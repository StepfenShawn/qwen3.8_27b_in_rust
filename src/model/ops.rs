use crate::gguf::{Gguf, TensorEntry};
use crate::kernel::Q38Q8KBlock;
use crate::model::{
    Q38Attention, Q38Layer, Q38Model, Q38ModelOps, Q38Scratch, Q38_ATTN_HEADS,
    Q38_ATTN_KV_DIM, Q38_ATTN_Q_DIM, Q38_FFN, Q38_HIDDEN, Q38_LAYERS,
    Q38_LINEAR_HEAD_DIM, Q38_LINEAR_QKV_DIM, Q38_LINEAR_V_DIM, Q38_LINEAR_V_HEADS,
    Q38_RECURRENT_LAYERS, Q38_TOTAL_FULL_LAYERS, Q38_VOCAB,
};

fn layer_tensor<'a>(
    gguf: &'a Gguf,
    layer: usize,
    suffix: &str,
) -> Result<&'a TensorEntry<'a>, i32> {
    let name = format!("blk.{layer}.{suffix}");
    gguf.find_tensor(&name).ok_or(-1)
}

fn bind_layer<'a>(gguf: &'a Gguf, layer: usize) -> Result<Q38Layer<'a>, i32> {
    let attn_norm = layer_tensor(gguf, layer, "attn_norm.weight")?;
    let post_norm = layer_tensor(gguf, layer, "post_attention_norm.weight")?;
    let ffn_gate = layer_tensor(gguf, layer, "ffn_gate.weight")?;
    let ffn_up = layer_tensor(gguf, layer, "ffn_up.weight")?;
    let ffn_down = layer_tensor(gguf, layer, "ffn_down.weight")?;

    let attention = if (layer + 1) % 4 == 0 {
        Q38Attention::Full {
            q: layer_tensor(gguf, layer, "attn_q.weight")?,
            k: layer_tensor(gguf, layer, "attn_k.weight")?,
            v: layer_tensor(gguf, layer, "attn_v.weight")?,
            out: layer_tensor(gguf, layer, "attn_output.weight")?,
            q_norm: layer_tensor(gguf, layer, "attn_q_norm.weight")?,
            k_norm: layer_tensor(gguf, layer, "attn_k_norm.weight")?,
        }
    } else {
        Q38Attention::Linear {
            qkv: layer_tensor(gguf, layer, "attn_qkv.weight")?,
            z: layer_tensor(gguf, layer, "attn_gate.weight")?,
            alpha: layer_tensor(gguf, layer, "ssm_alpha.weight")?,
            beta: layer_tensor(gguf, layer, "ssm_beta.weight")?,
            conv: layer_tensor(gguf, layer, "ssm_conv1d.weight")?,
            dt: layer_tensor(gguf, layer, "ssm_dt.bias")?,
            a: layer_tensor(gguf, layer, "ssm_a")?,
            norm: layer_tensor(gguf, layer, "ssm_norm.weight")?,
            out: layer_tensor(gguf, layer, "ssm_out.weight")?,
        }
    };

    Ok(Q38Layer {
        attn_norm,
        post_norm,
        ffn_gate,
        ffn_up,
        ffn_down,
        attention,
    })
}

fn alloc_scratch(context_length: u32) -> Q38Scratch {
    Q38Scratch {
        hidden: vec![0.0f32; Q38_HIDDEN],
        norm: vec![0.0f32; Q38_HIDDEN],
        branch: vec![0.0f32; Q38_HIDDEN],
        wide0: vec![0.0f32; Q38_FFN],
        wide1: vec![0.0f32; Q38_FFN],
        q: vec![0.0f32; Q38_ATTN_Q_DIM],
        gate: vec![0.0f32; Q38_ATTN_Q_DIM],
        k: vec![0.0f32; Q38_ATTN_KV_DIM],
        v: vec![0.0f32; Q38_ATTN_KV_DIM],
        attn: vec![0.0f32; Q38_LINEAR_V_DIM],
        scores: vec![0.0f32; Q38_ATTN_HEADS * context_length as usize],
        beta: vec![0.0f32; Q38_LINEAR_V_HEADS],
        alpha: vec![0.0f32; Q38_LINEAR_V_HEADS],
        logits: vec![0.0f32; Q38_VOCAB],
        quantized: (0..Q38_FFN / 256).map(|_| Q38Q8KBlock::default()).collect(),
    }
}

impl<'a> Q38Model<'a> {
    pub fn bind(gguf: &'a Gguf, context_length: u32) -> Result<Self, i32> {
        let embedding = gguf.find_tensor("token_embd.weight").ok_or(-1)?;
        let output_norm = gguf.find_tensor("output_norm.weight").ok_or(-1)?;
        let output = gguf.find_tensor("output.weight").ok_or(-1)?;

        let mut layers = Vec::with_capacity(Q38_LAYERS);
        for layer in 0..Q38_LAYERS {
            layers.push(bind_layer(gguf, layer)?);
        }

        let conv_count = Q38_RECURRENT_LAYERS * Q38_LINEAR_QKV_DIM * 3;
        let delta_count = Q38_RECURRENT_LAYERS
            * Q38_LINEAR_V_HEADS
            * Q38_LINEAR_HEAD_DIM
            * Q38_LINEAR_HEAD_DIM;
        let kv_count = Q38_TOTAL_FULL_LAYERS * context_length as usize * Q38_ATTN_KV_DIM;

        Ok(Q38Model {
            gguf,
            mtp_gguf: None,
            embedding,
            output_norm,
            output,
            mtp_embedding: None,
            mtp_output: None,
            layers,
            mtp: None,
            context_length,
            position: 0,
            mtp_position: 0,
            mtp_weights_loaded: false,
            mtp_enabled: false,
            mtp_input_hidden: vec![0.0f32; Q38_HIDDEN],
            checkpoint_conv_state: Vec::new(),
            checkpoint_delta_state: Vec::new(),
            checkpoint_mtp_hidden: Vec::new(),
            checkpoint_position: 0,
            checkpoint_mtp_position: 0,
            conv_state: vec![0.0f32; conv_count],
            delta_state: vec![0.0f32; delta_count],
            key_cache: vec![0.0f32; kv_count],
            value_cache: vec![0.0f32; kv_count],
            scratch: alloc_scratch(context_length),
        })
    }
}

impl<'a> Q38ModelOps<'a> for Q38Model<'a> {
    fn open_gguf(gguf: &'a Gguf, context_length: u32) -> Result<Self, i32> {
        if context_length == 0 {
            return Err(-1);
        }
        Self::bind(gguf, context_length)
    }

    fn reset(&mut self) {
        self.conv_state.fill(0.0);
        self.delta_state.fill(0.0);
        self.key_cache.fill(0.0);
        self.value_cache.fill(0.0);
        self.mtp_input_hidden.fill(0.0);
        self.position = 0;
        self.mtp_position = 0;
    }

    fn forward_token(&mut self, token_id: u32) -> Result<&[f32], i32> {
        if token_id as usize >= Q38_VOCAB || self.position >= self.context_length {
            return Err(-1);
        }
        self.scratch.logits.fill(0.0);
        self.position += 1;
        Ok(&self.scratch.logits)
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<&[f32], i32> {
        if tokens.is_empty() {
            return Err(-1);
        }
        if tokens.len() as u32 > self.context_length - self.position {
            return Err(-1);
        }
        self.scratch.logits.fill(0.0);
        self.position += tokens.len() as u32;
        Ok(&self.scratch.logits)
    }

    fn vocab_size(&self) -> u32 {
        Q38_VOCAB as u32
    }

    fn position(&self) -> u32 {
        self.position
    }

    fn context_length(&self) -> u32 {
        self.context_length
    }
}
