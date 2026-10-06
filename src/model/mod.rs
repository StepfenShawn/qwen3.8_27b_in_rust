use crate::{
    gguf::{Gguf, TensorEntry},
    kernel::{Q38Q8KBlock, Q38_Q8_K_BLOCK_SIZE},
};

pub const Q38_HIDDEN: usize = 5120;
pub const Q38_FFN: usize = 17408;
pub const Q38_LAYERS: usize = 64;
pub const Q38_TOTAL_LAYERS: usize = 65;
pub const Q38_RECURRENT_LAYERS: usize = 48;
pub const Q38_FULL_LAYERS: usize = 16;
pub const Q38_TOTAL_FULL_LAYERS: usize = 17;
pub const Q38_LINEAR_QK_HEADS: usize = 16;
pub const Q38_LINEAR_V_HEADS: usize = 48;
pub const Q38_LINEAR_HEAD_DIM: usize = 128;
pub const Q38_LINEAR_QK_DIM: usize = 2048;
pub const Q38_LINEAR_V_DIM: usize = 6144;
pub const Q38_LINEAR_QKV_DIM: usize = 10240;
pub const Q38_ATTN_HEADS: usize = 24;
pub const Q38_ATTN_KV_HEADS: usize = 4;
pub const Q38_ATTN_HEAD_DIM: usize = 256;
pub const Q38_ATTN_Q_DIM: usize = 6144;
pub const Q38_ATTN_QG_DIM: usize = 12288;
pub const Q38_ATTN_KV_DIM: usize = 1024;
pub const Q38_VOCAB: usize = 248320;
pub const Q38_RMS_EPS: f32 = 1e-6;

pub enum Q38Attention<'a> {
    Linear {
        qkv: &'a TensorEntry<'a>,
        z: &'a TensorEntry<'a>,
        alpha: &'a TensorEntry<'a>,
        beta: &'a TensorEntry<'a>,
        conv: &'a TensorEntry<'a>,
        dt: &'a TensorEntry<'a>,
        a: &'a TensorEntry<'a>,
        norm: &'a TensorEntry<'a>,
        out: &'a TensorEntry<'a>,
    },
    Full {
        q: &'a TensorEntry<'a>,
        k: &'a TensorEntry<'a>,
        v: &'a TensorEntry<'a>,
        out: &'a TensorEntry<'a>,
        q_norm: &'a TensorEntry<'a>,
        k_norm: &'a TensorEntry<'a>,
    },
}

pub struct Q38Layer<'a> {
    pub attn_norm: &'a TensorEntry<'a>,
    pub post_norm: &'a TensorEntry<'a>,
    pub ffn_gate: &'a TensorEntry<'a>,
    pub ffn_up: &'a TensorEntry<'a>,
    pub ffn_down: &'a TensorEntry<'a>,
    pub attention: Q38Attention<'a>,
}

#[derive(Clone, Copy)]
pub struct Q38MTPWeights<'a> {
    pub eh_proj: &'a TensorEntry<'a>,
    pub enorm: &'a TensorEntry<'a>,
    pub hnorm: &'a TensorEntry<'a>,
    pub shared_head_norm: &'a TensorEntry<'a>,
}

pub struct Q38Scratch {
    pub hidden: Vec<f32>,
    pub norm: Vec<f32>,
    pub branch: Vec<f32>,
    pub wide0: Vec<f32>,
    pub wide1: Vec<f32>,
    pub q: Vec<f32>,
    pub gate: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn: Vec<f32>,
    pub scores: Vec<f32>,
    pub beta: Vec<f32>,
    pub alpha: Vec<f32>,
    pub logits: Vec<f32>,
    pub quantized: Vec<Q38Q8KBlock>,
}

pub struct Q38BatchScratch {
    pub hidden: Vec<f32>,
    pub norm: Vec<f32>,
    pub branch: Vec<f32>,
    pub wide0: Vec<f32>,
    pub wide1: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub attention: Vec<f32>,
    pub beta: Vec<f32>,
    pub alpha: Vec<f32>,
    pub target_norm: Vec<f32>,
    pub quantized: Vec<Q38Q8KBlock>,
}

impl Q38BatchScratch {
    /// Reserve the token-major scratch used by [`Q38ModelOps::prefill`] and the
    /// speculative verifier. Each buffer keeps the stride the kernel expects,
    /// which is why `wide0` is sized for the FFN even though it also carries
    /// the narrower attention activations.
    pub fn new(count: usize) -> Self {
        let quantized = (0..count * (Q38_FFN / Q38_Q8_K_BLOCK_SIZE))
            .map(|_| Q38Q8KBlock::default())
            .collect();
        Self {
            hidden: vec![0.0f32; count * Q38_HIDDEN],
            norm: vec![0.0f32; count * Q38_HIDDEN],
            branch: vec![0.0f32; count * Q38_HIDDEN],
            wide0: vec![0.0f32; count * Q38_FFN],
            wide1: vec![0.0f32; count * Q38_FFN],
            key: vec![0.0f32; count * Q38_ATTN_KV_DIM],
            value: vec![0.0f32; count * Q38_ATTN_KV_DIM],
            attention: vec![0.0f32; count * Q38_LINEAR_V_DIM],
            beta: vec![0.0f32; count * Q38_LINEAR_V_HEADS],
            alpha: vec![0.0f32; count * Q38_LINEAR_V_HEADS],
            target_norm: vec![0.0f32; count * Q38_HIDDEN],
            quantized,
        }
    }
}

pub struct Q38Model<'a> {
    pub gguf: &'a Gguf,
    pub mtp_gguf: Option<&'a Gguf>,

    pub embedding: &'a TensorEntry<'a>,
    pub output_norm: &'a TensorEntry<'a>,
    pub output: &'a TensorEntry<'a>,

    pub mtp_embedding: Option<&'a TensorEntry<'a>>,
    pub mtp_output: Option<&'a TensorEntry<'a>>,

    pub layers: Vec<Q38Layer<'a>>,
    pub mtp: Option<Q38MTPWeights<'a>>,

    pub context_length: u32,
    pub position: u32,
    pub mtp_position: u32,

    pub mtp_weights_loaded: bool,
    pub mtp_enabled: bool,

    pub mtp_input_hidden: Vec<f32>,
    pub checkpoint_conv_state: Vec<f32>,
    pub checkpoint_delta_state: Vec<f32>,
    pub checkpoint_mtp_hidden: Vec<f32>,

    pub checkpoint_position: u32,
    pub checkpoint_mtp_position: u32,

    pub conv_state: Vec<f32>,
    pub delta_state: Vec<f32>,
    pub key_cache: Vec<f32>,
    pub value_cache: Vec<f32>,

    pub scratch: Q38Scratch,
}

pub trait Q38ModelOps<'a>: Sized {
    /// Open a checkpoint, bind every base weight and reserve the inference
    /// state for `context_length` positions.
    fn open_gguf(gguf: &'a Gguf, context_length: u32) -> Result<Self, i32>;

    /// Attach a checkpoint that hosts the native one-layer MTP head. Only
    /// valid before the first token has been evaluated.
    fn attach_mtp_gguf(&mut self, gguf: &'a Gguf) -> Result<(), i32>;

    fn reset(&mut self);

    /// Evaluate one token through the complete model. The returned logits stay
    /// owned by the model and are replaced by the next call.
    fn forward_token(&mut self, token_id: u32) -> Result<&[f32], i32>;

    /// Layer-major prompt evaluation; the output is the logits after the final
    /// token and recurrent/KV state advances by `tokens.len()` positions.
    fn prefill(&mut self, tokens: &[u32]) -> Result<&[f32], i32>;

    /// Enable and probe the checkpoint's native one-layer MTP head.
    fn enable_mtp(&mut self) -> Result<(), i32>;

    /// Draft one token with the MTP head.
    fn mtp_forward(&mut self, token_id: u32) -> Result<&[f32], i32>;

    /// Consume the pending target token and verify up to `draft_count` MTP
    /// proposals. `accepted` receives the newly available tokens after
    /// `token_id` and the returned count is how many were committed; the final
    /// token stays pending for the next call. Greedy only, and the emitted
    /// sequence matches the target model's greedy decode exactly.
    fn speculative_greedy(
        &mut self,
        token_id: u32,
        stop_token: u32,
        draft_count: u32,
        accepted: &mut [u32],
    ) -> Result<(&[f32], u32), i32>;

    fn vocab_size(&self) -> u32;
    fn position(&self) -> u32;
    fn context_length(&self) -> u32;
}

pub mod ops;
pub mod sampler;
