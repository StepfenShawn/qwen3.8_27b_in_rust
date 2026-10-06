//! Qwen3.8-27B inference graph.
//!
//! The stack is a hybrid: three out
//! of every four blocks run a gated DeltaNet ("linear") attention with a
//! per-head recurrent state, and every fourth block runs full causal attention
//! over a KV cache. Both are followed by a gated residual + SwiGLU FFN, and the
//! checkpoint ships a native one-layer MTP head that the speculative decoder
//! uses to draft tokens.
//!
//! Layout notes carried over from the reference:
//!
//! * `shape[0]` is the input width and `shape[1]` the output rows, so a row of
//!   a projection is contiguous and can be streamed without materializing the
//!   whole matrix.
//! * Activations are quantized to Q8_K once and reused by every projection that
//!   consumes the same vector; only the first projection of a group pays for
//!   the quantization.
//! * The batched (`prefill`) path walks weight rows in the outer loop so a
//!   resident row serves every token before the kernel advances through the
//!   file.

use crate::gguf::{GgmlType, Gguf, TensorEntry};
use crate::kernel::quant::quantize_q8_k;
use crate::kernel::{Q38Q8KBlock, Q38TensorOps, Q38_Q8_K_BLOCK_SIZE, q38_parallel_over};
use crate::model::{
    Q38Attention, Q38BatchScratch, Q38Layer, Q38MTPWeights, Q38Model, Q38ModelOps, Q38Scratch,
    Q38_ATTN_HEADS, Q38_ATTN_HEAD_DIM, Q38_ATTN_KV_DIM, Q38_ATTN_KV_HEADS, Q38_ATTN_QG_DIM,
    Q38_ATTN_Q_DIM, Q38_FFN, Q38_HIDDEN, Q38_LAYERS, Q38_LINEAR_HEAD_DIM, Q38_LINEAR_QK_DIM,
    Q38_LINEAR_QK_HEADS, Q38_LINEAR_QKV_DIM, Q38_LINEAR_V_DIM, Q38_LINEAR_V_HEADS,
    Q38_RECURRENT_LAYERS, Q38_RMS_EPS, Q38_TOTAL_FULL_LAYERS, Q38_TOTAL_LAYERS, Q38_VOCAB,
};
use rayon::prelude::*;

const BLOCK: usize = Q38_Q8_K_BLOCK_SIZE;

/// Full-attention layers sit at every fourth position, and the MTP block lives
/// past the end of the base stack.
#[inline]
fn is_full_attention(layer: usize) -> bool {
    (layer + 1) % 4 == 0 || layer == Q38_LAYERS
}

/// Index this layer has in the recurrent/KV state arrays.
#[inline]
fn recurrent_index(layer: usize) -> usize {
    layer - (layer + 1) / 4
}

// ---------------------------------------------------------------------------
// Weight binding
// ---------------------------------------------------------------------------

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

    let attention = if is_full_attention(layer) {
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

/// True when a matrix type can be consumed by [`quantize_q8_k`]-style integer
/// dot products. `Q8_0` is deliberately excluded: its 32-wide blocks would
/// force a separate activation layout, so it goes through the f32 path.
fn matrix_type_q8_k(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::Q2K
            | GgmlType::Q3K
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
            | GgmlType::Iq2Xxs
            | GgmlType::Iq2Xs
            | GgmlType::Iq3Xxs
            | GgmlType::Iq4Nl
            | GgmlType::Iq3S
            | GgmlType::Iq2S
            | GgmlType::Iq4Xs
            | GgmlType::Iq1S
            | GgmlType::Iq1M
    )
}

fn matrix_type_supported(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16 | GgmlType::Q8_0
    ) || matrix_type_q8_k(ty)
}

fn expect_tensor(
    tensor: &TensorEntry<'_>,
    name: &str,
    width: u64,
    rows: u64,
    f32_only: bool,
) -> Result<(), i32> {
    let dimensions = if rows != 0 { 2 } else { 1 };
    let shape_ok = tensor.n_dims == dimensions
        && tensor.shape[0] == width
        && (rows == 0 || tensor.shape[1] == rows);
    let type_ok = if f32_only {
        tensor.ty == GgmlType::F32
    } else {
        matrix_type_supported(tensor.ty)
    };
    if shape_ok && type_ok {
        return Ok(());
    }
    eprintln!(
        "qwen38: incompatible tensor {name}; expected {}{width}{}",
        if f32_only { "F32 " } else { "" },
        if rows != 0 {
            format!(",{rows}")
        } else {
            String::new()
        }
    );
    Err(-1)
}

fn validate_layer(weights: &Q38Layer<'_>, layer: usize) -> Result<(), i32> {
    let name = |suffix: &str| format!("blk.{layer}.{suffix}");
    expect_tensor(
        weights.attn_norm,
        &name("attn_norm.weight"),
        Q38_HIDDEN as u64,
        0,
        true,
    )?;
    expect_tensor(
        weights.post_norm,
        &name("post_attention_norm.weight"),
        Q38_HIDDEN as u64,
        0,
        true,
    )?;
    expect_tensor(
        weights.ffn_gate,
        &name("ffn_gate.weight"),
        Q38_HIDDEN as u64,
        Q38_FFN as u64,
        false,
    )?;
    expect_tensor(
        weights.ffn_up,
        &name("ffn_up.weight"),
        Q38_HIDDEN as u64,
        Q38_FFN as u64,
        false,
    )?;
    expect_tensor(
        weights.ffn_down,
        &name("ffn_down.weight"),
        Q38_FFN as u64,
        Q38_HIDDEN as u64,
        false,
    )?;

    match &weights.attention {
        Q38Attention::Full {
            q,
            k,
            v,
            out,
            q_norm,
            k_norm,
        } => {
            expect_tensor(
                q,
                &name("attn_q.weight"),
                Q38_HIDDEN as u64,
                Q38_ATTN_QG_DIM as u64,
                false,
            )?;
            expect_tensor(
                k,
                &name("attn_k.weight"),
                Q38_HIDDEN as u64,
                Q38_ATTN_KV_DIM as u64,
                false,
            )?;
            expect_tensor(
                v,
                &name("attn_v.weight"),
                Q38_HIDDEN as u64,
                Q38_ATTN_KV_DIM as u64,
                false,
            )?;
            expect_tensor(
                out,
                &name("attn_output.weight"),
                Q38_ATTN_Q_DIM as u64,
                Q38_HIDDEN as u64,
                false,
            )?;
            expect_tensor(
                q_norm,
                &name("attn_q_norm.weight"),
                Q38_ATTN_HEAD_DIM as u64,
                0,
                true,
            )?;
            expect_tensor(
                k_norm,
                &name("attn_k_norm.weight"),
                Q38_ATTN_HEAD_DIM as u64,
                0,
                true,
            )?;
        }
        Q38Attention::Linear {
            qkv,
            z,
            alpha,
            beta,
            conv,
            dt,
            a,
            norm,
            out,
        } => {
            expect_tensor(
                qkv,
                &name("attn_qkv.weight"),
                Q38_HIDDEN as u64,
                Q38_LINEAR_QKV_DIM as u64,
                false,
            )?;
            expect_tensor(
                z,
                &name("attn_gate.weight"),
                Q38_HIDDEN as u64,
                Q38_LINEAR_V_DIM as u64,
                false,
            )?;
            expect_tensor(
                alpha,
                &name("ssm_alpha.weight"),
                Q38_HIDDEN as u64,
                Q38_LINEAR_V_HEADS as u64,
                false,
            )?;
            expect_tensor(
                beta,
                &name("ssm_beta.weight"),
                Q38_HIDDEN as u64,
                Q38_LINEAR_V_HEADS as u64,
                false,
            )?;
            expect_tensor(conv, &name("ssm_conv1d.weight"), 4, Q38_LINEAR_QKV_DIM as u64, true)?;
            expect_tensor(dt, &name("ssm_dt.bias"), Q38_LINEAR_V_HEADS as u64, 0, true)?;
            expect_tensor(a, &name("ssm_a"), Q38_LINEAR_V_HEADS as u64, 0, true)?;
            expect_tensor(
                norm,
                &name("ssm_norm.weight"),
                Q38_LINEAR_HEAD_DIM as u64,
                0,
                true,
            )?;
            expect_tensor(
                out,
                &name("ssm_out.weight"),
                Q38_LINEAR_V_DIM as u64,
                Q38_HIDDEN as u64,
                false,
            )?;
        }
    }
    Ok(())
}

fn meta_string_is(gguf: &Gguf, key: &str, expected: &str) -> bool {
    gguf.meta_string(key)
        .map(|value| value.eq_str(expected))
        .unwrap_or(false)
}

fn meta_u32_is(gguf: &Gguf, key: &str, expected: u32) -> bool {
    gguf.meta_u32(key) == Some(expected)
}

fn validate_base_contract(
    gguf: &Gguf,
    embedding: &TensorEntry<'_>,
    output_norm: &TensorEntry<'_>,
    output: &TensorEntry<'_>,
    layers: &[Q38Layer<'_>],
) -> Result<(), i32> {
    let block_count = gguf.meta_u32("qwen35.block_count").ok_or(-1)?;
    let nextn_layers = gguf.meta_u32("qwen35.nextn_predict_layers").ok_or(-1)?;
    let layers_ok = (block_count == Q38_LAYERS as u32 && nextn_layers == 0)
        || (block_count == Q38_TOTAL_LAYERS as u32 && nextn_layers == 1);
    let metadata_ok = layers_ok
        && meta_string_is(gguf, "general.architecture", "qwen35")
        && meta_u32_is(gguf, "qwen35.embedding_length", Q38_HIDDEN as u32)
        && meta_u32_is(gguf, "qwen35.feed_forward_length", Q38_FFN as u32)
        && meta_u32_is(gguf, "qwen35.attention.head_count", Q38_ATTN_HEADS as u32)
        && meta_u32_is(gguf, "qwen35.attention.head_count_kv", Q38_ATTN_KV_HEADS as u32)
        && meta_u32_is(gguf, "qwen35.full_attention_interval", 4)
        && meta_u32_is(gguf, "qwen35.ssm.state_size", Q38_LINEAR_HEAD_DIM as u32)
        && meta_u32_is(gguf, "qwen35.ssm.group_count", Q38_LINEAR_QK_HEADS as u32)
        && meta_u32_is(gguf, "qwen35.ssm.time_step_rank", Q38_LINEAR_V_HEADS as u32)
        && meta_u32_is(gguf, "qwen35.ssm.inner_size", Q38_LINEAR_V_DIM as u32);
    if !metadata_ok {
        eprintln!("qwen38: checkpoint metadata is not Qwen3.8-27B");
        return Err(-1);
    }

    expect_tensor(
        embedding,
        "token_embd.weight",
        Q38_HIDDEN as u64,
        Q38_VOCAB as u64,
        false,
    )?;
    expect_tensor(
        output_norm,
        "output_norm.weight",
        Q38_HIDDEN as u64,
        0,
        true,
    )?;
    expect_tensor(
        output,
        "output.weight",
        Q38_HIDDEN as u64,
        Q38_VOCAB as u64,
        false,
    )?;
    for (layer, weights) in layers.iter().enumerate() {
        validate_layer(weights, layer)?;
    }
    Ok(())
}

fn validate_mtp_contract(
    gguf: &Gguf,
    embedding: &TensorEntry<'_>,
    output: &TensorEntry<'_>,
    layer: &Q38Layer<'_>,
    mtp: &Q38MTPWeights<'_>,
) -> Result<(), i32> {
    let metadata_ok = meta_string_is(gguf, "general.architecture", "qwen35")
        && meta_u32_is(gguf, "qwen35.block_count", Q38_TOTAL_LAYERS as u32)
        && meta_u32_is(gguf, "qwen35.nextn_predict_layers", 1)
        && meta_u32_is(gguf, "qwen35.embedding_length", Q38_HIDDEN as u32)
        && meta_u32_is(gguf, "qwen35.feed_forward_length", Q38_FFN as u32);
    if !metadata_ok {
        eprintln!("qwen38: MTP checkpoint metadata is not Qwen3.8-27B");
        return Err(-1);
    }
    expect_tensor(
        embedding,
        "token_embd.weight",
        Q38_HIDDEN as u64,
        Q38_VOCAB as u64,
        false,
    )?;
    expect_tensor(
        output,
        "output.weight",
        Q38_HIDDEN as u64,
        Q38_VOCAB as u64,
        false,
    )?;
    validate_layer(layer, Q38_LAYERS)?;
    expect_tensor(
        mtp.eh_proj,
        "blk.64.nextn.eh_proj.weight",
        2 * Q38_HIDDEN as u64,
        Q38_HIDDEN as u64,
        false,
    )?;
    expect_tensor(
        mtp.enorm,
        "blk.64.nextn.enorm.weight",
        Q38_HIDDEN as u64,
        0,
        true,
    )?;
    expect_tensor(
        mtp.hnorm,
        "blk.64.nextn.hnorm.weight",
        Q38_HIDDEN as u64,
        0,
        true,
    )?;
    expect_tensor(
        mtp.shared_head_norm,
        "blk.64.nextn.shared_head_norm.weight",
        Q38_HIDDEN as u64,
        0,
        true,
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Elementwise primitives
// ---------------------------------------------------------------------------

/// Read element `index` of a 1-D F32 tensor. Model scalars (`ssm_a`,
/// `ssm_dt.bias`) and every RMSNorm weight are stored unpacked.
fn scalar(tensor: &TensorEntry<'_>, index: usize) -> f32 {
    if tensor.ty != GgmlType::F32 || tensor.n_dims != 1 || index as u64 >= tensor.shape[0] {
        return f32::NAN;
    }
    let at = index * 4;
    f32::from_le_bytes([
        tensor.data[at],
        tensor.data[at + 1],
        tensor.data[at + 2],
        tensor.data[at + 3],
    ])
}

#[inline]
fn f32_at(data: &[u8], index: usize) -> f32 {
    let at = index * 4;
    f32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]])
}

/// Reciprocal RMS scale, accumulated in f64 exactly like the reference.
fn rms_scale(values: &[f32]) -> f32 {
    let mut sum = 0.0f64;
    for value in values {
        sum += (*value as f64) * (*value as f64);
    }
    let mean = (sum / values.len() as f64) as f32;
    1.0 / (mean + Q38_RMS_EPS).sqrt()
}

fn rmsnorm(output: &mut [f32], input: &[f32], weight: &TensorEntry<'_>, length: usize) {
    let scale = rms_scale(&input[..length]);
    for i in 0..length {
        output[i] = input[i] * scale * scalar(weight, i);
    }
}

/// In-place variant for the cases where the reference aliases input and output.
fn rmsnorm_inplace(values: &mut [f32], weight: &TensorEntry<'_>, length: usize) {
    let scale = rms_scale(&values[..length]);
    for i in 0..length {
        values[i] = values[i] * scale * scalar(weight, i);
    }
}

#[inline]
fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[inline]
fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

#[inline]
fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        return value;
    }
    (1.0 + value.exp()).ln()
}

fn silu_inplace(values: &mut [f32], length: usize) {
    for value in values[..length].iter_mut() {
        *value = silu(*value);
    }
}

/// `values *= silu(gate)`.
fn silu_multiply(values: &mut [f32], gate: &[f32], length: usize) {
    for i in 0..length {
        values[i] *= silu(gate[i]);
    }
}

/// `gate = silu(gate) * up`, the SwiGLU nonlinearity.
fn swiglu(gate: &mut [f32], up: &[f32], length: usize) {
    let gate = &mut gate[..length];
    let up = &up[..length];
    if q38_parallel_over(length, 1) {
        gate.par_iter_mut()
            .zip(up.par_iter())
            .for_each(|(g, u)| *g = silu(*g) * *u);
    } else {
        for (g, u) in gate.iter_mut().zip(up.iter()) {
            *g = silu(*g) * *u;
        }
    }
}

/// `accumulator += addend`, elementwise, over the first `length` lanes.
fn accumulate(accumulator: &mut [f32], addend: &[f32], length: usize) {
    let accumulator = &mut accumulator[..length];
    let addend = &addend[..length];
    if q38_parallel_over(length, 1) {
        accumulator
            .par_iter_mut()
            .zip(addend.par_iter())
            .for_each(|(a, b)| *a += *b);
    } else {
        for (a, b) in accumulator.iter_mut().zip(addend.iter()) {
            *a += *b;
        }
    }
}

fn l2norm(values: &mut [f32]) {
    let mut sum = 0.0f64;
    for value in values.iter() {
        sum += (*value as f64) * (*value as f64);
    }
    let scale = 1.0 / (sum as f32).sqrt().max(Q38_RMS_EPS);
    for value in values.iter_mut() {
        *value *= scale;
    }
}

fn dot_128(left: &[f32], right: &[f32]) -> f32 {
    let mut total = 0.0f32;
    for i in 0..Q38_LINEAR_HEAD_DIM {
        total = left[i].mul_add(right[i], total);
    }
    total
}

/// `output += input * scale`, then the dot product of the updated `output`
/// against `right`. Fusing the two passes is what keeps the DeltaNet state
/// update in registers.
fn mad_dot_128(output: &mut [f32], input: &[f32], scale: f32, right: &[f32]) -> f32 {
    let mut total = 0.0f32;
    for i in 0..Q38_LINEAR_HEAD_DIM {
        output[i] = input[i].mul_add(scale, output[i]);
        total = output[i].mul_add(right[i], total);
    }
    total
}

/// Half-rotation RoPE: the first 32 lanes of every head pair with the second
/// 32, with a per-pair geometrically decaying frequency.
fn rope(values: &mut [f32], heads: usize, position: u32) {
    let theta_scale = 10000000.0f32.powf(-2.0f32 / 64.0f32);
    for head in 0..heads {
        let vector = &mut values[head * Q38_ATTN_HEAD_DIM..(head + 1) * Q38_ATTN_HEAD_DIM];
        let mut theta = position as f32;
        for pair in 0..32 {
            let cosine = theta.cos();
            let sine = theta.sin();
            let x0 = vector[pair];
            let x1 = vector[pair + 32];
            vector[pair] = x0 * cosine - x1 * sine;
            vector[pair + 32] = x0 * sine + x1 * cosine;
            theta *= theta_scale;
        }
    }
}

// ---------------------------------------------------------------------------
// Projections
// ---------------------------------------------------------------------------

/// Quantize `input` into `quantized` so several projections can share it.
fn quantize_input(quantized: &mut [Q38Q8KBlock], input: &[f32]) -> Result<(), i32> {
    quantize_q8_k(quantized, input).map_err(|_| -1)
}

/// One projection: integer dot products when the weight allows it, otherwise a
/// decoded f32 mat-vec.
fn project(
    output: &mut [f32],
    input: &[f32],
    quantized: &[Q38Q8KBlock],
    length: usize,
    weight: &TensorEntry<'_>,
) -> Result<(), i32> {
    debug_assert_eq!(length as u64, weight.shape[0]);
    let result = if matrix_type_q8_k(weight.ty) {
        let blocks = length / BLOCK;
        let activations = quantized.get(..blocks).ok_or(-1)?;
        weight.tensor_gemv_q8_k(output, activations)
    } else {
        weight.tensor_gemv_f32(output, input.get(..length).ok_or(-1)?)
    };
    result.map_err(|_| -1)
}

// ---------------------------------------------------------------------------
// Gated DeltaNet attention
// ---------------------------------------------------------------------------

/// Depthwise causal convolution (kernel 4) into the qkv buffer, then the
/// gated delta-rule recurrence over `Q38_LINEAR_V_HEADS` heads.
#[allow(clippy::too_many_arguments)]
fn linear_core(
    weights: &Q38Layer<'_>,
    conv_state: &mut [f32],
    delta_state: &mut [f32],
    qkv: &mut [f32],
    z: &[f32],
    beta: &mut [f32],
    alpha: &mut [f32],
    attention: &mut [f32],
) -> Result<(), i32> {
    let Q38Attention::Linear { conv, dt, a, norm, .. } = &weights.attention else {
        return Err(-1);
    };

    for channel in 0..Q38_LINEAR_QKV_DIM {
        let state = &mut conv_state[channel * 3..channel * 3 + 3];
        let mut value = state[0] * f32_at(conv.data, channel * 4);
        value += state[1] * f32_at(conv.data, channel * 4 + 1);
        value += state[2] * f32_at(conv.data, channel * 4 + 2);
        value += qkv[channel] * f32_at(conv.data, channel * 4 + 3);
        state[0] = state[1];
        state[1] = state[2];
        state[2] = qkv[channel];
        qkv[channel] = value;
    }
    silu_inplace(qkv, Q38_LINEAR_QKV_DIM);

    let (query, rest) = qkv.split_at_mut(Q38_LINEAR_QK_DIM);
    let (key, value) = rest.split_at_mut(Q38_LINEAR_QK_DIM);
    for head in 0..Q38_LINEAR_QK_HEADS {
        l2norm(&mut query[head * Q38_LINEAR_HEAD_DIM..][..Q38_LINEAR_HEAD_DIM]);
        l2norm(&mut key[head * Q38_LINEAR_HEAD_DIM..][..Q38_LINEAR_HEAD_DIM]);
    }
    for head in 0..Q38_LINEAR_V_HEADS {
        beta[head] = sigmoid(beta[head]);
        alpha[head] = scalar(a, head) * softplus(alpha[head] + scalar(dt, head));
    }

    let query_scale = 1.0f32 / (Q38_LINEAR_HEAD_DIM as f32).sqrt();
    let head = Q38_LINEAR_HEAD_DIM * Q38_LINEAR_HEAD_DIM;
    let query: &[f32] = query;
    let key: &[f32] = key;
    let value: &[f32] = value;
    // Each value head owns one `head`-sized slice of the recurrent state and one
    // `Q38_LINEAR_HEAD_DIM`-wide slice of the output, both contiguous, and only
    // reads the shared q/k/v projections. That makes the head the natural unit
    // of work, exactly as in the reference's OpenMP region.
    let run_head = |value_head: usize, state: &mut [f32], head_output: &mut [f32]| {
        let key_head = value_head % Q38_LINEAR_QK_HEADS;
        let query_head = &query[key_head * Q38_LINEAR_HEAD_DIM..][..Q38_LINEAR_HEAD_DIM];
        let key_vec = &key[key_head * Q38_LINEAR_HEAD_DIM..][..Q38_LINEAR_HEAD_DIM];
        let value_vec = &value[value_head * Q38_LINEAR_HEAD_DIM..][..Q38_LINEAR_HEAD_DIM];
        let decay = alpha[value_head].exp();
        for entry in state.iter_mut() {
            *entry *= decay;
        }
        for column in 0..Q38_LINEAR_HEAD_DIM {
            let state_column = &mut state[column * Q38_LINEAR_HEAD_DIM..][..Q38_LINEAR_HEAD_DIM];
            let prediction = dot_128(state_column, key_vec);
            let delta = (value_vec[column] - prediction) * beta[value_head];
            let result = mad_dot_128(state_column, key_vec, delta, query_head);
            head_output[column] = result * query_scale;
        }
    };
    let states = &mut delta_state[..Q38_LINEAR_V_HEADS * head];
    let outputs = &mut attention[..Q38_LINEAR_V_HEADS * Q38_LINEAR_HEAD_DIM];
    if q38_parallel_over(Q38_LINEAR_V_HEADS, head) {
        states
            .par_chunks_mut(head)
            .zip(outputs.par_chunks_mut(Q38_LINEAR_HEAD_DIM))
            .enumerate()
            .for_each(|(value_head, (state, head_output))| {
                run_head(value_head, state, head_output)
            });
    } else {
        for (value_head, (state, head_output)) in states
            .chunks_mut(head)
            .zip(outputs.chunks_mut(Q38_LINEAR_HEAD_DIM))
            .enumerate()
        {
            run_head(value_head, state, head_output);
        }
    }

    for value_head in 0..Q38_LINEAR_V_HEADS {
        let head_output =
            &mut attention[value_head * Q38_LINEAR_HEAD_DIM..][..Q38_LINEAR_HEAD_DIM];
        rmsnorm_inplace(head_output, norm, Q38_LINEAR_HEAD_DIM);
        let gate = &z[value_head * Q38_LINEAR_HEAD_DIM..][..Q38_LINEAR_HEAD_DIM];
        silu_multiply(head_output, gate, Q38_LINEAR_HEAD_DIM);
    }
    Ok(())
}

fn linear_attention(
    weights: &Q38Layer<'_>,
    scratch: &mut Q38Scratch,
    conv_state: &mut [f32],
    delta_state: &mut [f32],
) -> Result<(), i32> {
    let Q38Attention::Linear {
        qkv,
        z,
        alpha,
        beta,
        out,
        ..
    } = &weights.attention
    else {
        return Err(-1);
    };
    let input = &scratch.norm[..Q38_HIDDEN];
    quantize_input(&mut scratch.quantized, input)?;
    project(
        &mut scratch.wide0,
        input,
        &scratch.quantized,
        Q38_HIDDEN,
        qkv,
    )?;
    project(
        &mut scratch.wide1,
        input,
        &scratch.quantized,
        Q38_HIDDEN,
        z,
    )?;
    project(
        &mut scratch.beta,
        input,
        &scratch.quantized,
        Q38_HIDDEN,
        beta,
    )?;
    project(
        &mut scratch.alpha,
        input,
        &scratch.quantized,
        Q38_HIDDEN,
        alpha,
    )?;
    linear_core(
        weights,
        conv_state,
        delta_state,
        &mut scratch.wide0,
        &scratch.wide1,
        &mut scratch.beta,
        &mut scratch.alpha,
        &mut scratch.attn,
    )?;

    quantize_input(&mut scratch.quantized, &scratch.attn[..Q38_LINEAR_V_DIM])?;
    project(
        &mut scratch.branch,
        &scratch.attn,
        &scratch.quantized,
        Q38_LINEAR_V_DIM,
        out,
    )
}

// ---------------------------------------------------------------------------
// Full causal attention
// ---------------------------------------------------------------------------

/// Q/K RMSNorm per head, RoPE, KV cache append, causal softmax attention and
/// the output gate. `current_position` is the slot being written.
#[allow(clippy::too_many_arguments)]
fn full_core(
    weights: &Q38Layer<'_>,
    q_and_gate: &[f32],
    key: &mut [f32],
    value: &[f32],
    q: &mut [f32],
    gate: &mut [f32],
    scores: &mut [f32],
    attention: &mut [f32],
    key_cache: &mut [f32],
    value_cache: &mut [f32],
    context_length: usize,
    current_position: u32,
    layer: usize,
) -> Result<(), i32> {
    let Q38Attention::Full { q_norm, k_norm, .. } = &weights.attention else {
        return Err(-1);
    };

    for head in 0..Q38_ATTN_HEADS {
        let source = &q_and_gate[head * 2 * Q38_ATTN_HEAD_DIM..];
        q[head * Q38_ATTN_HEAD_DIM..][..Q38_ATTN_HEAD_DIM]
            .copy_from_slice(&source[..Q38_ATTN_HEAD_DIM]);
        gate[head * Q38_ATTN_HEAD_DIM..][..Q38_ATTN_HEAD_DIM]
            .copy_from_slice(&source[Q38_ATTN_HEAD_DIM..][..Q38_ATTN_HEAD_DIM]);
        rmsnorm_inplace(
            &mut q[head * Q38_ATTN_HEAD_DIM..],
            q_norm,
            Q38_ATTN_HEAD_DIM,
        );
    }
    for head in 0..Q38_ATTN_KV_HEADS {
        rmsnorm_inplace(
            &mut key[head * Q38_ATTN_HEAD_DIM..],
            k_norm,
            Q38_ATTN_HEAD_DIM,
        );
    }
    rope(&mut q[..Q38_ATTN_Q_DIM], Q38_ATTN_HEADS, current_position);
    rope(&mut key[..Q38_ATTN_KV_DIM], Q38_ATTN_KV_HEADS, current_position);

    let full_index = layer / 4;
    let cache_row = (full_index * context_length + current_position as usize) * Q38_ATTN_KV_DIM;
    if cache_row + Q38_ATTN_KV_DIM > key_cache.len() || cache_row + Q38_ATTN_KV_DIM > value_cache.len()
    {
        return Err(-1);
    }
    key_cache[cache_row..cache_row + Q38_ATTN_KV_DIM]
        .copy_from_slice(&key[..Q38_ATTN_KV_DIM]);
    value_cache[cache_row..cache_row + Q38_ATTN_KV_DIM]
        .copy_from_slice(&value[..Q38_ATTN_KV_DIM]);

    let groups = Q38_ATTN_HEADS / Q38_ATTN_KV_HEADS;
    let score_scale = 1.0 / (Q38_ATTN_HEAD_DIM as f32).sqrt();
    let positions = current_position as usize + 1;
    let q: &[f32] = q;
    let gate: &[f32] = gate;
    let key_cache: &[f32] = key_cache;
    let value_cache: &[f32] = value_cache;
    // Head `h` writes only `scores[h * context_length..]` and
    // `attention[h * Q38_ATTN_HEAD_DIM..]`, and reads the KV cache. Both write
    // targets are contiguous per head, so the split is a plain chunk and the
    // per-head work is unchanged from the reference's OpenMP region.
    let run_head = |head: usize, head_scores: &mut [f32], head_output: &mut [f32]| {
        let kv_head = head / groups;
        let q_head = &q[head * Q38_ATTN_HEAD_DIM..][..Q38_ATTN_HEAD_DIM];

        let mut maximum = f32::NEG_INFINITY;
        for position in 0..positions {
            let row = (full_index * context_length + position) * Q38_ATTN_KV_DIM
                + kv_head * Q38_ATTN_HEAD_DIM;
            let k = &key_cache[row..][..Q38_ATTN_HEAD_DIM];
            let mut score = 0.0f32;
            for i in 0..Q38_ATTN_HEAD_DIM {
                score = q_head[i].mul_add(k[i], score);
            }
            score *= score_scale;
            head_scores[position] = score;
            if score > maximum {
                maximum = score;
            }
        }
        let mut denominator = 0.0f32;
        for position in 0..positions {
            head_scores[position] = (head_scores[position] - maximum).exp();
            denominator += head_scores[position];
        }

        head_output.fill(0.0);
        for position in 0..positions {
            let row = (full_index * context_length + position) * Q38_ATTN_KV_DIM
                + kv_head * Q38_ATTN_HEAD_DIM;
            let v = &value_cache[row..][..Q38_ATTN_HEAD_DIM];
            let probability = head_scores[position] / denominator;
            for i in 0..Q38_ATTN_HEAD_DIM {
                head_output[i] = probability.mul_add(v[i], head_output[i]);
            }
        }
        let head_gate = &gate[head * Q38_ATTN_HEAD_DIM..][..Q38_ATTN_HEAD_DIM];
        for i in 0..Q38_ATTN_HEAD_DIM {
            head_output[i] *= sigmoid(head_gate[i]);
        }
    };
    let work = positions * Q38_ATTN_HEAD_DIM;
    let scores = &mut scores[..Q38_ATTN_HEADS * context_length];
    let attention = &mut attention[..Q38_ATTN_HEADS * Q38_ATTN_HEAD_DIM];
    if q38_parallel_over(Q38_ATTN_HEADS, work) {
        scores
            .par_chunks_mut(context_length)
            .zip(attention.par_chunks_mut(Q38_ATTN_HEAD_DIM))
            .enumerate()
            .for_each(|(head, (head_scores, head_output))| {
                run_head(head, head_scores, head_output)
            });
    } else {
        for (head, (head_scores, head_output)) in scores
            .chunks_mut(context_length)
            .zip(attention.chunks_mut(Q38_ATTN_HEAD_DIM))
            .enumerate()
        {
            run_head(head, head_scores, head_output);
        }
    }
    Ok(())
}

fn full_attention(
    weights: &Q38Layer<'_>,
    scratch: &mut Q38Scratch,
    key_cache: &mut [f32],
    value_cache: &mut [f32],
    context_length: usize,
    position: u32,
    layer: usize,
) -> Result<(), i32> {
    let Q38Attention::Full { q, k, v, out, .. } = &weights.attention else {
        return Err(-1);
    };
    let input = &scratch.norm[..Q38_HIDDEN];
    quantize_input(&mut scratch.quantized, input)?;
    project(&mut scratch.wide0, input, &scratch.quantized, Q38_HIDDEN, q)?;
    project(&mut scratch.k, input, &scratch.quantized, Q38_HIDDEN, k)?;
    project(&mut scratch.v, input, &scratch.quantized, Q38_HIDDEN, v)?;
    full_core(
        weights,
        &scratch.wide0[..Q38_ATTN_QG_DIM],
        &mut scratch.k,
        &scratch.v,
        &mut scratch.q,
        &mut scratch.gate,
        &mut scratch.scores,
        &mut scratch.attn,
        key_cache,
        value_cache,
        context_length,
        position,
        layer,
    )?;
    quantize_input(&mut scratch.quantized, &scratch.attn[..Q38_ATTN_Q_DIM])?;
    project(
        &mut scratch.branch,
        &scratch.attn,
        &scratch.quantized,
        Q38_ATTN_Q_DIM,
        out,
    )
}

// ---------------------------------------------------------------------------
// Batched (prefill / verifier) variants
// ---------------------------------------------------------------------------

fn batch_norm(output: &mut [f32], input: &[f32], weight: &TensorEntry<'_>, count: usize) {
    for token in 0..count {
        let at = token * Q38_HIDDEN;
        rmsnorm(
            &mut output[at..],
            &input[at..(token + 1) * Q38_HIDDEN],
            weight,
            Q38_HIDDEN,
        );
    }
}

fn batch_quantize(
    quantized: &mut [Q38Q8KBlock],
    input: &[f32],
    count: usize,
    length: usize,
) -> Result<(), i32> {
    let blocks = length / BLOCK;
    for token in 0..count {
        let activations = quantized
            .get_mut(token * blocks..(token + 1) * blocks)
            .ok_or(-1)?;
        quantize_input(activations, &input[token * length..(token + 1) * length])?;
    }
    Ok(())
}

fn batch_project(
    output: &mut [f32],
    input: &[f32],
    quantized: &[Q38Q8KBlock],
    count: usize,
    length: usize,
    weight: &TensorEntry<'_>,
) -> Result<(), i32> {
    debug_assert_eq!(length as u64, weight.shape[0]);
    let result = if matrix_type_q8_k(weight.ty) {
        weight.tensor_gemm_q8_k(output, quantized, count as u32)
    } else {
        weight.tensor_gemm_f32(output, input, count as u32)
    };
    result.map_err(|_| -1)
}

fn batch_linear_attention(
    model: &mut Q38Model<'_>,
    batch: &mut Q38BatchScratch,
    layer: usize,
    count: usize,
) -> Result<(), i32> {
    let Q38Attention::Linear {
        qkv,
        z,
        alpha,
        beta,
        out,
        ..
    } = &model.layers[layer].attention
    else {
        return Err(-1);
    };
    batch_quantize(&mut batch.quantized, &batch.norm, count, Q38_HIDDEN)?;
    batch_project(
        &mut batch.wide0,
        &batch.norm,
        &batch.quantized,
        count,
        Q38_HIDDEN,
        qkv,
    )?;
    batch_project(
        &mut batch.wide1,
        &batch.norm,
        &batch.quantized,
        count,
        Q38_HIDDEN,
        z,
    )?;
    batch_project(
        &mut batch.beta,
        &batch.norm,
        &batch.quantized,
        count,
        Q38_HIDDEN,
        beta,
    )?;
    batch_project(
        &mut batch.alpha,
        &batch.norm,
        &batch.quantized,
        count,
        Q38_HIDDEN,
        alpha,
    )?;

    let index = recurrent_index(layer);
    let conv_base = index * Q38_LINEAR_QKV_DIM * 3;
    let delta_base = index * Q38_LINEAR_V_HEADS * Q38_LINEAR_HEAD_DIM * Q38_LINEAR_HEAD_DIM;
    let delta_len = Q38_LINEAR_V_HEADS * Q38_LINEAR_HEAD_DIM * Q38_LINEAR_HEAD_DIM;
    for token in 0..count {
        linear_core(
            &model.layers[layer],
            &mut model.conv_state[conv_base..conv_base + Q38_LINEAR_QKV_DIM * 3],
            &mut model.delta_state[delta_base..delta_base + delta_len],
            &mut batch.wide0[token * Q38_LINEAR_QKV_DIM..]
                [..Q38_LINEAR_QKV_DIM],
            &batch.wide1[token * Q38_LINEAR_V_DIM..][..Q38_LINEAR_V_DIM],
            &mut batch.beta[token * Q38_LINEAR_V_HEADS..][..Q38_LINEAR_V_HEADS],
            &mut batch.alpha[token * Q38_LINEAR_V_HEADS..][..Q38_LINEAR_V_HEADS],
            &mut batch.attention[token * Q38_LINEAR_V_DIM..][..Q38_LINEAR_V_DIM],
        )?;
    }

    batch_quantize(
        &mut batch.quantized,
        &batch.attention,
        count,
        Q38_LINEAR_V_DIM,
    )?;
    batch_project(
        &mut batch.branch,
        &batch.attention,
        &batch.quantized,
        count,
        Q38_LINEAR_V_DIM,
        out,
    )
}

fn batch_full_attention(
    model: &mut Q38Model<'_>,
    batch: &mut Q38BatchScratch,
    layer: usize,
    count: usize,
    base_position: u32,
) -> Result<(), i32> {
    let Q38Attention::Full { q, k, v, out, .. } = &model.layers[layer].attention else {
        return Err(-1);
    };
    batch_quantize(&mut batch.quantized, &batch.norm, count, Q38_HIDDEN)?;
    batch_project(
        &mut batch.wide0,
        &batch.norm,
        &batch.quantized,
        count,
        Q38_HIDDEN,
        q,
    )?;
    batch_project(
        &mut batch.key,
        &batch.norm,
        &batch.quantized,
        count,
        Q38_HIDDEN,
        k,
    )?;
    batch_project(
        &mut batch.value,
        &batch.norm,
        &batch.quantized,
        count,
        Q38_HIDDEN,
        v,
    )?;

    let context_length = model.context_length as usize;
    let key_cache = &mut model.key_cache;
    let value_cache = &mut model.value_cache;
    for token in 0..count {
        full_core(
            &model.layers[layer],
            &batch.wide0[token * Q38_ATTN_QG_DIM..][..Q38_ATTN_QG_DIM],
            &mut batch.key[token * Q38_ATTN_KV_DIM..][..Q38_ATTN_KV_DIM],
            &batch.value[token * Q38_ATTN_KV_DIM..][..Q38_ATTN_KV_DIM],
            &mut model.scratch.q,
            &mut model.scratch.gate,
            &mut model.scratch.scores,
            &mut batch.attention[token * Q38_ATTN_Q_DIM..][..Q38_ATTN_Q_DIM],
            key_cache,
            value_cache,
            context_length,
            base_position + token as u32,
            layer,
        )?;
    }

    batch_quantize(
        &mut batch.quantized,
        &batch.attention,
        count,
        Q38_ATTN_Q_DIM,
    )?;
    batch_project(
        &mut batch.branch,
        &batch.attention,
        &batch.quantized,
        count,
        Q38_ATTN_Q_DIM,
        out,
    )
}

fn batch_layer(
    model: &mut Q38Model<'_>,
    batch: &mut Q38BatchScratch,
    layer: usize,
    count: usize,
    base_position: u32,
) -> Result<(), i32> {
    batch_norm(
        &mut batch.norm,
        &batch.hidden,
        model.layers[layer].attn_norm,
        count,
    );
    if is_full_attention(layer) {
        batch_full_attention(model, batch, layer, count, base_position)?;
    } else {
        batch_linear_attention(model, batch, layer, count)?;
    }
    accumulate(&mut batch.hidden, &batch.branch, count * Q38_HIDDEN);

    batch_norm(
        &mut batch.norm,
        &batch.hidden,
        model.layers[layer].post_norm,
        count,
    );
    batch_quantize(&mut batch.quantized, &batch.norm, count, Q38_HIDDEN)?;
    batch_project(
        &mut batch.wide0,
        &batch.norm,
        &batch.quantized,
        count,
        Q38_HIDDEN,
        model.layers[layer].ffn_gate,
    )?;
    batch_project(
        &mut batch.wide1,
        &batch.norm,
        &batch.quantized,
        count,
        Q38_HIDDEN,
        model.layers[layer].ffn_up,
    )?;
    swiglu(&mut batch.wide0, &batch.wide1, count * Q38_FFN);
    batch_quantize(&mut batch.quantized, &batch.wide0, count, Q38_FFN)?;
    batch_project(
        &mut batch.branch,
        &batch.wide0,
        &batch.quantized,
        count,
        Q38_FFN,
        model.layers[layer].ffn_down,
    )?;
    accumulate(&mut batch.hidden, &batch.branch, count * Q38_HIDDEN);
    Ok(())
}

/// The MTP block mirrors a full-attention layer but reads/Writes the MTP slot
/// of the KV cache and the dedicated MTP position counter.
fn mtp_layer_forward(model: &mut Q38Model<'_>, position: u32) -> Result<(), i32> {
    if !model.mtp_weights_loaded {
        return Err(-1);
    }
    let weights = &model.layers[Q38_LAYERS];
    rmsnorm(
        &mut model.scratch.norm,
        &model.scratch.hidden,
        weights.attn_norm,
        Q38_HIDDEN,
    );
    full_attention(
        &model.layers[Q38_LAYERS],
        &mut model.scratch,
        &mut model.key_cache,
        &mut model.value_cache,
        model.context_length as usize,
        position,
        Q38_LAYERS,
    )?;
    accumulate(&mut model.scratch.hidden, &model.scratch.branch, Q38_HIDDEN);

    rmsnorm(
        &mut model.scratch.norm,
        &model.scratch.hidden,
        weights.post_norm,
        Q38_HIDDEN,
    );
    let input = &model.scratch.norm[..Q38_HIDDEN];
    quantize_input(&mut model.scratch.quantized, input)?;
    project(
        &mut model.scratch.wide0,
        input,
        &model.scratch.quantized,
        Q38_HIDDEN,
        weights.ffn_gate,
    )?;
    project(
        &mut model.scratch.wide1,
        input,
        &model.scratch.quantized,
        Q38_HIDDEN,
        weights.ffn_up,
    )?;
    swiglu(&mut model.scratch.wide0, &model.scratch.wide1, Q38_FFN);
    quantize_input(
        &mut model.scratch.quantized,
        &model.scratch.wide0[..Q38_FFN],
    )?;
    project(
        &mut model.scratch.branch,
        &model.scratch.wide0,
        &model.scratch.quantized,
        Q38_FFN,
        weights.ffn_down,
    )?;
    accumulate(&mut model.scratch.hidden, &model.scratch.branch, Q38_HIDDEN);
    Ok(())
}

/// Replay the tokens the target model committed through the MTP head so its
/// state stays level with the target's.
fn mtp_catchup(
    model: &mut Q38Model<'_>,
    batch: &mut Q38BatchScratch,
    tokens: &[u32],
    base_position: u32,
) -> Result<(), i32> {
    let count = tokens.len();
    if count == 0 {
        return Err(-1);
    }
    let (enorm, hnorm, eh_proj) = match &model.mtp {
        Some(mtp) => (mtp.enorm, mtp.hnorm, mtp.eh_proj),
        None => return Err(-1),
    };
    let embedding = model.mtp_embedding.ok_or(-1)?;
    for (token, &id) in tokens.iter().enumerate() {
        if id as usize >= Q38_VOCAB {
            return Err(-1);
        }
        let hidden = &mut batch.hidden[token * Q38_HIDDEN..(token + 1) * Q38_HIDDEN];
        embedding.tensor_row_f32(hidden, id as u64).map_err(|_| -1)?;
        let input = &mut batch.wide0[token * 2 * Q38_HIDDEN..][..2 * Q38_HIDDEN];
        rmsnorm(&mut input[..Q38_HIDDEN], hidden, enorm, Q38_HIDDEN);
        let previous = if token == 0 {
            &model.mtp_input_hidden[..Q38_HIDDEN]
        } else {
            &batch.target_norm[(token - 1) * Q38_HIDDEN..][..Q38_HIDDEN]
        };
        rmsnorm(&mut input[Q38_HIDDEN..], previous, hnorm, Q38_HIDDEN);
    }

    batch_quantize(&mut batch.quantized, &batch.wide0, count, 2 * Q38_HIDDEN)?;
    batch_project(
        &mut batch.hidden,
        &batch.wide0,
        &batch.quantized,
        count,
        2 * Q38_HIDDEN,
        eh_proj,
    )?;
    batch_mtp_layer(model, batch, count, base_position)?;
    model.mtp_position = base_position + count as u32;
    let last = (count - 1) * Q38_HIDDEN;
    model
        .mtp_input_hidden
        .copy_from_slice(&batch.target_norm[last..last + Q38_HIDDEN]);
    Ok(())
}

fn batch_mtp_layer(
    model: &mut Q38Model<'_>,
    batch: &mut Q38BatchScratch,
    count: usize,
    base_position: u32,
) -> Result<(), i32> {
    if !model.mtp_weights_loaded {
        return Err(-1);
    }
    let attn_norm = model.layers[Q38_LAYERS].attn_norm;
    batch_norm(&mut batch.norm, &batch.hidden, attn_norm, count);
    batch_full_attention(model, batch, Q38_LAYERS, count, base_position)?;
    accumulate(&mut batch.hidden, &batch.branch, count * Q38_HIDDEN);
    let post_norm = model.layers[Q38_LAYERS].post_norm;
    let gate = model.layers[Q38_LAYERS].ffn_gate;
    let up = model.layers[Q38_LAYERS].ffn_up;
    let down = model.layers[Q38_LAYERS].ffn_down;
    batch_norm(&mut batch.norm, &batch.hidden, post_norm, count);
    batch_quantize(&mut batch.quantized, &batch.norm, count, Q38_HIDDEN)?;
    batch_project(
        &mut batch.wide0,
        &batch.norm,
        &batch.quantized,
        count,
        Q38_HIDDEN,
        gate,
    )?;
    batch_project(
        &mut batch.wide1,
        &batch.norm,
        &batch.quantized,
        count,
        Q38_HIDDEN,
        up,
    )?;
    swiglu(&mut batch.wide0, &batch.wide1, count * Q38_FFN);
    batch_quantize(&mut batch.quantized, &batch.wide0, count, Q38_FFN)?;
    batch_project(
        &mut batch.branch,
        &batch.wide0,
        &batch.quantized,
        count,
        Q38_FFN,
        down,
    )?;
    accumulate(&mut batch.hidden, &batch.branch, count * Q38_HIDDEN);
    Ok(())
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

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
        quantized: (0..Q38_FFN / BLOCK).map(|_| Q38Q8KBlock::default()).collect(),
    }
}

fn argmax(values: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut maximum = values[0];
    for (index, value) in values.iter().enumerate().skip(1) {
        if *value > maximum {
            maximum = *value;
            best = index as u32;
        }
    }
    best
}

impl<'a> Q38Model<'a> {
    pub fn bind(gguf: &'a Gguf, context_length: u32) -> Result<Self, i32> {
        let embedding = gguf.find_tensor("token_embd.weight").ok_or(-1)?;
        let output_norm = gguf.find_tensor("output_norm.weight").ok_or(-1)?;
        let output = gguf.find_tensor("output.weight").ok_or(-1)?;

        let mut layers = Vec::with_capacity(Q38_TOTAL_LAYERS);
        for layer in 0..Q38_LAYERS {
            layers.push(bind_layer(gguf, layer)?);
        }
        validate_base_contract(gguf, embedding, output_norm, output, &layers)?;

        let conv_count = Q38_RECURRENT_LAYERS * Q38_LINEAR_QKV_DIM * 3;
        let delta_count =
            Q38_RECURRENT_LAYERS * Q38_LINEAR_V_HEADS * Q38_LINEAR_HEAD_DIM * Q38_LINEAR_HEAD_DIM;
        let kv_count = Q38_TOTAL_FULL_LAYERS * context_length as usize * Q38_ATTN_KV_DIM;

        let mut model = Q38Model {
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
        };

        // A merged checkpoint carries the MTP head as block 64.
        let block_count = gguf.meta_u32("qwen35.block_count").ok_or(-1)?;
        let nextn_layers = gguf.meta_u32("qwen35.nextn_predict_layers").ok_or(-1)?;
        if block_count == Q38_TOTAL_LAYERS as u32 && nextn_layers == 1 {
            model.bind_mtp_weights(gguf)?;
        }
        Ok(model)
    }

    fn bind_mtp_weights(&mut self, gguf: &'a Gguf) -> Result<(), i32> {
        let embedding = gguf.find_tensor("token_embd.weight").ok_or(-1)?;
        let output = gguf.find_tensor("output.weight").ok_or(-1)?;
        let layer = bind_layer(gguf, Q38_LAYERS)?;
        let mtp = Q38MTPWeights {
            eh_proj: layer_tensor(gguf, Q38_LAYERS, "nextn.eh_proj.weight")?,
            enorm: layer_tensor(gguf, Q38_LAYERS, "nextn.enorm.weight")?,
            hnorm: layer_tensor(gguf, Q38_LAYERS, "nextn.hnorm.weight")?,
            shared_head_norm: layer_tensor(gguf, Q38_LAYERS, "nextn.shared_head_norm.weight")?,
        };
        validate_mtp_contract(gguf, embedding, output, &layer, &mtp)?;

        if self.layers.len() == Q38_LAYERS {
            self.layers.push(layer);
        } else {
            self.layers[Q38_LAYERS] = layer;
        }
        self.mtp_embedding = Some(embedding);
        self.mtp_output = Some(output);
        self.mtp = Some(mtp);
        self.mtp_weights_loaded = true;
        Ok(())
    }

    /// Evaluate one token through the first `layer_count` blocks and hand back
    /// the resulting hidden state. A fresh/reset model is required when
    /// comparing layer boundaries.
    pub fn forward_token_layers(
        &mut self,
        token_id: u32,
        layer_count: usize,
        hidden_out: &mut [f32],
    ) -> Result<(), i32> {
        if token_id as usize >= Q38_VOCAB
            || layer_count > Q38_LAYERS
            || self.position >= self.context_length
            || hidden_out.len() < Q38_HIDDEN
        {
            return Err(-1);
        }
        let embedding = self.embedding;
        embedding
            .tensor_row_f32(&mut self.scratch.hidden, token_id as u64)
            .map_err(|_| -1)?;
        for layer in 0..layer_count {
            self.layer_forward(layer)?;
        }
        hidden_out[..Q38_HIDDEN].copy_from_slice(&self.scratch.hidden);
        self.position += 1;
        Ok(())
    }

    fn layer_forward(&mut self, layer: usize) -> Result<(), i32> {
        rmsnorm(
            &mut self.scratch.norm,
            &self.scratch.hidden,
            self.layers[layer].attn_norm,
            Q38_HIDDEN,
        );
        if is_full_attention(layer) {
            let context_length = self.context_length as usize;
            full_attention(
                &self.layers[layer],
                &mut self.scratch,
                &mut self.key_cache,
                &mut self.value_cache,
                context_length,
                self.position,
                layer,
            )?;
        } else {
            let index = recurrent_index(layer);
            let conv_base = index * Q38_LINEAR_QKV_DIM * 3;
            let delta_base =
                index * Q38_LINEAR_V_HEADS * Q38_LINEAR_HEAD_DIM * Q38_LINEAR_HEAD_DIM;
            let delta_len = Q38_LINEAR_V_HEADS * Q38_LINEAR_HEAD_DIM * Q38_LINEAR_HEAD_DIM;
            linear_attention(
                &self.layers[layer],
                &mut self.scratch,
                &mut self.conv_state[conv_base..conv_base + Q38_LINEAR_QKV_DIM * 3],
                &mut self.delta_state[delta_base..delta_base + delta_len],
            )?;
        }
        accumulate(&mut self.scratch.hidden, &self.scratch.branch, Q38_HIDDEN);

        let (gate, up, down) = (
            self.layers[layer].ffn_gate,
            self.layers[layer].ffn_up,
            self.layers[layer].ffn_down,
        );
        rmsnorm(
            &mut self.scratch.norm,
            &self.scratch.hidden,
            self.layers[layer].post_norm,
            Q38_HIDDEN,
        );
        let input = &self.scratch.norm[..Q38_HIDDEN];
        quantize_input(&mut self.scratch.quantized, input)?;
        project(
            &mut self.scratch.wide0,
            input,
            &self.scratch.quantized,
            Q38_HIDDEN,
            gate,
        )?;
        project(
            &mut self.scratch.wide1,
            input,
            &self.scratch.quantized,
            Q38_HIDDEN,
            up,
        )?;
        swiglu(&mut self.scratch.wide0, &self.scratch.wide1, Q38_FFN);
        quantize_input(&mut self.scratch.quantized, &self.scratch.wide0[..Q38_FFN])?;
        project(
            &mut self.scratch.branch,
            &self.scratch.wide0,
            &self.scratch.quantized,
            Q38_FFN,
            down,
        )?;
        accumulate(&mut self.scratch.hidden, &self.scratch.branch, Q38_HIDDEN);
        Ok(())
    }

    /// Snapshot everything `speculative_greedy` may have to roll back.
    fn checkpoint_state(&mut self) {
        self.checkpoint_conv_state.clear();
        self.checkpoint_conv_state
            .extend_from_slice(&self.conv_state);
        self.checkpoint_delta_state.clear();
        self.checkpoint_delta_state
            .extend_from_slice(&self.delta_state);
        self.checkpoint_mtp_hidden.clear();
        self.checkpoint_mtp_hidden
            .extend_from_slice(&self.mtp_input_hidden);
        self.checkpoint_position = self.position;
        self.checkpoint_mtp_position = self.mtp_position;
    }

    fn restore_target_state(&mut self) {
        self.conv_state.copy_from_slice(&self.checkpoint_conv_state);
        self.delta_state.copy_from_slice(&self.checkpoint_delta_state);
        self.position = self.checkpoint_position;
    }

    fn restore_mtp_state(&mut self) {
        self.mtp_input_hidden
            .copy_from_slice(&self.checkpoint_mtp_hidden);
        self.mtp_position = self.checkpoint_mtp_position;
    }
}

impl<'a> Q38ModelOps<'a> for Q38Model<'a> {
    fn open_gguf(gguf: &'a Gguf, context_length: u32) -> Result<Self, i32> {
        if context_length == 0 {
            return Err(-1);
        }
        Self::bind(gguf, context_length)
    }

    fn attach_mtp_gguf(&mut self, gguf: &'a Gguf) -> Result<(), i32> {
        if self.position != 0
            || self.mtp_position != 0
            || self.mtp_enabled
            || self.mtp_weights_loaded
            || self.mtp_gguf.is_some()
        {
            return Err(-1);
        }
        match self.bind_mtp_weights(gguf) {
            Ok(()) => {
                self.mtp_gguf = Some(gguf);
                Ok(())
            }
            Err(code) => {
                if self.layers.len() > Q38_LAYERS {
                    self.layers.truncate(Q38_LAYERS);
                }
                self.mtp = None;
                self.mtp_embedding = None;
                self.mtp_output = None;
                self.mtp_weights_loaded = false;
                Err(code)
            }
        }
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
        let embedding = self.embedding;
        embedding
            .tensor_row_f32(&mut self.scratch.hidden, token_id as u64)
            .map_err(|_| -1)?;
        for layer in 0..Q38_LAYERS {
            self.layer_forward(layer)?;
        }

        let output_norm = self.output_norm;
        let output = self.output;
        rmsnorm(
            &mut self.scratch.norm,
            &self.scratch.hidden,
            output_norm,
            Q38_HIDDEN,
        );
        if self.mtp_enabled {
            self.mtp_input_hidden
                .copy_from_slice(&self.scratch.norm[..Q38_HIDDEN]);
        }
        let input = &self.scratch.norm[..Q38_HIDDEN];
        quantize_input(&mut self.scratch.quantized, input)?;
        project(
            &mut self.scratch.logits,
            input,
            &self.scratch.quantized,
            Q38_HIDDEN,
            output,
        )?;
        self.position += 1;
        Ok(&self.scratch.logits)
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<&[f32], i32> {
        if tokens.is_empty() || tokens.len() as u32 > self.context_length - self.position {
            return Err(-1);
        }
        let count = tokens.len();
        let mut batch = Q38BatchScratch::new(count);
        let base_position = self.position;
        for (token, &id) in tokens.iter().enumerate() {
            if id as usize >= Q38_VOCAB {
                return Err(-1);
            }
            let embedding = self.embedding;
            embedding
                .tensor_row_f32(
                    &mut batch.hidden[token * Q38_HIDDEN..][..Q38_HIDDEN],
                    id as u64,
                )
                .map_err(|_| -1)?;
        }
        for layer in 0..Q38_LAYERS {
            batch_layer(self, &mut batch, layer, count, base_position)?;
        }
        let output_norm = self.output_norm;
        batch_norm(&mut batch.target_norm, &batch.hidden, output_norm, count);
        if self.mtp_enabled {
            mtp_catchup(self, &mut batch, tokens, base_position)?;
        }

        let last = &batch.target_norm[(count - 1) * Q38_HIDDEN..][..Q38_HIDDEN];
        self.scratch.norm[..Q38_HIDDEN].copy_from_slice(last);
        let output = self.output;
        let input = &self.scratch.norm[..Q38_HIDDEN];
        quantize_input(&mut self.scratch.quantized, input)?;
        project(
            &mut self.scratch.logits,
            input,
            &self.scratch.quantized,
            Q38_HIDDEN,
            output,
        )?;
        self.position = base_position + count as u32;
        Ok(&self.scratch.logits)
    }

    fn enable_mtp(&mut self) -> Result<(), i32> {
        if self.position != 0 || self.mtp_position != 0 {
            return Err(-1);
        }
        if !self.mtp_weights_loaded {
            eprintln!("qwen38: MTP weights are unavailable; attach the MTP GGUF first");
            return Err(-1);
        }
        let conv_count = Q38_RECURRENT_LAYERS * Q38_LINEAR_QKV_DIM * 3;
        let delta_count =
            Q38_RECURRENT_LAYERS * Q38_LINEAR_V_HEADS * Q38_LINEAR_HEAD_DIM * Q38_LINEAR_HEAD_DIM;
        self.checkpoint_conv_state = vec![0.0f32; conv_count];
        self.checkpoint_delta_state = vec![0.0f32; delta_count];
        self.checkpoint_mtp_hidden = vec![0.0f32; Q38_HIDDEN];
        self.mtp_enabled = true;
        Ok(())
    }

    fn mtp_forward(&mut self, token_id: u32) -> Result<&[f32], i32> {
        if !self.mtp_enabled
            || token_id as usize >= Q38_VOCAB
            || self.mtp_position >= self.context_length
        {
            return Err(-1);
        }
        let (enorm, hnorm, eh_proj, shared_head_norm) = match &self.mtp {
            Some(mtp) => (mtp.enorm, mtp.hnorm, mtp.eh_proj, mtp.shared_head_norm),
            None => return Err(-1),
        };
        let embedding = self.mtp_embedding.ok_or(-1)?;
        let output = self.mtp_output.ok_or(-1)?;

        embedding
            .tensor_row_f32(&mut self.scratch.hidden, token_id as u64)
            .map_err(|_| -1)?;
        rmsnorm(
            &mut self.scratch.wide0[..Q38_HIDDEN],
            &self.scratch.hidden[..Q38_HIDDEN],
            enorm,
            Q38_HIDDEN,
        );
        rmsnorm(
            &mut self.scratch.wide0[Q38_HIDDEN..][..Q38_HIDDEN],
            &self.mtp_input_hidden[..Q38_HIDDEN],
            hnorm,
            Q38_HIDDEN,
        );
        let joined = &self.scratch.wide0[..2 * Q38_HIDDEN];
        quantize_input(&mut self.scratch.quantized, joined)?;
        project(
            &mut self.scratch.hidden,
            joined,
            &self.scratch.quantized,
            2 * Q38_HIDDEN,
            eh_proj,
        )?;

        let position = self.mtp_position;
        mtp_layer_forward(self, position)?;
        rmsnorm(
            &mut self.scratch.norm,
            &self.scratch.hidden,
            shared_head_norm,
            Q38_HIDDEN,
        );
        self.mtp_input_hidden
            .copy_from_slice(&self.scratch.norm[..Q38_HIDDEN]);
        let input = &self.scratch.norm[..Q38_HIDDEN];
        quantize_input(&mut self.scratch.quantized, input)?;
        project(
            &mut self.scratch.logits,
            input,
            &self.scratch.quantized,
            Q38_HIDDEN,
            output,
        )?;
        self.mtp_position += 1;
        Ok(&self.scratch.logits)
    }

    fn speculative_greedy(
        &mut self,
        token_id: u32,
        stop_token: u32,
        draft_count: u32,
        accepted: &mut [u32],
    ) -> Result<(&[f32], u32), i32> {
        if !self.mtp_enabled
            || token_id as usize >= Q38_VOCAB
            || draft_count == 0
            || self.position != self.mtp_position
            || draft_count > self.context_length - self.position
            || accepted.len() < draft_count as usize
        {
            return Err(-1);
        }

        let capacity = draft_count as usize;
        let mut drafts = vec![0u32; capacity];
        let mut inputs = vec![0u32; capacity];
        let mut batch_logits = vec![0.0f32; capacity * Q38_VOCAB];

        self.checkpoint_state();
        let mut actual_count = 0usize;
        let mut draft_input = token_id;
        while actual_count < capacity {
            match self.mtp_forward(draft_input) {
                Ok(logits) => {
                    let next = argmax(logits);
                    draft_input = next;
                    drafts[actual_count] = next;
                    actual_count += 1;
                    if next == stop_token {
                        break;
                    }
                }
                Err(_) => {
                    self.restore_mtp_state();
                    return Err(-1);
                }
            }
        }

        inputs[0] = token_id;
        for i in 1..actual_count {
            inputs[i] = drafts[i - 1];
        }
        let mut batch = Q38BatchScratch::new(capacity);
        let mut ok = batch_target_eval(
            self,
            &mut batch,
            &inputs[..actual_count],
            &mut batch_logits[..actual_count * Q38_VOCAB],
        )
        .is_ok();

        let mut matched = 0usize;
        let mut next = 0u32;
        if ok {
            while matched < actual_count {
                let target = argmax(&batch_logits[matched * Q38_VOCAB..][..Q38_VOCAB]);
                if target != drafts[matched] {
                    next = target;
                    break;
                }
                matched += 1;
            }
            if matched == actual_count {
                next = drafts[actual_count - 1];
            }
        }
        let consumed = if matched < actual_count {
            matched + 1
        } else {
            actual_count
        };

        // The rejected tail must not stay in the recurrent state, so re-run
        // only the committed prefix from the checkpoint.
        if ok && consumed < actual_count {
            self.restore_target_state();
            ok = batch_target_eval(
                self,
                &mut batch,
                &inputs[..consumed],
                &mut batch_logits[..consumed * Q38_VOCAB],
            )
            .is_ok();
        }
        if ok {
            let last = &batch_logits[(consumed - 1) * Q38_VOCAB..][..Q38_VOCAB];
            self.scratch.logits.copy_from_slice(last);
            self.restore_mtp_state();
            let base = self.checkpoint_mtp_position;
            ok = mtp_catchup(self, &mut batch, &inputs[..consumed], base).is_ok();
        }

        if ok {
            for i in 0..matched {
                accepted[i] = drafts[i];
            }
            if matched < actual_count {
                accepted[matched] = next;
            }
            Ok((&self.scratch.logits, consumed as u32))
        } else {
            self.restore_target_state();
            self.restore_mtp_state();
            Err(-1)
        }
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

/// Evaluate `tokens` through the whole stack and produce the batch-major
/// logits, advancing the target state by `tokens.len()`.
fn batch_target_eval(
    model: &mut Q38Model<'_>,
    batch: &mut Q38BatchScratch,
    tokens: &[u32],
    logits: &mut [f32],
) -> Result<(), i32> {
    let count = tokens.len();
    if count == 0 || logits.len() < count * Q38_VOCAB {
        return Err(-1);
    }
    for (token, &id) in tokens.iter().enumerate() {
        if id as usize >= Q38_VOCAB {
            return Err(-1);
        }
        let embedding = model.embedding;
        embedding
            .tensor_row_f32(&mut batch.hidden[token * Q38_HIDDEN..][..Q38_HIDDEN], id as u64)
            .map_err(|_| -1)?;
    }
    let base_position = model.position;
    for layer in 0..Q38_LAYERS {
        batch_layer(model, batch, layer, count, base_position)?;
    }
    let output_norm = model.output_norm;
    let output = model.output;
    batch_norm(&mut batch.target_norm, &batch.hidden, output_norm, count);
    batch_quantize(&mut batch.quantized, &batch.target_norm, count, Q38_HIDDEN)?;
    batch_project(
        logits,
        &batch.target_norm,
        &batch.quantized,
        count,
        Q38_HIDDEN,
        output,
    )?;
    model.position = base_position + count as u32;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe_token() -> u32 {
        std::env::var("Q38_PROBE_TOKEN")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(1000)
    }

    fn probe_layers() -> Vec<usize> {
        std::env::var("Q38_PROBE_LAYERS")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .filter_map(|part| part.trim().parse().ok())
                    .collect()
            })
            .filter(|layers: &Vec<usize>| !layers.is_empty())
            .unwrap_or_else(|| vec![1, 4, 16, 32, 48, 64])
    }

    /// Mirrors `c_repo/src/cli/qwen38_forward_probe.c` so the two
    /// implementations can be diffed layer by layer.
    #[test]
    fn forward_layer_probe() {
        let gguf = Gguf::open("Qwen3.8-27B-UD-Q4_K_M.gguf").unwrap();
        let token = probe_token();
        for layers in probe_layers() {
            let mut model = Q38Model::open_gguf(&gguf, 16).unwrap();
            let mut hidden = vec![0.0f32; Q38_HIDDEN];
            model
                .forward_token_layers(token, layers, &mut hidden)
                .unwrap();
            let sum: f64 = hidden.iter().map(|value| *value as f64).sum();
            let abs: f64 = hidden.iter().map(|value| (*value as f64).abs()).sum();
            print!("layers={layers} sum={sum:.9} abs={abs:.9} first=");
            for (i, value) in hidden[..8].iter().enumerate() {
                print!("{}{value:.9}", if i > 0 { "," } else { "" });
            }
            println!();
        }
    }

    /// Dumps the next-token distribution for real prompts.
    ///
    /// A correct model puts the mass on a plausible continuation. A degenerate
    /// one shows it immediately: `NaN`/`inf` entries, an essentially flat
    /// distribution, or a special token (like EOS) winning the argmax on every
    /// prompt regardless of the question.
    ///
    /// Runs one prompt per weight pass, and a pass is memory-bandwidth bound,
    /// so this takes minutes rather than seconds.
    #[test]
    fn logits_probe() {
        let gguf = Gguf::open("Qwen3.8-27B-UD-Q4_K_M.gguf").unwrap();
        let tokenizer = crate::tokenizer::build_tokenizer_from_gguf(&gguf).unwrap();
        let eos = gguf.meta_u32("tokenizer.ggml.eos_token_id").unwrap_or(0);
        println!("eos={eos} vocab={}", Q38_VOCAB);

        let prompts: [(&str, &str); 3] = [
            (
                "arithmetic, thinking off",
                "<|im_start|>user\n1+1等于几？只回答数字。<|im_end|>\n\
                 <|im_start|>assistant\n<think>\n\n</think>\n\n",
            ),
            (
                "arithmetic, thinking on",
                "<|im_start|>user\n1+1等于几？<|im_end|>\n<|im_start|>assistant\n<think>\n",
            ),
            (
                "factual, thinking off",
                "<|im_start|>user\n中国的首都是哪座城市？只回答城市名。<|im_end|>\n\
                 <|im_start|>assistant\n<think>\n\n</think>\n\n",
            ),
        ];

        let mut model = Q38Model::open_gguf(&gguf, 256).unwrap();
        for (label, prompt) in prompts {
            model.reset();
            let ids: Vec<u32> = tokenizer
                .encode(prompt, false)
                .unwrap()
                .get_ids()
                .to_vec();
            let logits = model.prefill(&ids).unwrap().to_vec();

            let finite = logits.iter().filter(|value| value.is_finite()).count();
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mean = logits.iter().map(|value| *value as f64).sum::<f64>() / logits.len() as f64;
            let sd = (logits
                .iter()
                .map(|value| (*value as f64 - mean).powi(2))
                .sum::<f64>()
                / logits.len() as f64)
                .sqrt();

            println!(
                "\n== {label} == {} tokens, finite={finite}, max={max:.3}, mean={mean:.3}, sd={sd:.3}",
                ids.len()
            );
            let mut order: Vec<u32> = (0..logits.len() as u32).collect();
            order.sort_by(|a, b| logits[*b as usize].total_cmp(&logits[*a as usize]));
            for &id in order.iter().take(8) {
                let text = tokenizer.decode(&[id], false).unwrap_or_default();
                let logit = logits[id as usize];
                println!("  {id:>7} {logit:>10.4}  {text:?}");
            }
        }
    }
}
