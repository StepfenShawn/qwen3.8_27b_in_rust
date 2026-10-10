#![allow(unsafe_op_in_unsafe_fn)]

use crate::gguf::{GgmlType, TensorEntry};
use crate::kernel::iq;
use crate::kernel::quant::{bf16_to_f32, f16_to_f32};
use crate::kernel::{
    Q38CoreError, Q38CoreResult, Q38Q8KBlock, Q38TensorOps, Q38_Q8_K_BLOCK_SIZE,
    q38_parallel_over,
};
use rayon::prelude::*;

const BLOCK: usize = Q38_Q8_K_BLOCK_SIZE;

#[inline]
pub(crate) fn read_u16(data: &[u8]) -> u16 {
    u16::from_le_bytes([data[0], data[1]])
}

#[inline]
pub(crate) fn read_f32(data: &[u8]) -> f32 {
    f32::from_le_bytes([data[0], data[1], data[2], data[3]])
}

pub(crate) fn scale_min_k4(index: usize, packed: &[u8]) -> (u8, u8) {
    if index < 4 {
        (packed[index] & 63, packed[index + 4] & 63)
    } else {
        (
            (packed[index + 4] & 15) | ((packed[index - 4] >> 6) << 4),
            (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4),
        )
    }
}

pub(crate) fn unpack_q3_scales(packed: &[u8], scales: &mut [i8; 16]) {
    for index in 0..16 {
        let low = if index < 8 {
            packed[index] & 15
        } else {
            packed[index - 8] >> 4
        };
        let high = (packed[8 + index % 4] >> (2 * (index / 4))) & 3;
        scales[index] = ((low | (high << 4)) as i32 - 32) as i8;
    }
}

pub(crate) fn row_bytes(ty: GgmlType, width: usize) -> Option<usize> {
    let block_elements = ty.block_elements() as usize;
    let block_bytes = ty.block_bytes() as usize;
    if block_elements == 0 || block_bytes == 0 || width % block_elements != 0 {
        return None;
    }
    Some(width / block_elements * block_bytes)
}

fn supported_f32_gemv(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::Bf16
            | GgmlType::Q8_0
            | GgmlType::Q2K
            | GgmlType::Q3K
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
    ) || iq::supports_f32(ty)
}

fn supported_q8_k(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::Q2K | GgmlType::Q3K | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K
    ) || iq::supports_q8_k(ty)
}

// ---- scalar dot products (f32 input) ----

fn dot_f32_scalar(data: &[u8], input: &[f32], n: usize) -> f32 {
    let mut total = 0.0f32;
    for i in 0..n {
        total = read_f32(&data[i * 4..]).mul_add(input[i], total);
    }
    total
}

fn dot_f16_scalar(data: &[u8], input: &[f32], n: usize) -> f32 {
    let mut total = 0.0f32;
    for i in 0..n {
        total = f16_to_f32(read_u16(&data[i * 2..])).mul_add(input[i], total);
    }
    total
}

fn dot_bf16_scalar(data: &[u8], input: &[f32], n: usize) -> f32 {
    let mut total = 0.0f32;
    for i in 0..n {
        total = bf16_to_f32(&data[i * 2..]).mul_add(input[i], total);
    }
    total
}

pub(crate) fn dot_q8_0_scalar(data: &[u8], input: &[f32], n: usize) -> f32 {
    let mut total = 0.0f32;
    let mut base = 0usize;
    let mut block = 0usize;
    while base < n {
        let d = f16_to_f32(read_u16(&data[block * 34..]));
        for i in 0..32 {
            let q = data[block * 34 + 2 + i] as i8;
            total = (d * q as f32).mul_add(input[base + i], total);
        }
        base += 32;
        block += 1;
    }
    total
}

fn dot_q2_k_scalar(data: &[u8], input: &[f32], n: usize) -> f32 {
    let mut total = 0.0f32;
    let mut base = 0usize;
    let mut block = 0usize;
    while base < n {
        let b = block * 84;
        let scales = &data[b..b + 16];
        let quants = &data[b + 16..b + 80];
        let d = f16_to_f32(read_u16(&data[b + 80..]));
        let dmin = f16_to_f32(read_u16(&data[b + 82..]));
        let mut scale_index = 0usize;
        for half in 0..2 {
            for field in 0..4 {
                let shift = field * 2;
                for group in 0..2 {
                    let packed_scale = scales[scale_index];
                    scale_index += 1;
                    let scale = d * (packed_scale & 15) as f32;
                    let minimum = dmin * (packed_scale >> 4) as f32;
                    let offset = base + half * 128 + field * 32 + group * 16;
                    let packed = &quants[half * 32 + group * 16..half * 32 + group * 16 + 16];
                    for lane in 0..16 {
                        let q = (packed[lane] >> shift) & 3;
                        total = (scale * q as f32 - minimum).mul_add(input[offset + lane], total);
                    }
                }
            }
        }
        base += BLOCK;
        block += 1;
    }
    total
}

fn dot_q3_k_scalar(data: &[u8], input: &[f32], n: usize) -> f32 {
    let mut total = 0.0f32;
    let mut base = 0usize;
    let mut block = 0usize;
    while base < n {
        let b = block * 110;
        let high = &data[b..b + 32];
        let low = &data[b + 32..b + 96];
        let mut scales = [0i8; 16];
        // Q3_K packs its 16 six-bit scales into 12 bytes: 96..108. The `d`
        // half sits immediately after, at 108..110.
        unpack_q3_scales(&data[b + 96..b + 108], &mut scales);
        let d = f16_to_f32(read_u16(&data[b + 108..]));
        let mut group = 0usize;
        let mut high_bit: u8 = 1;
        for half in 0..2 {
            for field in 0..4 {
                let shift = field * 2;
                for lane in 0..32 {
                    let q = ((low[half * 32 + lane] >> shift) & 3) as i32
                        - if high[lane] & high_bit != 0 { 0 } else { 4 };
                    let scale = scales[group * 2 + lane / 16] as i32;
                    let offset = base + half * 128 + field * 32 + lane;
                    total =
                        (d * scale as f32 * q as f32).mul_add(input[offset], total);
                }
                group += 1;
                high_bit <<= 1;
            }
        }
        base += BLOCK;
        block += 1;
    }
    total
}

fn dot_q4_k_scalar(data: &[u8], input: &[f32], n: usize) -> f32 {
    let mut total = 0.0f32;
    let mut base = 0usize;
    let mut block = 0usize;
    while base < n {
        let b = block * 144;
        let d = f16_to_f32(read_u16(&data[b..]));
        let dmin = f16_to_f32(read_u16(&data[b + 2..]));
        let scales = &data[b + 4..b + 16];
        let mut quants = b + 16;
        let mut scale_index = 0usize;
        for chunk in 0..4 {
            let (scale0, min0) = scale_min_k4(scale_index, scales);
            let (scale1, min1) = scale_min_k4(scale_index + 1, scales);
            let d0 = d * scale0 as f32;
            let m0 = dmin * min0 as f32;
            let d1 = d * scale1 as f32;
            let m1 = dmin * min1 as f32;
            let x = base + chunk * 64;
            for i in 0..32 {
                total = (d0 * (data[quants + i] & 15) as f32 - m0).mul_add(input[x + i], total);
            }
            for i in 0..32 {
                total = (d1 * (data[quants + i] >> 4) as f32 - m1).mul_add(input[x + 32 + i], total);
            }
            quants += 32;
            scale_index += 2;
        }
        base += BLOCK;
        block += 1;
    }
    total
}

fn dot_q5_k_scalar(data: &[u8], input: &[f32], n: usize) -> f32 {
    let mut total = 0.0f32;
    let mut base = 0usize;
    let mut block = 0usize;
    while base < n {
        let b = block * 176;
        let d = f16_to_f32(read_u16(&data[b..]));
        let dmin = f16_to_f32(read_u16(&data[b + 2..]));
        let scales = &data[b + 4..b + 16];
        let high = &data[b + 16..b + 48];
        let low = &data[b + 48..b + 176];
        let mut scale_index = 0usize;
        let mut high0: u8 = 1;
        let mut high1: u8 = 2;
        for chunk in 0..4 {
            let (scale0, min0) = scale_min_k4(scale_index, scales);
            let (scale1, min1) = scale_min_k4(scale_index + 1, scales);
            let d0 = d * scale0 as f32;
            let m0 = dmin * min0 as f32;
            let d1 = d * scale1 as f32;
            let m1 = dmin * min1 as f32;
            let x = base + chunk * 64;
            for i in 0..32 {
                let q = (low[chunk * 32 + i] & 15) as i32
                    + if high[i] & high0 != 0 { 16 } else { 0 };
                total = (d0 * q as f32 - m0).mul_add(input[x + i], total);
            }
            for i in 0..32 {
                let q = (low[chunk * 32 + i] >> 4) as i32
                    + if high[i] & high1 != 0 { 16 } else { 0 };
                total = (d1 * q as f32 - m1).mul_add(input[x + 32 + i], total);
            }
            scale_index += 2;
            high0 <<= 2;
            high1 <<= 2;
        }
        base += BLOCK;
        block += 1;
    }
    total
}

fn dot_q6_k_scalar(data: &[u8], input: &[f32], n: usize) -> f32 {
    let mut total = 0.0f32;
    let mut base = 0usize;
    let mut block = 0usize;
    while base < n {
        let b = block * 210;
        let d = f16_to_f32(read_u16(&data[b + 208..]));
        let scales = &data[b + 192..b + 208];
        for half in 0..2 {
            let x = base + half * 128;
            for i in 0..32 {
                let si = half * 8 + i / 16;
                let low_i = data[b + half * 64 + i];
                let low_i2 = data[b + half * 64 + i + 32];
                let h = data[b + 128 + half * 32 + i];
                let q0 = ((low_i & 15) | ((h & 3) << 4)) as i32 - 32;
                let q1 = ((low_i2 & 15) | (((h >> 2) & 3) << 4)) as i32 - 32;
                let q2 = ((low_i >> 4) | (((h >> 4) & 3) << 4)) as i32 - 32;
                let q3 = ((low_i2 >> 4) | (((h >> 6) & 3) << 4)) as i32 - 32;
                total = (d * scales[si] as i8 as f32 * q0 as f32).mul_add(input[x + i], total);
                total = (d * scales[si + 2] as i8 as f32 * q1 as f32).mul_add(input[x + 32 + i], total);
                total = (d * scales[si + 4] as i8 as f32 * q2 as f32).mul_add(input[x + 64 + i], total);
                total = (d * scales[si + 6] as i8 as f32 * q3 as f32).mul_add(input[x + 96 + i], total);
            }
        }
        base += BLOCK;
        block += 1;
    }
    total
}

// ---- scalar dot products (Q8_K input) ----

fn dot_q2_k_q8_k_scalar(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut total = 0.0f32;
    for block in 0..blocks {
        let b = &q[block];
        let scales = &data[block * 84..block * 84 + 16];
        let quants = &data[block * 84 + 16..block * 84 + 80];
        let d = f16_to_f32(read_u16(&data[block * 84 + 80..])) * b.scale;
        let dmin = f16_to_f32(read_u16(&data[block * 84 + 82..])) * b.scale;
        let mut weighted = 0i32;
        let mut minimum = 0i32;
        for half in 0..2 {
            for field in 0..4 {
                let scale_index = half * 8 + field * 2;
                let scale0 = (scales[scale_index] & 15) as i32;
                let scale1 = (scales[scale_index + 1] & 15) as i32;
                let shift = field * 2;
                let mut dot0 = 0i32;
                let mut dot1 = 0i32;
                for lane in 0..16 {
                    dot0 += ((quants[half * 32 + lane] >> shift) & 3) as i32
                        * b.quants[half * 128 + field * 32 + lane] as i32;
                    dot1 += ((quants[half * 32 + 16 + lane] >> shift) & 3) as i32
                        * b.quants[half * 128 + field * 32 + 16 + lane] as i32;
                }
                weighted += scale0 * dot0 + scale1 * dot1;
                minimum += (scales[scale_index] >> 4) as i32 * b.sums[scale_index] as i32;
                minimum += (scales[scale_index + 1] >> 4) as i32 * b.sums[scale_index + 1] as i32;
            }
        }
        total += d * weighted as f32 - dmin * minimum as f32;
    }
    total
}

pub(crate) fn dot_q3_k_q8_k_scalar(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut total = 0.0f32;
    for block in 0..blocks {
        let b = &q[block];
        let high = &data[block * 110..block * 110 + 32];
        let low = &data[block * 110 + 32..block * 110 + 96];
        let mut scales = [0i8; 16];
        unpack_q3_scales(&data[block * 110 + 96..block * 110 + 108], &mut scales);
        let d = f16_to_f32(read_u16(&data[block * 110 + 108..])) * b.scale;
        let mut weighted = 0i32;
        let mut group = 0usize;
        for half in 0..2 {
            for field in 0..4 {
                let high_bit: u8 = 1 << group;
                let scale0 = scales[group * 2] as i32;
                let scale1 = scales[group * 2 + 1] as i32;
                for lane in 0..32 {
                    let qv = ((low[half * 32 + lane] >> (field * 2)) & 3) as i32
                        - if high[lane] & high_bit != 0 { 0 } else { 4 };
                    let scale = if lane < 16 { scale0 } else { scale1 };
                    weighted += scale * qv * b.quants[half * 128 + field * 32 + lane] as i32;
                }
                group += 1;
            }
        }
        total += d * weighted as f32;
    }
    total
}

fn dot_q4_k_q8_k_scalar(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut total = 0.0f32;
    for block in 0..blocks {
        let b = &q[block];
        let d = f16_to_f32(read_u16(&data[block * 144..])) * b.scale;
        let dmin = f16_to_f32(read_u16(&data[block * 144 + 2..])) * b.scale;
        let scales = &data[block * 144 + 4..block * 144 + 16];
        let quants = &data[block * 144 + 16..block * 144 + 144];
        let mut weighted = 0i32;
        let mut minimum = 0i32;
        for chunk in 0..4 {
            let (scale0, min0) = scale_min_k4(chunk * 2, scales);
            let (scale1, min1) = scale_min_k4(chunk * 2 + 1, scales);
            let mut dot0 = 0i32;
            let mut dot1 = 0i32;
            for i in 0..32 {
                dot0 += (quants[chunk * 32 + i] & 15) as i32 * b.quants[chunk * 64 + i] as i32;
                dot1 += (quants[chunk * 32 + i] >> 4) as i32 * b.quants[chunk * 64 + 32 + i] as i32;
            }
            weighted += scale0 as i32 * dot0 + scale1 as i32 * dot1;
            minimum += min0 as i32 * (b.sums[chunk * 4] + b.sums[chunk * 4 + 1]) as i32;
            minimum += min1 as i32 * (b.sums[chunk * 4 + 2] + b.sums[chunk * 4 + 3]) as i32;
        }
        total += d * weighted as f32 - dmin * minimum as f32;
    }
    total
}

pub(crate) fn dot_q5_k_q8_k_scalar(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut total = 0.0f32;
    for block in 0..blocks {
        let b = &q[block];
        let d = f16_to_f32(read_u16(&data[block * 176..])) * b.scale;
        let dmin = f16_to_f32(read_u16(&data[block * 176 + 2..])) * b.scale;
        let scales = &data[block * 176 + 4..block * 176 + 16];
        let high = &data[block * 176 + 16..block * 176 + 48];
        let low = &data[block * 176 + 48..block * 176 + 176];
        let mut minimum = 0i32;
        let mut weighted = 0i32;
        let mut high0: u8 = 1;
        let mut high1: u8 = 2;
        for chunk in 0..4 {
            let (scale0, min0) = scale_min_k4(chunk * 2, scales);
            let (scale1, min1) = scale_min_k4(chunk * 2 + 1, scales);
            let mut dot0 = 0i32;
            let mut dot1 = 0i32;
            for i in 0..32 {
                let qv0 = (low[chunk * 32 + i] & 15) as i32
                    + if high[i] & high0 != 0 { 16 } else { 0 };
                let qv1 = (low[chunk * 32 + i] >> 4) as i32
                    + if high[i] & high1 != 0 { 16 } else { 0 };
                dot0 += qv0 * b.quants[chunk * 64 + i] as i32;
                dot1 += qv1 * b.quants[chunk * 64 + 32 + i] as i32;
            }
            weighted += scale0 as i32 * dot0 + scale1 as i32 * dot1;
            minimum += min0 as i32 * (b.sums[chunk * 4] + b.sums[chunk * 4 + 1]) as i32;
            minimum += min1 as i32 * (b.sums[chunk * 4 + 2] + b.sums[chunk * 4 + 3]) as i32;
            high0 <<= 2;
            high1 <<= 2;
        }
        total += d * weighted as f32 - dmin * minimum as f32;
    }
    total
}

fn dot_q6_k_q8_k_scalar(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut total = 0.0f32;
    for block in 0..blocks {
        let b = &q[block];
        let d = f16_to_f32(read_u16(&data[block * 210 + 208..])) * b.scale;
        let scales = &data[block * 210 + 192..block * 210 + 208];
        let mut weighted = 0i32;
        for half in 0..2 {
            for i in 0..32 {
                let si = half * 8 + i / 16;
                let base = half * 128;
                let low_i = data[block * 210 + half * 64 + i];
                let low_i2 = data[block * 210 + half * 64 + i + 32];
                let h = data[block * 210 + 128 + half * 32 + i];
                let q0 = ((low_i & 15) | ((h & 3) << 4)) as i32 - 32;
                let q1 = ((low_i2 & 15) | (((h >> 2) & 3) << 4)) as i32 - 32;
                let q2 = ((low_i >> 4) | (((h >> 4) & 3) << 4)) as i32 - 32;
                let q3 = ((low_i2 >> 4) | (((h >> 6) & 3) << 4)) as i32 - 32;
                weighted += scales[si] as i8 as i32 * q0 * b.quants[base + i] as i32;
                weighted += scales[si + 2] as i8 as i32 * q1 * b.quants[base + 32 + i] as i32;
                weighted += scales[si + 4] as i8 as i32 * q2 * b.quants[base + 64 + i] as i32;
                weighted += scales[si + 6] as i8 as i32 * q3 * b.quants[base + 96 + i] as i32;
            }
        }
        total += d * weighted as f32;
    }
    total
}

// ---- runtime dispatch ----

#[inline]
fn dot_f32(data: &[u8], input: &[f32], n: usize) -> f32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if let Some(v) = crate::kernel::simd::tensor::try_dot_f32(data, input, n) {
        return v;
    }
    dot_f32_scalar(data, input, n)
}

#[inline]
fn dot_f16(data: &[u8], input: &[f32], n: usize) -> f32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if let Some(v) = crate::kernel::simd::tensor::try_dot_f16(data, input, n) {
        return v;
    }
    dot_f16_scalar(data, input, n)
}

#[inline]
fn dot_q4_k_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if let Some(v) = crate::kernel::simd::tensor::try_dot_q4_k_q8_k(data, q, blocks) {
        return v;
    }
    dot_q4_k_q8_k_scalar(data, q, blocks)
}

#[inline]
fn dot_q6_k_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if let Some(v) = crate::kernel::simd::tensor::try_dot_q6_k_q8_k(data, q, blocks) {
        return v;
    }
    dot_q6_k_q8_k_scalar(data, q, blocks)
}

#[inline]
fn dot_q3_k_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if let Some(v) = crate::kernel::simd::tensor::try_dot_q3_k_q8_k(data, q, blocks) {
        return v;
    }
    dot_q3_k_q8_k_scalar(data, q, blocks)
}

#[inline]
fn dot_q5_k_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if let Some(v) = crate::kernel::simd::tensor::try_dot_q5_k_q8_k(data, q, blocks) {
        return v;
    }
    dot_q5_k_q8_k_scalar(data, q, blocks)
}

#[inline]
fn dot_q8_0(data: &[u8], input: &[f32], n: usize) -> f32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if let Some(v) = crate::kernel::simd::tensor::try_dot_q8_0(data, input, n) {
        return v;
    }
    dot_q8_0_scalar(data, input, n)
}

fn dot_row(data: &[u8], input: &[f32], n: usize, ty: GgmlType) -> f32 {
    match ty {
        GgmlType::F32 => dot_f32(data, input, n),
        GgmlType::F16 => dot_f16(data, input, n),
        GgmlType::Bf16 => dot_bf16_scalar(data, input, n),
        GgmlType::Q8_0 => dot_q8_0(data, input, n),
        GgmlType::Q2K => dot_q2_k_scalar(data, input, n),
        GgmlType::Q3K => dot_q3_k_scalar(data, input, n),
        GgmlType::Q4K => dot_q4_k_scalar(data, input, n),
        GgmlType::Q5K => dot_q5_k_scalar(data, input, n),
        GgmlType::Q6K => dot_q6_k_scalar(data, input, n),
        ty if iq::supports_f32(ty) => iq::dot_row_f32(data, input, n, ty),
        _ => f32::NAN,
    }
}

fn dot_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize, ty: GgmlType) -> f32 {
    match ty {
        GgmlType::Q2K => dot_q2_k_q8_k_scalar(data, q, blocks),
        GgmlType::Q3K => dot_q3_k_q8_k(data, q, blocks),
        GgmlType::Q4K => dot_q4_k_q8_k(data, q, blocks),
        GgmlType::Q5K => dot_q5_k_q8_k(data, q, blocks),
        GgmlType::Q6K => dot_q6_k_q8_k(data, q, blocks),
        ty if iq::supports_q8_k(ty) => iq::dot_q8_k(data, q, blocks, ty),
        _ => f32::NAN,
    }
}

fn dequantize_quant_row(data: &[u8], output: &mut [f32], ty: GgmlType) {
    if iq::supports_f32(ty) {
        iq::dequantize_row(data, output, ty);
        return;
    }
    let stride = ty.block_elements() as usize;
    let block_bytes = ty.block_bytes() as usize;
    let mut basis = [0.0f32; BLOCK];
    for (block_index, block) in output.chunks_mut(stride).enumerate() {
        for i in 0..stride {
            basis[i] = 1.0;
            block[i] = dot_row(
                &data[block_index * block_bytes..(block_index + 1) * block_bytes],
                &basis[..stride],
                stride,
                ty,
            );
            basis[i] = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::Gguf;
    use crate::kernel::quant::quantize_q8_k;

    /// Deterministic activation vector shared by every per-type check.
    fn activation(width: usize) -> Vec<f32> {
        (0..width)
            .map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0)
            .collect()
    }

    fn relative(a: f32, b: f32) -> f32 {
        (a - b).abs() / b.abs().max(1.0)
    }

    /// Cross-validates the integer (`dot_q8_k`) and decoded (`dot_row`) paths
    /// against a plain f32 dot over the dequantized row, for *every*
    /// quantization type the checkpoint actually uses.
    ///
    /// `dot_row` and `dot_q8_k` are independent implementations of the same
    /// product, so their agreement is a real oracle without needing the C
    /// reference. Each row is sliced to exactly `row_bytes`: a kernel that
    /// reads past its last block therefore trips a bounds check here instead of
    /// silently corrupting a run. Row 0 and the final row are both covered
    /// because an over-read only runs off the end of the tensor at the last
    /// row.
    #[test]
    fn kernel_consistency() {
        let gguf = Gguf::open("Qwen3.8-27B-UD-Q4_K_M.gguf").unwrap();

        // One representative per distinct quant type, in a stable order.
        let mut representatives: Vec<(String, &TensorEntry<'static>)> = Vec::new();
        let mut types_seen = 0usize;
        for tensor in gguf.tensors() {
            if tensor.n_dims != 2 || tensor.shape[0] == 0 || tensor.shape[1] == 0 {
                continue;
            }
            if !supported_f32_gemv(tensor.ty) {
                continue;
            }
            if row_bytes(tensor.ty, tensor.shape[0] as usize).is_none() {
                continue;
            }
            types_seen += 1;
            let key = format!("{:?}", tensor.ty);
            if !representatives.iter().any(|(seen, _)| *seen == key) {
                representatives.push((key, tensor));
            }
        }
        assert!(!representatives.is_empty(), "no supported 2-D tensors found");
        println!(
            "{types_seen} 2-D tensors across {} distinct quant types",
            representatives.len()
        );

        for (key, tensor) in &representatives {
            let width = tensor.shape[0] as usize;
            let rows = tensor.shape[1];
            let rbytes = row_bytes(tensor.ty, width).unwrap();
            let x = activation(width);
            // The integer path quantizes the activation in 256-wide Q8_K
            // blocks, which narrow tensors (`ssm_conv1d.weight`, width 4) never
            // take. `project` only routes block-aligned matrices there.
            let q8_k_applicable = supported_q8_k(tensor.ty) && width % BLOCK == 0;
            let mut blocks = (0..width / BLOCK)
                .map(|_| Q38Q8KBlock::default())
                .collect::<Vec<_>>();
            if q8_k_applicable {
                quantize_q8_k(&mut blocks, &x).unwrap();
            }

            for row in [0u64, rows - 1] {
                let at = row as usize * rbytes;
                let data = &tensor.data[at..at + rbytes];

                let mut w = vec![0.0f32; width];
                tensor.tensor_row_f32(&mut w, row).unwrap();
                let reference: f32 = w.iter().zip(&x).map(|(a, b)| a * b).sum();

                let via_f32 = dot_row(data, &x, width, tensor.ty);
                assert!(
                    relative(via_f32, reference) < 1e-3,
                    "{key} row {row}: decoded path {} vs dequantized dot {reference}",
                    via_f32
                );

                if q8_k_applicable {
                    let via_q8_k = dot_q8_k(data, &blocks, width / BLOCK, tensor.ty);
                    assert!(
                        relative(via_q8_k, reference) < 0.05,
                        "{key} row {row}: q8_k path {} vs dequantized dot {reference}",
                        via_q8_k
                    );
                    println!(
                        "{key} row={row}: ref={reference} f32={via_f32} q8_k={via_q8_k} \
                         rel_f32={:.2e} rel_q8={:.2e}",
                        relative(via_f32, reference),
                        relative(via_q8_k, reference)
                    );
                } else {
                    println!(
                        "{key} row={row}: ref={reference} f32={via_f32} \
                         rel_f32={:.2e} (f32-only type)",
                        relative(via_f32, reference)
                    );
                }
            }
        }
    }

    /// Every supported type must stay inside its declared byte span.
    ///
    /// Two bugs lived here and neither was visible the usual way: the Q3_K
    /// kernels read four bytes past the end of a block's 12-byte scale array,
    /// and the IQ4_NL kernels iterate 256-element super-blocks while the type
    /// table declared 32-element ones. Both only fault once the *last* block of
    /// a row runs off the end of the tensor, and only for the types the current
    /// checkpoint happens to use. Sizing each synthetic row to exactly
    /// `row_bytes` turns any over-read into a bounds panic for every supported
    /// type, and needs no model file, so it also guards types this checkpoint
    /// does not ship.
    #[test]
    fn kernel_byte_span_is_contained() {
        use crate::gguf::{GGUF_MAX_DIMS, GgufString};

        let width = 2048usize;
        let types = [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::Bf16,
            GgmlType::Q8_0,
            GgmlType::Q2K,
            GgmlType::Q3K,
            GgmlType::Q4K,
            GgmlType::Q5K,
            GgmlType::Q6K,
            GgmlType::Iq1S,
            GgmlType::Iq1M,
            GgmlType::Iq2Xxs,
            GgmlType::Iq2Xs,
            GgmlType::Iq2S,
            GgmlType::Iq3Xxs,
            GgmlType::Iq3S,
            GgmlType::Iq4Xs,
            GgmlType::Iq4Nl,
        ];

        let x = activation(width);
        let mut blocks = (0..width / BLOCK)
            .map(|_| Q38Q8KBlock::default())
            .collect::<Vec<_>>();
        quantize_q8_k(&mut blocks, &x).unwrap();

        let mut checked = 0;
        for ty in types {
            if !supported_f32_gemv(ty) {
                continue;
            }
            let rbytes = row_bytes(ty, width).unwrap();
            // All-zero weights keep every scale finite, and the exact length
            // is what makes an over-read observable.
            let data = vec![0u8; rbytes];
            let mut shape = [0u64; GGUF_MAX_DIMS];
            shape[0] = width as u64;
            shape[1] = 1;
            let tensor = TensorEntry {
                name: GgufString {
                    data: b"probe".as_slice(),
                },
                n_dims: 2,
                shape,
                ty,
                offset: 0,
                nbytes: rbytes as u64,
                data: &data,
                iq1_s_repack: None,
            };

            let mut decoded = vec![0.0f32; width];
            tensor.tensor_row_f32(&mut decoded, 0).unwrap();
            let via_f32 = dot_row(&data, &x, width, ty);
            let via_q8_k = if supported_q8_k(ty) {
                dot_q8_k(&data, &blocks, width / BLOCK, ty)
            } else {
                f32::NAN
            };

            assert!(
                decoded.iter().all(|value| value.is_finite()),
                "{ty:?}: decoded row is not finite"
            );
            assert!(via_f32.is_finite(), "{ty:?}: f32 dot is not finite");
            assert!(
                via_q8_k.is_nan() || via_q8_k.is_finite(),
                "{ty:?}: q8_k dot is not finite"
            );
            println!("{ty:?}: {rbytes} byte row, dot={via_f32}, q8_k={via_q8_k}");
            checked += 1;
        }
        assert!(checked >= 15, "only {checked} types were exercised");
    }

    /// Locks down the parallel row loops.
    ///
    /// The pool-driven GEMV has to reproduce exactly what walking the rows one
    /// at a time produces, and the batched variants have to land their results
    /// at `token * rows + row` — that is the layout the model reads back, and
    /// it is the one part of the parallel path Rust cannot express with safe
    /// slicing (see `RowTargets`). Agreement is bitwise: every output element
    /// is still the same `dot_*` call on the same bytes, only issued from a
    /// different thread.
    #[test]
    fn gemv_and_gemm_match_serial_row_dot() {
        let gguf = Gguf::open("Qwen3.8-27B-UD-Q4_K_M.gguf").unwrap();
        let batch = 3usize;

        // The narrow projections are deliberately left off the pool, so a
        // matrix that only clears the row count but not the work gate has to
        // stay serial.
        let narrow = gguf.find_tensor("blk.0.ssm_alpha.weight").unwrap();
        assert_eq!(narrow.shape[1], 48);
        assert!(
            !q38_parallel_over(narrow.shape[1] as usize, narrow.shape[0] as usize),
            "a 48-row projection must stay on the calling thread"
        );

        for name in [
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_down.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.ffn_up.weight",
        ] {
            let tensor = gguf.find_tensor(name).unwrap();
            assert!(supported_f32_gemv(tensor.ty), "{name}: unsupported type");
            let width = tensor.shape[0] as usize;
            let rows = tensor.shape[1] as usize;
            let rbytes = row_bytes(tensor.ty, width).unwrap();
            assert!(
                q38_parallel_over(rows, width),
                "{name}: {rows}x{width} would not be split across the pool"
            );

            let x = activation(width);
            // Row 0 and the last row bracket the strided writes; the middle row
            // catches a chunk-boundary mix-up in the batched kernel.
            let probes = [0usize, 1, rows / 2, rows - 2, rows - 1];
            let row_of = |row: usize| &tensor.data[row * rbytes..(row + 1) * rbytes];

            // ---- decoded f32 path -------------------------------------------
            let mut gemv = vec![0.0f32; rows];
            tensor.tensor_gemv_f32(&mut gemv, &x).unwrap();
            for &row in &probes {
                let expected = dot_row(row_of(row), &x, width, tensor.ty);
                assert_eq!(
                    gemv[row].to_bits(),
                    expected.to_bits(),
                    "{name} ({:?}) gemv row {row}",
                    tensor.ty
                );
            }

            let mut tokens = vec![0.0f32; batch * width];
            for token in 0..batch {
                let lane = &mut tokens[token * width..(token + 1) * width];
                for (i, value) in lane.iter_mut().enumerate() {
                    *value = x[i] + token as f32 * 0.125;
                }
            }
            let mut gemm = vec![0.0f32; batch * rows];
            tensor.tensor_gemm_f32(&mut gemm, &tokens, batch as u32).unwrap();
            for &row in &probes {
                for token in 0..batch {
                    let expected = dot_row(
                        row_of(row),
                        &tokens[token * width..(token + 1) * width],
                        width,
                        tensor.ty,
                    );
                    assert_eq!(
                        gemm[token * rows + row].to_bits(),
                        expected.to_bits(),
                        "{name} ({:?}) gemm token {token} row {row}",
                        tensor.ty
                    );
                }
            }

            // ---- Q8_K integer path ------------------------------------------
            if !supported_q8_k(tensor.ty) {
                println!("{name} ({:?}): f32 path only", tensor.ty);
                continue;
            }
            let blocks = width / BLOCK;
            let mut quantized: Vec<Q38Q8KBlock> =
                (0..blocks).map(|_| Q38Q8KBlock::default()).collect();
            quantize_q8_k(&mut quantized, &x).unwrap();
            let mut gemv = vec![0.0f32; rows];
            tensor.tensor_gemv_q8_k(&mut gemv, &quantized).unwrap();
            for &row in &probes {
                let expected = dot_q8_k(row_of(row), &quantized, blocks, tensor.ty);
                assert_eq!(
                    gemv[row].to_bits(),
                    expected.to_bits(),
                    "{name} ({:?}) gemv_q8_k row {row}",
                    tensor.ty
                );
            }

            let mut wide: Vec<Q38Q8KBlock> =
                (0..batch * blocks).map(|_| Q38Q8KBlock::default()).collect();
            for token in 0..batch {
                let mut per_token = x.clone();
                for value in per_token.iter_mut() {
                    *value += token as f32 * 0.125;
                }
                quantize_q8_k(&mut wide[token * blocks..(token + 1) * blocks], &per_token).unwrap();
            }
            assert!(q38_parallel_over(rows, width * batch));
            let mut gemm = vec![0.0f32; batch * rows];
            tensor.tensor_gemm_q8_k(&mut gemm, &wide, batch as u32).unwrap();
            for &row in &probes {
                for token in 0..batch {
                    let expected = dot_q8_k(
                        row_of(row),
                        &wide[token * blocks..(token + 1) * blocks],
                        blocks,
                        tensor.ty,
                    );
                    assert_eq!(
                        gemm[token * rows + row].to_bits(),
                        expected.to_bits(),
                        "{name} ({:?}) gemm_q8_k token {token} row {row}",
                        tensor.ty
                    );
                }
            }
            println!("{name} ({:?}): {rows}x{width} serial == parallel", tensor.ty);
        }
    }
}

impl TensorEntry<'_> {
    /// Repacked IQ1_S view of this tensor, when [`crate::kernel::Q38Iq1sRepack`]
    /// built one and it covers every row of every block.
    #[inline]
    fn repacked_iq1_s(&self, blocks: usize) -> Option<&[u8]> {
        if self.ty != GgmlType::Iq1S {
            return None;
        }
        let repacked = self.iq1_s_repack.as_deref()?;
        let rows = self.shape[1] as usize;
        if repacked.len() < rows * blocks * iq::IQ1S_REPACK_BLOCK_BYTES {
            return None;
        }
        Some(repacked)
    }
}

/// Runs `dot(row)` for every row, in order, and stores the result.
///
/// The row is the outer loop because the weight row is what is being streamed
/// out of the 16 GB mapping: visiting row `r` once and reusing it for the whole
/// batch is what keeps the matrix from being re-read per token. Splitting by
/// row rather than by token preserves that order, and the result is
/// bit-identical to the serial walk because each output element is still the
/// same `dot` call.
fn gemv_rows<F>(output: &mut [f32], rows: usize, per_row: usize, dot: F)
where
    F: Fn(usize) -> f32 + Sync,
{
    let output = &mut output[..rows];
    if q38_parallel_over(rows, per_row) {
        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(row, slot)| *slot = dot(row));
    } else {
        for (row, slot) in output.iter_mut().enumerate() {
            *slot = dot(row);
        }
    }
}

/// Same as [`gemv_rows`], for the batched kernels whose output is token-major:
/// `output[token * rows + row]`.
///
/// Rust has no safe way to spell "column block of a token-major slice", so the
/// row loop writes through [`RowTargets`] rather than reordering the loop to
/// put the token outermost — that would re-read the whole matrix once per
/// token and undo the single-pass advantage.
fn gemm_rows<F>(output: &mut [f32], rows: usize, batch: usize, per_row: usize, dot: F)
where
    F: Fn(usize, usize) -> f32 + Sync,
{
    if q38_parallel_over(rows, per_row) {
        let targets = RowTargets {
            base: output.as_mut_ptr(),
            rows,
        };
        (0..rows).into_par_iter().for_each(|row| {
            for token in 0..batch {
                let value = dot(row, token);
                // SAFETY: `gemm_rows` only calls `at` with its own `row`, and
                // `RowTargets`'s contract makes those destinations disjoint.
                unsafe { targets.at(row, token).write(value) };
            }
        });
    } else {
        for row in 0..rows {
            for token in 0..batch {
                output[token * rows + row] = dot(row, token);
            }
        }
    }
}

/// The strided write view the batched row loop needs.
///
/// Row `r` owns exactly `{token * rows + r | token < batch}`, and `rows` is the
/// stride, so two different rows can never name the same slot. That
/// disjointness is what makes handing one row per worker sound, and it is the
/// only invariant this type carries.
#[derive(Clone, Copy)]
struct RowTargets {
    base: *mut f32,
    rows: usize,
}

// SAFETY: `RowTargets` is only ever copied into workers that each own a
// distinct `row`, so no two threads write the same address.
unsafe impl Send for RowTargets {}
// SAFETY: see `Send`.
unsafe impl Sync for RowTargets {}

impl RowTargets {
    /// Pointer to `output[token * rows + row]`.
    ///
    /// # Safety
    /// The caller must pass a `row` no other live worker is using, and the
    /// buffer must hold at least `(token + 1) * rows` elements.
    #[inline]
    unsafe fn at(&self, row: usize, token: usize) -> *mut f32 {
        unsafe { self.base.add(token * self.rows + row) }
    }
}

impl Q38TensorOps for TensorEntry<'_> {
    fn tensor_row_f32(&self, output: &mut [f32], row: u64) -> Q38CoreResult<()> {
        let width = self.shape[0] as usize;
        let rows = self.shape[1];
        if self.n_dims != 2 || row >= rows {
            return Err(Q38CoreError::RowOutOfBounds { row, rows });
        }
        let rbytes = row_bytes(self.ty, width)
            .ok_or(Q38CoreError::UnsupportedQuantType(self.ty as u32))?;
        if output.len() < width {
            return Err(Q38CoreError::ShapeMismatch {
                expected: width as u64,
                got: output.len() as u64,
            });
        }
        let data = &self.data[row as usize * rbytes..(row as usize + 1) * rbytes];
        match self.ty {
            GgmlType::F32 => {
                for (i, out) in output[..width].iter_mut().enumerate() {
                    *out = read_f32(&data[i * 4..]);
                }
            }
            GgmlType::F16 => {
                for (i, out) in output[..width].iter_mut().enumerate() {
                    *out = f16_to_f32(read_u16(&data[i * 2..]));
                }
            }
            GgmlType::Bf16 => {
                for (i, out) in output[..width].iter_mut().enumerate() {
                    *out = bf16_to_f32(&data[i * 2..]);
                }
            }
            ty if supported_f32_gemv(ty) => {
                dequantize_quant_row(data, &mut output[..width], self.ty);
            }
            _ => return Err(Q38CoreError::UnsupportedQuantType(self.ty as u32)),
        }
        Ok(())
    }

    fn tensor_dot_row_f32(&self, input: &[f32], row: u64) -> Q38CoreResult<f32> {
        let width = self.shape[0] as usize;
        let rows = self.shape[1];
        if self.n_dims != 2 || row >= rows {
            return Err(Q38CoreError::RowOutOfBounds { row, rows });
        }
        if !supported_f32_gemv(self.ty) {
            return Err(Q38CoreError::UnsupportedQuantType(self.ty as u32));
        }
        let rbytes = row_bytes(self.ty, width)
            .ok_or(Q38CoreError::UnsupportedQuantType(self.ty as u32))?;
        if input.len() < width {
            return Err(Q38CoreError::ShapeMismatch {
                expected: width as u64,
                got: input.len() as u64,
            });
        }
        let data = &self.data[row as usize * rbytes..(row as usize + 1) * rbytes];
        Ok(dot_row(data, &input[..width], width, self.ty))
    }

    fn tensor_gemv_f32(&self, output: &mut [f32], input: &[f32]) -> Q38CoreResult<()> {
        let width = self.shape[0] as usize;
        let rows = self.shape[1] as usize;
        if self.n_dims != 2 {
            return Err(Q38CoreError::Gguf("tensor is not 2-dimensional".into()));
        }
        if !supported_f32_gemv(self.ty) {
            return Err(Q38CoreError::UnsupportedQuantType(self.ty as u32));
        }
        let rbytes = row_bytes(self.ty, width)
            .ok_or(Q38CoreError::UnsupportedQuantType(self.ty as u32))?;
        if input.len() < width {
            return Err(Q38CoreError::ShapeMismatch {
                expected: width as u64,
                got: input.len() as u64,
            });
        }
        if output.len() < rows {
            return Err(Q38CoreError::ShapeMismatch {
                expected: rows as u64,
                got: output.len() as u64,
            });
        }
        let weights = self.data;
        let input = &input[..width];
        gemv_rows(output, rows, width, |row| {
            dot_row(
                &weights[row * rbytes..(row + 1) * rbytes],
                input,
                width,
                self.ty,
            )
        });
        Ok(())
    }

    fn tensor_gemv_q8_k(&self, output: &mut [f32], input: &[Q38Q8KBlock]) -> Q38CoreResult<()> {
        let width = self.shape[0] as usize;
        let rows = self.shape[1] as usize;
        if self.n_dims != 2 {
            return Err(Q38CoreError::Gguf("tensor is not 2-dimensional".into()));
        }
        if !supported_q8_k(self.ty) {
            return Err(Q38CoreError::UnsupportedQuantType(self.ty as u32));
        }
        let rbytes = row_bytes(self.ty, width)
            .ok_or(Q38CoreError::UnsupportedQuantType(self.ty as u32))?;
        let blocks = width / BLOCK;
        if input.len() < blocks {
            return Err(Q38CoreError::ShapeMismatch {
                expected: blocks as u64,
                got: input.len() as u64,
            });
        }
        if output.len() < rows {
            return Err(Q38CoreError::ShapeMismatch {
                expected: rows as u64,
                got: output.len() as u64,
            });
        }
        let weights = self.data;
        let activations = &input[..blocks];
        if let Some(repacked) = self.repacked_iq1_s(blocks) {
            let stride = blocks * iq::IQ1S_REPACK_BLOCK_BYTES;
            gemv_rows(output, rows, width, |row| {
                iq::dot_iq1_s_repacked_q8_k(&repacked[row * stride..], activations, blocks)
            });
            return Ok(());
        }
        gemv_rows(output, rows, width, |row| {
            dot_q8_k(
                &weights[row * rbytes..(row + 1) * rbytes],
                activations,
                blocks,
                self.ty,
            )
        });
        Ok(())
    }

    fn tensor_gemm_f32(
        &self,
        output: &mut [f32],
        input: &[f32],
        batch_size: u32,
    ) -> Q38CoreResult<()> {
        let width = self.shape[0] as usize;
        let rows = self.shape[1] as usize;
        let batch = batch_size as usize;
        if self.n_dims != 2 || batch == 0 {
            return Err(Q38CoreError::Gguf("tensor is not 2-dimensional".into()));
        }
        if !supported_f32_gemv(self.ty) {
            return Err(Q38CoreError::UnsupportedQuantType(self.ty as u32));
        }
        let rbytes = row_bytes(self.ty, width)
            .ok_or(Q38CoreError::UnsupportedQuantType(self.ty as u32))?;
        if input.len() < batch * width {
            return Err(Q38CoreError::ShapeMismatch {
                expected: (batch * width) as u64,
                got: input.len() as u64,
            });
        }
        if output.len() < batch * rows {
            return Err(Q38CoreError::ShapeMismatch {
                expected: (batch * rows) as u64,
                got: output.len() as u64,
            });
        }
        let weights = self.data;
        gemm_rows(output, rows, batch, width * batch, |row, token| {
            let start = token * width;
            dot_row(
                &weights[row * rbytes..(row + 1) * rbytes],
                &input[start..start + width],
                width,
                self.ty,
            )
        });
        Ok(())
    }

    fn tensor_gemm_q8_k(
        &self,
        output: &mut [f32],
        input: &[Q38Q8KBlock],
        batch_size: u32,
    ) -> Q38CoreResult<()> {
        let width = self.shape[0] as usize;
        let rows = self.shape[1] as usize;
        let batch = batch_size as usize;
        if self.n_dims != 2 || batch == 0 {
            return Err(Q38CoreError::Gguf("tensor is not 2-dimensional".into()));
        }
        if !supported_q8_k(self.ty) {
            return Err(Q38CoreError::UnsupportedQuantType(self.ty as u32));
        }
        let rbytes = row_bytes(self.ty, width)
            .ok_or(Q38CoreError::UnsupportedQuantType(self.ty as u32))?;
        let blocks = width / BLOCK;
        if input.len() < batch * blocks {
            return Err(Q38CoreError::ShapeMismatch {
                expected: (batch * blocks) as u64,
                got: input.len() as u64,
            });
        }
        if output.len() < batch * rows {
            return Err(Q38CoreError::ShapeMismatch {
                expected: (batch * rows) as u64,
                got: output.len() as u64,
            });
        }
        let weights = self.data;
        let activations = &input[..batch * blocks];
        if let Some(repacked) = self.repacked_iq1_s(blocks) {
            let stride = blocks * iq::IQ1S_REPACK_BLOCK_BYTES;
            gemm_rows(output, rows, batch, width * batch, |row, token| {
                let start = token * blocks;
                iq::dot_iq1_s_repacked_q8_k(
                    &repacked[row * stride..],
                    &activations[start..start + blocks],
                    blocks,
                )
            });
            return Ok(());
        }
        gemm_rows(output, rows, batch, width * batch, |row, token| {
            let start = token * blocks;
            dot_q8_k(
                &weights[row * rbytes..(row + 1) * rbytes],
                &activations[start..start + blocks],
                blocks,
                self.ty,
            )
        });
        Ok(())
    }
}
