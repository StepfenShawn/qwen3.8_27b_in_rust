#![allow(unsafe_op_in_unsafe_fn)]

use crate::gguf::{GgmlType, TensorEntry};
use crate::kernel::iq;
use crate::kernel::quant::{bf16_to_f32, f16_to_f32};
use crate::kernel::{
    Q38CoreError, Q38CoreResult, Q38Q8KBlock, Q38TensorOps, Q38_Q8_K_BLOCK_SIZE,
    q38_parallel_over,
};
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

const BLOCK: usize = Q38_Q8_K_BLOCK_SIZE;

#[inline]
fn read_u16(data: &[u8]) -> u16 {
    u16::from_le_bytes([data[0], data[1]])
}

#[inline]
fn read_f32(data: &[u8]) -> f32 {
    f32::from_le_bytes([data[0], data[1], data[2], data[3]])
}

fn scale_min_k4(index: usize, packed: &[u8]) -> (u8, u8) {
    if index < 4 {
        (packed[index] & 63, packed[index + 4] & 63)
    } else {
        (
            (packed[index + 4] & 15) | ((packed[index - 4] >> 6) << 4),
            (packed[index + 4] >> 4) | ((packed[index] >> 6) << 4),
        )
    }
}

fn unpack_q3_scales(packed: &[u8], scales: &mut [i8; 16]) {
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

fn row_bytes(ty: GgmlType, width: usize) -> Option<usize> {
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

fn dot_q8_0_scalar(data: &[u8], input: &[f32], n: usize) -> f32 {
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

fn dot_q3_k_q8_k_scalar(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
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

fn dot_q5_k_q8_k_scalar(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
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

// ---- AVX2 hot paths ----

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sum_f32_8(values: __m256) -> f32 {
    let mut sum = _mm_add_ps(_mm256_extractf128_ps(values, 1), _mm256_castps256_ps128(values));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    _mm_cvtss_f32(sum)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn products_u8_s8_32(unsigned_values: __m256i, signed_values: __m256i) -> __m256i {
    let pair16 = _mm256_maddubs_epi16(unsigned_values, signed_values);
    _mm256_madd_epi16(pair16, _mm256_set1_epi16(1))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn load_quants(b: &Q38Q8KBlock, offset: usize) -> __m256i {
    _mm256_loadu_si256(b.quants.as_ptr().add(offset) as *const __m256i)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q6_scale_pair(scales: &[u8], index: usize) -> __m256i {
    _mm256_set_m128i(
        _mm_set1_epi16(scales[index + 1] as i8 as i16),
        _mm_set1_epi16(scales[index] as i8 as i16),
    )
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_avx2(data: &[u8], input: &[f32], n: usize) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= n {
        let w = _mm256_loadu_ps(data[i * 4..].as_ptr() as *const f32);
        let v = _mm256_loadu_ps(input[i..].as_ptr());
        acc = _mm256_fmadd_ps(w, v, acc);
        i += 8;
    }
    let mut total = sum_f32_8(acc);
    while i < n {
        total = read_f32(&data[i * 4..]).mul_add(input[i], total);
        i += 1;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_f16_avx2(data: &[u8], input: &[f32], n: usize) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut i = 0usize;
    while i + 8 <= n {
        let h = _mm_loadu_si128(data[i * 2..].as_ptr() as *const __m128i);
        let w = _mm256_cvtph_ps(h);
        let v = _mm256_loadu_ps(input[i..].as_ptr());
        acc = _mm256_fmadd_ps(w, v, acc);
        i += 8;
    }
    let mut total = sum_f32_8(acc);
    while i < n {
        total = f16_to_f32(read_u16(&data[i * 2..])).mul_add(input[i], total);
        i += 1;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q4_k_q8_k_avx2(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut accumulated = _mm256_setzero_ps();
    let mut accumulated_min = _mm_setzero_ps();
    for block in 0..blocks {
        let b = &q[block];
        let d = f16_to_f32(read_u16(&data[block * 144..])) * b.scale;
        let dmin = f16_to_f32(read_u16(&data[block * 144 + 2..])) * b.scale;
        let scales = &data[block * 144 + 4..block * 144 + 16];
        let quants = &data[block * 144 + 16..block * 144 + 144];
        let mut weighted_lanes = _mm256_setzero_si256();
        let mut minimum_scales = [0i16; 8];
        for chunk in 0..4 {
            let (scale0, min0) = scale_min_k4(chunk * 2, scales);
            let (scale1, min1) = scale_min_k4(chunk * 2 + 1, scales);
            let packed = _mm256_loadu_si256(quants[chunk * 32..].as_ptr() as *const __m256i);
            let mask = _mm256_set1_epi8(15);
            let low = _mm256_and_si256(packed, mask);
            let high = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask);
            let dot0 = products_u8_s8_32(low, load_quants(b, chunk * 64));
            let dot1 = products_u8_s8_32(high, load_quants(b, chunk * 64 + 32));
            weighted_lanes = _mm256_add_epi32(
                weighted_lanes,
                _mm256_mullo_epi32(dot0, _mm256_set1_epi32(scale0 as i32)),
            );
            weighted_lanes = _mm256_add_epi32(
                weighted_lanes,
                _mm256_mullo_epi32(dot1, _mm256_set1_epi32(scale1 as i32)),
            );
            minimum_scales[chunk * 2] = min0 as i16;
            minimum_scales[chunk * 2 + 1] = min1 as i16;
        }
        accumulated = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes),
            accumulated,
        );
        let sums0 = _mm_loadu_si128(b.sums.as_ptr() as *const __m128i);
        let sums1 = _mm_loadu_si128(b.sums[8..].as_ptr() as *const __m128i);
        let paired_sums = _mm_hadd_epi16(sums0, sums1);
        let mins = _mm_loadu_si128(minimum_scales.as_ptr() as *const __m128i);
        let minimum_products = _mm_madd_epi16(mins, paired_sums);
        accumulated_min = _mm_fmadd_ps(
            _mm_set1_ps(-dmin),
            _mm_cvtepi32_ps(minimum_products),
            accumulated_min,
        );
    }
    let mut minimum = _mm_add_ps(accumulated_min, _mm_movehl_ps(accumulated_min, accumulated_min));
    minimum = _mm_add_ss(minimum, _mm_movehdup_ps(minimum));
    sum_f32_8(accumulated) + _mm_cvtss_f32(minimum)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q6_k_q8_k_avx2(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut accumulated = _mm256_setzero_ps();
    for block in 0..blocks {
        let b = &q[block];
        let d = f16_to_f32(read_u16(&data[block * 210 + 208..])) * b.scale;
        let mask2 = _mm256_set1_epi8(3);
        let mask4 = _mm256_set1_epi8(15);
        let input_sums = _mm256_loadu_si256(b.sums.as_ptr() as *const __m256i);
        let packed_scales = _mm_loadu_si128(data[block * 210 + 192..].as_ptr() as *const __m128i);
        let wide_scales = _mm256_cvtepi8_epi16(packed_scales);
        let offset_products = _mm256_slli_epi32(_mm256_madd_epi16(input_sums, wide_scales), 5);
        let mut weighted_lanes = _mm256_setzero_si256();
        for half in 0..2 {
            let low0 = _mm256_loadu_si256(data[block * 210 + half * 64..].as_ptr() as *const __m256i);
            let low1 = _mm256_loadu_si256(
                data[block * 210 + half * 64 + 32..].as_ptr() as *const __m256i,
            );
            let upper = _mm256_loadu_si256(
                data[block * 210 + 128 + half * 32..].as_ptr() as *const __m256i,
            );
            let q0 = _mm256_or_si256(
                _mm256_and_si256(low0, mask4),
                _mm256_slli_epi16(_mm256_and_si256(upper, mask2), 4),
            );
            let q1 = _mm256_or_si256(
                _mm256_and_si256(low1, mask4),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(upper, 2), mask2), 4),
            );
            let q2 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(low0, 4), mask4),
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(upper, 4), mask2), 4),
            );
            let q3 = _mm256_or_si256(
                _mm256_and_si256(_mm256_srli_epi16(low1, 4), mask4),
                _mm256_srli_epi16(_mm256_and_si256(upper, _mm256_set1_epi8(0xc0u8 as i8)), 2),
            );
            let base = half * 128;
            let si = half * 8;
            let mut product0 = _mm256_maddubs_epi16(q0, load_quants(b, base));
            let mut product1 = _mm256_maddubs_epi16(q1, load_quants(b, base + 32));
            let mut product2 = _mm256_maddubs_epi16(q2, load_quants(b, base + 64));
            let mut product3 = _mm256_maddubs_epi16(q3, load_quants(b, base + 96));
            product0 = _mm256_madd_epi16(
                q6_scale_pair(&data[block * 210 + 192..], si),
                product0,
            );
            product1 = _mm256_madd_epi16(
                q6_scale_pair(&data[block * 210 + 192..], si + 2),
                product1,
            );
            product2 = _mm256_madd_epi16(
                q6_scale_pair(&data[block * 210 + 192..], si + 4),
                product2,
            );
            product3 = _mm256_madd_epi16(
                q6_scale_pair(&data[block * 210 + 192..], si + 6),
                product3,
            );
            weighted_lanes = _mm256_add_epi32(
                weighted_lanes,
                _mm256_add_epi32(product0, product1),
            );
            weighted_lanes = _mm256_add_epi32(
                weighted_lanes,
                _mm256_add_epi32(product2, product3),
            );
        }
        weighted_lanes = _mm256_sub_epi32(weighted_lanes, offset_products);
        accumulated = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes),
            accumulated,
        );
    }
    sum_f32_8(accumulated)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn products_s8_s8_32(left: __m256i, right: __m256i) -> __m256i {
    let magnitudes = _mm256_abs_epi8(left);
    let signed_right = _mm256_sign_epi8(right, left);
    products_u8_s8_32(magnitudes, signed_right)
}

/// `packed >> (2 * field)`, for the four 2-bit fields of a Q3_K low byte.
///
/// `_mm256_srli_epi16` takes its shift as a const generic, so the loop index
/// has to be turned back into a literal here.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn q3_field_shift(packed: __m256i, field: usize) -> __m256i {
    match field {
        0 => packed,
        1 => _mm256_srli_epi16::<2>(packed),
        2 => _mm256_srli_epi16::<4>(packed),
        _ => _mm256_srli_epi16::<6>(packed),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q3_k_q8_k_avx2(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut accumulated = _mm256_setzero_ps();
    for block in 0..blocks {
        let b = &q[block];
        let high = &data[block * 110..block * 110 + 32];
        let low = &data[block * 110 + 32..block * 110 + 96];
        let mut scales = [0i8; 16];
        unpack_q3_scales(&data[block * 110 + 96..block * 110 + 108], &mut scales);
        let d = f16_to_f32(read_u16(&data[block * 110 + 108..])) * b.scale;
        let hbits = _mm256_loadu_si256(high.as_ptr() as *const __m256i);
        let mask = _mm256_set1_epi8(3);
        let four = _mm256_set1_epi8(4);
        let zero = _mm256_setzero_si256();
        let mut weighted_lanes = _mm256_setzero_si256();
        let mut group = 0usize;
        for half in 0..2 {
            let packed = _mm256_loadu_si256(low[half * 32..].as_ptr() as *const __m256i);
            for field in 0..4 {
                // A clear `high` bit means the 2-bit field is storing
                // `value - 4` rather than `value`.
                let low_values = _mm256_and_si256(q3_field_shift(packed, field), mask);
                let absent = _mm256_cmpeq_epi8(
                    _mm256_and_si256(hbits, _mm256_set1_epi8((1u8 << group) as i8)),
                    zero,
                );
                let quant = _mm256_sub_epi8(low_values, _mm256_and_si256(absent, four));
                let activation = _mm256_loadu_si256(
                    b.quants[half * 128 + field * 32..].as_ptr() as *const __m256i,
                );
                let dot = products_s8_s8_32(quant, activation);
                // Lanes 0..15 use the first scale, 16..31 the second.
                let scale = _mm256_set_m128i(
                    _mm_set1_epi32(scales[group * 2 + 1] as i32),
                    _mm_set1_epi32(scales[group * 2] as i32),
                );
                weighted_lanes =
                    _mm256_add_epi32(weighted_lanes, _mm256_mullo_epi32(dot, scale));
                group += 1;
            }
        }
        accumulated = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes),
            accumulated,
        );
    }
    sum_f32_8(accumulated)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q5_k_q8_k_avx2(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut accumulated = _mm256_setzero_ps();
    let mut accumulated_min = _mm_setzero_ps();
    for block in 0..blocks {
        let b = &q[block];
        let d = f16_to_f32(read_u16(&data[block * 176..])) * b.scale;
        let dmin = f16_to_f32(read_u16(&data[block * 176 + 2..])) * b.scale;
        let scales = &data[block * 176 + 4..block * 176 + 16];
        let high = &data[block * 176 + 16..block * 176 + 48];
        let low = &data[block * 176 + 48..block * 176 + 176];
        let mut minimum = 0i32;
        let mut high0: u8 = 1;
        let mut high1: u8 = 2;
        let mut weighted_lanes = _mm256_setzero_si256();
        for chunk in 0..4 {
            let (scale0, min0) = scale_min_k4(chunk * 2, scales);
            let (scale1, min1) = scale_min_k4(chunk * 2 + 1, scales);
            let packed = _mm256_loadu_si256(low[chunk * 32..].as_ptr() as *const __m256i);
            let high_bits = _mm256_loadu_si256(high.as_ptr() as *const __m256i);
            let nibble_mask = _mm256_set1_epi8(15);
            let zero = _mm256_setzero_si256();
            let sixteen = _mm256_set1_epi8(16);
            let mut q0 = _mm256_and_si256(packed, nibble_mask);
            let mut q1 = _mm256_and_si256(_mm256_srli_epi16::<4>(packed), nibble_mask);
            // Bit 4 of each quant lives in `high`; where it is clear the value
            // stays below 16, where it is set the vector adds the missing 16.
            let absent0 = _mm256_cmpeq_epi8(
                _mm256_and_si256(high_bits, _mm256_set1_epi8(high0 as i8)),
                zero,
            );
            let absent1 = _mm256_cmpeq_epi8(
                _mm256_and_si256(high_bits, _mm256_set1_epi8(high1 as i8)),
                zero,
            );
            q0 = _mm256_add_epi8(q0, _mm256_andnot_si256(absent0, sixteen));
            q1 = _mm256_add_epi8(q1, _mm256_andnot_si256(absent1, sixteen));
            let dot0 = products_u8_s8_32(q0, load_quants(b, chunk * 64));
            let dot1 = products_u8_s8_32(q1, load_quants(b, chunk * 64 + 32));
            weighted_lanes = _mm256_add_epi32(
                weighted_lanes,
                _mm256_mullo_epi32(dot0, _mm256_set1_epi32(scale0 as i32)),
            );
            weighted_lanes = _mm256_add_epi32(
                weighted_lanes,
                _mm256_mullo_epi32(dot1, _mm256_set1_epi32(scale1 as i32)),
            );
            minimum += min0 as i32 * (b.sums[chunk * 4] as i32 + b.sums[chunk * 4 + 1] as i32);
            minimum += min1 as i32 * (b.sums[chunk * 4 + 2] as i32 + b.sums[chunk * 4 + 3] as i32);
            high0 <<= 2;
            high1 <<= 2;
        }
        accumulated = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes),
            accumulated,
        );
        accumulated_min =
            _mm_fmadd_ss(_mm_set_ss(-dmin), _mm_set_ss(minimum as f32), accumulated_min);
    }
    sum_f32_8(accumulated) + _mm_cvtss_f32(accumulated_min)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_q8_0_avx2(data: &[u8], input: &[f32], n: usize) -> f32 {
    // Q8_0 blocks are 32 values wide, so anything else (never produced for
    // this type) goes back to the scalar loop.
    if n % 32 != 0 {
        return dot_q8_0_scalar(data, input, n);
    }
    let mut accumulated = _mm256_setzero_ps();
    let mut base = 0usize;
    let mut block = 0usize;
    while base < n {
        let d = _mm256_set1_ps(f16_to_f32(read_u16(&data[block * 34..])));
        let first = _mm_loadu_si128(data[block * 34 + 2..].as_ptr() as *const __m128i);
        let second = _mm_loadu_si128(data[block * 34 + 18..].as_ptr() as *const __m128i);
        let words = [
            _mm256_cvtepi8_epi32(first),
            _mm256_cvtepi8_epi32(_mm_srli_si128::<8>(first)),
            _mm256_cvtepi8_epi32(second),
            _mm256_cvtepi8_epi32(_mm_srli_si128::<8>(second)),
        ];
        for (quarter, word) in words.iter().enumerate() {
            let weights = _mm256_mul_ps(d, _mm256_cvtepi32_ps(*word));
            let activations = _mm256_loadu_ps(input[base + quarter * 8..].as_ptr());
            accumulated = _mm256_fmadd_ps(weights, activations, accumulated);
        }
        base += 32;
        block += 1;
    }
    sum_f32_8(accumulated)
}

// ---- runtime dispatch ----

#[inline]
fn dot_f32(data: &[u8], input: &[f32], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { dot_f32_avx2(data, input, n) };
        }
    }
    dot_f32_scalar(data, input, n)
}

#[inline]
fn dot_f16(data: &[u8], input: &[f32], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c")
        {
            return unsafe { dot_f16_avx2(data, input, n) };
        }
    }
    dot_f16_scalar(data, input, n)
}

#[inline]
fn dot_q4_k_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { dot_q4_k_q8_k_avx2(data, q, blocks) };
        }
    }
    dot_q4_k_q8_k_scalar(data, q, blocks)
}

#[inline]
fn dot_q6_k_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { dot_q6_k_q8_k_avx2(data, q, blocks) };
        }
    }
    dot_q6_k_q8_k_scalar(data, q, blocks)
}

#[inline]
fn dot_q3_k_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { dot_q3_k_q8_k_avx2(data, q, blocks) };
        }
    }
    dot_q3_k_q8_k_scalar(data, q, blocks)
}

#[inline]
fn dot_q5_k_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { dot_q5_k_q8_k_avx2(data, q, blocks) };
        }
    }
    dot_q5_k_q8_k_scalar(data, q, blocks)
}

#[inline]
fn dot_q8_0(data: &[u8], input: &[f32], n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c")
        {
            return unsafe { dot_q8_0_avx2(data, input, n) };
        }
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

    /// True when the AVX2 kernels can actually run on this machine.
    fn avx2_available() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            return std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma");
        }
        #[cfg(not(target_arch = "x86_64"))]
        false
    }

    /// The AVX2 integer kernels must agree with their scalar twins.
    ///
    /// Both accumulate exactly the same `i8 * i8` products into `i32`, so the
    /// only legitimate difference is the order of the final `f32` reduction.
    /// A wrong nibble, a misplaced scale or an off-by-one shift would instead
    /// show up as a large gap. Row 0 and the last row are both checked because
    /// an over-read only faults at the end of a tensor.
    #[test]
    fn avx2_matches_scalar_q8_k() {
        if !avx2_available() {
            println!("AVX2 + FMA unavailable; skipping");
            return;
        }
        let gguf = Gguf::open("Qwen3.8-27B-UD-Q4_K_M.gguf").unwrap();
        let mut seen: Vec<String> = Vec::new();
        let mut checked = 0usize;

        for tensor in gguf.tensors() {
            if tensor.n_dims != 2 || !matches!(tensor.ty, GgmlType::Q3K | GgmlType::Q5K) {
                continue;
            }
            let key = format!("{:?}", tensor.ty);
            if seen.contains(&key) {
                continue;
            }
            let width = tensor.shape[0] as usize;
            if width == 0 || width % BLOCK != 0 {
                continue;
            }
            seen.push(key.clone());

            let rows = tensor.shape[1];
            let rbytes = row_bytes(tensor.ty, width).unwrap();
            let x = activation(width);
            let block_count = width / BLOCK;
            let mut blocks = (0..block_count)
                .map(|_| Q38Q8KBlock::default())
                .collect::<Vec<_>>();
            quantize_q8_k(&mut blocks, &x).unwrap();

            for row in [0u64, rows - 1] {
                let at = row as usize * rbytes;
                let data = &tensor.data[at..at + rbytes];
                let (scalar, vector) = unsafe {
                    match tensor.ty {
                        GgmlType::Q3K => (
                            dot_q3_k_q8_k_scalar(data, &blocks, block_count),
                            dot_q3_k_q8_k_avx2(data, &blocks, block_count),
                        ),
                        _ => (
                            dot_q5_k_q8_k_scalar(data, &blocks, block_count),
                            dot_q5_k_q8_k_avx2(data, &blocks, block_count),
                        ),
                    }
                };
                let rel = relative(vector, scalar);
                println!("{key} row={row}: scalar={scalar} avx2={vector} rel={rel:.3e}");
                assert!(rel < 1e-5, "{key} row {row}: avx2 {vector} vs scalar {scalar}");
                checked += 1;
            }
        }
        assert!(checked >= 4, "only {checked} rows exercised");
    }

    /// Same check for the one f32-input kernel that gained an AVX2 path.
    ///
    /// This one genuinely reorders `f32` additions, so the bar is a relative
    /// tolerance rather than exactness.
    #[test]
    fn avx2_matches_scalar_f32() {
        if !avx2_available() {
            println!("AVX2 + FMA unavailable; skipping");
            return;
        }
        let gguf = Gguf::open("Qwen3.8-27B-UD-Q4_K_M.gguf").unwrap();
        let tensor = gguf
            .find_tensor("blk.0.ssm_alpha.weight")
            .expect("Q8_0 representative");
        assert_eq!(tensor.ty, GgmlType::Q8_0);

        let width = tensor.shape[0] as usize;
        let rbytes = row_bytes(tensor.ty, width).unwrap();
        let x = activation(width);
        let rows = tensor.shape[1];
        let mut checked = 0usize;
        for row in [0u64, rows - 1] {
            let at = row as usize * rbytes;
            let data = &tensor.data[at..at + rbytes];
            let scalar = dot_q8_0_scalar(data, &x, width);
            let vector = unsafe { dot_q8_0_avx2(data, &x, width) };
            let rel = relative(vector, scalar);
            println!("Q8_0 row={row}: scalar={scalar} avx2={vector} rel={rel:.3e}");
            assert!(rel < 1e-5, "Q8_0 row {row}: avx2 {vector} vs scalar {scalar}");
            checked += 1;
        }
        assert_eq!(checked, 2);
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
