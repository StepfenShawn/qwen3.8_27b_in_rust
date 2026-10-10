#![allow(unsafe_op_in_unsafe_fn)]

//! AVX2 IQ kernels.

use crate::kernel::iq::{
    f16_at, iq4_nl_group, repack_base_d, repack_grid_index, repack_signed_scale, u16_at, u32_at,
    IQ1S_REPACK_BLOCK_BYTES, IQ1_DELTA,
};
use crate::kernel::iq_tables::{IQ1S_GRID, IQ3S_GRID, KVALUES_IQ4NL};
use crate::kernel::Q38Q8KBlock;
use std::arch::x86_64::*;

#[inline]
fn avx2_fma() -> bool {
    std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
}

#[inline]
pub(crate) fn try_dot_iq1_s_repacked_q8_k(
    weights: &[u8],
    q: &[Q38Q8KBlock],
    blocks: usize,
) -> Option<f32> {
    if !avx2_fma() {
        return None;
    }
    Some(unsafe { dot_iq1_s_repacked_avx2(weights, q, blocks) })
}

#[inline]
pub(crate) fn try_dot_iq4_xs_q8_k(
    data: &[u8],
    q: &[Q38Q8KBlock],
    blocks: usize,
) -> Option<f32> {
    if !avx2_fma() {
        return None;
    }
    Some(unsafe { dot_iq4_xs_q8_k_avx2(data, q, blocks) })
}

#[inline]
pub(crate) fn try_dot_iq4_nl_q8_k(
    data: &[u8],
    q: &[Q38Q8KBlock],
    blocks: usize,
) -> Option<f32> {
    if !avx2_fma() {
        return None;
    }
    Some(unsafe { dot_iq4_nl_q8_k_avx2(data, q, blocks) })
}

#[inline]
pub(crate) fn try_dot_iq3_s_q8_k(
    data: &[u8],
    q: &[Q38Q8KBlock],
    blocks: usize,
) -> Option<f32> {
    if !avx2_fma() {
        return None;
    }
    Some(unsafe { dot_iq3_s_q8_k_avx2(data, q, blocks) })
}

// ---------------------------------------------------------------------------
// AVX2 integer kernels
// ---------------------------------------------------------------------------
//
// Mirrors the `#if defined(__AVX2__)` branches of `qwen38_quant.c`. The integer
// accumulators below are exact: every kernel is a sum of `i8 * i8` products
// into `i32`, so it agrees with the scalar path bit for bit. Only the final
// `f32` horizontal reduction reorders the additions, which is why the
// consistency test compares against the scalar kernel with a tolerance rather
// than `to_bits()`.

#[target_feature(enable = "avx2,fma")]
unsafe fn sum_f32x8(values: __m256) -> f32 {
    let mut sum = _mm_add_ps(_mm256_extractf128_ps(values, 1), _mm256_castps256_ps128(values));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    _mm_cvtss_f32(sum)
}

/// `sum(|left| * (sign(left) * right)) * scale` over 32 signed bytes.
#[target_feature(enable = "avx2")]
unsafe fn products_s8_s8_scaled_32(left: __m256i, right: __m256i, scale: i32) -> __m256i {
    let magnitudes = _mm256_abs_epi8(left);
    let signed_right = _mm256_sign_epi8(right, left);
    let pair16 = _mm256_maddubs_epi16(magnitudes, signed_right);
    _mm256_madd_epi16(pair16, _mm256_set1_epi16(scale as i16))
}

#[target_feature(enable = "avx2,fma")]
unsafe fn dot_iq1_s_repacked_avx2(weights: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut accumulated = _mm256_setzero_ps();
    let mut accumulated_correction = 0.0f32;
    for block in 0..blocks {
        let weight = &weights[block * IQ1S_REPACK_BLOCK_BYTES..][..IQ1S_REPACK_BLOCK_BYTES];
        let activation = &q[block];
        let mut weighted_lanes = _mm256_setzero_si256();
        let mut correction = 0i32;
        for group in (0..8).step_by(2) {
            let mut grid = [0i64; 8];
            for lane in 0..8 {
                let at = group * 4 + lane;
                grid[lane] = IQ1S_GRID[repack_grid_index(weight, at)] as i64;
            }
            let quant0 = _mm256_set_epi64x(grid[3], grid[2], grid[1], grid[0]);
            let quant1 = _mm256_set_epi64x(grid[7], grid[6], grid[5], grid[4]);
            let activation0 =
                _mm256_loadu_si256(activation.quants[group * 32..].as_ptr() as *const __m256i);
            let activation1 = _mm256_loadu_si256(
                activation.quants[(group + 1) * 32..].as_ptr() as *const __m256i,
            );
            let signed0 = repack_signed_scale(weight, group);
            let signed1 = repack_signed_scale(weight, group + 1);
            weighted_lanes = _mm256_add_epi32(
                weighted_lanes,
                products_s8_s8_scaled_32(quant0, activation0, signed0.abs()),
            );
            weighted_lanes = _mm256_add_epi32(
                weighted_lanes,
                products_s8_s8_scaled_32(quant1, activation1, signed1.abs()),
            );
            correction += signed0
                * (activation.sums[group * 2] as i32 + activation.sums[group * 2 + 1] as i32);
            correction += signed1
                * (activation.sums[(group + 1) * 2] as i32
                    + activation.sums[(group + 1) * 2 + 1] as i32);
        }
        let d = repack_base_d(weight) * activation.scale;
        accumulated = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes),
            accumulated,
        );
        accumulated_correction += d * correction as f32;
    }
    sum_f32x8(accumulated) + IQ1_DELTA * accumulated_correction
}

/// `sum(left * right)` over 32 signed bytes, widened to 8 × `i32`.
///
/// `vpmaddubsw` needs one unsigned operand, so the classic sign trick is used:
/// negate `right` where `left` is negative and feed `|left|` instead.
#[target_feature(enable = "avx2")]
unsafe fn products_s8_s8_32(left: __m256i, right: __m256i) -> __m256i {
    let magnitudes = _mm256_abs_epi8(left);
    let signed_right = _mm256_sign_epi8(right, left);
    let pair16 = _mm256_maddubs_epi16(magnitudes, signed_right);
    _mm256_madd_epi16(pair16, _mm256_set1_epi16(1))
}

/// `0xff` in every lane whose `signs` bit is set, `0x00` otherwise.
///
/// `signs` holds four bytes; byte `n` covers lanes `[8n, 8n+8)` and its bits
/// cover those lanes in order.
#[target_feature(enable = "avx2")]
unsafe fn raw_iq_sign_mask(signs: &[u8]) -> __m256i {
    const SIGN_SOURCE: [u8; 32] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        3, 3,
    ];
    const SIGN_BIT: [u8; 32] = [
        1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128, 1,
        2, 4, 8, 16, 32, 64, 128,
    ];
    let packed = _mm256_set1_epi32(u32_at(signs) as i32);
    let source = _mm256_loadu_si256(SIGN_SOURCE.as_ptr() as *const __m256i);
    let bits = _mm256_loadu_si256(SIGN_BIT.as_ptr() as *const __m256i);
    let selected = _mm256_and_si256(_mm256_shuffle_epi8(packed, source), bits);
    _mm256_cmpeq_epi8(selected, bits)
}

/// Negate `activation` in the lanes selected by `signs`.
#[target_feature(enable = "avx2")]
unsafe fn apply_raw_iq_signs(activation: __m256i, signs: &[u8]) -> __m256i {
    let negative = raw_iq_sign_mask(signs);
    _mm256_sub_epi8(_mm256_xor_si256(negative, activation), negative)
}

/// Expand two consecutive IQ3_S groups (16 codes) into their 3-bit grid vectors.
///
/// `codes` holds the low 8 bits of the 16 grid indices and `high` supplies bit
/// 8 for each of the two groups, matching the scalar IQ3_S index build.
#[target_feature(enable = "avx2")]
unsafe fn iq3_s_grid_vectors(codes: &[u8], high: &[u8], output: &mut [__m256i; 2]) {
    let packed = _mm_loadu_si128(codes.as_ptr() as *const __m128i);
    let shifts = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    let high_mask = _mm256_set1_epi32(0x100);
    let low0 = _mm256_cvtepu8_epi32(packed);
    let low1 = _mm256_cvtepu8_epi32(_mm_srli_si128::<8>(packed));
    let high0 = _mm256_and_si256(
        _mm256_sllv_epi32(_mm256_set1_epi32(high[0] as i32), shifts),
        high_mask,
    );
    let high1 = _mm256_and_si256(
        _mm256_sllv_epi32(_mm256_set1_epi32(high[1] as i32), shifts),
        high_mask,
    );
    let mut indices = [0u32; 16];
    _mm256_storeu_si256(
        indices.as_mut_ptr() as *mut __m256i,
        _mm256_or_si256(low0, high0),
    );
    _mm256_storeu_si256(
        indices.as_mut_ptr().add(8) as *mut __m256i,
        _mm256_or_si256(low1, high1),
    );
    let mut grids = [0i32; 16];
    for (slot, index) in grids.iter_mut().zip(indices.iter()) {
        *slot = IQ3S_GRID[*index as usize] as i32;
    }
    output[0] = _mm256_loadu_si256(grids.as_ptr() as *const __m256i);
    output[1] = _mm256_loadu_si256(grids.as_ptr().add(8) as *const __m256i);
}

#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_iq4_xs_q8_k_avx2(
    data: &[u8],
    q: &[Q38Q8KBlock],
    blocks: usize,
) -> f32 {
    let table = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let nibble_mask = _mm_set1_epi8(15);
    let mut accumulated = _mm256_setzero_ps();
    for block in 0..blocks {
        let weights = &data[block * 136..(block + 1) * 136];
        let activation = &q[block];
        let d = f16_at(weights) * activation.scale;
        let scale_high = u16_at(&weights[2..]);
        let scale_low = &weights[4..8];
        let quants = &weights[8..136];
        let mut weighted_lanes = _mm256_setzero_si256();
        for group in 0..8 {
            let scale = ((scale_low[group / 2] >> (4 * (group % 2))) & 15) as i32
                | (((scale_high >> (2 * group)) & 3) as i32) << 4;
            let packed = _mm_loadu_si128(quants[group * 16..].as_ptr() as *const __m128i);
            let low = _mm_shuffle_epi8(table, _mm_and_si128(packed, nibble_mask));
            let high = _mm_shuffle_epi8(
                table,
                _mm_and_si128(_mm_srli_epi16::<4>(packed), nibble_mask),
            );
            let quant = _mm256_set_m128i(high, low);
            let activation_group =
                _mm256_loadu_si256(activation.quants[group * 32..].as_ptr() as *const __m256i);
            let dot = products_s8_s8_32(quant, activation_group);
            weighted_lanes = _mm256_add_epi32(
                weighted_lanes,
                _mm256_mullo_epi32(dot, _mm256_set1_epi32(scale - 32)),
            );
        }
        accumulated = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes),
            accumulated,
        );
    }
    sum_f32x8(accumulated)
}

#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_iq4_nl_q8_k_avx2(
    data: &[u8],
    q: &[Q38Q8KBlock],
    blocks: usize,
) -> f32 {
    let table = _mm_loadu_si128(KVALUES_IQ4NL.as_ptr() as *const __m128i);
    let nibble_mask = _mm_set1_epi8(15);
    let mut accumulated = _mm256_setzero_ps();
    for block in 0..blocks {
        let weights = &data[block * 144..(block + 1) * 144];
        let activation = &q[block];
        // IQ4_NL carries a per-group scale, so each group folds into the
        // accumulator on its own; matches `q38_dot_iq4_nl_q8_k`.
        for group in 0..8 {
            let (base_d, quants) = iq4_nl_group(weights, group);
            let d = base_d * activation.scale;
            let packed = _mm_loadu_si128(quants.as_ptr() as *const __m128i);
            let low = _mm_shuffle_epi8(table, _mm_and_si128(packed, nibble_mask));
            let high = _mm_shuffle_epi8(
                table,
                _mm_and_si128(_mm_srli_epi16::<4>(packed), nibble_mask),
            );
            let quant = _mm256_set_m128i(high, low);
            let activation_group =
                _mm256_loadu_si256(activation.quants[group * 32..].as_ptr() as *const __m256i);
            let dot = products_s8_s8_32(quant, activation_group);
            accumulated = _mm256_fmadd_ps(
                _mm256_set1_ps(d),
                _mm256_cvtepi32_ps(dot),
                accumulated,
            );
        }
    }
    sum_f32x8(accumulated)
}

#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_iq3_s_q8_k_avx2(
    data: &[u8],
    q: &[Q38Q8KBlock],
    blocks: usize,
) -> f32 {
    let mut accumulated = _mm256_setzero_ps();
    for block in 0..blocks {
        let weights = &data[block * 110..(block + 1) * 110];
        let activation = &q[block];
        let d = f16_at(weights) * activation.scale;
        let indices = &weights[2..66];
        let high = &weights[66..74];
        let signs = &weights[74..106];
        let scales = &weights[106..110];
        let mut weighted_lanes = _mm256_setzero_si256();
        for group in (0..8).step_by(2) {
            let mut quant = [_mm256_setzero_si256(); 2];
            iq3_s_grid_vectors(
                &indices[group * 8..group * 8 + 16],
                &high[group..group + 2],
                &mut quant,
            );
            for pair in 0..2 {
                let current = group + pair;
                let activation_group = _mm256_loadu_si256(
                    activation.quants[current * 32..].as_ptr() as *const __m256i,
                );
                let signed = apply_raw_iq_signs(activation_group, &signs[current * 4..]);
                let dot = products_u8_s8_32(quant[pair], signed);
                let scale = 1 + 2 * ((scales[group / 2] >> (4 * pair)) & 15) as i32;
                weighted_lanes = _mm256_add_epi32(
                    weighted_lanes,
                    _mm256_mullo_epi32(dot, _mm256_set1_epi32(scale)),
                );
            }
        }
        accumulated = _mm256_fmadd_ps(
            _mm256_set1_ps(d),
            _mm256_cvtepi32_ps(weighted_lanes),
            accumulated,
        );
    }
    sum_f32x8(accumulated)
}

/// `sum(a * b)` over 32 unsigned bytes `a` and 32 signed bytes `b`.
#[target_feature(enable = "avx2")]
unsafe fn products_u8_s8_32(unsigned_values: __m256i, signed_values: __m256i) -> __m256i {
    let pair16 = _mm256_maddubs_epi16(unsigned_values, signed_values);
    _mm256_madd_epi16(pair16, _mm256_set1_epi16(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::iq::{dot_iq3_s_q8_k, dot_iq4_nl_q8_k, dot_iq4_xs_q8_k};
    use crate::kernel::quant::quantize_q8_k;
    use crate::kernel::Q38_Q8_K_BLOCK_SIZE;

    struct Rng(u32);

    impl Rng {
        fn next(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = x;
            x
        }

        fn byte(&mut self) -> u8 {
            (self.next() >> 13) as u8
        }
    }

    fn avx2_available() -> bool {
        avx2_fma()
    }

    fn synthetic_blocks(blocks: usize, block_bytes: usize, seed: u32) -> Vec<u8> {
        let mut rng = Rng(seed);
        let mut data = vec![0u8; blocks * block_bytes];
        for block in 0..blocks {
            let at = block * block_bytes;
            for i in 0..block_bytes {
                data[at + i] = rng.byte();
            }
            let scale = 0x3800u16 | (rng.next() as u16 & 0x03ff);
            data[at..at + 2].copy_from_slice(&scale.to_le_bytes());
        }
        data
    }

    fn synthetic_iq4_nl(blocks: usize, seed: u32) -> Vec<u8> {
        let mut rng = Rng(seed);
        let mut data = vec![0u8; blocks * 144];
        for block in 0..blocks {
            for group in 0..8 {
                let at = block * 144 + group * 18;
                for i in 0..18 {
                    data[at + i] = rng.byte();
                }
                let scale = 0x3800u16 | (rng.next() as u16 & 0x03ff);
                data[at..at + 2].copy_from_slice(&scale.to_le_bytes());
            }
        }
        data
    }

    fn activation_blocks(blocks: usize, seed: u32) -> Vec<Q38Q8KBlock> {
        let width = blocks * Q38_Q8_K_BLOCK_SIZE;
        let mut rng = Rng(seed);
        let values = (0..width)
            .map(|_| (rng.next() as f32 / u32::MAX as f32) * 4.0 - 2.0)
            .collect::<Vec<_>>();
        let mut activation = (0..blocks)
            .map(|_| Q38Q8KBlock::default())
            .collect::<Vec<_>>();
        quantize_q8_k(&mut activation, &values).unwrap();
        activation
    }

    #[test]
    fn avx2_matches_scalar_iq_kernels() {
        if !avx2_available() {
            println!("AVX2 + FMA unavailable; skipping");
            return;
        }
        let blocks = 16usize;
        for seed in [0x2545_f491u32, 0x9e37_79b9, 0x0123_4567] {
            let activation = activation_blocks(blocks, seed ^ 0xdead_beef);

            let xs = synthetic_blocks(blocks, 136, seed);
            let xs_scalar = dot_iq4_xs_q8_k(&xs, &activation, blocks);
            let xs_vector = unsafe { dot_iq4_xs_q8_k_avx2(&xs, &activation, blocks) };

            let nl = synthetic_iq4_nl(blocks, seed);
            let nl_scalar = dot_iq4_nl_q8_k(&nl, &activation, blocks);
            let nl_vector = unsafe { dot_iq4_nl_q8_k_avx2(&nl, &activation, blocks) };

            let iq3 = synthetic_blocks(blocks, 110, seed);
            let iq3_scalar = dot_iq3_s_q8_k(&iq3, &activation, blocks);
            let iq3_vector = unsafe { dot_iq3_s_q8_k_avx2(&iq3, &activation, blocks) };

            let rel = |a: f32, b: f32| (a - b).abs() / b.abs().max(1.0);
            println!(
                "seed={seed:#x}: IQ4_XS rel={:.3e} IQ4_NL rel={:.3e} IQ3_S rel={:.3e}",
                rel(xs_vector, xs_scalar),
                rel(nl_vector, nl_scalar),
                rel(iq3_vector, iq3_scalar)
            );
            assert!(
                rel(xs_vector, xs_scalar) < 1e-5,
                "IQ4_XS seed {seed:#x}: avx2 {xs_vector} vs scalar {xs_scalar}"
            );
            assert!(
                rel(nl_vector, nl_scalar) < 1e-5,
                "IQ4_NL seed {seed:#x}: avx2 {nl_vector} vs scalar {nl_scalar}"
            );
            assert!(
                rel(iq3_vector, iq3_scalar) < 1e-5,
                "IQ3_S seed {seed:#x}: avx2 {iq3_vector} vs scalar {iq3_scalar}"
            );
        }
    }
}
