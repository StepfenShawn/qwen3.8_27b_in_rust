#![allow(unsafe_op_in_unsafe_fn)]

//! AVX2 tensor kernels.

use crate::kernel::Q38Q8KBlock;
use crate::kernel::quant::f16_to_f32;
use crate::kernel::tensor::{read_f32, read_u16, scale_min_k4, unpack_q3_scales};
use std::arch::x86_64::*;

#[inline]
pub(crate) fn dot_f32(data: &[u8], input: &[f32], n: usize) -> f32 {
    unsafe { dot_f32_avx2(data, input, n) }
}

#[inline]
pub(crate) fn dot_f16(data: &[u8], input: &[f32], n: usize) -> f32 {
    unsafe { dot_f16_avx2(data, input, n) }
}

#[inline]
pub(crate) fn dot_q8_0(data: &[u8], input: &[f32], n: usize) -> f32 {
    unsafe { dot_q8_0_avx2(data, input, n) }
}

#[inline]
pub(crate) fn dot_q3_k_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    unsafe { dot_q3_k_q8_k_avx2(data, q, blocks) }
}

#[inline]
pub(crate) fn dot_q4_k_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    unsafe { dot_q4_k_q8_k_avx2(data, q, blocks) }
}

#[inline]
pub(crate) fn dot_q5_k_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    unsafe { dot_q5_k_q8_k_avx2(data, q, blocks) }
}

#[inline]
pub(crate) fn dot_q6_k_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    unsafe { dot_q6_k_q8_k_avx2(data, q, blocks) }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn sum_f32_8(values: __m256) -> f32 {
    let mut sum = _mm_add_ps(
        _mm256_extractf128_ps(values, 1),
        _mm256_castps256_ps128(values),
    );
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    _mm_cvtss_f32(sum)
}

#[target_feature(enable = "avx2")]
unsafe fn products_u8_s8_32(unsigned_values: __m256i, signed_values: __m256i) -> __m256i {
    let pair16 = _mm256_maddubs_epi16(unsigned_values, signed_values);
    _mm256_madd_epi16(pair16, _mm256_set1_epi16(1))
}

#[target_feature(enable = "avx2")]
unsafe fn load_quants(b: &Q38Q8KBlock, offset: usize) -> __m256i {
    _mm256_loadu_si256(b.quants.as_ptr().add(offset) as *const __m256i)
}

#[target_feature(enable = "avx2")]
unsafe fn q6_scale_pair(scales: &[u8], index: usize) -> __m256i {
    _mm256_set_m128i(
        _mm_set1_epi16(scales[index + 1] as i8 as i16),
        _mm_set1_epi16(scales[index] as i8 as i16),
    )
}

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
    let mut minimum = _mm_add_ps(
        accumulated_min,
        _mm_movehl_ps(accumulated_min, accumulated_min),
    );
    minimum = _mm_add_ss(minimum, _mm_movehdup_ps(minimum));
    sum_f32_8(accumulated) + _mm_cvtss_f32(minimum)
}

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
            let low0 =
                _mm256_loadu_si256(data[block * 210 + half * 64..].as_ptr() as *const __m256i);
            let low1 =
                _mm256_loadu_si256(data[block * 210 + half * 64 + 32..].as_ptr() as *const __m256i);
            let upper = _mm256_loadu_si256(
                data[block * 210 + 128 + half * 32..].as_ptr() as *const __m256i
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
            product0 = _mm256_madd_epi16(q6_scale_pair(&data[block * 210 + 192..], si), product0);
            product1 =
                _mm256_madd_epi16(q6_scale_pair(&data[block * 210 + 192..], si + 2), product1);
            product2 =
                _mm256_madd_epi16(q6_scale_pair(&data[block * 210 + 192..], si + 4), product2);
            product3 =
                _mm256_madd_epi16(q6_scale_pair(&data[block * 210 + 192..], si + 6), product3);
            weighted_lanes = _mm256_add_epi32(weighted_lanes, _mm256_add_epi32(product0, product1));
            weighted_lanes = _mm256_add_epi32(weighted_lanes, _mm256_add_epi32(product2, product3));
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
#[target_feature(enable = "avx2")]
unsafe fn q3_field_shift(packed: __m256i, field: usize) -> __m256i {
    match field {
        0 => packed,
        1 => _mm256_srli_epi16::<2>(packed),
        2 => _mm256_srli_epi16::<4>(packed),
        _ => _mm256_srli_epi16::<6>(packed),
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_q3_k_q8_k_avx2(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
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
                    b.quants[half * 128 + field * 32..].as_ptr() as *const __m256i
                );
                let dot = products_s8_s8_32(quant, activation);
                // Lanes 0..15 use the first scale, 16..31 the second.
                let scale = _mm256_set_m128i(
                    _mm_set1_epi32(scales[group * 2 + 1] as i32),
                    _mm_set1_epi32(scales[group * 2] as i32),
                );
                weighted_lanes = _mm256_add_epi32(weighted_lanes, _mm256_mullo_epi32(dot, scale));
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

#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_q5_k_q8_k_avx2(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
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
        accumulated_min = _mm_fmadd_ss(
            _mm_set_ss(-dmin),
            _mm_set_ss(minimum as f32),
            accumulated_min,
        );
    }
    sum_f32_8(accumulated) + _mm_cvtss_f32(accumulated_min)
}

#[target_feature(enable = "avx2,fma,f16c")]
pub(crate) unsafe fn dot_q8_0_avx2(data: &[u8], input: &[f32], n: usize) -> f32 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgmlType, Gguf};
    use crate::kernel::Q38_Q8_K_BLOCK_SIZE;
    use crate::kernel::quant::quantize_q8_k;
    use crate::kernel::tensor::{
        dot_q3_k_q8_k_scalar, dot_q5_k_q8_k_scalar, dot_q8_0_scalar, row_bytes,
    };

    const BLOCK: usize = Q38_Q8_K_BLOCK_SIZE;

    fn activation(width: usize) -> Vec<f32> {
        (0..width)
            .map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0)
            .collect()
    }

    fn relative(a: f32, b: f32) -> f32 {
        (a - b).abs() / b.abs().max(1.0)
    }

    fn avx2_available() -> bool {
        crate::kernel::avx2_fma()
    }

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
                assert!(
                    rel < 1e-5,
                    "{key} row {row}: avx2 {vector} vs scalar {scalar}"
                );
                checked += 1;
            }
        }
        assert!(checked >= 4, "only {checked} rows exercised");
    }

    #[test]
    fn avx2_matches_scalar_f32() {
        if !avx2_available() || !std::arch::is_x86_feature_detected!("f16c") {
            println!("AVX2 + FMA + F16C unavailable; skipping");
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
            assert!(
                rel < 1e-5,
                "Q8_0 row {row}: avx2 {vector} vs scalar {scalar}"
            );
            checked += 1;
        }
        assert_eq!(checked, 2);
    }
}
