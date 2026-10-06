#![allow(unsafe_op_in_unsafe_fn)]

//! Importance-quantization (IQ) kernels.

use crate::gguf::GgmlType;
use crate::kernel::iq_tables::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS,
    KSIGNS_IQ2XS, KVALUES_IQ4NL,
};
use crate::kernel::quant::f16_to_f32;
use crate::kernel::{Q38Q8KBlock, Q38_Q8_K_BLOCK_SIZE, q38_parallel_over};
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Widest kernel iteration unit in elements. `IQ4_NL` super-blocks are the
/// only ones that reach this; every other IQ type is narrower.
const MAX_BLOCK: usize = Q38_Q8_K_BLOCK_SIZE;

/// `IQ1S_DELTA` / `IQ1M_DELTA` from ggml-common.h.
const IQ1_DELTA: f32 = 0.125;

/// True when [`dot_row_f32`] handles `ty`.
pub fn supports_f32(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::Iq1S
            | GgmlType::Iq1M
            | GgmlType::Iq2Xxs
            | GgmlType::Iq2Xs
            | GgmlType::Iq2S
            | GgmlType::Iq3Xxs
            | GgmlType::Iq3S
            | GgmlType::Iq4Xs
            | GgmlType::Iq4Nl
    )
}

/// True when [`dot_q8_k`] handles `ty`.
pub fn supports_q8_k(ty: GgmlType) -> bool {
    supports_f32(ty)
}

/// Elements consumed by one kernel iteration.
///
/// GGUF stores `IQ4_NL` as 32 element / 18 byte blocks, but every kernel in
/// this module consumes eight consecutive blocks (144 bytes) at once so each
/// block's `d` can be hoisted out of the nibble loops. For every other IQ type
/// the GGUF block already is the kernel's iteration unit, so these two differ
/// only for `IQ4_NL`.
fn super_block_elements(ty: GgmlType) -> usize {
    match ty {
        GgmlType::Iq4Nl => 8 * ty.block_elements() as usize,
        _ => ty.block_elements() as usize,
    }
}

/// Bytes that [`super_block_elements`] worth of weights occupy.
fn super_block_bytes(ty: GgmlType) -> usize {
    let per_block_bytes = ty.block_bytes() as usize;
    super_block_elements(ty) / ty.block_elements() as usize * per_block_bytes
}

#[inline]
fn u16_at(data: &[u8]) -> u16 {
    u16::from_le_bytes([data[0], data[1]])
}

#[inline]
fn u32_at(data: &[u8]) -> u32 {
    u32::from_le_bytes([data[0], data[1], data[2], data[3]])
}

#[inline]
fn f16_at(data: &[u8]) -> f32 {
    f16_to_f32(u16_at(data))
}

/// Little-endian bytes of a grid entry, interpreted by the caller as signed
/// or unsigned to match the C reference.
#[inline]
fn grid64(entry: u64) -> [u8; 8] {
    entry.to_le_bytes()
}

#[inline]
fn grid32(entry: u32) -> [u8; 4] {
    entry.to_le_bytes()
}

#[inline]
fn sign_of(mask: u8, bit: u8) -> i32 {
    if mask & bit != 0 { -1 } else { 1 }
}

/// IQ4_NL scale: low and high nibble share one f16 block scale.
#[inline]
fn iq4_nl_group(block: &[u8], group: usize) -> (f32, &[u8]) {
    let packed = &block[group * 18..group * 18 + 18];
    (f16_at(packed), &packed[2..18])
}

// ---------------------------------------------------------------------------
// f32 decode: one block -> elements
// ---------------------------------------------------------------------------

/// Decode one kernel iteration unit into `out[..super_block_elements(ty)]`.
///
/// Used by the f32 dot product; the loop structure matches the q8_k kernels
/// below so both paths agree on the decoded values.
fn dequant_block(block: &[u8], out: &mut [f32; MAX_BLOCK], ty: GgmlType) {
    match ty {
        GgmlType::Iq4Nl => {
            for group in 0..8 {
                let (d, quants) = iq4_nl_group(block, group);
                for lane in 0..16 {
                    let packed = quants[lane];
                    out[group * 32 + lane] = d * KVALUES_IQ4NL[(packed & 15) as usize] as f32;
                    out[group * 32 + lane + 16] = d * KVALUES_IQ4NL[(packed >> 4) as usize] as f32;
                }
            }
        }
        GgmlType::Iq4Xs => {
            let d = f16_at(block);
            let scale_high = u16_at(&block[2..]);
            let scale_low = &block[4..8];
            let quants = &block[8..136];
            for group in 0..8 {
                let scale = ((scale_low[group / 2] >> (4 * (group % 2))) & 15) as i32
                    | (((scale_high >> (2 * group)) & 3) as i32) << 4;
                let ds = d * (scale - 32) as f32;
                for lane in 0..16 {
                    let packed = quants[group * 16 + lane];
                    out[group * 32 + lane] = ds * KVALUES_IQ4NL[(packed & 15) as usize] as f32;
                    out[group * 32 + lane + 16] =
                        ds * KVALUES_IQ4NL[(packed >> 4) as usize] as f32;
                }
            }
        }
        GgmlType::Iq3S => {
            let d = f16_at(block);
            let indices = &block[2..66];
            let high = &block[66..74];
            let signs = &block[74..106];
            let scales = &block[106..110];
            for group in 0..8 {
                let code = &indices[group * 8..group * 8 + 8];
                let high_bits = high[group] as u32;
                let group_signs = &signs[group * 4..group * 4 + 4];
                let scale =
                    1 + 2 * ((scales[group / 2] >> (4 * (group % 2))) & 15) as i32;
                let ds = d * scale as f32;
                for section in 0..4 {
                    let index0 =
                        code[section * 2] as u32 | ((high_bits << (8 - 2 * section)) & 0x100);
                    let index1 =
                        code[section * 2 + 1] as u32 | ((high_bits << (7 - 2 * section)) & 0x100);
                    let g0 = grid32(IQ3S_GRID[index0 as usize]);
                    let g1 = grid32(IQ3S_GRID[index1 as usize]);
                    for lane in 0..4 {
                        let sign0 = sign_of(group_signs[section], 1 << lane) as f32;
                        let sign1 = sign_of(group_signs[section], 1 << (lane + 4)) as f32;
                        out[group * 32 + section * 8 + lane] = ds * g0[lane] as f32 * sign0;
                        out[group * 32 + section * 8 + lane + 4] = ds * g1[lane] as f32 * sign1;
                    }
                }
            }
        }
        GgmlType::Iq3Xxs => {
            let d = 0.25 * f16_at(block);
            let indices = &block[2..66];
            let metadata = &block[66..98];
            for group in 0..8 {
                let packed = u32_at(&metadata[group * 4..]);
                let scale = 1 + 2 * (packed >> 28) as i32;
                let ds = d * scale as f32;
                let code = &indices[group * 8..group * 8 + 8];
                for section in 0..4 {
                    let g0 = grid32(IQ3XXS_GRID[code[section * 2] as usize]);
                    let g1 = grid32(IQ3XXS_GRID[code[section * 2 + 1] as usize]);
                    let signs = KSIGNS_IQ2XS[((packed >> (7 * section)) & 127) as usize];
                    for lane in 0..4 {
                        let sign0 = sign_of(signs, KMASK_IQ2XS[lane]) as f32;
                        let sign1 = sign_of(signs, KMASK_IQ2XS[lane + 4]) as f32;
                        out[group * 32 + section * 8 + lane] = ds * g0[lane] as f32 * sign0;
                        out[group * 32 + section * 8 + lane + 4] = ds * g1[lane] as f32 * sign1;
                    }
                }
            }
        }
        GgmlType::Iq2Xxs => {
            let d = IQ1_DELTA * f16_at(block);
            let codes = &block[2..66];
            for group in 0..8 {
                let code = &codes[group * 8..group * 8 + 8];
                let metadata = u32_at(&code[4..]);
                let scale = 1 + 2 * (metadata >> 28) as i32;
                let ds = d * scale as f32;
                for section in 0..4 {
                    let grid = grid64(IQ2XXS_GRID[code[section] as usize]);
                    let signs = KSIGNS_IQ2XS[((metadata >> (7 * section)) & 127) as usize];
                    for lane in 0..8 {
                        let sign = sign_of(signs, KMASK_IQ2XS[lane]) as f32;
                        out[group * 32 + section * 8 + lane] = ds * grid[lane] as f32 * sign;
                    }
                }
            }
        }
        GgmlType::Iq2Xs => {
            let d = IQ1_DELTA * f16_at(block);
            let codes = &block[2..66];
            let scales = &block[66..74];
            for group in 0..8 {
                let mut code = [0u16; 4];
                for section in 0..4 {
                    code[section] = u16_at(&codes[group * 8 + section * 2..]);
                }
                let scale0 = 1 + 2 * (scales[group] & 15) as i32;
                let scale1 = 1 + 2 * (scales[group] >> 4) as i32;
                for section in 0..4 {
                    let grid = grid64(IQ2XS_GRID[(code[section] & 511) as usize]);
                    let signs = KSIGNS_IQ2XS[(code[section] >> 9) as usize];
                    let ds = d * (if section < 2 { scale0 } else { scale1 }) as f32;
                    for lane in 0..8 {
                        let sign = sign_of(signs, KMASK_IQ2XS[lane]) as f32;
                        out[group * 32 + section * 8 + lane] = ds * grid[lane] as f32 * sign;
                    }
                }
            }
        }
        GgmlType::Iq2S => {
            let d = IQ1_DELTA * f16_at(block);
            let indices = &block[2..34];
            let signs = &block[34..66];
            let high = &block[66..74];
            let scales = &block[74..82];
            for group in 0..8 {
                let scale0 = 1 + 2 * (scales[group] & 15) as i32;
                let scale1 = 1 + 2 * (scales[group] >> 4) as i32;
                for section in 0..4 {
                    let index = indices[group * 4 + section] as u32
                        | ((high[group] as u32) << (8 - 2 * section)) & 0x300;
                    let grid = grid64(IQ2S_GRID[index as usize]);
                    let ds = d * (if section < 2 { scale0 } else { scale1 }) as f32;
                    let sign_byte = signs[group * 4 + section];
                    for lane in 0..8 {
                        let sign = sign_of(sign_byte, KMASK_IQ2XS[lane]) as f32;
                        out[group * 32 + section * 8 + lane] = ds * grid[lane] as f32 * sign;
                    }
                }
            }
        }
        GgmlType::Iq1S => {
            let d = f16_at(block);
            let indices = &block[2..34];
            let high = &block[34..50];
            for group in 0..8 {
                let packed = u16_at(&high[group * 2..]);
                let scale = 1 + 2 * ((packed >> 12) & 7) as i32;
                let delta = if packed & 0x8000 != 0 { -1.0 } else { 1.0 };
                let ds = d * scale as f32;
                for section in 0..4 {
                    let index = indices[group * 4 + section] as u32
                        | (((packed >> (3 * section)) & 7) as u32) << 8;
                    let grid = grid64(IQ1S_GRID[index as usize]);
                    for lane in 0..8 {
                        out[group * 32 + section * 8 + lane] =
                            ds * (grid[lane] as i8 as f32 + IQ1_DELTA * delta);
                    }
                }
            }
        }
        GgmlType::Iq1M => {
            let indices = &block[0..32];
            let high = &block[32..48];
            let packed_scales = &block[48..56];
            let scale_bits = (u16_at(packed_scales) >> 12)
                | ((u16_at(&packed_scales[2..]) >> 8) & 0x00f0)
                | ((u16_at(&packed_scales[4..]) >> 4) & 0x0f00)
                | (u16_at(&packed_scales[6..]) & 0xf000);
            let d = f16_to_f32(scale_bits);
            for group in 0..8 {
                let scales = u16_at(&packed_scales[(group / 2) * 2..]);
                let shift = 6 * (group % 2);
                let scale0 = 1 + 2 * ((scales >> shift) & 7) as i32;
                let scale1 = 1 + 2 * ((scales >> (shift + 3)) & 7) as i32;
                for section in 0..4 {
                    let group_high = high[group * 2 + section / 2];
                    let index = indices[group * 4 + section] as u32
                        | ((group_high as u32) << (8 - 4 * (section % 2))) & 0x700;
                    let grid = grid64(IQ1S_GRID[index as usize]);
                    let delta_bit = if section % 2 == 1 { 0x80 } else { 0x08 };
                    let delta = if group_high & delta_bit != 0 { -1.0 } else { 1.0 };
                    let ds = d * (if section < 2 { scale0 } else { scale1 }) as f32;
                    for lane in 0..8 {
                        out[group * 32 + section * 8 + lane] =
                            ds * (grid[lane] as i8 as f32 + IQ1_DELTA * delta);
                    }
                }
            }
        }
        _ => unreachable!("dequant_block called for non-IQ type"),
    }
}

// ---------------------------------------------------------------------------
// f32 dot products
// ---------------------------------------------------------------------------

/// Dot product between an f32 activation row and one IQ-quantized row.
pub fn dot_row_f32(data: &[u8], input: &[f32], n: usize, ty: GgmlType) -> f32 {
    let elements = super_block_elements(ty);
    let block_bytes = super_block_bytes(ty);
    let mut decoded = [0.0f32; MAX_BLOCK];
    let mut total = 0.0f32;
    let mut base = 0usize;
    let mut block = 0usize;
    while base < n {
        dequant_block(
            &data[block * block_bytes..(block + 1) * block_bytes],
            &mut decoded,
            ty,
        );
        for lane in 0..elements {
            total = decoded[lane].mul_add(input[base + lane], total);
        }
        base += elements;
        block += 1;
    }
    total
}

/// Decode a whole IQ row into `output` (one f32 per weight).
///
/// Cheaper than the C reference's basis-vector trick (`q38_tensor_row_f32`)
/// while producing identical values.
pub fn dequantize_row(data: &[u8], output: &mut [f32], ty: GgmlType) {
    let elements = super_block_elements(ty);
    let block_bytes = super_block_bytes(ty);
    let mut decoded = [0.0f32; MAX_BLOCK];
    for (block, chunk) in output.chunks_mut(elements).enumerate() {
        dequant_block(
            &data[block * block_bytes..(block + 1) * block_bytes],
            &mut decoded,
            ty,
        );
        chunk.copy_from_slice(&decoded[..elements]);
    }
}

// ---------------------------------------------------------------------------
// Q8_K integer dot products (exact port of qwen38_quant.c)
// ---------------------------------------------------------------------------

fn dot_iq4_nl_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut total = 0.0f32;
    for block in 0..blocks {
        let weights = &data[block * 144..(block + 1) * 144];
        let activation = &q[block];
        // IQ4_NL carries a per-group scale, so `d` and the accumulator are
        // both local to the group (see q38_dot_iq4_nl_q8_k).
        for group in 0..8 {
            let (base_d, quants) = iq4_nl_group(weights, group);
            let d = base_d * activation.scale;
            let mut weighted = 0i32;
            for lane in 0..16 {
                let packed = quants[lane];
                weighted += activation.quants[group * 32 + lane] as i32
                    * KVALUES_IQ4NL[(packed & 15) as usize] as i32;
                weighted += activation.quants[group * 32 + lane + 16] as i32
                    * KVALUES_IQ4NL[(packed >> 4) as usize] as i32;
            }
            total += d * weighted as f32;
        }
    }
    total
}

fn dot_iq4_xs_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut total = 0.0f32;
    for block in 0..blocks {
        let weights = &data[block * 136..(block + 1) * 136];
        let activation = &q[block];
        let d = f16_at(weights) * activation.scale;
        let scale_high = u16_at(&weights[2..]);
        let scale_low = &weights[4..8];
        let quants = &weights[8..136];
        let mut weighted = 0i32;
        for group in 0..8 {
            let scale = ((scale_low[group / 2] >> (4 * (group % 2))) & 15) as i32
                | (((scale_high >> (2 * group)) & 3) as i32) << 4;
            let mut subtotal = 0i32;
            for lane in 0..16 {
                let packed = quants[group * 16 + lane];
                subtotal += activation.quants[group * 32 + lane] as i32
                    * KVALUES_IQ4NL[(packed & 15) as usize] as i32;
                subtotal += activation.quants[group * 32 + lane + 16] as i32
                    * KVALUES_IQ4NL[(packed >> 4) as usize] as i32;
            }
            weighted += (scale - 32) * subtotal;
        }
        total += d * weighted as f32;
    }
    total
}

fn dot_iq3_s_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut total = 0.0f32;
    for block in 0..blocks {
        let weights = &data[block * 110..(block + 1) * 110];
        let activation = &q[block];
        let d = f16_at(weights) * activation.scale;
        let indices = &weights[2..66];
        let high = &weights[66..74];
        let signs = &weights[74..106];
        let scales = &weights[106..110];
        let mut weighted = 0i32;
        for group in 0..8 {
            let code = &indices[group * 8..group * 8 + 8];
            let high_bits = high[group] as u32;
            let group_signs = &signs[group * 4..group * 4 + 4];
            let scale = 1 + 2 * ((scales[group / 2] >> (4 * (group % 2))) & 15) as i32;
            let mut subtotal = 0i32;
            for section in 0..4 {
                let index0 =
                    code[section * 2] as u32 | ((high_bits << (8 - 2 * section)) & 0x100);
                let index1 =
                    code[section * 2 + 1] as u32 | ((high_bits << (7 - 2 * section)) & 0x100);
                let g0 = grid32(IQ3S_GRID[index0 as usize]);
                let g1 = grid32(IQ3S_GRID[index1 as usize]);
                for lane in 0..4 {
                    let sign0 = sign_of(group_signs[section], 1 << lane);
                    let sign1 = sign_of(group_signs[section], 1 << (lane + 4));
                    subtotal += activation.quants[group * 32 + section * 8 + lane] as i32
                        * g0[lane] as i32
                        * sign0;
                    subtotal += activation.quants[group * 32 + section * 8 + lane + 4] as i32
                        * g1[lane] as i32
                        * sign1;
                }
            }
            weighted += scale * subtotal;
        }
        total += d * weighted as f32;
    }
    total
}

fn dot_iq3_xxs_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut total = 0.0f32;
    for block in 0..blocks {
        let weights = &data[block * 98..(block + 1) * 98];
        let activation = &q[block];
        let d = 0.25 * f16_at(weights) * activation.scale;
        let indices = &weights[2..66];
        let metadata = &weights[66..98];
        let mut weighted = 0i32;
        for group in 0..8 {
            let packed = u32_at(&metadata[group * 4..]);
            let scale = 1 + 2 * (packed >> 28) as i32;
            let code = &indices[group * 8..group * 8 + 8];
            let mut subtotal = 0i32;
            for section in 0..4 {
                let g0 = grid32(IQ3XXS_GRID[code[section * 2] as usize]);
                let g1 = grid32(IQ3XXS_GRID[code[section * 2 + 1] as usize]);
                let signs = KSIGNS_IQ2XS[((packed >> (7 * section)) & 127) as usize];
                for lane in 0..4 {
                    let sign0 = sign_of(signs, KMASK_IQ2XS[lane]);
                    let sign1 = sign_of(signs, KMASK_IQ2XS[lane + 4]);
                    subtotal += activation.quants[group * 32 + section * 8 + lane] as i32
                        * g0[lane] as i32
                        * sign0;
                    subtotal += activation.quants[group * 32 + section * 8 + lane + 4] as i32
                        * g1[lane] as i32
                        * sign1;
                }
            }
            weighted += scale * subtotal;
        }
        total += d * weighted as f32;
    }
    total
}

fn dot_iq2_xxs_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut total = 0.0f32;
    for block in 0..blocks {
        let weights = &data[block * 66..(block + 1) * 66];
        let activation = &q[block];
        let d = IQ1_DELTA * f16_at(weights) * activation.scale;
        let codes = &weights[2..66];
        let mut weighted = 0i32;
        for group in 0..8 {
            let code = &codes[group * 8..group * 8 + 8];
            let metadata = u32_at(&code[4..]);
            let scale = 1 + 2 * (metadata >> 28) as i32;
            let mut subtotal = 0i32;
            for section in 0..4 {
                let grid = grid64(IQ2XXS_GRID[code[section] as usize]);
                let signs = KSIGNS_IQ2XS[((metadata >> (7 * section)) & 127) as usize];
                for lane in 0..8 {
                    let sign = sign_of(signs, KMASK_IQ2XS[lane]);
                    subtotal += activation.quants[group * 32 + section * 8 + lane] as i32
                        * grid[lane] as i32
                        * sign;
                }
            }
            weighted += scale * subtotal;
        }
        total += d * weighted as f32;
    }
    total
}

fn dot_iq2_xs_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut total = 0.0f32;
    for block in 0..blocks {
        let weights = &data[block * 74..(block + 1) * 74];
        let activation = &q[block];
        let d = IQ1_DELTA * f16_at(weights) * activation.scale;
        let codes = &weights[2..66];
        let scales = &weights[66..74];
        let mut weighted = 0i32;
        for group in 0..8 {
            let mut code = [0u16; 4];
            for section in 0..4 {
                code[section] = u16_at(&codes[group * 8 + section * 2..]);
            }
            let scale0 = 1 + 2 * (scales[group] & 15) as i32;
            let scale1 = 1 + 2 * (scales[group] >> 4) as i32;
            let mut subtotal = [0i32; 2];
            for section in 0..4 {
                let grid = grid64(IQ2XS_GRID[(code[section] & 511) as usize]);
                let signs = KSIGNS_IQ2XS[(code[section] >> 9) as usize];
                for lane in 0..8 {
                    let sign = sign_of(signs, KMASK_IQ2XS[lane]);
                    subtotal[section / 2] += activation.quants[group * 32 + section * 8 + lane]
                        as i32
                        * grid[lane] as i32
                        * sign;
                }
            }
            weighted += scale0 * subtotal[0] + scale1 * subtotal[1];
        }
        total += d * weighted as f32;
    }
    total
}

fn dot_iq2_s_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut total = 0.0f32;
    for block in 0..blocks {
        let weights = &data[block * 82..(block + 1) * 82];
        let activation = &q[block];
        let d = IQ1_DELTA * f16_at(weights) * activation.scale;
        let indices = &weights[2..34];
        let signs = &weights[34..66];
        let high = &weights[66..74];
        let scales = &weights[74..82];
        let mut weighted = 0i32;
        for group in 0..8 {
            let scale0 = 1 + 2 * (scales[group] & 15) as i32;
            let scale1 = 1 + 2 * (scales[group] >> 4) as i32;
            let mut subtotal = [0i32; 2];
            for lane in 0..32 {
                let section = lane / 8;
                let index = indices[group * 4 + section] as u32
                    | ((high[group] as u32) << (8 - 2 * section)) & 0x300;
                let grid = grid64(IQ2S_GRID[index as usize]);
                let sign = sign_of(signs[group * 4 + section], KMASK_IQ2XS[lane % 8]);
                subtotal[section / 2] +=
                    activation.quants[group * 32 + lane] as i32 * grid[lane % 8] as i32 * sign;
            }
            weighted += scale0 * subtotal[0] + scale1 * subtotal[1];
        }
        total += d * weighted as f32;
    }
    total
}

fn dot_iq1_s_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut total = 0.0f32;
    for block in 0..blocks {
        let weights = &data[block * 50..(block + 1) * 50];
        let activation = &q[block];
        let d = f16_at(weights) * activation.scale;
        let indices = &weights[2..34];
        let high = &weights[34..50];
        let mut weighted = 0i32;
        let mut correction = 0i32;
        for group in 0..8 {
            let packed = u16_at(&high[group * 2..]);
            let scale = 1 + 2 * ((packed >> 12) & 7) as i32;
            let delta = if packed & 0x8000 != 0 { -1 } else { 1 };
            let mut subtotal = 0i32;
            for section in 0..4 {
                let index = indices[group * 4 + section] as u32
                    | (((packed >> (3 * section)) & 7) as u32) << 8;
                let grid = grid64(IQ1S_GRID[index as usize]);
                for lane in 0..8 {
                    subtotal += activation.quants[group * 32 + section * 8 + lane] as i32
                        * (grid[lane] as i8) as i32;
                }
            }
            weighted += scale * subtotal;
            correction += scale
                * delta
                * (activation.sums[group * 2] as i32 + activation.sums[group * 2 + 1] as i32);
        }
        total += d * (weighted as f32 + IQ1_DELTA * correction as f32);
    }
    total
}

fn dot_iq1_m_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut total = 0.0f32;
    for block in 0..blocks {
        let weights = &data[block * 56..(block + 1) * 56];
        let activation = &q[block];
        let indices = &weights[0..32];
        let high = &weights[32..48];
        let packed_scales = &weights[48..56];
        let scale_bits = (u16_at(packed_scales) >> 12)
            | ((u16_at(&packed_scales[2..]) >> 8) & 0x00f0)
            | ((u16_at(&packed_scales[4..]) >> 4) & 0x0f00)
            | (u16_at(&packed_scales[6..]) & 0xf000);
        let d = f16_to_f32(scale_bits) * activation.scale;
        let mut weighted = 0i32;
        let mut correction = 0i32;
        for group in 0..8 {
            let mut dot = [0i32; 2];
            let mut correction_dot = [0i32; 2];
            for section in 0..4 {
                let group_high = high[group * 2 + section / 2];
                let index = indices[group * 4 + section] as u32
                    | ((group_high as u32) << (8 - 4 * (section % 2))) & 0x700;
                let grid = grid64(IQ1S_GRID[index as usize]);
                let delta_bit = if section % 2 == 1 { 0x80 } else { 0x08 };
                let delta = if group_high & delta_bit != 0 { -1 } else { 1 };
                for lane in 0..8 {
                    let activation_value =
                        activation.quants[group * 32 + section * 8 + lane] as i32;
                    dot[section / 2] += activation_value * (grid[lane] as i8) as i32;
                    correction_dot[section / 2] += activation_value * delta;
                }
            }
            let scales = u16_at(&packed_scales[(group / 2) * 2..]);
            let shift = 6 * (group % 2);
            let scale0 = 1 + 2 * ((scales >> shift) & 7) as i32;
            let scale1 = 1 + 2 * ((scales >> (shift + 3)) & 7) as i32;
            weighted += scale0 * dot[0] + scale1 * dot[1];
            correction += scale0 * correction_dot[0] + scale1 * correction_dot[1];
        }
        total += d * (weighted as f32 + IQ1_DELTA * correction as f32);
    }
    total
}

// ---------------------------------------------------------------------------
// IQ1_S repack (SIMD-friendly view of the GGUF payload)
// ---------------------------------------------------------------------------

/// Raw size of one IQ1_S block in the GGUF mapping.
pub const IQ1S_BLOCK_BYTES: usize = 50;

/// Size of one repacked block; mirrors `Q38IQ1SRepackedBlock`.
///
/// ```text
/// +0   float    base_d
/// +4   uint16   grid_index[32]
/// +68  int8     signed_scale[8]
/// ```
pub const IQ1S_REPACK_BLOCK_BYTES: usize = 76;

#[inline]
fn repack_base_d(block: &[u8]) -> f32 {
    f32::from_le_bytes([block[0], block[1], block[2], block[3]])
}

#[inline]
fn repack_grid_index(block: &[u8], index: usize) -> usize {
    u16::from_le_bytes([block[4 + index * 2], block[5 + index * 2]]) as usize
}

#[inline]
fn repack_signed_scale(block: &[u8], group: usize) -> i32 {
    block[IQ1S_REPACK_BLOCK_BYTES - 8 + group] as i8 as i32
}

/// Repacks one IQ1_S block; the mapping between the two layouts is documented
/// on [`IQ1S_REPACK_BLOCK_BYTES`].
fn repack_iq1_s_block(target: &mut [u8], source: &[u8]) {
    target[..4].copy_from_slice(&f16_at(source).to_le_bytes());
    let low = &source[2..34];
    let high = &source[34..50];
    for group in 0..8 {
        let packed = u16_at(&high[group * 2..]);
        let scale = 1 + 2 * ((packed >> 12) & 7) as i32;
        let signed_scale = if packed & 0x8000 != 0 { -scale } else { scale };
        target[IQ1S_REPACK_BLOCK_BYTES - 8 + group] = signed_scale as i8 as u8;
        for section in 0..4 {
            let index = low[group * 4 + section] as u16 | (((packed >> (3 * section)) & 7) << 8);
            let at = 4 + (group * 4 + section) * 2;
            target[at..at + 2].copy_from_slice(&index.to_le_bytes());
        }
    }
}

/// Hoist the per-block dequantization work out of the inner loop: the grid
/// indices and the sign-carrying scales depend only on the weight block, so
/// pre-computing them lets the dot product stream `iq1s_grid` directly into
/// vector registers.
pub fn repack_iq1_s(data: &[u8], block_count: usize) -> Vec<u8> {
    let total = block_count * IQ1S_REPACK_BLOCK_BYTES;
    let mut output = vec![0u8; total];
    // Every block writes its own 76-byte slot and reads its own 50-byte slot,
    // so the layout is addressable independently of the iteration order.
    if q38_parallel_over(block_count, IQ1S_BLOCK_BYTES) {
        output[..total]
            .par_chunks_mut(IQ1S_REPACK_BLOCK_BYTES)
            .zip(data.par_chunks(IQ1S_BLOCK_BYTES))
            .for_each(|(target, source)| repack_iq1_s_block(target, source));
    } else {
        for (target, source) in output[..total]
            .chunks_mut(IQ1S_REPACK_BLOCK_BYTES)
            .zip(data.chunks(IQ1S_BLOCK_BYTES))
        {
            repack_iq1_s_block(target, source);
        }
    }
    output
}

fn dot_iq1_s_repacked_scalar(weights: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    let mut total = 0.0f32;
    for block in 0..blocks {
        let weight = &weights[block * IQ1S_REPACK_BLOCK_BYTES..][..IQ1S_REPACK_BLOCK_BYTES];
        let activation = &q[block];
        let d = repack_base_d(weight) * activation.scale;
        let mut weighted = 0i32;
        let mut correction = 0i32;
        for group in 0..8 {
            let signed_scale = repack_signed_scale(weight, group);
            let mut subtotal = 0i32;
            for section in 0..4 {
                let grid = grid64(IQ1S_GRID[repack_grid_index(weight, group * 4 + section)]);
                for lane in 0..8 {
                    subtotal += activation.quants[group * 32 + section * 8 + lane] as i32
                        * grid[lane] as i8 as i32;
                }
            }
            weighted += signed_scale.abs() * subtotal;
            correction += signed_scale
                * (activation.sums[group * 2] as i32 + activation.sums[group * 2 + 1] as i32);
        }
        total += d * (weighted as f32 + IQ1_DELTA * correction as f32);
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn sum_f32x8(values: __m256) -> f32 {
    let mut sum = _mm_add_ps(_mm256_extractf128_ps(values, 1), _mm256_castps256_ps128(values));
    sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    _mm_cvtss_f32(sum)
}

/// `sum(|left| * (sign(left) * right)) * scale` over 32 signed bytes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn products_s8_s8_scaled_32(left: __m256i, right: __m256i, scale: i32) -> __m256i {
    let magnitudes = _mm256_abs_epi8(left);
    let signed_right = _mm256_sign_epi8(right, left);
    let pair16 = _mm256_maddubs_epi16(magnitudes, signed_right);
    _mm256_madd_epi16(pair16, _mm256_set1_epi16(scale as i16))
}

#[cfg(target_arch = "x86_64")]
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
            let activation0 = _mm256_loadu_si256(
                activation.quants[group * 32..].as_ptr() as *const __m256i,
            );
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

/// Integer dot product against a repacked IQ1_S matrix.
///
/// `weights` points at the first repacked block of the row; the result is
/// numerically identical to [`dot_q8_k`] for `GgmlType::Iq1S`.
pub fn dot_iq1_s_repacked_q8_k(
    weights: &[u8],
    q: &[Q38Q8KBlock],
    blocks: usize,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { dot_iq1_s_repacked_avx2(weights, q, blocks) };
        }
    }
    dot_iq1_s_repacked_scalar(weights, q, blocks)
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

/// `sum(left * right)` over 32 signed bytes, widened to 8 × `i32`.
///
/// `vpmaddubsw` needs one unsigned operand, so the classic sign trick is used:
/// negate `right` where `left` is negative and feed `|left|` instead.
#[cfg(target_arch = "x86_64")]
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
#[cfg(target_arch = "x86_64")]
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
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_raw_iq_signs(activation: __m256i, signs: &[u8]) -> __m256i {
    let negative = raw_iq_sign_mask(signs);
    _mm256_sub_epi8(_mm256_xor_si256(negative, activation), negative)
}

/// Expand two consecutive IQ3_S groups (16 codes) into their 3-bit grid vectors.
///
/// `codes` holds the low 8 bits of the 16 grid indices and `high` supplies bit
/// 8 for each of the two groups, matching [`dot_iq3_s_q8_k`]'s index build.
#[cfg(target_arch = "x86_64")]
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_iq4_xs_q8_k_avx2(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_iq4_nl_q8_k_avx2(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_iq3_s_q8_k_avx2(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
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
///
/// The AVX2 counterpart of `tensor::products_u8_s8_32`; kept local because the
/// two modules do not share private helpers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn products_u8_s8_32(unsigned_values: __m256i, signed_values: __m256i) -> __m256i {
    let pair16 = _mm256_maddubs_epi16(unsigned_values, signed_values);
    _mm256_madd_epi16(pair16, _mm256_set1_epi16(1))
}

// ---- runtime dispatch ----

fn dot_iq4_xs(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { dot_iq4_xs_q8_k_avx2(data, q, blocks) };
        }
    }
    dot_iq4_xs_q8_k(data, q, blocks)
}

fn dot_iq4_nl(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { dot_iq4_nl_q8_k_avx2(data, q, blocks) };
        }
    }
    dot_iq4_nl_q8_k(data, q, blocks)
}

fn dot_iq3_s(data: &[u8], q: &[Q38Q8KBlock], blocks: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return unsafe { dot_iq3_s_q8_k_avx2(data, q, blocks) };
        }
    }
    dot_iq3_s_q8_k(data, q, blocks)
}

/// Integer dot product between a `Q8_K` activation row and one IQ row.
pub fn dot_q8_k(data: &[u8], q: &[Q38Q8KBlock], blocks: usize, ty: GgmlType) -> f32 {
    match ty {
        GgmlType::Iq1S => dot_iq1_s_q8_k(data, q, blocks),
        GgmlType::Iq1M => dot_iq1_m_q8_k(data, q, blocks),
        GgmlType::Iq2Xxs => dot_iq2_xxs_q8_k(data, q, blocks),
        GgmlType::Iq2Xs => dot_iq2_xs_q8_k(data, q, blocks),
        GgmlType::Iq2S => dot_iq2_s_q8_k(data, q, blocks),
        GgmlType::Iq3Xxs => dot_iq3_xxs_q8_k(data, q, blocks),
        GgmlType::Iq3S => dot_iq3_s(data, q, blocks),
        GgmlType::Iq4Xs => dot_iq4_xs(data, q, blocks),
        GgmlType::Iq4Nl => dot_iq4_nl(data, q, blocks),
        _ => f32::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::quant::quantize_q8_k;

    /// Deterministic xorshift so the test does not depend on a random source.
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

    /// Any byte pattern is a legal IQ1_S block: the grid indices are masked
    /// into `[0, 2048)`, so only `d` has to stay a finite f16.
    fn synthetic_iq1_s(blocks: usize, seed: u32) -> Vec<u8> {
        let mut rng = Rng(seed);
        let mut data = vec![0u8; blocks * IQ1S_BLOCK_BYTES];
        for block in 0..blocks {
            let at = block * IQ1S_BLOCK_BYTES;
            for i in 2..IQ1S_BLOCK_BYTES {
                data[at + i] = rng.byte();
            }
            // Exponent 14 keeps `d` finite and in `[0.5, 1.0)`.
            let scale = 0x3800u16 | (rng.next() as u16 & 0x03ff);
            data[at..at + 2].copy_from_slice(&scale.to_le_bytes());
        }
        data
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

    /// Random bytes with a finite leading f16 scale.
    ///
    /// Every remaining byte is a legal code for the types under test, so the
    /// PRNG reaches nibble and sign combinations a real checkpoint rarely
    /// emits — a stronger check than sampling rows out of the model file.
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

    /// `IQ4_NL` keeps its scale per 18-byte sub-block, so each one needs the
    /// finite-scale fix-up rather than just the first.
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

    /// The AVX2 IQ kernels must reproduce the scalar kernels.
    ///
    /// Both paths build the same `i32` accumulator out of the same grid
    /// lookups, so agreement here means the shuffle-based decoding
    /// (`vpshufb` on `kvalues_iq4nl`, the sign mask, the IQ3_S grid build) is
    /// translating the codes the same way. Only the trailing `f32` reduction
    /// differs in order, hence a tight relative tolerance rather than bit
    /// equality.
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

    #[test]
    fn iq1_s_repack_matches_packed_dot() {
        let blocks = 8usize;
        let width = blocks * Q38_Q8_K_BLOCK_SIZE;
        let weights = synthetic_iq1_s(blocks, 0x2545_f491);

        let mut rng = Rng(0x9e37_79b9);
        let mut activation_values = vec![0.0f32; width];
        for value in activation_values.iter_mut() {
            *value = (rng.next() as f32 / u32::MAX as f32) * 4.0 - 2.0;
        }
        let mut activation = (0..blocks)
            .map(|_| Q38Q8KBlock::default())
            .collect::<Vec<_>>();
        quantize_q8_k(&mut activation, &activation_values).unwrap();

        let packed = dot_iq1_s_q8_k(&weights, &activation, blocks);
        let repacked = repack_iq1_s(&weights, blocks);
        assert_eq!(repacked.len(), blocks * IQ1S_REPACK_BLOCK_BYTES);
        let baseline = dot_iq1_s_repacked_scalar(&repacked, &activation, blocks);
        let dispatched = dot_iq1_s_repacked_q8_k(&repacked, &activation, blocks);

        assert!(
            (packed - baseline).abs() <= 1e-3 * packed.abs().max(1.0),
            "repacked scalar {baseline} vs packed {packed}"
        );
        assert!(
            (packed - dispatched).abs() <= 1e-3 * packed.abs().max(1.0),
            "repacked dispatch {dispatched} vs packed {packed}"
        );
    }
}
