use crate::kernel::{
    Q38CoreError, Q38CoreResult, Q38Q8KBlock, Q38QuantOps, Q38_Q8_K_BLOCK_SIZE,
    q38_parallel_over,
};
use rayon::prelude::*;

pub struct Q38Quant;

fn load_u16(data: &[u8]) -> u16 {
    data[0] as u16 | (data[1] as u16) << 8
}

fn load_u32(data: &[u8]) -> u32 {
    data[0] as u32 | (data[1] as u32) << 8 | (data[2] as u32) << 16 | (data[3] as u32) << 24
}

fn nearest_int(value: f32) -> i32 {
    let shifted = value + 12582912.0f32;
    let bits = shifted.to_bits();
    (bits & 0x007f_ffff) as i32 - 0x0040_0000
}

pub fn f16_to_f32(value: u16) -> f32 {
    let sign = ((value & 0x8000) as u32) << 16;
    let exponent = ((value >> 10) & 0x1f) as u32;
    let mantissa = (value & 0x03ff) as u32;

    let bits = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            let mut normalized = mantissa;
            let mut shift = 0u32;
            while normalized & 0x400 == 0 {
                normalized <<= 1;
                shift += 1;
            }
            normalized &= 0x3ff;
            sign | (127u32 - 14u32 - shift) << 23 | normalized << 13
        }
    } else if exponent == 31 {
        sign | 0x7f80_0000 | mantissa << 13
    } else {
        sign | (exponent + 112u32) << 23 | mantissa << 13
    };

    f32::from_bits(bits)
}

pub fn bf16_to_f32(data: &[u8]) -> f32 {
    f32::from_bits((load_u16(data) as u32) << 16)
}

/// Quantizes one 256-wide block. Blocks are independent, which is what lets
/// the block loop run across the pool.
fn quantize_block(quantized: &mut Q38Q8KBlock, values: &[f32]) {
    const BLOCK: usize = Q38_Q8_K_BLOCK_SIZE;
    const GROUP: usize = 16;

    let mut maximum = 0.0f32;
    let mut absolute_maximum = 0.0f32;
    for &value in values {
        let absolute = value.abs();
        if absolute > absolute_maximum {
            absolute_maximum = absolute;
            maximum = value;
        }
    }

    if absolute_maximum == 0.0f32 {
        quantized.scale = 0.0;
        quantized.quants.fill(0);
        quantized.sums.fill(0);
        return;
    }

    let inverse_scale = -127.0f32 / maximum;
    for (i, &value) in values.iter().enumerate() {
        let mut quant = nearest_int(inverse_scale * value);
        if quant > 127 {
            quant = 127;
        }
        quantized.quants[i] = quant as i8;
    }

    debug_assert_eq!(values.len(), BLOCK);
    for group in 0..BLOCK / GROUP {
        let mut sum = 0i32;
        for i in 0..GROUP {
            sum += quantized.quants[group * GROUP + i] as i32;
        }
        quantized.sums[group] = sum as i16;
    }

    quantized.scale = 1.0f32 / inverse_scale;
}

pub fn quantize_q8_k(output: &mut [Q38Q8KBlock], input: &[f32]) -> Q38CoreResult<()> {
    const BLOCK: usize = Q38_Q8_K_BLOCK_SIZE;

    if input.len() % BLOCK != 0 {
        return Err(Q38CoreError::LengthNotDivisibleByBlock {
            length: input.len() as u64,
        });
    }
    let blocks = input.len() / BLOCK;
    if output.len() < blocks {
        return Err(Q38CoreError::ShapeMismatch {
            expected: blocks as u64,
            got: output.len() as u64,
        });
    }

    let output = &mut output[..blocks];
    // Worth a handoff only for prompt-sized activations; one token is 20
    // blocks and stays on the calling thread.
    if q38_parallel_over(blocks, BLOCK) {
        output
            .par_iter_mut()
            .zip(input.par_chunks(BLOCK))
            .for_each(|(quantized, values)| quantize_block(quantized, values));
    } else {
        for (quantized, values) in output.iter_mut().zip(input.chunks(BLOCK)) {
            quantize_block(quantized, values);
        }
    }

    Ok(())
}

impl Q38QuantOps for Q38Quant {
    fn f16_to_f32(value: u16) -> f32 {
        f16_to_f32(value)
    }

    fn quantize_q8_k(output: &mut [Q38Q8KBlock], input: &[f32]) -> Q38CoreResult<()> {
        quantize_q8_k(output, input)
    }
}
