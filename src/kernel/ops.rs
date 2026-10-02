use crate::gguf::{Gguf, TensorEntry};
use crate::kernel::{Q38Iq1sRepack, Q38QuantOps, Q38TensorOps};

pub struct Q38Quant;

impl Q38QuantOps for Q38Quant {
    fn f16_to_f32(value: u16) -> f32 {
        todo!()
    }

    fn quantize_q8_k(output: &mut [super::Q38Q8KBlock], input: &[f32]) -> super::Q38CoreResult<()> {
        todo!()
    }
}

impl Q38TensorOps for TensorEntry<'_> {
    fn tensor_row_f32(&self, output: &mut [f32], row: u64) -> super::Q38CoreResult<()> {
        todo!()
    }

    fn tensor_dot_row_f32(&self, input: &[f32], row: u64) -> super::Q38CoreResult<f32> {
        todo!()
    }

    fn tensor_gemv_f32(&self, output: &mut [f32], input: &[f32]) -> super::Q38CoreResult<()> {
        todo!()
    }

    fn tensor_gemv_q8_k(
        &self,
        output: &mut [f32],
        input: &[super::Q38Q8KBlock],
    ) -> super::Q38CoreResult<()> {
        todo!()
    }

    fn tensor_gemm_f32(
        &self,
        output: &mut [f32],
        input: &[f32],
        batch_size: u32,
    ) -> super::Q38CoreResult<()> {
        todo!()
    }

    fn tensor_gemm_q8_k(
        &self,
        output: &mut [f32],
        input: &[super::Q38Q8KBlock],
        batch_size: u32,
    ) -> super::Q38CoreResult<()> {
        todo!()
    }
}

impl Q38Iq1sRepack for Gguf {
    fn prepare_iq1_s_repacks(&mut self) -> super::Q38CoreResult<()> {
        todo!()
    }

    fn release_iq1_s_repacks(&mut self) {
        todo!()
    }
}
