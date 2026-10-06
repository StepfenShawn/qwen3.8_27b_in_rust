//! Optional derived views over the GGUF mapping.
//!
//! The mapped tensor payload is authoritative and is never modified. IQ1_S
//! blocks however are laid out for a bit-unpacking decoder, which forces the
//! inner loop to rebuild grid indices and sign masks on every dot product.
//! [`Q38Iq1sRepack`] builds a second, SIMD-friendly copy of just those blocks
//! so the hot loop can stream `iq1s_grid` straight into vector registers, and
//! releases it again when the caller is done.

use crate::gguf::{GgmlType, Gguf};
use crate::kernel::iq::{repack_iq1_s, IQ1S_BLOCK_BYTES};
use crate::kernel::{Q38CoreError, Q38CoreResult, Q38Iq1sRepack, Q38_Q8_K_BLOCK_SIZE};

impl Q38Iq1sRepack for Gguf {
    fn prepare_iq1_s_repacks(&mut self) -> Q38CoreResult<()> {
        for tensor in self.tensors_mut() {
            if tensor.ty != GgmlType::Iq1S || tensor.n_dims != 2 {
                continue;
            }
            let width = tensor.shape[0];
            let rows = tensor.shape[1];
            if width % Q38_Q8_K_BLOCK_SIZE as u64 != 0 {
                continue;
            }
            let blocks = (width / Q38_Q8_K_BLOCK_SIZE as u64) as usize;
            let block_count =
                blocks
                    .checked_mul(rows as usize)
                    .ok_or(Q38CoreError::ShapeMismatch {
                        expected: width,
                        got: rows,
                    })?;
            let source_bytes = block_count * IQ1S_BLOCK_BYTES;
            if source_bytes > tensor.data.len() {
                return Err(Q38CoreError::ShapeMismatch {
                    expected: tensor.data.len() as u64,
                    got: source_bytes as u64,
                });
            }
            tensor.iq1_s_repack = Some(repack_iq1_s(tensor.data, block_count).into_boxed_slice());
        }
        Ok(())
    }
}
