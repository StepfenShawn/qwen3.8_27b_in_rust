//! NEON IQ kernels — not implemented yet; always fall back to scalar.

use crate::kernel::Q38Q8KBlock;

#[inline]
pub(crate) fn try_dot_iq1_s_repacked_q8_k(
    _weights: &[u8],
    _q: &[Q38Q8KBlock],
    _blocks: usize,
) -> f32 {
    todo!()
}

#[inline]
pub(crate) fn dot_iq4_xs_q8_k(_data: &[u8], _q: &[Q38Q8KBlock], _blocks: usize) -> f32 {
    todo!()
}

#[inline]
pub(crate) fn dot_iq4_nl_q8_k(_data: &[u8], _q: &[Q38Q8KBlock], _blocks: usize) -> f32 {
    todo!()
}

#[inline]
pub(crate) fn dot_iq3_s_q8_k(_data: &[u8], _q: &[Q38Q8KBlock], _blocks: usize) -> f32 {
    todo!()
}
