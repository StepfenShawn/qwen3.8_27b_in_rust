//! NEON tensor kernels — not implemented yet; always fall back to scalar.

use crate::kernel::Q38Q8KBlock;

#[inline]
pub(crate) fn try_dot_f32(_data: &[u8], _input: &[f32], _n: usize) -> Option<f32> {
    None
}

#[inline]
pub(crate) fn try_dot_f16(_data: &[u8], _input: &[f32], _n: usize) -> Option<f32> {
    None
}

#[inline]
pub(crate) fn try_dot_q8_0(_data: &[u8], _input: &[f32], _n: usize) -> Option<f32> {
    None
}

#[inline]
pub(crate) fn try_dot_q3_k_q8_k(
    _data: &[u8],
    _q: &[Q38Q8KBlock],
    _blocks: usize,
) -> Option<f32> {
    None
}

#[inline]
pub(crate) fn try_dot_q4_k_q8_k(
    _data: &[u8],
    _q: &[Q38Q8KBlock],
    _blocks: usize,
) -> Option<f32> {
    None
}

#[inline]
pub(crate) fn try_dot_q5_k_q8_k(
    _data: &[u8],
    _q: &[Q38Q8KBlock],
    _blocks: usize,
) -> Option<f32> {
    None
}

#[inline]
pub(crate) fn try_dot_q6_k_q8_k(
    _data: &[u8],
    _q: &[Q38Q8KBlock],
    _blocks: usize,
) -> Option<f32> {
    None
}
