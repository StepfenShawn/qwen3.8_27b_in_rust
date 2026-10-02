/// Number of weights per Q8_K block.
pub const Q38_Q8_K_BLOCK_SIZE: usize = 256;

/// One Q8_K activation quantization block.
pub struct Q38Q8KBlock {
    /// Per-block dequantization scala.
    pub scale: f32,
    /// Signed 8-bit quants, one per weight.
    pub quants: [i8; Q38_Q8_K_BLOCK_SIZE],
    /// Sum of each 16-wide group of quants.
    pub sums: [i16; Q38_Q8_K_BLOCK_SIZE / 16],
}

impl Default for Q38Q8KBlock {
    #[inline]
    fn default() -> Self {
        Self {
            scale: 0.0,
            quants: [0; Q38_Q8_K_BLOCK_SIZE],
            sums: [0; Q38_Q8_K_BLOCK_SIZE / 16],
        }
    }
}

/// Errors produced by the quantization and tensor kenel.
#[derive(Debug)]
pub enum Q38CoreError {
    /// Activation length is not an integer number of Q8_K blocks.
    LengthNotDivisibleByBlock { length: u64 },
    /// A caller-supplied buffer has the wrong length.
    ShapeMismatch { expected: u64, got: u64 },
    /// Requested row lies outside `[0, rows)`.
    RowOutOfBounds { row: u64, rows: u64 },
    /// The tensor's quant type is not handled by this kernel.
    UnsupportedQuantType(u32),
    /// Underlying GGUF read failed.
    Gguf(String),
}

impl std::fmt::Display for Q38CoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Q38CoreError::LengthNotDivisibleByBlock { length } => write!(
                f,
                "input length {length} is not a multiple of {Q38_Q8_K_BLOCK_SIZE}"
            ),
            Q38CoreError::ShapeMismatch { expected, got } => {
                write!(f, "buffer length mismatch: expected {expected}, got {got}")
            }
            Q38CoreError::RowOutOfBounds { row, rows } => {
                write!(f, "row {row} is outside [0, {rows})")
            }
            Q38CoreError::UnsupportedQuantType(t) => {
                write!(f, "unsupported quant type {t}")
            }
            Q38CoreError::Gguf(msg) => write!(f, "GGUF error: {msg}"),
        }
    }
}

impl std::error::Error for Q38CoreError {}
pub type Q38CoreResult<T> = Result<T, Q38CoreError>;

pub trait Q38QuantOps {
    fn f16_to_f32(value: u16) -> f32;
    fn quantize_q8_k(output: &mut [Q38Q8KBlock], input: &[f32]) -> Q38CoreResult<()>;
}

/// Row-level and matrix-level kernels over a GGUF tensor.
pub trait Q38TensorOps {
    /// Decode a single logical row into `output` as f32.
    fn tensor_row_f32(&self, output: &mut [f32], row: u64) -> Q38CoreResult<()>;

    /// Dot product between `input` and one logical row of the tensor.
    fn tensor_dot_row_f32(&self, input: &[f32], row: u64) -> Q38CoreResult<f32>;

    /// Matrix-vector product: `output = tensor * input`.
    fn tensor_gemv_f32(&self, output: &mut [f32], input: &[f32]) -> Q38CoreResult<()>;

    /// Integer dot-product GEMV for supported K-quant / IQ matrices.
    fn tensor_gemv_q8_k(&self, output: &mut [f32], input: &[Q38Q8KBlock]) -> Q38CoreResult<()>;

    /// Batched GEMV. Weight rows form the outer loop, so a resident row is
    /// used against every input row before the kernel advances through the
    /// model file.
    fn tensor_gemm_f32(
        &self,
        output: &mut [f32],
        input: &[f32],
        batch_size: u32,
    ) -> Q38CoreResult<()>;

    /// Batched integer dot-product GEMV.
    fn tensor_gemm_q8_k(
        &self,
        output: &mut [f32],
        input: &[Q38Q8KBlock],
        batch_size: u32,
    ) -> Q38CoreResult<()>;
}

/// SIMD-friendly IQ1_S repacks.
pub trait Q38Iq1sRepack {
    /// Build the SIMD views. Idempotent if already prepared.
    fn prepare_iq1_s_repacks(&mut self) -> Q38CoreResult<()>;

    /// Release the SIMD views and any associated allocations.
    fn release_iq1_s_repacks(&mut self);
}

pub mod ops;
