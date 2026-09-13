//! Safetensors reader for deepseek-v4-flash checkpoint.
//!
//! Safetensors format:
//! `[8 bytes little-endian N][N bytes of JSON header][tensor data]`.
//! Each header entry: `{"dtype": ..., "shape": [...], "data_offsets": [start, end]}`.
//! So it's easy to parse it with serde_json library in rust. Right?
//!

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dtype {
    Unknown,
    U8,
    F8,
    Bf16,
    F16,
    F32,
    I8R,
    I64,
}

impl Dtype {
    pub fn elemsize(self) -> usize {
        match self {
            Dtype::U8 | Dtype::I8R | Dtype::F8 => 1,
            Dtype::Bf16 | Dtype::F16 => 2,
            Dtype::F32 => 4,
            Dtype::I64 => 8,
            Dtype::Unknown => 0,
        }
    }

    pub fn from_str(s: &str) -> Dtype {
        match s {
            "U8" => Dtype::U8,
            "BF16" => Dtype::Bf16,
            "F16" => Dtype::F16,
            "F32" => Dtype::F32,
            "I8" | "I8R" => Dtype::I8R,
            "F8_E4M3" => Dtype::F8,
            "I64" => Dtype::I64,
            _ => Dtype::Unknown,
        }
    }
}

pub const MAX_DIMS: usize = 4;

#[derive(Clone, Debug)]
pub struct Tensor {
    pub name: Box<str>,
    pub shard: usize,
    pub dtype: Dtype,
    shape: [u64; MAX_DIMS],
    rank: usize,
    pub mmap: Arc<Mmap>,
    pub data_range: std::ops::Range<usize>,
}

impl Tensor {
    #[inline]
    pub fn shape(&self) -> &[u64] {
        &self.shape[..self.rank]
    }

    /// Product of all shape dims, or 1 for a scalar (`shape == []`)
    pub fn num_elements(&self) -> u64 {
        self.shape().iter().fold(1, |acc, x| acc * x)
    }

    pub fn data(&self) -> &[u8] {
        &self.mmap[self.data_range.clone()]
    }

    pub fn byte_size(&self) -> usize {
        self.num_elements() as usize * self.dtype.elemsize()
    }
}

/// FNV-1a, hash function for the string.
#[inline]
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 14695981039346656037;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h
}

pub struct Safetensor {
    pub tensors: Vec<Tensor>,
    /// fnv1a(name) -> position in `tensors`
    pub index: HashMap<u64, u32>,
}

impl Safetensor {
    /// Open every `*.safetensors` in `dir` and index every tensor.
    pub fn open(dir: &Path) -> io::Result<Safetensor> {
        let mut shard_paths = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            if entry
                .file_name()
                .to_string_lossy()
                .ends_with(".safetensors")
            {
                shard_paths.push(entry.path());
            }
        }
        shard_paths.sort();
        if shard_paths.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("no .safetensors in {}", dir.display()),
            ));
        }
        let mut tensors: Vec<Tensor> = Vec::new();
        let mut index: HashMap<u64, u32> = HashMap::new();

        for (shard, path) in shard_paths.iter().enumerate() {
            let file = File::open(path)?;
            // Safety: the model file will not be modified by other processes while it is mapped.
            let mmap = unsafe { Mmap::map(&file)? };
            let mmap = Arc::new(mmap);

            // format: `[8 bytes little-endian N][N bytes of JSON header][tensor data]`.
            let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
            let header_json = &mmap[8..8 + header_len];
            let data_section_start = 8 + header_len;

            // Just parse the JSON now
            let root: serde_json::Value = serde_json::from_slice(&header_json).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{} header is not valid JSON: {}", path.display(), e),
                )
            })?;

            let obj = root.as_object().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{} header is not a JSON object", path.display()),
                )
            })?;

            for (name, val) in obj {
                // __metadata__ is not a tensor; its shape is arbitrary.
                if name == "__metadata__" {
                    continue;
                }
                let entry = val.as_object().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("k3_st: {}: entry {} is not an object", path.display(), name),
                    )
                })?;
                // Build a `Tensor` from a parsed header entry
                let dv = entry.get("dtype").and_then(|v| v.as_str()).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "{}: {} is missing dtype or data_offsets",
                            path.display(),
                            name
                        ),
                    )
                })?;
                let dtype = Dtype::from_str(dv);
                if dtype == Dtype::Unknown {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("{}: unsupported dtype '{}' on {}", path.display(), dv, name),
                    ));
                }

                let mut shape = [0u64; MAX_DIMS];
                let mut rank = 0usize;
                match entry.get("shape") {
                    Some(serde_json::Value::Array(a)) => {
                        if a.len() > MAX_DIMS {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!(
                                    "k3_st: {}: {} has rank {} (max {})",
                                    path.display(),
                                    name,
                                    a.len(),
                                    MAX_DIMS
                                ),
                            ));
                        }
                        for d in a {
                            let v = d.as_u64().ok_or_else(|| {
                                io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    format!(
                                        "k3_st: {}: {} has non-integer shape",
                                        path.display(),
                                        name
                                    ),
                                )
                            })?;
                            shape[rank] = v;
                            rank += 1;
                        }
                    }
                    Some(serde_json::Value::Null) | None => {}
                    Some(_) => {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("k3_st: {}: {} shape is not an array", path.display(), name),
                        ));
                    }
                }
                let offs = entry.get("data_offsets").ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "k3_st: {}: {} is missing dtype or data_offsets",
                            path.display(),
                            name
                        ),
                    )
                })?;
                let arr = offs.as_array().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "k3_st: {}: {} data_offsets is not an array",
                            path.display(),
                            name
                        ),
                    )
                })?;
                if arr.len() != 2 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "k3_st: {}: {} data_offsets is not a pair",
                            path.display(),
                            name
                        ),
                    ));
                }
                let start = data_section_start + arr[0].as_u64().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "k3_st: {}: {} data_offsets has non-integer start",
                            path.display(),
                            name
                        ),
                    )
                })? as usize;
                let end = data_section_start + arr[1].as_u64().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "k3_st: {}: {} data_offsets has non-integer end",
                            path.display(),
                            name
                        ),
                    )
                })? as usize;
                let h = fnv1a(name);
                index.insert(h, tensors.len() as u32);
                tensors.push(Tensor {
                    name: Box::from(name.as_str()),
                    shard,
                    dtype,
                    shape,
                    rank,
                    mmap: Arc::clone(&mmap),
                    data_range: start..end,
                });
            }
        }

        Ok(Safetensor { tensors, index })
    }
}
