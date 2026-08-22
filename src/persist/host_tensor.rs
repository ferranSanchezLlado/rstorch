//! [`HostTensor`]: dtype + dims + raw little-endian bytes, with **no**
//! dependency on the live runtime tensor type.
//!
//! This is the persistence layer's currency. The crate's checkpoint runtime
//! bridges live tensors to and from `HostTensor`s; this module never touches a
//! live tensor, storage, or layout. Keeping the two
//! apart means the on-disk format can evolve without a tensor-core change and
//! vice-versa.

use crate::dtype::DType;
use crate::error::{Error, Result};
use safetensors::Dtype as StDtype;

/// A host-side, storage-independent tensor payload: its [`DType`], its
/// dimensions, and its contiguous **row-major little-endian** bytes.
///
/// A `HostTensor` is a plain owned value. It carries no device, no autograd
/// history, and no strides — it is always the canonical contiguous
/// interchange form. Constructing one validates that `bytes.len()` is exactly
/// `dtype.size_in_bytes() * dims.product()`, so a `HostTensor` in hand is
/// always internally consistent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostTensor {
    dtype: DType,
    dims: Vec<usize>,
    bytes: Vec<u8>,
}

impl HostTensor {
    /// Build a `HostTensor` from its parts, validating the byte length.
    ///
    /// Errors with [`Error::Persistence`] if `dims` overflows `usize` when
    /// multiplied out, or if `bytes.len()` does not equal
    /// `dtype.size_in_bytes() * dims.product()`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Persistence`] if `dims`'s element count overflows
    /// `usize`, or if `bytes.len()` does not equal
    /// `dtype.size_in_bytes() * dims.product()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rstorch::persist::HostTensor;
    /// use rstorch::DType;
    ///
    /// let bytes = 1.0f32.to_le_bytes().repeat(4);
    /// let host = HostTensor::from_bytes(DType::F32, vec![2, 2], bytes)?;
    /// assert_eq!(host.dims(), &[2, 2]);
    /// assert_eq!(host.dtype(), DType::F32);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn from_bytes(dtype: DType, dims: Vec<usize>, bytes: Vec<u8>) -> Result<Self> {
        let expected = byte_len(dtype, &dims)?;
        if bytes.len() != expected {
            return Err(Error::persistence(format!(
                "tensor byte length mismatch: dtype {dtype} dims {dims:?} expect {expected} bytes, got {}",
                bytes.len()
            )));
        }
        Ok(Self { dtype, dims, bytes })
    }

    /// The element dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// The dimensions, outermost first.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// The number of logical elements (`dims.product()`).
    pub fn num_elements(&self) -> usize {
        self.dims.iter().product()
    }

    /// The raw contiguous little-endian bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume the tensor, returning its `(dtype, dims, bytes)` parts.
    pub fn into_parts(self) -> (DType, Vec<usize>, Vec<u8>) {
        (self.dtype, self.dims, self.bytes)
    }
}

/// The checked byte length of a contiguous tensor with these `dims` and
/// `dtype`, or [`Error::Persistence`] on `usize` overflow.
pub(crate) fn byte_len(dtype: DType, dims: &[usize]) -> Result<usize> {
    let mut elems = 1usize;
    for &d in dims {
        elems = elems.checked_mul(d).ok_or_else(|| {
            Error::persistence(format!("element count overflow for dims {dims:?}"))
        })?;
    }
    elems.checked_mul(dtype.size_in_bytes()).ok_or_else(|| {
        Error::persistence(format!(
            "byte length overflow for dtype {dtype} dims {dims:?}"
        ))
    })
}

/// Map a crate [`DType`] to the safetensors [`Dtype`](safetensors::Dtype).
///
/// All six crate dtypes have exact safetensors counterparts, so this is total.
pub(crate) fn to_st_dtype(dtype: DType) -> StDtype {
    match dtype {
        DType::F16 => StDtype::F16,
        DType::BF16 => StDtype::BF16,
        DType::F32 => StDtype::F32,
        DType::F64 => StDtype::F64,
        DType::I64 => StDtype::I64,
        DType::Bool => StDtype::BOOL,
    }
}

/// Map a safetensors [`Dtype`](safetensors::Dtype) back to a crate [`DType`].
///
/// Errors with [`Error::Persistence`] for any dtype the crate does not model
/// (e.g. `U8`, `I32`, `C64`): the file is valid safetensors but not loadable
/// into this library, and that is reported loudly rather than silently
/// reinterpreted.
pub(crate) fn from_st_dtype(dtype: StDtype) -> Result<DType> {
    Ok(match dtype {
        StDtype::F16 => DType::F16,
        StDtype::BF16 => DType::BF16,
        StDtype::F32 => DType::F32,
        StDtype::F64 => DType::F64,
        StDtype::I64 => DType::I64,
        StDtype::BOOL => DType::Bool,
        other => {
            return Err(Error::persistence(format!(
                "unsupported safetensors dtype {other:?} (not one of the crate's six)"
            )));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_bytes_validates_length() {
        // 2 x 3 f32 => 24 bytes.
        let ok = HostTensor::from_bytes(DType::F32, vec![2, 3], vec![0u8; 24]);
        assert!(ok.is_ok());
        let t = ok.unwrap();
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.dims(), &[2, 3]);
        assert_eq!(t.num_elements(), 6);
        assert_eq!(t.bytes().len(), 24);

        let bad = HostTensor::from_bytes(DType::F32, vec![2, 3], vec![0u8; 20]);
        assert!(matches!(bad, Err(Error::Persistence { .. })));
    }

    #[test]
    fn into_parts_round_trips_from_bytes() {
        // The zero-copy inverse of `from_bytes`: the accessors only borrow,
        // so this is how a caller takes ownership of loaded tensor data
        // without copying it.
        let bytes: Vec<u8> = (0..24).collect();
        let t = HostTensor::from_bytes(DType::F32, vec![2, 3], bytes.clone()).unwrap();
        let (dtype, dims, out) = t.into_parts();
        assert_eq!(dtype, DType::F32);
        assert_eq!(dims, vec![2, 3]);
        assert_eq!(out, bytes);
        // and the parts rebuild the identical tensor.
        let rebuilt = HostTensor::from_bytes(dtype, dims, out).unwrap();
        assert_eq!(
            rebuilt,
            HostTensor::from_bytes(DType::F32, vec![2, 3], bytes).unwrap()
        );
    }

    #[test]
    fn scalar_and_empty() {
        // rank-0 scalar: one element.
        let scalar = HostTensor::from_bytes(DType::I64, vec![], vec![0u8; 8]).unwrap();
        assert_eq!(scalar.num_elements(), 1);
        // an axis of length 0 => zero elements, zero bytes.
        let empty = HostTensor::from_bytes(DType::F32, vec![0, 4], vec![]).unwrap();
        assert_eq!(empty.num_elements(), 0);
        assert!(empty.bytes().is_empty());
    }

    #[test]
    fn overflow_is_reported() {
        let bad = HostTensor::from_bytes(DType::F64, vec![usize::MAX, 2], vec![]);
        assert!(matches!(bad, Err(Error::Persistence { .. })));
    }

    #[test]
    fn dtype_round_trips_through_safetensors() {
        for dt in [
            DType::F16,
            DType::BF16,
            DType::F32,
            DType::F64,
            DType::I64,
            DType::Bool,
        ] {
            assert_eq!(from_st_dtype(to_st_dtype(dt)).unwrap(), dt);
        }
    }

    #[test]
    fn unsupported_safetensors_dtype_is_loud() {
        assert!(matches!(
            from_st_dtype(StDtype::U8),
            Err(Error::Persistence { .. })
        ));
        assert!(matches!(
            from_st_dtype(StDtype::I32),
            Err(Error::Persistence { .. })
        ));
    }
}
