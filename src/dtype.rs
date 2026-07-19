//! Runtime dtypes and the sealed [`Element`] trait.
//!
//! Dtype is runtime data (exploration §4.1): [`DType`] is a plain enum, and
//! the only public generic in the tensor math surface is `T: Element` on
//! `from_vec` / `to_vec` / `to_scalar`.
//!
//! Dtype scope (exploration §4.2): epoch-1 kernels implement `F32`, `I64`,
//! `Bool`; `F16`/`BF16` arrive in the dedicated precision epoch (T60) under
//! the `Acc` contract; `F64` follows demand. All six [`Element`] impls exist
//! from the start so signatures never change.

use crate::error::{Error, Result};
use crate::storage::CpuStorage;
use std::sync::Arc;

/// The runtime element type of a tensor.
///
/// There is **no implicit promotion** between dtypes: mixing dtypes in an
/// op is a structured [`Error::DTypeMismatch`] telling you to cast with
/// `to_dtype`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DType {
    /// IEEE 754 half precision (16-bit) float.
    F16,
    /// bfloat16: 8-bit exponent, 7-bit mantissa.
    BF16,
    /// IEEE 754 single precision (32-bit) float.
    F32,
    /// IEEE 754 double precision (64-bit) float.
    F64,
    /// 64-bit signed integer (class labels, indices).
    I64,
    /// Boolean (comparison results, masks).
    Bool,
}

impl DType {
    /// Size of one element in bytes.
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::F16 | DType::BF16 => 2,
            DType::F32 => 4,
            DType::F64 | DType::I64 => 8,
            DType::Bool => 1,
        }
    }

    /// Whether this is a floating-point dtype (`F16`/`BF16`/`F32`/`F64`).
    ///
    /// Only floating-point tensors participate in autograd; stochastic
    /// constructors (`rand`, `randn`) require a float dtype.
    pub fn is_float(self) -> bool {
        matches!(self, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::F32 => "f32",
            DType::F64 => "f64",
            DType::I64 => "i64",
            DType::Bool => "bool",
        };
        f.write_str(s)
    }
}

mod sealed {
    /// The seal: a marker with no methods, in a private module so no
    /// downstream crate can name it and thus none can implement
    /// [`Element`](super::Element). This closes the dtype set (adding a dtype
    /// is a crate-internal, compiler-guided change with no semver event)
    /// **without** putting any crate-private type on the public interface.
    pub trait Sealed {}
}

/// A Rust scalar type that can be a tensor element. Sealed: exactly six
/// implementations exist ([`f32`], [`f64`], [`half::f16`], [`half::bf16`],
/// [`i64`], [`bool`]), mirroring [`DType`].
///
/// # The `Acc` wide-accumulation contract
///
/// `Acc` is the type kernels **must** accumulate in for reductions, matmul
/// inner products, and broadcast-reducing backward passes (exploration
/// §3.1, the implemented fix for the v2 f16 sum-saturation bug): for
/// `f16`/`bf16`, `Acc = f32`; every other element accumulates as itself.
/// Counts (e.g. `mean` divisors) are computed in `Acc` as well; the result
/// is cast back to `Self` exactly once at output.
pub trait Element: sealed::Sealed + Copy + Send + Sync + std::fmt::Debug + 'static {
    /// The [`DType`] tag corresponding to this element type.
    const DTYPE: DType;

    /// The wide accumulator type (see trait-level docs).
    type Acc: Element;

    /// Widen for accumulation.
    fn to_acc(self) -> Self::Acc;

    /// Narrow an accumulated value back to the element type (the single
    /// output cast of the accumulation contract).
    fn from_acc(acc: Self::Acc) -> Self;
}

/// Crate-private host ↔ [`CpuStorage`] plumbing, kept **off** the public
/// [`Element`] trait so the crate-internal `CpuStorage` never leaks into the
/// public interface. Implemented for exactly the six [`Element`] types;
/// generic CPU kernel/constructor code bounds `E: HostConv`.
// Consumed by T10a/T20 (from_vec/to_vec, host transfer); the integrator
// removes this allow once those land.
#[allow(dead_code)]
pub(crate) trait HostConv: Element {
    /// Wrap a host vector in the matching [`CpuStorage`] variant.
    fn into_cpu_storage(v: Vec<Self>) -> CpuStorage;

    /// Extract a host vector from [`CpuStorage`], erroring with
    /// [`Error::DTypeMismatch`] if the storage holds a different dtype. The
    /// storage must be contiguous interchange data (offset 0); this is a
    /// plain element copy, not a strided gather.
    fn try_from_cpu_storage(s: &CpuStorage, op: &'static str) -> Result<Vec<Self>>;
}

macro_rules! impl_element {
    ($ty:ty, $dtype:ident, $acc:ty, $variant:ident, to_acc: $to_acc:expr, from_acc: $from_acc:expr) => {
        impl sealed::Sealed for $ty {}

        impl HostConv for $ty {
            fn into_cpu_storage(v: Vec<Self>) -> CpuStorage {
                CpuStorage::$variant(Arc::new(v))
            }

            fn try_from_cpu_storage(s: &CpuStorage, op: &'static str) -> Result<Vec<Self>> {
                match s {
                    CpuStorage::$variant(v) => Ok(v.as_ref().clone()),
                    other => Err(Error::DTypeMismatch {
                        op,
                        expected: DType::$dtype,
                        got: other.dtype(),
                    }),
                }
            }
        }

        impl Element for $ty {
            const DTYPE: DType = DType::$dtype;
            type Acc = $acc;

            fn to_acc(self) -> Self::Acc {
                ($to_acc)(self)
            }

            fn from_acc(acc: Self::Acc) -> Self {
                ($from_acc)(acc)
            }
        }
    };
}

impl_element!(half::f16, F16, f32, F16, to_acc: half::f16::to_f32, from_acc: half::f16::from_f32);
impl_element!(half::bf16, BF16, f32, BF16, to_acc: half::bf16::to_f32, from_acc: half::bf16::from_f32);
impl_element!(f32, F32, f32, F32, to_acc: |x| x, from_acc: |x| x);
impl_element!(f64, F64, f64, F64, to_acc: |x| x, from_acc: |x| x);
impl_element!(i64, I64, i64, I64, to_acc: |x| x, from_acc: |x| x);
// `bool` is never accumulated; `Acc = bool` documents that reductions over
// Bool are not part of the kernel contract (compare/mask ops only).
impl_element!(bool, Bool, bool, Bool, to_acc: |x| x, from_acc: |x| x);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_tags_match() {
        assert_eq!(<half::f16 as Element>::DTYPE, DType::F16);
        assert_eq!(<half::bf16 as Element>::DTYPE, DType::BF16);
        assert_eq!(<f32 as Element>::DTYPE, DType::F32);
        assert_eq!(<f64 as Element>::DTYPE, DType::F64);
        assert_eq!(<i64 as Element>::DTYPE, DType::I64);
        assert_eq!(<bool as Element>::DTYPE, DType::Bool);
    }

    #[test]
    fn acc_widens_half_precision() {
        // 2049 is not representable in f16 (rounds to 2048): the Acc
        // contract exists exactly so sums like this do not saturate.
        let x = half::f16::from_f32(2048.0);
        let acc = x.to_acc() + 1.0f32;
        assert_eq!(acc, 2049.0f32);
        assert_eq!(<half::f16 as Element>::DTYPE.size_in_bytes(), 2);
    }

    #[test]
    fn float_classification() {
        assert!(DType::F16.is_float());
        assert!(DType::BF16.is_float());
        assert!(DType::F32.is_float());
        assert!(DType::F64.is_float());
        assert!(!DType::I64.is_float());
        assert!(!DType::Bool.is_float());
    }

    #[test]
    fn display_names() {
        assert_eq!(DType::F32.to_string(), "f32");
        assert_eq!(DType::BF16.to_string(), "bf16");
        assert_eq!(DType::Bool.to_string(), "bool");
    }

    #[test]
    fn cpu_storage_round_trip_checks_dtype() {
        let s = <f32 as HostConv>::into_cpu_storage(vec![1.0, 2.0]);
        assert_eq!(
            <f32 as HostConv>::try_from_cpu_storage(&s, "test").unwrap(),
            vec![1.0, 2.0]
        );
        assert!(matches!(
            <i64 as HostConv>::try_from_cpu_storage(&s, "test"),
            Err(Error::DTypeMismatch {
                op: "test",
                expected: DType::I64,
                got: DType::F32
            })
        ));
    }
}
