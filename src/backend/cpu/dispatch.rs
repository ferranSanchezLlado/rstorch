//! The CPU backend's one runtime-dtype dispatch.
//!
//! Every kernel family faces the same problem: a [`View`](crate::backend::View)
//! carries its dtype as runtime data, but the loops want a concrete element
//! type. The answer is always the same shape — match the
//! [`DType`](crate::dtype::DType) **once**, outside the element loop, bind a
//! concrete type, and let the body monomorphize. Written out by hand that is a
//! six-arm match per kernel; six kernel modules had grown their own spelling
//! of it.
//!
//! This module owns both halves of that pattern:
//!
//! - [`CpuElement`] — the per-dtype plumbing a kernel needs *after* it has been
//!   routed: borrowing its operands out of the runtime-tagged [`CpuStorage`]
//!   and re-tagging its owned outputs. [`CpuFloat`] adds the accumulation-dtype
//!   and scalar-narrowing accessors only the fused kernels need.
//! - Three dispatch macros, one per dtype family, because the families really
//!   are different and folding them together would need a guard argument at
//!   every call site:
//!   [`dispatch_all!`] (all six — `index_select`, `gather`, `where`, host
//!   materialization), [`dispatch_numeric!`] (the five with a wide accumulator;
//!   `Bool` is [`Error::Unsupported`](crate::error::Error::Unsupported) — reduce,
//!   matmul, conv, index-add), and [`dispatch_float!`] (the four floats —
//!   softmax, `LayerNorm`, the optimizer steps, where an integer dtype has
//!   already been rejected upstream).
//!
//! Every accessor runs once per kernel call, outside the element loop; the
//! loops stay monomorphized over the concrete type exactly as they were when
//! each dtype had its own hand-written match arm.

use std::sync::Arc;

use crate::dtype::Element;
use crate::storage::{CpuStorage, Storage};

/// The per-dtype plumbing shared by every CPU kernel: borrow this dtype's
/// elements out of a [`CpuStorage`], and re-tag an owned buffer as storage.
///
/// Implemented for exactly the six [`Element`] types, so the set of impls
/// mirrors [`CpuStorage`]'s variants and a dispatch cannot miss one.
///
/// The borrowing accessors are **infallible**: every caller has already
/// dispatched on the view's validated runtime dtype, so a different variant is
/// an internal invariant break, not a user error. A kernel that must diagnose a
/// dtype mismatch (elementwise's `cpu_slice`, whose operands are not all
/// dispatched on) checks [`CpuStorage::dtype`] itself and keeps its own error
/// text.
pub(super) trait CpuElement: Element {
    /// The additive identity. Used only to pre-size output buffers whose slots
    /// are all overwritten, so `bool`'s `false` is a placeholder, not a claim
    /// about boolean arithmetic.
    const ZERO: Self;

    /// Borrow this dtype's elements.
    fn slice(storage: &CpuStorage) -> &[Self];

    /// Borrow the shared buffer itself, for kernels that re-share it rather
    /// than copying (host materialization's zero-copy case).
    fn buffer(storage: &CpuStorage) -> &Arc<Vec<Self>>;

    /// Borrow the shared buffer for in-place mutation, for the
    /// copy-into-destination contract.
    fn buffer_mut(storage: &mut CpuStorage) -> &mut Arc<Vec<Self>>;

    /// Re-tag an already-shared buffer as CPU storage.
    fn from_buffer(buffer: Arc<Vec<Self>>) -> CpuStorage;

    /// Re-tag an owned element buffer as CPU storage.
    fn cpu_storage(values: Vec<Self>) -> CpuStorage;

    /// Re-tag an owned element buffer as device storage.
    fn storage(values: Vec<Self>) -> Storage;
}

/// The extra plumbing the fused kernels need: the accumulation dtype (which
/// stays wide for `f16`/`bf16` parameters) and the `f64`-encoded scalars.
///
/// Implemented for the four float dtypes only — the same gate
/// [`FloatAcc`](super::acc::FloatAcc) applies one level down, so an integer or
/// bool dtype has no impl for a [`dispatch_float!`] to land on.
pub(super) trait CpuFloat: CpuElement {
    /// Borrow accumulation-dtype elements: optimizer state, or `LayerNorm`'s
    /// saved statistics.
    fn acc_slice(storage: &CpuStorage) -> &[Self::Acc];

    /// Re-tag an owned accumulation-dtype buffer as CPU storage.
    fn acc_storage(values: Vec<Self::Acc>) -> Storage;

    /// Narrow one `f64`-encoded scalar to the accumulation dtype, so a
    /// hyperparameter reaches the kernel at the precision the step runs in.
    fn scalar(value: f64) -> Self::Acc;
}

/// Generate one [`CpuElement`] impl.
///
/// The `#[inline]` on each method is load-bearing, not decoration: these are
/// one-line accessors, and leaving them out of line grew one module's share of
/// the crate's codegen units enough to change how `reduce.rs`'s hot loop was
/// partitioned, costing `max_last_f32` ~50% under the default
/// `codegen-units = 16`. Measured, then fixed by inlining. The same applies to
/// `impl_cpu_float!` below.
macro_rules! impl_cpu_element {
    ($ty:ty, $variant:ident, $zero:expr) => {
        impl CpuElement for $ty {
            const ZERO: Self = $zero;

            #[inline]
            fn slice(storage: &CpuStorage) -> &[Self] {
                Self::buffer(storage).as_slice()
            }

            #[inline]
            fn buffer(storage: &CpuStorage) -> &Arc<Vec<Self>> {
                match storage {
                    CpuStorage::$variant(values) => values,
                    _ => unreachable!("dispatched on the validated dtype"),
                }
            }

            #[inline]
            fn buffer_mut(storage: &mut CpuStorage) -> &mut Arc<Vec<Self>> {
                match storage {
                    CpuStorage::$variant(values) => values,
                    _ => unreachable!("dispatched on the validated dtype"),
                }
            }

            #[inline]
            fn from_buffer(buffer: Arc<Vec<Self>>) -> CpuStorage {
                CpuStorage::$variant(buffer)
            }

            #[inline]
            fn cpu_storage(values: Vec<Self>) -> CpuStorage {
                CpuStorage::$variant(Arc::new(values))
            }

            #[inline]
            fn storage(values: Vec<Self>) -> Storage {
                Storage::Cpu(Self::cpu_storage(values))
            }
        }
    };
}

impl_cpu_element!(half::f16, F16, half::f16::ZERO);
impl_cpu_element!(half::bf16, BF16, half::bf16::ZERO);
impl_cpu_element!(f32, F32, 0.0);
impl_cpu_element!(f64, F64, 0.0);
impl_cpu_element!(i64, I64, 0);
impl_cpu_element!(bool, Bool, false);

/// Generate one [`CpuFloat`] impl. See [`impl_cpu_element!`] on the `#[inline]`.
macro_rules! impl_cpu_float {
    ($ty:ty, $acc_variant:ident, $scalar:expr) => {
        impl CpuFloat for $ty {
            #[inline]
            fn acc_slice(storage: &CpuStorage) -> &[Self::Acc] {
                match storage {
                    CpuStorage::$acc_variant(values) => values.as_slice(),
                    _ => unreachable!("dispatched on the validated dtype"),
                }
            }

            #[inline]
            fn acc_storage(values: Vec<Self::Acc>) -> Storage {
                Storage::Cpu(CpuStorage::$acc_variant(Arc::new(values)))
            }

            #[inline]
            fn scalar(value: f64) -> Self::Acc {
                ($scalar)(value)
            }
        }
    };
}

impl_cpu_float!(half::f16, F32, |value| value as f32);
impl_cpu_float!(half::bf16, F32, |value| value as f32);
impl_cpu_float!(f32, F32, |value| value as f32);
impl_cpu_float!(f64, F64, |value| value);

/// Route a runtime [`DType`](crate::dtype::DType) to one monomorphized copy of
/// `$body`, with `$elem` bound to the concrete element type. All six dtypes,
/// so there is no fallback arm and no error to report.
macro_rules! dispatch_all {
    ($dtype:expr, $elem:ident => $body:block) => {
        match $dtype {
            $crate::dtype::DType::F16 => {
                type $elem = half::f16;
                $body
            }
            $crate::dtype::DType::BF16 => {
                type $elem = half::bf16;
                $body
            }
            $crate::dtype::DType::F32 => {
                type $elem = f32;
                $body
            }
            $crate::dtype::DType::F64 => {
                type $elem = f64;
                $body
            }
            $crate::dtype::DType::I64 => {
                type $elem = i64;
                $body
            }
            $crate::dtype::DType::Bool => {
                type $elem = bool;
                $body
            }
        }
    };
}

/// [`dispatch_all!`] over the five dtypes that have a wide accumulator.
///
/// `Bool` has none — [`NumAcc`](super::acc::NumAcc) is deliberately not
/// implemented for it — so a boolean reduction, matmul, convolution, or
/// index-accumulate is rejected here rather than given invented semantics one
/// level down. `$body` must therefore evaluate to a `Result`.
macro_rules! dispatch_numeric {
    ($dtype:expr, $op:expr, $device:expr, $elem:ident => $body:block) => {
        match $dtype {
            $crate::dtype::DType::F16 => {
                type $elem = half::f16;
                $body
            }
            $crate::dtype::DType::BF16 => {
                type $elem = half::bf16;
                $body
            }
            $crate::dtype::DType::F32 => {
                type $elem = f32;
                $body
            }
            $crate::dtype::DType::F64 => {
                type $elem = f64;
                $body
            }
            $crate::dtype::DType::I64 => {
                type $elem = i64;
                $body
            }
            $crate::dtype::DType::Bool => Err($crate::error::Error::Unsupported {
                op: $op,
                device: $device,
                dtype: $crate::dtype::DType::Bool,
            }),
        }
    };
}

/// [`dispatch_all!`] over the four float dtypes.
///
/// Unlike [`dispatch_numeric!`] this has no error arm: its callers (softmax,
/// `LayerNorm`, the optimizer steps) reject a non-float dtype up front with their
/// own `Error::Unsupported`, before any operand is borrowed.
macro_rules! dispatch_float {
    ($dtype:expr, $elem:ident => $body:block) => {
        match $dtype {
            $crate::dtype::DType::F16 => {
                type $elem = half::f16;
                $body
            }
            $crate::dtype::DType::BF16 => {
                type $elem = half::bf16;
                $body
            }
            $crate::dtype::DType::F32 => {
                type $elem = f32;
                $body
            }
            $crate::dtype::DType::F64 => {
                type $elem = f64;
                $body
            }
            $crate::dtype::DType::I64 | $crate::dtype::DType::Bool => {
                unreachable!("validated float dtype")
            }
        }
    };
}

pub(super) use {dispatch_all, dispatch_float, dispatch_numeric};
