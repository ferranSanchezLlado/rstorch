//! Host-boundary CPU kernels: transfer, contiguous materialization,
//! fills, and dtype casts.
//!
//! Signatures frozen by T01; **T10a** fills the bodies. Semantics are
//! specified on [`BackendOps`](crate::backend::BackendOps) — these are the
//! delegation targets of `CpuBackend`.

use crate::backend::View;
use crate::dtype::DType;
use crate::error::Result;
use crate::storage::{CpuStorage, Storage};

/// See [`BackendOps::transfer_in`](crate::backend::BackendOps::transfer_in).
/// For CPU this wraps the buffer unchanged.
pub(crate) fn transfer_in(host: CpuStorage) -> Result<Storage> {
    let _ = host;
    todo!("T10a: host transfer in")
}

/// See [`BackendOps::transfer_out`](crate::backend::BackendOps::transfer_out).
pub(crate) fn transfer_out(x: View<'_>) -> Result<CpuStorage> {
    let _ = x;
    todo!("T10a: host transfer out")
}

/// See [`BackendOps::copy_strided`](crate::backend::BackendOps::copy_strided).
pub(crate) fn copy_strided(x: View<'_>) -> Result<Storage> {
    let _ = x;
    todo!("T10a: strided copy")
}

/// See [`BackendOps::full`](crate::backend::BackendOps::full).
pub(crate) fn full(len: usize, dtype: DType, value: f64) -> Result<Storage> {
    let _ = (len, dtype, value);
    todo!("T10a: fill")
}

/// See [`BackendOps::cast`](crate::backend::BackendOps::cast). Lanes now:
/// F32↔I64, F32↔Bool, I64↔Bool (T60 adds F16/BF16).
pub(crate) fn cast(x: View<'_>, to: DType) -> Result<Storage> {
    let _ = (x, to);
    todo!("T10a: cast kernel")
}
