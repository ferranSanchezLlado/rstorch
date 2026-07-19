//! Element-wise CPU kernels: the strided + broadcast-aware iteration
//! engine (the largest chunk of new tensor-core work, exploration §3.2)
//! and the binary/unary/compare/where/masked-fill families built on it.
//!
//! Signatures frozen by T01; **T10b** fills the bodies (one internal
//! dtype-dispatch macro; contiguous fast path ported from v2; `parallel.rs`
//! rayon switch). Semantics on [`BackendOps`](crate::backend::BackendOps):
//! multi-input kernels receive pre-broadcast, shape-identical views.

use crate::backend::{BinaryOp, CmpOp, UnaryOp, View};
use crate::error::Result;
use crate::storage::Storage;

/// See [`BackendOps::binary`](crate::backend::BackendOps::binary).
pub(crate) fn binary(op: BinaryOp, lhs: View<'_>, rhs: View<'_>) -> Result<Storage> {
    let _ = (op, lhs, rhs);
    todo!("T10b: binary kernels")
}

/// See [`BackendOps::binary_scalar`](crate::backend::BackendOps::binary_scalar).
pub(crate) fn binary_scalar(op: BinaryOp, x: View<'_>, scalar: f64) -> Result<Storage> {
    let _ = (op, x, scalar);
    todo!("T10b: scalar binary kernels")
}

/// See [`BackendOps::unary`](crate::backend::BackendOps::unary). Note the
/// exact-GELU contract on [`UnaryOp::Gelu`](crate::backend::UnaryOp).
pub(crate) fn unary(op: UnaryOp, x: View<'_>) -> Result<Storage> {
    let _ = (op, x);
    todo!("T10b: unary kernels")
}

/// See [`BackendOps::compare`](crate::backend::BackendOps::compare).
pub(crate) fn compare(op: CmpOp, lhs: View<'_>, rhs: View<'_>) -> Result<Storage> {
    let _ = (op, lhs, rhs);
    todo!("T10b: comparison kernels")
}

/// See [`BackendOps::where_cond`](crate::backend::BackendOps::where_cond).
pub(crate) fn where_cond(cond: View<'_>, on_true: View<'_>, on_false: View<'_>) -> Result<Storage> {
    let _ = (cond, on_true, on_false);
    todo!("T10b: where kernel")
}

/// See [`BackendOps::masked_fill`](crate::backend::BackendOps::masked_fill).
pub(crate) fn masked_fill(x: View<'_>, mask: View<'_>, value: f64) -> Result<Storage> {
    let _ = (x, mask, value);
    todo!("T10b: masked_fill kernel")
}
