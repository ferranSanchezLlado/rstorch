//! Reduction CPU kernels.
//!
//! Signatures frozen by T01; **T11** fills the bodies — ported v2 loops
//! **adapted to the `Element::Acc` contract** (v2's native paths
//! accumulate in dtype; do not port verbatim). Semantics on
//! [`BackendOps`](crate::backend::BackendOps).

use crate::backend::{ArgReduceOp, ReduceOp, View};
use crate::error::Result;
use crate::storage::Storage;

/// See [`BackendOps::reduce`](crate::backend::BackendOps::reduce).
pub(crate) fn reduce(op: ReduceOp, x: View<'_>, axis: usize) -> Result<Storage> {
    let _ = (op, x, axis);
    todo!("T11: reduction kernels (Acc contract)")
}

/// See [`BackendOps::arg_reduce`](crate::backend::BackendOps::arg_reduce).
pub(crate) fn arg_reduce(op: ArgReduceOp, x: View<'_>, axis: usize) -> Result<Storage> {
    let _ = (op, x, axis);
    todo!("T11: arg-reduction kernels")
}
