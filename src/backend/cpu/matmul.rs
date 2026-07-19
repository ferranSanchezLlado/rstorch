//! Matmul CPU kernel.
//!
//! Signature frozen by T01; **T11** fills the body — ported v2 loops
//! adapted to the `Element::Acc` contract, batched over leading dims.
//! Semantics on [`BackendOps`](crate::backend::BackendOps).

use crate::backend::View;
use crate::error::Result;
use crate::storage::Storage;

/// See [`BackendOps::matmul`](crate::backend::BackendOps::matmul).
pub(crate) fn matmul(lhs: View<'_>, rhs: View<'_>) -> Result<Storage> {
    let _ = (lhs, rhs);
    todo!("T11: batched matmul (Acc contract)")
}
