//! Indexing CPU kernels.
//!
//! Signatures frozen by T01; **T25** fills the bodies. Semantics on
//! [`BackendOps`](crate::backend::BackendOps): index values are bounds
//! checked ([`Error::IndexOutOfBounds`](crate::Error)), never UB;
//! `scatter_add` accumulates in `Acc`.

use crate::backend::View;
use crate::error::Result;
use crate::storage::Storage;

/// See [`BackendOps::index_select`](crate::backend::BackendOps::index_select).
pub(crate) fn index_select(x: View<'_>, axis: usize, indices: View<'_>) -> Result<Storage> {
    let _ = (x, axis, indices);
    todo!("T25: index_select kernel")
}

/// See [`BackendOps::index_add`](crate::backend::BackendOps::index_add).
pub(crate) fn index_add(
    x: View<'_>,
    axis: usize,
    indices: View<'_>,
    src: View<'_>,
) -> Result<Storage> {
    let _ = (x, axis, indices, src);
    todo!("T25: index_add kernel (index_select backward)")
}

/// See [`BackendOps::gather`](crate::backend::BackendOps::gather).
pub(crate) fn gather(x: View<'_>, axis: usize, indices: View<'_>) -> Result<Storage> {
    let _ = (x, axis, indices);
    todo!("T25: gather kernel")
}

/// See [`BackendOps::scatter_add`](crate::backend::BackendOps::scatter_add).
pub(crate) fn scatter_add(
    x: View<'_>,
    axis: usize,
    indices: View<'_>,
    src: View<'_>,
) -> Result<Storage> {
    let _ = (x, axis, indices, src);
    todo!("T25: scatter_add kernel")
}
