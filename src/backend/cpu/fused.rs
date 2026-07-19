//! Fused CPU kernels (softmax, layernorm, optimizer updates) — the
//! measured answer to v2's fixed optimizer-allocation hotspot.
//!
//! Signature frozen by T01; **T48** fills the body, guarded by T47's
//! benchmark numbers. Until then the body must return
//! [`Error::Unsupported`](crate::Error) so the op layer composes the
//! unfused form (the only sanctioned fallback — same-device, no host
//! round-trip).

use crate::backend::{FusedOp, View};
use crate::error::Result;
use crate::storage::Storage;

/// See [`BackendOps::fused`](crate::backend::BackendOps::fused).
pub(crate) fn fused(op: FusedOp, inputs: &[View<'_>], scalars: &[f64]) -> Result<Storage> {
    let _ = (op, inputs, scalars);
    todo!("T48: fused kernels (return Unsupported until implemented)")
}
