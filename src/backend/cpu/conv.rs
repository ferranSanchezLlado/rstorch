//! Convolution/pooling CPU kernels.
//!
//! Signature frozen by T01; **T26** fills the body. Per-variant input
//! contracts on [`ConvOp`](crate::backend::ConvOp); geometry in
//! [`Conv2dParams`](crate::backend::Conv2dParams); accumulation in `Acc`.

use crate::backend::{Conv2dParams, ConvOp, View};
use crate::error::Result;
use crate::storage::Storage;

/// See [`BackendOps::conv`](crate::backend::BackendOps::conv).
pub(crate) fn conv(op: ConvOp, inputs: &[View<'_>], params: &Conv2dParams) -> Result<Storage> {
    let _ = (op, inputs, params);
    todo!("T26: conv/pool kernels")
}
