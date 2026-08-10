//! Op families (shape, elementwise, reductions, matmul, indexing, conv,
//! losses), one file per family.
//!
//! The operand preconditions every family shares live here, once: there is no
//! implicit device transfer and no implicit dtype promotion, so mixing either
//! is a structured [`Error`](crate::Error) naming the public method. Each
//! family calls these rather than re-deriving them, so the wording of "wrong
//! device" is written in exactly one place.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

pub(crate) mod conv;
pub(crate) mod elementwise;
pub(crate) mod index;
pub(crate) mod loss;
pub(crate) mod matmul;
pub(crate) mod reduce;
pub(crate) mod shape;
pub(crate) mod sugar;

/// Both operands must live on the same device; there are no implicit
/// transfers. `expected` is `lhs`'s device — the op adopts the first operand's
/// placement and reports the second one as the offender.
pub(crate) fn same_device(op: &'static str, lhs: &Tensor, rhs: &Tensor) -> Result<()> {
    if lhs.device() != rhs.device() {
        return Err(Error::DeviceMismatch {
            op,
            expected: lhs.device(),
            got: rhs.device(),
        });
    }
    Ok(())
}

/// Both operands must share a dtype; there is no implicit promotion.
pub(crate) fn same_dtype(op: &'static str, lhs: &Tensor, rhs: &Tensor) -> Result<()> {
    if lhs.dtype() != rhs.dtype() {
        return Err(Error::DTypeMismatch {
            op,
            expected: lhs.dtype(),
            got: rhs.dtype(),
        });
    }
    Ok(())
}

/// Both operands must have the same rank (`cat`/`stack` operand lists).
pub(crate) fn same_rank(op: &'static str, lhs: &Tensor, rhs: &Tensor) -> Result<()> {
    if lhs.rank() != rhs.rank() {
        return Err(Error::RankMismatch {
            op,
            expected: lhs.rank(),
            got: rhs.rank(),
        });
    }
    Ok(())
}

/// The operand must have exactly `expected` dtype — a
/// [`Bool`](DType::Bool) mask/condition, or the
/// [`I64`](DType::I64) of an index or class-label tensor.
pub(crate) fn require_dtype(op: &'static str, t: &Tensor, expected: DType) -> Result<()> {
    if t.dtype() != expected {
        return Err(Error::DTypeMismatch {
            op,
            expected,
            got: t.dtype(),
        });
    }
    Ok(())
}
