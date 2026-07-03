use crate::backend::Backend;
use crate::dtype::FloatDType;
use crate::error::Result;
use crate::shape::ShapeSpec;
use crate::tensor::{Scalar, Tensor};

/// Reduction mode for scalar-returning losses.
///
/// `None` is intentionally not a variant: existing loss methods return a
/// scalar. A future unreduced loss should be a separate tensor-returning method,
/// not a new variant here.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Reduction {
    #[default]
    Mean,
    Sum,
}

#[derive(Debug, Clone, Copy, Default)]
#[non_exhaustive]
pub struct CrossEntropyOpts {
    pub reduction: Reduction,
    pub ignore_index: Option<usize>,
}

pub fn mse_loss<S, E, B>(pred: &Tensor<S, E, B>, target: &Tensor<S, E, B>) -> Result<Scalar<E, B>>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    let diff = pred.sub(target)?;
    diff.mul(&diff)?
        .sum()?
        .div_scalar(E::from_usize(pred.numel()))
}
