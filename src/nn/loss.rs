use crate::backend::Backend;
use crate::dtype::FloatDType;
use crate::error::Result;
use crate::shape::{D2, DimSpec, ShapeSpec};
use crate::tensor::{Scalar, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    Mean,
    Sum,
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

pub fn cross_entropy<A, Cc, E, B>(
    logits: &Tensor<D2<A, Cc>, E, B>,
    targets: &[usize],
) -> Result<Scalar<E, B>>
where
    A: DimSpec,
    Cc: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    cross_entropy_with_reduction(logits, targets, Reduction::Mean)
}

pub fn cross_entropy_with_reduction<A, Cc, E, B>(
    logits: &Tensor<D2<A, Cc>, E, B>,
    targets: &[usize],
    reduction: Reduction,
) -> Result<Scalar<E, B>>
where
    A: DimSpec,
    Cc: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    let loss = logits.cross_entropy(targets)?;
    match reduction {
        Reduction::Mean => Ok(loss),
        Reduction::Sum => loss.mul_scalar(E::from_usize(targets.len())),
    }
}
