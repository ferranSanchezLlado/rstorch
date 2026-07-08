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
    pub label_smoothing: f64,
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

pub fn l1_loss<S, E, B>(
    pred: &Tensor<S, E, B>,
    target: &Tensor<S, E, B>,
    reduction: Reduction,
) -> Result<Scalar<E, B>>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    reduce_loss(&pred.sub(target)?.abs()?, reduction, pred.numel())
}

pub fn huber_loss<S, E, B>(
    pred: &Tensor<S, E, B>,
    target: &Tensor<S, E, B>,
    delta: E,
    reduction: Reduction,
) -> Result<Scalar<E, B>>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    let diff = pred.sub(target)?;
    let abs = diff.abs()?;
    let quadratic = diff.mul(&diff)?.mul_scalar(E::from_f64(0.5))?;
    let linear = abs
        .sub_scalar(delta * E::from_f64(0.5))?
        .mul_scalar(delta)?;
    let mask = abs.le_scalar(delta)?;
    reduce_loss(
        &quadratic.where_mask(&mask, &linear)?,
        reduction,
        pred.numel(),
    )
}

pub fn bce_with_logits_loss<S, E, B>(
    pred: &Tensor<S, E, B>,
    target: &Tensor<S, E, B>,
    reduction: Reduction,
) -> Result<Scalar<E, B>>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    let max_zero = pred.relu()?;
    let logits_target = pred.mul(target)?;
    let log_term = pred.neg()?.abs()?.neg()?.exp()?.add_scalar(E::ONE)?.ln()?;
    reduce_loss(
        &max_zero.sub(&logits_target)?.add(&log_term)?,
        reduction,
        pred.numel(),
    )
}

fn reduce_loss<S, E, B>(
    losses: &Tensor<S, E, B>,
    reduction: Reduction,
    count: usize,
) -> Result<Scalar<E, B>>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    match reduction {
        Reduction::Mean => losses.sum()?.div_scalar(E::from_usize(count)),
        Reduction::Sum => losses.sum(),
    }
}
