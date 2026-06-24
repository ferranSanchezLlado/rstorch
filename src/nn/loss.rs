use crate::backend::Backend;
use crate::dtype::FloatDType;
use crate::error::Result;
use crate::shape::ShapeSpec;
use crate::tensor::{Scalar, Tensor};

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
