mod adam;
mod scheduler;
mod sgd;

use crate::backend::Backend;
use crate::dtype::FloatDType;
use crate::error::Result;
use crate::nn::{ParameterRef, ParameterRefMut};

pub use adam::{Adam, AdamW};
pub use scheduler::{ConstantLr, CosineLr, LrSchedule, StepLr, WarmupLr};
pub use sgd::Sgd;

pub trait Optimizer<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn step(&mut self, params: &mut [ParameterRefMut<'_, E, B>]) -> Result<()>;

    fn zero_grad(&mut self, params: &[ParameterRef<'_, E, B>]) {
        for param in params {
            param.zero_grad();
        }
    }
}

pub fn clip_grad_norm<E, B>(params: &mut [ParameterRefMut<'_, E, B>], max_norm: E) -> Result<E>
where
    E: FloatDType,
    B: Backend<E>,
{
    let mut sum_sq = 0.0;
    for param in params.iter() {
        if let Some(grad) = param.grad()? {
            for value in grad {
                let value = value.to_f64();
                sum_sq += value * value;
            }
        }
    }

    let norm = sum_sq.sqrt();
    if norm == 0.0 || norm <= max_norm.to_f64() {
        return Ok(E::from_f64(norm));
    }

    let scale = E::from_f64(max_norm.to_f64() / norm);
    for param in params.iter_mut() {
        if let Some(grad) = param.grad()? {
            param.set_grad(grad.into_iter().map(|value| value * scale).collect())?;
        }
    }
    Ok(E::from_f64(norm))
}
