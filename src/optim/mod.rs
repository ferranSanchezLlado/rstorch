mod adam;
mod sgd;

use crate::backend::Backend;
use crate::dtype::FloatDType;
use crate::error::Result;
use crate::nn::{ParameterRef, ParameterRefMut};

pub use adam::Adam;
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
