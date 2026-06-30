use crate::backend::Backend;
use crate::dtype::FloatDType;
use crate::error::Result;
use crate::nn::{
    HasParameters, Layer, Module, ParameterRef, ParameterRefMut, RngSource, TrainingMode,
};
use crate::shape::ShapeSpec;
use crate::tensor::{Mask, Tensor};

pub struct Dropout<E = f32>
where
    E: FloatDType,
{
    p: E,
}

impl<E> Dropout<E>
where
    E: FloatDType,
{
    pub fn new(p: E) -> Self {
        Self { p }
    }

    pub fn p(&self) -> E {
        self.p
    }
}

impl<S, E, B> Layer<Tensor<S, E, B>> for Dropout<E>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    type Output = Tensor<S, E, B>;
}

impl<S, E, B, Ctx> Module<Tensor<S, E, B>, Ctx> for Dropout<E>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
    Ctx: TrainingMode + RngSource,
{
    fn forward(&self, input: &Tensor<S, E, B>, ctx: &mut Ctx) -> Result<Self::Output> {
        if !ctx.is_training() || self.p <= E::ZERO {
            return Ok(input.clone());
        }

        let rng = ctx.rng();
        let keep_prob = E::ONE - self.p;
        let keep = (0..input.numel())
            .map(|_| rng.uniform(E::ZERO, E::ONE) < keep_prob)
            .collect();
        let mask = Mask::<S>::from_vec_with_shape(keep, input.shape().clone())?;
        let zeros = Tensor::<S, E, B>::zeros_with_shape(input.shape().clone())?;
        input
            .mul_scalar(E::ONE / keep_prob)?
            .where_mask(&mask, &zeros)
    }
}

impl<E, B> HasParameters<E, B> for Dropout<E>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn parameters<'a>(&'a self, _out: &mut Vec<ParameterRef<'a, E, B>>) {}

    fn parameters_mut<'a>(&'a mut self, _out: &mut Vec<ParameterRefMut<'a, E, B>>) {}
}
