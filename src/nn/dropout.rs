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

impl<S, E, B, Context> Module<Tensor<S, E, B>, Context> for Dropout<E>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
    Context: TrainingMode + RngSource,
{
    fn forward(&self, input: &Tensor<S, E, B>, ctx: &mut Context) -> Result<Self::Output> {
        if !ctx.is_training() || self.p <= E::ZERO {
            return Ok(input.clone());
        }

        let rng = ctx.rng();
        let keep_prob = E::ONE - self.p;
        let keep = (0..input.numel())
            .map(|_| rng.uniform(E::ZERO, E::ONE) < keep_prob)
            .collect();
        let mask = Mask::<S, B>::from_vec_with_shape(keep, input.shape().clone())?;
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
    fn visit_parameters<'a>(
        &'a self,
        _prefix: &str,
        _visit: &mut dyn FnMut(&str, ParameterRef<'a, E, B>),
    ) {
    }

    fn visit_parameters_mut<'a>(
        &'a mut self,
        _prefix: &str,
        _visit: &mut dyn FnMut(&str, ParameterRefMut<'a, E, B>),
    ) {
    }
}
