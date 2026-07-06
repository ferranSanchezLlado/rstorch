use crate::backend::Backend;
use crate::dtype::FloatDType;
use crate::error::Result;
use crate::nn::{Layer, Module, RngSource, TrainingMode};
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

crate::nn::has_parameters! {
    impl[E, B] Dropout<E>
    where { }
    {
        params { }
        children { }
        transparent_children { }
    }
}
