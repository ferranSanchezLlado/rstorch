use crate::backend::{Backend, Cpu};
use crate::dtype::FloatDType;
use crate::error::Result;
use crate::nn::{Layer, Module, Parameter};
use crate::shape::{C, D1, D2, DimSpec};
use crate::tensor::Tensor;

/// Last-dimension layer normalization for 2D tensors.
///
/// Variance is computed inside this layer from existing tensor operations.
/// Parameter names are part of the persistence contract:
/// `weight` and `bias` both have shape `[FEATURES]`.
pub struct LayerNorm<const FEATURES: usize, E = f32, B = Cpu>
where
    E: FloatDType,
    B: Backend<E>,
{
    weight: Parameter<D1<C<FEATURES>>, E, B>,
    bias: Parameter<D1<C<FEATURES>>, E, B>,
    eps: E,
}

impl<const FEATURES: usize, E, B> LayerNorm<FEATURES, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn new() -> Result<Self> {
        Self::with_eps(E::from_f64(1e-5))
    }

    pub fn with_eps(eps: E) -> Result<Self> {
        Ok(Self {
            weight: Parameter::new(Tensor::ones()?),
            bias: Parameter::new(Tensor::zeros()?),
            eps,
        })
    }

    pub fn weight(&self) -> &Parameter<D1<C<FEATURES>>, E, B> {
        &self.weight
    }

    pub fn bias(&self) -> &Parameter<D1<C<FEATURES>>, E, B> {
        &self.bias
    }

    fn normalize<A>(
        &self,
        input: &Tensor<D2<A, C<FEATURES>>, E, B>,
    ) -> Result<Tensor<D2<A, C<FEATURES>>, E, B>>
    where
        A: DimSpec,
    {
        let mean = input.mean_last()?;
        let centered = input.sub_leading_dim(&mean)?;
        let var = centered.mul(&centered)?.mean_last()?;
        let denom = var.add_scalar(self.eps)?.sqrt()?;
        centered
            .div_leading_dim(&denom)?
            .mul_last_dim(self.weight.tensor())?
            .add_last_dim(self.bias.tensor())
    }
}

impl<const FEATURES: usize, A, E, B> Layer<Tensor<D2<A, C<FEATURES>>, E, B>>
    for LayerNorm<FEATURES, E, B>
where
    A: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    type Output = Tensor<D2<A, C<FEATURES>>, E, B>;
}

impl<const FEATURES: usize, A, E, B, Context> Module<Tensor<D2<A, C<FEATURES>>, E, B>, Context>
    for LayerNorm<FEATURES, E, B>
where
    A: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    fn forward(
        &self,
        input: &Tensor<D2<A, C<FEATURES>>, E, B>,
        _ctx: &mut Context,
    ) -> Result<Self::Output> {
        self.normalize(input)
    }
}

crate::nn::has_parameters! {
    impl[const FEATURES: usize, E, B] LayerNorm<FEATURES, E, B>
    where { }
    {
        params { weight, bias }
        children { }
        transparent_children { }
    }
}
