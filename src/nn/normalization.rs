use crate::backend::{Backend, Cpu};
use crate::dtype::FloatDType;
use crate::error::Result;
use crate::nn::{HasParameters, Layer, Module, Parameter, ParameterRef, ParameterRefMut};
use crate::shape::{C, D1, D2, DimSpec};
use crate::tensor::Tensor;

/// Last-dimension layer normalization for 2D tensors.
///
/// Variance is computed inside this layer from existing tensor operations.
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
    pub fn new(eps: E) -> Result<Self> {
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
        let mean = input.mean_axis1()?;
        let centered = input.sub_col(&mean)?;
        let var = centered.mul(&centered)?.mean_axis1()?;
        let denom = var.add_scalar(self.eps)?.sqrt()?;
        centered
            .div_col(&denom)?
            .mul_row(self.weight.tensor())?
            .add_row(self.bias.tensor())
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

impl<const FEATURES: usize, A, E, B, Ctx> Module<Tensor<D2<A, C<FEATURES>>, E, B>, Ctx>
    for LayerNorm<FEATURES, E, B>
where
    A: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    fn forward(
        &self,
        input: &Tensor<D2<A, C<FEATURES>>, E, B>,
        _ctx: &mut Ctx,
    ) -> Result<Self::Output> {
        self.normalize(input)
    }
}

impl<const FEATURES: usize, E, B> HasParameters<E, B> for LayerNorm<FEATURES, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn parameters<'a>(&'a self, out: &mut Vec<ParameterRef<'a, E, B>>) {
        out.push(self.weight.as_ref());
        out.push(self.bias.as_ref());
    }

    fn parameters_mut<'a>(&'a mut self, out: &mut Vec<ParameterRefMut<'a, E, B>>) {
        out.push(self.weight.as_mut());
        out.push(self.bias.as_mut());
    }
}
