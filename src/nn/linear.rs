use super::parameter::{HasParameters, Layer, Module, Parameter, ParameterRef, ParameterRefMut};
use crate::backend::{Backend, Cpu};
use crate::dtype::FloatDType;
use crate::error::{Result, const_check};
use crate::random::SmallRng;
use crate::shape::{C, D1, D2, DimSpec};
use crate::tensor::Tensor;

pub struct Linear<const IN: usize, const OUT: usize, E = f32, B = Cpu>
where
    E: FloatDType,
    B: Backend<E>,
{
    weight: Parameter<D2<C<IN>, C<OUT>>, E, B>,
    bias: Parameter<D1<C<OUT>>, E, B>,
}

impl<const IN: usize, const OUT: usize, E, B> Linear<IN, OUT, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn zeros() -> Result<Self> {
        Ok(Self {
            weight: Parameter::new(Tensor::zeros()?),
            bias: Parameter::new(Tensor::zeros()?),
        })
    }

    pub fn xavier_uniform(rng: &mut SmallRng) -> Result<Self> {
        const {
            const_check::sum_nonzero(
                IN,
                OUT,
                "xavier_uniform",
                "fan_in (IN)",
                "fan_out (OUT)",
                "fan_in + fan_out (IN+OUT)",
            );
        };

        let limit = (6.0 / ((IN + OUT) as f64)).sqrt();
        Self::uniform(rng, -E::from_f64(limit), E::from_f64(limit))
    }

    pub fn kaiming_uniform(rng: &mut SmallRng) -> Result<Self> {
        const { const_check::nonzero(IN, "kaiming_uniform", "fan_in (IN)") };

        let limit = (6.0 / (IN as f64)).sqrt();
        Self::uniform(rng, -E::from_f64(limit), E::from_f64(limit))
    }

    fn uniform(rng: &mut SmallRng, low: E, high: E) -> Result<Self> {
        const { const_check::mul_fits(IN, OUT, "linear_uniform", "IN", "OUT") };

        let weight = (0..IN * OUT).map(|_| rng.uniform(low, high)).collect();
        Ok(Self {
            weight: Parameter::new(Tensor::from_vec(weight)?),
            bias: Parameter::new(Tensor::zeros()?),
        })
    }

    pub fn weight(&self) -> &Parameter<D2<C<IN>, C<OUT>>, E, B> {
        &self.weight
    }

    pub fn bias(&self) -> &Parameter<D1<C<OUT>>, E, B> {
        &self.bias
    }
}

impl<Batch, const IN: usize, const OUT: usize, E, B> Layer<Tensor<D2<Batch, C<IN>>, E, B>>
    for Linear<IN, OUT, E, B>
where
    Batch: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    type Output = Tensor<D2<Batch, C<OUT>>, E, B>;
}

impl<Batch, const IN: usize, const OUT: usize, E, B, Context>
    Module<Tensor<D2<Batch, C<IN>>, E, B>, Context> for Linear<IN, OUT, E, B>
where
    Batch: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    fn forward(
        &self,
        input: &Tensor<D2<Batch, C<IN>>, E, B>,
        _ctx: &mut Context,
    ) -> Result<Self::Output> {
        input
            .matmul(self.weight.tensor())?
            .add_last_dim(self.bias.tensor())
    }
}

impl<const IN: usize, const OUT: usize, E, B> HasParameters<E, B> for Linear<IN, OUT, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn visit_parameters<'a>(
        &'a self,
        prefix: &str,
        visit: &mut dyn FnMut(&str, ParameterRef<'a, E, B>),
    ) {
        visit(
            &crate::nn::parameter_path(prefix, "weight"),
            self.weight.as_ref(),
        );
        visit(
            &crate::nn::parameter_path(prefix, "bias"),
            self.bias.as_ref(),
        );
    }

    fn visit_parameters_mut<'a>(
        &'a mut self,
        prefix: &str,
        visit: &mut dyn FnMut(&str, ParameterRefMut<'a, E, B>),
    ) {
        visit(
            &crate::nn::parameter_path(prefix, "weight"),
            self.weight.as_mut(),
        );
        visit(
            &crate::nn::parameter_path(prefix, "bias"),
            self.bias.as_mut(),
        );
    }
}
