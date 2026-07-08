use crate::backend::{Backend, Cpu};
use crate::dtype::FloatDType;
use crate::error::Result;
use crate::nn::context::TrainingMode;
use crate::nn::parameter::Buffer;
use crate::nn::{Layer, Module, Parameter};
use crate::shape::{C, D1, D2, D4, DimSpec};
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

/// RMS normalization — normalizes by root-mean-square, no mean subtraction, no bias.
///
/// Parameter names are part of the persistence contract: `weight` has shape `[FEATURES]`.
pub struct RMSNorm<const FEATURES: usize, E = f32, B = Cpu>
where
    E: FloatDType,
    B: Backend<E>,
{
    weight: Parameter<D1<C<FEATURES>>, E, B>,
    eps: E,
}

impl<const FEATURES: usize, E, B> RMSNorm<FEATURES, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn new() -> Result<Self> {
        Self::with_eps(E::from_f64(1e-6))
    }

    pub fn with_eps(eps: E) -> Result<Self> {
        Ok(Self {
            weight: Parameter::new(Tensor::ones()?),
            eps,
        })
    }
}

impl<const FEATURES: usize, A, E, B> Layer<Tensor<D2<A, C<FEATURES>>, E, B>>
    for RMSNorm<FEATURES, E, B>
where
    A: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    type Output = Tensor<D2<A, C<FEATURES>>, E, B>;
}

impl<const FEATURES: usize, A, E, B, Context> Module<Tensor<D2<A, C<FEATURES>>, E, B>, Context>
    for RMSNorm<FEATURES, E, B>
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
        let sq = input.mul(input)?;
        let mean_sq = sq.mean_last()?;
        let rms = mean_sq.add_scalar(self.eps)?.sqrt()?;
        input
            .div_leading_dim(&rms)?
            .mul_last_dim(self.weight.tensor())
    }
}

crate::nn::has_parameters! {
    impl[const FEATURES: usize, E, B] RMSNorm<FEATURES, E, B>
    where { }
    {
        params { weight }
        children { }
        transparent_children { }
    }
}

/// 2D batch normalization with learnable scale/shift and exponential moving-average statistics.
///
/// Parameter names are part of the persistence contract: `weight` and `bias` have
/// shape `[CH]`. Buffer names: `running_mean` and `running_var` have shape `[CH]`.
pub struct BatchNorm2d<const CH: usize, E = f32, B = Cpu>
where
    E: FloatDType,
    B: Backend<E>,
{
    weight: Parameter<D1<C<CH>>, E, B>,
    bias: Parameter<D1<C<CH>>, E, B>,
    running_mean: Buffer<D1<C<CH>>, E, B>,
    running_var: Buffer<D1<C<CH>>, E, B>,
    eps: f64,
    momentum: f64,
}

impl<const CH: usize, E, B> BatchNorm2d<CH, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn new() -> Result<Self> {
        Self::with_params(1e-5, 0.1)
    }

    pub fn with_params(eps: f64, momentum: f64) -> Result<Self> {
        Ok(Self {
            weight: Parameter::new(Tensor::ones()?),
            bias: Parameter::new(Tensor::zeros()?),
            running_mean: Buffer::new(Tensor::zeros()?),
            running_var: Buffer::new(Tensor::ones()?),
            eps,
            momentum,
        })
    }
}

impl<const CH: usize, A, HH, WW, E, B> Layer<Tensor<D4<A, C<CH>, HH, WW>, E, B>>
    for BatchNorm2d<CH, E, B>
where
    A: DimSpec,
    HH: DimSpec,
    WW: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    type Output = Tensor<D4<A, C<CH>, HH, WW>, E, B>;
}

impl<const CH: usize, A, HH, WW, E, B, Context> Module<Tensor<D4<A, C<CH>, HH, WW>, E, B>, Context>
    for BatchNorm2d<CH, E, B>
where
    A: DimSpec,
    HH: DimSpec,
    WW: DimSpec,
    E: FloatDType,
    B: Backend<E>,
    Context: TrainingMode,
{
    fn forward(
        &self,
        input: &Tensor<D4<A, C<CH>, HH, WW>, E, B>,
        ctx: &mut Context,
    ) -> Result<Self::Output> {
        if ctx.is_training() {
            let (out, batch_mean, batch_var) = input.batch_norm2d(
                self.weight.tensor(),
                self.bias.tensor(),
                self.eps,
                self.momentum,
            )?;
            let old_mean = self.running_mean.tensor().to_vec()?;
            let old_var = self.running_var.tensor().to_vec()?;
            let updated_mean: Vec<E> = old_mean
                .iter()
                .zip(&batch_mean)
                .map(|(&old, &new_val)| {
                    E::from_f64(
                        (1.0 - self.momentum) * old.to_f64() + self.momentum * new_val.to_f64(),
                    )
                })
                .collect();
            let updated_var: Vec<E> = old_var
                .iter()
                .zip(&batch_var)
                .map(|(&old, &new_val)| {
                    E::from_f64(
                        (1.0 - self.momentum) * old.to_f64() + self.momentum * new_val.to_f64(),
                    )
                })
                .collect();
            self.running_mean.as_ref().set_data(updated_mean)?;
            self.running_var.as_ref().set_data(updated_var)?;
            Ok(out)
        } else {
            let mean_vals = self.running_mean.tensor().to_vec()?;
            let var_vals = self.running_var.tensor().to_vec()?;
            input.batch_norm2d_eval(
                self.weight.tensor(),
                self.bias.tensor(),
                &mean_vals,
                &var_vals,
                self.eps,
            )
        }
    }
}

crate::nn::has_parameters! {
    impl[const CH: usize, E, B] BatchNorm2d<CH, E, B>
    where { }
    {
        params { weight, bias }
        children { }
        transparent_children { }
        buffers { running_mean, running_var }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::AnyDim;

    #[test]
    fn rmsnorm_identity_weight_scales_by_rms() {
        let norm = RMSNorm::<4>::new().unwrap();
        let input: Tensor<D2<AnyDim, C<4>>, f32> =
            Tensor::from_vec_with_shape(vec![3.0f32, 4.0, 0.0, 0.0], [1, 4]).unwrap();
        let out = norm.forward(&input, &mut ()).unwrap();
        let vals = out.to_vec().unwrap();
        // rms = sqrt((9+16)/4 + 1e-6) ≈ sqrt(6.25) = 2.5
        assert!((vals[0] - 1.2_f32).abs() < 1e-4, "got {}", vals[0]);
        assert!((vals[1] - 1.6_f32).abs() < 1e-4, "got {}", vals[1]);
    }

    #[test]
    fn batchnorm2d_train_normalizes_single_element() {
        use crate::nn::context::TrainContext;
        let bn = BatchNorm2d::<2>::new().unwrap();
        // 1 sample, 2 channels, 1x1 spatial — with 1 element per channel mean=x, var=0
        let input: Tensor<D4<AnyDim, C<2>, AnyDim, AnyDim>, f32> =
            Tensor::from_vec_with_shape(vec![2.0f32, 4.0], [1, 2, 1, 1]).unwrap();
        let mut ctx = TrainContext::training(0);
        let out = bn.forward(&input, &mut ctx).unwrap();
        let vals = out.to_vec().unwrap();
        assert!(vals[0].abs() < 1e-4, "got {}", vals[0]);
        assert!(vals[1].abs() < 1e-4, "got {}", vals[1]);
    }
}
