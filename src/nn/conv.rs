use super::parameter::{HasParameters, Layer, Module, Parameter, ParameterRef, ParameterRefMut};
use crate::backend::{Backend, Cpu};
use crate::dtype::FloatDType;
use crate::error::{Result, ShapeError};
use crate::random::SmallRng;
use crate::shape::{C, D1, D4, DimSpec};
use crate::tensor::{Conv2dOptions, Tensor};

pub struct Conv2d<
    const IN_CH: usize,
    const OUT_CH: usize,
    const K_H: usize,
    const K_W: usize,
    const OUT_H: usize,
    const OUT_W: usize,
    E = f32,
    B = Cpu,
> where
    E: FloatDType,
    B: Backend<E>,
{
    weight: Parameter<D4<C<OUT_CH>, C<IN_CH>, C<K_H>, C<K_W>>, E, B>,
    bias: Option<Parameter<D1<C<OUT_CH>>, E, B>>,
    options: Conv2dOptions,
}

impl<
    const IN_CH: usize,
    const OUT_CH: usize,
    const K_H: usize,
    const K_W: usize,
    const OUT_H: usize,
    const OUT_W: usize,
    E,
    B,
> Conv2d<IN_CH, OUT_CH, K_H, K_W, OUT_H, OUT_W, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn zeros(options: Conv2dOptions) -> Result<Self> {
        Self::zeros_with_bias(options, true)
    }

    pub fn zeros_without_bias(options: Conv2dOptions) -> Result<Self> {
        Self::zeros_with_bias(options, false)
    }

    fn zeros_with_bias(options: Conv2dOptions, bias: bool) -> Result<Self> {
        Ok(Self {
            weight: Parameter::new(Tensor::zeros()?),
            bias: bias
                .then(|| Tensor::zeros().map(Parameter::new))
                .transpose()?,
            options,
        })
    }

    pub fn kaiming_uniform(rng: &mut SmallRng, options: Conv2dOptions) -> Result<Self> {
        Self::uniform(rng, options, true)
    }

    pub fn kaiming_uniform_without_bias(
        rng: &mut SmallRng,
        options: Conv2dOptions,
    ) -> Result<Self> {
        Self::uniform(rng, options, false)
    }

    fn uniform(rng: &mut SmallRng, options: Conv2dOptions, bias: bool) -> Result<Self> {
        let fan_in = fan_in::<IN_CH, K_H, K_W>("conv2d_kaiming_uniform")?;
        let limit = (6.0 / (fan_in as f64)).sqrt();
        let weight_len = OUT_CH
            .checked_mul(fan_in)
            .ok_or(ShapeError::InvalidSpatialParam {
                op: "conv2d_kaiming_uniform",
                param: "weight_len",
                value: OUT_CH,
                reason: "weight element count overflow",
            })?;
        let weight = (0..weight_len)
            .map(|_| rng.uniform(-E::from_f64(limit), E::from_f64(limit)))
            .collect();
        Ok(Self {
            weight: Parameter::new(Tensor::from_vec(weight)?),
            bias: bias
                .then(|| Tensor::zeros().map(Parameter::new))
                .transpose()?,
            options,
        })
    }

    pub fn weight(&self) -> &Parameter<D4<C<OUT_CH>, C<IN_CH>, C<K_H>, C<K_W>>, E, B> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Parameter<D1<C<OUT_CH>>, E, B>> {
        self.bias.as_ref()
    }

    pub fn options(&self) -> Conv2dOptions {
        self.options
    }
}

impl<
    Batch,
    Height,
    Width,
    const IN_CH: usize,
    const OUT_CH: usize,
    const K_H: usize,
    const K_W: usize,
    const OUT_H: usize,
    const OUT_W: usize,
    E,
    B,
> Layer<Tensor<D4<Batch, C<IN_CH>, Height, Width>, E, B>>
    for Conv2d<IN_CH, OUT_CH, K_H, K_W, OUT_H, OUT_W, E, B>
where
    Batch: DimSpec,
    Height: DimSpec,
    Width: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    type Output = Tensor<D4<Batch, C<OUT_CH>, C<OUT_H>, C<OUT_W>>, E, B>;
}

impl<
    Batch,
    Height,
    Width,
    const IN_CH: usize,
    const OUT_CH: usize,
    const K_H: usize,
    const K_W: usize,
    const OUT_H: usize,
    const OUT_W: usize,
    E,
    B,
    Context,
> Module<Tensor<D4<Batch, C<IN_CH>, Height, Width>, E, B>, Context>
    for Conv2d<IN_CH, OUT_CH, K_H, K_W, OUT_H, OUT_W, E, B>
where
    Batch: DimSpec,
    Height: DimSpec,
    Width: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    fn forward(
        &self,
        input: &Tensor<D4<Batch, C<IN_CH>, Height, Width>, E, B>,
        _ctx: &mut Context,
    ) -> Result<Self::Output> {
        let out = input.conv2d::<C<OUT_CH>, C<K_H>, C<K_W>, OUT_H, OUT_W>(
            self.weight.tensor(),
            self.options,
        )?;
        match &self.bias {
            Some(bias) => out.add_channel_dim(bias.tensor()),
            None => Ok(out),
        }
    }
}

impl<
    const IN_CH: usize,
    const OUT_CH: usize,
    const K_H: usize,
    const K_W: usize,
    const OUT_H: usize,
    const OUT_W: usize,
    E,
    B,
> HasParameters<E, B> for Conv2d<IN_CH, OUT_CH, K_H, K_W, OUT_H, OUT_W, E, B>
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
        if let Some(bias) = &self.bias {
            visit(&crate::nn::parameter_path(prefix, "bias"), bias.as_ref());
        }
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
        if let Some(bias) = &mut self.bias {
            visit(&crate::nn::parameter_path(prefix, "bias"), bias.as_mut());
        }
    }
}

fn fan_in<const IN_CH: usize, const K_H: usize, const K_W: usize>(
    op: &'static str,
) -> Result<usize> {
    if IN_CH == 0 {
        return Err(ShapeError::InvalidSpatialParam {
            op,
            param: "in_channels",
            value: IN_CH,
            reason: "fan-in must be greater than 0",
        }
        .into());
    }
    if K_H == 0 {
        return Err(ShapeError::InvalidSpatialParam {
            op,
            param: "kernel_h",
            value: K_H,
            reason: "kernel size must be greater than 0",
        }
        .into());
    }
    if K_W == 0 {
        return Err(ShapeError::InvalidSpatialParam {
            op,
            param: "kernel_w",
            value: K_W,
            reason: "kernel size must be greater than 0",
        }
        .into());
    }
    IN_CH
        .checked_mul(K_H)
        .and_then(|value| value.checked_mul(K_W))
        .ok_or_else(|| {
            ShapeError::InvalidSpatialParam {
                op,
                param: "fan_in",
                value: IN_CH,
                reason: "fan-in overflow",
            }
            .into()
        })
}
