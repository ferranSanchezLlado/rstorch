use super::parameter::{Layer, Module};
use crate::backend::Backend;
use crate::dtype::FloatDType;
use crate::error::Result;
use crate::shape::{C, D4, DimSpec};
use crate::tensor::{Pool2dOptions, Tensor};

#[derive(Debug, Clone, Copy)]
pub struct MaxPool2d<const K_H: usize, const K_W: usize, const OUT_H: usize, const OUT_W: usize> {
    options: Pool2dOptions,
}

impl<const K_H: usize, const K_W: usize, const OUT_H: usize, const OUT_W: usize>
    MaxPool2d<K_H, K_W, OUT_H, OUT_W>
{
    pub fn new() -> Self {
        Self {
            options: Pool2dOptions::new(K_H, K_W),
        }
    }

    pub const fn with_stride_padding(
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
    ) -> Self {
        Self {
            options: Pool2dOptions::with_stride_padding(
                K_H, K_W, stride_h, stride_w, padding_h, padding_w,
            ),
        }
    }

    pub fn options(&self) -> Pool2dOptions {
        self.options
    }
}

impl<const K_H: usize, const K_W: usize, const OUT_H: usize, const OUT_W: usize> Default
    for MaxPool2d<K_H, K_W, OUT_H, OUT_W>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
    Batch,
    Channels,
    Height,
    Width,
    const K_H: usize,
    const K_W: usize,
    const OUT_H: usize,
    const OUT_W: usize,
    E,
    B,
> Layer<Tensor<D4<Batch, Channels, Height, Width>, E, B>> for MaxPool2d<K_H, K_W, OUT_H, OUT_W>
where
    Batch: DimSpec,
    Channels: DimSpec,
    Height: DimSpec,
    Width: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    type Output = Tensor<D4<Batch, Channels, C<OUT_H>, C<OUT_W>>, E, B>;
}

impl<
    Batch,
    Channels,
    Height,
    Width,
    const K_H: usize,
    const K_W: usize,
    const OUT_H: usize,
    const OUT_W: usize,
    E,
    B,
    Context,
> Module<Tensor<D4<Batch, Channels, Height, Width>, E, B>, Context>
    for MaxPool2d<K_H, K_W, OUT_H, OUT_W>
where
    Batch: DimSpec,
    Channels: DimSpec,
    Height: DimSpec,
    Width: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    fn forward(
        &self,
        input: &Tensor<D4<Batch, Channels, Height, Width>, E, B>,
        _ctx: &mut Context,
    ) -> Result<Self::Output> {
        input.max_pool2d::<OUT_H, OUT_W>(self.options)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct AvgPool2d<const K_H: usize, const K_W: usize, const OUT_H: usize, const OUT_W: usize> {
    options: Pool2dOptions,
}

impl<const K_H: usize, const K_W: usize, const OUT_H: usize, const OUT_W: usize>
    AvgPool2d<K_H, K_W, OUT_H, OUT_W>
{
    pub fn new() -> Self {
        Self {
            options: Pool2dOptions::new(K_H, K_W),
        }
    }

    pub const fn with_stride_padding(
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
    ) -> Self {
        Self {
            options: Pool2dOptions::with_stride_padding(
                K_H, K_W, stride_h, stride_w, padding_h, padding_w,
            ),
        }
    }

    pub fn options(&self) -> Pool2dOptions {
        self.options
    }
}

impl<const K_H: usize, const K_W: usize, const OUT_H: usize, const OUT_W: usize> Default
    for AvgPool2d<K_H, K_W, OUT_H, OUT_W>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<
    Batch,
    Channels,
    Height,
    Width,
    const K_H: usize,
    const K_W: usize,
    const OUT_H: usize,
    const OUT_W: usize,
    E,
    B,
> Layer<Tensor<D4<Batch, Channels, Height, Width>, E, B>> for AvgPool2d<K_H, K_W, OUT_H, OUT_W>
where
    Batch: DimSpec,
    Channels: DimSpec,
    Height: DimSpec,
    Width: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    type Output = Tensor<D4<Batch, Channels, C<OUT_H>, C<OUT_W>>, E, B>;
}

impl<
    Batch,
    Channels,
    Height,
    Width,
    const K_H: usize,
    const K_W: usize,
    const OUT_H: usize,
    const OUT_W: usize,
    E,
    B,
    Context,
> Module<Tensor<D4<Batch, Channels, Height, Width>, E, B>, Context>
    for AvgPool2d<K_H, K_W, OUT_H, OUT_W>
where
    Batch: DimSpec,
    Channels: DimSpec,
    Height: DimSpec,
    Width: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    fn forward(
        &self,
        input: &Tensor<D4<Batch, Channels, Height, Width>, E, B>,
        _ctx: &mut Context,
    ) -> Result<Self::Output> {
        input.avg_pool2d::<OUT_H, OUT_W>(self.options)
    }
}

macro_rules! empty_parameters {
    ($name:ident<$($const_name:ident),+>) => {
        crate::nn::has_parameters! {
            impl[$(const $const_name: usize,)+ E, B] $name<$($const_name,)+>
            where { }
            {
                params { }
                children { }
                transparent_children { }
            }
        }
    };
}

empty_parameters!(MaxPool2d<K_H, K_W, OUT_H, OUT_W>);
empty_parameters!(AvgPool2d<K_H, K_W, OUT_H, OUT_W>);
