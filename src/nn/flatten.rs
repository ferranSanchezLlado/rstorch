use super::parameter::{HasParameters, Layer, Module, ParameterRef, ParameterRefMut};
use crate::backend::Backend;
use crate::dtype::FloatDType;
use crate::error::Result;
use crate::shape::{C, D2, D4, DimSpec};
use crate::tensor::Tensor;

#[derive(Debug, Clone, Copy, Default)]
pub struct Flatten<const OUT: usize>;

impl<Batch, Channels, Height, Width, const OUT: usize, E, B>
    Layer<Tensor<D4<Batch, Channels, Height, Width>, E, B>> for Flatten<OUT>
where
    Batch: DimSpec,
    Channels: DimSpec,
    Height: DimSpec,
    Width: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    type Output = Tensor<D2<Batch, C<OUT>>, E, B>;
}

impl<Batch, Channels, Height, Width, const OUT: usize, E, B, Context>
    Module<Tensor<D4<Batch, Channels, Height, Width>, E, B>, Context> for Flatten<OUT>
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
        input.flatten_spatial::<OUT>()
    }
}

impl<const OUT: usize, E, B> HasParameters<E, B> for Flatten<OUT>
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
