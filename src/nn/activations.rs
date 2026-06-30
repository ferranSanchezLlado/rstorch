use crate::backend::Backend;
use crate::dtype::FloatDType;
use crate::error::Result;
use crate::nn::{HasParameters, Layer, Module, ParameterRef, ParameterRefMut};
use crate::shape::ShapeSpec;
use crate::tensor::Tensor;

pub fn relu<S, E, B>(input: &Tensor<S, E, B>) -> Result<Tensor<S, E, B>>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    input.relu()
}

pub fn sigmoid<S, E, B>(input: &Tensor<S, E, B>) -> Result<Tensor<S, E, B>>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    input.sigmoid()
}

pub fn tanh<S, E, B>(input: &Tensor<S, E, B>) -> Result<Tensor<S, E, B>>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    input.tanh()
}

pub fn gelu<S, E, B>(input: &Tensor<S, E, B>) -> Result<Tensor<S, E, B>>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    input.gelu()
}

/// Defines a stateless activation `Module` that forwards to a tensor method.
///
/// These shape-preserving units carry no parameters, so they can be dropped
/// into a [`Sequential`](crate::nn::Sequential) stack between layers.
macro_rules! activation_module {
    ($($(#[$meta:meta])* $name:ident => $method:ident);+ $(;)?) => {
        $(activation_module!(@single $(#[$meta])* $name => $method);)+
    };

    (@single $(#[$meta:meta])* $name:ident => $method:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $name;

        impl<S, E, B> Layer<Tensor<S, E, B>> for $name
        where
            S: ShapeSpec,
            E: FloatDType,
            B: Backend<E>,
        {
            type Output = Tensor<S, E, B>;
        }

        impl<S, E, B, Ctx> Module<Tensor<S, E, B>, Ctx> for $name
        where
            S: ShapeSpec,
            E: FloatDType,
            B: Backend<E>,
        {

            fn forward(&self, input: &Tensor<S, E, B>, _ctx: &mut Ctx) -> Result<Self::Output> {
                input.$method()
            }
        }

        impl<E, B> HasParameters<E, B> for $name
        where
            E: FloatDType,
            B: Backend<E>,
        {
            fn parameters<'a>(&'a self, _out: &mut Vec<ParameterRef<'a, E, B>>) {}

            fn parameters_mut<'a>(&'a mut self, _out: &mut Vec<ParameterRefMut<'a, E, B>>) {}
        }
    };
}

activation_module!(
    /// Rectified linear unit module form of [`relu`].
    Relu => relu;
    /// Logistic sigmoid module form of [`sigmoid`].
    Sigmoid => sigmoid;
    /// Hyperbolic tangent module form of [`tanh`].
    Tanh => tanh;
    /// Gaussian error linear unit module form of [`gelu`].
    Gelu => gelu;
);
