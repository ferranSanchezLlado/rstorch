use crate::backend::Backend;
use crate::dtype::FloatDType;
use crate::nn::{HasParameters, Layer, Module, ParameterRef, ParameterRefMut};
use crate::shape::ShapeSpec;
use crate::tensor::Tensor;

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

        impl<S, E, B, Context> Module<Tensor<S, E, B>, Context> for $name
        where
            S: ShapeSpec,
            E: FloatDType,
            B: Backend<E>,
        {

            fn forward(&self, input: &Tensor<S, E, B>, _ctx: &mut Context) -> crate::error::Result<Self::Output> {
                input.$method()
            }
        }

        impl<E, B> HasParameters<E, B> for $name
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
    };
}

activation_module!(
    /// Rectified linear unit module form of [`Tensor::relu`](crate::tensor::Tensor::relu).
    Relu => relu;
    /// Logistic sigmoid module form of [`Tensor::sigmoid`](crate::tensor::Tensor::sigmoid).
    Sigmoid => sigmoid;
    /// Hyperbolic tangent module form of [`Tensor::tanh`](crate::tensor::Tensor::tanh).
    Tanh => tanh;
    /// Gaussian error linear unit module form of [`Tensor::gelu`](crate::tensor::Tensor::gelu).
    Gelu => gelu;
);
