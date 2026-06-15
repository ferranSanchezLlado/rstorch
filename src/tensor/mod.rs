//! Tensor type declarations for the restart architecture.

pub mod autograd;
pub mod ops;

use crate::backend::{Backend, Cpu};
use crate::dtype::FloatElement;
use crate::shape::{D0, D1, D2, Shape};
use std::marker::PhantomData;

/// Generic tensor placeholder. Epoch 01 adds owned storage and constructors.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Tensor<S, E = f32, B = Cpu>
where
    S: Shape,
    E: FloatElement,
    B: Backend<E>,
{
    _shape: PhantomData<S>,
    _element: PhantomData<E>,
    _backend: PhantomData<B>,
}

/// Scalar tensor alias.
pub type Scalar<E = f32, B = Cpu> = Tensor<D0, E, B>;

/// One-dimensional tensor alias.
pub type Tensor1D<const N: usize, E = f32, B = Cpu> = Tensor<D1<N>, E, B>;

/// Two-dimensional tensor alias.
pub type Tensor2D<const M: usize, const N: usize, E = f32, B = Cpu> = Tensor<D2<M, N>, E, B>;
