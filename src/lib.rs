#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub mod backend;
pub mod dtype;
pub mod nn;
pub mod optim;
pub mod shape;
pub mod tensor;

pub mod prelude {
    pub use crate::backend::Cpu;
    pub use crate::dtype::FloatElement;
    pub use crate::shape::{D0, D1, D2, Shape};
    pub use crate::tensor::{Scalar, Tensor, Tensor1D, Tensor2D, TensorError};
}
