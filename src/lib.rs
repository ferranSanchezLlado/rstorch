#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub mod backend;
pub mod dtype;
pub mod nn;
pub mod optim;
pub mod rng;
pub mod shape;
pub mod tensor;

pub mod prelude {
    pub use crate::backend::Cpu;
    #[cfg(feature = "cuda")]
    pub use crate::backend::Cuda;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub use crate::backend::Metal;
    pub use crate::dtype::FloatElement;
    pub use crate::nn::{Linear, Module, Parameter};
    pub use crate::optim::{OptimParameter, SGD};
    pub use crate::rng::SmallRng;
    pub use crate::shape::{D0, D1, D2, Shape};
    pub use crate::tensor::autograd::{NoGradGuard, is_grad_enabled, no_grad, with_no_grad};
    pub use crate::tensor::{Scalar, Tensor, Tensor1D, Tensor2D, TensorError};
}
