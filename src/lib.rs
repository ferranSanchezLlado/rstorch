pub mod backend;
pub mod dtype;
pub mod error;
pub mod nn;
pub mod optim;
pub mod random;
pub mod shape;
pub mod tensor;

pub use backend::{Backend, Cpu, CpuDevice, CpuError};
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use backend::{Metal, MetalDevice, MetalError};
pub use dtype::{DType, DTypeId, FloatDType};
pub use error::{DTypeError, DeviceError, Error, Result, ShapeError};
pub use nn::{HasParameters, Linear, Module, Parameter, ParameterId, mse_loss, relu};
pub use optim::{Adam, Optimizer, Sgd};
pub use random::SmallRng;
pub use shape::{
    AnyDim, C, D0, D1, D2, D3, D4, DimSpec, Layout, Shape, ShapeSpec, StaticShape, Sym,
};
pub use tensor::{
    NoGradGuard, Scalar, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D, is_grad_enabled, no_grad,
};

pub mod prelude {
    pub use crate::backend::{Backend, Cpu};
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub use crate::backend::{Metal, MetalDevice, MetalError};
    pub use crate::dtype::{DType, FloatDType};
    pub use crate::error::Result;
    pub use crate::nn::{HasParameters, Linear, Module, Parameter, mse_loss, relu};
    pub use crate::optim::{Adam, Optimizer, Sgd};
    pub use crate::random::SmallRng;
    pub use crate::shape::{AnyDim, C, D0, D1, D2, D3, D4, DimSpec, ShapeSpec, StaticShape, Sym};
    pub use crate::tensor::{
        Scalar, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D, is_grad_enabled, no_grad,
    };
}
