pub mod backend;
pub mod dtype;
pub mod error;
pub mod shape;
pub mod tensor;

pub use backend::{Backend, Cpu, CpuDevice, CpuError};
pub use dtype::{DType, DTypeId};
pub use error::{DTypeError, DeviceError, Error, Result, ShapeError};
pub use shape::{
    AnyDim, C, D0, D1, D2, D3, D4, DimSpec, Layout, Shape, ShapeSpec, StaticShape, Sym,
};
pub use tensor::{Scalar, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D};

pub mod prelude {
    pub use crate::backend::{Backend, Cpu};
    pub use crate::dtype::DType;
    pub use crate::error::Result;
    pub use crate::shape::{AnyDim, C, D0, D1, D2, D3, D4, DimSpec, ShapeSpec, StaticShape, Sym};
    pub use crate::tensor::{Scalar, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D};
}
