//! The compile-time checked tensor surface for downstream users.

pub use super::{
    Cpu, DYN, DeviceCtx, FloatElement, IndexElement, NumericElement, Placement, Tensor0, Tensor1,
    Tensor2, Tensor3, Tensor4, Tensor5, Tensor6, Tensor7, Tensor8, TypedGradsExt, TypedTensor,
};

#[cfg(all(feature = "metal", target_os = "macos"))]
pub use super::Metal;

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
pub use super::Cuda;

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
pub use super::Wgpu;
