//! Backend abstractions and backend-specific placeholders.

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "metal")]
pub mod metal;
#[cfg(feature = "wgpu")]
pub mod wgpu;

pub use cpu::{Cpu, CpuDevice, CpuStorage};

use crate::dtype::FloatElement;

/// Storage backend for tensor data.
pub trait Backend<E: FloatElement>: Clone + Send + Sync + 'static {
    type Device: Clone + Send + Sync + 'static;
    type Storage: Clone + Send + Sync + 'static;

    fn default_device() -> Self::Device;
}
