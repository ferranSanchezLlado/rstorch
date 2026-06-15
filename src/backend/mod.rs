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

    fn zeros<const N: usize>(device: &Self::Device) -> Self::Storage
    where
        [(); N]:;

    fn ones<const N: usize>(device: &Self::Device) -> Self::Storage
    where
        [(); N]:;

    fn from_array<const N: usize>(device: &Self::Device, data: [E; N]) -> Self::Storage
    where
        [(); N]:;

    fn from_vec(device: &Self::Device, data: Vec<E>) -> Self::Storage;
    fn to_vec(storage: &Self::Storage) -> Vec<E>;
}
