mod cpu;
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
mod cuda;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;
#[cfg(feature = "wgpu")]
mod wgpu;

use crate::dtype::DType;

pub use cpu::{Cpu, CpuDevice, CpuError};
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
pub use cuda::{Cuda, CudaDevice, CudaError};
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use metal::{Metal, MetalDevice, MetalError};
#[cfg(feature = "wgpu")]
pub use wgpu::{Wgpu, WgpuDevice, WgpuError};

pub trait Backend<E: DType>: Clone + Send + Sync + 'static {
    type Device: Clone + Send + Sync + PartialEq + std::fmt::Debug + 'static;
    type Storage: Clone + Send + Sync + 'static;
    type Error: std::error::Error + Send + Sync + 'static;

    fn default_device() -> std::result::Result<Self::Device, Self::Error>;
    fn zeros(device: &Self::Device, len: usize) -> std::result::Result<Self::Storage, Self::Error>;
    fn ones(device: &Self::Device, len: usize) -> std::result::Result<Self::Storage, Self::Error>;
    fn from_vec(
        device: &Self::Device,
        data: Vec<E>,
    ) -> std::result::Result<Self::Storage, Self::Error>;
    fn to_vec(
        device: &Self::Device,
        storage: &Self::Storage,
    ) -> std::result::Result<Vec<E>, Self::Error>;
    fn storage_len(storage: &Self::Storage) -> usize;

    fn matmul(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        m: usize,
        k: usize,
        n: usize,
    ) -> std::result::Result<Self::Storage, Self::Error>;

    fn add(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error>;

    fn sub(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error>;

    fn mul(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error>;

    fn div(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error>;

    fn add_scalar(
        device: &Self::Device,
        input: &Self::Storage,
        rhs: E,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error>;

    fn sub_scalar(
        device: &Self::Device,
        input: &Self::Storage,
        rhs: E,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error>;

    fn mul_scalar(
        device: &Self::Device,
        input: &Self::Storage,
        rhs: E,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error>;

    fn div_scalar(
        device: &Self::Device,
        input: &Self::Storage,
        rhs: E,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error>;

    fn sum(
        device: &Self::Device,
        input: &Self::Storage,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error>;
}
