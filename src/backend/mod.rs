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

    fn add<const N: usize>(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
    ) -> Self::Storage
    where
        [(); N]:;

    fn sub<const N: usize>(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
    ) -> Self::Storage
    where
        [(); N]:;

    fn mul<const N: usize>(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
    ) -> Self::Storage
    where
        [(); N]:;

    fn div<const N: usize>(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
    ) -> Self::Storage
    where
        [(); N]:;

    fn add_scalar<const N: usize>(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: E,
    ) -> Self::Storage
    where
        [(); N]:;

    fn sub_scalar<const N: usize>(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: E,
    ) -> Self::Storage
    where
        [(); N]:;

    fn mul_scalar<const N: usize>(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: E,
    ) -> Self::Storage
    where
        [(); N]:;

    fn div_scalar<const N: usize>(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: E,
    ) -> Self::Storage
    where
        [(); N]:;

    fn powf<const N: usize>(
        device: &Self::Device,
        lhs: &Self::Storage,
        exponent: E,
    ) -> Self::Storage
    where
        [(); N]:;

    fn relu<const N: usize>(device: &Self::Device, input: &Self::Storage) -> Self::Storage
    where
        [(); N]:;

    fn exp<const N: usize>(device: &Self::Device, input: &Self::Storage) -> Self::Storage
    where
        [(); N]:;

    fn ln<const N: usize>(device: &Self::Device, input: &Self::Storage) -> Self::Storage
    where
        [(); N]:;

    fn sum<const N: usize>(device: &Self::Device, input: &Self::Storage) -> Self::Storage
    where
        [(); N]:;

    fn mean<const N: usize>(device: &Self::Device, input: &Self::Storage) -> Self::Storage
    where
        [(); N]:;

    fn matmul<const M: usize, const K: usize, const N: usize>(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
    ) -> Self::Storage
    where
        [(); M * K]:,
        [(); K * N]:,
        [(); M * N]:;

    fn transpose<const M: usize, const N: usize>(
        device: &Self::Device,
        input: &Self::Storage,
    ) -> Self::Storage
    where
        [(); M * N]:,
        [(); N * M]:;

    fn add_row<const M: usize, const N: usize>(
        device: &Self::Device,
        input: &Self::Storage,
        row: &Self::Storage,
    ) -> Self::Storage
    where
        [(); M * N]:,
        [(); N]:;

    fn add_col<const M: usize, const N: usize>(
        device: &Self::Device,
        input: &Self::Storage,
        col: &Self::Storage,
    ) -> Self::Storage
    where
        [(); M * N]:,
        [(); M]:;
}
