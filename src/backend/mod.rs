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

    fn zeros(device: &Self::Device, len: usize) -> Self::Storage;

    fn ones(device: &Self::Device, len: usize) -> Self::Storage;

    fn from_array<const N: usize>(device: &Self::Device, data: [E; N]) -> Self::Storage
    where
        [(); N]:;

    fn from_vec(device: &Self::Device, data: Vec<E>) -> Self::Storage;
    fn to_vec(storage: &Self::Storage) -> Vec<E>;

    fn add(device: &Self::Device, lhs: &Self::Storage, rhs: &Self::Storage) -> Self::Storage;

    fn sub(device: &Self::Device, lhs: &Self::Storage, rhs: &Self::Storage) -> Self::Storage;

    fn mul(device: &Self::Device, lhs: &Self::Storage, rhs: &Self::Storage) -> Self::Storage;

    fn div(device: &Self::Device, lhs: &Self::Storage, rhs: &Self::Storage) -> Self::Storage;

    fn add_scalar(device: &Self::Device, lhs: &Self::Storage, rhs: E) -> Self::Storage;

    fn sub_scalar(device: &Self::Device, lhs: &Self::Storage, rhs: E) -> Self::Storage;

    fn mul_scalar(device: &Self::Device, lhs: &Self::Storage, rhs: E) -> Self::Storage;

    fn div_scalar(device: &Self::Device, lhs: &Self::Storage, rhs: E) -> Self::Storage;

    fn powf(device: &Self::Device, lhs: &Self::Storage, exponent: E) -> Self::Storage;

    fn relu(device: &Self::Device, input: &Self::Storage) -> Self::Storage;

    fn exp(device: &Self::Device, input: &Self::Storage) -> Self::Storage;

    fn ln(device: &Self::Device, input: &Self::Storage) -> Self::Storage;

    fn sum(device: &Self::Device, input: &Self::Storage) -> Self::Storage;

    fn mean(device: &Self::Device, input: &Self::Storage) -> Self::Storage;

    fn matmul(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> Self::Storage;

    fn transpose(
        device: &Self::Device,
        input: &Self::Storage,
        rows: usize,
        cols: usize,
    ) -> Self::Storage;

    fn add_row(
        device: &Self::Device,
        input: &Self::Storage,
        row: &Self::Storage,
        rows: usize,
        cols: usize,
    ) -> Self::Storage;

    fn add_col(
        device: &Self::Device,
        input: &Self::Storage,
        col: &Self::Storage,
        rows: usize,
        cols: usize,
    ) -> Self::Storage;
}
