//! Safe owned CPU backend foundation.

use crate::backend::Backend;
use crate::dtype::FloatElement;

/// CPU backend marker.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Cpu;

/// CPU device marker.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CpuDevice;

/// CPU storage owns tensor elements.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuStorage<E> {
    data: Vec<E>,
}

impl<E: FloatElement> Backend<E> for Cpu {
    type Device = CpuDevice;
    type Storage = CpuStorage<E>;

    fn default_device() -> Self::Device {
        CpuDevice
    }

    fn zeros<const N: usize>(_device: &Self::Device) -> Self::Storage
    where
        [(); N]:,
    {
        CpuStorage {
            data: vec![E::zero(); N],
        }
    }

    fn ones<const N: usize>(_device: &Self::Device) -> Self::Storage
    where
        [(); N]:,
    {
        CpuStorage {
            data: vec![E::one(); N],
        }
    }

    fn from_array<const N: usize>(_device: &Self::Device, data: [E; N]) -> Self::Storage
    where
        [(); N]:,
    {
        CpuStorage {
            data: Vec::from(data),
        }
    }

    fn from_vec(_device: &Self::Device, data: Vec<E>) -> Self::Storage {
        CpuStorage { data }
    }

    fn to_vec(storage: &Self::Storage) -> Vec<E> {
        storage.data.clone()
    }
}
