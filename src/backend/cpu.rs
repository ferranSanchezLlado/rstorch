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
    _data: Vec<E>,
}

impl<E: FloatElement> Backend<E> for Cpu {
    type Device = CpuDevice;
    type Storage = CpuStorage<E>;

    fn default_device() -> Self::Device {
        CpuDevice
    }
}
