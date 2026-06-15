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

    fn zeros(_device: &Self::Device, len: usize) -> Self::Storage {
        CpuStorage {
            data: vec![E::zero(); len],
        }
    }

    fn ones(_device: &Self::Device, len: usize) -> Self::Storage {
        CpuStorage {
            data: vec![E::one(); len],
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

    fn add(device: &Self::Device, lhs: &Self::Storage, rhs: &Self::Storage) -> Self::Storage {
        elementwise::<E, _>(device, lhs, rhs, |lhs, rhs| lhs + rhs)
    }

    fn sub(device: &Self::Device, lhs: &Self::Storage, rhs: &Self::Storage) -> Self::Storage {
        elementwise::<E, _>(device, lhs, rhs, |lhs, rhs| lhs - rhs)
    }

    fn mul(device: &Self::Device, lhs: &Self::Storage, rhs: &Self::Storage) -> Self::Storage {
        elementwise::<E, _>(device, lhs, rhs, |lhs, rhs| lhs * rhs)
    }

    fn div(device: &Self::Device, lhs: &Self::Storage, rhs: &Self::Storage) -> Self::Storage {
        elementwise::<E, _>(device, lhs, rhs, |lhs, rhs| lhs / rhs)
    }

    fn add_scalar(device: &Self::Device, lhs: &Self::Storage, rhs: E) -> Self::Storage {
        unary::<E, _>(device, lhs, |lhs| lhs + rhs)
    }

    fn sub_scalar(device: &Self::Device, lhs: &Self::Storage, rhs: E) -> Self::Storage {
        unary::<E, _>(device, lhs, |lhs| lhs - rhs)
    }

    fn mul_scalar(device: &Self::Device, lhs: &Self::Storage, rhs: E) -> Self::Storage {
        unary::<E, _>(device, lhs, |lhs| lhs * rhs)
    }

    fn div_scalar(device: &Self::Device, lhs: &Self::Storage, rhs: E) -> Self::Storage {
        unary::<E, _>(device, lhs, |lhs| lhs / rhs)
    }

    fn powf(device: &Self::Device, lhs: &Self::Storage, exponent: E) -> Self::Storage {
        unary::<E, _>(device, lhs, |lhs| lhs.powf(exponent))
    }

    fn relu(device: &Self::Device, input: &Self::Storage) -> Self::Storage {
        unary::<E, _>(device, input, |value| {
            if value > E::zero() { value } else { E::zero() }
        })
    }

    fn exp(device: &Self::Device, input: &Self::Storage) -> Self::Storage {
        unary::<E, _>(device, input, E::exp)
    }

    fn ln(device: &Self::Device, input: &Self::Storage) -> Self::Storage {
        unary::<E, _>(device, input, E::ln)
    }

    fn sum(device: &Self::Device, input: &Self::Storage) -> Self::Storage {
        let value = input
            .data
            .iter()
            .copied()
            .fold(E::zero(), |total, value| total + value);
        Self::from_array(device, [value])
    }

    fn mean(device: &Self::Device, input: &Self::Storage) -> Self::Storage {
        let total = input
            .data
            .iter()
            .copied()
            .fold(E::zero(), |total, value| total + value);
        Self::from_array(device, [total / E::from_usize(input.data.len())])
    }

    fn matmul(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> Self::Storage {
        let mut data = vec![E::zero(); rows * cols];

        for row in 0..rows {
            for col in 0..cols {
                let mut total = E::zero();
                for index in 0..inner {
                    total = total + lhs.data[row * inner + index] * rhs.data[index * cols + col];
                }
                data[row * cols + col] = total;
            }
        }

        Self::from_vec(device, data)
    }

    fn transpose(
        device: &Self::Device,
        input: &Self::Storage,
        rows: usize,
        cols: usize,
    ) -> Self::Storage {
        let mut data = vec![E::zero(); rows * cols];

        for row in 0..rows {
            for col in 0..cols {
                data[col * rows + row] = input.data[row * cols + col];
            }
        }

        Self::from_vec(device, data)
    }

    fn add_row(
        device: &Self::Device,
        input: &Self::Storage,
        row: &Self::Storage,
        rows: usize,
        cols: usize,
    ) -> Self::Storage {
        let mut data = Vec::with_capacity(rows * cols);

        for matrix_row in 0..rows {
            for col in 0..cols {
                data.push(input.data[matrix_row * cols + col] + row.data[col]);
            }
        }

        Self::from_vec(device, data)
    }

    fn add_col(
        device: &Self::Device,
        input: &Self::Storage,
        col: &Self::Storage,
        rows: usize,
        cols: usize,
    ) -> Self::Storage {
        let mut data = Vec::with_capacity(rows * cols);

        for row in 0..rows {
            for matrix_col in 0..cols {
                data.push(input.data[row * cols + matrix_col] + col.data[row]);
            }
        }

        Self::from_vec(device, data)
    }
}

fn elementwise<E, F>(
    device: &CpuDevice,
    lhs: &CpuStorage<E>,
    rhs: &CpuStorage<E>,
    op: F,
) -> CpuStorage<E>
where
    E: FloatElement,
    F: Fn(E, E) -> E,
{
    let data = lhs
        .data
        .iter()
        .zip(rhs.data.iter())
        .map(|(lhs, rhs)| op(*lhs, *rhs))
        .collect();
    Cpu::from_vec(device, data)
}

fn unary<E, F>(device: &CpuDevice, input: &CpuStorage<E>, op: F) -> CpuStorage<E>
where
    E: FloatElement,
    F: Fn(E) -> E,
{
    let data = input.data.iter().copied().map(op).collect();
    Cpu::from_vec(device, data)
}
