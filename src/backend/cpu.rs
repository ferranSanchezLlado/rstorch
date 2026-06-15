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

    fn add<const N: usize>(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
    ) -> Self::Storage
    where
        [(); N]:,
    {
        elementwise::<E, N, _>(device, lhs, rhs, |lhs, rhs| lhs + rhs)
    }

    fn sub<const N: usize>(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
    ) -> Self::Storage
    where
        [(); N]:,
    {
        elementwise::<E, N, _>(device, lhs, rhs, |lhs, rhs| lhs - rhs)
    }

    fn mul<const N: usize>(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
    ) -> Self::Storage
    where
        [(); N]:,
    {
        elementwise::<E, N, _>(device, lhs, rhs, |lhs, rhs| lhs * rhs)
    }

    fn div<const N: usize>(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
    ) -> Self::Storage
    where
        [(); N]:,
    {
        elementwise::<E, N, _>(device, lhs, rhs, |lhs, rhs| lhs / rhs)
    }

    fn add_scalar<const N: usize>(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: E,
    ) -> Self::Storage
    where
        [(); N]:,
    {
        unary::<E, N, _>(device, lhs, |lhs| lhs + rhs)
    }

    fn sub_scalar<const N: usize>(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: E,
    ) -> Self::Storage
    where
        [(); N]:,
    {
        unary::<E, N, _>(device, lhs, |lhs| lhs - rhs)
    }

    fn mul_scalar<const N: usize>(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: E,
    ) -> Self::Storage
    where
        [(); N]:,
    {
        unary::<E, N, _>(device, lhs, |lhs| lhs * rhs)
    }

    fn div_scalar<const N: usize>(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: E,
    ) -> Self::Storage
    where
        [(); N]:,
    {
        unary::<E, N, _>(device, lhs, |lhs| lhs / rhs)
    }

    fn powf<const N: usize>(
        device: &Self::Device,
        lhs: &Self::Storage,
        exponent: E,
    ) -> Self::Storage
    where
        [(); N]:,
    {
        unary::<E, N, _>(device, lhs, |lhs| lhs.powf(exponent))
    }

    fn relu<const N: usize>(device: &Self::Device, input: &Self::Storage) -> Self::Storage
    where
        [(); N]:,
    {
        unary::<E, N, _>(device, input, |value| {
            if value > E::zero() { value } else { E::zero() }
        })
    }

    fn exp<const N: usize>(device: &Self::Device, input: &Self::Storage) -> Self::Storage
    where
        [(); N]:,
    {
        unary::<E, N, _>(device, input, E::exp)
    }

    fn ln<const N: usize>(device: &Self::Device, input: &Self::Storage) -> Self::Storage
    where
        [(); N]:,
    {
        unary::<E, N, _>(device, input, E::ln)
    }

    fn sum<const N: usize>(device: &Self::Device, input: &Self::Storage) -> Self::Storage
    where
        [(); N]:,
    {
        let value = input
            .data
            .iter()
            .copied()
            .fold(E::zero(), |total, value| total + value);
        Self::from_array(device, [value])
    }

    fn mean<const N: usize>(device: &Self::Device, input: &Self::Storage) -> Self::Storage
    where
        [(); N]:,
    {
        let total = input
            .data
            .iter()
            .copied()
            .fold(E::zero(), |total, value| total + value);
        Self::from_array(device, [total / E::from_usize(N)])
    }

    fn matmul<const M: usize, const K: usize, const N: usize>(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
    ) -> Self::Storage
    where
        [(); M * K]:,
        [(); K * N]:,
        [(); M * N]:,
    {
        let mut data = vec![E::zero(); M * N];

        for row in 0..M {
            for col in 0..N {
                let mut total = E::zero();
                for inner in 0..K {
                    total = total + lhs.data[row * K + inner] * rhs.data[inner * N + col];
                }
                data[row * N + col] = total;
            }
        }

        Self::from_vec(device, data)
    }

    fn transpose<const M: usize, const N: usize>(
        device: &Self::Device,
        input: &Self::Storage,
    ) -> Self::Storage
    where
        [(); M * N]:,
        [(); N * M]:,
    {
        let mut data = vec![E::zero(); M * N];

        for row in 0..M {
            for col in 0..N {
                data[col * M + row] = input.data[row * N + col];
            }
        }

        Self::from_vec(device, data)
    }

    fn add_row<const M: usize, const N: usize>(
        device: &Self::Device,
        input: &Self::Storage,
        row: &Self::Storage,
    ) -> Self::Storage
    where
        [(); M * N]:,
        [(); N]:,
    {
        let mut data = Vec::with_capacity(M * N);

        for matrix_row in 0..M {
            for col in 0..N {
                data.push(input.data[matrix_row * N + col] + row.data[col]);
            }
        }

        Self::from_vec(device, data)
    }

    fn add_col<const M: usize, const N: usize>(
        device: &Self::Device,
        input: &Self::Storage,
        col: &Self::Storage,
    ) -> Self::Storage
    where
        [(); M * N]:,
        [(); M]:,
    {
        let mut data = Vec::with_capacity(M * N);

        for row in 0..M {
            for matrix_col in 0..N {
                data.push(input.data[row * N + matrix_col] + col.data[row]);
            }
        }

        Self::from_vec(device, data)
    }
}

fn elementwise<E, const N: usize, F>(
    device: &CpuDevice,
    lhs: &CpuStorage<E>,
    rhs: &CpuStorage<E>,
    op: F,
) -> CpuStorage<E>
where
    E: FloatElement,
    F: Fn(E, E) -> E,
    [(); N]:,
{
    let data = lhs
        .data
        .iter()
        .zip(rhs.data.iter())
        .map(|(lhs, rhs)| op(*lhs, *rhs))
        .collect();
    Cpu::from_vec(device, data)
}

fn unary<E, const N: usize, F>(device: &CpuDevice, input: &CpuStorage<E>, op: F) -> CpuStorage<E>
where
    E: FloatElement,
    F: Fn(E) -> E,
    [(); N]:,
{
    let data = input.data.iter().copied().map(op).collect();
    Cpu::from_vec(device, data)
}
