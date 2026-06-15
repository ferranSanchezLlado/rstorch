#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use rstorch::backend::{Backend, CpuDevice, CpuStorage};
use rstorch::prelude::*;

#[derive(Clone, Copy)]
struct OtherBackend;

impl Backend<f32> for OtherBackend {
    type Device = CpuDevice;
    type Storage = CpuStorage<f32>;

    fn default_device() -> Self::Device {
        CpuDevice
    }

    fn zeros(_device: &Self::Device, _len: usize) -> Self::Storage {
        todo!()
    }

    fn ones(_device: &Self::Device, _len: usize) -> Self::Storage {
        todo!()
    }

    fn from_array<const N: usize>(_device: &Self::Device, _data: [f32; N]) -> Self::Storage
    where
        [(); N]:,
    {
        todo!()
    }

    fn from_vec(_device: &Self::Device, _data: Vec<f32>) -> Self::Storage {
        todo!()
    }

    fn to_vec(_storage: &Self::Storage) -> Vec<f32> {
        todo!()
    }

    fn add(
        _device: &Self::Device,
        _lhs: &Self::Storage,
        _rhs: &Self::Storage,
    ) -> Self::Storage {
        todo!()
    }

    fn sub(
        _device: &Self::Device,
        _lhs: &Self::Storage,
        _rhs: &Self::Storage,
    ) -> Self::Storage {
        todo!()
    }

    fn mul(
        _device: &Self::Device,
        _lhs: &Self::Storage,
        _rhs: &Self::Storage,
    ) -> Self::Storage {
        todo!()
    }

    fn div(
        _device: &Self::Device,
        _lhs: &Self::Storage,
        _rhs: &Self::Storage,
    ) -> Self::Storage {
        todo!()
    }

    fn add_scalar(
        _device: &Self::Device,
        _lhs: &Self::Storage,
        _rhs: f32,
    ) -> Self::Storage {
        todo!()
    }

    fn sub_scalar(
        _device: &Self::Device,
        _lhs: &Self::Storage,
        _rhs: f32,
    ) -> Self::Storage {
        todo!()
    }

    fn mul_scalar(
        _device: &Self::Device,
        _lhs: &Self::Storage,
        _rhs: f32,
    ) -> Self::Storage {
        todo!()
    }

    fn div_scalar(
        _device: &Self::Device,
        _lhs: &Self::Storage,
        _rhs: f32,
    ) -> Self::Storage {
        todo!()
    }

    fn powf(
        _device: &Self::Device,
        _lhs: &Self::Storage,
        _exponent: f32,
    ) -> Self::Storage {
        todo!()
    }

    fn relu(_device: &Self::Device, _input: &Self::Storage) -> Self::Storage {
        todo!()
    }

    fn exp(_device: &Self::Device, _input: &Self::Storage) -> Self::Storage {
        todo!()
    }

    fn ln(_device: &Self::Device, _input: &Self::Storage) -> Self::Storage {
        todo!()
    }

    fn sum(_device: &Self::Device, _input: &Self::Storage) -> Self::Storage {
        todo!()
    }

    fn mean(_device: &Self::Device, _input: &Self::Storage) -> Self::Storage {
        todo!()
    }

    fn matmul(
        _device: &Self::Device,
        _lhs: &Self::Storage,
        _rhs: &Self::Storage,
        _rows: usize,
        _inner: usize,
        _cols: usize,
    ) -> Self::Storage {
        todo!()
    }

    fn transpose(
        _device: &Self::Device,
        _input: &Self::Storage,
        _rows: usize,
        _cols: usize,
    ) -> Self::Storage {
        todo!()
    }

    fn add_row(
        _device: &Self::Device,
        _input: &Self::Storage,
        _row: &Self::Storage,
        _rows: usize,
        _cols: usize,
    ) -> Self::Storage {
        todo!()
    }

    fn add_col(
        _device: &Self::Device,
        _input: &Self::Storage,
        _col: &Self::Storage,
        _rows: usize,
        _cols: usize,
    ) -> Self::Storage {
        todo!()
    }
}

fn main() {
    let cpu: Tensor2D<32, 10, f32, Cpu> = Tensor2D::zeros();
    let other: Tensor2D<32, 10, f32, OtherBackend> = Tensor2D::zeros();
    let _ = cpu.add(&other);
}
