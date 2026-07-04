use super::{Backend, sealed};
use crate::dtype::DType;
use std::error;
use std::fmt;

#[derive(Debug, Clone, Copy, Default)]
pub struct Cpu;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuDevice {
    Cpu,
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CpuError {
    LengthMismatch {
        lhs: usize,
        rhs: usize,
    },
    BadMatmulDims {
        m: usize,
        k: usize,
        n: usize,
        lhs_len: usize,
        rhs_len: usize,
    },
}

impl fmt::Display for CpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { lhs, rhs } => write!(f, "length mismatch: {lhs} != {rhs}"),
            Self::BadMatmulDims {
                m,
                k,
                n,
                lhs_len,
                rhs_len,
            } => write!(
                f,
                "bad matmul dims ({m}, {k}, {n}) for operand lengths {lhs_len} and {rhs_len}"
            ),
        }
    }
}

impl error::Error for CpuError {}

impl sealed::SealedBackend for Cpu {}

impl<E: DType> Backend<E> for Cpu {
    type Device = CpuDevice;
    type Storage = Vec<E>;
    type Error = CpuError;

    fn default_device() -> std::result::Result<Self::Device, Self::Error> {
        Ok(CpuDevice::Cpu)
    }

    fn zeros(
        _device: &Self::Device,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        Ok(vec![E::ZERO; len])
    }

    fn ones(_device: &Self::Device, len: usize) -> std::result::Result<Self::Storage, Self::Error> {
        Ok(vec![E::ONE; len])
    }

    fn from_vec(
        _device: &Self::Device,
        data: Vec<E>,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        Ok(data)
    }

    fn to_vec(
        _device: &Self::Device,
        storage: &Self::Storage,
    ) -> std::result::Result<Vec<E>, Self::Error> {
        Ok(storage.clone())
    }

    fn storage_len(storage: &Self::Storage) -> usize {
        storage.len()
    }

    fn matmul(
        _device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        m: usize,
        k: usize,
        n: usize,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        if lhs.len() != m.saturating_mul(k) || rhs.len() != k.saturating_mul(n) {
            return Err(CpuError::BadMatmulDims {
                m,
                k,
                n,
                lhs_len: lhs.len(),
                rhs_len: rhs.len(),
            });
        }

        // Iteration order keeps the inner loop unit-stride over `rhs` and the
        // output row instead of striding `rhs` by `n` per step. Each output
        // element still accumulates its `k` terms in ascending-`inner` order,
        // so results are bitwise identical to the naive row/col/inner loop.
        let mut out = vec![E::ZERO; m.saturating_mul(n)];
        for row in 0..m {
            let out_row = &mut out[row * n..(row + 1) * n];
            for inner in 0..k {
                let scale = lhs[row * k + inner];
                let rhs_row = &rhs[inner * n..(inner + 1) * n];
                for (acc, &value) in out_row.iter_mut().zip(rhs_row) {
                    *acc += scale * value;
                }
            }
        }
        Ok(out)
    }

    fn add(
        _device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        binary(lhs, rhs, len, |a, b| a + b)
    }

    fn sub(
        _device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        binary(lhs, rhs, len, |a, b| a - b)
    }

    fn mul(
        _device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        binary(lhs, rhs, len, |a, b| a * b)
    }

    fn div(
        _device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        binary(lhs, rhs, len, |a, b| a / b)
    }

    fn add_scalar(
        _device: &Self::Device,
        input: &Self::Storage,
        rhs: E,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        unary_scalar(input, rhs, len, |a, b| a + b)
    }

    fn sub_scalar(
        _device: &Self::Device,
        input: &Self::Storage,
        rhs: E,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        unary_scalar(input, rhs, len, |a, b| a - b)
    }

    fn mul_scalar(
        _device: &Self::Device,
        input: &Self::Storage,
        rhs: E,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        unary_scalar(input, rhs, len, |a, b| a * b)
    }

    fn div_scalar(
        _device: &Self::Device,
        input: &Self::Storage,
        rhs: E,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        unary_scalar(input, rhs, len, |a, b| a / b)
    }

    fn sum(
        _device: &Self::Device,
        input: &Self::Storage,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        if input.len() != len {
            return Err(CpuError::LengthMismatch {
                lhs: input.len(),
                rhs: len,
            });
        }

        Ok(vec![input.iter().fold(E::ZERO, |acc, &value| acc + value)])
    }
}

fn binary<E, F>(lhs: &[E], rhs: &[E], len: usize, f: F) -> std::result::Result<Vec<E>, CpuError>
where
    E: DType,
    F: Fn(E, E) -> E,
{
    if lhs.len() != len || rhs.len() != len {
        return Err(CpuError::LengthMismatch {
            lhs: lhs.len(),
            rhs: rhs.len(),
        });
    }

    Ok(lhs.iter().zip(rhs.iter()).map(|(&a, &b)| f(a, b)).collect())
}

fn unary_scalar<E, F>(
    input: &[E],
    rhs: E,
    len: usize,
    f: F,
) -> std::result::Result<Vec<E>, CpuError>
where
    E: DType,
    F: Fn(E, E) -> E,
{
    if input.len() != len {
        return Err(CpuError::LengthMismatch {
            lhs: input.len(),
            rhs: len,
        });
    }

    Ok(input.iter().map(|&value| f(value, rhs)).collect())
}
