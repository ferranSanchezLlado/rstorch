use std::fmt;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use rstorch::{Backend, DType, DTypeId, FloatDType};

#[derive(Debug, Clone, Copy, Default, PartialEq, PartialOrd)]
struct RogueDType(f32);

impl Add for RogueDType {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for RogueDType {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub for RogueDType {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl SubAssign for RogueDType {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl Mul for RogueDType {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl MulAssign for RogueDType {
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl Div for RogueDType {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl DivAssign for RogueDType {
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl Neg for RogueDType {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl DType for RogueDType {
    const ID: DTypeId = DTypeId::F32;
    const ZERO: Self = Self(0.0);
    const ONE: Self = Self(1.0);
}

impl FloatDType for RogueDType {
    fn from_usize(value: usize) -> Self {
        Self(value as f32)
    }

    fn from_f64(value: f64) -> Self {
        Self(value as f32)
    }

    fn to_f64(self) -> f64 {
        self.0 as f64
    }

    fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }
}

#[derive(Debug, Clone, Copy)]
struct RogueBackend;

#[derive(Debug)]
struct RogueBackendError;

impl fmt::Display for RogueBackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rogue backend error")
    }
}

impl std::error::Error for RogueBackendError {}

impl Backend<f32> for RogueBackend {
    type Device = ();
    type Storage = Vec<f32>;
    type Error = RogueBackendError;

    fn default_device() -> Result<Self::Device, Self::Error> {
        Ok(())
    }

    fn zeros(_device: &Self::Device, len: usize) -> Result<Self::Storage, Self::Error> {
        Ok(vec![0.0; len])
    }

    fn ones(_device: &Self::Device, len: usize) -> Result<Self::Storage, Self::Error> {
        Ok(vec![1.0; len])
    }

    fn from_vec(_device: &Self::Device, data: Vec<f32>) -> Result<Self::Storage, Self::Error> {
        Ok(data)
    }

    fn to_vec(_device: &Self::Device, storage: &Self::Storage) -> Result<Vec<f32>, Self::Error> {
        Ok(storage.clone())
    }

    fn storage_len(storage: &Self::Storage) -> usize {
        storage.len()
    }

    fn matmul(
        _device: &Self::Device,
        _lhs: &Self::Storage,
        _rhs: &Self::Storage,
        m: usize,
        _k: usize,
        n: usize,
    ) -> Result<Self::Storage, Self::Error> {
        Ok(vec![0.0; m * n])
    }

    fn add(
        _device: &Self::Device,
        lhs: &Self::Storage,
        _rhs: &Self::Storage,
        _len: usize,
    ) -> Result<Self::Storage, Self::Error> {
        Ok(lhs.clone())
    }

    fn sub(
        _device: &Self::Device,
        lhs: &Self::Storage,
        _rhs: &Self::Storage,
        _len: usize,
    ) -> Result<Self::Storage, Self::Error> {
        Ok(lhs.clone())
    }

    fn mul(
        _device: &Self::Device,
        lhs: &Self::Storage,
        _rhs: &Self::Storage,
        _len: usize,
    ) -> Result<Self::Storage, Self::Error> {
        Ok(lhs.clone())
    }

    fn div(
        _device: &Self::Device,
        lhs: &Self::Storage,
        _rhs: &Self::Storage,
        _len: usize,
    ) -> Result<Self::Storage, Self::Error> {
        Ok(lhs.clone())
    }

    fn add_scalar(
        _device: &Self::Device,
        input: &Self::Storage,
        _rhs: f32,
        _len: usize,
    ) -> Result<Self::Storage, Self::Error> {
        Ok(input.clone())
    }

    fn sub_scalar(
        _device: &Self::Device,
        input: &Self::Storage,
        _rhs: f32,
        _len: usize,
    ) -> Result<Self::Storage, Self::Error> {
        Ok(input.clone())
    }

    fn mul_scalar(
        _device: &Self::Device,
        input: &Self::Storage,
        _rhs: f32,
        _len: usize,
    ) -> Result<Self::Storage, Self::Error> {
        Ok(input.clone())
    }

    fn div_scalar(
        _device: &Self::Device,
        input: &Self::Storage,
        _rhs: f32,
        _len: usize,
    ) -> Result<Self::Storage, Self::Error> {
        Ok(input.clone())
    }

    fn sum(
        _device: &Self::Device,
        _input: &Self::Storage,
        _len: usize,
    ) -> Result<Self::Storage, Self::Error> {
        Ok(vec![0.0])
    }
}

fn main() {}
