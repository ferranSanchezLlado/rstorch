use std::ops::{Add, Div, Mul, Neg, Sub};

pub use half::{bf16, f16};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DTypeId {
    F16,
    BF16,
    F32,
    F64,
}

pub trait DType:
    Copy
    + Default
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + PartialOrd
{
    const ID: DTypeId;

    fn zero() -> Self;
    fn one() -> Self;
}

pub trait FloatDType: DType + Neg<Output = Self> {
    fn from_usize(value: usize) -> Self;
    fn from_f64(value: f64) -> Self;
    fn to_f64(self) -> f64;
    fn sqrt(self) -> Self;
}

impl DType for f32 {
    const ID: DTypeId = DTypeId::F32;

    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }
}

impl DType for f16 {
    const ID: DTypeId = DTypeId::F16;

    fn zero() -> Self {
        Self::from_f32(0.0)
    }

    fn one() -> Self {
        Self::from_f32(1.0)
    }
}

impl FloatDType for f16 {
    fn from_usize(value: usize) -> Self {
        Self::from_f32(value as f32)
    }

    fn from_f64(value: f64) -> Self {
        Self::from_f64(value)
    }

    fn to_f64(self) -> f64 {
        self.to_f64()
    }

    fn sqrt(self) -> Self {
        Self::from_f32(self.to_f32().sqrt())
    }
}

impl DType for bf16 {
    const ID: DTypeId = DTypeId::BF16;

    fn zero() -> Self {
        Self::from_f32(0.0)
    }

    fn one() -> Self {
        Self::from_f32(1.0)
    }
}

impl FloatDType for bf16 {
    fn from_usize(value: usize) -> Self {
        Self::from_f32(value as f32)
    }

    fn from_f64(value: f64) -> Self {
        Self::from_f64(value)
    }

    fn to_f64(self) -> f64 {
        self.to_f64()
    }

    fn sqrt(self) -> Self {
        Self::from_f32(self.to_f32().sqrt())
    }
}

impl FloatDType for f32 {
    fn from_usize(value: usize) -> Self {
        value as Self
    }

    fn from_f64(value: f64) -> Self {
        value as Self
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }
}

impl DType for f64 {
    const ID: DTypeId = DTypeId::F64;

    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }
}

impl FloatDType for f64 {
    fn from_usize(value: usize) -> Self {
        value as Self
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn to_f64(self) -> f64 {
        self
    }

    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }
}
