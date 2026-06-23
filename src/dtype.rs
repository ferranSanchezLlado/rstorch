use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DTypeId {
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

impl FloatDType for f32 {
    fn from_usize(value: usize) -> Self {
        value as Self
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
}
