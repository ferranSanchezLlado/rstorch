use std::ops::{Add, Div, Mul, Sub};

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

impl DType for f32 {
    const ID: DTypeId = DTypeId::F32;

    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
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
