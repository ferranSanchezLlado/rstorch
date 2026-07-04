use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

pub use half::{bf16, f16};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DTypeId {
    F16,
    BF16,
    F32,
    F64,
}

mod sealed {
    pub trait SealedDType {}
}

/// Element type supported by RsTorch tensors.
///
/// The dtype set is sealed before 1.0 while backend and kernel coverage is
/// still expanding. New dtypes will be added in-tree without requiring
/// downstream implementations to track an unstable trait surface.
pub trait DType:
    sealed::SealedDType
    + Copy
    + Default
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + PartialEq
    + PartialOrd
{
    const ID: DTypeId;
    const BYTE_SIZE: usize;
    const ZERO: Self;
    const ONE: Self;

    fn write_le_bytes(self, out: &mut Vec<u8>);
    fn read_le_bytes(bytes: &[u8]) -> Option<Self>;

    fn zero() -> Self {
        Self::ZERO
    }

    fn one() -> Self {
        Self::ONE
    }
}

/// Floating-point dtype operations.
///
/// This trait is sealed for the same reason as [`DType`]: the set of required
/// operations remains crate-owned until dtype and backend expansion settles.
pub trait FloatDType: DType + Neg<Output = Self> {
    fn from_usize(value: usize) -> Self;
    fn from_f32(value: f32) -> Self {
        Self::from_f64(value as f64)
    }
    fn from_f64(value: f64) -> Self;
    fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }
    fn to_f64(self) -> f64;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self {
        Self::from_f64(self.to_f64().exp())
    }
    fn ln(self) -> Self {
        Self::from_f64(self.to_f64().ln())
    }
    fn tanh(self) -> Self {
        Self::from_f64(self.to_f64().tanh())
    }
    fn powf(self, exponent: Self) -> Self {
        Self::from_f64(self.to_f64().powf(exponent.to_f64()))
    }
}

impl sealed::SealedDType for f32 {}

impl DType for f32 {
    const ID: DTypeId = DTypeId::F32;
    const BYTE_SIZE: usize = 4;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    fn write_le_bytes(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }

    fn read_le_bytes(bytes: &[u8]) -> Option<Self> {
        let bytes: [u8; 4] = bytes.try_into().ok()?;
        Some(Self::from_le_bytes(bytes))
    }
}

impl sealed::SealedDType for f16 {}

impl DType for f16 {
    const ID: DTypeId = DTypeId::F16;
    const BYTE_SIZE: usize = 2;
    const ZERO: Self = f16::ZERO;
    const ONE: Self = f16::ONE;

    fn write_le_bytes(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_bits().to_le_bytes());
    }

    fn read_le_bytes(bytes: &[u8]) -> Option<Self> {
        let bytes: [u8; 2] = bytes.try_into().ok()?;
        Some(Self::from_bits(u16::from_le_bytes(bytes)))
    }
}

impl FloatDType for f16 {
    fn from_usize(value: usize) -> Self {
        Self::from_f32(value as f32)
    }

    fn from_f32(value: f32) -> Self {
        f16::from_f32(value)
    }

    fn from_f64(value: f64) -> Self {
        f16::from_f64(value)
    }

    fn to_f32(self) -> f32 {
        f16::to_f32(self)
    }

    fn to_f64(self) -> f64 {
        f16::to_f64(self)
    }

    fn sqrt(self) -> Self {
        Self::from_f32(self.to_f32().sqrt())
    }

    fn exp(self) -> Self {
        Self::from_f32(self.to_f32().exp())
    }

    fn ln(self) -> Self {
        Self::from_f32(self.to_f32().ln())
    }

    fn tanh(self) -> Self {
        Self::from_f32(self.to_f32().tanh())
    }

    fn powf(self, exponent: Self) -> Self {
        Self::from_f32(self.to_f32().powf(exponent.to_f32()))
    }
}

impl sealed::SealedDType for bf16 {}

impl DType for bf16 {
    const ID: DTypeId = DTypeId::BF16;
    const BYTE_SIZE: usize = 2;
    const ZERO: Self = bf16::ZERO;
    const ONE: Self = bf16::ONE;

    fn write_le_bytes(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_bits().to_le_bytes());
    }

    fn read_le_bytes(bytes: &[u8]) -> Option<Self> {
        let bytes: [u8; 2] = bytes.try_into().ok()?;
        Some(Self::from_bits(u16::from_le_bytes(bytes)))
    }
}

impl FloatDType for bf16 {
    fn from_usize(value: usize) -> Self {
        Self::from_f32(value as f32)
    }

    fn from_f32(value: f32) -> Self {
        bf16::from_f32(value)
    }

    fn from_f64(value: f64) -> Self {
        bf16::from_f64(value)
    }

    fn to_f32(self) -> f32 {
        bf16::to_f32(self)
    }

    fn to_f64(self) -> f64 {
        bf16::to_f64(self)
    }

    fn sqrt(self) -> Self {
        Self::from_f32(self.to_f32().sqrt())
    }

    fn exp(self) -> Self {
        Self::from_f32(self.to_f32().exp())
    }

    fn ln(self) -> Self {
        Self::from_f32(self.to_f32().ln())
    }

    fn tanh(self) -> Self {
        Self::from_f32(self.to_f32().tanh())
    }

    fn powf(self, exponent: Self) -> Self {
        Self::from_f32(self.to_f32().powf(exponent.to_f32()))
    }
}

impl FloatDType for f32 {
    fn from_usize(value: usize) -> Self {
        value as Self
    }

    fn from_f32(value: f32) -> Self {
        value
    }

    fn from_f64(value: f64) -> Self {
        value as Self
    }

    fn to_f32(self) -> f32 {
        self
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }

    fn exp(self) -> Self {
        f32::exp(self)
    }

    fn ln(self) -> Self {
        f32::ln(self)
    }

    fn tanh(self) -> Self {
        f32::tanh(self)
    }

    fn powf(self, exponent: Self) -> Self {
        f32::powf(self, exponent)
    }
}

impl sealed::SealedDType for f64 {}

impl DType for f64 {
    const ID: DTypeId = DTypeId::F64;
    const BYTE_SIZE: usize = 8;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    fn write_le_bytes(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }

    fn read_le_bytes(bytes: &[u8]) -> Option<Self> {
        let bytes: [u8; 8] = bytes.try_into().ok()?;
        Some(Self::from_le_bytes(bytes))
    }
}

impl FloatDType for f64 {
    fn from_usize(value: usize) -> Self {
        value as Self
    }

    fn from_f32(value: f32) -> Self {
        value as Self
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn to_f32(self) -> f32 {
        self as f32
    }

    fn to_f64(self) -> f64 {
        self
    }

    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    fn exp(self) -> Self {
        f64::exp(self)
    }

    fn ln(self) -> Self {
        f64::ln(self)
    }

    fn tanh(self) -> Self {
        f64::tanh(self)
    }

    fn powf(self, exponent: Self) -> Self {
        f64::powf(self, exponent)
    }
}
