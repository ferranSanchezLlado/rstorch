use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

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
    const ZERO: Self;
    const ONE: Self;

    fn zero() -> Self {
        Self::ZERO
    }

    fn one() -> Self {
        Self::ONE
    }
}

pub trait FloatDType: DType + Neg<Output = Self> {
    const HALF: Self;
    const THREE: Self;
    const GELU_K: Self;
    const GELU_C: Self;

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

impl DType for f32 {
    const ID: DTypeId = DTypeId::F32;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}

impl DType for f16 {
    const ID: DTypeId = DTypeId::F16;
    const ZERO: Self = f16::ZERO;
    const ONE: Self = f16::ONE;
}

impl FloatDType for f16 {
    const HALF: Self = f16::from_f32_const(0.5);
    const THREE: Self = f16::from_f32_const(3.0);
    const GELU_K: Self = f16::from_f32_const(0.797_884_6);
    const GELU_C: Self = f16::from_f32_const(0.044_715);

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

impl DType for bf16 {
    const ID: DTypeId = DTypeId::BF16;
    const ZERO: Self = bf16::ZERO;
    const ONE: Self = bf16::ONE;
}

impl FloatDType for bf16 {
    const HALF: Self = bf16::from_f32_const(0.5);
    const THREE: Self = bf16::from_f32_const(3.0);
    const GELU_K: Self = bf16::from_f32_const(0.797_884_6);
    const GELU_C: Self = bf16::from_f32_const(0.044_715);

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
    const HALF: Self = 0.5;
    const THREE: Self = 3.0;
    const GELU_K: Self = 0.797_884_6;
    const GELU_C: Self = 0.044_715;

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

impl DType for f64 {
    const ID: DTypeId = DTypeId::F64;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}

impl FloatDType for f64 {
    const HALF: Self = 0.5;
    const THREE: Self = 3.0;
    const GELU_K: Self = 0.797_884_560_802_865_4;
    const GELU_C: Self = 0.044_715;

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
