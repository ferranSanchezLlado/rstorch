//! Floating-point element types supported by tensors.

/// Project-owned floating-point trait used by tensor and backend code.
pub trait FloatElement:
    Copy
    + Default
    + Send
    + Sync
    + PartialEq
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + 'static
{
    fn zero() -> Self;
    fn one() -> Self;
    fn from_usize(value: usize) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sqrt(self) -> Self;
    fn powf(self, exponent: Self) -> Self;
}

impl FloatElement for f32 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn from_usize(value: usize) -> Self {
        value as Self
    }

    fn exp(self) -> Self {
        f32::exp(self)
    }

    fn ln(self) -> Self {
        f32::ln(self)
    }

    fn sqrt(self) -> Self {
        f32::sqrt(self)
    }

    fn powf(self, exponent: Self) -> Self {
        f32::powf(self, exponent)
    }
}

impl FloatElement for f64 {
    fn zero() -> Self {
        0.0
    }

    fn one() -> Self {
        1.0
    }

    fn from_usize(value: usize) -> Self {
        value as Self
    }

    fn exp(self) -> Self {
        f64::exp(self)
    }

    fn ln(self) -> Self {
        f64::ln(self)
    }

    fn sqrt(self) -> Self {
        f64::sqrt(self)
    }

    fn powf(self, exponent: Self) -> Self {
        f64::powf(self, exponent)
    }
}

#[cfg(test)]
mod tests {
    use super::FloatElement;

    #[test]
    fn default_float_element_is_f32_compatible() {
        assert_eq!(f32::zero(), 0.0);
        assert_eq!(f32::one(), 1.0);
        assert_eq!(f32::from_usize(3), 3.0);
    }
}
