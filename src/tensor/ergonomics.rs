use super::Tensor;
use crate::backend::Backend;
use crate::dtype::{DType, FloatDType};
use crate::error::Result;
use crate::shape::{ShapeSpec, StaticShape};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

const DISPLAY_HEAD: usize = 6;
const DISPLAY_TAIL: usize = 2;

impl<S, E, B> fmt::Display for Tensor<S, E, B>
where
    S: ShapeSpec,
    E: DType + fmt::Display,
    B: Backend<E>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(shape={:?}, dtype={:?}, values=",
            self.shape().dims(),
            self.dtype()
        )?;
        match self.to_vec() {
            Ok(values) => fmt_values(f, &values)?,
            Err(err) => write!(f, "<unavailable: {err}>")?,
        }
        write!(f, ")")
    }
}

fn fmt_values<E>(f: &mut fmt::Formatter<'_>, values: &[E]) -> fmt::Result
where
    E: fmt::Display,
{
    write!(f, "[")?;
    if values.len() <= DISPLAY_HEAD + DISPLAY_TAIL + 1 {
        for (idx, value) in values.iter().enumerate() {
            if idx > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{value}")?;
        }
    } else {
        for (idx, value) in values.iter().take(DISPLAY_HEAD).enumerate() {
            if idx > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{value}")?;
        }
        write!(f, ", ...")?;
        for value in values.iter().skip(values.len() - DISPLAY_TAIL) {
            write!(f, ", {value}")?;
        }
    }
    write!(f, "]")
}

macro_rules! impl_binary_tensor_op {
    ($trait:ident, $method:ident, $tensor_method:ident, $doc:literal) => {
        #[doc = $doc]
        impl<'b, S, E, B> $trait<&'b Tensor<S, E, B>> for &Tensor<S, E, B>
        where
            S: StaticShape,
            E: FloatDType,
            B: Backend<E>,
        {
            type Output = Result<Tensor<S, E, B>>;

            fn $method(self, rhs: &'b Tensor<S, E, B>) -> Self::Output {
                Tensor::$tensor_method(self, rhs)
            }
        }

        #[doc = $doc]
        impl<S, E, B> $trait<Tensor<S, E, B>> for &Tensor<S, E, B>
        where
            S: StaticShape,
            E: FloatDType,
            B: Backend<E>,
        {
            type Output = Result<Tensor<S, E, B>>;

            fn $method(self, rhs: Tensor<S, E, B>) -> Self::Output {
                Tensor::$tensor_method(self, &rhs)
            }
        }

        #[doc = $doc]
        impl<'b, S, E, B> $trait<&'b Tensor<S, E, B>> for Tensor<S, E, B>
        where
            S: StaticShape,
            E: FloatDType,
            B: Backend<E>,
        {
            type Output = Result<Tensor<S, E, B>>;

            fn $method(self, rhs: &'b Tensor<S, E, B>) -> Self::Output {
                Tensor::$tensor_method(&self, rhs)
            }
        }

        #[doc = $doc]
        impl<S, E, B> $trait<Tensor<S, E, B>> for Tensor<S, E, B>
        where
            S: StaticShape,
            E: FloatDType,
            B: Backend<E>,
        {
            type Output = Result<Tensor<S, E, B>>;

            fn $method(self, rhs: Tensor<S, E, B>) -> Self::Output {
                Tensor::$tensor_method(&self, &rhs)
            }
        }
    };
}

macro_rules! impl_scalar_rhs_op {
    ($trait:ident, $method:ident, $tensor_method:ident, $doc:literal) => {
        #[doc = $doc]
        impl<S, E, B> $trait<E> for &Tensor<S, E, B>
        where
            S: StaticShape,
            E: FloatDType,
            B: Backend<E>,
        {
            type Output = Result<Tensor<S, E, B>>;

            fn $method(self, rhs: E) -> Self::Output {
                Tensor::$tensor_method(self, rhs)
            }
        }

        #[doc = $doc]
        impl<S, E, B> $trait<E> for Tensor<S, E, B>
        where
            S: StaticShape,
            E: FloatDType,
            B: Backend<E>,
        {
            type Output = Result<Tensor<S, E, B>>;

            fn $method(self, rhs: E) -> Self::Output {
                Tensor::$tensor_method(&self, rhs)
            }
        }
    };
}

impl_binary_tensor_op!(
    Add,
    add,
    add,
    "Adds two same-static-shape tensors. The operator returns `Result` because backend kernels remain fallible; use `Tensor::add` for the explicit method form."
);
impl_binary_tensor_op!(
    Sub,
    sub,
    sub,
    "Subtracts two same-static-shape tensors. The operator returns `Result` because backend kernels remain fallible; use `Tensor::sub` for the explicit method form."
);
impl_binary_tensor_op!(
    Mul,
    mul,
    mul,
    "Multiplies two same-static-shape tensors. The operator returns `Result` because backend kernels remain fallible; use `Tensor::mul` for the explicit method form."
);
impl_binary_tensor_op!(
    Div,
    div,
    div,
    "Divides two same-static-shape tensors. The operator returns `Result` because backend kernels remain fallible; use `Tensor::div` for the explicit method form."
);

impl_scalar_rhs_op!(
    Add,
    add,
    add_scalar,
    "Adds a scalar to every element of a static-shape tensor. The operator returns `Result` because backend kernels remain fallible; use `Tensor::add_scalar` for the explicit method form."
);
impl_scalar_rhs_op!(
    Sub,
    sub,
    sub_scalar,
    "Subtracts a scalar from every element of a static-shape tensor. The operator returns `Result` because backend kernels remain fallible; use `Tensor::sub_scalar` for the explicit method form."
);
impl_scalar_rhs_op!(
    Mul,
    mul,
    mul_scalar,
    "Multiplies every element of a static-shape tensor by a scalar. The operator returns `Result` because backend kernels remain fallible; use `Tensor::mul_scalar` for the explicit method form."
);
impl_scalar_rhs_op!(
    Div,
    div,
    div_scalar,
    "Divides every element of a static-shape tensor by a scalar. The operator returns `Result` because backend kernels remain fallible; use `Tensor::div_scalar` for the explicit method form."
);

/// Negates a static-shape tensor. The operator returns `Result` because tensor
/// materialization remains fallible; use [`Tensor::neg`] for the explicit method
/// form.
impl<S, E, B> Neg for &Tensor<S, E, B>
where
    S: StaticShape,
    E: FloatDType,
    B: Backend<E>,
{
    type Output = Result<Tensor<S, E, B>>;

    fn neg(self) -> Self::Output {
        Tensor::neg(self)
    }
}

/// Negates a static-shape tensor. The operator returns `Result` because tensor
/// materialization remains fallible; use [`Tensor::neg`] for the explicit method
/// form.
impl<S, E, B> Neg for Tensor<S, E, B>
where
    S: StaticShape,
    E: FloatDType,
    B: Backend<E>,
{
    type Output = Result<Tensor<S, E, B>>;

    fn neg(self) -> Self::Output {
        Tensor::neg(&self)
    }
}
