#![allow(clippy::type_complexity)]

mod image;
mod indexing;
mod matmul;
mod shape_ops;

pub use image::{Conv2dOptions, Padding2d, Pool2dOptions};

use super::autograd::{
    AnyTensor, raw_div, raw_div_scalar, raw_from_vec_like, raw_full_like, raw_mul, raw_mul_scalar,
    raw_neg,
};
use super::{Mask, RawTensor, Scalar, Tensor};
use crate::backend::Backend;
use crate::dtype::{DType, FloatDType};
use crate::error::{DeviceError, Error, Result, ShapeError};
use crate::shape::ShapeSpec;

type BinaryKernel<E, B> =
    fn(
        &<B as Backend<E>>::Device,
        &<B as Backend<E>>::Storage,
        &<B as Backend<E>>::Storage,
        usize,
    ) -> std::result::Result<<B as Backend<E>>::Storage, <B as Backend<E>>::Error>;

type ScalarKernel<E, B> =
    fn(
        &<B as Backend<E>>::Device,
        &<B as Backend<E>>::Storage,
        E,
        usize,
    ) -> std::result::Result<<B as Backend<E>>::Storage, <B as Backend<E>>::Error>;

impl<S, E, B> Tensor<S, E, B>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    pub fn add(&self, rhs: &Self) -> Result<Self> {
        self.binary_same_shape(rhs, "add", B::add, |grad, _lhs, _rhs| {
            Ok((grad.clone(), grad.clone()))
        })
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self> {
        self.binary_same_shape(rhs, "sub", B::sub, |grad, _lhs, _rhs| {
            Ok((grad.clone(), raw_neg(grad)?))
        })
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        self.binary_same_shape(rhs, "mul", B::mul, |grad, lhs, rhs| {
            Ok((raw_mul(grad, rhs)?, raw_mul(grad, lhs)?))
        })
    }

    pub fn div(&self, rhs: &Self) -> Result<Self> {
        self.binary_same_shape(rhs, "div", B::div, |grad, lhs, rhs| {
            let dx = raw_div(grad, rhs)?;
            let rhs_sq = raw_mul(rhs, rhs)?;
            let dy = raw_neg(&raw_div(&raw_mul(grad, lhs)?, &rhs_sq)?)?;
            Ok((dx, dy))
        })
    }

    pub fn add_scalar(&self, rhs: E) -> Result<Self> {
        self.unary_scalar(rhs, B::add_scalar, |grad| Ok(grad.clone()))
    }

    pub fn sub_scalar(&self, rhs: E) -> Result<Self> {
        self.unary_scalar(rhs, B::sub_scalar, |grad| Ok(grad.clone()))
    }

    pub fn mul_scalar(&self, rhs: E) -> Result<Self> {
        self.unary_scalar(rhs, B::mul_scalar, move |grad| raw_mul_scalar(grad, rhs))
    }

    pub fn div_scalar(&self, rhs: E) -> Result<Self> {
        self.unary_scalar(rhs, B::div_scalar, move |grad| raw_div_scalar(grad, rhs))
    }

    pub fn sum(&self) -> Result<Scalar<E, B>> {
        let input = self.contiguous()?;
        let storage =
            B::sum(input.device(), input.raw().storage(), input.numel()).map_err(Error::backend)?;
        let raw = RawTensor::from_storage_on(
            self.device().clone(),
            storage,
            crate::shape::Shape::known([]),
        )?;
        let input_raw = self.raw().clone();
        Scalar::<E, B>::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            let seed = grad.to_vec()?[0];
            Ok(vec![Some(raw_full_like(&input_raw, seed)?)])
        })
    }

    pub fn mean(&self) -> Result<Scalar<E, B>> {
        if self.numel() == 0 {
            return Err(ShapeError::ZeroDimension {
                op: "mean",
                axis: 0,
            }
            .into());
        }
        let input_values = self.to_vec()?;
        let sum = input_values
            .iter()
            .fold(<E::Acc as DType>::ZERO, |acc, &value| {
                acc + E::Acc::from_f64(value.to_f64())
            });
        let denom = E::Acc::from_usize(self.numel());
        let mean = E::from_f64((sum / denom).to_f64());
        let raw = RawTensor::from_vec_on(
            self.device().clone(),
            vec![mean],
            crate::shape::Shape::known([]),
        )?;
        let input_raw = self.raw().clone();
        let numel = self.numel();
        Scalar::<E, B>::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            let seed = grad.to_vec()?[0] / E::from_usize(numel);
            Ok(vec![Some(raw_full_like(&input_raw, seed)?)])
        })
    }

    pub fn var(&self) -> Result<Scalar<E, B>> {
        if self.numel() == 0 {
            return Err(ShapeError::ZeroDimension { op: "var", axis: 0 }.into());
        }
        let values = self.to_vec()?;
        let numel = self.numel();
        let denom = E::Acc::from_usize(numel);
        let mean_acc = values.iter().fold(<E::Acc as DType>::ZERO, |acc, &value| {
            acc + E::Acc::from_f64(value.to_f64())
        }) / denom;
        let var_acc = values.iter().fold(<E::Acc as DType>::ZERO, |acc, &value| {
            let diff = E::Acc::from_f64(value.to_f64()) - mean_acc;
            acc + diff * diff
        }) / denom;
        let var = E::from_f64(var_acc.to_f64());
        let raw = RawTensor::from_vec_on(
            self.device().clone(),
            vec![var],
            crate::shape::Shape::known([]),
        )?;
        let input_raw = self.raw().clone();
        let mean = E::from_f64(mean_acc.to_f64());
        Scalar::<E, B>::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            let seed = grad.to_vec()?[0];
            let scale = E::from_f64(2.0) / E::from_usize(numel);
            let grad_values = values
                .iter()
                .map(|&value| seed * scale * (value - mean))
                .collect();
            Ok(vec![Some(raw_from_vec_like(&input_raw, grad_values)?)])
        })
    }

    pub fn std(&self) -> Result<Scalar<E, B>> {
        self.var()?.sqrt()
    }

    pub fn min(&self) -> Result<Scalar<E, B>> {
        self.full_extreme("min", |value, best| value < best)
    }

    pub fn max(&self) -> Result<Scalar<E, B>> {
        self.full_extreme("max", |value, best| value > best)
    }

    pub fn abs(&self) -> Result<Self> {
        self.unary_map(
            |x| if x < E::ZERO { -x } else { x },
            |x, _y| {
                if x < E::ZERO {
                    -E::ONE
                } else if x > E::ZERO {
                    E::ONE
                } else {
                    E::ZERO
                }
            },
        )
    }

    pub fn relu(&self) -> Result<Self> {
        let input = self.contiguous()?;
        let values = input
            .to_vec()?
            .into_iter()
            .map(|value| if value > E::ZERO { value } else { E::ZERO })
            .collect();
        let raw = RawTensor::from_vec_on(self.device().clone(), values, self.shape().clone())?;
        let input_raw = self.raw().clone();
        Self::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            let mask = input_raw
                .to_vec()?
                .into_iter()
                .map(|value| if value > E::ZERO { E::ONE } else { E::ZERO });
            let grad_values = grad
                .to_vec()?
                .into_iter()
                .zip(mask)
                .map(|(g, m)| g * m)
                .collect();
            Ok(vec![Some(raw_from_vec_like(&input_raw, grad_values)?)])
        })
    }

    pub fn neg(&self) -> Result<Self> {
        self.unary_map(|x| -x, |_x, _y| -E::ONE)
    }

    pub fn exp(&self) -> Result<Self> {
        self.unary_map(|x| x.exp(), |_x, y| y)
    }

    pub fn ln(&self) -> Result<Self> {
        self.unary_map(|x| x.ln(), |x, _y| E::ONE / x)
    }

    pub fn tanh(&self) -> Result<Self> {
        self.unary_map(|x| x.tanh(), |_x, y| E::ONE - y * y)
    }

    pub fn sigmoid(&self) -> Result<Self> {
        self.unary_map(|x| E::ONE / (E::ONE + (-x).exp()), |_x, y| y * (E::ONE - y))
    }

    /// Raises each element to `exponent`. Gradients follow the mathematical
    /// derivative and may produce infinities or NaNs at values outside the
    /// real-valued derivative domain for the chosen exponent.
    pub fn powf(&self, exponent: E) -> Result<Self> {
        self.unary_map(
            move |x| x.powf(exponent),
            move |x, _y| exponent * x.powf(exponent - E::ONE),
        )
    }

    /// Computes elementwise square root. Gradients follow the mathematical
    /// derivative and may produce infinities or NaNs at non-positive inputs.
    pub fn sqrt(&self) -> Result<Self> {
        self.unary_map(|x| x.sqrt(), |x, _y| half::<E>() / x.sqrt())
    }

    /// Computes elementwise reciprocal square root. Gradients follow the
    /// mathematical derivative and may produce infinities or NaNs at
    /// non-positive inputs.
    pub fn rsqrt(&self) -> Result<Self> {
        self.unary_map(|x| E::ONE / x.sqrt(), |x, _y| -half::<E>() / (x * x.sqrt()))
    }

    pub fn clamp(&self, min: E, max: E) -> Result<Self> {
        self.unary_map(
            move |x| {
                if x < min {
                    min
                } else if x > max {
                    max
                } else {
                    x
                }
            },
            move |x, _y| {
                if x < min || x > max { E::ZERO } else { E::ONE }
            },
        )
    }

    pub fn gelu(&self) -> Result<Self> {
        self.unary_map(
            move |x| {
                let x3 = x * x * x;
                half::<E>() * x * (E::ONE + (gelu_k::<E>() * (x + gelu_c::<E>() * x3)).tanh())
            },
            move |x, _y| {
                let x2 = x * x;
                let inner = gelu_k::<E>() * (x + gelu_c::<E>() * x * x2);
                let t = inner.tanh();
                let sech2 = E::ONE - t * t;
                half::<E>() * (E::ONE + t)
                    + half::<E>()
                        * x
                        * sech2
                        * gelu_k::<E>()
                        * (E::ONE + three::<E>() * gelu_c::<E>() * x2)
            },
        )
    }

    pub fn masked_fill(&self, mask: &Mask<S, B>, value: E) -> Result<Self> {
        if self.shape() != mask.shape() {
            return Err(ShapeError::LengthMismatch {
                op: "masked_fill",
                expected: self.numel(),
                found: mask.to_vec()?.len(),
            }
            .into());
        }
        let mask_values = mask.to_vec()?;
        let values = self
            .to_vec()?
            .into_iter()
            .zip(&mask_values)
            .map(|(x, &m)| if m { value } else { x })
            .collect();
        let raw = RawTensor::from_vec_on(self.device().clone(), values, self.shape().clone())?;
        let input_raw = self.raw().clone();
        Self::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            let grad_values = grad
                .to_vec()?
                .into_iter()
                .zip(&mask_values)
                .map(|(g, &m)| if m { E::ZERO } else { g })
                .collect();
            Ok(vec![Some(raw_from_vec_like(&input_raw, grad_values)?)])
        })
    }

    pub fn where_mask(&self, mask: &Mask<S, B>, other: &Self) -> Result<Self> {
        self.ensure_same_device(other, "where_mask")?;
        if self.shape() != mask.shape() || other.shape() != mask.shape() {
            return Err(ShapeError::LengthMismatch {
                op: "where_mask",
                expected: self.numel(),
                found: mask.to_vec()?.len(),
            }
            .into());
        }
        let mask_values = mask.to_vec()?;
        let lhs_values = self.to_vec()?;
        let rhs_values = other.to_vec()?;
        let values = lhs_values
            .into_iter()
            .zip(rhs_values)
            .zip(&mask_values)
            .map(|((a, b), &m)| if m { a } else { b })
            .collect();
        let raw = RawTensor::from_vec_on(self.device().clone(), values, self.shape().clone())?;
        let lhs_raw = self.raw().clone();
        let rhs_raw = other.raw().clone();
        Self::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(other)],
            move |grad| {
                let grad_values = grad.to_vec()?;
                let lhs_grad = grad_values
                    .iter()
                    .zip(&mask_values)
                    .map(|(&g, &m)| if m { g } else { E::ZERO })
                    .collect();
                let rhs_grad = grad_values
                    .into_iter()
                    .zip(&mask_values)
                    .map(|(g, &m)| if m { E::ZERO } else { g })
                    .collect();
                Ok(vec![
                    Some(raw_from_vec_like(&lhs_raw, lhs_grad)?),
                    Some(raw_from_vec_like(&rhs_raw, rhs_grad)?),
                ])
            },
        )
    }

    fn ensure_same_device(&self, rhs: &Self, op: &'static str) -> Result<()> {
        ensure_same_device::<E, B>(self.device(), rhs.device(), op)
    }

    fn binary_same_shape(
        &self,
        rhs: &Self,
        op: &'static str,
        kernel: BinaryKernel<E, B>,
        backward: impl Fn(
            &RawTensor<E, B>,
            &RawTensor<E, B>,
            &RawTensor<E, B>,
        ) -> Result<(RawTensor<E, B>, RawTensor<E, B>)>
        + Send
        + Sync
        + 'static,
    ) -> Result<Self> {
        self.ensure_same_device(rhs, op)?;

        let lhs_dims = self.shape().dims();
        let rhs_dims = rhs.shape().dims();
        crate::shape::bind_and_check(
            op,
            S::dim_entries(0)
                .into_iter()
                .zip(lhs_dims.iter().copied())
                .chain(S::dim_entries(1).into_iter().zip(rhs_dims.iter().copied())),
        )?;

        for (axis, (&lhs, &rhs)) in lhs_dims.iter().zip(rhs_dims.iter()).enumerate() {
            if lhs != rhs {
                return Err(ShapeError::DimMismatch {
                    op,
                    operand: 1,
                    axis,
                    expected: lhs,
                    found: rhs,
                }
                .into());
            }
        }

        let lhs_input = self.contiguous()?;
        let rhs_input = rhs.contiguous()?;
        let storage = kernel(
            lhs_input.device(),
            lhs_input.raw().storage(),
            rhs_input.raw().storage(),
            lhs_input.numel(),
        )
        .map_err(Error::backend)?;
        let raw = RawTensor::from_storage_on(self.device().clone(), storage, self.shape().clone())?;
        let lhs_raw = self.raw().clone();
        let rhs_raw = rhs.raw().clone();
        Self::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(rhs)],
            move |grad| {
                let (lhs_grad, rhs_grad) = backward(grad, &lhs_raw, &rhs_raw)?;
                Ok(vec![Some(lhs_grad), Some(rhs_grad)])
            },
        )
    }

    fn unary_scalar(
        &self,
        rhs: E,
        kernel: ScalarKernel<E, B>,
        backward: impl Fn(&RawTensor<E, B>) -> Result<RawTensor<E, B>> + Send + Sync + 'static,
    ) -> Result<Self> {
        let input = self.contiguous()?;
        let storage = kernel(input.device(), input.raw().storage(), rhs, input.numel())
            .map_err(Error::backend)?;
        let raw = RawTensor::from_storage_on(self.device().clone(), storage, self.shape().clone())?;
        Self::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            Ok(vec![Some(backward(grad)?)])
        })
    }

    fn unary_map(
        &self,
        forward: impl Fn(E) -> E + Copy + Send + Sync + 'static,
        backward: impl Fn(E, E) -> E + Copy + Send + Sync + 'static,
    ) -> Result<Self> {
        let input_values = self.to_vec()?;
        let output_values = input_values
            .iter()
            .copied()
            .map(forward)
            .collect::<Vec<_>>();
        let raw = RawTensor::from_vec_on(
            self.device().clone(),
            output_values.clone(),
            self.shape().clone(),
        )?;
        let input_raw = self.raw().clone();
        Self::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            let grad_values = grad
                .to_vec()?
                .into_iter()
                .zip(
                    input_values
                        .iter()
                        .copied()
                        .zip(output_values.iter().copied()),
                )
                .map(|(g, (x, y))| g * backward(x, y))
                .collect();
            Ok(vec![Some(raw_from_vec_like(&input_raw, grad_values)?)])
        })
    }

    fn full_extreme(
        &self,
        op: &'static str,
        better: impl Fn(E, E) -> bool + Copy + Send + Sync + 'static,
    ) -> Result<Scalar<E, B>> {
        if self.numel() == 0 {
            return Err(ShapeError::ZeroDimension { op, axis: 0 }.into());
        }
        let values = self.to_vec()?;
        let mut best = values[0];
        for &value in &values[1..] {
            if better(value, best) {
                best = value;
            }
        }
        let raw = RawTensor::from_vec_on(
            self.device().clone(),
            vec![best],
            crate::shape::Shape::known([]),
        )?;
        let input_raw = self.raw().clone();
        Scalar::<E, B>::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            let seed = grad.to_vec()?[0];
            let count = values.iter().filter(|&&value| value == best).count();
            let each = seed / E::from_usize(count);
            let grad_values = values
                .iter()
                .map(|&value| if value == best { each } else { E::ZERO })
                .collect();
            Ok(vec![Some(raw_from_vec_like(&input_raw, grad_values)?)])
        })
    }
}

impl<S, E, B> Tensor<S, E, B>
where
    S: ShapeSpec,
    E: DType,
    B: Backend<E>,
{
    pub fn gt_scalar(&self, rhs: E) -> Result<Mask<S, B>> {
        self.compare_scalar(rhs, |a, b| a > b)
    }

    pub fn ge_scalar(&self, rhs: E) -> Result<Mask<S, B>> {
        self.compare_scalar(rhs, |a, b| a >= b)
    }

    pub fn lt_scalar(&self, rhs: E) -> Result<Mask<S, B>> {
        self.compare_scalar(rhs, |a, b| a < b)
    }

    pub fn le_scalar(&self, rhs: E) -> Result<Mask<S, B>> {
        self.compare_scalar(rhs, |a, b| a <= b)
    }

    pub fn eq_scalar(&self, rhs: E) -> Result<Mask<S, B>> {
        self.compare_scalar(rhs, |a, b| a == b)
    }

    fn compare_scalar(&self, rhs: E, compare: impl Fn(E, E) -> bool) -> Result<Mask<S, B>> {
        Mask::from_vec_with_shape(
            self.to_vec()?
                .into_iter()
                .map(|value| compare(value, rhs))
                .collect(),
            self.shape().clone(),
        )
    }
}

fn ensure_same_device<E, B>(lhs: &B::Device, rhs: &B::Device, op: &'static str) -> Result<()>
where
    E: DType,
    B: Backend<E>,
{
    if lhs != rhs {
        return Err(DeviceError::Mismatch {
            op,
            lhs: format!("{lhs:?}"),
            rhs: format!("{rhs:?}"),
        }
        .into());
    }
    Ok(())
}

fn ensure_nonzero_dim(op: &'static str, axis: usize, size: usize) -> Result<()> {
    if size == 0 {
        return Err(ShapeError::ZeroDimension { op, axis }.into());
    }
    Ok(())
}

fn half<E: FloatDType>() -> E {
    E::from_f64(0.5)
}

fn three<E: FloatDType>() -> E {
    E::from_f64(3.0)
}

fn gelu_k<E: FloatDType>() -> E {
    E::from_f64(0.797_884_560_802_865_4)
}

fn gelu_c<E: FloatDType>() -> E {
    E::from_f64(0.044_715)
}
