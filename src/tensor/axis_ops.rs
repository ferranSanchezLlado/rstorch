use super::autograd::{AnyTensor, raw_from_vec_like};
use super::{RawTensor, Tensor};
use crate::backend::{Backend, parallel};
use crate::dtype::{DType, FloatDType};
use crate::error::{DeviceError, Result, ShapeError, const_check};
use crate::shape::{DimSpec, LastAxis, LeadingAxis, Shape, ShapeSpec, bind_and_check};

impl<S, E, B> Tensor<S, E, B>
where
    S: LastAxis,
    E: DType,
    B: Backend<E>,
{
    /// Returns the index of the first maximum value along the last axis.
    pub fn argmax_last(&self) -> Result<Vec<usize>> {
        const { const_check::known_nonzero(<S::Last as DimSpec>::KNOWN, "argmax_last", "last axis") };

        let dims = self.shape().dims();
        let last_axis = S::RANK - 1;
        let rows = product(&dims[..last_axis]);
        let cols = dims[last_axis];
        ensure_nonzero_dim("argmax_last", last_axis, cols)?;
        let values = self.host_values()?;
        let mut out = Vec::with_capacity(rows);
        for row in 0..rows {
            let start = row * cols;
            let mut best = 0usize;
            let mut best_value = values[start];
            for col in 1..cols {
                let value = values[start + col];
                if value > best_value {
                    best = col;
                    best_value = value;
                }
            }
            out.push(best);
        }
        Ok(out)
    }

    /// Returns [`argmax_last`](Self::argmax_last) as an i64 data tensor.
    ///
    /// This CPU-oriented id path is available only when the backend also
    /// implements `Backend<i64>`. The universal `Vec<usize>` method remains the
    /// every-backend path. i64 tensors do not support autograd, arithmetic, or
    /// matmul, and GPU i64 storage is intentionally out of scope before 1.0.
    pub fn argmax_last_tensor(&self) -> Result<Tensor<S::Reduced, i64, B>>
    where
        B: Backend<i64, Device = <B as Backend<E>>::Device>,
    {
        let dims = self.shape().dims();
        let last_axis = S::RANK - 1;
        let shape = Shape::known(dims[..last_axis].to_vec());
        let values = self
            .argmax_last()?
            .into_iter()
            .map(|value| value as i64)
            .collect();
        let raw = RawTensor::<i64, B>::from_vec_on(self.device().clone(), values, shape)?;
        Tensor::<S::Reduced, i64, B>::from_raw(raw)
    }
}

impl<S, E, B> Tensor<S, E, B>
where
    S: LastAxis,
    E: FloatDType,
    B: Backend<E>,
{
    pub fn add_last_dim(&self, rhs: &Tensor<S::Row, E, B>) -> Result<Self> {
        self.broadcast_last_axis(
            rhs,
            "add_last_dim",
            |a, b| a + b,
            |_a, _b, g| g,
            |_a, _b, g| g,
        )
    }

    pub fn sub_last_dim(&self, rhs: &Tensor<S::Row, E, B>) -> Result<Self> {
        self.broadcast_last_axis(
            rhs,
            "sub_last_dim",
            |a, b| a - b,
            |_a, _b, g| g,
            |_a, _b, g| -g,
        )
    }

    pub fn mul_last_dim(&self, rhs: &Tensor<S::Row, E, B>) -> Result<Self> {
        self.broadcast_last_axis(
            rhs,
            "mul_last_dim",
            |a, b| a * b,
            |_a, b, g| g * b,
            |a, _b, g| g * a,
        )
    }

    pub fn div_last_dim(&self, rhs: &Tensor<S::Row, E, B>) -> Result<Self> {
        self.broadcast_last_axis(
            rhs,
            "div_last_dim",
            |a, b| a / b,
            |_a, b, g| g / b,
            |a, b, g| -(g * a) / (b * b),
        )
    }

    /// Reduces the last axis with `keepdim = false`.
    pub fn sum_last(&self) -> Result<Tensor<S::Reduced, E, B>> {
        let dims = self.shape().dims();
        let last_axis = S::RANK - 1;
        let rows = product(&dims[..last_axis]);
        let cols = dims[last_axis];
        let values = self.host_values()?;
        let mut out = vec![E::ZERO; rows];
        for row in 0..rows {
            for col in 0..cols {
                out[row] += values[row * cols + col];
            }
        }
        let raw = RawTensor::from_vec_on(
            self.device().clone(),
            out,
            Shape::known(dims[..last_axis].to_vec()),
        )?;
        let input_raw = self.raw().clone();
        Tensor::<S::Reduced, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self)],
            move |grad| {
                let seed = grad.host_values()?;
                let mut values = Vec::with_capacity(rows * cols);
                for &g in seed.iter() {
                    values.extend(std::iter::repeat_n(g, cols));
                }
                Ok(vec![Some(raw_from_vec_like(&input_raw, values)?)])
            },
        )
    }

    pub fn mean_last(&self) -> Result<Tensor<S::Reduced, E, B>> {
        const { const_check::known_nonzero(<S::Last as DimSpec>::KNOWN, "mean_last", "last axis") };
        let axis = S::RANK - 1;
        let cols = self.shape().dims()[axis];
        ensure_nonzero_dim("mean_last", axis, cols)?;
        self.sum_last()?.div_scalar(E::from_usize(cols))
    }

    pub fn var_last(&self) -> Result<Tensor<S::Reduced, E, B>> {
        const { const_check::known_nonzero(<S::Last as DimSpec>::KNOWN, "var_last", "last axis") };
        let axis = S::RANK - 1;
        let cols = self.shape().dims()[axis];
        ensure_nonzero_dim("var_last", axis, cols)?;
        let dims = self.shape().dims();
        let rows = product(&dims[..axis]);
        let values = self.host_values()?.into_owned();
        let mut means = vec![E::ZERO; rows];
        let mut out = vec![E::ZERO; rows];
        for row in 0..rows {
            let start = row * cols;
            let mean_acc = values[start..start + cols]
                .iter()
                .fold(<E::Acc as crate::dtype::DType>::ZERO, |acc, &value| {
                    acc + E::Acc::from_f64(value.to_f64())
                })
                / E::Acc::from_usize(cols);
            let var_acc = values[start..start + cols].iter().fold(
                <E::Acc as crate::dtype::DType>::ZERO,
                |acc, &value| {
                    let diff = E::Acc::from_f64(value.to_f64()) - mean_acc;
                    acc + diff * diff
                },
            ) / E::Acc::from_usize(cols);
            means[row] = E::from_f64(mean_acc.to_f64());
            out[row] = E::from_f64(var_acc.to_f64());
        }
        let raw = RawTensor::from_vec_on(
            self.device().clone(),
            out,
            Shape::known(dims[..axis].to_vec()),
        )?;
        let input_raw = self.raw().clone();
        Tensor::<S::Reduced, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self)],
            move |grad| {
                let seed = grad.host_values()?;
                let mut grad_values = vec![E::ZERO; rows * cols];
                let scale = E::from_f64(2.0) / E::from_usize(cols);
                for row in 0..rows {
                    let start = row * cols;
                    for col in 0..cols {
                        grad_values[start + col] =
                            seed[row] * scale * (values[start + col] - means[row]);
                    }
                }
                Ok(vec![Some(raw_from_vec_like(&input_raw, grad_values)?)])
            },
        )
    }

    pub fn std_last(&self) -> Result<Tensor<S::Reduced, E, B>> {
        self.var_last()?.sqrt()
    }

    /// Reduces the last axis with max. Backward splits gradient evenly across
    /// tied maxima instead of selecting the first maximum.
    pub fn max_last(&self) -> Result<Tensor<S::Reduced, E, B>> {
        const { const_check::known_nonzero(<S::Last as DimSpec>::KNOWN, "max_last", "last axis") };

        let dims = self.shape().dims();
        let last_axis = S::RANK - 1;
        let rows = product(&dims[..last_axis]);
        let cols = dims[last_axis];
        ensure_nonzero_dim("max_last", last_axis, cols)?;
        let values = self.host_values()?.into_owned();
        let mut out = Vec::with_capacity(rows);
        for row in 0..rows {
            let start = row * cols;
            let mut max = values[start];
            for &value in &values[start + 1..start + cols] {
                if value > max {
                    max = value;
                }
            }
            out.push(max);
        }
        let raw = RawTensor::from_vec_on(
            self.device().clone(),
            out.clone(),
            Shape::known(dims[..last_axis].to_vec()),
        )?;
        let input_raw = self.raw().clone();
        Tensor::<S::Reduced, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self)],
            move |grad| {
                let seed = grad.host_values()?;
                let mut grad_values = vec![E::ZERO; rows * cols];
                for row in 0..rows {
                    let start = row * cols;
                    let count = values[start..start + cols]
                        .iter()
                        .filter(|&&value| value == out[row])
                        .count();
                    let each = seed[row] / E::from_usize(count);
                    for col in 0..cols {
                        if values[start + col] == out[row] {
                            grad_values[start + col] = each;
                        }
                    }
                }
                Ok(vec![Some(raw_from_vec_like(&input_raw, grad_values)?)])
            },
        )
    }

    pub fn min_last(&self) -> Result<Tensor<S::Reduced, E, B>> {
        const { const_check::known_nonzero(<S::Last as DimSpec>::KNOWN, "min_last", "last axis") };

        let dims = self.shape().dims();
        let last_axis = S::RANK - 1;
        let rows = product(&dims[..last_axis]);
        let cols = dims[last_axis];
        ensure_nonzero_dim("min_last", last_axis, cols)?;
        let values = self.host_values()?.into_owned();
        let mut out = Vec::with_capacity(rows);
        for row in 0..rows {
            let start = row * cols;
            let mut min = values[start];
            for &value in &values[start + 1..start + cols] {
                if value < min {
                    min = value;
                }
            }
            out.push(min);
        }
        let raw = RawTensor::from_vec_on(
            self.device().clone(),
            out.clone(),
            Shape::known(dims[..last_axis].to_vec()),
        )?;
        let input_raw = self.raw().clone();
        Tensor::<S::Reduced, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self)],
            move |grad| {
                let seed = grad.host_values()?;
                let mut grad_values = vec![E::ZERO; rows * cols];
                for row in 0..rows {
                    let start = row * cols;
                    let count = values[start..start + cols]
                        .iter()
                        .filter(|&&value| value == out[row])
                        .count();
                    let each = seed[row] / E::from_usize(count);
                    for col in 0..cols {
                        if values[start + col] == out[row] {
                            grad_values[start + col] = each;
                        }
                    }
                }
                Ok(vec![Some(raw_from_vec_like(&input_raw, grad_values)?)])
            },
        )
    }

    pub fn logsumexp_last(&self) -> Result<Tensor<S::Reduced, E, B>> {
        const { const_check::known_nonzero(<S::Last as DimSpec>::KNOWN, "logsumexp_last", "last axis") };

        let dims = self.shape().dims();
        let last_axis = S::RANK - 1;
        let rows = product(&dims[..last_axis]);
        let cols = dims[last_axis];
        ensure_nonzero_dim("logsumexp_last", last_axis, cols)?;
        let values = self.host_values()?.into_owned();
        let mut softmax = vec![E::ZERO; rows * cols];
        let mut out = Vec::with_capacity(rows);
        for row in 0..rows {
            let start = row * cols;
            let max = row_max(&values[start..start + cols]);
            let mut sum = <E::Acc as DType>::ZERO;
            for col in 0..cols {
                let exp = E::Acc::from_f64((values[start + col] - max).to_f64()).exp();
                sum += exp;
                softmax[start + col] = E::from_f64(exp.to_f64());
            }
            for col in 0..cols {
                softmax[start + col] =
                    E::from_f64((E::Acc::from_f64(softmax[start + col].to_f64()) / sum).to_f64());
            }
            out.push(E::from_f64(
                (E::Acc::from_f64(max.to_f64()) + sum.ln()).to_f64(),
            ));
        }
        let raw = RawTensor::from_vec_on(
            self.device().clone(),
            out,
            Shape::known(dims[..last_axis].to_vec()),
        )?;
        let input_raw = self.raw().clone();
        Tensor::<S::Reduced, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self)],
            move |grad| {
                let seed = grad.host_values()?;
                let mut grad_values = vec![E::ZERO; rows * cols];
                for row in 0..rows {
                    for col in 0..cols {
                        grad_values[row * cols + col] = seed[row] * softmax[row * cols + col];
                    }
                }
                Ok(vec![Some(raw_from_vec_like(&input_raw, grad_values)?)])
            },
        )
    }

    pub fn softmax_last(&self) -> Result<Self> {
        const { const_check::known_nonzero(<S::Last as DimSpec>::KNOWN, "softmax_last", "last axis") };

        let dims = self.shape().dims();
        let last_axis = S::RANK - 1;
        let rows = product(&dims[..last_axis]);
        let cols = dims[last_axis];
        ensure_nonzero_dim("softmax_last", last_axis, cols)?;
        let values = stable_row_softmax(&self.host_values()?, rows, cols);
        let raw =
            RawTensor::from_vec_on(self.device().clone(), values.clone(), self.shape().clone())?;
        let input_raw = self.raw().clone();
        Self::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            let grad = grad.host_values()?;
            let mut out = vec![E::ZERO; rows * cols];
            for row in 0..rows {
                let start = row * cols;
                let dot = (0..cols).fold(<E::Acc as DType>::ZERO, |acc, col| {
                    acc + E::Acc::from_f64((grad[start + col] * values[start + col]).to_f64())
                });
                for col in 0..cols {
                    out[start + col] = E::from_f64(
                        (E::Acc::from_f64(values[start + col].to_f64())
                            * (E::Acc::from_f64(grad[start + col].to_f64()) - dot))
                            .to_f64(),
                    );
                }
            }
            Ok(vec![Some(raw_from_vec_like(&input_raw, out)?)])
        })
    }

    pub fn log_softmax_last(&self) -> Result<Self> {
        const {
            const_check::known_nonzero(<S::Last as DimSpec>::KNOWN, "log_softmax_last", "last axis")
        };

        let dims = self.shape().dims();
        let last_axis = S::RANK - 1;
        let rows = product(&dims[..last_axis]);
        let cols = dims[last_axis];
        ensure_nonzero_dim("log_softmax_last", last_axis, cols)?;
        let input = self.host_values()?;
        let softmax = stable_row_softmax(&input, rows, cols);
        let values = stable_row_log_softmax(&input, rows, cols);
        let raw = RawTensor::from_vec_on(self.device().clone(), values, self.shape().clone())?;
        let input_raw = self.raw().clone();
        Self::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            let grad = grad.host_values()?;
            let mut out = vec![E::ZERO; rows * cols];
            for row in 0..rows {
                let start = row * cols;
                let row_sum = (0..cols).fold(<E::Acc as DType>::ZERO, |acc, col| {
                    acc + E::Acc::from_f64(grad[start + col].to_f64())
                });
                for col in 0..cols {
                    out[start + col] = E::from_f64(
                        (E::Acc::from_f64(grad[start + col].to_f64())
                            - E::Acc::from_f64(softmax[start + col].to_f64()) * row_sum)
                            .to_f64(),
                    );
                }
            }
            Ok(vec![Some(raw_from_vec_like(&input_raw, out)?)])
        })
    }

    fn broadcast_last_axis(
        &self,
        rhs: &Tensor<S::Row, E, B>,
        op: &'static str,
        forward: impl Fn(E, E) -> E + Copy + Send + Sync + 'static,
        lhs_backward: impl Fn(E, E, E) -> E + Copy + Send + Sync + 'static,
        rhs_backward: impl Fn(E, E, E) -> E + Copy + Send + Sync + 'static,
    ) -> Result<Self> {
        ensure_same_device::<E, B>(self.device(), rhs.device(), op)?;
        let dims = self.shape().dims();
        let last_axis = S::RANK - 1;
        let rows = product(&dims[..last_axis]);
        let cols = dims[last_axis];
        let rhs_dims = rhs.shape().dims();
        let rhs_cols = rhs_dims[0];
        bind_and_check(
            op,
            S::dim_entries(0)
                .into_iter()
                .zip(dims.iter().copied())
                .chain(
                    <S::Row as ShapeSpec>::dim_entries(1)
                        .into_iter()
                        .zip(rhs_dims.iter().copied()),
                ),
        )?;
        if rhs_cols != cols {
            return Err(ShapeError::DimMismatch {
                op,
                operand: 1,
                axis: 0,
                expected: cols,
                found: rhs_cols,
            }
            .into());
        }
        let lhs_values = self.host_values()?.into_owned();
        let rhs_values = rhs.host_values()?.into_owned();
        let values = lhs_values
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, value)| forward(value, rhs_values[idx % cols]))
            .collect();
        let raw = RawTensor::from_vec_on(self.device().clone(), values, self.shape().clone())?;
        let lhs_raw = self.raw().clone();
        let rhs_raw = rhs.raw().clone();
        Self::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(rhs)],
            move |grad| {
                let grad_values = grad.host_values()?;
                let mut lhs_grad = Vec::with_capacity(rows * cols);
                let mut rhs_grad = vec![E::ZERO; cols];
                for idx in 0..rows * cols {
                    let col = idx % cols;
                    let g = grad_values[idx];
                    lhs_grad.push(lhs_backward(lhs_values[idx], rhs_values[col], g));
                    rhs_grad[col] += rhs_backward(lhs_values[idx], rhs_values[col], g);
                }
                Ok(vec![
                    Some(raw_from_vec_like(&lhs_raw, lhs_grad)?),
                    Some(raw_from_vec_like(&rhs_raw, rhs_grad)?),
                ])
            },
        )
    }
}

impl<S, E, B> Tensor<S, E, B>
where
    S: LeadingAxis,
    E: FloatDType,
    B: Backend<E>,
{
    pub fn add_leading_dim(&self, rhs: &Tensor<S::Col, E, B>) -> Result<Self> {
        self.broadcast_leading_axis(
            rhs,
            "add_leading_dim",
            |a, b| a + b,
            |_a, _b, g| g,
            |_a, _b, g| g,
        )
    }

    pub fn sub_leading_dim(&self, rhs: &Tensor<S::Col, E, B>) -> Result<Self> {
        self.broadcast_leading_axis(
            rhs,
            "sub_leading_dim",
            |a, b| a - b,
            |_a, _b, g| g,
            |_a, _b, g| -g,
        )
    }

    pub fn mul_leading_dim(&self, rhs: &Tensor<S::Col, E, B>) -> Result<Self> {
        self.broadcast_leading_axis(
            rhs,
            "mul_leading_dim",
            |a, b| a * b,
            |_a, b, g| g * b,
            |a, _b, g| g * a,
        )
    }

    pub fn div_leading_dim(&self, rhs: &Tensor<S::Col, E, B>) -> Result<Self> {
        self.broadcast_leading_axis(
            rhs,
            "div_leading_dim",
            |a, b| a / b,
            |_a, b, g| g / b,
            |a, b, g| -(g * a) / (b * b),
        )
    }

    /// Reduces the leading axis with `keepdim = false`.
    pub fn sum_leading(&self) -> Result<Tensor<S::Reduced, E, B>> {
        let dims = self.shape().dims();
        let leading = dims[0];
        let inner = product(&dims[1..]);
        let values = self.host_values()?;
        let mut out = vec![E::ZERO; inner];
        for row in 0..leading {
            for col in 0..inner {
                out[col] += values[row * inner + col];
            }
        }
        let raw =
            RawTensor::from_vec_on(self.device().clone(), out, Shape::known(dims[1..].to_vec()))?;
        let input_raw = self.raw().clone();
        Tensor::<S::Reduced, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self)],
            move |grad| {
                let seed = grad.host_values()?;
                let mut values = Vec::with_capacity(leading * inner);
                for _ in 0..leading {
                    values.extend(seed.iter().copied());
                }
                Ok(vec![Some(raw_from_vec_like(&input_raw, values)?)])
            },
        )
    }

    pub fn mean_leading(&self) -> Result<Tensor<S::Reduced, E, B>> {
        const {
            const_check::known_nonzero(
                <S::Leading as DimSpec>::KNOWN,
                "mean_leading",
                "leading axis",
            )
        };
        let leading = self.shape().dims()[0];
        ensure_nonzero_dim("mean_leading", 0, leading)?;
        self.sum_leading()?.div_scalar(E::from_usize(leading))
    }

    fn broadcast_leading_axis(
        &self,
        rhs: &Tensor<S::Col, E, B>,
        op: &'static str,
        forward: impl Fn(E, E) -> E + Copy + Send + Sync + 'static,
        lhs_backward: impl Fn(E, E, E) -> E + Copy + Send + Sync + 'static,
        rhs_backward: impl Fn(E, E, E) -> E + Copy + Send + Sync + 'static,
    ) -> Result<Self> {
        ensure_same_device::<E, B>(self.device(), rhs.device(), op)?;
        let dims = self.shape().dims();
        let leading = dims[0];
        let inner = product(&dims[1..]);
        let rhs_dims = rhs.shape().dims();
        let rhs_leading = rhs_dims[0];
        bind_and_check(
            op,
            S::dim_entries(0)
                .into_iter()
                .zip(dims.iter().copied())
                .chain(
                    <S::Col as ShapeSpec>::dim_entries(1)
                        .into_iter()
                        .zip(rhs_dims.iter().copied()),
                ),
        )?;
        if rhs_leading != leading {
            return Err(ShapeError::DimMismatch {
                op,
                operand: 1,
                axis: 0,
                expected: leading,
                found: rhs_leading,
            }
            .into());
        }
        let lhs_values = self.host_values()?.into_owned();
        let rhs_values = rhs.host_values()?.into_owned();
        let values = lhs_values
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, value)| forward(value, rhs_values[idx / inner]))
            .collect();
        let raw = RawTensor::from_vec_on(self.device().clone(), values, self.shape().clone())?;
        let lhs_raw = self.raw().clone();
        let rhs_raw = rhs.raw().clone();
        Self::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(rhs)],
            move |grad| {
                let grad_values = grad.host_values()?;
                let mut lhs_grad = Vec::with_capacity(leading * inner);
                let mut rhs_grad = vec![E::ZERO; leading];
                for idx in 0..leading * inner {
                    let row = idx / inner;
                    let g = grad_values[idx];
                    lhs_grad.push(lhs_backward(lhs_values[idx], rhs_values[row], g));
                    rhs_grad[row] += rhs_backward(lhs_values[idx], rhs_values[row], g);
                }
                Ok(vec![
                    Some(raw_from_vec_like(&lhs_raw, lhs_grad)?),
                    Some(raw_from_vec_like(&rhs_raw, rhs_grad)?),
                ])
            },
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

fn product(dims: &[usize]) -> usize {
    dims.iter().copied().product()
}

fn row_max<E: FloatDType>(values: &[E]) -> E {
    let mut max = values[0];
    for &value in &values[1..] {
        if value > max {
            max = value;
        }
    }
    max
}

fn stable_row_softmax<E: FloatDType>(values: &[E], rows: usize, cols: usize) -> Vec<E> {
    let mut out = vec![E::ZERO; rows * cols];
    parallel::for_each_chunk_mut(&mut out, cols, |row, out_row| {
        let start = row * cols;
        let max = row_max(&values[start..start + cols]);
        let mut sum = <E::Acc as DType>::ZERO;
        for col in 0..cols {
            let exp = E::Acc::from_f64((values[start + col] - max).to_f64()).exp();
            sum += exp;
            out_row[col] = E::from_f64(exp.to_f64());
        }
        for slot in out_row.iter_mut() {
            *slot = E::from_f64((E::Acc::from_f64(slot.to_f64()) / sum).to_f64());
        }
    });
    out
}

fn stable_row_log_softmax<E: FloatDType>(values: &[E], rows: usize, cols: usize) -> Vec<E> {
    let mut out = vec![E::ZERO; rows * cols];
    parallel::for_each_chunk_mut(&mut out, cols, |row, out_row| {
        let start = row * cols;
        let max = row_max(&values[start..start + cols]);
        let sum = (0..cols)
            .map(|col| E::Acc::from_f64((values[start + col] - max).to_f64()).exp())
            .fold(<E::Acc as DType>::ZERO, |acc, value| acc + value);
        let logsumexp = E::Acc::from_f64(max.to_f64()) + sum.ln();
        for (col, slot) in out_row.iter_mut().enumerate() {
            *slot =
                E::from_f64((E::Acc::from_f64(values[start + col].to_f64()) - logsumexp).to_f64());
        }
    });
    out
}
