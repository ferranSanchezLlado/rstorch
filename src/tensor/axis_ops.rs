use super::autograd::{AnyTensor, is_grad_enabled, raw_from_vec_like};
use super::{RawTensor, Tensor};
use crate::backend::{Backend, NativeBinaryOp, NativeRowOp, parallel};
use crate::dtype::{DType, FloatDType};
use crate::error::{DeviceError, Error, Result, ShapeError, const_check};
use crate::shape::{D1, D2, DimSpec, LastAxis, LeadingAxis, Shape, ShapeSpec, bind_and_check};

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
            NativeBinaryOp::Add,
        )
    }

    pub fn sub_last_dim(&self, rhs: &Tensor<S::Row, E, B>) -> Result<Self> {
        self.broadcast_last_axis(
            rhs,
            "sub_last_dim",
            |a, b| a - b,
            |_a, _b, g| g,
            |_a, _b, g| -g,
            NativeBinaryOp::Sub,
        )
    }

    pub fn mul_last_dim(&self, rhs: &Tensor<S::Row, E, B>) -> Result<Self> {
        self.broadcast_last_axis(
            rhs,
            "mul_last_dim",
            |a, b| a * b,
            |_a, b, g| g * b,
            |a, _b, g| g * a,
            NativeBinaryOp::Mul,
        )
    }

    pub fn div_last_dim(&self, rhs: &Tensor<S::Row, E, B>) -> Result<Self> {
        self.broadcast_last_axis(
            rhs,
            "div_last_dim",
            |a, b| a / b,
            |_a, b, g| g / b,
            |a, b, g| -(g * a) / (b * b),
            NativeBinaryOp::Div,
        )
    }

    /// Reduces the last axis with `keepdim = false`.
    pub fn sum_last(&self) -> Result<Tensor<S::Reduced, E, B>> {
        let dims = self.shape().dims();
        let last_axis = S::RANK - 1;
        let rows = product(&dims[..last_axis]);
        let cols = dims[last_axis];
        let out_shape = Shape::known(dims[..last_axis].to_vec());
        let input = self.contiguous()?;
        let raw = if let Some(storage) =
            B::try_sum_last(input.device(), input.raw().storage(), rows, cols)
                .map_err(Error::backend)?
        {
            RawTensor::from_storage_on(self.device().clone(), storage, out_shape.clone())?
        } else {
            crate::backend::record_reference_fall("sum_last");
            let values = input.host_values()?;
            let mut out = vec![E::ZERO; rows];
            for row in 0..rows {
                for col in 0..cols {
                    out[row] += values[row * cols + col];
                }
            }
            RawTensor::from_vec_on(self.device().clone(), out, out_shape)?
        };
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
        let input = self.contiguous()?;
        let raw = if let Some(storage) = B::try_row_softmax(
            input.device(),
            input.raw().storage(),
            rows,
            cols,
            NativeRowOp::Softmax,
        )
        .map_err(Error::backend)?
        {
            RawTensor::from_storage_on(self.device().clone(), storage, self.shape().clone())?
        } else {
            crate::backend::record_reference_fall("softmax_last");
            let values = stable_row_softmax(&input.host_values()?, rows, cols);
            RawTensor::from_vec_on(self.device().clone(), values, self.shape().clone())?
        };
        if !is_grad_enabled() || !self.requires_grad() {
            return Self::from_raw_non_leaf(raw);
        }
        let values = raw.to_vec()?;
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
        let input = self.contiguous()?;
        let raw = if let Some(storage) = B::try_row_softmax(
            input.device(),
            input.raw().storage(),
            rows,
            cols,
            NativeRowOp::LogSoftmax,
        )
        .map_err(Error::backend)?
        {
            RawTensor::from_storage_on(self.device().clone(), storage, self.shape().clone())?
        } else {
            crate::backend::record_reference_fall("log_softmax_last");
            let values = stable_row_log_softmax(&input.host_values()?, rows, cols);
            RawTensor::from_vec_on(self.device().clone(), values, self.shape().clone())?
        };
        if !is_grad_enabled() || !self.requires_grad() {
            return Self::from_raw_non_leaf(raw);
        }
        let input_values = input.host_values()?;
        let softmax = stable_row_softmax(&input_values, rows, cols);
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
        native_op: NativeBinaryOp,
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
        let lhs_input = self.contiguous()?;
        let rhs_input = rhs.contiguous()?;
        let raw = if let Some(storage) = B::try_broadcast_last(
            lhs_input.device(),
            lhs_input.raw().storage(),
            rhs_input.raw().storage(),
            rows,
            cols,
            native_op,
        )
        .map_err(Error::backend)?
        {
            RawTensor::from_storage_on(self.device().clone(), storage, self.shape().clone())?
        } else {
            crate::backend::record_reference_fall("broadcast_last");
            let lhs_values = lhs_input.host_values()?;
            let rhs_values = rhs_input.host_values()?;
            let values = lhs_values
                .iter()
                .copied()
                .enumerate()
                .map(|(idx, value)| forward(value, rhs_values[idx % cols]))
                .collect();
            RawTensor::from_vec_on(self.device().clone(), values, self.shape().clone())?
        };
        let lhs_values = lhs_input.host_values()?.into_owned();
        let rhs_values = rhs_input.host_values()?.into_owned();
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
            NativeBinaryOp::Add,
        )
    }

    pub fn sub_leading_dim(&self, rhs: &Tensor<S::Col, E, B>) -> Result<Self> {
        self.broadcast_leading_axis(
            rhs,
            "sub_leading_dim",
            |a, b| a - b,
            |_a, _b, g| g,
            |_a, _b, g| -g,
            NativeBinaryOp::Sub,
        )
    }

    pub fn mul_leading_dim(&self, rhs: &Tensor<S::Col, E, B>) -> Result<Self> {
        self.broadcast_leading_axis(
            rhs,
            "mul_leading_dim",
            |a, b| a * b,
            |_a, b, g| g * b,
            |a, _b, g| g * a,
            NativeBinaryOp::Mul,
        )
    }

    pub fn div_leading_dim(&self, rhs: &Tensor<S::Col, E, B>) -> Result<Self> {
        self.broadcast_leading_axis(
            rhs,
            "div_leading_dim",
            |a, b| a / b,
            |_a, b, g| g / b,
            |a, b, g| -(g * a) / (b * b),
            NativeBinaryOp::Div,
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
        native_op: NativeBinaryOp,
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
        let lhs_input = self.contiguous()?;
        let rhs_input = rhs.contiguous()?;
        let raw = if let Some(storage) = B::try_broadcast_leading(
            lhs_input.device(),
            lhs_input.raw().storage(),
            rhs_input.raw().storage(),
            leading,
            inner,
            native_op,
        )
        .map_err(Error::backend)?
        {
            RawTensor::from_storage_on(self.device().clone(), storage, self.shape().clone())?
        } else {
            crate::backend::record_reference_fall("broadcast_leading");
            let lhs_values = lhs_input.host_values()?;
            let rhs_values = rhs_input.host_values()?;
            let values = lhs_values
                .iter()
                .copied()
                .enumerate()
                .map(|(idx, value)| forward(value, rhs_values[idx / inner]))
                .collect();
            RawTensor::from_vec_on(self.device().clone(), values, self.shape().clone())?
        };
        let lhs_values = lhs_input.host_values()?.into_owned();
        let rhs_values = rhs_input.host_values()?.into_owned();
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

impl<A, N, E, B> Tensor<D2<A, N>, E, B>
where
    A: DimSpec,
    N: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    #[allow(clippy::needless_range_loop)]
    pub fn layer_norm_last(
        &self,
        weight: &Tensor<D1<N>, E, B>,
        bias: &Tensor<D1<N>, E, B>,
        eps: E,
    ) -> Result<Self> {
        ensure_same_device::<E, B>(self.device(), weight.device(), "layer_norm_last")?;
        ensure_same_device::<E, B>(self.device(), bias.device(), "layer_norm_last")?;
        let dims = self.shape().dims();
        let rows = dims[0];
        let cols = dims[1];
        if weight.shape().dims()[0] != cols || bias.shape().dims()[0] != cols {
            return Err(ShapeError::LengthMismatch {
                op: "layer_norm_last",
                expected: cols,
                found: weight.shape().dims()[0].min(bias.shape().dims()[0]),
            }
            .into());
        }

        let input = self.contiguous()?;
        let weight_input = weight.contiguous()?;
        let bias_input = bias.contiguous()?;
        let input_values = input.host_values()?.into_owned();
        let weight_values = weight_input.host_values()?.into_owned();
        let bias_values = bias_input.host_values()?.into_owned();
        let (x_hat, inv_std, reference_values) = layer_norm_values(
            &input_values,
            &weight_values,
            Some(&bias_values),
            rows,
            cols,
            eps.to_f64(),
            false,
        );
        let raw = if let Some(storage) = B::try_layer_norm(
            input.device(),
            input.raw().storage(),
            weight_input.raw().storage(),
            bias_input.raw().storage(),
            rows,
            cols,
            eps.to_f64(),
        )
        .map_err(Error::backend)?
        {
            RawTensor::from_storage_on(self.device().clone(), storage, self.shape().clone())?
        } else {
            crate::backend::record_reference_fall("layer_norm_last");
            RawTensor::from_vec_on(
                self.device().clone(),
                reference_values,
                self.shape().clone(),
            )?
        };
        let input_raw = self.raw().clone();
        let weight_raw = weight.raw().clone();
        let bias_raw = bias.raw().clone();
        Self::autograd_output(
            raw,
            vec![
                AnyTensor::from_shape(self),
                AnyTensor::from_shape(weight),
                AnyTensor::from_shape(bias),
            ],
            move |grad| {
                let grad = grad.host_values()?;
                let mut dx = vec![E::ZERO; rows * cols];
                let mut dweight = vec![<E::Acc as DType>::ZERO; cols];
                let mut dbias = vec![<E::Acc as DType>::ZERO; cols];
                for row in 0..rows {
                    let start = row * cols;
                    let mut sum_dxhat = 0.0;
                    let mut sum_dxhat_xhat = 0.0;
                    for col in 0..cols {
                        let idx = start + col;
                        let g = grad[idx].to_f64();
                        let xh = x_hat[idx];
                        let dxh = g * weight_values[col].to_f64();
                        sum_dxhat += dxh;
                        sum_dxhat_xhat += dxh * xh;
                        dweight[col] += E::Acc::from_f64(g * xh);
                        dbias[col] += E::Acc::from_f64(g);
                    }
                    let denom = cols as f64;
                    for col in 0..cols {
                        let idx = start + col;
                        let dxh = grad[idx].to_f64() * weight_values[col].to_f64();
                        dx[idx] = E::from_f64(
                            inv_std[row]
                                * (dxh - sum_dxhat / denom - x_hat[idx] * sum_dxhat_xhat / denom),
                        );
                    }
                }
                Ok(vec![
                    Some(raw_from_vec_like(&input_raw, dx)?),
                    Some(raw_from_vec_like(
                        &weight_raw,
                        dweight
                            .into_iter()
                            .map(|v| E::from_f64(v.to_f64()))
                            .collect(),
                    )?),
                    Some(raw_from_vec_like(
                        &bias_raw,
                        dbias.into_iter().map(|v| E::from_f64(v.to_f64())).collect(),
                    )?),
                ])
            },
        )
    }

    #[allow(clippy::needless_range_loop)]
    pub fn rms_norm_last(&self, weight: &Tensor<D1<N>, E, B>, eps: E) -> Result<Self> {
        ensure_same_device::<E, B>(self.device(), weight.device(), "rms_norm_last")?;
        let dims = self.shape().dims();
        let rows = dims[0];
        let cols = dims[1];
        if weight.shape().dims()[0] != cols {
            return Err(ShapeError::LengthMismatch {
                op: "rms_norm_last",
                expected: cols,
                found: weight.shape().dims()[0],
            }
            .into());
        }

        let input = self.contiguous()?;
        let weight_input = weight.contiguous()?;
        let input_values = input.host_values()?.into_owned();
        let weight_values = weight_input.host_values()?.into_owned();
        let (scaled_input, inv_rms, reference_values) = layer_norm_values(
            &input_values,
            &weight_values,
            None,
            rows,
            cols,
            eps.to_f64(),
            true,
        );
        let raw = if let Some(storage) = B::try_rms_norm(
            input.device(),
            input.raw().storage(),
            weight_input.raw().storage(),
            rows,
            cols,
            eps.to_f64(),
        )
        .map_err(Error::backend)?
        {
            RawTensor::from_storage_on(self.device().clone(), storage, self.shape().clone())?
        } else {
            crate::backend::record_reference_fall("rms_norm_last");
            RawTensor::from_vec_on(
                self.device().clone(),
                reference_values,
                self.shape().clone(),
            )?
        };
        let input_raw = self.raw().clone();
        let weight_raw = weight.raw().clone();
        Self::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(weight)],
            move |grad| {
                let grad = grad.host_values()?;
                let mut dx = vec![E::ZERO; rows * cols];
                let mut dweight = vec![<E::Acc as DType>::ZERO; cols];
                for row in 0..rows {
                    let start = row * cols;
                    let mut dot = 0.0;
                    for col in 0..cols {
                        let idx = start + col;
                        let g = grad[idx].to_f64();
                        dweight[col] += E::Acc::from_f64(g * scaled_input[idx]);
                        dot += g * weight_values[col].to_f64() * input_values[idx].to_f64();
                    }
                    let coeff = inv_rms[row] * inv_rms[row] * inv_rms[row] * dot / cols as f64;
                    for col in 0..cols {
                        let idx = start + col;
                        let dxhat = grad[idx].to_f64() * weight_values[col].to_f64();
                        dx[idx] =
                            E::from_f64(dxhat * inv_rms[row] - input_values[idx].to_f64() * coeff);
                    }
                }
                Ok(vec![
                    Some(raw_from_vec_like(&input_raw, dx)?),
                    Some(raw_from_vec_like(
                        &weight_raw,
                        dweight
                            .into_iter()
                            .map(|v| E::from_f64(v.to_f64()))
                            .collect(),
                    )?),
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

#[allow(clippy::needless_range_loop)]
fn layer_norm_values<E: FloatDType>(
    input: &[E],
    weight: &[E],
    bias: Option<&[E]>,
    rows: usize,
    cols: usize,
    eps: f64,
    rms_only: bool,
) -> (Vec<f64>, Vec<f64>, Vec<E>) {
    let mut normalized = vec![0.0; rows * cols];
    let mut inv = vec![0.0; rows];
    let mut out = vec![E::ZERO; rows * cols];
    for row in 0..rows {
        let start = row * cols;
        let mean = if rms_only {
            0.0
        } else {
            input[start..start + cols]
                .iter()
                .map(|value| value.to_f64())
                .sum::<f64>()
                / cols as f64
        };
        let variance = input[start..start + cols]
            .iter()
            .map(|value| {
                let diff = value.to_f64() - mean;
                diff * diff
            })
            .sum::<f64>()
            / cols as f64;
        inv[row] = 1.0 / (variance + eps).sqrt();
        for col in 0..cols {
            let idx = start + col;
            normalized[idx] = (input[idx].to_f64() - mean) * inv[row];
            let bias = bias.map_or(0.0, |bias| bias[col].to_f64());
            out[idx] = E::from_f64(normalized[idx] * weight[col].to_f64() + bias);
        }
    }
    (normalized, inv, out)
}
