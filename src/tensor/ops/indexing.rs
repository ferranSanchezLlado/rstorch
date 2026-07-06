use super::super::autograd::{AnyTensor, raw_from_vec_like};
use super::super::{RawTensor, Scalar, Tensor};
use super::ensure_nonzero_dim;
use crate::backend::Backend;
use crate::dtype::{DType, FloatDType};
use crate::error::{DataError, Error, Result, ShapeError, const_check};
use crate::nn::CrossEntropyOpts;
use crate::shape::{D0, D1, D2, DimEntry, DimSpec, Shape, bind_and_check};

impl<A, K, E, B> Tensor<D2<A, K>, E, B>
where
    A: DimSpec,
    K: DimSpec,
    E: DType,
    B: Backend<E>,
{
    /// Counts predictions matching `targets` after [`Self::argmax_last`].
    pub fn correct_count(&self, targets: &[usize]) -> Result<usize> {
        let dims = self.shape().dims();
        let rows = dims[0];
        let cols = dims[1];
        if targets.len() != rows {
            return Err(ShapeError::LengthMismatch {
                expected: rows,
                found: targets.len(),
            }
            .into());
        }
        for &target in targets {
            if target >= cols {
                return Err(DataError::IndexOutOfBounds {
                    index: target,
                    len: cols,
                }
                .into());
            }
        }

        Ok(self
            .argmax_last()?
            .into_iter()
            .zip(targets.iter().copied())
            .filter(|(predicted, target)| predicted == target)
            .count())
    }

    /// Returns classification accuracy over `targets` after [`Self::argmax_last`].
    pub fn accuracy(&self, targets: &[usize]) -> Result<f64> {
        let rows = self.shape().dims()[0];
        let correct = self.correct_count(targets)?;
        if rows == 0 {
            return Err(Error::InvalidInput {
                op: "accuracy",
                reason: "target set must not be empty",
            });
        }
        Ok(correct as f64 / rows as f64)
    }
}

impl<A, N, E, B> Tensor<D2<A, N>, E, B>
where
    A: DimSpec,
    N: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    pub fn cross_entropy(&self, targets: &[usize]) -> Result<Scalar<E, B>> {
        self.cross_entropy_with(targets, CrossEntropyOpts::default())
    }

    pub fn cross_entropy_with(
        &self,
        targets: &[usize],
        opts: CrossEntropyOpts,
    ) -> Result<Scalar<E, B>> {
        const { const_check::known_nonzero(N::KNOWN, "cross_entropy", "axis 1") };

        let dims = self.shape().dims();
        let rows = dims[0];
        let cols = dims[1];
        ensure_nonzero_dim("cross_entropy", 1, cols)?;
        if targets.len() != rows {
            return Err(ShapeError::LengthMismatch {
                expected: rows,
                found: targets.len(),
            }
            .into());
        }
        for &target in targets {
            if Some(target) == opts.ignore_index {
                continue;
            }
            if target >= cols {
                return Err(DataError::IndexOutOfBounds {
                    index: target,
                    len: cols,
                }
                .into());
            }
        }
        let input = self.to_vec()?;
        let softmax = stable_row_softmax(&input, rows, cols);
        let log_probs = stable_row_log_softmax(&input, rows, cols);
        let valid_count = targets
            .iter()
            .filter(|&&target| Some(target) != opts.ignore_index)
            .count();
        let loss_sum = targets
            .iter()
            .enumerate()
            .filter(|&(_, &target)| Some(target) != opts.ignore_index)
            .fold(E::ZERO, |acc, (row, &target)| {
                acc - log_probs[row * cols + target]
            });
        let scale = match opts.reduction {
            crate::nn::Reduction::Mean => E::from_usize(valid_count),
            crate::nn::Reduction::Sum => E::ONE,
        };
        let loss = if valid_count == 0 {
            E::ZERO
        } else {
            loss_sum / scale
        };
        let raw = RawTensor::from_vec_on(self.device().clone(), vec![loss], Shape::known([]))?;
        let input_raw = self.raw().clone();
        let targets = targets.to_vec();
        Tensor::<D0, E, B>::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            let seed = grad.to_vec()?[0];
            let mut values = vec![E::ZERO; rows * cols];
            if valid_count == 0 {
                return Ok(vec![Some(raw_from_vec_like(&input_raw, values)?)]);
            }
            for row in 0..rows {
                if Some(targets[row]) == opts.ignore_index {
                    continue;
                }
                for col in 0..cols {
                    values[row * cols + col] = softmax[row * cols + col];
                }
            }
            for row in 0..rows {
                if Some(targets[row]) == opts.ignore_index {
                    continue;
                }
                values[row * cols + targets[row]] -= E::ONE;
            }
            for value in &mut values {
                *value = *value * seed / scale;
            }
            Ok(vec![Some(raw_from_vec_like(&input_raw, values)?)])
        })
    }

    pub fn select_row(&self, row: usize) -> Result<Tensor<D1<N>, E, B>> {
        let dims = self.shape().dims();
        let rows = dims[0];
        let cols = dims[1];
        if row >= rows {
            return Err(DataError::IndexOutOfBounds {
                index: row,
                len: rows,
            }
            .into());
        }
        let values = self.to_vec()?[row * cols..(row + 1) * cols].to_vec();
        let raw = RawTensor::from_vec_on(self.device().clone(), values, Shape::known([cols]))?;
        let input_raw = self.raw().clone();
        Tensor::<D1<N>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self)],
            move |grad| {
                let mut values = vec![E::ZERO; rows * cols];
                values[row * cols..(row + 1) * cols].copy_from_slice(&grad.to_vec()?);
                Ok(vec![Some(raw_from_vec_like(&input_raw, values)?)])
            },
        )
    }

    pub fn index_select_rows<T>(&self, indices: &[usize]) -> Result<Tensor<D2<T, N>, E, B>>
    where
        T: DimSpec,
    {
        let dims = self.shape().dims();
        let rows = dims[0];
        let cols = dims[1];
        bind_and_check(
            "index_select_rows",
            [(DimEntry::of::<T>(0, 0), indices.len())],
        )?;
        for &index in indices {
            if index >= rows {
                return Err(DataError::IndexOutOfBounds { index, len: rows }.into());
            }
        }
        let input = self.to_vec()?;
        let mut values = Vec::with_capacity(indices.len() * cols);
        for &index in indices {
            values.extend_from_slice(&input[index * cols..(index + 1) * cols]);
        }
        let raw = RawTensor::from_vec_on(
            self.device().clone(),
            values,
            Shape::known([indices.len(), cols]),
        )?;
        let input_raw = self.raw().clone();
        let indices = indices.to_vec();
        Tensor::<D2<T, N>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self)],
            move |grad| {
                let grad = grad.to_vec()?;
                let mut values = vec![E::ZERO; rows * cols];
                for (out_row, &source_row) in indices.iter().enumerate() {
                    for col in 0..cols {
                        values[source_row * cols + col] += grad[out_row * cols + col];
                    }
                }
                Ok(vec![Some(raw_from_vec_like(&input_raw, values)?)])
            },
        )
    }
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
    for row in 0..rows {
        let start = row * cols;
        let max = row_max(&values[start..start + cols]);
        let mut sum = E::ZERO;
        for col in 0..cols {
            let exp = (values[start + col] - max).exp();
            sum += exp;
            out[start + col] = exp;
        }
        for col in 0..cols {
            out[start + col] /= sum;
        }
    }
    out
}

fn stable_row_log_softmax<E: FloatDType>(values: &[E], rows: usize, cols: usize) -> Vec<E> {
    let mut out = vec![E::ZERO; rows * cols];
    for row in 0..rows {
        let start = row * cols;
        let max = row_max(&values[start..start + cols]);
        let sum = (0..cols)
            .map(|col| (values[start + col] - max).exp())
            .fold(E::ZERO, |acc, value| acc + value);
        let logsumexp = max + sum.ln();
        for col in 0..cols {
            out[start + col] = values[start + col] - logsumexp;
        }
    }
    out
}
