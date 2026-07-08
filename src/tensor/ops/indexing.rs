use super::super::autograd::{AnyTensor, is_grad_enabled, raw_from_vec_like};
use super::super::{RawTensor, Scalar, Tensor};
use super::{ensure_nonzero_dim, ensure_same_device};
use crate::backend::Backend;
use crate::dtype::{DType, FloatDType};
use crate::error::{DataError, Error, Result, ShapeError, const_check};
use crate::nn::{CrossEntropyOpts, Reduction};
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
                op: "correct_count",
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

    /// Tensor-label sibling of [`cross_entropy`](Self::cross_entropy).
    ///
    /// This path is gated on `B: Backend<i64>` and is intended for CPU id and
    /// label plumbing. The slice-based `usize` path remains the universal
    /// every-backend API. Negative ids are rejected; they do not wrap Python-style.
    pub fn cross_entropy_ids(&self, targets: &Tensor<D1<A>, i64, B>) -> Result<Scalar<E, B>>
    where
        B: Backend<i64, Device = <B as Backend<E>>::Device>,
    {
        self.cross_entropy_ids_with(targets, CrossEntropyOpts::default())
    }

    /// Option-bearing tensor-label sibling of [`cross_entropy_with`](Self::cross_entropy_with).
    ///
    /// `ignore_index` is still a `usize`, so negative ids are invalid rather
    /// than ignored. i64 tensors do not participate in autograd.
    pub fn cross_entropy_ids_with(
        &self,
        targets: &Tensor<D1<A>, i64, B>,
        opts: CrossEntropyOpts,
    ) -> Result<Scalar<E, B>>
    where
        B: Backend<i64, Device = <B as Backend<E>>::Device>,
    {
        ensure_same_device::<E, B>(self.device(), targets.device(), "cross_entropy_ids")?;
        let targets = i64_indices_to_usize(&targets.host_values()?)?;
        self.cross_entropy_with(&targets, opts)
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
                op: "cross_entropy",
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
        let input_tensor = self.contiguous()?;
        let input = input_tensor.host_values()?;
        let softmax = stable_row_softmax(&input, rows, cols);
        let log_probs = stable_row_log_softmax(&input, rows, cols);
        let smoothing = E::from_f64(opts.label_smoothing);
        if smoothing < E::ZERO || smoothing >= E::ONE {
            return Err(Error::InvalidInput {
                op: "cross_entropy",
                reason: "label_smoothing must be in [0, 1)",
            });
        }
        let valid_count = targets
            .iter()
            .filter(|&&target| Some(target) != opts.ignore_index)
            .count();
        let loss_sum = targets
            .iter()
            .enumerate()
            .filter(|&(_, &target)| Some(target) != opts.ignore_index)
            .fold(<E::Acc as DType>::ZERO, |acc, (row, &target)| {
                let row_start = row * cols;
                let nll = E::Acc::from_f64((-log_probs[row_start + target]).to_f64());
                let smooth = -(0..cols)
                    .map(|col| E::Acc::from_f64(log_probs[row_start + col].to_f64()))
                    .fold(<E::Acc as DType>::ZERO, |acc, value| acc + value)
                    / E::Acc::from_usize(cols);
                acc + E::Acc::from_f64((E::ONE - smoothing).to_f64()) * nll
                    + E::Acc::from_f64(smoothing.to_f64()) * smooth
            });
        let scale_acc = match opts.reduction {
            crate::nn::Reduction::Mean => E::Acc::from_usize(valid_count),
            crate::nn::Reduction::Sum => <E::Acc as DType>::ONE,
        };
        let scale = E::from_f64(scale_acc.to_f64());
        let loss = if valid_count == 0 {
            E::ZERO
        } else {
            E::from_f64((loss_sum / scale_acc).to_f64())
        };
        let raw = if let Some(storage) = B::try_cross_entropy(
            input_tensor.device(),
            input_tensor.raw().storage(),
            targets,
            rows,
            cols,
            opts.ignore_index,
            opts.label_smoothing,
            matches!(opts.reduction, Reduction::Mean),
        )
        .map_err(Error::backend)?
        {
            RawTensor::from_storage_on(self.device().clone(), storage, Shape::known([]))?
        } else {
            crate::backend::record_reference_fall("cross_entropy");
            RawTensor::from_vec_on(self.device().clone(), vec![loss], Shape::known([]))?
        };
        let input_raw = self.raw().clone();
        let targets = targets.to_vec();
        Tensor::<D0, E, B>::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            let seed = grad.host_values()?[0];
            let mut values = vec![E::ZERO; rows * cols];
            if valid_count == 0 {
                return Ok(vec![Some(raw_from_vec_like(&input_raw, values)?)]);
            }
            for row in 0..rows {
                if Some(targets[row]) == opts.ignore_index {
                    continue;
                }
                let smooth_target = smoothing / E::from_usize(cols);
                for col in 0..cols {
                    values[row * cols + col] = softmax[row * cols + col] - smooth_target;
                }
                values[row * cols + targets[row]] -= E::ONE - smoothing;
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
        let input = self.host_values()?;
        let values = input[row * cols..(row + 1) * cols].to_vec();
        let raw = RawTensor::from_vec_on(self.device().clone(), values, Shape::known([cols]))?;
        let input_raw = self.raw().clone();
        Tensor::<D1<N>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self)],
            move |grad| {
                let mut values = vec![E::ZERO; rows * cols];
                values[row * cols..(row + 1) * cols].copy_from_slice(&grad.host_values()?);
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
        let input_tensor = self.contiguous()?;
        let raw = if let Some(storage) = B::try_index_select_rows(
            input_tensor.device(),
            input_tensor.raw().storage(),
            indices,
            rows,
            cols,
        )
        .map_err(Error::backend)?
        {
            RawTensor::from_storage_on(
                self.device().clone(),
                storage,
                Shape::known([indices.len(), cols]),
            )?
        } else {
            crate::backend::record_reference_fall("index_select_rows");
            let input = input_tensor.host_values()?;
            let mut values = Vec::with_capacity(indices.len() * cols);
            for &index in indices {
                values.extend_from_slice(&input[index * cols..(index + 1) * cols]);
            }
            RawTensor::from_vec_on(
                self.device().clone(),
                values,
                Shape::known([indices.len(), cols]),
            )?
        };
        if !is_grad_enabled() || !self.requires_grad() {
            return Tensor::<D2<T, N>, E, B>::from_raw_non_leaf(raw);
        }
        let input_raw = self.raw().clone();
        let indices = indices.to_vec();
        Tensor::<D2<T, N>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self)],
            move |grad| {
                let grad = grad.host_values()?;
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

    /// Tensor-index sibling of [`index_select_rows`](Self::index_select_rows).
    ///
    /// This is available only when the backend can store i64 tensors. The
    /// `&[usize]` method remains the universal every-backend path. Negative
    /// indices are rejected; RsTorch does not apply Python-style wrapping.
    pub fn index_select_rows_ids<T, I>(
        &self,
        indices: &Tensor<D1<I>, i64, B>,
    ) -> Result<Tensor<D2<T, N>, E, B>>
    where
        T: DimSpec,
        I: DimSpec,
        B: Backend<i64, Device = <B as Backend<E>>::Device>,
    {
        ensure_same_device::<E, B>(self.device(), indices.device(), "index_select_rows_ids")?;
        let indices = i64_indices_to_usize(&indices.host_values()?)?;
        self.index_select_rows(&indices)
    }
}

fn i64_indices_to_usize(values: &[i64]) -> Result<Vec<usize>> {
    values
        .iter()
        .copied()
        .map(|index| {
            if index < 0 {
                return Err(DataError::NegativeIndex { index }.into());
            }
            usize::try_from(index).map_err(|_| {
                DataError::IndexOutOfBounds {
                    index: usize::MAX,
                    len: usize::MAX,
                }
                .into()
            })
        })
        .collect()
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
        let mut sum = <E::Acc as DType>::ZERO;
        for col in 0..cols {
            let exp = E::Acc::from_f64((values[start + col] - max).to_f64()).exp();
            sum += exp;
            out[start + col] = E::from_f64(exp.to_f64());
        }
        for col in 0..cols {
            out[start + col] =
                E::from_f64((E::Acc::from_f64(out[start + col].to_f64()) / sum).to_f64());
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
            .map(|col| E::Acc::from_f64((values[start + col] - max).to_f64()).exp())
            .fold(<E::Acc as DType>::ZERO, |acc, value| acc + value);
        let logsumexp = E::Acc::from_f64(max.to_f64()) + sum.ln();
        for col in 0..cols {
            out[start + col] =
                E::from_f64((E::Acc::from_f64(values[start + col].to_f64()) - logsumexp).to_f64());
        }
    }
    out
}
