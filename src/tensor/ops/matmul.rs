use super::super::autograd::{AnyTensor, raw_from_vec_like};
use super::super::{RawTensor, Tensor};
use super::ensure_same_device;
use crate::backend::Backend;
use crate::dtype::{DType, FloatDType};
use crate::error::{Error, Result, ShapeError};
use crate::shape::{D2, D3, DimEntry, DimSpec, Shape, bind_and_check};

impl<A, K, E, B> Tensor<D2<A, K>, E, B>
where
    A: DimSpec,
    K: DimSpec,
    E: DType,
    B: Backend<E>,
{
    pub fn transpose(&self) -> Result<Tensor<D2<K, A>, E, B>> {
        let raw = self.raw().view_with_layout(self.layout().transpose2()?)?;
        Tensor::<D2<K, A>, E, B>::autograd_output(raw, vec![AnyTensor::from_shape(self)], |grad| {
            Ok(vec![Some(
                grad.view_with_layout(grad.layout().transpose2()?)?,
            )])
        })
    }
}

impl<A, K, E, B> Tensor<D2<A, K>, E, B>
where
    A: DimSpec,
    K: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    pub fn matmul<Cc>(&self, rhs: &Tensor<D2<K, Cc>, E, B>) -> Result<Tensor<D2<A, Cc>, E, B>>
    where
        Cc: DimSpec,
    {
        ensure_same_device::<E, B>(self.device(), rhs.device(), "matmul")?;

        let lhs_dims = self.shape().dims();
        let rhs_dims = rhs.shape().dims();
        let m = lhs_dims[0];
        let k = lhs_dims[1];
        let rhs_k = rhs_dims[0];
        let n = rhs_dims[1];

        bind_and_check(
            "matmul",
            [
                (DimEntry::of::<A>(0, 0), m),
                (DimEntry::of::<K>(0, 1), k),
                (DimEntry::of::<K>(1, 0), rhs_k),
                (DimEntry::of::<Cc>(1, 1), n),
            ],
        )?;

        if k != rhs_k {
            return Err(ShapeError::DimMismatch {
                op: "matmul",
                operand: 1,
                axis: 0,
                expected: k,
                found: rhs_k,
            }
            .into());
        }

        let lhs_input = self.contiguous()?;
        let rhs_input = rhs.contiguous()?;
        let storage = B::matmul(
            lhs_input.device(),
            lhs_input.raw().storage(),
            rhs_input.raw().storage(),
            m,
            k,
            n,
        )
        .map_err(Error::backend)?;
        let raw = RawTensor::from_storage_on(self.device().clone(), storage, Shape::known([m, n]))?;
        let lhs_raw = self.raw().clone();
        let rhs_raw = rhs.raw().clone();
        Tensor::<D2<A, Cc>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(rhs)],
            move |grad| {
                let rhs_t = rhs_raw.view_with_layout(rhs_raw.layout().transpose2()?)?;
                let lhs_t = lhs_raw.view_with_layout(lhs_raw.layout().transpose2()?)?;
                Ok(vec![
                    Some(raw_matmul(grad, &rhs_t)?),
                    Some(raw_matmul(&lhs_t, grad)?),
                ])
            },
        )
    }
}

impl<Batch, M, K, E, B> Tensor<D3<Batch, M, K>, E, B>
where
    Batch: DimSpec,
    M: DimSpec,
    K: DimSpec,
    E: DType,
    B: Backend<E>,
{
    pub fn transpose_last2(&self) -> Result<Tensor<D3<Batch, K, M>, E, B>> {
        let raw = self
            .raw()
            .view_with_layout(self.layout().transpose_axes(1, 2)?)?;
        Tensor::<D3<Batch, K, M>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self)],
            |grad| {
                Ok(vec![Some(
                    grad.view_with_layout(grad.layout().transpose_axes(1, 2)?)?,
                )])
            },
        )
    }
}

impl<Batch, M, K, E, B> Tensor<D3<Batch, M, K>, E, B>
where
    Batch: DimSpec,
    M: DimSpec,
    K: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    pub fn bmm<N>(
        &self,
        rhs: &Tensor<D3<Batch, K, N>, E, B>,
    ) -> Result<Tensor<D3<Batch, M, N>, E, B>>
    where
        N: DimSpec,
    {
        ensure_same_device::<E, B>(self.device(), rhs.device(), "bmm")?;
        let lhs_dims = self.shape().dims();
        let rhs_dims = rhs.shape().dims();
        let batch = lhs_dims[0];
        let m = lhs_dims[1];
        let k = lhs_dims[2];
        let rhs_batch = rhs_dims[0];
        let rhs_k = rhs_dims[1];
        let n = rhs_dims[2];
        bind_and_check(
            "bmm",
            [
                (DimEntry::of::<Batch>(0, 0), batch),
                (DimEntry::of::<M>(0, 1), m),
                (DimEntry::of::<K>(0, 2), k),
                (DimEntry::of::<Batch>(1, 0), rhs_batch),
                (DimEntry::of::<K>(1, 1), rhs_k),
                (DimEntry::of::<N>(1, 2), n),
            ],
        )?;
        if batch != rhs_batch {
            return Err(ShapeError::DimMismatch {
                op: "bmm",
                operand: 1,
                axis: 0,
                expected: batch,
                found: rhs_batch,
            }
            .into());
        }
        if k != rhs_k {
            return Err(ShapeError::DimMismatch {
                op: "bmm",
                operand: 1,
                axis: 1,
                expected: k,
                found: rhs_k,
            }
            .into());
        }
        let lhs_values = self.host_values()?.into_owned();
        let rhs_values = rhs.host_values()?.into_owned();
        let values = bmm_values(&lhs_values, &rhs_values, batch, m, k, n);
        let raw =
            RawTensor::from_vec_on(self.device().clone(), values, Shape::known([batch, m, n]))?;
        let lhs_raw = self.raw().clone();
        let rhs_raw = rhs.raw().clone();
        Tensor::<D3<Batch, M, N>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(rhs)],
            move |grad| {
                let grad = grad.host_values()?;
                let mut lhs_grad = vec![E::ZERO; batch * m * k];
                let mut rhs_grad = vec![E::ZERO; batch * k * n];
                for b in 0..batch {
                    let lhs_base = b * m * k;
                    let rhs_base = b * k * n;
                    let grad_base = b * m * n;
                    for i in 0..m {
                        for kk in 0..k {
                            let mut acc = <E::Acc as DType>::ZERO;
                            for j in 0..n {
                                acc += E::Acc::from_f64(
                                    (grad[grad_base + i * n + j]
                                        * rhs_values[rhs_base + kk * n + j])
                                        .to_f64(),
                                );
                            }
                            lhs_grad[lhs_base + i * k + kk] = E::from_f64(acc.to_f64());
                        }
                    }
                    for kk in 0..k {
                        for j in 0..n {
                            let mut acc = <E::Acc as DType>::ZERO;
                            for i in 0..m {
                                acc += E::Acc::from_f64(
                                    (lhs_values[lhs_base + i * k + kk]
                                        * grad[grad_base + i * n + j])
                                        .to_f64(),
                                );
                            }
                            rhs_grad[rhs_base + kk * n + j] = E::from_f64(acc.to_f64());
                        }
                    }
                }
                Ok(vec![
                    Some(raw_from_vec_like(&lhs_raw, lhs_grad)?),
                    Some(raw_from_vec_like(&rhs_raw, rhs_grad)?),
                ])
            },
        )
    }
}

fn raw_matmul<E, B>(lhs: &RawTensor<E, B>, rhs: &RawTensor<E, B>) -> Result<RawTensor<E, B>>
where
    E: FloatDType,
    B: Backend<E>,
{
    // The backend kernel expects operand storage of exactly `m * k` and
    // `k * n` elements in row-major order; operands that already satisfy that
    // (typically incoming gradients) share storage instead of round-tripping
    // through the host.
    let materialize = |input: &RawTensor<E, B>| -> Result<RawTensor<E, B>> {
        if input.is_contiguous() && B::storage_len(input.storage()) == input.numel() {
            return Ok(input.clone());
        }
        RawTensor::from_vec_on(
            input.device().clone(),
            input.host_values()?.into_owned(),
            input.shape().clone(),
        )
    };
    let lhs_input = materialize(lhs)?;
    let rhs_input = materialize(rhs)?;
    let m = lhs.shape().dims()[0];
    let k = lhs.shape().dims()[1];
    let n = rhs.shape().dims()[1];
    let storage = B::matmul(
        lhs.device(),
        lhs_input.storage(),
        rhs_input.storage(),
        m,
        k,
        n,
    )
    .map_err(Error::backend)?;
    RawTensor::from_storage_on(lhs.device().clone(), storage, Shape::known([m, n]))
}

fn bmm_values<E: FloatDType>(
    lhs: &[E],
    rhs: &[E],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<E> {
    let mut out = vec![E::ZERO; batch * m * n];
    for b in 0..batch {
        let lhs_base = b * m * k;
        let rhs_base = b * k * n;
        let out_base = b * m * n;
        for i in 0..m {
            for j in 0..n {
                let mut acc = <E::Acc as DType>::ZERO;
                for kk in 0..k {
                    acc += E::Acc::from_f64(
                        (lhs[lhs_base + i * k + kk] * rhs[rhs_base + kk * n + j]).to_f64(),
                    );
                }
                out[out_base + i * n + j] = E::from_f64(acc.to_f64());
            }
        }
    }
    out
}
