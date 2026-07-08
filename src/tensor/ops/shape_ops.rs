use super::super::autograd::{AnyTensor, raw_from_vec_like};
use super::super::{RawTensor, Tensor, Tensor1D, Tensor4D};
use super::ensure_same_device;
use crate::backend::Backend;
use crate::dtype::DType;
use crate::error::{Result, ShapeError, const_check};
use crate::shape::{
    C, D0, D1, D2, D3, D4, DimEntry, DimSpec, Shape, ShapeSpec, StaticShape, bind_and_check,
};

impl<S, E, B> Tensor<S, E, B>
where
    S: ShapeSpec,
    E: DType,
    B: Backend<E>,
{
    pub fn reshape<T>(&self) -> Result<Tensor<T, E, B>>
    where
        T: StaticShape,
    {
        const {
            const_check::known_size_eq(
                S::KNOWN_NUMEL,
                T::KNOWN_NUMEL,
                "reshape",
                "source elements",
                "target elements",
            );
        };

        let shape = T::static_shape();
        let expected = shape.numel()?;
        if expected != self.numel() {
            return Err(ShapeError::LengthMismatch {
                op: "reshape",
                expected,
                found: self.numel(),
            }
            .into());
        }

        let source = self.contiguous()?;
        let layout = source.raw().layout().reshape_contiguous(shape)?;
        let raw = source.raw().view_with_layout(layout)?;
        let input_raw = self.raw().clone();
        Tensor::<T, E, B>::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            let grad = RawTensor::from_vec_on(
                grad.device().clone(),
                grad.host_values()?.into_owned(),
                grad.shape().clone(),
            )?;
            let layout = grad
                .layout()
                .reshape_contiguous(input_raw.shape().clone())?;
            Ok(vec![Some(grad.view_with_layout(layout)?)])
        })
    }

    pub fn reshape_with_shape<T>(&self, shape: impl Into<Shape>) -> Result<Tensor<T, E, B>>
    where
        T: ShapeSpec,
    {
        const {
            const_check::known_size_eq(
                S::KNOWN_NUMEL,
                T::KNOWN_NUMEL,
                "reshape_with_shape",
                "source elements",
                "target elements",
            );
        };

        let shape = shape.into();
        let expected = shape.numel()?;
        if expected != self.numel() {
            return Err(ShapeError::LengthMismatch {
                op: "reshape_with_shape",
                expected,
                found: self.numel(),
            }
            .into());
        }

        let source = self.contiguous()?;
        let layout = source.raw().layout().reshape_contiguous(shape)?;
        let raw = source.raw().view_with_layout(layout)?;
        let input_raw = self.raw().clone();
        Tensor::<T, E, B>::autograd_output(raw, vec![AnyTensor::from_shape(self)], move |grad| {
            let grad = RawTensor::from_vec_on(
                grad.device().clone(),
                grad.host_values()?.into_owned(),
                grad.shape().clone(),
            )?;
            let layout = grad
                .layout()
                .reshape_contiguous(input_raw.shape().clone())?;
            Ok(vec![Some(grad.view_with_layout(layout)?)])
        })
    }

    pub fn reshape0(&self) -> Result<Tensor<D0, E, B>> {
        self.reshape::<D0>()
    }

    pub fn reshape1<const N: usize>(&self) -> Result<Tensor<D1<C<N>>, E, B>> {
        self.reshape::<D1<C<N>>>()
    }

    pub fn reshape2<const M: usize, const N: usize>(&self) -> Result<Tensor<D2<C<M>, C<N>>, E, B>> {
        self.reshape::<D2<C<M>, C<N>>>()
    }

    pub fn reshape3<const X: usize, const Y: usize, const Z: usize>(
        &self,
    ) -> Result<Tensor<D3<C<X>, C<Y>, C<Z>>, E, B>> {
        self.reshape::<D3<C<X>, C<Y>, C<Z>>>()
    }

    pub fn reshape4<const N: usize, const CH: usize, const H: usize, const W: usize>(
        &self,
    ) -> Result<Tensor4D<N, CH, H, W, E, B>> {
        self.reshape::<D4<C<N>, C<CH>, C<H>, C<W>>>()
    }

    pub fn flatten<const N: usize>(&self) -> Result<Tensor<D1<C<N>>, E, B>> {
        self.reshape1::<N>()
    }
}

impl<A, E, B> Tensor<D1<A>, E, B>
where
    A: DimSpec,
    E: DType,
    B: Backend<E>,
{
    pub fn cat<R, const N: usize>(&self, rhs: &Tensor<D1<R>, E, B>) -> Result<Tensor1D<N, E, B>>
    where
        R: DimSpec,
    {
        const {
            const_check::known_sum_eq(
                A::KNOWN,
                R::KNOWN,
                N,
                "cat",
                "lhs length",
                "rhs length",
                "output length",
            );
        };

        ensure_same_device::<E, B>(self.device(), rhs.device(), "cat")?;
        let found =
            self.numel()
                .checked_add(rhs.numel())
                .ok_or_else(|| ShapeError::NumelOverflow {
                    dims: vec![self.numel(), rhs.numel()].into_boxed_slice(),
                })?;
        if found != N {
            return Err(ShapeError::LengthMismatch {
                op: "cat",
                expected: N,
                found,
            }
            .into());
        }

        let mut data = self.host_values()?.into_owned();
        data.extend(rhs.host_values()?.iter().copied());
        let raw = RawTensor::from_vec_on(self.device().clone(), data, Shape::known([N]))?;
        let left_len = self.numel();
        let lhs_shape = self.shape().clone();
        let rhs_shape = rhs.shape().clone();
        Tensor::<D1<C<N>>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(rhs)],
            move |grad| {
                let data = grad.host_values()?;
                let lhs = RawTensor::from_vec_on(
                    grad.device().clone(),
                    data[..left_len].to_vec(),
                    lhs_shape.clone(),
                )?;
                let rhs = RawTensor::from_vec_on(
                    grad.device().clone(),
                    data[left_len..].to_vec(),
                    rhs_shape.clone(),
                )?;
                Ok(vec![Some(lhs), Some(rhs)])
            },
        )
    }

    /// Stacks two 1-D tensors along a new leading axis of length 2.
    pub fn stack<R>(&self, rhs: &Tensor<D1<R>, E, B>) -> Result<Tensor<D2<C<2>, A>, E, B>>
    where
        R: DimSpec,
    {
        ensure_same_device::<E, B>(self.device(), rhs.device(), "stack")?;
        let len = self.shape().dims()[0];
        let rhs_len = rhs.shape().dims()[0];
        bind_and_check(
            "stack",
            [
                (DimEntry::of::<A>(0, 0), len),
                (DimEntry::of::<R>(1, 0), rhs_len),
            ],
        )?;
        if len != rhs_len {
            return Err(ShapeError::DimMismatch {
                op: "stack",
                operand: 1,
                axis: 0,
                expected: len,
                found: rhs_len,
            }
            .into());
        }
        let mut values = self.host_values()?.into_owned();
        values.extend(rhs.host_values()?.iter().copied());
        let raw = RawTensor::from_vec_on(self.device().clone(), values, Shape::known([2, len]))?;
        let lhs_raw = self.raw().clone();
        let rhs_raw = rhs.raw().clone();
        Tensor::<D2<C<2>, A>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(rhs)],
            move |grad| {
                let values = grad.host_values()?;
                Ok(vec![
                    Some(raw_from_vec_like(&lhs_raw, values[..len].to_vec())?),
                    Some(raw_from_vec_like(&rhs_raw, values[len..].to_vec())?),
                ])
            },
        )
    }

    pub fn unsqueeze_leading(&self) -> Result<Tensor<D2<C<1>, A>, E, B>> {
        self.reshape_with_shape::<D2<C<1>, A>>([1, self.shape().dims()[0]])
    }

    pub fn unsqueeze_last(&self) -> Result<Tensor<D2<A, C<1>>, E, B>> {
        self.reshape_with_shape::<D2<A, C<1>>>([self.shape().dims()[0], 1])
    }
}

impl<N, E, B> Tensor<D2<C<1>, N>, E, B>
where
    N: DimSpec,
    E: DType,
    B: Backend<E>,
{
    pub fn squeeze_leading(&self) -> Result<Tensor<D1<N>, E, B>> {
        self.reshape_with_shape::<D1<N>>([self.shape().dims()[1]])
    }
}

impl<A, N, E, B> Tensor<D2<A, N>, E, B>
where
    A: DimSpec,
    N: DimSpec,
    E: DType,
    B: Backend<E>,
{
    pub fn cat_leading<R, const OUT: usize>(
        &self,
        rhs: &Tensor<D2<R, N>, E, B>,
    ) -> Result<Tensor<D2<C<OUT>, N>, E, B>>
    where
        R: DimSpec,
    {
        const {
            const_check::known_sum_eq(
                A::KNOWN,
                R::KNOWN,
                OUT,
                "cat_leading",
                "lhs rows",
                "rhs rows",
                "output rows",
            );
        };

        ensure_same_device::<E, B>(self.device(), rhs.device(), "cat_leading")?;
        let lhs_dims = self.shape().dims();
        let rhs_dims = rhs.shape().dims();
        let cols = lhs_dims[1];
        bind_and_check(
            "cat_leading",
            [
                (DimEntry::of::<A>(0, 0), lhs_dims[0]),
                (DimEntry::of::<N>(0, 1), lhs_dims[1]),
                (DimEntry::of::<R>(1, 0), rhs_dims[0]),
                (DimEntry::of::<N>(1, 1), rhs_dims[1]),
            ],
        )?;
        if rhs_dims[1] != cols {
            return Err(ShapeError::DimMismatch {
                op: "cat_leading",
                operand: 1,
                axis: 1,
                expected: cols,
                found: rhs_dims[1],
            }
            .into());
        }
        let found = lhs_dims[0] + rhs_dims[0];
        if found != OUT {
            return Err(ShapeError::LengthMismatch {
                op: "cat_leading",
                expected: OUT,
                found,
            }
            .into());
        }
        let mut values = self.host_values()?.into_owned();
        values.extend(rhs.host_values()?.iter().copied());
        let raw = RawTensor::from_vec_on(self.device().clone(), values, Shape::known([OUT, cols]))?;
        let lhs_len = self.numel();
        let lhs_raw = self.raw().clone();
        let rhs_raw = rhs.raw().clone();
        Tensor::<D2<C<OUT>, N>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(rhs)],
            move |grad| {
                let values = grad.host_values()?;
                Ok(vec![
                    Some(raw_from_vec_like(&lhs_raw, values[..lhs_len].to_vec())?),
                    Some(raw_from_vec_like(&rhs_raw, values[lhs_len..].to_vec())?),
                ])
            },
        )
    }

    pub fn cat_last<R, const OUT: usize>(
        &self,
        rhs: &Tensor<D2<A, R>, E, B>,
    ) -> Result<Tensor<D2<A, C<OUT>>, E, B>>
    where
        R: DimSpec,
    {
        const {
            const_check::known_sum_eq(
                N::KNOWN,
                R::KNOWN,
                OUT,
                "cat_last",
                "lhs cols",
                "rhs cols",
                "output cols",
            );
        };

        ensure_same_device::<E, B>(self.device(), rhs.device(), "cat_last")?;
        let lhs_dims = self.shape().dims();
        let rhs_dims = rhs.shape().dims();
        let rows = lhs_dims[0];
        bind_and_check(
            "cat_last",
            [
                (DimEntry::of::<A>(0, 0), lhs_dims[0]),
                (DimEntry::of::<N>(0, 1), lhs_dims[1]),
                (DimEntry::of::<A>(1, 0), rhs_dims[0]),
                (DimEntry::of::<R>(1, 1), rhs_dims[1]),
            ],
        )?;
        if rhs_dims[0] != rows {
            return Err(ShapeError::DimMismatch {
                op: "cat_last",
                operand: 1,
                axis: 0,
                expected: rows,
                found: rhs_dims[0],
            }
            .into());
        }
        let found = lhs_dims[1] + rhs_dims[1];
        if found != OUT {
            return Err(ShapeError::LengthMismatch {
                op: "cat_last",
                expected: OUT,
                found,
            }
            .into());
        }
        let lhs_values = self.host_values()?;
        let rhs_values = rhs.host_values()?;
        let mut values = Vec::with_capacity(rows * OUT);
        for row in 0..rows {
            values.extend_from_slice(&lhs_values[row * lhs_dims[1]..(row + 1) * lhs_dims[1]]);
            values.extend_from_slice(&rhs_values[row * rhs_dims[1]..(row + 1) * rhs_dims[1]]);
        }
        let raw = RawTensor::from_vec_on(self.device().clone(), values, Shape::known([rows, OUT]))?;
        let lhs_raw = self.raw().clone();
        let rhs_raw = rhs.raw().clone();
        let lhs_cols = lhs_dims[1];
        let rhs_cols = rhs_dims[1];
        Tensor::<D2<A, C<OUT>>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(rhs)],
            move |grad| {
                let grad = grad.host_values()?;
                let mut lhs_grad = Vec::with_capacity(rows * lhs_cols);
                let mut rhs_grad = Vec::with_capacity(rows * rhs_cols);
                for row in 0..rows {
                    let start = row * OUT;
                    lhs_grad.extend_from_slice(&grad[start..start + lhs_cols]);
                    rhs_grad.extend_from_slice(&grad[start + lhs_cols..start + OUT]);
                }
                Ok(vec![
                    Some(raw_from_vec_like(&lhs_raw, lhs_grad)?),
                    Some(raw_from_vec_like(&rhs_raw, rhs_grad)?),
                ])
            },
        )
    }

    pub fn stack<R, M>(&self, rhs: &Tensor<D2<R, M>, E, B>) -> Result<Tensor<D3<C<2>, A, N>, E, B>>
    where
        R: DimSpec,
        M: DimSpec,
    {
        ensure_same_device::<E, B>(self.device(), rhs.device(), "stack")?;
        let lhs_dims = self.shape().dims();
        let rhs_dims = rhs.shape().dims();
        bind_and_check(
            "stack",
            [
                (DimEntry::of::<A>(0, 0), lhs_dims[0]),
                (DimEntry::of::<N>(0, 1), lhs_dims[1]),
                (DimEntry::of::<R>(1, 0), rhs_dims[0]),
                (DimEntry::of::<M>(1, 1), rhs_dims[1]),
            ],
        )?;
        if lhs_dims != rhs_dims {
            return Err(ShapeError::LengthMismatch {
                op: "stack",
                expected: self.numel(),
                found: rhs.numel(),
            }
            .into());
        }
        let mut values = self.host_values()?.into_owned();
        values.extend(rhs.host_values()?.iter().copied());
        let raw = RawTensor::from_vec_on(
            self.device().clone(),
            values,
            Shape::known([2, lhs_dims[0], lhs_dims[1]]),
        )?;
        let lhs_len = self.numel();
        let lhs_raw = self.raw().clone();
        let rhs_raw = rhs.raw().clone();
        Tensor::<D3<C<2>, A, N>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(rhs)],
            move |grad| {
                let values = grad.host_values()?;
                Ok(vec![
                    Some(raw_from_vec_like(&lhs_raw, values[..lhs_len].to_vec())?),
                    Some(raw_from_vec_like(&rhs_raw, values[lhs_len..].to_vec())?),
                ])
            },
        )
    }
}
