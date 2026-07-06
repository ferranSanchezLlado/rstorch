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
                grad.to_vec()?,
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
                grad.to_vec()?,
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
            return Err(ShapeError::LengthMismatch { expected: N, found }.into());
        }

        let mut data = self.to_vec()?;
        data.extend(rhs.to_vec()?);
        let raw = RawTensor::from_vec_on(self.device().clone(), data, Shape::known([N]))?;
        let left_len = self.numel();
        let lhs_shape = self.shape().clone();
        let rhs_shape = rhs.shape().clone();
        Tensor::<D1<C<N>>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(rhs)],
            move |grad| {
                let data = grad.to_vec()?;
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
        let mut values = self.to_vec()?;
        values.extend(rhs.to_vec()?);
        let raw = RawTensor::from_vec_on(self.device().clone(), values, Shape::known([2, len]))?;
        let lhs_raw = self.raw().clone();
        let rhs_raw = rhs.raw().clone();
        Tensor::<D2<C<2>, A>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(rhs)],
            move |grad| {
                let values = grad.to_vec()?;
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
