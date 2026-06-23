use super::{RawTensor, Scalar, Tensor, Tensor1D, Tensor4D};
use crate::backend::Backend;
use crate::dtype::DType;
use crate::error::{DeviceError, Error, Result, ShapeError};
use crate::shape::{
    C, D0, D1, D2, D3, D4, DimEntry, DimSpec, Shape, ShapeSpec, StaticShape, bind_and_check,
};

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
    E: DType,
    B: Backend<E>,
{
    pub fn reshape<T>(&self) -> Result<Tensor<T, E, B>>
    where
        T: StaticShape,
    {
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
        Tensor::<T, E, B>::from_raw(raw)
    }

    pub fn reshape_with_shape<T>(&self, shape: impl Into<Shape>) -> Result<Tensor<T, E, B>>
    where
        T: ShapeSpec,
    {
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
        Tensor::<T, E, B>::from_raw(raw)
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

    pub fn add(&self, rhs: &Self) -> Result<Self> {
        self.binary_same_shape(rhs, "add", B::add)
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self> {
        self.binary_same_shape(rhs, "sub", B::sub)
    }

    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        self.binary_same_shape(rhs, "mul", B::mul)
    }

    pub fn div(&self, rhs: &Self) -> Result<Self> {
        self.binary_same_shape(rhs, "div", B::div)
    }

    pub fn add_scalar(&self, rhs: E) -> Result<Self> {
        self.unary_scalar(rhs, B::add_scalar)
    }

    pub fn sub_scalar(&self, rhs: E) -> Result<Self> {
        self.unary_scalar(rhs, B::sub_scalar)
    }

    pub fn mul_scalar(&self, rhs: E) -> Result<Self> {
        self.unary_scalar(rhs, B::mul_scalar)
    }

    pub fn div_scalar(&self, rhs: E) -> Result<Self> {
        self.unary_scalar(rhs, B::div_scalar)
    }

    pub fn sum(&self) -> Result<Scalar<E, B>> {
        let input = self.contiguous()?;
        let storage =
            B::sum(input.device(), input.raw().storage(), input.numel()).map_err(Error::backend)?;
        let raw = RawTensor::from_storage_on(self.device().clone(), storage, Shape::known([]))?;
        Tensor::<D0, E, B>::from_raw(raw)
    }

    fn ensure_same_device(&self, rhs: &Self, op: &'static str) -> Result<()> {
        ensure_same_device::<E, B>(self.device(), rhs.device(), op)
    }

    fn binary_same_shape(
        &self,
        rhs: &Self,
        op: &'static str,
        kernel: BinaryKernel<E, B>,
    ) -> Result<Self> {
        self.ensure_same_device(rhs, op)?;

        let lhs_dims = self.shape().dims();
        let rhs_dims = rhs.shape().dims();
        bind_and_check(
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
        Self::from_raw(raw)
    }

    fn unary_scalar(&self, rhs: E, kernel: ScalarKernel<E, B>) -> Result<Self> {
        let input = self.contiguous()?;
        let storage = kernel(input.device(), input.raw().storage(), rhs, input.numel())
            .map_err(Error::backend)?;
        let raw = RawTensor::from_storage_on(self.device().clone(), storage, self.shape().clone())?;
        Self::from_raw(raw)
    }
}

impl<A, E, B> Tensor<D1<A>, E, B>
where
    A: DimSpec,
    E: DType,
    B: Backend<E>,
{
    pub fn cat1<R, const N: usize>(&self, rhs: &Tensor<D1<R>, E, B>) -> Result<Tensor1D<N, E, B>>
    where
        R: DimSpec,
    {
        ensure_same_device::<E, B>(self.device(), rhs.device(), "cat1")?;
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
        Tensor::<D1<C<N>>, E, B>::from_raw(raw)
    }
}

impl<A, K, E, B> Tensor<D2<A, K>, E, B>
where
    A: DimSpec,
    K: DimSpec,
    E: DType,
    B: Backend<E>,
{
    pub fn transpose(&self) -> Result<Tensor<D2<K, A>, E, B>> {
        let raw = self.raw().view_with_layout(self.layout().transpose2()?)?;
        Tensor::<D2<K, A>, E, B>::from_raw(raw)
    }

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
        Tensor::<D2<A, Cc>, E, B>::from_raw(raw)
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
