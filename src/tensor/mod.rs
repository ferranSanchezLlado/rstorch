mod autograd;
mod axis_ops;
mod ergonomics;
mod ops;
mod raw;

use crate::backend::{Backend, Cpu};
use crate::dtype::{DType, DTypeId, FloatDType, bf16, f16};
use crate::error::Result;
use crate::random::SmallRng;
use crate::shape::{C, D0, D1, D2, D3, D4, Shape, ShapeSpec, StaticShape};
pub use autograd::{NoGradGuard, is_grad_enabled, no_grad};
pub use ops::{Conv2dOptions, Padding2d, Pool2dOptions};
use raw::RawTensor;
use std::borrow::Cow;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

pub struct Tensor<S, E = f32, B = Cpu>
where
    S: ShapeSpec,
    E: DType,
    B: Backend<E>,
{
    inner: Arc<TensorInner<E, B>>,
    _shape: PhantomData<S>,
}

struct TensorInner<E, B>
where
    E: DType,
    B: Backend<E>,
{
    raw: RawTensor<E, B>,
    autograd: autograd::AutogradMeta<E, B>,
}

pub type Scalar<E = f32, B = Cpu> = Tensor<D0, E, B>;
pub type Tensor1D<const N: usize, E = f32, B = Cpu> = Tensor<D1<C<N>>, E, B>;
pub type Tensor2D<const M: usize, const N: usize, E = f32, B = Cpu> = Tensor<D2<C<M>, C<N>>, E, B>;
pub type Tensor3D<const X: usize, const Y: usize, const Z: usize, E = f32, B = Cpu> =
    Tensor<D3<C<X>, C<Y>, C<Z>>, E, B>;
pub type Tensor4D<
    const N: usize,
    const CH: usize,
    const H: usize,
    const W: usize,
    E = f32,
    B = Cpu,
> = Tensor<D4<C<N>, C<CH>, C<H>, C<W>>, E, B>;

/// Boolean mask with shape and backend markers.
///
/// The current implementation stores host `bool` values internally. The
/// backend type parameter is part of the public type now so a future bool-dtype
/// or backend-owned mask representation can be added without a structural API
/// change.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mask<S, B = Cpu>
where
    S: ShapeSpec,
{
    shape: Shape,
    values: Vec<bool>,
    _shape: PhantomData<S>,
    _backend: PhantomData<B>,
}

impl<S, B> Mask<S, B>
where
    S: ShapeSpec,
{
    pub fn from_vec_with_shape(values: Vec<bool>, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        S::validate(&shape)?;
        let expected = shape.numel()?;
        if values.len() != expected {
            return Err(crate::error::ShapeError::LengthMismatch {
                op: "mask_from_vec_with_shape",
                expected,
                found: values.len(),
            }
            .into());
        }
        Ok(Self {
            shape,
            values,
            _shape: PhantomData,
            _backend: PhantomData,
        })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn to_vec(&self) -> Result<Vec<bool>> {
        Ok(self.values.clone())
    }

    pub fn and(&self, rhs: &Self) -> Result<Self> {
        self.binary_mask(rhs, "mask_and", |lhs, rhs| lhs && rhs)
    }

    pub fn or(&self, rhs: &Self) -> Result<Self> {
        self.binary_mask(rhs, "mask_or", |lhs, rhs| lhs || rhs)
    }

    pub fn not(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            values: self.values.iter().map(|value| !value).collect(),
            _shape: PhantomData,
            _backend: PhantomData,
        }
    }

    fn binary_mask(
        &self,
        rhs: &Self,
        op: &'static str,
        f: impl Fn(bool, bool) -> bool,
    ) -> Result<Self> {
        if self.shape != rhs.shape {
            return Err(crate::error::ShapeError::LengthMismatch {
                op,
                expected: self.values.len(),
                found: rhs.values.len(),
            }
            .into());
        }
        Ok(Self {
            shape: self.shape.clone(),
            values: self
                .values
                .iter()
                .copied()
                .zip(rhs.values.iter().copied())
                .map(|(lhs, rhs)| f(lhs, rhs))
                .collect(),
            _shape: PhantomData,
            _backend: PhantomData,
        })
    }
}

impl<A, N, B> Mask<D2<A, N>, B>
where
    A: crate::shape::DimSpec,
    N: crate::shape::DimSpec,
{
    pub fn expand_leading<const R: usize>(&self) -> Result<Mask<D3<crate::shape::C<R>, A, N>, B>> {
        let dims = self.shape.dims();
        let mut values = Vec::with_capacity(R * self.values.len());
        for _ in 0..R {
            values.extend(self.values.iter().copied());
        }
        Mask::from_vec_with_shape(values, [R, dims[0], dims[1]])
    }
}

impl<S, B> Mask<S, B>
where
    S: StaticShape,
{
    pub fn from_vec(values: Vec<bool>) -> Result<Self> {
        Self::from_vec_with_shape(values, S::static_shape())
    }
}

impl<S, E, B> Clone for Tensor<S, E, B>
where
    S: ShapeSpec,
    E: DType,
    B: Backend<E>,
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            _shape: PhantomData,
        }
    }
}

impl<S, E, B> Debug for Tensor<S, E, B>
where
    S: ShapeSpec,
    E: DType,
    B: Backend<E>,
    B::Device: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("dtype", &self.dtype())
            .field("device", self.device())
            .field("shape", self.shape())
            .finish()
    }
}

impl<S, E, B> Tensor<S, E, B>
where
    S: StaticShape,
    E: DType,
    B: Backend<E>,
{
    pub fn zeros() -> Result<Self> {
        Self::from_raw(RawTensor::zeros(S::static_shape())?)
    }

    pub fn ones() -> Result<Self> {
        Self::from_raw(RawTensor::ones(S::static_shape())?)
    }

    pub fn full(value: E) -> Result<Self> {
        let shape = S::static_shape();
        let len = shape.numel()?;
        Self::from_raw(RawTensor::from_vec(vec![value; len], shape)?)
    }

    pub fn from_vec(data: Vec<E>) -> Result<Self> {
        Self::from_raw(RawTensor::from_vec(data, S::static_shape())?)
    }
}

impl<S, E, B> Tensor<S, E, B>
where
    S: StaticShape,
    E: FloatDType,
    B: Backend<E>,
{
    pub fn rand(rng: &mut SmallRng) -> Result<Self> {
        let shape = S::static_shape();
        let len = shape.numel()?;
        let values = (0..len).map(|_| rng.uniform(E::ZERO, E::ONE)).collect();
        Self::from_raw(RawTensor::from_vec(values, shape)?)
    }

    pub fn randn(rng: &mut SmallRng) -> Result<Self> {
        let shape = S::static_shape();
        let len = shape.numel()?;
        let values = (0..len).map(|_| rng.normal::<E>()).collect();
        Self::from_raw(RawTensor::from_vec(values, shape)?)
    }
}

impl<const N: usize, E, B> Tensor<D1<C<N>>, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn arange() -> Result<Self> {
        Self::from_vec((0..N).map(E::from_usize).collect())
    }
}

impl<const N: usize, B> Tensor<D1<C<N>>, i64, B>
where
    B: Backend<i64>,
{
    /// Builds an i64 data tensor containing `0..N`.
    ///
    /// i64 tensors are intended for ids, labels, and indices. Integer
    /// autograd, typed-layer arithmetic, matmul, and GPU i64 backends are
    /// intentionally not part of the pre-1.0 surface.
    pub fn arange() -> Result<Self> {
        Self::from_vec((0..N).map(|value| value as i64).collect())
    }
}

impl<S, E, B> Tensor<S, E, B>
where
    S: ShapeSpec,
    E: DType,
    B: Backend<E>,
{
    pub fn zeros_with_shape(shape: impl Into<Shape>) -> Result<Self> {
        Self::from_raw(RawTensor::zeros(shape.into())?)
    }

    pub fn ones_with_shape(shape: impl Into<Shape>) -> Result<Self> {
        Self::from_raw(RawTensor::ones(shape.into())?)
    }

    pub fn from_vec_with_shape(data: Vec<E>, shape: impl Into<Shape>) -> Result<Self> {
        Self::from_raw(RawTensor::from_vec(data, shape.into())?)
    }

    pub fn full_with_shape(value: E, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let len = shape.numel()?;
        Self::from_raw(RawTensor::from_vec(vec![value; len], shape)?)
    }

    pub fn dtype(&self) -> DTypeId {
        self.raw().dtype()
    }

    pub fn device(&self) -> &B::Device {
        self.raw().device()
    }

    pub fn shape(&self) -> &Shape {
        self.raw().shape()
    }

    pub(crate) fn layout(&self) -> &crate::shape::Layout {
        self.raw().layout()
    }

    pub fn rank(&self) -> usize {
        self.raw().rank()
    }

    pub fn numel(&self) -> usize {
        self.raw().numel()
    }

    pub fn to_vec(&self) -> Result<Vec<E>> {
        self.raw().to_vec()
    }

    pub(crate) fn host_values(&self) -> Result<Cow<'_, [E]>> {
        self.raw().host_values()
    }

    /// Casts this tensor to another dtype and returns a detached leaf.
    ///
    /// Integer tensors are data containers, not differentiable numeric tensors:
    /// i64 casts do not enable integer autograd, arithmetic, or matmul. Casting
    /// i64 to a float uses the nearest representable float value and may lose
    /// precision above 2^53. Casting a float to i64 truncates toward zero and
    /// returns an error for non-finite or out-of-range values.
    ///
    /// Conversion methods intentionally detach from autograd graphs. Graphs are
    /// single-dtype and single-backend; compose `cast`, `to_backend`, and
    /// `to_backend_on` before enabling gradients for converted tensors.
    pub fn cast<F>(&self) -> Result<Tensor<S, F, B>>
    where
        F: DType,
        B: Backend<F, Device = <B as Backend<E>>::Device>,
    {
        let data = self
            .to_vec()?
            .into_iter()
            .map(cast_value::<E, F>)
            .collect::<Result<Vec<_>>>()?;
        let raw =
            RawTensor::<F, B>::from_vec_on(self.device().clone(), data, self.shape().clone())?;
        Tensor::<S, F, B>::from_raw(raw)
    }

    /// Moves this tensor to another backend's default device as a detached leaf.
    ///
    /// Conversion methods intentionally detach from autograd graphs. Graphs are
    /// single-dtype and single-backend; compose `cast`, `to_backend`, and
    /// `to_backend_on` before enabling gradients for converted tensors.
    pub fn to_backend<C>(&self) -> Result<Tensor<S, E, C>>
    where
        C: Backend<E>,
    {
        let device = C::default_device().map_err(crate::error::Error::backend)?;
        self.to_backend_on::<C>(device)
    }

    /// Moves this tensor to an explicit backend device as a detached leaf.
    ///
    /// Conversion methods intentionally detach from autograd graphs. Graphs are
    /// single-dtype and single-backend; compose `cast`, `to_backend`, and
    /// `to_backend_on` before enabling gradients for converted tensors.
    pub fn to_backend_on<C>(&self, device: C::Device) -> Result<Tensor<S, E, C>>
    where
        C: Backend<E>,
    {
        let raw = RawTensor::<E, C>::from_vec_on(device, self.to_vec()?, self.shape().clone())?;
        Tensor::<S, E, C>::from_raw(raw)
    }

    pub fn is_contiguous(&self) -> bool {
        self.raw().is_contiguous()
    }

    pub fn is_view(&self) -> bool {
        self.raw().has_storage_view()
    }

    pub fn contiguous(&self) -> Result<Self> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }

        let raw =
            RawTensor::from_vec_on(self.device().clone(), self.to_vec()?, self.shape().clone())?;
        Self::autograd_output(raw, vec![autograd::AnyTensor::from_shape(self)], |grad| {
            Ok(vec![Some(grad.clone())])
        })
    }

    pub fn shares_storage_with<T>(&self, other: &Tensor<T, E, B>) -> bool
    where
        T: ShapeSpec,
    {
        self.raw().shares_storage_with(other.raw())
    }

    fn from_raw(raw: RawTensor<E, B>) -> Result<Self> {
        S::validate(raw.shape())?;
        Ok(Self {
            inner: Arc::new(TensorInner {
                raw,
                autograd: autograd::AutogradMeta::leaf(),
            }),
            _shape: PhantomData,
        })
    }

    pub(crate) fn from_raw_non_leaf(raw: RawTensor<E, B>) -> Result<Self> {
        S::validate(raw.shape())?;
        Ok(Self {
            inner: Arc::new(TensorInner {
                raw,
                autograd: autograd::AutogradMeta::non_leaf(),
            }),
            _shape: PhantomData,
        })
    }

    fn raw(&self) -> &RawTensor<E, B> {
        &self.inner.raw
    }
}

fn cast_value<E, F>(value: E) -> Result<F>
where
    E: DType,
    F: DType,
{
    cast_scalar_to_dtype::<F>(cast_scalar_from_dtype(value), E::ID)
}

enum CastScalar {
    Float(f64),
    Int(i64),
}

fn cast_scalar_from_dtype<E: DType>(value: E) -> CastScalar {
    let mut bytes = Vec::with_capacity(E::BYTE_SIZE);
    value.write_le_bytes(&mut bytes);
    match E::ID {
        DTypeId::F16 => CastScalar::Float(
            <f16 as DType>::read_le_bytes(&bytes)
                .expect("dtype id and byte width match")
                .to_f64(),
        ),
        DTypeId::BF16 => CastScalar::Float(
            <bf16 as DType>::read_le_bytes(&bytes)
                .expect("dtype id and byte width match")
                .to_f64(),
        ),
        DTypeId::F32 => CastScalar::Float(
            <f32 as DType>::read_le_bytes(&bytes).expect("dtype id and byte width match") as f64,
        ),
        DTypeId::F64 => CastScalar::Float(
            <f64 as DType>::read_le_bytes(&bytes).expect("dtype id and byte width match"),
        ),
        DTypeId::I64 => CastScalar::Int(
            <i64 as DType>::read_le_bytes(&bytes).expect("dtype id and byte width match"),
        ),
    }
}

fn cast_scalar_to_dtype<F: DType>(value: CastScalar, from: DTypeId) -> Result<F> {
    match F::ID {
        DTypeId::F16 => Ok(retype_cast_value::<F, f16>(match value {
            CastScalar::Float(value) => <f16 as FloatDType>::from_f64(value),
            CastScalar::Int(value) => <f16 as FloatDType>::from_f64(value as f64),
        })),
        DTypeId::BF16 => Ok(retype_cast_value::<F, bf16>(match value {
            CastScalar::Float(value) => <bf16 as FloatDType>::from_f64(value),
            CastScalar::Int(value) => <bf16 as FloatDType>::from_f64(value as f64),
        })),
        DTypeId::F32 => Ok(retype_cast_value::<F, f32>(match value {
            CastScalar::Float(value) => value as f32,
            CastScalar::Int(value) => value as f32,
        })),
        DTypeId::F64 => Ok(retype_cast_value::<F, f64>(match value {
            CastScalar::Float(value) => value,
            CastScalar::Int(value) => value as f64,
        })),
        DTypeId::I64 => match value {
            CastScalar::Int(value) => Ok(retype_cast_value::<F, i64>(value)),
            CastScalar::Float(value) => {
                let value = cast_float_to_i64(value, from)?;
                Ok(retype_cast_value::<F, i64>(value))
            }
        },
    }
}

fn cast_float_to_i64(value: f64, from: DTypeId) -> Result<i64> {
    if !value.is_finite() {
        return Err(crate::error::DTypeError::InvalidCast {
            op: "cast",
            from,
            to: DTypeId::I64,
            reason: "float to i64 cast requires a finite value",
        }
        .into());
    }
    let truncated = value.trunc();
    if !(truncated >= i64::MIN as f64 && truncated < 9_223_372_036_854_775_808.0) {
        return Err(crate::error::DTypeError::InvalidCast {
            op: "cast",
            from,
            to: DTypeId::I64,
            reason: "float to i64 cast is out of range after truncation",
        }
        .into());
    }
    Ok(truncated as i64)
}

fn retype_cast_value<F, T>(value: T) -> F
where
    F: DType,
    T: DType,
{
    let mut bytes = Vec::with_capacity(T::BYTE_SIZE);
    value.write_le_bytes(&mut bytes);
    F::read_le_bytes(&bytes).expect("matched dtype id preserves byte width")
}

impl<E, B> Tensor<D0, E, B>
where
    E: DType,
    B: Backend<E>,
{
    pub fn item(&self) -> Result<E> {
        Ok(self.to_vec()?[0])
    }
}

impl<S, E, B> Tensor<S, E, B>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    pub fn rand_with_shape(rng: &mut SmallRng, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let len = shape.numel()?;
        let values = (0..len).map(|_| rng.uniform(E::ZERO, E::ONE)).collect();
        Self::from_raw(RawTensor::from_vec(values, shape)?)
    }

    pub fn randn_with_shape(rng: &mut SmallRng, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let len = shape.numel()?;
        let values = (0..len).map(|_| rng.normal::<E>()).collect();
        Self::from_raw(RawTensor::from_vec(values, shape)?)
    }

    pub(crate) fn replace_data(&mut self, data: Vec<E>) -> Result<()> {
        let raw = RawTensor::from_vec_on(self.device().clone(), data, self.shape().clone())?;
        *self = Self::from_raw(raw)?.with_requires_grad(true);
        Ok(())
    }
}

#[cfg(test)]
#[allow(dead_code)]
pub(super) mod test_support {
    use super::*;
    use crate::backend::sealed;
    use crate::dtype::DType;
    use std::borrow::Cow;
    use std::error;
    use std::fmt;

    #[derive(Debug)]
    pub(super) struct Batch;

    #[derive(Debug)]
    pub(super) struct Hidden;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(super) enum TestDevice {
        A,
        B,
    }

    #[derive(Debug, Clone, Copy)]
    pub(super) struct TestBackend;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(super) struct TestBackendError;

    impl fmt::Display for TestBackendError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "test backend error")
        }
    }

    impl error::Error for TestBackendError {}

    impl sealed::SealedBackend for TestBackend {}

    impl<E: DType> Backend<E> for TestBackend {
        type Device = TestDevice;
        type Storage = Vec<E>;
        type Error = TestBackendError;

        fn default_device() -> std::result::Result<Self::Device, Self::Error> {
            Ok(TestDevice::A)
        }

        fn zeros(
            _device: &Self::Device,
            len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Ok(vec![E::ZERO; len])
        }

        fn ones(
            _device: &Self::Device,
            len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Ok(vec![E::ONE; len])
        }

        fn from_vec(
            _device: &Self::Device,
            data: Vec<E>,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Ok(data)
        }

        fn to_vec(
            _device: &Self::Device,
            storage: &Self::Storage,
        ) -> std::result::Result<Vec<E>, Self::Error> {
            Ok(storage.clone())
        }

        fn host_access<'a>(
            _device: &Self::Device,
            storage: &'a Self::Storage,
        ) -> std::result::Result<Cow<'a, [E]>, Self::Error> {
            Ok(Cow::Borrowed(storage.as_slice()))
        }

        fn storage_len(storage: &Self::Storage) -> usize {
            storage.len()
        }

        fn matmul(
            _device: &Self::Device,
            _lhs: &Self::Storage,
            _rhs: &Self::Storage,
            m: usize,
            _k: usize,
            n: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Ok(vec![E::ZERO; m * n])
        }

        fn add(
            _device: &Self::Device,
            lhs: &Self::Storage,
            rhs: &Self::Storage,
            len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Ok(lhs
                .iter()
                .zip(rhs.iter())
                .take(len)
                .map(|(&lhs, &rhs)| lhs + rhs)
                .collect())
        }

        fn sub(
            _device: &Self::Device,
            lhs: &Self::Storage,
            rhs: &Self::Storage,
            len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Ok(lhs
                .iter()
                .zip(rhs.iter())
                .take(len)
                .map(|(&lhs, &rhs)| lhs - rhs)
                .collect())
        }

        fn mul(
            _device: &Self::Device,
            lhs: &Self::Storage,
            rhs: &Self::Storage,
            len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Ok(lhs
                .iter()
                .zip(rhs.iter())
                .take(len)
                .map(|(&lhs, &rhs)| lhs * rhs)
                .collect())
        }

        fn div(
            _device: &Self::Device,
            lhs: &Self::Storage,
            rhs: &Self::Storage,
            len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Ok(lhs
                .iter()
                .zip(rhs.iter())
                .take(len)
                .map(|(&lhs, &rhs)| lhs / rhs)
                .collect())
        }

        fn add_scalar(
            _device: &Self::Device,
            input: &Self::Storage,
            rhs: E,
            len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Ok(input.iter().take(len).map(|&value| value + rhs).collect())
        }

        fn sub_scalar(
            _device: &Self::Device,
            input: &Self::Storage,
            rhs: E,
            len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Ok(input.iter().take(len).map(|&value| value - rhs).collect())
        }

        fn mul_scalar(
            _device: &Self::Device,
            input: &Self::Storage,
            rhs: E,
            len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Ok(input.iter().take(len).map(|&value| value * rhs).collect())
        }

        fn div_scalar(
            _device: &Self::Device,
            input: &Self::Storage,
            rhs: E,
            len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Ok(input.iter().take(len).map(|&value| value / rhs).collect())
        }

        fn sum(
            _device: &Self::Device,
            input: &Self::Storage,
            len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Ok(vec![
                input
                    .iter()
                    .take(len)
                    .fold(E::ZERO, |acc, &value| acc + value),
            ])
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub(super) struct FailingBackend;

    #[derive(Debug)]
    pub(super) struct FailingBackendError;

    impl fmt::Display for FailingBackendError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "failing backend error")
        }
    }

    impl error::Error for FailingBackendError {}

    impl sealed::SealedBackend for FailingBackend {}

    impl<E: DType> Backend<E> for FailingBackend {
        type Device = TestDevice;
        type Storage = Vec<E>;
        type Error = FailingBackendError;

        fn default_device() -> std::result::Result<Self::Device, Self::Error> {
            Ok(TestDevice::A)
        }

        fn zeros(
            _device: &Self::Device,
            _len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Err(FailingBackendError)
        }

        fn ones(
            _device: &Self::Device,
            _len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Err(FailingBackendError)
        }

        fn from_vec(
            _device: &Self::Device,
            _data: Vec<E>,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Err(FailingBackendError)
        }

        fn to_vec(
            _device: &Self::Device,
            _storage: &Self::Storage,
        ) -> std::result::Result<Vec<E>, Self::Error> {
            Err(FailingBackendError)
        }

        fn host_access<'a>(
            _device: &Self::Device,
            _storage: &'a Self::Storage,
        ) -> std::result::Result<Cow<'a, [E]>, Self::Error> {
            Err(FailingBackendError)
        }

        fn storage_len(storage: &Self::Storage) -> usize {
            storage.len()
        }

        fn matmul(
            _device: &Self::Device,
            _lhs: &Self::Storage,
            _rhs: &Self::Storage,
            _m: usize,
            _k: usize,
            _n: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Err(FailingBackendError)
        }

        fn add(
            _device: &Self::Device,
            _lhs: &Self::Storage,
            _rhs: &Self::Storage,
            _len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Err(FailingBackendError)
        }

        fn sub(
            _device: &Self::Device,
            _lhs: &Self::Storage,
            _rhs: &Self::Storage,
            _len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Err(FailingBackendError)
        }

        fn mul(
            _device: &Self::Device,
            _lhs: &Self::Storage,
            _rhs: &Self::Storage,
            _len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Err(FailingBackendError)
        }

        fn div(
            _device: &Self::Device,
            _lhs: &Self::Storage,
            _rhs: &Self::Storage,
            _len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Err(FailingBackendError)
        }

        fn add_scalar(
            _device: &Self::Device,
            _input: &Self::Storage,
            _rhs: E,
            _len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Err(FailingBackendError)
        }

        fn sub_scalar(
            _device: &Self::Device,
            _input: &Self::Storage,
            _rhs: E,
            _len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Err(FailingBackendError)
        }

        fn mul_scalar(
            _device: &Self::Device,
            _input: &Self::Storage,
            _rhs: E,
            _len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Err(FailingBackendError)
        }

        fn div_scalar(
            _device: &Self::Device,
            _input: &Self::Storage,
            _rhs: E,
            _len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Err(FailingBackendError)
        }

        fn sum(
            _device: &Self::Device,
            _input: &Self::Storage,
            _len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Err(FailingBackendError)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::Cpu;
    use crate::error::{Error, ShapeError};
    use crate::shape::Sym;
    use crate::tensor::test_support::Batch;

    #[test]
    fn static_constructors_create_cpu_tensors() {
        let zeros = Tensor2D::<2, 3>::zeros().unwrap();
        assert_eq!(zeros.shape().dims(), &[2, 3]);
        assert_eq!(zeros.to_vec().unwrap(), vec![0.0; 6]);

        let ones = Tensor::<D2<C<2>, C<3>>, f64, Cpu>::ones().unwrap();
        assert_eq!(ones.to_vec().unwrap(), vec![1.0; 6]);
    }

    #[test]
    fn dynamic_constructor_validates_shape_markers() {
        let tensor = Tensor::<D2<Sym<Batch>, C<784>>>::zeros_with_shape([32, 784]).unwrap();
        assert_eq!(tensor.shape().dims(), &[32, 784]);

        let err = Tensor::<D2<Sym<Batch>, C<784>>>::zeros_with_shape([32, 10]).unwrap_err();
        assert!(matches!(
            err,
            Error::Shape(ShapeError::DimMismatch {
                op: "validate",
                operand: 0,
                axis: 1,
                expected: 784,
                found: 10,
            })
        ));
    }

    #[test]
    fn from_vec_rejects_length_mismatch() {
        let err = Tensor2D::<2, 3>::from_vec(vec![1.0; 5]).unwrap_err();
        assert!(matches!(
            err,
            Error::Shape(ShapeError::LengthMismatch {
                op: "raw_from_vec_on",
                expected: 6,
                found: 5,
            })
        ));

        let err =
            Tensor::<D2<Sym<Batch>, C<3>>>::from_vec_with_shape(vec![1.0; 5], [2, 3]).unwrap_err();
        assert!(matches!(
            err,
            Error::Shape(ShapeError::LengthMismatch {
                op: "raw_from_vec_on",
                expected: 6,
                found: 5,
            })
        ));
    }
}
