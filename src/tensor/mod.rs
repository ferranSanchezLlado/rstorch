mod autograd;
mod ops;
mod raw;

use crate::backend::{Backend, Cpu};
use crate::dtype::{DTypeId, FloatDType};
use crate::error::Result;
use crate::shape::{C, D0, D1, D2, D3, D4, Layout, Shape, ShapeSpec, StaticShape};
pub use autograd::{NoGradGuard, is_grad_enabled, no_grad};
use raw::RawTensor;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

pub struct Tensor<S, E = f32, B = Cpu>
where
    S: ShapeSpec,
    E: FloatDType,
    B: Backend<E>,
{
    inner: Arc<TensorInner<E, B>>,
    _shape: PhantomData<S>,
}

struct TensorInner<E, B>
where
    E: FloatDType,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mask<S>
where
    S: ShapeSpec,
{
    shape: Shape,
    values: Vec<bool>,
    _shape: PhantomData<S>,
}

impl<S> Mask<S>
where
    S: ShapeSpec,
{
    pub fn from_vec_with_shape(values: Vec<bool>, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        S::validate(&shape)?;
        let expected = shape.numel()?;
        if values.len() != expected {
            return Err(crate::error::ShapeError::LengthMismatch {
                expected,
                found: values.len(),
            }
            .into());
        }
        Ok(Self {
            shape,
            values,
            _shape: PhantomData,
        })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn values(&self) -> &[bool] {
        &self.values
    }
}

impl<S> Mask<S>
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
    E: FloatDType,
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
    E: FloatDType,
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
    E: FloatDType,
    B: Backend<E>,
{
    pub fn zeros() -> Result<Self> {
        Self::from_raw(RawTensor::zeros(S::static_shape())?)
    }

    pub fn ones() -> Result<Self> {
        Self::from_raw(RawTensor::ones(S::static_shape())?)
    }

    pub fn from_vec(data: Vec<E>) -> Result<Self> {
        Self::from_raw(RawTensor::from_vec(data, S::static_shape())?)
    }
}

impl<S, E, B> Tensor<S, E, B>
where
    S: ShapeSpec,
    E: FloatDType,
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

    pub fn dtype(&self) -> DTypeId {
        self.raw().dtype()
    }

    pub fn device(&self) -> &B::Device {
        self.raw().device()
    }

    pub fn shape(&self) -> &Shape {
        self.raw().shape()
    }

    pub fn layout(&self) -> &Layout {
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

    pub fn to<F, C>(&self) -> Result<Tensor<S, F, C>>
    where
        F: FloatDType,
        C: Backend<F>,
    {
        let device = C::default_device().map_err(crate::error::Error::backend)?;
        self.to_on::<F, C>(device)
    }

    pub fn cast<F>(&self) -> Result<Tensor<S, F, B>>
    where
        F: FloatDType,
        B: Backend<F, Device = <B as Backend<E>>::Device>,
    {
        self.to_on::<F, B>(self.device().clone())
    }

    pub fn to_device<C>(&self) -> Result<Tensor<S, E, C>>
    where
        C: Backend<E>,
    {
        self.to::<E, C>()
    }

    pub fn to_backend<C>(&self) -> Result<Tensor<S, E, C>>
    where
        C: Backend<E>,
    {
        self.to_device::<C>()
    }

    pub fn to_device_on<C>(&self, device: C::Device) -> Result<Tensor<S, E, C>>
    where
        C: Backend<E>,
    {
        self.to_on::<E, C>(device)
    }

    pub fn to_on<F, C>(&self, device: C::Device) -> Result<Tensor<S, F, C>>
    where
        F: FloatDType,
        C: Backend<F>,
    {
        let data = self
            .to_vec()?
            .into_iter()
            .map(|value| F::from_f64(value.to_f64()))
            .collect();
        let raw = RawTensor::<F, C>::from_vec_on(device, data, self.shape().clone())?;
        Tensor::<S, F, C>::from_raw(raw)
    }

    pub(crate) fn replace_data(&mut self, data: Vec<E>) -> Result<()> {
        let raw = RawTensor::from_vec_on(self.device().clone(), data, self.shape().clone())?;
        *self = Self::from_raw(raw)?.with_requires_grad(true);
        Ok(())
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

#[cfg(test)]
pub(super) mod test_support {
    use super::*;
    use crate::dtype::DType;
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
                expected: 6,
                found: 5,
            })
        ));

        let err =
            Tensor::<D2<Sym<Batch>, C<3>>>::from_vec_with_shape(vec![1.0; 5], [2, 3]).unwrap_err();
        assert!(matches!(
            err,
            Error::Shape(ShapeError::LengthMismatch {
                expected: 6,
                found: 5,
            })
        ));
    }
}
