mod ops;
mod raw;

use crate::backend::{Backend, Cpu};
use crate::dtype::{DType, DTypeId};
use crate::error::Result;
use crate::shape::{C, D0, D1, D2, D3, D4, Layout, Shape, ShapeSpec, StaticShape};
use raw::RawTensor;
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

    pub fn from_vec(data: Vec<E>) -> Result<Self> {
        Self::from_raw(RawTensor::from_vec(data, S::static_shape())?)
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
        Self::from_raw(raw)
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
            inner: Arc::new(TensorInner { raw }),
            _shape: PhantomData,
        })
    }

    fn raw(&self) -> &RawTensor<E, B> {
        &self.inner.raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Cpu, CpuDevice};
    use crate::dtype::DTypeId;
    use crate::error::{DeviceError, Error, ShapeError};
    use crate::shape::{AnyDim, Sym};
    use std::error;
    use std::fmt;

    #[derive(Debug)]
    struct Batch;

    #[derive(Debug)]
    struct Hidden;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum TestDevice {
        A,
        B,
    }

    #[derive(Debug, Clone, Copy)]
    struct TestBackend;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TestBackendError;

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
            Ok(vec![E::zero(); len])
        }

        fn ones(
            _device: &Self::Device,
            len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Ok(vec![E::one(); len])
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
            Ok(vec![E::zero(); m * n])
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
                    .fold(E::zero(), |acc, &value| acc + value),
            ])
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct FailingBackend;

    #[derive(Debug)]
    struct FailingBackendError;

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

    #[test]
    fn reshape_views_to_static_target() {
        let tensor = Tensor2D::<2, 6>::from_vec((0..12).map(|x| x as f32).collect()).unwrap();
        let reshaped = tensor.reshape1::<12>().unwrap();
        assert_eq!(reshaped.shape().dims(), &[12]);
        assert!(reshaped.shares_storage_with(&tensor));

        let reshaped = tensor.reshape2::<3, 4>().unwrap();
        assert_eq!(reshaped.shape().dims(), &[3, 4]);
        assert!(reshaped.is_contiguous());
        assert!(reshaped.is_view());
        assert_eq!(
            reshaped.to_vec().unwrap(),
            (0..12).map(|x| x as f32).collect::<Vec<_>>()
        );

        let reshaped = tensor.reshape3::<2, 3, 2>().unwrap();
        assert_eq!(reshaped.shape().dims(), &[2, 3, 2]);
        assert!(reshaped.shares_storage_with(&tensor));

        let reshaped = tensor.reshape4::<1, 2, 2, 3>().unwrap();
        assert_eq!(reshaped.shape().dims(), &[1, 2, 2, 3]);
        assert!(reshaped.shares_storage_with(&tensor));

        let err = tensor.reshape2::<5, 2>().unwrap_err();
        assert!(matches!(
            err,
            Error::Shape(ShapeError::LengthMismatch {
                expected: 10,
                found: 12,
            })
        ));
    }

    #[test]
    fn backend_built_storage_lengths_match_numel() {
        let zeros = Tensor2D::<2, 3>::zeros().unwrap();
        assert_eq!(<Cpu as Backend<f32>>::storage_len(zeros.raw().storage()), 6);
        assert_eq!(zeros.numel(), 6);

        let ones = Tensor2D::<2, 3>::ones().unwrap();
        assert_eq!(<Cpu as Backend<f32>>::storage_len(ones.raw().storage()), 6);
        assert_eq!(ones.numel(), 6);
    }

    #[test]
    fn matmul_computes_and_preserves_types() {
        let lhs = Tensor::<D2<C<2>, C<3>>>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let rhs =
            Tensor::<D2<C<3>, C<2>>>::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let out: Tensor<D2<C<2>, C<2>>> = lhs.matmul(&rhs).unwrap();

        assert_eq!(out.shape().dims(), &[2, 2]);
        assert_eq!(out.to_vec().unwrap(), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn matmul_anydim_still_checks_structure() {
        let lhs = Tensor::<D2<C<2>, AnyDim>>::from_vec_with_shape(vec![1.0; 6], [2, 3]).unwrap();
        let rhs = Tensor::<D2<AnyDim, C<2>>>::from_vec_with_shape(vec![1.0; 8], [4, 2]).unwrap();
        let err = lhs.matmul(&rhs).unwrap_err();

        assert!(matches!(
            err,
            Error::Shape(ShapeError::DimMismatch {
                op: "matmul",
                operand: 1,
                axis: 0,
                expected: 3,
                found: 4,
            })
        ));
    }

    #[test]
    fn matmul_shared_sym_reports_symbol_mismatch() {
        let lhs =
            Tensor::<D2<C<2>, Sym<Hidden>>>::from_vec_with_shape(vec![1.0; 6], [2, 3]).unwrap();
        let rhs =
            Tensor::<D2<Sym<Hidden>, C<2>>>::from_vec_with_shape(vec![1.0; 8], [4, 2]).unwrap();
        let err = lhs.matmul(&rhs).unwrap_err();

        assert!(matches!(
            err,
            Error::Shape(ShapeError::SymbolMismatch { op: "matmul", .. })
        ));
    }

    #[test]
    fn add_and_mul_compute_elementwise() {
        let lhs = Tensor2D::<2, 2>::from_vec(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let rhs = Tensor2D::<2, 2>::from_vec(vec![5.0, 6.0, 7.0, 8.0]).unwrap();

        assert_eq!(
            lhs.add(&rhs).unwrap().to_vec().unwrap(),
            vec![6.0, 8.0, 10.0, 12.0]
        );
        assert_eq!(
            lhs.mul(&rhs).unwrap().to_vec().unwrap(),
            vec![5.0, 12.0, 21.0, 32.0]
        );
    }

    #[test]
    fn same_shape_ops_reject_runtime_mismatches_for_anydim() {
        let lhs = Tensor::<D1<AnyDim>>::from_vec_with_shape(vec![1.0, 2.0], [2]).unwrap();
        let rhs = Tensor::<D1<AnyDim>>::from_vec_with_shape(vec![3.0, 4.0, 5.0], [3]).unwrap();

        let err = lhs.add(&rhs).unwrap_err();
        assert!(matches!(
            err,
            Error::Shape(ShapeError::DimMismatch {
                op: "add",
                operand: 1,
                axis: 0,
                expected: 2,
                found: 3,
            })
        ));
    }

    #[test]
    fn sub_and_div_compute_elementwise() {
        let lhs = Tensor2D::<2, 2>::from_vec(vec![8.0, 9.0, 10.0, 12.0]).unwrap();
        let rhs = Tensor2D::<2, 2>::from_vec(vec![2.0, 3.0, 5.0, 6.0]).unwrap();

        assert_eq!(
            lhs.sub(&rhs).unwrap().to_vec().unwrap(),
            vec![6.0, 6.0, 5.0, 6.0]
        );
        assert_eq!(
            lhs.div(&rhs).unwrap().to_vec().unwrap(),
            vec![4.0, 3.0, 2.0, 2.0]
        );
    }

    #[test]
    fn sum_reduces_to_scalar() {
        let tensor = Tensor2D::<2, 3>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let out = tensor.sum().unwrap();

        assert_eq!(out.shape().dims(), &[]);
        assert_eq!(out.rank(), 0);
        assert_eq!(out.numel(), 1);
        assert_eq!(out.to_vec().unwrap(), vec![21.0]);
    }

    #[test]
    fn reshape_with_shape_supports_symbolic_targets() {
        let tensor = Tensor1D::<6>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let reshaped = tensor
            .reshape_with_shape::<D2<Sym<Batch>, C<2>>>([3, 2])
            .unwrap();

        assert_eq!(reshaped.shape().dims(), &[3, 2]);
        assert!(reshaped.shares_storage_with(&tensor));
        assert_eq!(
            reshaped.to_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );

        let err = tensor
            .reshape_with_shape::<D2<Sym<Batch>, C<2>>>([2, 3])
            .unwrap_err();
        assert!(matches!(
            err,
            Error::Shape(ShapeError::DimMismatch {
                op: "validate",
                operand: 0,
                axis: 1,
                expected: 2,
                found: 3,
            })
        ));
    }

    #[test]
    fn flatten_uses_caller_named_static_target() {
        let tensor = Tensor2D::<2, 3>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let flattened = tensor.flatten::<6>().unwrap();

        assert_eq!(flattened.shape().dims(), &[6]);
        assert!(flattened.shares_storage_with(&tensor));
        assert_eq!(
            flattened.to_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );

        let err = tensor.flatten::<5>().unwrap_err();
        assert!(matches!(
            err,
            Error::Shape(ShapeError::LengthMismatch {
                expected: 5,
                found: 6,
            })
        ));
    }

    #[test]
    fn transpose_swaps_2d_shape_and_values() {
        let tensor = Tensor2D::<2, 3>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let transposed = tensor.transpose().unwrap();

        assert_eq!(transposed.shape().dims(), &[3, 2]);
        assert_eq!(transposed.layout().strides(), &[1, 3]);
        assert!(transposed.shares_storage_with(&tensor));
        assert!(!transposed.is_contiguous());
        assert_eq!(
            transposed.to_vec().unwrap(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn contiguous_materializes_non_contiguous_views() {
        let tensor = Tensor2D::<2, 3>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let transposed = tensor.transpose().unwrap();
        let contiguous = transposed.contiguous().unwrap();

        assert!(contiguous.is_contiguous());
        assert!(!contiguous.shares_storage_with(&tensor));
        assert_eq!(
            contiguous.to_vec().unwrap(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn ops_handle_non_contiguous_views() {
        let tensor = Tensor2D::<2, 3>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let view = tensor.transpose().unwrap();

        assert_eq!(
            view.add(&view).unwrap().to_vec().unwrap(),
            vec![2.0, 8.0, 4.0, 10.0, 6.0, 12.0]
        );
        assert_eq!(
            view.mul_scalar(2.0).unwrap().to_vec().unwrap(),
            vec![2.0, 8.0, 4.0, 10.0, 6.0, 12.0]
        );
        assert_eq!(view.sum().unwrap().to_vec().unwrap(), vec![21.0]);
    }

    #[test]
    fn matmul_handles_transposed_views() {
        let base = Tensor2D::<2, 3>::from_vec(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
        let lhs = base.transpose().unwrap();
        let rhs = Tensor2D::<2, 1>::from_vec(vec![10.0, 1.0]).unwrap();
        let out = lhs.matmul(&rhs).unwrap();

        assert_eq!(out.shape().dims(), &[3, 1]);
        assert_eq!(out.to_vec().unwrap(), vec![15.0, 43.0, 26.0]);
    }

    #[test]
    fn cat1_concatenates_1d_tensors_with_static_target() {
        let lhs = Tensor::<D1<Sym<Batch>>>::from_vec_with_shape(vec![1.0, 2.0], [2]).unwrap();
        let rhs = Tensor1D::<3>::from_vec(vec![3.0, 4.0, 5.0]).unwrap();
        let out = lhs.cat1::<C<3>, 5>(&rhs).unwrap();

        assert_eq!(out.shape().dims(), &[5]);
        assert_eq!(out.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let err = lhs.cat1::<C<3>, 4>(&rhs).unwrap_err();
        assert!(matches!(
            err,
            Error::Shape(ShapeError::LengthMismatch {
                expected: 4,
                found: 5,
            })
        ));
    }

    #[test]
    fn scalar_ops_compute_for_cpu_f32_and_preserve_metadata() {
        let tensor = Tensor2D::<2, 2>::from_vec(vec![2.0, 4.0, 6.0, 8.0]).unwrap();

        let add = tensor.add_scalar(1.5).unwrap();
        assert_eq!(add.to_vec().unwrap(), vec![3.5, 5.5, 7.5, 9.5]);
        assert_eq!(add.shape(), tensor.shape());
        assert_eq!(add.rank(), 2);
        assert_eq!(add.numel(), 4);
        assert_eq!(add.dtype(), DTypeId::F32);
        assert_eq!(add.device(), &CpuDevice::Cpu);
        assert_eq!(add.layout(), tensor.layout());
        assert_eq!(tensor.to_vec().unwrap(), vec![2.0, 4.0, 6.0, 8.0]);

        assert_eq!(
            tensor.sub_scalar(1.0).unwrap().to_vec().unwrap(),
            vec![1.0, 3.0, 5.0, 7.0]
        );
        assert_eq!(
            tensor.mul_scalar(0.5).unwrap().to_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            tensor.div_scalar(2.0).unwrap().to_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn scalar_ops_compute_for_cpu_f64() {
        let tensor = Tensor::<D1<C<3>>, f64, Cpu>::from_vec(vec![1.0, 2.0, 4.0]).unwrap();

        assert_eq!(
            tensor.add_scalar(2.0).unwrap().to_vec().unwrap(),
            vec![3.0, 4.0, 6.0]
        );
        assert_eq!(
            tensor.sub_scalar(0.5).unwrap().to_vec().unwrap(),
            vec![0.5, 1.5, 3.5]
        );
        assert_eq!(
            tensor.mul_scalar(3.0).unwrap().to_vec().unwrap(),
            vec![3.0, 6.0, 12.0]
        );
        assert_eq!(
            tensor.div_scalar(2.0).unwrap().to_vec().unwrap(),
            vec![0.5, 1.0, 2.0]
        );
    }

    #[test]
    fn scalar_ops_work_for_scalar_tensor() {
        let tensor = Scalar::<f32, Cpu>::from_vec(vec![3.0]).unwrap();
        let out = tensor.mul_scalar(2.0).unwrap();

        assert_eq!(out.shape().dims(), &[]);
        assert_eq!(out.rank(), 0);
        assert_eq!(out.numel(), 1);
        assert_eq!(out.to_vec().unwrap(), vec![6.0]);
    }

    #[test]
    fn scalar_ops_work_for_symbolic_shape() {
        let tensor = Tensor::<D2<Sym<Batch>, C<2>>>::from_vec_with_shape(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3, 2],
        )
        .unwrap();
        let out = tensor.add_scalar(10.0).unwrap();

        assert_eq!(out.shape().dims(), &[3, 2]);
        assert_eq!(
            out.to_vec().unwrap(),
            vec![11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
        );
    }

    #[test]
    fn multi_input_ops_reject_device_mismatch() {
        type TestTensor = Tensor<D1<C<2>>, f32, TestBackend>;

        let lhs = TestTensor::from_raw(
            RawTensor::from_vec_on(TestDevice::A, vec![1.0, 2.0], Shape::known([2])).unwrap(),
        )
        .unwrap();
        let rhs = TestTensor::from_raw(
            RawTensor::from_vec_on(TestDevice::B, vec![3.0, 4.0], Shape::known([2])).unwrap(),
        )
        .unwrap();

        let err = lhs.add(&rhs).unwrap_err();
        assert!(matches!(
            err,
            Error::Device(DeviceError::Mismatch {
                op: "add",
                lhs,
                rhs,
            }) if lhs == "A" && rhs == "B"
        ));
    }

    #[test]
    fn backend_errors_are_boxed_sources() {
        let err = match RawTensor::<f32, FailingBackend>::zeros(Shape::known([2])) {
            Ok(_) => panic!("expected backend error"),
            Err(err) => err,
        };
        assert!(matches!(err, Error::Backend(_)));
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn scalar_backend_errors_are_boxed_sources() {
        let raw = RawTensor::<f32, FailingBackend>::from_storage_on(
            TestDevice::A,
            vec![1.0, 2.0],
            Shape::known([2]),
        )
        .unwrap();
        let tensor = Tensor::<D1<C<2>>, f32, FailingBackend>::from_raw(raw).unwrap();

        let err = tensor.add_scalar(1.0).unwrap_err();
        assert!(matches!(err, Error::Backend(_)));
        assert!(std::error::Error::source(&err).is_some());
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    mod metal_tests {
        use super::*;
        use crate::backend::{Metal, MetalDevice};

        fn device() -> Option<MetalDevice> {
            <Metal as Backend<f32>>::default_device().ok()
        }

        fn tensor_1d(device: MetalDevice, values: Vec<f32>) -> Tensor<D1<C<4>>, f32, Metal> {
            let raw =
                RawTensor::<f32, Metal>::from_vec_on(device, values, Shape::known([4])).unwrap();
            Tensor::<D1<C<4>>, f32, Metal>::from_raw(raw).unwrap()
        }

        #[test]
        fn metal_scalar_add_round_trips() {
            let Some(device) = device() else {
                return;
            };
            let raw = RawTensor::<f32, Metal>::from_vec_on(
                device,
                vec![1.0, 2.0, 3.0, 4.0],
                Shape::known([2, 2]),
            )
            .unwrap();
            let tensor = Tensor::<D2<C<2>, C<2>>, f32, Metal>::from_raw(raw).unwrap();

            let out = tensor.add_scalar(1.0).unwrap();
            assert_eq!(out.to_vec().unwrap(), vec![2.0, 3.0, 4.0, 5.0]);
        }

        #[test]
        fn metal_scalar_sub_mul_and_div_round_trip() {
            let Some(device) = device() else {
                return;
            };
            let tensor = tensor_1d(device, vec![8.0, 9.0, 10.0, 12.0]);

            assert_eq!(
                tensor.sub_scalar(2.0).unwrap().to_vec().unwrap(),
                vec![6.0, 7.0, 8.0, 10.0]
            );
            assert_eq!(
                tensor.mul_scalar(3.0).unwrap().to_vec().unwrap(),
                vec![24.0, 27.0, 30.0, 36.0]
            );
            assert_eq!(
                tensor.div_scalar(2.0).unwrap().to_vec().unwrap(),
                vec![4.0, 4.5, 5.0, 6.0]
            );
        }

        #[test]
        fn metal_binary_ops_match_cpu() {
            let Some(device) = device() else {
                return;
            };
            let lhs_values = vec![8.0, 9.0, 10.0, 12.0];
            let rhs_values = vec![2.0, 3.0, 5.0, 6.0];
            let lhs = tensor_1d(device.clone(), lhs_values.clone());
            let rhs = tensor_1d(device.clone(), rhs_values.clone());
            let cpu_lhs = Tensor::<D1<C<4>>>::from_vec(lhs_values).unwrap();
            let cpu_rhs = Tensor::<D1<C<4>>>::from_vec(rhs_values).unwrap();

            assert_eq!(
                lhs.add(&rhs).unwrap().to_vec().unwrap(),
                cpu_lhs.add(&cpu_rhs).unwrap().to_vec().unwrap()
            );
            assert_eq!(
                lhs.mul(&rhs).unwrap().to_vec().unwrap(),
                cpu_lhs.mul(&cpu_rhs).unwrap().to_vec().unwrap()
            );

            let sub_storage = <Metal as Backend<f32>>::sub(
                lhs.device(),
                lhs.raw().storage(),
                rhs.raw().storage(),
                lhs.numel(),
            )
            .unwrap();
            let sub_raw = RawTensor::<f32, Metal>::from_storage_on(
                device.clone(),
                sub_storage,
                Shape::known([4]),
            )
            .unwrap();
            assert_eq!(sub_raw.to_vec().unwrap(), vec![6.0, 6.0, 5.0, 6.0]);

            let div_storage = <Metal as Backend<f32>>::div(
                lhs.device(),
                lhs.raw().storage(),
                rhs.raw().storage(),
                lhs.numel(),
            )
            .unwrap();
            let div_raw =
                RawTensor::<f32, Metal>::from_storage_on(device, div_storage, Shape::known([4]))
                    .unwrap();
            assert_eq!(div_raw.to_vec().unwrap(), vec![4.0, 3.0, 2.0, 2.0]);
        }

        #[test]
        fn metal_sum_uses_backend_reduction() {
            let Some(device) = device() else {
                return;
            };
            let tensor = tensor_1d(device, vec![1.0, 2.0, 3.0, 4.0]);

            let out = tensor.sum().unwrap();
            assert_eq!(out.shape().dims(), &[]);
            assert_eq!(out.to_vec().unwrap(), vec![10.0]);
        }

        #[test]
        fn metal_empty_sum_returns_zero_scalar() {
            let Some(device) = device() else {
                return;
            };
            let raw = RawTensor::<f32, Metal>::from_vec_on(device, Vec::new(), Shape::known([0]))
                .unwrap();
            let tensor = Tensor::<D1<C<0>>, f32, Metal>::from_raw(raw).unwrap();

            let out = tensor.sum().unwrap();
            assert_eq!(out.shape().dims(), &[]);
            assert_eq!(out.to_vec().unwrap(), vec![0.0]);
        }

        #[test]
        fn metal_transpose_and_reshape_with_shape_round_trip() {
            let Some(device) = device() else {
                return;
            };
            let raw = RawTensor::<f32, Metal>::from_vec_on(
                device,
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                Shape::known([2, 3]),
            )
            .unwrap();
            let tensor = Tensor::<D2<C<2>, C<3>>, f32, Metal>::from_raw(raw).unwrap();

            assert_eq!(
                tensor.transpose().unwrap().to_vec().unwrap(),
                vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
            );

            let reshaped = tensor
                .reshape_with_shape::<D2<Sym<Batch>, C<2>>>([3, 2])
                .unwrap();
            assert_eq!(reshaped.shape().dims(), &[3, 2]);
            assert_eq!(
                reshaped.to_vec().unwrap(),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            );
        }

        #[test]
        fn metal_matmul_matches_cpu() {
            let Some(device) = device() else {
                return;
            };
            let lhs_raw = RawTensor::<f32, Metal>::from_vec_on(
                device.clone(),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                Shape::known([2, 3]),
            )
            .unwrap();
            let rhs_raw = RawTensor::<f32, Metal>::from_vec_on(
                device,
                vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                Shape::known([3, 2]),
            )
            .unwrap();
            let lhs = Tensor::<D2<C<2>, C<3>>, f32, Metal>::from_raw(lhs_raw).unwrap();
            let rhs = Tensor::<D2<C<3>, C<2>>, f32, Metal>::from_raw(rhs_raw).unwrap();

            let out = lhs.matmul(&rhs).unwrap();
            assert_eq!(out.to_vec().unwrap(), vec![58.0, 64.0, 139.0, 154.0]);
        }

        #[test]
        fn metal_zeros_and_ones_have_expected_values_and_lengths() {
            let Some(device) = device() else {
                return;
            };

            let zeros =
                RawTensor::<f32, Metal>::zeros_on(device.clone(), Shape::known([3])).unwrap();
            assert_eq!(<Metal as Backend<f32>>::storage_len(zeros.storage()), 3);
            assert_eq!(zeros.to_vec().unwrap(), vec![0.0, 0.0, 0.0]);

            let ones = RawTensor::<f32, Metal>::ones_on(device, Shape::known([3])).unwrap();
            assert_eq!(<Metal as Backend<f32>>::storage_len(ones.storage()), 3);
            assert_eq!(ones.to_vec().unwrap(), vec![1.0, 1.0, 1.0]);
        }

        #[test]
        fn metal_preserves_typed_shape_facade() {
            let Some(device) = device() else {
                return;
            };
            let raw = RawTensor::<f32, Metal>::from_vec_on(
                device.clone(),
                vec![1.0, 2.0, 3.0, 4.0],
                Shape::known([2, 2]),
            )
            .unwrap();
            let tensor = Tensor::<D2<C<2>, C<2>>, f32, Metal>::from_raw(raw).unwrap();
            let out = tensor.mul_scalar(2.0).unwrap();

            assert_eq!(out.shape().dims(), &[2, 2]);
            assert_eq!(out.rank(), 2);
            assert_eq!(out.numel(), 4);
            assert_eq!(out.dtype(), DTypeId::F32);
            assert_eq!(out.device(), &device);
        }
    }
}
