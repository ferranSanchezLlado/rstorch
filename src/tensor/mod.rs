//! Tensor type declarations for the restart architecture.

pub mod autograd;
pub mod ops;

use crate::backend::{Backend, Cpu};
use crate::dtype::FloatElement;
use crate::shape::{D0, D1, D2, Shape};
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::Mutex;

/// Error returned by checked tensor constructors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    InvalidLength { expected: usize, actual: usize },
}

/// Generic tensor with shared owned backend storage.
pub struct Tensor<S, E = f32, B = Cpu>
where
    S: Shape,
    E: FloatElement,
    B: Backend<E>,
{
    inner: Arc<TensorInner<S, E, B>>,
}

struct TensorInner<S, E, B>
where
    S: Shape,
    E: FloatElement,
    B: Backend<E>,
{
    id: autograd::NodeId,
    data: B::Storage,
    device: B::Device,
    requires_grad: bool,
    is_leaf: bool,
    grad: Arc<Mutex<Option<B::Storage>>>,
    grad_fn: Option<Arc<autograd::GradFn<E, B>>>,
    shape: PhantomData<S>,
}

impl<S, E, B> Clone for Tensor<S, E, B>
where
    S: Shape,
    E: FloatElement,
    B: Backend<E>,
{
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<S, E, B> Tensor<S, E, B>
where
    S: Shape,
    E: FloatElement,
    B: Backend<E>,
{
    fn from_storage(device: B::Device, data: B::Storage) -> Self {
        Self::from_storage_with_autograd(device, data, false, true, None)
    }

    fn from_storage_with_autograd(
        device: B::Device,
        data: B::Storage,
        requires_grad: bool,
        is_leaf: bool,
        grad_fn: Option<Arc<autograd::GradFn<E, B>>>,
    ) -> Self {
        Self {
            inner: Arc::new(TensorInner {
                id: autograd::next_node_id(),
                data,
                device,
                requires_grad,
                is_leaf,
                grad: Arc::new(Mutex::new(None)),
                grad_fn,
                shape: PhantomData,
            }),
        }
    }
}

impl<S, E, B> Tensor<S, E, B>
where
    S: Shape,
    E: FloatElement,
    B: Backend<E>,
    [(); S::NUMEL]:,
{
    pub fn zeros() -> Self {
        Self::zeros_on(B::default_device())
    }

    pub fn ones() -> Self {
        Self::ones_on(B::default_device())
    }

    pub fn zeros_on(device: B::Device) -> Self {
        let data = B::zeros::<{ S::NUMEL }>(&device);
        Self::from_storage(device, data)
    }

    pub fn ones_on(device: B::Device) -> Self {
        let data = B::ones::<{ S::NUMEL }>(&device);
        Self::from_storage(device, data)
    }

    pub fn from_vec(data: Vec<E>) -> Result<Self, TensorError> {
        let actual = data.len();
        if actual != S::NUMEL {
            return Err(TensorError::InvalidLength {
                expected: S::NUMEL,
                actual,
            });
        }

        let device = B::default_device();
        let data = B::from_vec(&device, data);
        Ok(Self::from_storage(device, data))
    }

    pub fn to_vec(&self) -> Vec<E> {
        B::to_vec(&self.inner.data)
    }

    pub fn shape(&self) -> &'static [usize] {
        S::dims()
    }

    pub fn numel(&self) -> usize {
        S::NUMEL
    }

    pub fn device(&self) -> &B::Device {
        &self.inner.device
    }
}

/// Scalar tensor alias.
pub type Scalar<E = f32, B = Cpu> = Tensor<D0, E, B>;

/// One-dimensional tensor alias.
pub type Tensor1D<const N: usize, E = f32, B = Cpu> = Tensor<D1<N>, E, B>;

/// Two-dimensional tensor alias.
pub type Tensor2D<const M: usize, const N: usize, E = f32, B = Cpu> = Tensor<D2<M, N>, E, B>;

impl<const N: usize, E, B> Tensor1D<N, E, B>
where
    E: FloatElement,
    B: Backend<E>,
    [(); N]:,
{
    pub fn from_array(data: [E; N]) -> Self {
        let device = B::default_device();
        let data = B::from_array(&device, data);
        Self::from_storage(device, data)
    }
}

impl<const M: usize, const N: usize, E, B> Tensor2D<M, N, E, B>
where
    E: FloatElement,
    B: Backend<E>,
    [(); M * N]:,
{
    pub fn from_array(data: [[E; N]; M]) -> Self {
        let device = B::default_device();
        let data = data.into_iter().flat_map(|row| row.into_iter()).collect();
        let data = B::from_vec(&device, data);
        Self::from_storage(device, data)
    }
}

#[cfg(test)]
mod tests {
    use super::{Tensor, Tensor1D, Tensor2D, TensorError};

    #[test]
    fn tensor2d_has_static_shape_metadata() {
        let tensor = Tensor2D::<32, 784>::zeros();

        assert_eq!(tensor.shape(), &[32, 784]);
        assert_eq!(tensor.numel(), 25_088);
    }

    #[test]
    fn default_tensor_dtype_is_f32() {
        let tensor: Tensor2D<2, 3> = Tensor::ones();
        let values: Vec<f32> = tensor.to_vec();

        assert_eq!(values, vec![1.0; 6]);
    }

    #[test]
    fn from_vec_rejects_wrong_lengths() {
        let result = Tensor2D::<2, 3>::from_vec(vec![1.0, 2.0]);

        assert!(matches!(
            result,
            Err(TensorError::InvalidLength {
                expected: 6,
                actual: 2,
            })
        ));
    }

    #[test]
    fn from_array_owns_1d_data() {
        let data = [1.0_f32, 2.0, 3.0];
        let tensor = Tensor1D::<3>::from_array(data);

        assert_eq!(tensor.to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn from_array_owns_2d_data() {
        let data = [[1.0_f32, 2.0], [3.0, 4.0]];
        let tensor = Tensor2D::<2, 2>::from_array(data);

        assert_eq!(tensor.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn cloning_preserves_values() {
        let tensor = Tensor2D::<2, 2>::from_array([[1.0_f32, 2.0], [3.0, 4.0]]);
        let clone = tensor.clone();

        assert_eq!(clone.to_vec(), tensor.to_vec());
    }
}
