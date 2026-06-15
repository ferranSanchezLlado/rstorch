//! Forward tensor operations and their first-order backward formulas.

use super::autograd::{self, AnyTensor, GradFn};
use super::{Scalar, Tensor, Tensor1D, Tensor2D};
use crate::backend::Backend;
use crate::dtype::FloatElement;
use crate::shape::Shape;
use std::sync::Arc;

impl<S, E, B> Tensor<S, E, B>
where
    S: Shape,
    E: FloatElement,
    B: Backend<E>,
    [(); S::NUMEL]:,
{
    pub fn add(&self, rhs: &Self) -> Self {
        let device = self.inner.device.clone();
        let data = B::add::<{ S::NUMEL }>(&device, &self.inner.data, &rhs.inner.data);
        let requires_grad =
            autograd::should_track_grad(self.inner.requires_grad || rhs.inner.requires_grad);
        let grad_fn = requires_grad.then(|| {
            Arc::new(GradFn {
                parents: vec![AnyTensor::from_tensor(self), AnyTensor::from_tensor(rhs)],
                backward: Box::new(|grad| vec![grad.clone(), grad.clone()]),
            })
        });
        Self::from_storage_with_autograd(device, data, requires_grad, !requires_grad, grad_fn)
    }

    pub fn sub(&self, rhs: &Self) -> Self {
        let device = self.inner.device.clone();
        let data = B::sub::<{ S::NUMEL }>(&device, &self.inner.data, &rhs.inner.data);
        let backward_device = device.clone();
        let requires_grad =
            autograd::should_track_grad(self.inner.requires_grad || rhs.inner.requires_grad);
        let grad_fn = requires_grad.then(|| {
            Arc::new(GradFn {
                parents: vec![AnyTensor::from_tensor(self), AnyTensor::from_tensor(rhs)],
                backward: Box::new(move |grad| {
                    let negative =
                        B::mul_scalar::<{ S::NUMEL }>(&backward_device, grad, E::zero() - E::one());
                    vec![grad.clone(), negative]
                }),
            })
        });
        Self::from_storage_with_autograd(device, data, requires_grad, !requires_grad, grad_fn)
    }

    pub fn mul(&self, rhs: &Self) -> Self {
        let device = self.inner.device.clone();
        let data = B::mul::<{ S::NUMEL }>(&device, &self.inner.data, &rhs.inner.data);
        let lhs_data = self.inner.data.clone();
        let rhs_data = rhs.inner.data.clone();
        let backward_device = device.clone();
        let requires_grad =
            autograd::should_track_grad(self.inner.requires_grad || rhs.inner.requires_grad);
        let grad_fn = requires_grad.then(|| {
            Arc::new(GradFn {
                parents: vec![AnyTensor::from_tensor(self), AnyTensor::from_tensor(rhs)],
                backward: Box::new(move |grad| {
                    vec![
                        B::mul::<{ S::NUMEL }>(&backward_device, grad, &rhs_data),
                        B::mul::<{ S::NUMEL }>(&backward_device, grad, &lhs_data),
                    ]
                }),
            })
        });
        Self::from_storage_with_autograd(device, data, requires_grad, !requires_grad, grad_fn)
    }

    pub fn div(&self, rhs: &Self) -> Self {
        let device = self.inner.device.clone();
        let data = B::div::<{ S::NUMEL }>(&device, &self.inner.data, &rhs.inner.data);
        let lhs_data = self.inner.data.clone();
        let rhs_data = rhs.inner.data.clone();
        let backward_device = device.clone();
        let requires_grad =
            autograd::should_track_grad(self.inner.requires_grad || rhs.inner.requires_grad);
        let grad_fn = requires_grad.then(|| {
            Arc::new(GradFn {
                parents: vec![AnyTensor::from_tensor(self), AnyTensor::from_tensor(rhs)],
                backward: Box::new(move |grad| {
                    let dx = B::div::<{ S::NUMEL }>(&backward_device, grad, &rhs_data);
                    let rhs_squared =
                        B::powf::<{ S::NUMEL }>(&backward_device, &rhs_data, E::from_usize(2));
                    let numerator = B::mul::<{ S::NUMEL }>(&backward_device, grad, &lhs_data);
                    let dy = B::div::<{ S::NUMEL }>(&backward_device, &numerator, &rhs_squared);
                    let dy =
                        B::mul_scalar::<{ S::NUMEL }>(&backward_device, &dy, E::zero() - E::one());
                    vec![dx, dy]
                }),
            })
        });
        Self::from_storage_with_autograd(device, data, requires_grad, !requires_grad, grad_fn)
    }

    pub fn add_scalar(&self, rhs: E) -> Self {
        let device = self.inner.device.clone();
        let data = B::add_scalar::<{ S::NUMEL }>(&device, &self.inner.data, rhs);
        let requires_grad = autograd::should_track_grad(self.inner.requires_grad);
        let grad_fn = requires_grad.then(|| {
            Arc::new(GradFn {
                parents: vec![AnyTensor::from_tensor(self)],
                backward: Box::new(|grad| vec![grad.clone()]),
            })
        });
        Self::from_storage_with_autograd(device, data, requires_grad, !requires_grad, grad_fn)
    }

    pub fn sub_scalar(&self, rhs: E) -> Self {
        let device = self.inner.device.clone();
        let data = B::sub_scalar::<{ S::NUMEL }>(&device, &self.inner.data, rhs);
        let requires_grad = autograd::should_track_grad(self.inner.requires_grad);
        let grad_fn = requires_grad.then(|| {
            Arc::new(GradFn {
                parents: vec![AnyTensor::from_tensor(self)],
                backward: Box::new(|grad| vec![grad.clone()]),
            })
        });
        Self::from_storage_with_autograd(device, data, requires_grad, !requires_grad, grad_fn)
    }

    pub fn mul_scalar(&self, rhs: E) -> Self {
        let device = self.inner.device.clone();
        let data = B::mul_scalar::<{ S::NUMEL }>(&device, &self.inner.data, rhs);
        let backward_device = device.clone();
        let requires_grad = autograd::should_track_grad(self.inner.requires_grad);
        let grad_fn = requires_grad.then(|| {
            Arc::new(GradFn {
                parents: vec![AnyTensor::from_tensor(self)],
                backward: Box::new(move |grad| {
                    vec![B::mul_scalar::<{ S::NUMEL }>(&backward_device, grad, rhs)]
                }),
            })
        });
        Self::from_storage_with_autograd(device, data, requires_grad, !requires_grad, grad_fn)
    }

    pub fn div_scalar(&self, rhs: E) -> Self {
        let device = self.inner.device.clone();
        let data = B::div_scalar::<{ S::NUMEL }>(&device, &self.inner.data, rhs);
        let backward_device = device.clone();
        let requires_grad = autograd::should_track_grad(self.inner.requires_grad);
        let grad_fn = requires_grad.then(|| {
            Arc::new(GradFn {
                parents: vec![AnyTensor::from_tensor(self)],
                backward: Box::new(move |grad| {
                    vec![B::div_scalar::<{ S::NUMEL }>(&backward_device, grad, rhs)]
                }),
            })
        });
        Self::from_storage_with_autograd(device, data, requires_grad, !requires_grad, grad_fn)
    }

    pub fn powf(&self, exponent: E) -> Self {
        let device = self.inner.device.clone();
        let data = B::powf::<{ S::NUMEL }>(&device, &self.inner.data, exponent);
        let input_data = self.inner.data.clone();
        let backward_device = device.clone();
        let requires_grad = autograd::should_track_grad(self.inner.requires_grad);
        let grad_fn = requires_grad.then(|| {
            Arc::new(GradFn {
                parents: vec![AnyTensor::from_tensor(self)],
                backward: Box::new(move |grad| {
                    let exponent_minus_one = exponent - E::one();
                    let power =
                        B::powf::<{ S::NUMEL }>(&backward_device, &input_data, exponent_minus_one);
                    let scaled = B::mul_scalar::<{ S::NUMEL }>(&backward_device, &power, exponent);
                    vec![B::mul::<{ S::NUMEL }>(&backward_device, grad, &scaled)]
                }),
            })
        });
        Self::from_storage_with_autograd(device, data, requires_grad, !requires_grad, grad_fn)
    }

    pub fn relu(&self) -> Self {
        let device = self.inner.device.clone();
        let data = B::relu::<{ S::NUMEL }>(&device, &self.inner.data);
        let input_data = self.inner.data.clone();
        let backward_device = device.clone();
        let requires_grad = autograd::should_track_grad(self.inner.requires_grad);
        let grad_fn = requires_grad.then(|| {
            Arc::new(GradFn {
                parents: vec![AnyTensor::from_tensor(self)],
                backward: Box::new(move |grad| {
                    let input = B::to_vec(&input_data);
                    let grad = B::to_vec(grad);
                    let data = input
                        .into_iter()
                        .zip(grad)
                        .map(|(input, grad)| if input > E::zero() { grad } else { E::zero() })
                        .collect();
                    vec![B::from_vec(&backward_device, data)]
                }),
            })
        });
        Self::from_storage_with_autograd(device, data, requires_grad, !requires_grad, grad_fn)
    }

    pub fn exp(&self) -> Self {
        let device = self.inner.device.clone();
        let data = B::exp::<{ S::NUMEL }>(&device, &self.inner.data);
        let output_data = data.clone();
        let backward_device = device.clone();
        let requires_grad = autograd::should_track_grad(self.inner.requires_grad);
        let grad_fn = requires_grad.then(|| {
            Arc::new(GradFn {
                parents: vec![AnyTensor::from_tensor(self)],
                backward: Box::new(move |grad| {
                    vec![B::mul::<{ S::NUMEL }>(&backward_device, grad, &output_data)]
                }),
            })
        });
        Self::from_storage_with_autograd(device, data, requires_grad, !requires_grad, grad_fn)
    }

    pub fn ln(&self) -> Self {
        let device = self.inner.device.clone();
        let data = B::ln::<{ S::NUMEL }>(&device, &self.inner.data);
        let input_data = self.inner.data.clone();
        let backward_device = device.clone();
        let requires_grad = autograd::should_track_grad(self.inner.requires_grad);
        let grad_fn = requires_grad.then(|| {
            Arc::new(GradFn {
                parents: vec![AnyTensor::from_tensor(self)],
                backward: Box::new(move |grad| {
                    vec![B::div::<{ S::NUMEL }>(&backward_device, grad, &input_data)]
                }),
            })
        });
        Self::from_storage_with_autograd(device, data, requires_grad, !requires_grad, grad_fn)
    }

    pub fn sum(&self) -> Scalar<E, B> {
        let device = self.inner.device.clone();
        let data = B::sum::<{ S::NUMEL }>(&device, &self.inner.data);
        let backward_device = device.clone();
        let requires_grad = autograd::should_track_grad(self.inner.requires_grad);
        let grad_fn = requires_grad.then(|| {
            Arc::new(GradFn {
                parents: vec![AnyTensor::from_tensor(self)],
                backward: Box::new(move |grad| {
                    let seed = B::to_vec(grad)[0];
                    vec![B::from_vec(&backward_device, vec![seed; S::NUMEL])]
                }),
            })
        });
        Tensor::from_storage_with_autograd(device, data, requires_grad, !requires_grad, grad_fn)
    }

    pub fn mean(&self) -> Scalar<E, B> {
        let device = self.inner.device.clone();
        let data = B::mean::<{ S::NUMEL }>(&device, &self.inner.data);
        let backward_device = device.clone();
        let requires_grad = autograd::should_track_grad(self.inner.requires_grad);
        let grad_fn = requires_grad.then(|| {
            Arc::new(GradFn {
                parents: vec![AnyTensor::from_tensor(self)],
                backward: Box::new(move |grad| {
                    let seed = B::to_vec(grad)[0] / E::from_usize(S::NUMEL);
                    vec![B::from_vec(&backward_device, vec![seed; S::NUMEL])]
                }),
            })
        });
        Tensor::from_storage_with_autograd(device, data, requires_grad, !requires_grad, grad_fn)
    }
}

impl<const M: usize, const K: usize, E, B> Tensor2D<M, K, E, B>
where
    E: FloatElement,
    B: Backend<E>,
    [(); M * K]:,
{
    pub fn matmul<const N: usize>(&self, rhs: &Tensor2D<K, N, E, B>) -> Tensor2D<M, N, E, B>
    where
        [(); K * N]:,
        [(); N * K]:,
        [(); M * N]:,
        [(); K * M]:,
    {
        let device = self.inner.device.clone();
        let data = B::matmul::<M, K, N>(&device, &self.inner.data, &rhs.inner.data);
        let lhs_data = self.inner.data.clone();
        let rhs_data = rhs.inner.data.clone();
        let backward_device = device.clone();
        let requires_grad =
            autograd::should_track_grad(self.inner.requires_grad || rhs.inner.requires_grad);
        let grad_fn = requires_grad.then(|| {
            Arc::new(GradFn {
                parents: vec![AnyTensor::from_tensor(self), AnyTensor::from_tensor(rhs)],
                backward: Box::new(move |grad| {
                    let rhs_t = B::transpose::<K, N>(&backward_device, &rhs_data);
                    let lhs_t = B::transpose::<M, K>(&backward_device, &lhs_data);
                    vec![
                        B::matmul::<M, N, K>(&backward_device, grad, &rhs_t),
                        B::matmul::<K, M, N>(&backward_device, &lhs_t, grad),
                    ]
                }),
            })
        });
        Tensor::from_storage_with_autograd(device, data, requires_grad, !requires_grad, grad_fn)
    }
}

impl<const M: usize, const N: usize, E, B> Tensor2D<M, N, E, B>
where
    E: FloatElement,
    B: Backend<E>,
    [(); M * N]:,
{
    pub fn transpose(&self) -> Tensor2D<N, M, E, B>
    where
        [(); N * M]:,
    {
        let device = self.inner.device.clone();
        let data = B::transpose::<M, N>(&device, &self.inner.data);
        let backward_device = device.clone();
        let requires_grad = autograd::should_track_grad(self.inner.requires_grad);
        let grad_fn = requires_grad.then(|| {
            Arc::new(GradFn {
                parents: vec![AnyTensor::from_tensor(self)],
                backward: Box::new(move |grad| vec![B::transpose::<N, M>(&backward_device, grad)]),
            })
        });
        Tensor::from_storage_with_autograd(device, data, requires_grad, !requires_grad, grad_fn)
    }

    pub fn flatten(&self) -> Tensor1D<{ M * N }, E, B> {
        let device = self.inner.device.clone();
        let data = self.inner.data.clone();
        let requires_grad = autograd::should_track_grad(self.inner.requires_grad);
        let grad_fn = requires_grad.then(|| {
            Arc::new(GradFn {
                parents: vec![AnyTensor::from_tensor(self)],
                backward: Box::new(|grad| vec![grad.clone()]),
            })
        });
        Tensor::from_storage_with_autograd(device, data, requires_grad, !requires_grad, grad_fn)
    }

    pub fn add_row(&self, row: &Tensor1D<N, E, B>) -> Self
    where
        [(); N]:,
    {
        let device = self.inner.device.clone();
        let data = B::add_row::<M, N>(&device, &self.inner.data, &row.inner.data);
        let backward_device = device.clone();
        let requires_grad =
            autograd::should_track_grad(self.inner.requires_grad || row.inner.requires_grad);
        let grad_fn = requires_grad.then(|| {
            Arc::new(GradFn {
                parents: vec![AnyTensor::from_tensor(self), AnyTensor::from_tensor(row)],
                backward: Box::new(move |grad| {
                    let grad_values = B::to_vec(grad);
                    let mut row_grad = vec![E::zero(); N];
                    for matrix_row in 0..M {
                        for col in 0..N {
                            row_grad[col] = row_grad[col] + grad_values[matrix_row * N + col];
                        }
                    }
                    vec![grad.clone(), B::from_vec(&backward_device, row_grad)]
                }),
            })
        });
        Self::from_storage_with_autograd(device, data, requires_grad, !requires_grad, grad_fn)
    }

    pub fn add_col(&self, col: &Tensor1D<M, E, B>) -> Self
    where
        [(); M]:,
    {
        let device = self.inner.device.clone();
        let data = B::add_col::<M, N>(&device, &self.inner.data, &col.inner.data);
        let backward_device = device.clone();
        let requires_grad =
            autograd::should_track_grad(self.inner.requires_grad || col.inner.requires_grad);
        let grad_fn = requires_grad.then(|| {
            Arc::new(GradFn {
                parents: vec![AnyTensor::from_tensor(self), AnyTensor::from_tensor(col)],
                backward: Box::new(move |grad| {
                    let grad_values = B::to_vec(grad);
                    let mut col_grad = vec![E::zero(); M];
                    for row in 0..M {
                        for matrix_col in 0..N {
                            col_grad[row] = col_grad[row] + grad_values[row * N + matrix_col];
                        }
                    }
                    vec![grad.clone(), B::from_vec(&backward_device, col_grad)]
                }),
            })
        });
        Self::from_storage_with_autograd(device, data, requires_grad, !requires_grad, grad_fn)
    }
}

#[cfg(test)]
mod tests {
    use super::{Tensor1D, Tensor2D};
    use crate::backend::Cpu;

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn elementwise_ops_compute_values() {
        let lhs = Tensor1D::<4>::from_array([8.0, 6.0, 4.0, 2.0]);
        let rhs = Tensor1D::<4>::from_array([4.0, 3.0, 2.0, 1.0]);

        assert_eq!(lhs.add(&rhs).to_vec(), vec![12.0, 9.0, 6.0, 3.0]);
        assert_eq!(lhs.sub(&rhs).to_vec(), vec![4.0, 3.0, 2.0, 1.0]);
        assert_eq!(lhs.mul(&rhs).to_vec(), vec![32.0, 18.0, 8.0, 2.0]);
        assert_eq!(lhs.div(&rhs).to_vec(), vec![2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn scalar_ops_compute_values() {
        let tensor = Tensor1D::<3>::from_array([2.0, 4.0, 8.0]);

        assert_eq!(tensor.add_scalar(1.0).to_vec(), vec![3.0, 5.0, 9.0]);
        assert_eq!(tensor.sub_scalar(1.0).to_vec(), vec![1.0, 3.0, 7.0]);
        assert_eq!(tensor.mul_scalar(2.0).to_vec(), vec![4.0, 8.0, 16.0]);
        assert_eq!(tensor.div_scalar(2.0).to_vec(), vec![1.0, 2.0, 4.0]);
        assert_eq!(tensor.powf(2.0).to_vec(), vec![4.0, 16.0, 64.0]);
    }

    #[test]
    fn matmul_computes_known_matrix_product() {
        let lhs = Tensor2D::<2, 3>::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let rhs = Tensor2D::<3, 2>::from_array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);

        let product: Tensor2D<2, 2> = lhs.matmul(&rhs);

        assert_eq!(product.to_vec(), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn transpose_swaps_2d_axes() {
        let tensor = Tensor2D::<2, 3>::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let transposed: Tensor2D<3, 2> = tensor.transpose();

        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(transposed.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn full_reductions_return_scalars() {
        let tensor = Tensor2D::<2, 3>::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        assert_eq!(tensor.sum().shape(), &[]);
        assert_eq!(tensor.sum().to_vec(), vec![21.0]);
        assert_eq!(tensor.mean().to_vec(), vec![3.5]);
    }

    #[test]
    fn unary_ops_compute_values() {
        let tensor = Tensor1D::<3>::from_array([-1.0, 0.0, 1.0]);
        assert_eq!(tensor.relu().to_vec(), vec![0.0, 0.0, 1.0]);

        let positive = Tensor1D::<2>::from_array([1.0, 4.0]);
        assert_close(&positive.exp().ln().to_vec(), &[1.0, 4.0]);
    }

    #[test]
    fn flatten_preserves_shape_and_values() {
        let tensor = Tensor2D::<2, 3>::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let flattened: Tensor1D<6> = tensor.flatten();

        assert_eq!(flattened.shape(), &[6]);
        assert_eq!(flattened.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn explicit_row_and_column_addition_compute_values() {
        let tensor = Tensor2D::<2, 3>::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let row = Tensor1D::<3>::from_array([10.0, 20.0, 30.0]);
        let col = Tensor1D::<2>::from_array([100.0, 200.0]);

        assert_eq!(
            tensor.add_row(&row).to_vec(),
            vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]
        );
        assert_eq!(
            tensor.add_col(&col).to_vec(),
            vec![101.0, 102.0, 103.0, 204.0, 205.0, 206.0]
        );
    }

    #[test]
    fn operations_preserve_dtype_and_backend() {
        let lhs: Tensor1D<2, f64, Cpu> = Tensor1D::from_array([1.0, 2.0]);
        let rhs: Tensor1D<2, f64, Cpu> = Tensor1D::from_array([3.0, 4.0]);

        let result: Tensor1D<2, f64, Cpu> = lhs.add(&rhs);

        assert_eq!(result.to_vec(), vec![4.0, 6.0]);
    }
}
