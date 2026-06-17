//! Differentiable loss functions for training loops.

use crate::backend::Backend;
use crate::dtype::FloatElement;
use crate::shape::Shape;
use crate::tensor::{Scalar, Tensor};

/// Returns mean squared error: `mean((prediction - target)^2)`.
pub fn mse_loss<S, E, B>(prediction: &Tensor<S, E, B>, target: &Tensor<S, E, B>) -> Scalar<E, B>
where
    S: Shape,
    E: FloatElement,
    B: Backend<E>,
{
    prediction.sub(target).powf(E::from_usize(2)).mean()
}

/// Returns binary cross entropy with mean reduction.
///
/// `prediction` is expected to contain probabilities strictly between 0 and 1.
/// This function does not clamp inputs or add an epsilon policy.
pub fn binary_cross_entropy<S, E, B>(
    prediction: &Tensor<S, E, B>,
    target: &Tensor<S, E, B>,
) -> Scalar<E, B>
where
    S: Shape,
    E: FloatElement,
    B: Backend<E>,
{
    let one = E::one();
    let negative_one = -one;
    let one_minus_target = target.mul_scalar(negative_one).add_scalar(one);
    let one_minus_prediction = prediction.mul_scalar(negative_one).add_scalar(one);

    target
        .mul(&prediction.ln())
        .add(&one_minus_target.mul(&one_minus_prediction.ln()))
        .mul_scalar(negative_one)
        .mean()
}

#[cfg(test)]
mod tests {
    use super::{binary_cross_entropy, mse_loss};
    use crate::backend::Cpu;
    use crate::tensor::{Scalar, Tensor1D, Tensor2D};

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    }

    #[test]
    fn mse_loss_returns_expected_scalar_value() {
        let prediction = Tensor2D::<2, 2, f64>::from_array([[1.0, 3.0], [5.0, 7.0]]);
        let target = Tensor2D::<2, 2, f64>::from_array([[0.0, 1.0], [2.0, 3.0]]);

        let loss = mse_loss(&prediction, &target);

        assert_eq!(loss.shape(), &[]);
        assert_close(loss.to_vec()[0], 7.5);
    }

    #[test]
    fn binary_cross_entropy_returns_expected_scalar_value() {
        let prediction = Tensor1D::<2, f64>::from_array([0.25, 0.75]);
        let target = Tensor1D::<2, f64>::from_array([0.0, 1.0]);

        let loss = binary_cross_entropy(&prediction, &target);

        assert_close(loss.to_vec()[0], -0.75_f64.ln());
    }

    #[test]
    fn losses_preserve_dtype_and_backend_type() {
        let prediction: Tensor1D<2, f64, Cpu> = Tensor1D::from_array([0.25, 0.75]);
        let target: Tensor1D<2, f64, Cpu> = Tensor1D::from_array([0.0, 1.0]);

        let mse: Scalar<f64, Cpu> = mse_loss(&prediction, &target);
        let bce: Scalar<f64, Cpu> = binary_cross_entropy(&prediction, &target);

        assert_eq!(mse.shape(), &[]);
        assert_eq!(bce.shape(), &[]);
    }
}
