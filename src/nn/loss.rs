//! Differentiable loss functions for training loops.

use crate::backend::Backend;
use crate::const_check::nonzero;
use crate::dtype::FloatElement;
use crate::shape::Shape;
use crate::tensor::{Scalar, Tensor, Tensor2D};

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

/// Returns mean cross entropy from logits and one-hot classification targets.
pub fn cross_entropy_one_hot<const BATCH: usize, const CLASSES: usize, E, B>(
    logits: &Tensor2D<BATCH, CLASSES, E, B>,
    targets: &Tensor2D<BATCH, CLASSES, E, B>,
) -> Scalar<E, B>
where
    E: FloatElement,
    B: Backend<E>,
    [(); nonzero(BATCH, "cross_entropy_one_hot", "BATCH")]:,
    [(); nonzero(CLASSES, "cross_entropy_one_hot", "CLASSES")]:,
    [(); nonzero(CLASSES, "log_softmax_rows", "N")]:,
{
    targets
        .mul(&logits.log_softmax_rows())
        .sum()
        .mul_scalar(-E::one() / E::from_usize(BATCH))
}

#[cfg(test)]
mod tests {
    use super::{binary_cross_entropy, cross_entropy_one_hot, mse_loss};
    use crate::backend::Cpu;
    use crate::tensor::{Scalar, Tensor1D, Tensor2D};

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    }

    fn assert_close_slice(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() < tolerance,
                "{actual} != {expected}"
            );
        }
    }

    fn finite_difference<F>(values: &[f64], f: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        let epsilon = 1e-6;
        let mut gradient = Vec::with_capacity(values.len());

        for index in 0..values.len() {
            let mut plus = values.to_vec();
            plus[index] += epsilon;
            let mut minus = values.to_vec();
            minus[index] -= epsilon;
            gradient.push((f(&plus) - f(&minus)) / (2.0 * epsilon));
        }

        gradient
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
    fn cross_entropy_one_hot_prefers_better_logits() {
        let targets = Tensor2D::<2, 3, f64>::from_array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
        let good_logits = Tensor2D::<2, 3, f64>::from_array([[5.0, 0.0, -1.0], [-1.0, 0.0, 5.0]]);
        let bad_logits = Tensor2D::<2, 3, f64>::from_array([[0.0, 5.0, -1.0], [-1.0, 5.0, 0.0]]);

        let good_loss = cross_entropy_one_hot(&good_logits, &targets).to_vec()[0];
        let bad_loss = cross_entropy_one_hot(&bad_logits, &targets).to_vec()[0];

        assert!(
            good_loss < bad_loss,
            "{good_loss} should be less than {bad_loss}"
        );
    }

    #[test]
    fn cross_entropy_one_hot_gradients_match_finite_differences() {
        let targets = Tensor2D::<2, 3, f64>::from_array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        let values = [1.0, 0.0, -1.0, -0.5, 0.5, 1.5];
        let logits =
            Tensor2D::<2, 3, f64>::from_array([[1.0, 0.0, -1.0], [-0.5, 0.5, 1.5]]).requires_grad();

        cross_entropy_one_hot(&logits, &targets).backward();
        let expected = finite_difference(&values, |values| {
            let logits = Tensor2D::<2, 3, f64>::from_vec(values.to_vec()).unwrap();
            cross_entropy_one_hot(&logits, &targets).to_vec()[0]
        });

        assert_close_slice(&logits.grad().unwrap().to_vec(), &expected, 1e-6);
    }

    #[test]
    fn losses_preserve_dtype_and_backend_type() {
        let prediction: Tensor1D<2, f64, Cpu> = Tensor1D::from_array([0.25, 0.75]);
        let target: Tensor1D<2, f64, Cpu> = Tensor1D::from_array([0.0, 1.0]);

        let mse: Scalar<f64, Cpu> = mse_loss(&prediction, &target);
        let bce: Scalar<f64, Cpu> = binary_cross_entropy(&prediction, &target);
        let logits: Tensor2D<1, 2, f64, Cpu> = Tensor2D::from_array([[1.0, 2.0]]);
        let labels: Tensor2D<1, 2, f64, Cpu> = Tensor2D::from_array([[0.0, 1.0]]);
        let ce: Scalar<f64, Cpu> = cross_entropy_one_hot(&logits, &labels);

        assert_eq!(mse.shape(), &[]);
        assert_eq!(bce.shape(), &[]);
        assert_eq!(ce.shape(), &[]);
    }
}
