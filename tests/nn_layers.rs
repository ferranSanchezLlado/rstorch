//! Behavior of neural-network layers (`LayerNorm`, `Dropout`).

use rstorch::Tensor;
use rstorch::prelude::*;

#[derive(Debug)]
struct Batch;

#[test]
fn layer_norm_centers_each_row() {
    let norm = LayerNorm::<3>::new(1e-5).unwrap();
    let input = Tensor::<D2<Sym<Batch>, C<3>>>::from_vec_with_shape(
        vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
        [2, 3],
    )
    .unwrap();
    let mut ctx = TrainContext::eval();

    let row_means = norm
        .forward(&input, &mut ctx)
        .unwrap()
        .mean_last()
        .unwrap()
        .to_vec()
        .unwrap();

    assert_close(row_means[0], 0.0, 1e-5);
    assert_close(row_means[1], 0.0, 1e-5);
}

#[test]
fn layer_norm_gradient_matches_finite_difference() {
    let input_data = vec![1.0, 2.0, 4.0, -1.0, 0.5, 3.0];
    let probe = Tensor2D::<2, 3, f64>::from_vec(vec![0.5, -0.2, 0.7, -0.4, 0.3, 0.1]).unwrap();
    let norm = LayerNorm::<3, f64>::new(1e-5).unwrap();
    let input = Tensor2D::<2, 3, f64>::from_vec(input_data.clone())
        .unwrap()
        .with_requires_grad(true);
    let mut ctx = TrainContext::eval();

    norm.forward(&input, &mut ctx)
        .unwrap()
        .mul(&probe)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();

    let grad = input.grad().unwrap().to_vec().unwrap();
    for (idx, &analytic) in grad.iter().enumerate() {
        let numerical = finite_difference(&input_data, idx, 1e-6, |values| {
            let mut ctx = TrainContext::eval();
            LayerNorm::<3, f64>::new(1e-5)
                .unwrap()
                .forward(
                    &Tensor2D::<2, 3, f64>::from_vec(values.to_vec()).unwrap(),
                    &mut ctx,
                )
                .unwrap()
                .mul(&probe)
                .unwrap()
                .sum()
                .unwrap()
                .to_vec()
                .unwrap()[0]
        });
        assert_close_f64(analytic, numerical, 1e-5);
    }
}

#[test]
fn dropout_scales_in_training_and_is_identity_in_eval() {
    let dropout = Dropout::new(0.5);
    let mut train_ctx = TrainContext::training(7);
    let dropped = dropout
        .forward(&Tensor1D::<4>::ones().unwrap(), &mut train_ctx)
        .unwrap()
        .to_vec()
        .unwrap();
    assert!(dropped.iter().all(|&value| value == 0.0 || value == 2.0));

    let mut eval_ctx = TrainContext::eval();
    assert_eq!(
        dropout
            .forward(&Tensor1D::<2>::ones().unwrap(), &mut eval_ctx)
            .unwrap()
            .to_vec()
            .unwrap(),
        vec![1.0, 1.0]
    );
}

fn assert_close(actual: f32, expected: f32, tol: f32) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {actual} within {tol} of {expected}"
    );
}

fn assert_close_f64(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {actual} within {tol} of {expected}"
    );
}

fn finite_difference(values: &[f64], idx: usize, eps: f64, f: impl Fn(&[f64]) -> f64) -> f64 {
    let mut plus = values.to_vec();
    plus[idx] += eps;
    let mut minus = values.to_vec();
    minus[idx] -= eps;
    (f(&plus) - f(&minus)) / (2.0 * eps)
}
