use rstorch::prelude::*;

#[test]
fn cross_entropy_options_ignore_index_matches_value_and_gradient_rules() {
    let logits = Tensor2D::<3, 3>::from_vec(vec![1.0, 2.0, 3.0, 9.0, 8.0, 7.0, 0.5, 1.0, -0.5])
        .unwrap()
        .with_requires_grad(true);
    let targets = [2, 99, 1];

    let mut opts = CrossEntropyOpts::default();
    opts.ignore_index = Some(99);
    let loss = logits.cross_entropy_with(&targets, opts).unwrap();
    let expected = (row_loss(&[1.0, 2.0, 3.0], 2) + row_loss(&[0.5, 1.0, -0.5], 1)) / 2.0;
    assert_close(loss.to_vec().unwrap()[0], expected, 1e-6);

    loss.backward().unwrap();
    let grad = logits.grad().unwrap().to_vec().unwrap();
    assert_eq!(&grad[3..6], &[0.0, 0.0, 0.0]);

    let eps = 1e-3;
    let mut plus = vec![1.0, 2.0, 3.0, 9.0, 8.0, 7.0, 0.5, 1.0, -0.5];
    let mut minus = plus.clone();
    plus[0] += eps;
    minus[0] -= eps;
    let plus_loss = Tensor2D::<3, 3>::from_vec(plus)
        .unwrap()
        .cross_entropy_with(&targets, opts)
        .unwrap()
        .to_vec()
        .unwrap()[0];
    let minus_loss = Tensor2D::<3, 3>::from_vec(minus)
        .unwrap()
        .cross_entropy_with(&targets, opts)
        .unwrap()
        .to_vec()
        .unwrap()[0];
    assert_close(grad[0], (plus_loss - minus_loss) / (2.0 * eps), 1e-3);
}

#[test]
fn new_losses_and_label_smoothing_match_expected_values() {
    let pred = Tensor1D::<2>::from_vec(vec![0.0, 2.0]).unwrap();
    let target = Tensor1D::<2>::from_vec(vec![0.0, 1.0]).unwrap();

    assert_close(
        l1_loss(&pred, &target, Reduction::Mean)
            .unwrap()
            .item()
            .unwrap(),
        0.5,
        1e-6,
    );
    assert_close(
        huber_loss(&pred, &target, 1.0, Reduction::Sum)
            .unwrap()
            .item()
            .unwrap(),
        0.5,
        1e-6,
    );
    assert_close(
        bce_with_logits_loss(&pred, &target, Reduction::Mean)
            .unwrap()
            .item()
            .unwrap(),
        (2.0f32.ln() + (1.0 + (-2.0f32).exp()).ln()) / 2.0,
        1e-6,
    );

    let logits = Tensor2D::<1, 3>::from_vec(vec![1.0, 2.0, 3.0]).unwrap();
    let mut opts = CrossEntropyOpts::default();
    opts.label_smoothing = 0.1;
    let smoothed = logits
        .cross_entropy_with(&[2], opts)
        .unwrap()
        .item()
        .unwrap();
    let expected = 0.9 * row_loss(&[1.0, 2.0, 3.0], 2)
        + 0.1
            * (row_loss(&[1.0, 2.0, 3.0], 0)
                + row_loss(&[1.0, 2.0, 3.0], 1)
                + row_loss(&[1.0, 2.0, 3.0], 2))
            / 3.0;
    assert_close(smoothed, expected, 1e-6);
}

fn assert_close(actual: f32, expected: f32, tol: f32) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {actual} to be within {tol} of {expected}"
    );
}

fn row_loss(row: &[f32], target: usize) -> f32 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = row.iter().map(|value| (*value - max).exp()).sum();
    -(row[target] - (max + sum.ln()))
}
