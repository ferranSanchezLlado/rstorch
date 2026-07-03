//! Forward and autograd behavior of the typed tensor op surface.

use rstorch::Tensor;
use rstorch::prelude::*;

#[test]
fn numeric_ops_are_differentiable() {
    let x = Tensor1D::<2>::from_vec(vec![-1.0, 0.5])
        .unwrap()
        .with_requires_grad(true);

    x.exp()
        .unwrap()
        .ln()
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();

    assert_close(x.grad().unwrap().to_vec().unwrap()[0], 1.0, 1e-6);
    assert_close(x.grad().unwrap().to_vec().unwrap()[1], 1.0, 1e-6);

    assert_eq!(x.neg().unwrap().to_vec().unwrap(), vec![1.0, -0.5]);
    assert_close(
        x.sigmoid().unwrap().to_vec().unwrap()[1],
        0.622_459_35,
        1e-6,
    );
    assert_close(x.tanh().unwrap().to_vec().unwrap()[1], 0.462_117_17, 1e-6);
}

#[test]
fn reductions_softmax_and_cross_entropy_work() {
    let logits = Tensor2D::<2, 3>::from_vec(vec![1.0, 2.0, 3.0, 1.0, 0.0, -1.0])
        .unwrap()
        .with_requires_grad(true);

    assert_eq!(
        logits.sum_leading().unwrap().to_vec().unwrap(),
        vec![2.0, 2.0, 2.0]
    );
    assert_eq!(
        logits.mean_last().unwrap().to_vec().unwrap(),
        vec![2.0, 0.0]
    );

    let softmax = logits.softmax_last().unwrap().to_vec().unwrap();
    assert_close(softmax[0] + softmax[1] + softmax[2], 1.0, 1e-6);
    assert_close(softmax[3] + softmax[4] + softmax[5], 1.0, 1e-6);

    let loss = logits.cross_entropy(&[2, 0]).unwrap();
    assert_close(loss.to_vec().unwrap()[0], 0.407_605_95, 1e-6);
    loss.backward().unwrap();

    let grad = logits.grad().unwrap().to_vec().unwrap();
    assert_close(grad[2], (softmax[2] - 1.0) / 2.0, 1e-6);
    assert_close(grad[3], (softmax[3] - 1.0) / 2.0, 1e-6);
}

#[test]
fn cross_entropy_and_log_softmax_stay_finite_on_extreme_logits() {
    let extreme = Tensor2D::<1, 2>::from_vec(vec![0.0, -1000.0]).unwrap();
    assert_close(
        extreme.cross_entropy(&[1]).unwrap().to_vec().unwrap()[0],
        1000.0,
        1e-3,
    );
    assert_close(
        extreme.log_softmax_last().unwrap().to_vec().unwrap()[1],
        -1000.0,
        1e-3,
    );
}

#[test]
fn broadcasts_masks_and_indexing_have_gradients() {
    let x = Tensor2D::<2, 3>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .with_requires_grad(true);
    let col = Tensor1D::<2>::from_vec(vec![10.0, 20.0]).unwrap();
    assert_eq!(
        x.add_leading_dim(&col).unwrap().to_vec().unwrap(),
        vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]
    );

    let row = Tensor1D::<3>::from_vec(vec![1.0, 2.0, 3.0])
        .unwrap()
        .with_requires_grad(true);
    x.mul_last_dim(&row)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    assert_eq!(row.grad().unwrap().to_vec().unwrap(), vec![5.0, 7.0, 9.0]);
    x.zero_grad();

    let mask = x.gt_scalar(3.0).unwrap();
    assert_eq!(
        x.masked_fill(&mask, -1.0).unwrap().to_vec().unwrap(),
        vec![1.0, 2.0, 3.0, -1.0, -1.0, -1.0]
    );

    let picked = x.index_select_rows::<C<3>>(&[1, 0, 1]).unwrap();
    picked.sum().unwrap().backward().unwrap();
    assert_eq!(
        x.grad().unwrap().to_vec().unwrap(),
        vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    );
}

#[test]
fn bmm_computes_batched_matmul() {
    let lhs = Tensor::<D3<C<2>, C<2>, C<3>>>::from_vec(vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
    ])
    .unwrap();
    let rhs = Tensor::<D3<C<2>, C<3>, C<2>>>::from_vec(vec![
        1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 2.0, 1.0, 1.0,
    ])
    .unwrap();
    assert_eq!(
        lhs.bmm(&rhs).unwrap().to_vec().unwrap(),
        vec![4.0, 5.0, 10.0, 11.0, 3.0, 1.0, 0.0, 2.0]
    );
}

#[test]
fn gelu_softmax_and_log_softmax_gradients_match_finite_difference() {
    let gelu_data = vec![-0.7, 0.2];
    let x = Tensor1D::<2, f64>::from_vec(gelu_data.clone())
        .unwrap()
        .with_requires_grad(true);
    x.gelu().unwrap().sum().unwrap().backward().unwrap();
    let grad = x.grad().unwrap().to_vec().unwrap();
    for (idx, &analytic) in grad.iter().enumerate() {
        let numerical = finite_difference(&gelu_data, idx, 1e-6, |values| {
            Tensor1D::<2, f64>::from_vec(values.to_vec())
                .unwrap()
                .gelu()
                .unwrap()
                .sum()
                .unwrap()
                .to_vec()
                .unwrap()[0]
        });
        assert_close_f64(analytic, numerical, 1e-6);
    }

    let data = vec![0.2, -0.5, 1.0, -1.0, 0.3, 0.7];
    let weights = Tensor2D::<2, 3, f64>::from_vec(vec![0.2, -0.4, 0.8, -0.3, 0.7, -0.1]).unwrap();

    let x = Tensor2D::<2, 3, f64>::from_vec(data.clone())
        .unwrap()
        .with_requires_grad(true);
    x.softmax_last()
        .unwrap()
        .mul(&weights)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    let grad = x.grad().unwrap().to_vec().unwrap();
    for (idx, &analytic) in grad.iter().enumerate() {
        let numerical = finite_difference(&data, idx, 1e-6, |values| {
            Tensor2D::<2, 3, f64>::from_vec(values.to_vec())
                .unwrap()
                .softmax_last()
                .unwrap()
                .mul(&weights)
                .unwrap()
                .sum()
                .unwrap()
                .to_vec()
                .unwrap()[0]
        });
        assert_close_f64(analytic, numerical, 1e-6);
    }

    let x = Tensor2D::<2, 3, f64>::from_vec(data.clone())
        .unwrap()
        .with_requires_grad(true);
    x.log_softmax_last()
        .unwrap()
        .mul(&weights)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    let grad = x.grad().unwrap().to_vec().unwrap();
    for (idx, &analytic) in grad.iter().enumerate() {
        let numerical = finite_difference(&data, idx, 1e-6, |values| {
            Tensor2D::<2, 3, f64>::from_vec(values.to_vec())
                .unwrap()
                .log_softmax_last()
                .unwrap()
                .mul(&weights)
                .unwrap()
                .sum()
                .unwrap()
                .to_vec()
                .unwrap()[0]
        });
        assert_close_f64(analytic, numerical, 1e-6);
    }
}

#[test]
fn bmm_gradients_match_finite_difference() {
    let lhs_data = vec![1.0, 2.0, -1.0, 0.5];
    let rhs_data = vec![0.3, -0.2, 0.7, 1.1];
    let weights = Tensor::<D3<C<1>, C<2>, C<2>>, f64>::from_vec(vec![0.4, -0.5, 0.2, 0.9]).unwrap();

    let lhs = Tensor::<D3<C<1>, C<2>, C<2>>, f64>::from_vec(lhs_data.clone())
        .unwrap()
        .with_requires_grad(true);
    let rhs = Tensor::<D3<C<1>, C<2>, C<2>>, f64>::from_vec(rhs_data.clone())
        .unwrap()
        .with_requires_grad(true);
    lhs.bmm(&rhs)
        .unwrap()
        .mul(&weights)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();

    let lhs_grad = lhs.grad().unwrap().to_vec().unwrap();
    for (idx, &analytic) in lhs_grad.iter().enumerate() {
        let numerical = finite_difference(&lhs_data, idx, 1e-6, |values| {
            Tensor::<D3<C<1>, C<2>, C<2>>, f64>::from_vec(values.to_vec())
                .unwrap()
                .bmm(&Tensor::<D3<C<1>, C<2>, C<2>>, f64>::from_vec(rhs_data.clone()).unwrap())
                .unwrap()
                .mul(&weights)
                .unwrap()
                .sum()
                .unwrap()
                .to_vec()
                .unwrap()[0]
        });
        assert_close_f64(analytic, numerical, 1e-6);
    }
    let rhs_grad = rhs.grad().unwrap().to_vec().unwrap();
    for (idx, &analytic) in rhs_grad.iter().enumerate() {
        let numerical = finite_difference(&rhs_data, idx, 1e-6, |values| {
            Tensor::<D3<C<1>, C<2>, C<2>>, f64>::from_vec(lhs_data.clone())
                .unwrap()
                .bmm(&Tensor::<D3<C<1>, C<2>, C<2>>, f64>::from_vec(values.to_vec()).unwrap())
                .unwrap()
                .mul(&weights)
                .unwrap()
                .sum()
                .unwrap()
                .to_vec()
                .unwrap()[0]
        });
        assert_close_f64(analytic, numerical, 1e-6);
    }
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
