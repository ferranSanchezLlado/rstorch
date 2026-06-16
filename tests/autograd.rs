#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use rstorch::prelude::*;

const EPSILON: f64 = 1e-6;
const TOLERANCE: f64 = 1e-6;

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < TOLERANCE,
            "index {index}: {actual} != {expected}"
        );
    }
}

fn finite_difference(values: &[f64], f: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    let mut gradient = Vec::with_capacity(values.len());

    for index in 0..values.len() {
        let mut plus = values.to_vec();
        plus[index] += EPSILON;
        let mut minus = values.to_vec();
        minus[index] -= EPSILON;
        gradient.push((f(&plus) - f(&minus)) / (2.0 * EPSILON));
    }

    gradient
}

#[test]
fn finite_difference_checks_add_gradient() {
    let y = Tensor1D::<3, f64>::from_array([5.0, 8.0, 13.0]);
    let x = Tensor1D::<3, f64>::from_array([2.0, 4.0, 6.0]).requires_grad();

    x.add(&y).sum().backward();

    let expected = finite_difference(&[2.0, 4.0, 6.0], |values| {
        Tensor1D::<3, f64>::from_vec(values.to_vec())
            .unwrap()
            .add(&y)
            .sum()
            .to_vec()[0]
    });
    assert_close(&x.grad().unwrap().to_vec(), &expected);
}

#[test]
fn finite_difference_checks_sub_gradient() {
    let y = Tensor1D::<3, f64>::from_array([5.0, 8.0, 13.0]);
    let x = Tensor1D::<3, f64>::from_array([2.0, 4.0, 6.0]).requires_grad();

    x.sub(&y).sum().backward();

    let expected = finite_difference(&[2.0, 4.0, 6.0], |values| {
        Tensor1D::<3, f64>::from_vec(values.to_vec())
            .unwrap()
            .sub(&y)
            .sum()
            .to_vec()[0]
    });
    assert_close(&x.grad().unwrap().to_vec(), &expected);
}

#[test]
fn finite_difference_checks_mul_gradient() {
    let y = Tensor1D::<3, f64>::from_array([5.0, 8.0, 13.0]);
    let x = Tensor1D::<3, f64>::from_array([2.0, 4.0, 6.0]).requires_grad();

    x.mul(&y).sum().backward();

    let expected = finite_difference(&[2.0, 4.0, 6.0], |values| {
        Tensor1D::<3, f64>::from_vec(values.to_vec())
            .unwrap()
            .mul(&y)
            .sum()
            .to_vec()[0]
    });
    assert_close(&x.grad().unwrap().to_vec(), &expected);
}

#[test]
fn finite_difference_checks_div_gradient() {
    let y = Tensor1D::<3, f64>::from_array([5.0, 8.0, 13.0]);
    let x = Tensor1D::<3, f64>::from_array([2.0, 4.0, 6.0]).requires_grad();

    x.div(&y).sum().backward();

    let expected = finite_difference(&[2.0, 4.0, 6.0], |values| {
        Tensor1D::<3, f64>::from_vec(values.to_vec())
            .unwrap()
            .div(&y)
            .sum()
            .to_vec()[0]
    });
    assert_close(&x.grad().unwrap().to_vec(), &expected);
}

#[test]
fn finite_difference_checks_matmul_gradient() {
    let w = Tensor2D::<3, 2, f64>::from_array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);
    let x = Tensor2D::<2, 3, f64>::from_array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).requires_grad();

    x.matmul(&w).sum().backward();

    let expected = finite_difference(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], |values| {
        Tensor2D::<2, 3, f64>::from_vec(values.to_vec())
            .unwrap()
            .matmul(&w)
            .sum()
            .to_vec()[0]
    });
    assert_close(&x.grad().unwrap().to_vec(), &expected);
}

#[test]
fn finite_difference_checks_sum_gradient() {
    let x = Tensor1D::<3, f64>::from_array([1.0, 2.0, 4.0]).requires_grad();

    x.sum().backward();

    let expected = finite_difference(&[1.0, 2.0, 4.0], |values| {
        Tensor1D::<3, f64>::from_vec(values.to_vec())
            .unwrap()
            .sum()
            .to_vec()[0]
    });
    assert_close(&x.grad().unwrap().to_vec(), &expected);
}

#[test]
fn finite_difference_checks_mean_gradient() {
    let x = Tensor1D::<3, f64>::from_array([1.0, 2.0, 4.0]).requires_grad();

    x.mean().backward();

    let expected = finite_difference(&[1.0, 2.0, 4.0], |values| {
        Tensor1D::<3, f64>::from_vec(values.to_vec())
            .unwrap()
            .mean()
            .to_vec()[0]
    });
    assert_close(&x.grad().unwrap().to_vec(), &expected);
}

#[test]
fn finite_difference_checks_relu_gradient() {
    let x = Tensor1D::<3, f64>::from_array([-1.0, 2.0, 4.0]).requires_grad();

    x.relu().sum().backward();
    let expected = finite_difference(&[-1.0, 2.0, 4.0], |values| {
        Tensor1D::<3, f64>::from_vec(values.to_vec())
            .unwrap()
            .relu()
            .sum()
            .to_vec()[0]
    });
    assert_close(&x.grad().unwrap().to_vec(), &expected);
}

#[test]
fn finite_difference_checks_exp_gradient() {
    let x = Tensor1D::<3, f64>::from_array([1.0, 2.0, 4.0]).requires_grad();

    x.exp().sum().backward();
    let expected = finite_difference(&[1.0, 2.0, 4.0], |values| {
        Tensor1D::<3, f64>::from_vec(values.to_vec())
            .unwrap()
            .exp()
            .sum()
            .to_vec()[0]
    });
    assert_close(&x.grad().unwrap().to_vec(), &expected);
}

#[test]
fn finite_difference_checks_ln_gradient() {
    let x = Tensor1D::<3, f64>::from_array([1.0, 2.0, 4.0]).requires_grad();

    x.ln().sum().backward();
    let expected = finite_difference(&[1.0, 2.0, 4.0], |values| {
        Tensor1D::<3, f64>::from_vec(values.to_vec())
            .unwrap()
            .ln()
            .sum()
            .to_vec()[0]
    });
    assert_close(&x.grad().unwrap().to_vec(), &expected);
}

#[test]
fn finite_difference_checks_powf_gradient() {
    let x = Tensor1D::<3, f64>::from_array([1.0, 2.0, 4.0]).requires_grad();

    x.powf(2.5).sum().backward();
    let expected = finite_difference(&[1.0, 2.0, 4.0], |values| {
        Tensor1D::<3, f64>::from_vec(values.to_vec())
            .unwrap()
            .powf(2.5)
            .sum()
            .to_vec()[0]
    });
    assert_close(&x.grad().unwrap().to_vec(), &expected);
}

#[test]
fn non_leaf_diamond_graph_accumulates_before_parent_backward() {
    let x = Tensor1D::<3, f64>::from_array([-1.0, 2.0, 4.0]).requires_grad();

    let y = x.relu();
    y.mul(&y).sum().backward();

    assert_close(&x.grad().unwrap().to_vec(), &[0.0, 4.0, 8.0]);
}
