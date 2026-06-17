#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

mod common;

use common::{assert_close, finite_difference};
use rstorch::prelude::*;

#[test]
fn mse_loss_backward_matches_finite_difference() {
    let target = Tensor1D::<3, f64>::from_array([1.0, -2.0, 4.0]);
    let prediction = Tensor1D::<3, f64>::from_array([2.0, 0.0, 7.0]).requires_grad();

    mse_loss(&prediction, &target).backward();

    let expected = finite_difference(&[2.0, 0.0, 7.0], |values| {
        mse_loss(
            &Tensor1D::<3, f64>::from_vec(values.to_vec()).unwrap(),
            &target,
        )
        .to_vec()[0]
    });
    assert_close(&prediction.grad().unwrap().to_vec(), &expected);
}

#[test]
fn binary_cross_entropy_backward_matches_finite_difference() {
    let target = Tensor1D::<3, f64>::from_array([0.0, 1.0, 1.0]);
    let prediction = Tensor1D::<3, f64>::from_array([0.2, 0.7, 0.9]).requires_grad();

    binary_cross_entropy(&prediction, &target).backward();

    let expected = finite_difference(&[0.2, 0.7, 0.9], |values| {
        binary_cross_entropy(
            &Tensor1D::<3, f64>::from_vec(values.to_vec()).unwrap(),
            &target,
        )
        .to_vec()[0]
    });
    assert_close(&prediction.grad().unwrap().to_vec(), &expected);
}
