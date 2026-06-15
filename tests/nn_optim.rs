#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use rstorch::prelude::*;

#[test]
fn linear_parameters_default_to_requires_grad() {
    let layer = Linear::<3, 2>::new();

    assert!(layer.weight().tensor().requires_grad_enabled());
    assert!(layer.bias().tensor().requires_grad_enabled());
}

#[test]
fn sgd_step_updates_parameters_under_no_grad() {
    let mut parameter = Parameter::new(Tensor1D::<1>::from_array([2.0]));
    parameter.tensor().mul_scalar(3.0).sum().backward();

    let mut optimizer = SGD::new(0.5);
    optimizer.step(vec![&mut parameter as &mut dyn OptimParameter<f32, Cpu>]);

    assert_eq!(parameter.tensor().to_vec(), vec![0.5]);
    assert!(parameter.tensor().requires_grad_enabled());
    assert!(parameter.tensor().is_leaf());
}

#[test]
fn tiny_linear_regression_loss_decreases() {
    let mut layer = Linear::<1, 1>::new();
    let input = Tensor2D::<4, 1>::from_array([[0.0], [1.0], [2.0], [3.0]]);
    let target = Tensor2D::<4, 1>::from_array([[1.0], [3.0], [5.0], [7.0]]);
    let mut optimizer = SGD::new(0.1);
    let initial = layer.forward(&input).sub(&target).powf(2.0).mean().to_vec()[0];

    for _ in 0..80 {
        layer.zero_grad();
        let loss = layer.forward(&input).sub(&target).powf(2.0).mean();
        loss.backward();
        optimizer.step(layer.parameters_mut());
    }

    let final_loss = layer.forward(&input).sub(&target).powf(2.0).mean().to_vec()[0];
    assert!(final_loss < initial);
    assert!(final_loss < 0.1, "final loss too high: {final_loss}");
}
