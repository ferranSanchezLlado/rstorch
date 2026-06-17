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
    let initial = mse_loss(&layer.forward(&input), &target).to_vec()[0];

    for _ in 0..80 {
        layer.zero_grad();
        let loss = mse_loss(&layer.forward(&input), &target);
        loss.backward();
        optimizer.step(layer.parameters_mut());
    }

    let final_loss = mse_loss(&layer.forward(&input), &target).to_vec()[0];
    assert!(final_loss < initial);
    assert!(final_loss < 0.1, "final loss too high: {final_loss}");
}

#[test]
fn tiny_mlp_loss_decreases_with_adam() {
    let mut rng = SmallRng::seed_from_u64(42);
    let mut layer1 = Linear::<2, 4>::kaiming_uniform(&mut rng);
    let mut layer2 = Linear::<4, 1>::xavier_uniform(&mut rng);
    let input = Tensor2D::<4, 2>::from_array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
    let target = Tensor2D::<4, 1>::from_array([[0.0], [1.0], [1.0], [2.0]]);
    let mut optimizer = Adam::new(0.05);

    let initial = mse_loss(&layer2.forward(&layer1.forward(&input).tanh()), &target).to_vec()[0];

    for _ in 0..400 {
        layer1.zero_grad();
        layer2.zero_grad();
        let hidden = layer1.forward(&input).tanh();
        let prediction = layer2.forward(&hidden);
        let loss = mse_loss(&prediction, &target);
        loss.backward();
        optimizer.step(layer1.parameters_mut());
        optimizer.step(layer2.parameters_mut());
    }

    let final_loss = mse_loss(&layer2.forward(&layer1.forward(&input).tanh()), &target).to_vec()[0];
    assert!(
        final_loss < initial,
        "initial={initial}, final={final_loss}"
    );
    assert!(final_loss < 0.1, "final loss too high: {final_loss}");
}
