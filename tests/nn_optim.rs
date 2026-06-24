use rstorch::prelude::*;
use rstorch::{Adam, HasParameters, Optimizer, Sgd, Tensor};

#[derive(Debug)]
struct Batch;

#[test]
fn parameter_new_sets_requires_grad_and_unique_ids() {
    let a = Parameter::new(Tensor1D::<2>::zeros().unwrap());
    let b = Parameter::new(Tensor1D::<2>::zeros().unwrap());

    assert!(a.tensor().requires_grad());
    assert_ne!(a.id(), b.id());
}

#[test]
fn linear_zeros_forward_and_symbolic_batch() {
    let layer = Linear::<2, 1>::zeros().unwrap();
    let input =
        Tensor::<D2<Sym<Batch>, C<2>>>::from_vec_with_shape(vec![1.0, 2.0, 3.0, 4.0], [2, 2])
            .unwrap();

    let out: Tensor<D2<Sym<Batch>, C<1>>> = layer.forward(&input).unwrap();

    assert_eq!(out.shape().dims(), &[2, 1]);
    assert_eq!(out.to_vec().unwrap(), vec![0.0, 0.0]);
}

#[test]
fn seeded_initialization_is_repeatable_and_non_zero() {
    let mut rng_a = SmallRng::seed_from_u64(42);
    let mut rng_b = SmallRng::seed_from_u64(42);
    let a = Linear::<3, 2>::xavier_uniform(&mut rng_a).unwrap();
    let b = Linear::<3, 2>::xavier_uniform(&mut rng_b).unwrap();

    let wa = a.weight().tensor().to_vec().unwrap();
    let wb = b.weight().tensor().to_vec().unwrap();
    assert_eq!(wa, wb);
    assert!(wa.iter().any(|&value| value != 0.0));
}

#[test]
fn relu_forward_and_backward_use_zero_derivative_at_zero() {
    let x = Tensor1D::<4>::from_vec(vec![-1.0, 0.0, 2.0, 3.0])
        .unwrap()
        .with_requires_grad(true);
    let y = relu(&x).unwrap();

    assert_eq!(y.to_vec().unwrap(), vec![0.0, 0.0, 2.0, 3.0]);

    y.sum().unwrap().backward().unwrap();
    assert_eq!(
        x.grad().unwrap().to_vec().unwrap(),
        vec![0.0, 0.0, 1.0, 1.0]
    );
}

#[test]
fn mse_loss_returns_scalar_and_computes_gradients() {
    let pred = Tensor1D::<2>::from_vec(vec![1.0, 3.0])
        .unwrap()
        .with_requires_grad(true);
    let target = Tensor1D::<2>::from_vec(vec![0.0, 1.0]).unwrap();
    let loss = mse_loss(&pred, &target).unwrap();

    assert_eq!(loss.shape().dims(), &[]);
    assert_eq!(loss.to_vec().unwrap(), vec![2.5]);

    loss.backward().unwrap();
    assert_eq!(pred.grad().unwrap().to_vec().unwrap(), vec![1.0, 2.0]);
}

#[test]
fn plain_sgd_matches_hand_computed_update() {
    let mut layer = Linear::<1, 1>::zeros().unwrap();
    train_one_scalar_step(&mut layer, &mut Sgd::new(0.1));

    assert_close(layer.weight().tensor().to_vec().unwrap()[0], 1.6, 1e-6);
    assert_close(layer.bias().tensor().to_vec().unwrap()[0], 0.8, 1e-6);
}

#[test]
fn momentum_sgd_matches_two_step_update() {
    let mut layer = Linear::<1, 1>::zeros().unwrap();
    let mut opt = Sgd::with_momentum(0.1, 0.5);

    train_one_scalar_step(&mut layer, &mut opt);
    train_one_scalar_step(&mut layer, &mut opt);

    assert_close(layer.weight().tensor().to_vec().unwrap()[0], 2.4, 1e-6);
    assert_close(layer.bias().tensor().to_vec().unwrap()[0], 1.2, 1e-6);
}

#[test]
fn adam_first_step_matches_hand_computed_update() {
    let mut layer = Linear::<1, 1>::zeros().unwrap();
    train_one_scalar_step(&mut layer, &mut Adam::new(0.1));

    assert_close(layer.weight().tensor().to_vec().unwrap()[0], 0.1, 1e-5);
    assert_close(layer.bias().tensor().to_vec().unwrap()[0], 0.1, 1e-5);
}

#[test]
fn optimizers_skip_parameters_without_gradients() {
    let mut layer = Linear::<1, 1>::zeros().unwrap();
    let mut opt = Sgd::new(0.1);
    let mut params = Vec::new();
    layer.parameters_mut(&mut params);

    opt.step(&mut params).unwrap();
    drop(params);

    assert_eq!(layer.weight().tensor().to_vec().unwrap(), vec![0.0]);
    assert_eq!(layer.bias().tensor().to_vec().unwrap(), vec![0.0]);
}

#[test]
fn optimizer_zero_grad_clears_module_parameter_gradients() {
    let layer = Linear::<1, 1>::zeros().unwrap();
    scalar_loss(&layer).backward().unwrap();
    assert!(layer.weight().grad().is_some());

    let mut opt = Sgd::new(0.1);
    let mut params = Vec::new();
    layer.parameters(&mut params);
    opt.zero_grad(&params);

    assert!(layer.weight().grad().is_none());
    assert!(layer.bias().grad().is_none());
}

#[test]
fn tiny_linear_regression_loss_decreases() {
    let mut layer = Linear::<1, 1>::zeros().unwrap();
    let mut opt = Sgd::new(0.05);
    let first = scalar_loss(&layer).to_vec().unwrap()[0];

    for _ in 0..10 {
        scalar_loss(&layer).backward().unwrap();
        let mut params = Vec::new();
        layer.parameters_mut(&mut params);
        opt.step(&mut params).unwrap();
    }

    let last = scalar_loss(&layer).to_vec().unwrap()[0];
    assert!(last < first, "expected {last} < {first}");
}

fn scalar_loss(layer: &Linear<1, 1>) -> Scalar {
    let input = Tensor2D::<1, 1>::from_vec(vec![2.0]).unwrap();
    let target = Tensor2D::<1, 1>::from_vec(vec![4.0]).unwrap();
    mse_loss(&layer.forward(&input).unwrap(), &target).unwrap()
}

fn train_one_scalar_step<O>(layer: &mut Linear<1, 1>, opt: &mut O)
where
    O: Optimizer<f32, Cpu>,
{
    scalar_loss(layer).backward().unwrap();
    let mut params = Vec::new();
    layer.parameters_mut(&mut params);
    opt.step(&mut params).unwrap();
}

fn assert_close(actual: f32, expected: f32, tol: f32) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {actual} to be within {tol} of {expected}"
    );
}
