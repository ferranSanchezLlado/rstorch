//! Metal hardware lane for Epoch 16.7: device-resident optimizer parity and
//! cross-backend checkpoint restore.
//!
//! These run only with the `metal` feature on macOS and skip gracefully when no
//! Metal device is available (headless CI). CPU stays the exact reference; the
//! native Metal SGD/Adam step kernels are compared against it within the f32
//! tolerance table, and a Metal-trained checkpoint is restored onto CPU and
//! continued.
#![cfg(all(feature = "metal", target_os = "macos"))]

use rstorch::prelude::*;
use rstorch::{Adam, Backend, HasParameters, Metal, Optimizer, OptimizerState, Sgd, StateDict};

const INPUT: [f32; 12] = [
    0.5, -0.2, 0.1, 0.4, -0.3, 0.8, -0.1, 0.2, 0.6, 0.0, -0.4, 0.3,
];
const TARGET: [f32; 6] = [0.2, -0.4, -0.1, 0.5, 0.3, 0.1];

fn metal_available() -> bool {
    <Metal as Backend<f32>>::default_device().is_ok()
}

fn dataset<B: Backend<f32>>() -> (Tensor2D<3, 4, f32, B>, Tensor2D<3, 2, f32, B>) {
    (
        Tensor2D::from_vec(INPUT.to_vec()).unwrap(),
        Tensor2D::from_vec(TARGET.to_vec()).unwrap(),
    )
}

fn step<B: Backend<f32>, O: Optimizer<f32, B>>(
    layer: &mut Linear<4, 2, f32, B>,
    opt: &mut O,
    input: &Tensor2D<3, 4, f32, B>,
    target: &Tensor2D<3, 2, f32, B>,
) -> f32 {
    let mut ctx = TrainContext::eval();
    let out = layer.forward(input, &mut ctx).unwrap();
    let loss = mse_loss(&out, target).unwrap();
    let value = loss.item().unwrap();
    loss.backward().unwrap();
    let mut params = Vec::new();
    layer.parameters_mut(&mut params);
    opt.step(&mut params).unwrap();
    value
}

/// Trains a single `Linear<4, 2>` and returns its flattened `[weight, bias]`.
fn train_linear<B, O>(make_opt: impl FnOnce() -> O, steps: usize) -> Vec<f32>
where
    B: Backend<f32>,
    O: Optimizer<f32, B>,
{
    let mut rng = SmallRng::seed_from_u64(11);
    let mut layer = Linear::<4, 2, f32, B>::xavier_uniform(&mut rng).unwrap();
    let (input, target) = dataset::<B>();
    let mut opt = make_opt();
    for _ in 0..steps {
        step(&mut layer, &mut opt, &input, &target);
    }
    let mut out = layer.weight().tensor().to_vec().unwrap();
    out.extend(layer.bias().tensor().to_vec().unwrap());
    out
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "index {i}: metal {a} vs cpu {e} exceeds tolerance {tol}"
        );
    }
}

#[test]
fn metal_sgd_momentum_matches_cpu_within_tolerance() {
    if !metal_available() {
        return;
    }
    let cpu = train_linear::<Cpu, _>(|| Sgd::<f32, Cpu>::with_momentum(0.1, 0.9), 25);
    let metal = train_linear::<Metal, _>(|| Sgd::<f32, Metal>::with_momentum(0.1, 0.9), 25);
    assert_close(&metal, &cpu, 1e-4);
}

#[test]
fn metal_adam_matches_cpu_within_tolerance() {
    if !metal_available() {
        return;
    }
    let cpu = train_linear::<Cpu, _>(|| Adam::<f32, Cpu>::with_weight_decay(0.05, 0.01), 25);
    let metal = train_linear::<Metal, _>(|| Adam::<f32, Metal>::with_weight_decay(0.05, 0.01), 25);
    assert_close(&metal, &cpu, 1e-4);
}

/// A Metal-trained model and its Adam optimizer state (moment buffers) restore
/// onto CPU through the name-addressed host records and continue training.
#[test]
fn metal_trained_checkpoint_restores_onto_cpu_and_continues() {
    if !metal_available() {
        return;
    }

    // Train on Metal for a few steps, building non-trivial Adam moment state.
    let mut rng = SmallRng::seed_from_u64(11);
    let mut metal_layer = Linear::<4, 2, f32, Metal>::xavier_uniform(&mut rng).unwrap();
    let (metal_input, metal_target) = dataset::<Metal>();
    let mut metal_opt = Adam::<f32, Metal>::new(0.05);
    for _ in 0..8 {
        step(
            &mut metal_layer,
            &mut metal_opt,
            &metal_input,
            &metal_target,
        );
    }

    // Serialize model + optimizer state to host records.
    let model_state = StateDict::from_module(&metal_layer).unwrap();
    let opt_state = metal_opt.state_dict(&metal_layer).unwrap();

    // Restore onto a fresh CPU model + optimizer.
    let mut cpu_layer = Linear::<4, 2, f32, Cpu>::zeros().unwrap();
    model_state.load_module(&mut cpu_layer).unwrap();
    let mut cpu_opt = Adam::<f32, Cpu>::new(1.0); // hyperparams overwritten by load
    cpu_opt.load_state_dict(&cpu_layer, &opt_state).unwrap();

    // Model weights transferred exactly (host records are dtype-exact).
    assert_eq!(
        cpu_layer.weight().tensor().to_vec().unwrap(),
        metal_layer.weight().tensor().to_vec().unwrap()
    );

    // Continue training on CPU; loss must keep decreasing from the restore point.
    let (cpu_input, cpu_target) = dataset::<Cpu>();
    let resume_loss = {
        let mut ctx = TrainContext::eval();
        let out = cpu_layer.forward(&cpu_input, &mut ctx).unwrap();
        mse_loss(&out, &cpu_target).unwrap().item().unwrap()
    };
    let mut last = resume_loss;
    for _ in 0..10 {
        last = step(&mut cpu_layer, &mut cpu_opt, &cpu_input, &cpu_target);
    }
    assert!(
        last.is_finite() && last < resume_loss,
        "expected continued CPU loss {last} < restore-point loss {resume_loss}"
    );
}
