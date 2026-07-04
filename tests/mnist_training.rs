//! End-to-end classifier training loops, including a real MNIST integration run.

use rstorch::prelude::*;

#[test]
fn tiny_cross_entropy_classifier_loss_decreases() {
    let input = Tensor2D::<4, 2>::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
    let targets = [0, 1, 0, 1];
    let mut layer = Linear::<2, 2>::zeros().unwrap();
    let mut opt = Sgd::new(0.2);
    let mut ctx = TrainContext::training(0);

    let first = layer
        .forward(&input, &mut ctx)
        .unwrap()
        .cross_entropy(&targets)
        .unwrap()
        .to_vec()
        .unwrap()[0];
    for _ in 0..20 {
        layer
            .forward(&input, &mut ctx)
            .unwrap()
            .cross_entropy(&targets)
            .unwrap()
            .backward()
            .unwrap();
        sgd_step_and_zero(&mut layer, &mut opt);
    }
    let last = layer
        .forward(&input, &mut ctx)
        .unwrap()
        .cross_entropy(&targets)
        .unwrap()
        .to_vec()
        .unwrap()[0];

    assert!(
        last < first,
        "expected CE loss to decrease from {first} to {last}"
    );
}

#[test]
fn typed_sequential_mlp_trains_on_synthetic_classification() {
    // Two clusters separable through a hidden ReLU layer.
    let input = Tensor2D::<6, 4>::from_vec(vec![
        2.0, 1.0, 0.0, 0.0, 1.5, 2.0, 0.1, 0.0, 2.2, 1.3, 0.0, 0.2, 0.0, 0.1, 2.0, 1.0, 0.1, 0.0,
        1.7, 2.0, 0.0, 0.2, 2.1, 1.4,
    ])
    .unwrap();
    let targets = [0, 0, 0, 1, 1, 1];

    let mut rng = SmallRng::seed_from_u64(0);
    let mut model = Sequential::new(Linear::<4, 8>::xavier_uniform(&mut rng).unwrap(), Relu)
        .add_module(Linear::<8, 2>::xavier_uniform(&mut rng).unwrap());
    let mut opt = Sgd::new(0.5);
    let mut ctx = TrainContext::training(0);

    let first = model
        .forward(&input, &mut ctx)
        .unwrap()
        .cross_entropy(&targets)
        .unwrap()
        .to_vec()
        .unwrap()[0];
    for _ in 0..100 {
        let loss = model
            .forward(&input, &mut ctx)
            .unwrap()
            .cross_entropy(&targets)
            .unwrap();
        loss.backward().unwrap();
        sgd_step_and_zero(&mut model, &mut opt);
    }
    let logits = model.forward(&input, &mut ctx).unwrap();
    let last = logits.cross_entropy(&targets).unwrap().to_vec().unwrap()[0];

    assert!(
        last < first,
        "expected CE loss to decrease from {first} to {last}"
    );
    assert_eq!(logits.argmax_last().unwrap(), targets.to_vec());
}

#[cfg(feature = "hub")]
#[test]
#[ignore = "downloads the MNIST dataset over the network"]
fn typed_sequential_mlp_learns_on_mnist() {
    let hub = DatasetHub::default_cache();
    let dataset = Mnist::load(&hub, MnistSplit::Train).unwrap();
    let loader = DataLoader::new(
        dataset,
        SequentialSampler,
        MnistCollate::<f32>::new(),
        64,
        true,
    )
    .unwrap();
    let (images, labels) = loader.iter().next().unwrap().unwrap();
    let targets: Vec<usize> = labels.iter().map(|&label| label as usize).collect();

    let mut rng = SmallRng::seed_from_u64(0);
    let mut model = Sequential::new(Linear::<784, 128>::kaiming_uniform(&mut rng).unwrap(), Relu)
        .add_module(Linear::<128, 10>::kaiming_uniform(&mut rng).unwrap());
    let mut opt = Sgd::new(0.1);
    let mut ctx = TrainContext::training(0);

    let first = model
        .forward(&images, &mut ctx)
        .unwrap()
        .cross_entropy(&targets)
        .unwrap()
        .to_vec()
        .unwrap()[0];
    // Overfitting one real batch is a reliable architecture smoke test: it drives
    // the full collate -> Sequential -> cross_entropy -> SGD loop end to end.
    for _ in 0..300 {
        let loss = model
            .forward(&images, &mut ctx)
            .unwrap()
            .cross_entropy(&targets)
            .unwrap();
        loss.backward().unwrap();
        sgd_step_and_zero(&mut model, &mut opt);
    }
    let logits = model.forward(&images, &mut ctx).unwrap();
    let last = logits.cross_entropy(&targets).unwrap().to_vec().unwrap()[0];
    let correct = logits.correct_count(&targets).unwrap();
    let accuracy = correct as f64 / targets.len() as f64;

    assert!(
        last < first,
        "expected MNIST loss to decrease from {first} to {last}"
    );
    assert!(
        accuracy >= 0.9,
        "expected to overfit one batch, got accuracy {accuracy}"
    );
}

/// Applies an optimizer step over every parameter, then clears the gradients.
fn sgd_step_and_zero<M, O>(model: &mut M, opt: &mut O)
where
    M: HasParameters<f32, Cpu>,
    O: Optimizer<f32, Cpu>,
{
    let mut params = Vec::new();
    model.parameters_mut(&mut params);
    opt.step(&mut params).unwrap();
    drop(params);

    let mut refs = Vec::new();
    model.parameters(&mut refs);
    for param in &refs {
        param.zero_grad();
    }
}
