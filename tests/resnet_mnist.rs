//! Does the big model actually *learn*, on every backend?
//!
//! The throughput numbers live in `benches/resnet_mnist.rs`; this file owns the
//! other half of the question, because a backend that is fast and wrong is the
//! failure mode worth catching. Both targets share the network
//! (`benches/support/resnet.rs`), so the thing benched and the thing checked
//! cannot drift apart.
//!
//! # What runs when
//!
//! - [`resnet_smoke_trains_on_synthetic_data`] runs on every `cargo test`. It
//!   overfits four images with the tiny spec in well under a second, and fails
//!   if the residual model stops building, stops training, or starts producing
//!   non-finite losses.
//! - [`resnet_learns_on_every_device`] is `#[ignore]`d: it trains the real
//!   network for tens of steps per device and takes minutes.
//!
//! ```text
//! # Synthetic pixels: asserts the loss falls, on CPU and on Metal
//! cargo test --release --features rayon,metal --test resnet_mnist \
//!     -- --ignored --nocapture
//!
//! # Real MNIST: also asserts held-out accuracy beats chance by a wide margin
//! RSTORCH_RESNET_DATA=mnist cargo test --release --features rayon,metal,hub \
//!     --test resnet_mnist -- --ignored --nocapture
//! ```
//!
//! `--release` is not optional in practice: a debug build runs the
//! convolutions roughly two orders of magnitude slower.
//!
//! Beyond the shape variables documented in `benches/support/resnet.rs`, this
//! file reads `RSTORCH_RESNET_STEPS` (default 100) and
//! `RSTORCH_RESNET_EVAL_BATCHES` (default 4). At those defaults the CPU run
//! takes a bit over a minute and lands around 83% on held-out MNIST — 3200
//! training images is not a converged model, only a decisive one.

use std::time::Instant;

use rstorch::prelude::*;

#[path = "../benches/support/resnet.rs"]
mod support;

use support::{
    HostData, ResNet, ResNetSpec, Source, accuracy, batch_size, devices, env_usize, seed,
    train_step,
};

/// Overfitting one batch is the cheapest end-to-end proof that the residual
/// path, the strided shortcut, the global pool and the optimizer are all
/// wired up. Deterministic: same seed, same three losses.
#[test]
fn resnet_smoke_trains_on_synthetic_data() -> Result<()> {
    let device = Device::Cpu;
    let mut rng = Rng::seed(11);
    let mut model = ResNet::new(ResNetSpec::tiny(), 1, &device, &mut rng)?;
    let mut optimizer = Sgd::new(0.05).momentum(0.9);

    let inputs = Tensor::from_vec(
        (0..4 * 28 * 28)
            .map(|_| rng.normal(0.0, 1.0) as f32)
            .collect(),
        [4, 1, 28, 28],
        &device,
    )?;
    let targets = Tensor::from_vec(vec![0i64, 1, 2, 3], [4], &device)?;

    let mut losses = Vec::new();
    for _ in 0..3 {
        losses.push(train_step(&mut model, &mut optimizer, &inputs, &targets)?);
    }

    assert!(
        losses.iter().all(|loss| loss.is_finite()),
        "non-finite loss in {losses:?}"
    );
    assert!(
        losses[2] < losses[0],
        "three steps of overfitting one batch did not reduce the loss: {losses:?}"
    );
    Ok(())
}

/// Trains the configured network on each available device and asserts it
/// learns there. Prints a wall-clock summary per device as a sanity check on
/// the bench numbers — criterion is the instrument, this is the smell test.
#[test]
#[ignore = "trains a real network for minutes: run with --release --ignored --nocapture"]
fn resnet_learns_on_every_device() -> Result<()> {
    let spec = ResNetSpec::from_env();
    let source = Source::from_env();
    let batch = batch_size();
    let seed = seed();
    let steps = env_usize("RSTORCH_RESNET_STEPS", 100).max(2);
    let eval_batches = env_usize("RSTORCH_RESNET_EVAL_BATCHES", 4);

    let data = HostData::load(source, steps * batch, eval_batches * batch, seed)?;
    println!(
        "\nresnet/{}: {}, batch {}, {steps} steps, width {}, {} blocks/stage, {} stages -> {} convs",
        source.label(),
        support::build_label(),
        batch,
        spec.width,
        spec.blocks_per_stage,
        spec.stages,
        spec.conv_layers(),
    );

    for device in devices() {
        let (train, eval) = data.batches(&device, batch, steps, eval_batches, seed)?;
        let mut rng = Rng::seed(seed);
        let mut model = ResNet::new(spec, 1, &device, &mut rng)?;
        let mut optimizer = Sgd::new(0.05).momentum(0.9);

        let start = Instant::now();
        let mut losses = Vec::with_capacity(steps);
        for (inputs, targets) in &train {
            losses.push(train_step(&mut model, &mut optimizer, inputs, targets)?);
        }
        let elapsed = start.elapsed();

        // The tail rather than the last single step: one batch can be unlucky.
        let tail = losses.len() / 4;
        let first = mean(&losses[..tail.max(1)]);
        let last = mean(&losses[losses.len() - tail.max(1)..]);
        let accuracy = if eval.is_empty() {
            None
        } else {
            Some(accuracy(&mut model, &eval)?)
        };

        println!(
            "  {device:<10} {:>8.1} ms/step  {:>8.1} img/s  loss {first:.4} -> {last:.4}{}",
            elapsed.as_secs_f64() * 1e3 / steps as f64,
            (steps * batch) as f64 / elapsed.as_secs_f64(),
            match accuracy {
                Some(value) => format!("  accuracy {:.1}%", value * 100.0),
                None => String::new(),
            }
        );

        assert!(
            losses.iter().all(|loss| loss.is_finite()),
            "{device} produced a non-finite loss: {losses:?}"
        );
        assert!(
            last < 0.8 * first,
            "{device} did not train: mean loss went {first:.4} -> {last:.4} over {steps} steps"
        );
        if source == Source::Mnist {
            let accuracy = accuracy.expect("eval batches were requested");
            assert!(
                accuracy > 0.6,
                "{device} reached only {:.1}% on held-out MNIST after {steps} steps; \
                 chance is 10% and the default 100 steps land around 83% on the CPU",
                accuracy * 100.0
            );
        }
    }
    println!();
    Ok(())
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}
