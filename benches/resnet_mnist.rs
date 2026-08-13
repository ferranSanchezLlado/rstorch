//! ResNet-on-MNIST throughput: the big-model counterpart to
//! `benches/training.rs`, for comparing the CPU backend with and without
//! `rayon` and against Metal.
//!
//! `training.rs` benches a toy transformer and an MLP, where per-op host
//! overhead dominates. Here the workload is a residual network over
//! `[batch, 1, 28, 28]` images — convolutions large enough that thread scaling
//! and GPU occupancy are what the wall clock reflects.
//!
//! # This bench does not run with the others
//!
//! It is gated on the `bench-resnet` feature, so a plain `cargo bench` neither
//! builds nor runs it: one sample is a full training step (about a second on a
//! release CPU build at the default shape, minutes at a scaled-up one), which
//! has no business inside the routine suite.
//!
//! ```text
//! # CPU, single-threaded reference
//! cargo bench --features bench-resnet --bench resnet_mnist
//!
//! # CPU with the parallel façade (the A/B partner of the line above)
//! cargo bench --features bench-resnet,rayon --bench resnet_mnist
//!
//! # ... and Metal, which adds a `resnet/metal:0` group
//! cargo bench --features bench-resnet,rayon,metal --bench resnet_mnist
//!
//! # Real MNIST pixels instead of seeded noise (shapes and op counts are
//! # identical, so this changes the data, not the throughput)
//! RSTORCH_RESNET_DATA=mnist cargo bench --features bench-resnet,rayon,hub --bench resnet_mnist
//! ```
//!
//! Because `rayon` is a compile-time feature it cannot be swept at runtime:
//! run the first two lines and compare, using the repo's baseline workflow
//! (`--save-baseline no-rayon`, then `--baseline no-rayon`). `RAYON_NUM_THREADS`
//! bounds the pool for a thread-scaling curve. The model shape, batch size and
//! data source come from the environment variables documented in
//! `benches/support/resnet.rs`; the shape is part of every benchmark id
//! (`w16b1s3`), so a baseline is never compared against a different network.
//!
//! # What each row measures
//!
//! - `forward` — an inference forward (`Mode::EVAL`, so `BatchNorm` reads its
//!   running statistics and nothing is recorded), ending in a host read of the
//!   logits.
//! - `train_step` — forward, cross-entropy, `backward`, `Sgd::step`, and a host
//!   read of a head parameter. The difference between the two rows is the
//!   backward pass plus the optimizer.
//!
//! Both rows end at a host read on purpose: a deferred backend would otherwise
//! be timed encoding commands rather than running them. Iterations reuse the
//! same batches and keep mutating the model, exactly as `training.rs` does —
//! that is the steady state of a training loop, not a measurement trick.

use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, SamplingMode, criterion_group, criterion_main};
use rstorch::prelude::*;

#[path = "support/resnet.rs"]
mod support;

use support::{HostData, ResNet, ResNetSpec, Source, batch_size, devices, seed, train_step};

/// Training batches materialized per device. More than one so an iteration is
/// not a single batch memorized into the `BatchNorm` buffers, few enough that a
/// scaled-up shape does not exhaust GPU memory.
const BATCHES: usize = 4;

fn bench_resnet(c: &mut Criterion) {
    let spec = ResNetSpec::from_env();
    let source = Source::from_env();
    let batch = batch_size();
    let seed = seed();

    let data = HostData::load(source, BATCHES * batch, 0, seed).expect("bench data");

    println!(
        "resnet bench: {}, {} data, batch {}, width {}, {} blocks/stage, {} stages -> {} convs",
        support::build_label(),
        source.label(),
        batch,
        spec.width,
        spec.blocks_per_stage,
        spec.stages,
        spec.conv_layers(),
    );

    for device in devices() {
        let (batches, _) = data
            .batches(&device, batch, BATCHES, 0, seed)
            .expect("bench batches");
        let mut rng = Rng::seed(seed);
        let mut model = ResNet::new(spec, 1, &device, &mut rng).expect("bench model");
        let mut optimizer = Sgd::new(0.05).momentum(0.9);

        let mut group = c.benchmark_group(format!("resnet/{device}"));
        // A sample here is `BATCHES` whole training steps — hundreds of
        // milliseconds to seconds — which is what criterion's flat sampling
        // exists for: a fixed iteration count per sample instead of the linear
        // ramp, which at this scale would ask for hours. Ten samples is
        // criterion's floor and enough to see a regime change.
        //
        // The measurement time is a target: criterion divides it by the sample
        // count to pick iterations per sample, so keeping it short is what
        // holds `train_step` at one step per sample. When a sample alone
        // overruns it, criterion prints "unable to complete 10 samples" and
        // then runs them anyway — expected here, not a misconfiguration.
        group.sampling_mode(SamplingMode::Flat);
        group.sample_size(10);
        group.warm_up_time(Duration::from_secs(2));
        group.measurement_time(Duration::from_secs(20));

        println!(
            "  {device}: {} parameters, {BATCHES} batches of {batch}",
            model.params()
        );

        let id = format!("{}/{}x1x28x28", spec.id(), batch);

        group.bench_function(format!("forward/{id}"), |b| {
            b.iter(|| {
                for (inputs, _) in &batches {
                    // The host read is the synchronization point that makes a
                    // deferred backend's "forward" mean something.
                    black_box(
                        model
                            .forward(inputs, Mode::EVAL)
                            .unwrap()
                            .to_vec::<f32>()
                            .unwrap(),
                    );
                }
            });
        });

        group.bench_function(format!("train_step/{id}"), |b| {
            b.iter(|| {
                for (inputs, targets) in &batches {
                    black_box(train_step(&mut model, &mut optimizer, inputs, targets).unwrap());
                }
            });
        });

        group.finish();
    }
}

criterion_group!(benches, bench_resnet);
criterion_main!(benches);
