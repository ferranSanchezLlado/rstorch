//! `DataLoader` throughput over a full epoch, for both provided datasets.
//!
//! Ported from the v2 `benches/data_pipeline.rs` (api-reset2). The two v2
//! benchmark ids (`dataloader/sequential_epoch/1024x64`,
//! `dataloader/random_epoch/1024x64`) are preserved for the per-item dataset so
//! the numbers line up with
//! `docs/restart-v3/reference/v2-performance-baseline.md`.
//!
//! # What changed in the port
//!
//! - v2 had a `Sampler` trait plus three collators in two batch-axis flavors.
//!   v3 has one loader whose order is `new(ds, n)` / `.shuffle(seed)`, and the
//!   dataset owns collation — so `SequentialSampler` becomes the default and
//!   `RandomSampler::new(5)` becomes `.shuffle(5)`.
//! - v2's `VecDataset<(Vec<f32>, usize)>` held host `Vec`s and the
//!   `features::<64>()` collator built the batch tensor from them. v3's
//!   [`VecDataset`] holds one `Tensor` per item and collates with
//!   `Tensor::stack`, so the per-item lane now pays 32 tensor stacks per batch
//!   instead of one host copy. That is the honest cost of the v3 design and the
//!   reason [`TensorDataset`] exists; both lanes are measured.
//! - The `tensor_*` lanes are **new**: one `index_select` per batch on the
//!   whole split, which is the v3 idiom for data that already lives in tensors
//!   (and what `fixtures/mnist_mlp.rs` uses).
//! - The `collate` group is **new**: the two collation primitives isolated at
//!   one batch's shape, each at two or three sizes so the epoch cost above can
//!   be attributed instead of guessed. See [`bench_collate`].
//!
//! Inputs are deterministic (a seeded `rstorch::Rng`) and never touch disk or
//! the network. Throughput is reported in samples per second over a full epoch.

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rstorch::prelude::*;

const SAMPLES: usize = 1024;
const FEATURES: usize = 64;
const BATCH_SIZE: usize = 32;
const CLASSES: usize = 10;

/// A deterministic `[len]` f32 row in `[-1, 1)`.
fn row(rng: &mut Rng, len: usize, dev: &Device) -> Tensor {
    let values: Vec<f32> = (0..len).map(|_| rng.uniform(-1.0, 1.0) as f32).collect();
    Tensor::from_vec(values, [len], dev).expect("bench row")
}

/// `SAMPLES` items of a `[FEATURES]` f32 input and a rank-0 I64 label — the
/// per-item shape, stacked into `[n, FEATURES]` / `[n]` batches.
fn vec_dataset(dev: &Device) -> VecDataset {
    let mut rng = Rng::seed(5);
    let items = (0..SAMPLES)
        .map(|sample| {
            let values: Vec<f32> = (0..FEATURES)
                .map(|_| rng.uniform(-1.0, 1.0) as f32)
                .collect();
            let input = Tensor::from_vec(values, [FEATURES], dev).expect("bench input");
            let target =
                Tensor::from_vec(vec![(sample % CLASSES) as i64], [], dev).expect("bench label");
            (input, target)
        })
        .collect();
    VecDataset::new(items).expect("uniform bench items")
}

/// The same data as one `[SAMPLES, FEATURES]` input tensor and one `[SAMPLES]`
/// label tensor — the device-resident split.
fn tensor_dataset(dev: &Device) -> TensorDataset {
    let mut rng = Rng::seed(5);
    let values: Vec<f32> = (0..SAMPLES * FEATURES)
        .map(|_| rng.uniform(-1.0, 1.0) as f32)
        .collect();
    let inputs = Tensor::from_vec(values, [SAMPLES, FEATURES], dev).expect("bench input");
    let targets: Vec<i64> = (0..SAMPLES).map(|s| (s % CLASSES) as i64).collect();
    let targets = Tensor::from_vec(targets, [SAMPLES], dev).expect("bench labels");
    TensorDataset::new(inputs, targets).expect("matching bench lengths")
}

/// Walk one full epoch, consuming every batch.
fn drain<D: Dataset>(loader: &DataLoader<D>) {
    for batch in loader.batches() {
        black_box(batch.expect("bench batch"));
    }
}

fn bench_dataloader(c: &mut Criterion) {
    let dev = Device::Cpu;
    let mut group = c.benchmark_group("dataloader");
    group.throughput(Throughput::Elements(SAMPLES as u64));

    let items = vec_dataset(&dev);
    let split = tensor_dataset(&dev);

    // A self-check before timing: an epoch must cover every sample in
    // `num_batches` batches whose leading axes agree. A loader that silently
    // yielded fewer batches would look wonderfully fast.
    for (name, num_batches, first) in [
        (
            "vec",
            DataLoader::new(&items, BATCH_SIZE).num_batches(),
            DataLoader::new(&items, BATCH_SIZE)
                .batches()
                .next()
                .unwrap()
                .unwrap(),
        ),
        (
            "tensor",
            DataLoader::new(&split, BATCH_SIZE).num_batches(),
            DataLoader::new(&split, BATCH_SIZE)
                .batches()
                .next()
                .unwrap()
                .unwrap(),
        ),
    ] {
        assert_eq!(num_batches, SAMPLES / BATCH_SIZE, "{name}: batch count");
        assert_eq!(first.0.dims(), &[BATCH_SIZE, FEATURES], "{name}: inputs");
        assert_eq!(first.1.dims(), &[BATCH_SIZE], "{name}: targets");
    }

    let sequential = DataLoader::new(&items, BATCH_SIZE);
    group.bench_function("sequential_epoch/1024x64", |b| {
        b.iter(|| drain(black_box(&sequential)));
    });

    let random = DataLoader::new(&items, BATCH_SIZE).shuffle(5);
    group.bench_function("random_epoch/1024x64", |b| {
        b.iter(|| drain(black_box(&random)));
    });

    let tensor_sequential = DataLoader::new(&split, BATCH_SIZE);
    group.bench_function("tensor_sequential_epoch/1024x64", |b| {
        b.iter(|| drain(black_box(&tensor_sequential)));
    });

    let tensor_random = DataLoader::new(&split, BATCH_SIZE).shuffle(5);
    group.bench_function("tensor_random_epoch/1024x64", |b| {
        b.iter(|| drain(black_box(&tensor_random)));
    });

    group.finish();
}

/// **New in v3.** The two collation primitives, isolated at one batch's shape.
///
/// An epoch above is `SAMPLES / BATCH_SIZE` calls to `Dataset::batch`, and every
/// one of those is either a [`Tensor::stack`] of `BATCH_SIZE` per-item rows
/// ([`VecDataset`]) or a single [`Tensor::index_select`] over the whole split
/// ([`TensorDataset`]). Which of the two dominates the epoch cost is not
/// something the epoch rows can answer, so they are measured directly here:
/// multiply either row by `SAMPLES / BATCH_SIZE` and compare against the
/// matching epoch row to see how much of the epoch is collation and how much is
/// the loader's own index bookkeeping.
///
/// Each primitive is measured at more than one size, because "collation is
/// slow" is not actionable but "collation is slow **per part**" is. `stack` is
/// measured at 4x the elements with the part count fixed and at 4x the parts
/// with the elements per part fixed; whichever axis moves the number is the one
/// a fix has to attack. `index_select` gets the same treatment at 32 and 128
/// gathered rows.
fn bench_collate(c: &mut Criterion) {
    let dev = Device::Cpu;
    let mut group = c.benchmark_group("collate");
    let mut rng = Rng::seed(6);

    // `stack` at three points chosen to separate the two ways its cost can
    // scale: 4x the elements at a fixed part count, and 4x the parts at a
    // fixed element count per part. `Tensor::stack` goes through
    // `concat_values`, which calls `transfer_out` once **per part** (a fresh
    // host buffer each) before a memcpy assembly, so the prediction is that
    // parts, not elements, dominate. `stack_32x64` is the one an epoch row
    // above actually pays.
    for (parts, features) in [
        (BATCH_SIZE, FEATURES),
        (BATCH_SIZE, FEATURES * 4),
        (BATCH_SIZE * 4, FEATURES),
    ] {
        let rows: Vec<Tensor> = (0..parts).map(|_| row(&mut rng, features, &dev)).collect();
        let row_refs: Vec<&Tensor> = rows.iter().collect();
        group.throughput(Throughput::Elements((parts * features) as u64));
        group.bench_function(format!("stack_{parts}x{features}"), |b| {
            b.iter(|| black_box(Tensor::stack(black_box(&row_refs), 0).unwrap()));
        });
    }

    let split = row(&mut rng, SAMPLES * FEATURES, &dev)
        .reshape([SAMPLES, FEATURES])
        .expect("bench split");

    // The same split gathered at 32 and at 128 rows: one `index_select` whose
    // work is entirely per output element, for contrast with `stack` above.
    for count in [BATCH_SIZE, BATCH_SIZE * 4] {
        let positions: Vec<usize> = (0..count).map(|i| i * 7 % SAMPLES).collect();
        let positions = Tensor::index_vec(&positions, &dev).expect("bench positions");
        group.throughput(Throughput::Elements((count * FEATURES) as u64));
        group.bench_function(format!("index_select_{count}of{SAMPLES}x{FEATURES}"), |b| {
            b.iter(|| {
                black_box(
                    black_box(&split)
                        .index_select(0, black_box(&positions))
                        .unwrap(),
                )
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_dataloader, bench_collate);
criterion_main!(benches);
