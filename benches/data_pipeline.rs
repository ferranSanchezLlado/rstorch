//! DataLoader plus collator throughput on a synthetic in-memory dataset.
//!
//! Inputs are deterministic (seeded [`SmallRng`]) and never touch disk or the
//! network. Throughput is reported in samples per second over a full epoch.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use rstorch::prelude::*;
use std::hint::black_box;

const SAMPLES: usize = 1024;
const FEATURES: usize = 64;
const BATCH_SIZE: usize = 32;

fn synthetic_dataset() -> VecDataset<(Vec<f32>, usize)> {
    let mut rng = SmallRng::seed_from_u64(5);
    let samples = (0..SAMPLES)
        .map(|sample| {
            let values = (0..FEATURES).map(|_| rng.uniform(-1.0, 1.0)).collect();
            (values, sample % 10)
        })
        .collect();
    VecDataset::new(samples)
}

fn bench_dataloader(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataloader");
    group.throughput(Throughput::Elements(SAMPLES as u64));

    let sequential = DataLoader::new(
        synthetic_dataset(),
        SequentialSampler,
        features::<FEATURES>(),
        BATCH_SIZE,
        false,
    )
    .unwrap();
    group.bench_function("sequential_epoch/1024x64", |b| {
        b.iter(|| {
            for batch in sequential.iter() {
                black_box(batch.unwrap());
            }
        });
    });

    let random = DataLoader::new(
        synthetic_dataset(),
        RandomSampler::new(5),
        features::<FEATURES>(),
        BATCH_SIZE,
        false,
    )
    .unwrap();
    group.bench_function("random_epoch/1024x64", |b| {
        b.iter(|| {
            for batch in random.iter() {
                black_box(batch.unwrap());
            }
        });
    });

    group.finish();
}

criterion_group!(benches, bench_dataloader);
criterion_main!(benches);
