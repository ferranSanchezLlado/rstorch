//! Deferred element-wise chain and allocation-shape benchmarks.
//!
//! The benchmark keeps construction outside Criterion's timed closure. Each
//! sample therefore measures the operation path itself, including realization
//! at the explicit host boundary, rather than input generation.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rstorch::prelude::*;

fn input(seed: u64, elements: usize) -> Tensor {
    let mut rng = Rng::seed(seed);
    Tensor::randn([elements], DType::F32, &Device::Cpu, &mut rng).expect("bench input")
}

fn chain(x: &Tensor, length: usize) -> Tensor {
    let mut value = x.clone();
    for index in 0..length {
        value = match index % 4 {
            0 => value.add_scalar(0.125).expect("add_scalar"),
            1 => value.mul_scalar(1.25).expect("mul_scalar"),
            2 => value.neg().expect("neg"),
            _ => value.sub_scalar(0.25).expect("sub_scalar"),
        };
    }
    value
}

fn bench_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("lazy_chain");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(3));
    for elements in [1 << 10, 1 << 20] {
        let x = input(elements as u64, elements);
        group.throughput(Throughput::Elements(elements as u64));
        for length in [1usize, 2, 4, 8] {
            group.bench_function(
                BenchmarkId::new("eager", format!("{elements}/{length}")),
                |b| {
                    b.iter(|| {
                        let _guard = rstorch::lazy::set_fusion(rstorch::lazy::Fusion::Off);
                        let value = chain(black_box(&x), length);
                        black_box(value.to_vec::<f32>().expect("host read"));
                    });
                },
            );
            group.bench_function(
                BenchmarkId::new("deferred", format!("{elements}/{length}")),
                |b| {
                    b.iter(|| {
                        let _guard = rstorch::lazy::set_fusion(rstorch::lazy::Fusion::On);
                        let value = chain(black_box(&x), length);
                        black_box(value.to_vec::<f32>().expect("host read"));
                    });
                },
            );
        }
    }
    group.finish();
}

fn binary_chain(lhs: &Tensor, rhs: &Tensor, length: usize) -> Tensor {
    let mut value = lhs.clone();
    for _ in 0..length {
        value = value.add(rhs).expect("binary add");
    }
    value
}

fn bench_binary_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("lazy_binary_chain");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    let elements = 1 << 20;
    let lhs = input(101, elements);
    let rhs = input(102, elements);
    group.throughput(Throughput::Elements(elements as u64));
    for length in [2usize, 4, 8] {
        for mode in [rstorch::lazy::Fusion::Off, rstorch::lazy::Fusion::On] {
            let label = match mode {
                rstorch::lazy::Fusion::Off => "eager",
                rstorch::lazy::Fusion::On => "deferred",
            };
            group.bench_function(BenchmarkId::new(label, length), |b| {
                b.iter(|| {
                    let _guard = rstorch::lazy::set_fusion(mode);
                    let value = binary_chain(black_box(&lhs), black_box(&rhs), length);
                    black_box(value.to_vec::<f32>().expect("host read"));
                });
            });
        }
    }
    group.finish();
}

fn bench_training_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("lazy_training_chain");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(3));
    let x = input(7, 1024 * 1024);
    for traced in [false, true] {
        let label = if traced { "traced" } else { "untraced" };
        group.throughput(Throughput::Elements(x.num_elements() as u64));
        group.bench_function(label, |b| {
            b.iter(|| {
                let _guard = rstorch::lazy::set_fusion(rstorch::lazy::Fusion::On);
                let source = if traced {
                    x.traced().expect("trace")
                } else {
                    x.clone()
                };
                let value = chain(&source, 4);
                if traced {
                    let _ = black_box(value.backward().expect("backward"));
                } else {
                    black_box(value.to_vec::<f32>().expect("host read"));
                }
            });
        });
    }
    group.finish();
}

fn bench_headline_shapes(c: &mut Criterion) {
    let mut group = c.benchmark_group("lazy_headline");
    group.sample_size(10);
    let x = input(41, 1024 * 1024)
        .reshape([1024, 1024])
        .expect("reshape");
    let weight = input(42, 1024 * 1024)
        .reshape([1024, 1024])
        .expect("reshape");
    let bias = input(43, 1024);
    group.throughput(Throughput::Elements(1024 * 1024));
    for mode in [rstorch::lazy::Fusion::Off, rstorch::lazy::Fusion::On] {
        let label = match mode {
            rstorch::lazy::Fusion::Off => "relu_matmul_eager",
            rstorch::lazy::Fusion::On => "relu_matmul_deferred",
        };
        group.bench_function(label, |b| {
            b.iter(|| {
                let _guard = rstorch::lazy::set_fusion(mode);
                let value = x
                    .matmul(&weight)
                    .and_then(|value| value.add(&bias))
                    .and_then(|value| value.relu())
                    .expect("relu matmul chain");
                black_box(value.to_vec::<f32>().expect("host read"));
            });
        });
    }

    let shared = input(44, 1 << 20);
    group.throughput(Throughput::Elements((1 << 20) as u64));
    for mode in [rstorch::lazy::Fusion::Off, rstorch::lazy::Fusion::On] {
        let label = match mode {
            rstorch::lazy::Fusion::Off => "shared_eager",
            rstorch::lazy::Fusion::On => "shared_deferred",
        };
        group.bench_function(label, |b| {
            b.iter(|| {
                let _guard = rstorch::lazy::set_fusion(mode);
                let middle = shared.add_scalar(1.0).expect("shared middle");
                let value = middle.mul(&middle).expect("shared chain");
                black_box(value.to_vec::<f32>().expect("host read"));
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_chain,
    bench_binary_chain,
    bench_training_chain,
    bench_headline_shapes
);
criterion_main!(benches);
