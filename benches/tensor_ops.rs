//! Core tensor-operation benchmarks: elementwise arithmetic, matmul shapes,
//! the softmax family, batched matmul, and the f32 vs f16/bf16 dtype spread.
//!
//! Ported from the v2 `benches/tensor_ops.rs` (api-reset2) to the v3
//! zero-generic public API: shapes are runtime values, so the v2 const-generic
//! `macro_rules!` cases collapse into plain loops over shape tables.
//!
//! Inputs are deterministic (a seeded `rstorch::Rng`) and never touch disk or
//! the network. Matmul throughput is reported in multiply-accumulate operations
//! so sizes can be compared directly.
//!
//! Two v2 groups are intentionally absent until their owning tasks land:
//! `layernorm` (needs `nn::LayerNorm`, T42) and the `cross_entropy` case of the
//! softmax family (needs the loss ops, T27). T47 re-adds them alongside the
//! training/data-pipeline bench ports.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rstorch::prelude::*;

/// Deterministic standard-normal tensor on the CPU. Every bench input goes
/// through here so a group's data depends only on its seed and shape.
fn randn(seed: u64, dims: &[usize], dtype: DType) -> Tensor {
    let mut rng = Rng::seed(seed);
    Tensor::randn(dims.to_vec(), dtype, &Device::Cpu, &mut rng).expect("bench input")
}

/// `rows x cols` label matching the v2 benchmark ids (`128x512`).
fn label(dims: &[usize]) -> String {
    dims.iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("x")
}

fn bench_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise");

    for dims in [[32usize, 32], [256, 256], [1024, 1024]] {
        let numel = dims[0] * dims[1];
        let lhs = randn(1, &dims, DType::F32);
        let rhs = randn(2, &dims, DType::F32);
        group.throughput(Throughput::Elements(numel as u64));
        group.bench_function(BenchmarkId::new("add_f32", numel), |b| {
            b.iter(|| black_box(black_box(&lhs).add(black_box(&rhs)).unwrap()));
        });
        group.bench_function(BenchmarkId::new("mul_f32", numel), |b| {
            b.iter(|| black_box(black_box(&lhs).mul(black_box(&rhs)).unwrap()));
        });
        group.bench_function(BenchmarkId::new("relu_f32", numel), |b| {
            b.iter(|| black_box(black_box(&lhs).relu().unwrap()));
        });
    }

    // Broadcasting a row vector across rows is the bias-add shape; it exercises
    // the zero-stride path of the element engine rather than the flat one.
    let rows = randn(3, &[512, 512], DType::F32);
    let bias = randn(4, &[1, 512], DType::F32);
    group.throughput(Throughput::Elements(512 * 512));
    group.bench_function("add_broadcast_f32/512x512", |b| {
        b.iter(|| black_box(black_box(&rows).add(black_box(&bias)).unwrap()));
    });

    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(5));

    for n in [64usize, 128, 256, 512, 1024] {
        let lhs = randn(5, &[n, n], DType::F32);
        let rhs = randn(6, &[n, n], DType::F32);
        // A single 1024³ product is ~0.8 s on the reference machine, so the
        // default window cannot fit ten samples; give the largest size more.
        group.measurement_time(Duration::from_secs(if n >= 1024 { 10 } else { 5 }));
        group.throughput(Throughput::Elements((n * n * n) as u64));
        group.bench_function(BenchmarkId::new("square_f32", n), |b| {
            b.iter(|| black_box(black_box(&lhs).matmul(black_box(&rhs)).unwrap()));
        });
    }

    group.measurement_time(Duration::from_secs(5));

    // LM-shaped rectangles from the transformer path: an LM-head projection
    // (tokens x embed) @ (embed x vocab) and an FFN down-projection.
    for (name, m, k, n) in [
        ("lm_head_f32", 128usize, 256usize, 512usize),
        ("ffn_down_f32", 128, 1024, 256),
    ] {
        let lhs = randn(7, &[m, k], DType::F32);
        let rhs = randn(8, &[k, n], DType::F32);
        group.throughput(Throughput::Elements((m * k * n) as u64));
        group.bench_function(BenchmarkId::new(name, label(&[m, k, n])), |b| {
            b.iter(|| black_box(black_box(&lhs).matmul(black_box(&rhs)).unwrap()));
        });
    }

    // A transposed rhs view stays strided in v3 (no pre-materialization), so
    // this measures the stride-aware walk against the contiguous case above.
    let lhs = randn(9, &[256, 256], DType::F32);
    let rhs_t = randn(10, &[256, 256], DType::F32)
        .transpose(0, 1)
        .expect("transpose");
    group.throughput(Throughput::Elements(256 * 256 * 256));
    group.bench_function("square_transposed_rhs_f32/256", |b| {
        b.iter(|| black_box(black_box(&lhs).matmul(black_box(&rhs_t)).unwrap()));
    });

    group.finish();
}

fn bench_softmax_family(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_family");

    for dims in [[32usize, 128], [128, 512]] {
        let logits = randn(11, &dims, DType::F32);
        group.throughput(Throughput::Elements((dims[0] * dims[1]) as u64));
        group.bench_function(BenchmarkId::new("softmax_last_f32", label(&dims)), |b| {
            b.iter(|| black_box(black_box(&logits).softmax(-1).unwrap()));
        });
        group.bench_function(
            BenchmarkId::new("log_softmax_last_f32", label(&dims)),
            |b| {
                b.iter(|| black_box(black_box(&logits).log_softmax(-1).unwrap()));
            },
        );
        group.bench_function(BenchmarkId::new("max_last_f32", label(&dims)), |b| {
            b.iter(|| black_box(black_box(&logits).max(-1).unwrap()));
        });
        group.bench_function(BenchmarkId::new("sum_all_f32", label(&dims)), |b| {
            b.iter(|| black_box(black_box(&logits).sum_all().unwrap()));
        });
    }

    group.finish();
}

fn bench_bmm(c: &mut Criterion) {
    let mut group = c.benchmark_group("bmm");
    group.sample_size(10);

    let lhs = randn(12, &[8, 64, 64], DType::F32);
    let rhs = randn(13, &[8, 64, 64], DType::F32);
    group.throughput(Throughput::Elements(8 * 64 * 64 * 64));
    group.bench_function("f32/8x64x64x64", |b| {
        b.iter(|| black_box(black_box(&lhs).matmul(black_box(&rhs)).unwrap()));
    });

    // v3 broadcasts batch dims: a shared `[1, k, n]` weight against a batched
    // activation should cost the same as the fully materialized batch.
    let shared = randn(14, &[1, 64, 64], DType::F32);
    group.throughput(Throughput::Elements(8 * 64 * 64 * 64));
    group.bench_function("f32_broadcast_rhs/8x64x64x64", |b| {
        b.iter(|| black_box(black_box(&lhs).matmul(black_box(&shared)).unwrap()));
    });

    group.finish();
}

/// The f32 vs f16/bf16 spread makes the cost of scalar `half` emulation on CPU
/// visible: low-precision dtypes have no hardware arithmetic on this path.
fn bench_dtype_spread(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtype_spread");
    group.sample_size(10);

    for (dtype, name) in [
        (DType::F32, "f32"),
        (DType::F16, "f16"),
        (DType::BF16, "bf16"),
    ] {
        let lhs = randn(15, &[128, 128], dtype);
        let rhs = randn(16, &[128, 128], dtype);
        group.throughput(Throughput::Elements(128 * 128 * 128));
        group.bench_function(BenchmarkId::new("matmul_128", name), |b| {
            b.iter(|| black_box(black_box(&lhs).matmul(black_box(&rhs)).unwrap()));
        });

        let logits = randn(17, &[128, 512], dtype);
        group.throughput(Throughput::Elements(128 * 512));
        group.bench_function(BenchmarkId::new("softmax_last_128x512", name), |b| {
            b.iter(|| black_box(black_box(&logits).softmax(-1).unwrap()));
        });

        let values = randn(18, &[256, 256], dtype);
        group.throughput(Throughput::Elements(256 * 256));
        group.bench_function(BenchmarkId::new("add_256x256", name), |b| {
            b.iter(|| black_box(black_box(&values).add(black_box(&values)).unwrap()));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_elementwise,
    bench_matmul,
    bench_softmax_family,
    bench_bmm,
    bench_dtype_spread,
);
criterion_main!(benches);
