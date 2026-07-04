//! Core tensor-operation benchmarks: elementwise arithmetic, matmul shapes,
//! the softmax family, LayerNorm, bmm, and the f32 vs f16/bf16 dtype spread.
//!
//! Inputs are deterministic (seeded [`SmallRng`]) and never touch disk or the
//! network. Matmul throughput is reported in multiply-accumulate operations so
//! sizes can be compared directly.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rstorch::prelude::*;
use std::hint::black_box;

fn uniform_f32(rng: &mut SmallRng, len: usize) -> Vec<f32> {
    (0..len).map(|_| rng.uniform(-1.0, 1.0)).collect()
}

#[derive(Debug)]
struct Rows;

fn bench_elementwise(c: &mut Criterion) {
    let mut group = c.benchmark_group("elementwise");

    macro_rules! case {
        ($rows:literal x $cols:literal) => {{
            let numel = $rows * $cols;
            let mut rng = SmallRng::seed_from_u64(1);
            let lhs = Tensor2D::<$rows, $cols>::from_vec(uniform_f32(&mut rng, numel)).unwrap();
            let rhs = Tensor2D::<$rows, $cols>::from_vec(uniform_f32(&mut rng, numel)).unwrap();
            group.throughput(Throughput::Elements(numel as u64));
            group.bench_function(BenchmarkId::new("add_f32", numel), |b| {
                b.iter(|| black_box(black_box(&lhs).add(black_box(&rhs)).unwrap()));
            });
            group.bench_function(BenchmarkId::new("mul_f32", numel), |b| {
                b.iter(|| black_box(black_box(&lhs).mul(black_box(&rhs)).unwrap()));
            });
        }};
    }

    case!(32 x 32);
    case!(256 x 256);
    case!(1024 x 1024);
    group.finish();
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    group.sample_size(10);

    macro_rules! square {
        ($n:literal) => {{
            let mut rng = SmallRng::seed_from_u64(2);
            let lhs = Tensor2D::<$n, $n>::from_vec(uniform_f32(&mut rng, $n * $n)).unwrap();
            let rhs = Tensor2D::<$n, $n>::from_vec(uniform_f32(&mut rng, $n * $n)).unwrap();
            group.throughput(Throughput::Elements(($n * $n * $n) as u64));
            group.bench_function(BenchmarkId::new("square_f32", $n), |b| {
                b.iter(|| black_box(black_box(&lhs).matmul(black_box(&rhs)).unwrap()));
            });
        }};
    }

    square!(64);
    square!(128);
    square!(256);
    square!(512);
    square!(1024);

    // LM-shaped rectangles from the transformer path: an LM-head projection
    // (tokens x embed) @ (embed x vocab) and an FFN down-projection.
    let mut rng = SmallRng::seed_from_u64(3);
    let tokens_by_embed = Tensor2D::<128, 256>::from_vec(uniform_f32(&mut rng, 128 * 256)).unwrap();
    let embed_by_vocab = Tensor2D::<256, 512>::from_vec(uniform_f32(&mut rng, 256 * 512)).unwrap();
    group.throughput(Throughput::Elements(128 * 256 * 512));
    group.bench_function("lm_head_f32/128x256x512", |b| {
        b.iter(|| {
            black_box(
                black_box(&tokens_by_embed)
                    .matmul(black_box(&embed_by_vocab))
                    .unwrap(),
            )
        });
    });

    let tokens_by_ff = Tensor2D::<128, 1024>::from_vec(uniform_f32(&mut rng, 128 * 1024)).unwrap();
    let ff_by_embed = Tensor2D::<1024, 256>::from_vec(uniform_f32(&mut rng, 1024 * 256)).unwrap();
    group.throughput(Throughput::Elements(128 * 1024 * 256));
    group.bench_function("ffn_down_f32/128x1024x256", |b| {
        b.iter(|| {
            black_box(
                black_box(&tokens_by_ff)
                    .matmul(black_box(&ff_by_embed))
                    .unwrap(),
            )
        });
    });

    group.finish();
}

fn bench_softmax_family(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_family");

    macro_rules! case {
        ($rows:literal x $cols:literal) => {{
            let numel = $rows * $cols;
            let mut rng = SmallRng::seed_from_u64(4);
            let logits = Tensor2D::<$rows, $cols>::from_vec(uniform_f32(&mut rng, numel)).unwrap();
            let targets: Vec<usize> = (0..$rows).map(|row| row % $cols).collect();
            group.throughput(Throughput::Elements(numel as u64));
            group.bench_function(
                BenchmarkId::new("softmax_last_f32", concat!($rows, "x", $cols)),
                |b| b.iter(|| black_box(black_box(&logits).softmax_last().unwrap())),
            );
            group.bench_function(
                BenchmarkId::new("log_softmax_last_f32", concat!($rows, "x", $cols)),
                |b| b.iter(|| black_box(black_box(&logits).log_softmax_last().unwrap())),
            );
            group.bench_function(
                BenchmarkId::new("cross_entropy_f32", concat!($rows, "x", $cols)),
                |b| b.iter(|| black_box(black_box(&logits).cross_entropy(&targets).unwrap())),
            );
        }};
    }

    case!(32 x 128);
    case!(128 x 512);
    group.finish();
}

fn bench_layernorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("layernorm");

    let norm = LayerNorm::<256>::new(1e-5).unwrap();
    let mut rng = SmallRng::seed_from_u64(5);
    let input = Tensor::<D2<Sym<Rows>, C<256>>>::from_vec_with_shape(
        uniform_f32(&mut rng, 64 * 256),
        [64, 256],
    )
    .unwrap();
    let mut ctx = TrainContext::eval();
    group.throughput(Throughput::Elements(64 * 256));
    group.bench_function("forward_f32/64x256", |b| {
        b.iter(|| black_box(norm.forward(black_box(&input), &mut ctx).unwrap()));
    });
    group.finish();
}

fn bench_bmm(c: &mut Criterion) {
    let mut group = c.benchmark_group("bmm");

    let mut rng = SmallRng::seed_from_u64(6);
    let lhs =
        Tensor::<D3<C<8>, C<64>, C<64>>>::from_vec(uniform_f32(&mut rng, 8 * 64 * 64)).unwrap();
    let rhs =
        Tensor::<D3<C<8>, C<64>, C<64>>>::from_vec(uniform_f32(&mut rng, 8 * 64 * 64)).unwrap();
    group.throughput(Throughput::Elements(8 * 64 * 64 * 64));
    group.bench_function("f32/8x64x64x64", |b| {
        b.iter(|| black_box(black_box(&lhs).bmm(black_box(&rhs)).unwrap()));
    });
    group.finish();
}

/// The f32 vs f16/bf16 spread makes the cost of scalar `half` emulation on CPU
/// visible: low-precision dtypes have no hardware arithmetic on this path.
fn bench_dtype_spread(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtype_spread");

    macro_rules! case {
        ($dtype:ty, $label:literal, $convert:expr) => {{
            let convert = $convert;
            let mut rng = SmallRng::seed_from_u64(7);
            let lhs_values: Vec<$dtype> = uniform_f32(&mut rng, 128 * 128)
                .into_iter()
                .map(convert)
                .collect();
            let rhs_values: Vec<$dtype> = uniform_f32(&mut rng, 128 * 128)
                .into_iter()
                .map(convert)
                .collect();
            let lhs = Tensor2D::<128, 128, $dtype>::from_vec(lhs_values).unwrap();
            let rhs = Tensor2D::<128, 128, $dtype>::from_vec(rhs_values).unwrap();
            group.throughput(Throughput::Elements(128 * 128 * 128));
            group.bench_function(concat!("matmul_128/", $label), |b| {
                b.iter(|| black_box(black_box(&lhs).matmul(black_box(&rhs)).unwrap()));
            });

            let logits_values: Vec<$dtype> = uniform_f32(&mut rng, 128 * 512)
                .into_iter()
                .map(convert)
                .collect();
            let logits = Tensor2D::<128, 512, $dtype>::from_vec(logits_values).unwrap();
            group.throughput(Throughput::Elements(128 * 512));
            group.bench_function(concat!("softmax_last_128x512/", $label), |b| {
                b.iter(|| black_box(black_box(&logits).softmax_last().unwrap()));
            });
        }};
    }

    case!(f32, "f32", |value: f32| value);
    case!(f16, "f16", f16::from_f32);
    case!(bf16, "bf16", bf16::from_f32);
    group.finish();
}

criterion_group!(
    benches,
    bench_elementwise,
    bench_matmul,
    bench_softmax_family,
    bench_layernorm,
    bench_bmm,
    bench_dtype_spread,
);
criterion_main!(benches);
