//! End-to-end model benchmarks: transformer forward and training step at the
//! tiny test config, an MNIST-style MLP training epoch on synthetic data,
//! autograd overhead (forward-only vs forward+backward on the same graph), and
//! Conv2d/pooling forward and backward at small image sizes.
//!
//! Inputs are deterministic (seeded [`SmallRng`]) and never touch disk or the
//! network. Models mutate across iterations in training benchmarks; that is
//! the steady state a real training loop measures.

use criterion::{Criterion, criterion_group, criterion_main};
use rstorch::prelude::*;
use std::hint::black_box;

fn uniform_f32(rng: &mut SmallRng, len: usize) -> Vec<f32> {
    (0..len).map(|_| rng.uniform(-1.0, 1.0)).collect()
}

#[derive(Debug)]
struct Rows;

/// The tiny transformer config used by the integration tests:
/// vocab 6, seq 3, embed 4, 2 heads of dim 2, FFN 8, 2 layers.
type TinyTransformer = DecoderOnlyTransformer<6, 3, 4, 2, 2, 8, 2>;

fn zero_grads<M: HasParameters<f32, Cpu>>(model: &M) {
    let mut refs = Vec::new();
    model.parameters(&mut refs);
    for parameter in &refs {
        parameter.zero_grad();
    }
}

fn bench_transformer(c: &mut Criterion) {
    let mut group = c.benchmark_group("transformer");

    let input = [[2, 4, 5], [4, 5, 3]];
    let target = [[4, 5, 3], [5, 3, 0]];

    let mut rng = SmallRng::seed_from_u64(7);
    let model = TinyTransformer::new(&mut rng).unwrap();
    group.bench_function("forward", |b| {
        b.iter(|| black_box(model.forward(black_box(&input)).unwrap()));
    });

    group.bench_function("forward_backward", |b| {
        b.iter(|| {
            model.loss(&input, &target).unwrap().backward().unwrap();
            zero_grads(&model);
        });
    });

    let mut rng = SmallRng::seed_from_u64(7);
    let mut train_model = TinyTransformer::new(&mut rng).unwrap();
    let mut opt = AdamW::new(1e-3, 0.0);
    group.bench_function("train_step_adamw", |b| {
        b.iter(|| {
            train_model
                .loss(&input, &target)
                .unwrap()
                .backward()
                .unwrap();
            let mut params = Vec::new();
            train_model.parameters_mut(&mut params);
            opt.step(&mut params).unwrap();
            drop(params);
            zero_grads(&train_model);
        });
    });

    group.finish();
}

/// One epoch of an MNIST-shaped MLP (784 -> 128 -> 10) over eight synthetic
/// 64-sample batches: forward, cross-entropy, backward, SGD step per batch.
fn bench_mlp_epoch(c: &mut Criterion) {
    let mut group = c.benchmark_group("mlp");
    group.sample_size(10);

    let mut rng = SmallRng::seed_from_u64(0);
    let mut model = Sequential::new(Linear::<784, 128>::kaiming_uniform(&mut rng).unwrap(), Relu)
        .add_module(Linear::<128, 10>::kaiming_uniform(&mut rng).unwrap());
    let mut opt = Sgd::new(0.01);
    let mut ctx = TrainContext::training(0);

    let batches: Vec<(Tensor2D<64, 784>, Vec<usize>)> = (0..8)
        .map(|batch| {
            let input = Tensor2D::<64, 784>::from_vec(uniform_f32(&mut rng, 64 * 784)).unwrap();
            let targets = (0..64).map(|sample| (batch + sample) % 10).collect();
            (input, targets)
        })
        .collect();

    group.bench_function("train_epoch_sgd/8x64x784", |b| {
        b.iter(|| {
            for (input, targets) in &batches {
                let loss = model
                    .forward(input, &mut ctx)
                    .unwrap()
                    .cross_entropy(targets)
                    .unwrap();
                loss.backward().unwrap();
                let mut params = Vec::new();
                model.parameters_mut(&mut params);
                opt.step(&mut params).unwrap();
                drop(params);
                zero_grads(&model);
            }
        });
    });

    group.finish();
}

/// Forward-only vs forward+backward on the same MLP graph. The ratio between
/// these is the autograd overhead; the no_grad variant shows what recording
/// the graph itself costs.
fn bench_autograd_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("autograd_overhead");

    let mut rng = SmallRng::seed_from_u64(1);
    let model = Sequential::new(Linear::<784, 128>::kaiming_uniform(&mut rng).unwrap(), Relu)
        .add_module(Linear::<128, 10>::kaiming_uniform(&mut rng).unwrap());
    let input = Tensor2D::<64, 784>::from_vec(uniform_f32(&mut rng, 64 * 784)).unwrap();
    let targets: Vec<usize> = (0..64).map(|sample| sample % 10).collect();
    let mut ctx = TrainContext::training(0);

    group.bench_function("forward_no_grad", |b| {
        b.iter(|| {
            let _guard = no_grad();
            black_box(
                model
                    .forward(black_box(&input), &mut ctx)
                    .unwrap()
                    .cross_entropy(&targets)
                    .unwrap(),
            );
        });
    });

    group.bench_function("forward_recorded", |b| {
        b.iter(|| {
            black_box(
                model
                    .forward(black_box(&input), &mut ctx)
                    .unwrap()
                    .cross_entropy(&targets)
                    .unwrap(),
            );
        });
    });

    group.bench_function("forward_backward", |b| {
        b.iter(|| {
            model
                .forward(black_box(&input), &mut ctx)
                .unwrap()
                .cross_entropy(&targets)
                .unwrap()
                .backward()
                .unwrap();
            zero_grads(&model);
        });
    });

    group.finish();
}

fn bench_conv_pool(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv_pool");

    let mut rng = SmallRng::seed_from_u64(2);
    let conv =
        Conv2d::<1, 8, 3, 3, 14, 14>::kaiming_uniform(&mut rng, Conv2dOptions::default()).unwrap();
    let max_pool = MaxPool2d::<2, 2, 7, 7>::new();
    let avg_pool = AvgPool2d::<2, 2, 7, 7>::new();
    let input = Tensor::<D4<Sym<Rows>, C<1>, C<16>, C<16>>>::from_vec_with_shape(
        uniform_f32(&mut rng, 4 * 16 * 16),
        [4, 1, 16, 16],
    )
    .unwrap();
    let mut ctx = TrainContext::eval();

    group.bench_function("conv2d_forward/4x1x16x16", |b| {
        b.iter(|| black_box(conv.forward(black_box(&input), &mut ctx).unwrap()));
    });

    let feature_map = conv.forward(&input, &mut ctx).unwrap();
    group.bench_function("max_pool2d_forward/4x8x14x14", |b| {
        b.iter(|| black_box(max_pool.forward(black_box(&feature_map), &mut ctx).unwrap()));
    });
    group.bench_function("avg_pool2d_forward/4x8x14x14", |b| {
        b.iter(|| black_box(avg_pool.forward(black_box(&feature_map), &mut ctx).unwrap()));
    });

    group.bench_function("conv_max_pool_forward_backward/4x1x16x16", |b| {
        b.iter(|| {
            let features = conv.forward(&input, &mut ctx).unwrap();
            let pooled = max_pool.forward(&features, &mut ctx).unwrap();
            pooled.sum().unwrap().backward().unwrap();
            zero_grads(&conv);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_transformer,
    bench_mlp_epoch,
    bench_autograd_overhead,
    bench_conv_pool,
);
criterion_main!(benches);
