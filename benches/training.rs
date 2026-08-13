//! End-to-end model benchmarks: a tiny decoder-only transformer's forward,
//! forward+backward and `AdamW` training step; an MNIST-shaped MLP training
//! epoch; autograd overhead (no-recording vs recorded vs recorded+backward on
//! the same graph); Conv2d/pooling forward and backward at small image sizes;
//! and a whole-tensor-reduction rank scan.
//!
//! Group ids (`transformer/forward`, `mlp/train_epoch_sgd/8x64x784`, …) are
//! stable across revisions so successive runs compare row for row.
//!
//! Two things about the workload are worth knowing before reading a number:
//!
//! - **`forward_no_grad` measures [`Mode::EVAL`]**, which records nothing
//!   (`Param::get` hands back the plain value); `Mode::TRAIN` records.
//! - **There is no `zero_grad`** — `backward()` returns a [`Grads`] value that
//!   `step` consumes — so the training rows carry one fewer parameter walk per
//!   iteration than a framework that clears gradients in place.
//!
//! Inputs are deterministic (a seeded `rstorch::Rng`) and never touch disk or
//! the network. Training benches mutate their model across iterations; that is
//! the steady state a real training loop measures.

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use rstorch::prelude::*;

fn benchmark_devices() -> Vec<Device> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    let devices = vec![Device::Cpu, Device::Metal(0)];
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    let devices = vec![Device::Cpu];
    devices
}

/// Deterministic uniform `[-1, 1)` f32 tensor on the CPU.
fn uniform(rng: &mut Rng, dims: &[usize], device: &Device) -> Tensor {
    let len: usize = dims.iter().product();
    let values: Vec<f32> = (0..len).map(|_| rng.uniform(-1.0, 1.0) as f32).collect();
    Tensor::from_vec(values, dims.to_vec(), device).expect("bench input")
}

/// `[rows]` I64 class labels cycling through `classes`.
fn labels(rows: usize, classes: usize, offset: usize, device: &Device) -> Tensor {
    let values: Vec<i64> = (0..rows).map(|i| ((i + offset) % classes) as i64).collect();
    Tensor::from_vec(values, [rows], device).expect("bench labels")
}

fn observe_model(module: &dyn Module) {
    let value = rstorch::nn::state_dict(module)
        .into_values()
        .next()
        .expect("bench model has parameters");
    black_box(
        value
            .to_vec::<f32>()
            .expect("bench completion read")
            .first()
            .copied(),
    );
}

// ---------------------------------------------------------------------------
// The tiny decoder-only transformer (v2's `DecoderOnlyTransformer<6,3,4,2,2,8,2>`)
// ---------------------------------------------------------------------------

const VOCAB: usize = 6;
const SEQ: usize = 3;
const EMBED: usize = 4;
const HEADS: usize = 2;
const FF: usize = 8;
const LAYERS: usize = 2;
const BATCH: usize = 2;

/// One pre-norm decoder block: `x + attn(LN(x))`, then `h + fc2(gelu(fc1(LN(h))))`.
///
/// Child paths are `norm1`, `attn`, `norm2`, `fc1`, `fc2`, matching the v2
/// block's persistence contract closely enough that a reader of the old
/// baseline recognizes the model.
#[derive(Module)]
struct Block {
    norm1: LayerNorm,
    attn: MultiHeadAttention,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl Block {
    fn new(rng: &mut Rng, dev: &Device) -> Result<Block> {
        Ok(Block {
            norm1: LayerNorm::new([EMBED], dev)?,
            attn: MultiHeadAttention::new(EMBED, HEADS, dev, rng)?,
            norm2: LayerNorm::new([EMBED], dev)?,
            fc1: Linear::new(EMBED, FF, dev, rng)?,
            fc2: Linear::new(FF, EMBED, dev, rng)?,
        })
    }

    fn run(&mut self, x: &Tensor, mask: &Tensor, mode: Mode) -> Result<Tensor> {
        let attn = self
            .attn
            .attend(&self.norm1.forward(x, mode)?, Some(mask), mode)?;
        let hidden = x.add(&attn)?;
        let normed = self.norm2.forward(&hidden, mode)?;
        let ff = self
            .fc2
            .forward(&self.fc1.forward(&normed, mode)?.gelu()?, mode)?;
        hidden.add(&ff)
    }
}

/// The tiny LM: token + learned position embeddings, `LAYERS` pre-norm blocks,
/// a final norm and an untied `[EMBED, VOCAB]` head.
#[derive(Module)]
struct TinyTransformer {
    tokens: Embedding,
    positions: Embedding,
    blocks: Vec<Block>,
    final_norm: LayerNorm,
    lm_head: Linear,
}

impl TinyTransformer {
    fn new(rng: &mut Rng, dev: &Device) -> Result<TinyTransformer> {
        let mut blocks = Vec::with_capacity(LAYERS);
        for _ in 0..LAYERS {
            blocks.push(Block::new(rng, dev)?);
        }
        Ok(TinyTransformer {
            tokens: Embedding::new(VOCAB, EMBED, dev, rng)?,
            positions: Embedding::new(SEQ, EMBED, dev, rng)?,
            blocks,
            final_norm: LayerNorm::new([EMBED], dev)?,
            lm_head: Linear::new(EMBED, VOCAB, dev, rng)?,
        })
    }

    /// `ids` is `[BATCH, SEQ]` I64; the result is `[BATCH, SEQ, VOCAB]` logits.
    ///
    /// `positions` and `mask` are built once by the caller and reused, which is
    /// what v2's cached `Mutex<Option<(usize, Mask)>>` amounted to after the
    /// first call — so neither side pays mask construction in the steady state.
    fn logits(
        &mut self,
        ids: &Tensor,
        positions: &Tensor,
        mask: &Tensor,
        mode: Mode,
    ) -> Result<Tensor> {
        let mut hidden = self
            .tokens
            .lookup(ids, mode)?
            .add(&self.positions.lookup(positions, mode)?)?;
        for block in &mut self.blocks {
            hidden = block.run(&hidden, mask, mode)?;
        }
        let normed = self.final_norm.forward(&hidden, mode)?;
        self.lm_head.forward(&normed, mode)
    }

    /// Mean next-token cross-entropy over the flattened `[BATCH * SEQ]` rows.
    fn loss(
        &mut self,
        ids: &Tensor,
        positions: &Tensor,
        mask: &Tensor,
        targets: &Tensor,
        mode: Mode,
    ) -> Result<Tensor> {
        self.logits(ids, positions, mask, mode)?
            .reshape([BATCH * SEQ, VOCAB])?
            .cross_entropy(targets)
    }
}

/// The v2 `transformer` group: forward, forward+backward, and an `AdamW` step.
fn bench_transformer(c: &mut Criterion) {
    for dev in benchmark_devices() {
        let mut group = c.benchmark_group(format!("transformer/{dev}"));

        let ids = Tensor::from_vec(vec![2i64, 4, 5, 4, 5, 3], [BATCH, SEQ], &dev).unwrap();
        let targets = Tensor::from_vec(vec![4i64, 5, 3, 5, 3, 0], [BATCH * SEQ], &dev).unwrap();
        let positions = Tensor::index_range(SEQ, &dev).unwrap();
        let mask = Tensor::causal_mask(SEQ, &dev).unwrap();

        let mut rng = Rng::seed(7);
        let mut model = TinyTransformer::new(&mut rng, &dev).unwrap();

        // Self-check: the ported model must actually train. A finite loss proves
        // the forward composes; a `Sgd::step` that returns `Ok` proves *every*
        // non-frozen parameter received a gradient (the `MissingGrad` pre-pass),
        // which is the failure mode a hand-ported model is most likely to hide —
        // an untraced weight would make these numbers meaninglessly fast.
        {
            // 418 trainable scalars is the v2 tiny config's parameter count: 24 + 12
            // for the two tables, 172 per block, 8 for the final norm, 30 for the
            // head. Pinning it means a later edit to the ported model cannot quietly
            // change the workload these numbers describe.
            assert_eq!(
                rstorch::nn::num_params(&model),
                418,
                "ported transformer must match the v2 tiny config's parameter count"
            );
            let loss = model
                .loss(&ids, &positions, &mask, &targets, Mode::TRAIN)
                .unwrap();
            assert!(
                loss.item().unwrap().is_finite(),
                "transformer loss diverged"
            );
            let grads = loss.backward().unwrap();
            Sgd::new(0.0)
                .step(&mut model, grads)
                .expect("every transformer parameter must receive a gradient");
        }

        group.bench_function("forward", |b| {
            b.iter(|| {
                let output = model
                    .logits(black_box(&ids), &positions, &mask, Mode::EVAL)
                    .unwrap();
                black_box(output.to_vec::<f32>().unwrap().first().copied())
            });
        });

        group.bench_function("forward_backward", |b| {
            b.iter(|| {
                let loss = model
                    .loss(&ids, &positions, &mask, &targets, Mode::TRAIN)
                    .unwrap();
                let grads = loss.backward().unwrap();
                let _ = black_box(grads);
                black_box(loss.item().unwrap());
            });
        });

        let mut rng = Rng::seed(7);
        let mut train_model = TinyTransformer::new(&mut rng, &dev).unwrap();
        let mut opt = AdamW::new(1e-3, 0.0);
        let warm_grads = train_model
            .loss(&ids, &positions, &mask, &targets, Mode::TRAIN)
            .unwrap()
            .backward()
            .unwrap();
        opt.step(&mut train_model, warm_grads).unwrap();
        observe_model(&train_model);
        group.bench_function("train_step_adamw", |b| {
            b.iter(|| {
                let grads = train_model
                    .loss(&ids, &positions, &mask, &targets, Mode::TRAIN)
                    .unwrap()
                    .backward()
                    .unwrap();
                opt.step(&mut train_model, grads).unwrap();
                observe_model(&train_model);
            });
        });

        group.finish();
    }
}

/// A 784 → 128 → 10 MLP, the shape both `mlp` and `autograd_overhead` use.
fn mnist_mlp(rng: &mut Rng, dev: &Device) -> Sequential {
    Sequential::new()
        .push(Linear::new(784, 128, dev, rng).unwrap())
        .push(Relu)
        .push(Linear::new(128, 10, dev, rng).unwrap())
}

/// One epoch of an MNIST-shaped MLP over eight synthetic 64-sample batches:
/// forward, cross-entropy, backward, SGD step per batch.
fn bench_mlp_epoch(c: &mut Criterion) {
    for dev in benchmark_devices() {
        let mut group = c.benchmark_group(format!("mlp/{dev}"));
        group.sample_size(10);

        let mut rng = Rng::seed(0);
        let mut model = mnist_mlp(&mut rng, &dev);
        let mut opt = Sgd::new(0.01);

        let batches: Vec<(Tensor, Tensor)> = (0..8)
            .map(|batch| {
                (
                    uniform(&mut rng, &[64, 784], &dev),
                    labels(64, 10, batch, &dev),
                )
            })
            .collect();

        for (input, targets) in &batches {
            let grads = model
                .forward(input, Mode::TRAIN)
                .unwrap()
                .cross_entropy(targets)
                .unwrap()
                .backward()
                .unwrap();
            opt.step(&mut model, grads).unwrap();
        }
        observe_model(&model);

        group.bench_function("train_epoch_sgd/8x64x784", |b| {
            b.iter(|| {
                for (input, targets) in &batches {
                    let grads = model
                        .forward(input, Mode::TRAIN)
                        .unwrap()
                        .cross_entropy(targets)
                        .unwrap()
                        .backward()
                        .unwrap();
                    opt.step(&mut model, grads).unwrap();
                }
                observe_model(&model);
            });
        });

        group.finish();
    }
}

/// Forward-only vs recorded-forward vs forward+backward on the same MLP graph.
///
/// The ratio between the last two is the autograd overhead; the gap between
/// the first two is what *recording* the graph costs on its own.
fn bench_autograd_overhead(c: &mut Criterion) {
    let dev = Device::Cpu;
    let mut group = c.benchmark_group("autograd_overhead");

    let mut rng = Rng::seed(1);
    let mut model = mnist_mlp(&mut rng, &dev);
    let input = uniform(&mut rng, &[64, 784], &dev);
    let targets = labels(64, 10, 0, &dev);

    group.bench_function("forward_no_grad", |b| {
        b.iter(|| {
            black_box(
                model
                    .forward(black_box(&input), Mode::EVAL)
                    .unwrap()
                    .cross_entropy(&targets)
                    .unwrap(),
            )
        });
    });

    group.bench_function("forward_recorded", |b| {
        b.iter(|| {
            black_box(
                model
                    .forward(black_box(&input), Mode::TRAIN)
                    .unwrap()
                    .cross_entropy(&targets)
                    .unwrap(),
            )
        });
    });

    group.bench_function("forward_backward", |b| {
        b.iter(|| {
            black_box(
                model
                    .forward(black_box(&input), Mode::TRAIN)
                    .unwrap()
                    .cross_entropy(&targets)
                    .unwrap()
                    .backward()
                    .unwrap(),
            )
        });
    });

    group.finish();
}

/// Conv2d and pooling at v2's sizes: a `[4, 1, 16, 16]` image through a
/// `1 → 8` 3x3 convolution (`[4, 8, 14, 14]`), then 2x2 pooling.
fn bench_conv_pool(c: &mut Criterion) {
    let dev = Device::Cpu;
    let mut group = c.benchmark_group("conv_pool");

    let mut rng = Rng::seed(2);
    // Kaiming-uniform fan_in for a 1x3x3 patch, matching v2's `Conv2d`.
    let weight = Param::new(uniform(&mut rng, &[8, 1, 3, 3], &dev));
    let bias = Param::new(Tensor::zeros([8, 1, 1], DType::F32, &dev).unwrap());
    let input = uniform(&mut rng, &[4, 1, 16, 16], &dev);

    let conv = |mode: Mode| -> Result<Tensor> {
        input
            .conv2d(&weight.get(mode), (1, 1), (0, 0), (1, 1))?
            .add(&bias.get(mode))
    };

    group.bench_function("conv2d_forward/4x1x16x16", |b| {
        b.iter(|| black_box(conv(Mode::EVAL).unwrap()));
    });

    let feature_map = conv(Mode::EVAL).unwrap();
    assert_eq!(feature_map.dims(), &[4, 8, 14, 14]);
    group.bench_function("max_pool2d_forward/4x8x14x14", |b| {
        b.iter(|| {
            black_box(
                black_box(&feature_map)
                    .max_pool2d((2, 2), (2, 2), (0, 0))
                    .unwrap(),
            )
        });
    });
    group.bench_function("avg_pool2d_forward/4x8x14x14", |b| {
        b.iter(|| {
            black_box(
                black_box(&feature_map)
                    .avg_pool2d((2, 2), (2, 2), (0, 0))
                    .unwrap(),
            )
        });
    });

    group.bench_function("conv_max_pool_forward_backward/4x1x16x16", |b| {
        b.iter(|| {
            let pooled = conv(Mode::TRAIN)
                .unwrap()
                .max_pool2d((2, 2), (2, 2), (0, 0))
                .unwrap();
            let _ = black_box(pooled.sum_all().unwrap().backward().unwrap());
        });
    });

    group.finish();
}

/// **New in v3.** Whole-tensor reductions at a *constant* element count and
/// growing rank — the measurement for the known `*_all` lead.
///
/// `sum_all` is a fold: one `axis_reduce` per axis, each a separate kernel
/// launch **and** a separate recorded graph node
/// (`tensor/ops/reduce.rs::fold_all`), so a rank-4 `sum_all` is 4 launches and
/// 4 nodes where one would do. Holding the element count fixed and varying only
/// the rank isolates that cost from the arithmetic: if it is real, the rows
/// separate by rank, and the recorded lane separates further because `backward`
/// then walks `rank` nodes instead of one.
///
/// Two element counts, because the answer can differ: at 262144 elements the
/// first fold does almost all the work and later folds are geometrically
/// smaller, while at 1024 elements per-launch overhead is the whole cost. This
/// group exists to give the fused-kernel work a number instead of an argument.
fn bench_reduce_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("reduce_all");

    // Every shape in a row holds the same element count, so only rank varies.
    let sizes: [(&str, [&[usize]; 4]); 2] = [
        ("1k", [&[1024], &[32, 32], &[8, 16, 8], &[4, 4, 8, 8]]),
        (
            "256k",
            [&[262144], &[512, 512], &[64, 64, 64], &[16, 16, 32, 32]],
        ),
    ];

    let mut rng = Rng::seed(4);
    for (label, shapes) in &sizes {
        for dims in shapes {
            let x = uniform(&mut rng, dims, &Device::Cpu);
            let param = Param::new(x.clone());
            let rank = dims.len();

            group.bench_function(format!("sum_all_f32/{label}_rank{rank}"), |b| {
                b.iter(|| black_box(black_box(&x).sum_all().unwrap()));
            });
            group.bench_function(format!("sum_all_backward_f32/{label}_rank{rank}"), |b| {
                b.iter(|| {
                    black_box(
                        param
                            .get(Mode::TRAIN)
                            .sum_all()
                            .unwrap()
                            .backward()
                            .unwrap(),
                    )
                });
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_transformer,
    bench_mlp_epoch,
    bench_autograd_overhead,
    bench_conv_pool,
    bench_reduce_all,
);
criterion_main!(benches);
