//! A `ResNet` over MNIST-shaped images, shared by the `resnet_mnist` bench and
//! the `resnet_mnist` convergence test.
//!
//! Included with `#[path]` by both targets rather than living in the library:
//! it assembles `nn::Conv2d`/`nn::BatchNorm2d` into a residual network, which
//! is model structure, not a library primitive. Every item is `pub` and the
//! module allows dead code, because each including target uses a different
//! subset.
//!
//! # Why this model
//!
//! The criterion suites in `benches/tensor_ops.rs` and `benches/training.rs`
//! measure single ops and toy models, where per-op host overhead dominates.
//! This is the opposite regime: a residual network deep enough that thread
//! scaling (`rayon`) and GPU occupancy (`metal`) are what the wall clock
//! actually reflects.
//!
//! # Shape knobs
//!
//! Both including targets read the same environment variables, so a bench run
//! and a convergence run can be pointed at the same network:
//!
//! | Variable | Default | Meaning |
//! |---|---|---|
//! | `RSTORCH_RESNET_DATA` | `synthetic` | `mnist` (needs `--features hub`) or `synthetic` |
//! | `RSTORCH_RESNET_WIDTH` | `16` | Channels in the first stage; doubled per stage |
//! | `RSTORCH_RESNET_BLOCKS` | `1` | Residual blocks per stage |
//! | `RSTORCH_RESNET_STAGES` | `3` | Stages (each after the first halves the resolution) |
//! | `RSTORCH_RESNET_BATCH` | `32` | Images per step |
//! | `RSTORCH_RESNET_SEED` | `7` | Seed for the weights and the synthetic data |

#![allow(dead_code)]

use std::env;

use rstorch::prelude::*;

// ---------------------------------------------------------------------------
// Layers
// ---------------------------------------------------------------------------

/// Kaiming-normal (`fan_in`, `ReLU` gain), bias-free — what `PyTorch`'s
/// `kaiming_normal_(mode="fan_in", nonlinearity="relu")` does, and what a
/// `ResNet` needs to train at this depth without a warm-up schedule. Every
/// convolution here is followed by a [`BatchNorm2d`], whose own shift
/// subsumes the bias `nn::Conv2d::new`'s default would otherwise add.
fn conv2d(
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    device: &Device,
    rng: &mut Rng,
) -> Result<Conv2d> {
    Ok(Conv2d::with_init(
        in_channels,
        out_channels,
        (kernel, kernel),
        device,
        |shape, device| {
            rstorch::nn::init::kaiming_normal(shape.to_vec(), 2f64.sqrt(), DType::F32, device, rng)
        },
    )?
    .with_stride((stride, stride))
    .with_padding((padding, padding))
    .without_bias())
}

/// The `1x1`-convolution shortcut used when a block changes shape.
#[derive(Module)]
pub struct Downsample {
    conv: Conv2d,
    norm: BatchNorm2d,
}

impl Forward for Downsample {
    type Output = Tensor;

    fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        self.norm.forward(&self.conv.forward(x, mode)?, mode)
    }
}

/// The post-activation basic block of the original `ResNet` paper:
/// `conv3x3 → BN → ReLU → conv3x3 → BN → (+ shortcut) → ReLU`.
#[derive(Module)]
pub struct BasicBlock {
    conv1: Conv2d,
    norm1: BatchNorm2d,
    conv2: Conv2d,
    norm2: BatchNorm2d,
    shortcut: Option<Downsample>,
}

impl BasicBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        device: &Device,
        rng: &mut Rng,
    ) -> Result<BasicBlock> {
        let shortcut = if stride != 1 || in_channels != out_channels {
            Some(Downsample {
                conv: conv2d(in_channels, out_channels, 1, stride, 0, device, rng)?,
                norm: BatchNorm2d::new(out_channels, device)?,
            })
        } else {
            None
        };
        Ok(BasicBlock {
            conv1: conv2d(in_channels, out_channels, 3, stride, 1, device, rng)?,
            norm1: BatchNorm2d::new(out_channels, device)?,
            conv2: conv2d(out_channels, out_channels, 3, 1, 1, device, rng)?,
            norm2: BatchNorm2d::new(out_channels, device)?,
            shortcut,
        })
    }
}

impl Forward for BasicBlock {
    type Output = Tensor;

    fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        let identity = match &mut self.shortcut {
            Some(shortcut) => shortcut.forward(x, mode)?,
            None => x.clone(),
        };
        let hidden = self
            .norm1
            .forward(&self.conv1.forward(x, mode)?, mode)?
            .relu()?;
        let hidden = self
            .norm2
            .forward(&self.conv2.forward(&hidden, mode)?, mode)?;
        hidden.add(&identity)?.relu()
    }
}

// ---------------------------------------------------------------------------
// The network
// ---------------------------------------------------------------------------

/// The shape of the network, in the dimensions worth scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResNetSpec {
    /// Channels in the first stage. Every later stage doubles it.
    pub width: usize,
    /// Residual blocks per stage.
    pub blocks_per_stage: usize,
    /// Stages. Every stage after the first halves the spatial resolution.
    pub stages: usize,
    /// Output classes (10 for MNIST).
    pub classes: usize,
}

impl ResNetSpec {
    /// The default: three stages of one 16/32/64-channel block — a
    /// CIFAR-style ResNet-8 pointed at 28x28 single-channel images, sized so
    /// a criterion sample takes about a second on a release CPU build.
    ///
    /// `RSTORCH_RESNET_BLOCKS=3 RSTORCH_RESNET_WIDTH=32` is the ResNet-20
    /// -shaped step up for when the point is to load the machine.
    pub fn from_env() -> ResNetSpec {
        ResNetSpec {
            width: env_usize("RSTORCH_RESNET_WIDTH", 16).max(1),
            blocks_per_stage: env_usize("RSTORCH_RESNET_BLOCKS", 1).max(1),
            stages: env_usize("RSTORCH_RESNET_STAGES", 3).max(1),
            classes: 10,
        }
    }

    /// The smallest network that still exercises every path (strided block,
    /// shortcut convolution, global pool), for the smoke test.
    pub fn tiny() -> ResNetSpec {
        ResNetSpec {
            width: 4,
            blocks_per_stage: 1,
            stages: 2,
            classes: 10,
        }
    }

    /// Convolutions in the built network, shortcuts included: one stem, two
    /// per block, and one shortcut for every stage that changes shape (all
    /// but the first).
    pub fn conv_layers(&self) -> usize {
        1 + 2 * self.stages * self.blocks_per_stage + (self.stages - 1)
    }

    /// A compact id for a criterion benchmark name: `w16b1s3`. Baselines are
    /// only comparable within one shape, so the shape belongs in the id.
    pub fn id(&self) -> String {
        format!("w{}b{}s{}", self.width, self.blocks_per_stage, self.stages)
    }
}

/// `stem → [stage]* → global average pool → linear`.
#[derive(Module)]
pub struct ResNet {
    stem: Conv2d,
    stem_norm: BatchNorm2d,
    blocks: Vec<BasicBlock>,
    head: Linear,
}

impl ResNet {
    pub fn new(
        spec: ResNetSpec,
        in_channels: usize,
        device: &Device,
        rng: &mut Rng,
    ) -> Result<ResNet> {
        let stem = conv2d(in_channels, spec.width, 3, 1, 1, device, rng)?;
        let stem_norm = BatchNorm2d::new(spec.width, device)?;

        let mut blocks = Vec::new();
        let mut channels = spec.width;
        for stage in 0..spec.stages {
            let out_channels = spec.width * (1 << stage);
            for block in 0..spec.blocks_per_stage {
                // The first block of every stage but the first halves the
                // resolution, as `torchvision`'s `_make_layer` does.
                let stride = usize::from(block == 0 && stage > 0) + 1;
                blocks.push(BasicBlock::new(
                    channels,
                    out_channels,
                    stride,
                    device,
                    rng,
                )?);
                channels = out_channels;
            }
        }
        Ok(ResNet {
            stem,
            stem_norm,
            blocks,
            head: Linear::new(channels, spec.classes, device, rng)?,
        })
    }

    /// Trainable parameters.
    pub fn params(&self) -> usize {
        self.num_params()
    }

    /// Forces a deferred backend (Metal) to finish everything queued behind
    /// it, via [`Tensor::realize`](rstorch::Tensor::realize) on one head
    /// parameter.
    ///
    /// Both including targets call this at the end of a training step. It is
    /// not free and it is not meant to be: without it a Metal "step" would
    /// time command encoding, and a real loop pays the same flush every time
    /// it logs a loss.
    pub fn sync(&self) -> Result<()> {
        self.head.weight().value().realize()
    }
}

impl Forward for ResNet {
    type Output = Tensor;

    fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        let mut hidden = self
            .stem_norm
            .forward(&self.stem.forward(x, mode)?, mode)?
            .relu()?;
        for block in &mut self.blocks {
            hidden = block.forward(&hidden, mode)?;
        }
        let [batch, channels, rows, cols] = *hidden.dims() else {
            unreachable!("a ResNet trunk keeps NCHW");
        };
        let pooled = hidden
            .avg_pool2d((rows, cols), (rows, cols), (0, 0))?
            .reshape([batch, channels])?;
        self.head.forward(&pooled, mode)
    }
}

/// One training step: forward, loss, backward, optimizer, host sync. Returns
/// the loss so a caller can watch it fall.
///
/// This is the unit both targets measure — the bench times it, the test checks
/// that repeating it learns.
pub fn train_step(
    model: &mut ResNet,
    optimizer: &mut Sgd,
    inputs: &Tensor,
    targets: &Tensor,
) -> Result<f64> {
    let loss = model.forward(inputs, Mode::TRAIN)?.cross_entropy(targets)?;
    let value = loss.item()?;
    let grads = loss.backward()?;
    optimizer.step(model, grads)?;
    model.sync()?;
    Ok(value)
}

/// Top-1 accuracy over `batches`, in [`Mode::EVAL`] (so `BatchNorm` uses its
/// running statistics rather than the batch's).
pub fn accuracy(model: &mut ResNet, batches: &[(Tensor, Tensor)]) -> Result<f64> {
    let mut correct = 0usize;
    let mut total = 0usize;
    for (inputs, targets) in batches {
        let predicted = model
            .forward(inputs, Mode::EVAL)?
            .argmax(1)?
            .to_vec::<i64>()?;
        let expected = targets.to_vec::<i64>()?;
        correct += predicted
            .iter()
            .zip(&expected)
            .filter(|(p, e)| p == e)
            .count();
        total += expected.len();
    }
    Ok(correct as f64 / total as f64)
}

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

/// Where the pixels come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// The real dataset, through the `hub` cache (downloads once).
    Mnist,
    /// Seeded noise with cycling labels: same shapes, same op sequence, no
    /// disk and no network. Throughput is comparable; accuracy is not, since
    /// the labels are unrelated to the pixels.
    Synthetic,
}

impl Source {
    pub fn from_env() -> Source {
        match env::var("RSTORCH_RESNET_DATA")
            .unwrap_or_else(|_| "synthetic".to_string())
            .to_ascii_lowercase()
            .as_str()
        {
            "mnist" => Source::Mnist,
            "synthetic" => Source::Synthetic,
            other => panic!("RSTORCH_RESNET_DATA must be `mnist` or `synthetic`, got {other:?}"),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Source::Mnist => "mnist",
            Source::Synthetic => "synthetic",
        }
    }
}

/// Materialized `(inputs, targets)` batches, already on the target device.
pub type Batches = Vec<(Tensor, Tensor)>;

/// Train and eval tensors, held on the CPU so a device sweep uploads the same
/// bytes to each backend instead of re-reading the dataset per device.
pub struct HostData {
    train_inputs: Tensor,
    train_targets: Tensor,
    eval_inputs: Tensor,
    eval_targets: Tensor,
}

impl HostData {
    /// `train_items` training images and `eval_items` held-out ones, `NCHW`
    /// and standardized.
    pub fn load(
        source: Source,
        train_items: usize,
        eval_items: usize,
        seed: u64,
    ) -> Result<HostData> {
        match source {
            Source::Synthetic => HostData::synthetic(train_items, eval_items.max(1), seed),
            Source::Mnist => HostData::mnist(train_items, eval_items.max(1)),
        }
    }

    fn synthetic(train_items: usize, eval_items: usize, seed: u64) -> Result<HostData> {
        let mut rng = Rng::seed(seed);
        let mut images = |items: usize| -> Result<Tensor> {
            let values: Vec<f32> = (0..items * 28 * 28)
                .map(|_| rng.normal(0.0, 1.0) as f32)
                .collect();
            Tensor::from_vec(values, [items, 1, 28, 28], &Device::Cpu)
        };
        let labels = |items: usize| -> Result<Tensor> {
            let values: Vec<i64> = (0..items).map(|i| (i % 10) as i64).collect();
            Tensor::from_vec(values, [items], &Device::Cpu)
        };
        Ok(HostData {
            train_inputs: images(train_items)?,
            train_targets: labels(train_items)?,
            eval_inputs: images(eval_items)?,
            eval_targets: labels(eval_items)?,
        })
    }

    #[cfg(feature = "hub")]
    fn mnist(train_items: usize, eval_items: usize) -> Result<HostData> {
        use rstorch::data::Dataset as _;
        use rstorch::data::hub::{DatasetHub, MnistDataset, MnistLayout, MnistSplit};

        let hub = DatasetHub::default_cache();
        let prefix = |split: MnistSplit, items: usize| -> Result<(Tensor, Tensor)> {
            let dataset = MnistDataset::load(&hub, split, MnistLayout::Nchw, &Device::Cpu)?;
            assert!(
                items <= dataset.len(),
                "asked for {items} items of the MNIST {split:?} split, which has {}; \
                 lower the step count, the batch size, or the eval batches",
                dataset.len()
            );
            // The conventional MNIST standardization, applied once to the whole
            // split rather than per batch.
            let inputs = dataset
                .inputs()
                .narrow(0, 0, items)?
                .sub_scalar(0.1307)?
                .div_scalar(0.3081)?;
            Ok((inputs, dataset.targets().narrow(0, 0, items)?))
        };

        let (train_inputs, train_targets) = prefix(MnistSplit::Train, train_items)?;
        let (eval_inputs, eval_targets) = prefix(MnistSplit::Test, eval_items)?;
        Ok(HostData {
            train_inputs,
            train_targets,
            eval_inputs,
            eval_targets,
        })
    }

    #[cfg(not(feature = "hub"))]
    fn mnist(_train_items: usize, _eval_items: usize) -> Result<HostData> {
        panic!(
            "RSTORCH_RESNET_DATA=mnist needs the `hub` feature for the download + gzip path; \
             rerun with `--features hub`, or drop the variable to use synthetic pixels"
        );
    }

    /// The data uploaded to `device` and batched: `train_batches` shuffled
    /// training batches and `eval_batches` evaluation batches.
    ///
    /// Batching happens here so that neither target times `index_select` — the
    /// data pipeline has its own bench (`benches/data_pipeline.rs`).
    pub fn batches(
        &self,
        device: &Device,
        batch: usize,
        train_batches: usize,
        eval_batches: usize,
        seed: u64,
    ) -> Result<(Batches, Batches)> {
        let train = TensorDataset::new(
            self.train_inputs.to_device(device)?,
            self.train_targets.to_device(device)?,
        )?;
        let train = DataLoader::new(train, batch)
            .shuffle(seed)
            .drop_last(true)
            .batches()
            .take(train_batches)
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(
            train.len(),
            train_batches,
            "the training data yielded fewer whole batches than asked for"
        );

        let eval = TensorDataset::new(
            self.eval_inputs.to_device(device)?,
            self.eval_targets.to_device(device)?,
        )?;
        let eval = DataLoader::new(eval, batch)
            .drop_last(true)
            .batches()
            .take(eval_batches)
            .collect::<Result<Vec<_>>>()?;
        Ok((train, eval))
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Every device this build can reach, CPU first.
///
/// Metal is probed rather than assumed: `metal` is a default feature, so a
/// macOS build always compiles the branch, and a trial allocation is what
/// distinguishes a machine with a usable device from one without. Both
/// including targets honour `RSTORCH_SKIP_METAL_TESTS` for the same reason the
/// test suites do (README documents it as the opt-out on CPU-only machines and
/// doc builds).
pub fn devices() -> Vec<Device> {
    #[allow(unused_mut)]
    let mut devices = vec![Device::Cpu];
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if env::var_os("RSTORCH_SKIP_METAL_TESTS").is_none()
        && Tensor::zeros([1], DType::F32, &Device::Metal(0)).is_ok()
    {
        devices.push(Device::Metal(0));
    }
    devices
}

/// How the build was configured, for the report header: `rayon` is a
/// compile-time feature, so it can never be swept at runtime.
pub fn build_label() -> String {
    format!(
        "rayon {}, {} cores visible, {}",
        if cfg!(feature = "rayon") { "ON" } else { "off" },
        std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get),
        if cfg!(debug_assertions) {
            "DEBUG build (timings are meaningless; use --release)"
        } else {
            "release build"
        }
    )
}

pub fn env_usize(key: &str, default: usize) -> usize {
    match env::var(key) {
        Ok(raw) => raw
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("{key} must be a non-negative integer, got {raw:?}")),
        Err(_) => default,
    }
}

/// The batch size both targets use.
pub fn batch_size() -> usize {
    env_usize("RSTORCH_RESNET_BATCH", 32).max(1)
}

/// The seed for weights and synthetic data.
pub fn seed() -> u64 {
    env_usize("RSTORCH_RESNET_SEED", 7) as u64
}
