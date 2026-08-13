//! Train a compile-time shape-checked CNN on the real MNIST dataset.
//!
//! The downloaded dataset enters through the dynamic hub API, then crosses a
//! checked, zero-copy typed boundary. From there, image geometry, channels,
//! convolution weights, flattened feature count, and logits all stay in the
//! type system.
//!
//! ```text
//! cargo run --release --example typed_mnist --features typed,hub
//! ```
//!
//! `RSTORCH_MNIST_EPOCHS` (default 3) and `RSTORCH_MNIST_BATCH` (default 128)
//! can be used to change the training budget. Set `RSTORCH_MNIST_CHECKPOINT`
//! to save a strict model-plus-optimizer checkpoint after every epoch and
//! resume from it on the next run. Keep the batch size unchanged when resuming,
//! because completed optimizer steps determine the next deterministic epoch.

use rstorch::data::hub::{DatasetHub, MnistDataset, MnistLayout, MnistSplit};
use rstorch::persist::{Limits, LoadOptions};
use rstorch::typed::data::{DataLoader, TensorDataset};
use rstorch::typed::nn::{Forward, Linear, Mode, TypedModule, TypedParam};
use rstorch::typed::optim::{Adam, adam_step, load_adam_checkpoint, save_adam_checkpoint};
use rstorch::typed::prelude::*;
use rstorch::{Device, Error, Result, Rng};

const CHANNELS: usize = 8;
const CLASSES: usize = 10;
const IMAGE_SIDE: usize = 28;
const POOLED_SIDE: usize = IMAGE_SIDE / 2;
const FEATURES: usize = CHANNELS * POOLED_SIDE * POOLED_SIDE;

type Images = Tensor4<DYN, 1, IMAGE_SIDE, IMAGE_SIDE>;
type Labels = Tensor1<DYN, i64>;
type Split = TensorDataset<Tensor3<1, IMAGE_SIDE, IMAGE_SIDE>, Tensor0<i64>>;
type ConvOutput = Tensor4<DYN, CHANNELS, DYN, DYN>;

#[derive(TypedModule)]
struct Cnn {
    conv_weight: TypedParam<Tensor4<CHANNELS, 1, 3, 3>>,
    classifier: Linear<FEATURES, CLASSES>,
}

impl Cnn {
    fn new(ctx: &DeviceCtx<Cpu>, rng: &mut Rng) -> Result<Cnn> {
        let bound = (6.0f64 / 9.0).sqrt();
        let weights = (0..CHANNELS * 9)
            .map(|_| rng.uniform(-bound, bound) as f32)
            .collect();
        Ok(Cnn {
            conv_weight: TypedParam::new(Tensor4::from_vec(weights, [CHANNELS, 1, 3, 3], ctx)?)?,
            classifier: Linear::new(FEATURES, CLASSES, ctx, rng)?,
        })
    }
}

impl Forward<Images> for Cnn {
    type Output = Tensor2<DYN, CLASSES>;

    fn forward(&mut self, images: &Images, mode: Mode) -> Result<Self::Output> {
        let batch = images.dims()[0];
        let weights = self.conv_weight.get(mode)?;
        let features: ConvOutput = images.conv2d(&weights, (1, 1), (1, 1), (1, 1))?;
        let features: ConvOutput = features.relu()?.max_pool2d((2, 2), (2, 2), (0, 0))?;
        let features = features.reshape::<Tensor2<DYN, FEATURES>>([batch, FEATURES])?;
        self.classifier.forward(&features, mode)
    }
}

fn load_split(split: MnistSplit, ctx: &DeviceCtx<Cpu>) -> Result<Split> {
    let dataset = MnistDataset::load(
        &DatasetHub::default_cache(),
        split,
        MnistLayout::Nchw,
        &Device::Cpu,
    )?;
    let normalized = rstorch::data::TensorDataset::new(
        dataset.inputs().sub_scalar(0.1307)?.div_scalar(0.3081)?,
        dataset.targets().clone(),
    )?;
    Split::try_from_dynamic(normalized, ctx)
}

fn accuracy(model: &mut Cnn, dataset: &Split, batch_size: usize) -> Result<f64> {
    let mut correct = 0usize;
    let mut seen = 0usize;
    for batch in DataLoader::new(dataset, batch_size).batches() {
        let (images, labels): (Images, Labels) = batch?;
        let predictions = model
            .forward(&images, Mode::EVAL)?
            .argmax::<1>()?
            .to_vec()?;
        let labels = labels.to_vec()?;
        correct += predictions
            .iter()
            .zip(&labels)
            .filter(|(prediction, label)| prediction == label)
            .count();
        seen += labels.len();
    }
    Ok(correct as f64 / seen as f64)
}

fn main() -> Result<()> {
    let epochs = env_usize("RSTORCH_MNIST_EPOCHS", 3).max(1);
    let batch_size = env_usize("RSTORCH_MNIST_BATCH", 128).max(1);
    let ctx = DeviceCtx::cpu()?;

    println!("loading MNIST on typed Cpu (downloading it on the first run)...");
    let train = load_split(MnistSplit::Train, &ctx)?;
    let test = load_split(MnistSplit::Test, &ctx)?;
    let mut model = Cnn::new(&ctx, &mut Rng::seed(42))?;
    let mut optimizer = Adam::new(3e-3);
    let loader = DataLoader::new(&train, batch_size).shuffle(7);
    let checkpoint = std::env::var_os("RSTORCH_MNIST_CHECKPOINT")
        .filter(|path| !path.is_empty())
        .map(std::path::PathBuf::from);
    if let Some(path) = checkpoint.as_ref().filter(|path| path.exists()) {
        load_adam_checkpoint(&mut optimizer, &mut model, path, &LoadOptions::strict())?;
        println!("resumed model and optimizer from {}", path.display());
    }
    let completed_steps = usize::try_from(optimizer.steps()).unwrap_or(usize::MAX);
    if completed_steps % loader.num_batches() != 0 {
        return Err(Error::Persistence {
            msg: "MNIST checkpoint was not saved at an epoch boundary".into(),
        });
    }
    let start_epoch = completed_steps / loader.num_batches();

    println!(
        "test accuracy after {start_epoch} completed epoch(s): {:.2}%",
        100.0 * accuracy(&mut model, &test, batch_size)?
    );
    for epoch in start_epoch..epochs {
        let mut total_loss = 0.0;
        let mut batches = 0usize;
        for batch in loader.batches_for_epoch(epoch as u64) {
            let (images, labels): (Images, Labels) = batch?;
            let logits: Tensor2<DYN, CLASSES> = model.forward(&images, Mode::TRAIN)?;
            let loss = logits.cross_entropy(&labels)?;
            total_loss += loss.item()?;
            batches += 1;
            adam_step(&mut optimizer, &mut model, loss.backward()?)?;
        }
        println!(
            "epoch {:>2}/{epochs}  train loss {:.4}  test accuracy {:.2}%",
            epoch + 1,
            total_loss / batches as f64,
            100.0 * accuracy(&mut model, &test, batch_size)?,
        );
        if let Some(path) = &checkpoint {
            save_adam_checkpoint(&optimizer, &mut model, path, &Limits::defaults())?;
            println!("  checkpointed model and optimizer to {}", path.display());
        }
    }
    Ok(())
}

fn env_usize(key: &str, default: usize) -> usize {
    match std::env::var(key) {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|_| panic!("{key} must be a non-negative integer, got {value:?}")),
        Err(_) => default,
    }
}
