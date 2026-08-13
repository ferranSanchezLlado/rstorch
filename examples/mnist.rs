//! Train a small convolutional network on the real MNIST train split and
//! evaluate it on the official test split.
//!
//! The first run downloads MNIST into rstorch's dataset cache. Release mode is
//! strongly recommended:
//!
//! ```text
//! cargo run --release --example mnist --features hub
//! ```
//!
//! `RSTORCH_MNIST_EPOCHS` (default 3) and `RSTORCH_MNIST_BATCH` (default 128)
//! can be used to change the training budget.

use rstorch::data::hub::{DatasetHub, MnistDataset, MnistLayout, MnistSplit};
use rstorch::optim::schedule;
use rstorch::prelude::*;

const CHANNELS: usize = 8;
const CLASSES: usize = 10;
const IMAGE_SIDE: usize = 28;
const POOLED_SIDE: usize = IMAGE_SIDE / 2;
const LEARNING_RATE: f64 = 3e-3;
const MIN_LEARNING_RATE: f64 = 3e-4;

#[derive(Module)]
struct Cnn {
    conv_weight: Param,
    conv_bias: Param,
    classifier: Linear,
}

impl Cnn {
    fn new(device: &Device, rng: &mut Rng) -> Result<Cnn> {
        let fan_in = 3 * 3;
        let bound = (6.0 / fan_in as f64).sqrt();
        let weights = (0..CHANNELS * fan_in)
            .map(|_| rng.uniform(-bound, bound) as f32)
            .collect();
        Ok(Cnn {
            conv_weight: Param::new(Tensor::from_vec(weights, [CHANNELS, 1, 3, 3], device)?),
            conv_bias: Param::new(Tensor::zeros([CHANNELS, 1, 1], DType::F32, device)?),
            classifier: Linear::new(CHANNELS * POOLED_SIDE * POOLED_SIDE, CLASSES, device, rng)?,
        })
    }
}

impl Forward for Cnn {
    fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        let batch = x.dims()[0];
        let features = x
            .conv2d(&self.conv_weight.get(mode), (1, 1), (1, 1), (1, 1))?
            .add(&self.conv_bias.get(mode))?
            .relu()?
            .max_pool2d((2, 2), (2, 2), (0, 0))?
            .reshape([batch, CHANNELS * POOLED_SIDE * POOLED_SIDE])?;
        self.classifier.forward(&features, mode)
    }
}

fn standardized(dataset: MnistDataset) -> Result<TensorDataset> {
    TensorDataset::new(
        dataset.inputs().sub_scalar(0.1307)?.div_scalar(0.3081)?,
        dataset.targets().clone(),
    )
}

fn accuracy(model: &mut Cnn, dataset: &TensorDataset, batch_size: usize) -> Result<f64> {
    let mut correct = 0usize;
    let mut seen = 0usize;
    for batch in DataLoader::new(dataset, batch_size).batches() {
        let (images, labels) = batch?;
        let predictions = model
            .forward(&images, Mode::EVAL)?
            .argmax(-1)?
            .to_vec::<i64>()?;
        let labels = labels.to_vec::<i64>()?;
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
    let device = Device::best_available();
    let hub = DatasetHub::default_cache();

    println!("loading MNIST on {device} (downloading it on the first run)...");
    let train = standardized(MnistDataset::load(
        &hub,
        MnistSplit::Train,
        MnistLayout::Nchw,
        &device,
    )?)?;
    let test = standardized(MnistDataset::load(
        &hub,
        MnistSplit::Test,
        MnistLayout::Nchw,
        &device,
    )?)?;

    let mut model = Cnn::new(&device, &mut Rng::seed(42))?;
    let mut optimizer = Adam::new(LEARNING_RATE);
    let loader = DataLoader::new(&train, batch_size).shuffle(7);
    let total_steps = (epochs * loader.num_batches()) as u64;

    println!(
        "test accuracy before training: {:.2}%",
        100.0 * accuracy(&mut model, &test, batch_size)?
    );
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut batches = 0usize;
        for batch in loader.batches_for_epoch(epoch as u64) {
            let (images, labels) = batch?;
            let loss = model
                .forward(&images, Mode::TRAIN)?
                .cross_entropy(&labels)?;
            total_loss += loss.item()?;
            batches += 1;
            // Schedules use the count of successful updates, so a failed step
            // neither advances the optimizer nor silently skips the schedule.
            optimizer.set_lr(schedule::cosine(
                LEARNING_RATE,
                MIN_LEARNING_RATE,
                total_steps,
                optimizer.steps(),
            ));
            optimizer.step(&mut model, loss.backward()?)?;
        }
        println!(
            "epoch {:>2}/{epochs}  train loss {:.4}  lr {:.2e}  test accuracy {:.2}%",
            epoch + 1,
            total_loss / batches as f64,
            optimizer.lr(),
            100.0 * accuracy(&mut model, &test, batch_size)?,
        );
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
