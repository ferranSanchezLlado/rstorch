//! **The m4 CNN acceptance fixture**: a downstream, public-API-only model
//! trains end to end on an offline MNIST-shaped split in NCHW layout.
//!
//! The split is a small seeded 5x7 stroke-font problem encoded as real IDX
//! bytes. It therefore exercises the MNIST parser, `MnistLayout::Nchw`, device
//! batching, `conv2d` and `max_pool2d`, real autograd through both kernels,
//! and an optimizer update. It is intentionally synthetic rather than a
//! downloaded copy of MNIST so the test is deterministic and network-free.

use rstorch::data::hub::{Mnist, MnistDataset, MnistLayout};
use rstorch::prelude::*;

const ROWS: usize = 10;
const COLS: usize = 10;
const CLASSES: usize = 10;
const CHANNELS: usize = 4;
const POOLED_ROWS: usize = ROWS / 2;
const POOLED_COLS: usize = COLS / 2;

/// A 5x7 stroke font for digits 0 through 9.
#[rustfmt::skip]
const GLYPHS: [[&str; 7]; CLASSES] = [
    [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
    ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    ["#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."],
    ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
    ["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."],
    ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
    [".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."],
];

fn pick(rng: &mut Rng, n: usize) -> usize {
    (rng.uniform(0.0, n as f64) as usize).min(n - 1)
}

fn render(class: usize, rng: &mut Rng) -> Vec<u8> {
    let row_offset = 1 + pick(rng, 2);
    let col_offset = 2 + pick(rng, 2);
    let ink = 255.0 * rng.uniform(0.65, 1.0);
    let mut image = vec![0u8; ROWS * COLS];

    for (row, glyph_row) in GLYPHS[class].iter().enumerate() {
        for (col, byte) in glyph_row.bytes().enumerate() {
            if byte == b'#' {
                image[(row_offset + row) * COLS + col_offset + col] = ink as u8;
            }
        }
    }
    for pixel in &mut image {
        let noisy = f64::from(*pixel) + rng.uniform(-20.0, 20.0);
        *pixel = noisy.clamp(0.0, 255.0) as u8;
    }
    image
}

fn synthetic_split(per_class: usize, seed: u64) -> Result<Mnist> {
    let count = per_class * CLASSES;
    let mut rng = Rng::seed(seed);
    let mut pixels = Vec::with_capacity(count * ROWS * COLS);
    let mut labels = Vec::with_capacity(count);
    for _ in 0..per_class {
        for class in 0..CLASSES {
            pixels.extend(render(class, &mut rng));
            labels.push(class as u8);
        }
    }

    let mut images = Vec::with_capacity(16 + pixels.len());
    images.extend(2051u32.to_be_bytes());
    images.extend(u32::try_from(count).unwrap().to_be_bytes());
    images.extend(u32::try_from(ROWS).unwrap().to_be_bytes());
    images.extend(u32::try_from(COLS).unwrap().to_be_bytes());
    images.extend(pixels);

    let mut targets = Vec::with_capacity(8 + labels.len());
    targets.extend(2049u32.to_be_bytes());
    targets.extend(u32::try_from(count).unwrap().to_be_bytes());
    targets.extend(labels);
    Mnist::from_idx_bytes(&images, &targets)
}

fn splits(device: &Device) -> Result<(MnistDataset, MnistDataset)> {
    let train = MnistDataset::new(&synthetic_split(30, 11)?, MnistLayout::Nchw, device)?;
    let held_out = MnistDataset::new(&synthetic_split(10, 977)?, MnistLayout::Nchw, device)?;
    Ok((train, held_out))
}

/// One honest convolutional stage followed by a learned classifier.
#[derive(Module)]
struct Cnn {
    conv_weight: Param,
    conv_bias: Param,
    head: Linear,
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
            head: Linear::new(CHANNELS * POOLED_ROWS * POOLED_COLS, CLASSES, device, rng)?,
        })
    }
}

impl Forward for Cnn {
    type Output = Tensor;

    fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        let batch = x.dims()[0];
        let features = x
            .conv2d(&self.conv_weight.get(mode), (1, 1), (1, 1), (1, 1))?
            .add(&self.conv_bias.get(mode))?
            .relu()?
            .max_pool2d((2, 2), (2, 2), (0, 0))?
            .reshape([batch, CHANNELS * POOLED_ROWS * POOLED_COLS])?;
        self.head.forward(&features, mode)
    }
}

type Batch = (Tensor, Tensor);

fn mean_loss(model: &mut Cnn, loader: &DataLoader<&MnistDataset>) -> Result<f64> {
    let mut total = 0.0;
    let mut batches = 0;
    for batch in loader.batches() {
        let (x, y): Batch = batch?;
        total += model.forward(&x, Mode::EVAL)?.cross_entropy(&y)?.item()?;
        batches += 1;
    }
    Ok(total / f64::from(batches))
}

fn accuracy(model: &mut Cnn, loader: &DataLoader<&MnistDataset>) -> Result<f64> {
    let mut correct = 0usize;
    let mut seen = 0usize;
    for batch in loader.batches() {
        let (x, y): Batch = batch?;
        let predicted = model.forward(&x, Mode::EVAL)?.argmax(-1)?.to_vec::<i64>()?;
        let targets = y.to_vec::<i64>()?;
        correct += predicted
            .iter()
            .zip(&targets)
            .filter(|(prediction, target)| prediction == target)
            .count();
        seen += targets.len();
    }
    Ok(correct as f64 / seen as f64)
}

#[test]
fn cnn_learns_from_an_nchw_mnist_split() -> Result<()> {
    const EPOCHS: u64 = 10;

    let device = Device::best_available();
    let (train, held_out) = splits(&device)?;
    assert_eq!(train.layout(), MnistLayout::Nchw);
    assert_eq!(train.inputs().dims(), &[300, 1, ROWS, COLS]);

    let mut rng = Rng::seed(42);
    let mut model = Cnn::new(&device, &mut rng)?;
    let initial_conv = model.conv_weight.value().to_vec::<f32>()?;
    let mut optimizer = Adam::new(3e-3);
    let train_loader = DataLoader::new(&train, 30).shuffle(0);
    let eval_loader = DataLoader::new(&held_out, 50);

    let start_loss = mean_loss(&mut model, &eval_loader)?;
    let start_accuracy = accuracy(&mut model, &eval_loader)?;
    assert!(
        start_accuracy < 0.3,
        "the seeded untrained model should be near chance, got {start_accuracy}"
    );
    let mut epoch_losses = Vec::new();
    for epoch in 0..EPOCHS {
        let mut total = 0.0;
        for batch in train_loader.batches_for_epoch(epoch) {
            let (x, y): Batch = batch?;
            let loss = model.forward(&x, Mode::TRAIN)?.cross_entropy(&y)?;
            total += loss.item()?;
            optimizer.step(&mut model, loss.backward()?)?;
        }
        epoch_losses.push(total / train_loader.num_batches() as f64);
    }

    let final_loss = mean_loss(&mut model, &eval_loader)?;
    let final_accuracy = accuracy(&mut model, &eval_loader)?;
    assert!(
        epoch_losses.last().unwrap() < &(0.5 * epoch_losses[0]),
        "training loss must materially decrease: {epoch_losses:?}"
    );
    assert!(
        final_loss < 0.6 * start_loss,
        "held-out loss must decrease: {start_loss} -> {final_loss}"
    );
    assert!(
        final_accuracy > 0.8,
        "held-out accuracy must beat random ({start_accuracy} -> {final_accuracy})"
    );
    assert_ne!(
        model.conv_weight.value().to_vec::<f32>()?,
        initial_conv,
        "the optimizer must update the convolution kernel"
    );
    assert_eq!(
        optimizer.steps(),
        EPOCHS * train_loader.num_batches() as u64
    );
    Ok(())
}
