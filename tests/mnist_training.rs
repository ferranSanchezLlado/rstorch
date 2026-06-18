#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use rstorch::prelude::*;

const IMAGE_HEIGHT: usize = 28;
const IMAGE_WIDTH: usize = 28;
const IMAGE_PIXELS: usize = IMAGE_HEIGHT * IMAGE_WIDTH;
const CLASSES: usize = 10;

type MnistCollator = ImageOneHotClassification<IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES>;

#[derive(Clone)]
struct SyntheticMnist {
    samples: Vec<([[f32; IMAGE_WIDTH]; IMAGE_HEIGHT], u8)>,
}

impl SyntheticMnist {
    fn new(samples_per_class: usize, rng: &mut SmallRng) -> Self {
        let mut samples = Vec::with_capacity(samples_per_class * CLASSES);
        for _ in 0..samples_per_class {
            for label in 0..CLASSES {
                samples.push((synthetic_image(label as u8, rng), label as u8));
            }
        }
        Self { samples }
    }

    fn labels(&self) -> Vec<u8> {
        self.samples.iter().map(|(_, label)| *label).collect()
    }
}

impl Dataset for SyntheticMnist {
    type Item = ([[f32; IMAGE_WIDTH]; IMAGE_HEIGHT], u8);

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        self.samples.get(index).copied()
    }
}

fn synthetic_image(label: u8, rng: &mut SmallRng) -> [[f32; IMAGE_WIDTH]; IMAGE_HEIGHT] {
    let mut image = [[0.0; IMAGE_WIDTH]; IMAGE_HEIGHT];
    for row in &mut image {
        for pixel in row {
            *pixel = rng.uniform_f32(0.0, 0.03);
        }
    }

    let class = usize::from(label);
    let block_row = class / 5;
    let block_col = class % 5;
    let row_start = 3 + block_row * 12;
    let col_start = 2 + block_col * 5;

    for row in row_start..row_start + 8 {
        for col in col_start..col_start + 4 {
            image[row][col] = rng.uniform_f32(0.9, 1.0);
        }
    }

    image
}

fn accuracy<const N: usize>(logits: &Tensor2D<N, CLASSES>, labels: &[u8]) -> f32 {
    assert_eq!(labels.len(), N, "accuracy labels must match the batch size");

    let values = logits.to_vec();
    let mut correct = 0;
    for (row, label) in values.chunks_exact(CLASSES).zip(labels.iter().copied()) {
        let prediction = row
            .iter()
            .enumerate()
            .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
            .map(|(index, _)| index as u8)
            .unwrap();
        if prediction == label {
            correct += 1;
        }
    }

    correct as f32 / N as f32
}

#[test]
fn synthetic_mnist_sequential_pipeline_trains_offline() {
    const BATCH: usize = 20;

    let mut data_rng = SmallRng::seed_from_u64(7);
    let train = SyntheticMnist::new(12, &mut data_rng);
    let held_out = SyntheticMnist::new(4, &mut data_rng);
    let held_out_labels = held_out.labels();

    let mut shuffle_rng = SmallRng::seed_from_u64(11);
    let train_loader = DataLoader::new(train)
        .shuffle(&mut shuffle_rng)
        .collate::<MnistCollator>()
        .batch_size::<BATCH>();
    let held_out_loader = DataLoader::new(held_out)
        .collate::<MnistCollator>()
        .batch_size::<BATCH>();

    let mut model_rng = SmallRng::seed_from_u64(42);
    let mut model = Sequential::new()
        .add_module(Linear::<IMAGE_PIXELS, 32>::kaiming_uniform(&mut model_rng))
        .add_module(ReLU)
        .add_module(Linear::<32, CLASSES>::xavier_uniform(&mut model_rng));
    let mut optimizer = Adam::new(0.01);

    let (images, labels) = train_loader.iter().next().expect("training batch");
    let initial_loss =
        cross_entropy_one_hot(&model.forward(&images.flatten_2d()), &labels).to_vec()[0];

    for _ in 0..12 {
        for (images, labels) in train_loader.iter() {
            model.zero_grad();
            let logits = model.forward(&images.flatten_2d());
            let loss = cross_entropy_one_hot(&logits, &labels);
            loss.backward();
            optimizer.step(model.parameters_mut());
        }
    }

    let (images, labels) = train_loader.iter().next().expect("training batch");
    let final_loss =
        cross_entropy_one_hot(&model.forward(&images.flatten_2d()), &labels).to_vec()[0];
    let (images, _) = held_out_loader.iter().next().expect("held-out batch");
    let held_out_accuracy = accuracy(
        &model.forward(&images.flatten_2d()),
        &held_out_labels[..BATCH],
    );

    assert!(
        final_loss < initial_loss,
        "loss did not decrease: initial={initial_loss}, final={final_loss}"
    );
    assert!(
        held_out_accuracy > 0.90,
        "held-out accuracy too low: {held_out_accuracy}"
    );
}

#[cfg(feature = "datasets")]
#[test]
#[ignore = "downloads and trains on the real MNIST dataset"]
fn real_mnist_sequential_pipeline_trains() -> Result<(), Box<dyn std::error::Error>> {
    const BATCH: usize = 64;

    let train = Mnist::train()?;
    let test = Mnist::test()?;
    let test_labels = test.labels()[..BATCH].to_vec();

    let mut shuffle_rng = SmallRng::seed_from_u64(11);
    let train_loader = DataLoader::new(train)
        .shuffle(&mut shuffle_rng)
        .collate::<MnistCollator>()
        .batch_size::<BATCH>();
    let test_loader = DataLoader::new(test)
        .collate::<MnistCollator>()
        .batch_size::<BATCH>();

    let mut model_rng = SmallRng::seed_from_u64(42);
    let mut model = Sequential::new()
        .add_module(Linear::<IMAGE_PIXELS, 128>::kaiming_uniform(&mut model_rng))
        .add_module(ReLU)
        .add_module(Linear::<128, CLASSES>::xavier_uniform(&mut model_rng));
    let mut optimizer = Adam::new(0.001);

    let (images, labels) = train_loader.iter().next().expect("training batch");
    let initial_loss =
        cross_entropy_one_hot(&model.forward(&images.flatten_2d()), &labels).to_vec()[0];

    for _ in 0..2 {
        for (images, labels) in train_loader.iter() {
            model.zero_grad();
            let logits = model.forward(&images.flatten_2d());
            let loss = cross_entropy_one_hot(&logits, &labels);
            loss.backward();
            optimizer.step(model.parameters_mut());
        }
    }

    let (images, labels) = train_loader.iter().next().expect("training batch");
    let final_loss =
        cross_entropy_one_hot(&model.forward(&images.flatten_2d()), &labels).to_vec()[0];
    let (images, _) = test_loader.iter().next().expect("test batch");
    let test_accuracy = accuracy(&model.forward(&images.flatten_2d()), &test_labels);

    assert!(
        final_loss < initial_loss,
        "loss did not decrease: initial={initial_loss}, final={final_loss}"
    );
    assert!(
        test_accuracy > 0.85,
        "test accuracy too low: {test_accuracy}"
    );

    Ok(())
}
