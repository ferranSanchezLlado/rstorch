#![cfg(feature = "typed")]

//! Downstream typed MLP training and resume over an offline MNIST-like split.

use std::path::PathBuf;

use rstorch::data::TensorDataset as DynamicDataset;
use rstorch::nn::{Forward as DynamicForward, Linear as DynamicLinear};
use rstorch::persist::{Envelope, Limits, LoadOptions};
use rstorch::typed::data::{DataLoader, Dataset, TensorDataset};
use rstorch::typed::nn::{Forward, Linear, Mode, Relu, Sequential3, ToDevice, sequential};
use rstorch::typed::optim::{Adam, adam_step, load_adam_checkpoint, save_adam_checkpoint};
use rstorch::typed::persist::state_dict;
use rstorch::typed::{Cpu, DYN, DeviceCtx, Placement, Tensor0, Tensor1, Tensor2};
use rstorch::{DType, Device, Error, Result, Rng, Tensor};

const ROWS: usize = 10;
const COLS: usize = 10;
const FEATURES: usize = ROWS * COLS;
const HIDDEN: usize = 32;
const CLASSES: usize = 10;
const EPOCHS: u64 = 10;

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

type Input<P = Cpu> = Tensor2<DYN, FEATURES, f32, P>;
type Mlp<P = Cpu> =
    Sequential3<Input<P>, Linear<FEATURES, HIDDEN, f32, P>, Relu, Linear<HIDDEN, CLASSES, f32, P>>;
type TypedSplit = TensorDataset<Tensor1<FEATURES>, Tensor0<i64>>;
type Loader<'a> = DataLoader<&'a TypedSplit>;

fn build_model<P: Placement>(ctx: &DeviceCtx<P>, seed: u64) -> Result<Mlp<P>> {
    let mut rng = Rng::seed(seed);
    Ok(sequential::<Input<P>, _>((
        Linear::new(FEATURES, HIDDEN, ctx, &mut rng)?,
        Relu,
        Linear::new(HIDDEN, CLASSES, ctx, &mut rng)?,
    )))
}

fn dynamic_model(seed: u64) -> Result<rstorch::nn::Sequential> {
    let mut rng = Rng::seed(seed);
    Ok(rstorch::nn::Sequential::new()
        .push(DynamicLinear::new(
            FEATURES,
            HIDDEN,
            &Device::Cpu,
            &mut rng,
        )?)
        .push(rstorch::nn::Relu)
        .push(DynamicLinear::new(HIDDEN, CLASSES, &Device::Cpu, &mut rng)?))
}

fn pick(rng: &mut Rng, choices: usize) -> usize {
    (rng.uniform(0.0, choices as f64) as usize).min(choices - 1)
}

fn render(class: usize, rng: &mut Rng) -> Vec<f32> {
    let row_offset = 1 + pick(rng, 2);
    let col_offset = 2 + pick(rng, 2);
    let ink = rng.uniform(0.75, 1.0) as f32;
    let mut image = vec![0.0f32; FEATURES];
    for (row, stroke) in GLYPHS[class].iter().enumerate() {
        for (col, byte) in stroke.bytes().enumerate() {
            if byte == b'#' {
                image[(row + row_offset) * COLS + col + col_offset] = ink;
            }
        }
    }
    for pixel in &mut image {
        *pixel = (*pixel + rng.uniform(-0.06, 0.06) as f32).clamp(0.0, 1.0);
    }
    image
}

fn dynamic_split(per_class: usize, seed: u64) -> Result<DynamicDataset> {
    let mut rng = Rng::seed(seed);
    let mut features = Vec::with_capacity(per_class * CLASSES * FEATURES);
    let mut labels = Vec::with_capacity(per_class * CLASSES);
    for _ in 0..per_class {
        for class in 0..CLASSES {
            features.extend(render(class, &mut rng));
            labels.push(class as i64);
        }
    }
    DynamicDataset::new(
        Tensor::from_vec(features, [labels.len(), FEATURES], &Device::Cpu)?,
        Tensor::from_vec(labels.clone(), [labels.len()], &Device::Cpu)?,
    )
}

fn typed_split(per_class: usize, seed: u64, ctx: &DeviceCtx<Cpu>) -> Result<TypedSplit> {
    TensorDataset::try_from_dynamic(dynamic_split(per_class, seed)?, ctx)
}

/// Sample-weighted held-out mean, so uneven trailing batches are not
/// over-weighted and the number stays the dataset mean for any split size.
fn mean_loss(model: &mut Mlp, loader: &Loader<'_>) -> Result<f64> {
    let mut total = 0.0;
    let mut samples = 0usize;
    for batch in loader.batches() {
        let (features, labels): (Input, Tensor1<DYN, i64>) = batch?;
        let count = labels.dims()[0];
        let logits: Tensor2<DYN, CLASSES> = model.forward(&features, Mode::EVAL)?;
        total += logits.cross_entropy(&labels)?.item()? * count as f64;
        samples += count;
    }
    Ok(total / samples as f64)
}

fn dynamic_mean_loss(model: &mut rstorch::nn::Sequential, loader: &Loader<'_>) -> Result<f64> {
    let mut total = 0.0;
    let mut samples = 0usize;
    for batch in loader.batches() {
        let (features, labels): (Input, Tensor1<DYN, i64>) = batch?;
        let count = labels.dims()[0];
        let logits = model.forward(features.as_dynamic(), Mode::EVAL)?;
        total += logits.cross_entropy(labels.as_dynamic())?.item()? * count as f64;
        samples += count;
    }
    Ok(total / samples as f64)
}

fn train_epoch(
    model: &mut Mlp,
    optimizer: &mut Adam,
    loader: &Loader<'_>,
    epoch: u64,
) -> Result<()> {
    for batch in loader.batches_for_epoch(epoch) {
        let (features, labels): (Input, Tensor1<DYN, i64>) = batch?;
        let logits: Tensor2<DYN, CLASSES> = model.forward(&features, Mode::TRAIN)?;
        let loss = logits.cross_entropy(&labels)?;
        adam_step(optimizer, model, loss.backward()?)?;
    }
    Ok(())
}

/// Trains the dynamic peer off the same typed loader, so the batch order,
/// batch contents, and epoch schedule are the typed run's by construction.
fn dynamic_train_epoch(
    model: &mut rstorch::nn::Sequential,
    optimizer: &mut Adam,
    loader: &Loader<'_>,
    epoch: u64,
) -> Result<()> {
    for batch in loader.batches_for_epoch(epoch) {
        let (features, labels): (Input, Tensor1<DYN, i64>) = batch?;
        let logits = model.forward(features.as_dynamic(), Mode::TRAIN)?;
        let loss = logits.cross_entropy(labels.as_dynamic())?;
        optimizer.step(model, loss.backward()?)?;
    }
    Ok(())
}

/// Owns one unique directory for this run's checkpoints and removes it on
/// drop, so a failing assertion cannot strand `.rstorch` files in the temp dir.
struct CheckpointDir(PathBuf);

impl CheckpointDir {
    fn new() -> Result<CheckpointDir> {
        let root = std::env::temp_dir().join(format!(
            "rstorch-ct50-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&root)?;
        Ok(CheckpointDir(root))
    }

    fn path(&self, suffix: &str) -> PathBuf {
        self.0.join(format!("{suffix}.rstorch"))
    }
}

impl Drop for CheckpointDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-6,
            "value {index} differs: {actual} != {expected}"
        );
    }
}

#[test]
fn seeded_typed_mnist_mlp_learns_checkpoints_and_resumes() -> Result<()> {
    let ctx = DeviceCtx::<Cpu>::cpu()?;
    let train = typed_split(30, 11, &ctx)?;
    let held_out = typed_split(10, 977, &ctx)?;
    let train_loader = DataLoader::new(&train, 32).shuffle(5);
    let eval_loader = DataLoader::new(&held_out, 40);
    let mut model = build_model(&ctx, 42)?;
    let mut optimizer = Adam::new(0.01);

    let (batch_x, batch_y): (Input, Tensor1<DYN, i64>) = train.batch(&[0, 1, 2, 3, 4])?;
    assert_eq!(batch_x.dims(), [5, FEATURES]);
    assert_eq!(batch_y.dims(), [5]);
    assert_eq!(batch_x.as_dynamic().dtype(), DType::F32);
    assert_eq!(batch_y.as_dynamic().dtype(), DType::I64);
    assert_eq!(batch_x.as_dynamic().device(), Device::Cpu);

    let mut dynamic = dynamic_model(42)?;
    let mut dynamic_optimizer = Adam::new(0.01);
    let typed_logits: Tensor2<DYN, CLASSES> = model.forward(&batch_x, Mode::EVAL)?;
    let dynamic_logits = dynamic.forward(batch_x.as_dynamic(), Mode::EVAL)?;
    assert_close(&typed_logits.to_vec()?, &dynamic_logits.to_vec::<f32>()?);
    assert!(
        (typed_logits.cross_entropy(&batch_y)?.item()?
            - dynamic_logits.cross_entropy(batch_y.as_dynamic())?.item()?)
        .abs()
            <= 1e-6
    );

    let initial = mean_loss(&mut model, &eval_loader)?;
    let mut held_out_losses = Vec::new();
    for epoch in 0..EPOCHS {
        train_epoch(&mut model, &mut optimizer, &train_loader, epoch)?;
        dynamic_train_epoch(&mut dynamic, &mut dynamic_optimizer, &train_loader, epoch)?;
        held_out_losses.push(mean_loss(&mut model, &eval_loader)?);
    }
    let final_loss = *held_out_losses.last().unwrap();
    assert!(
        final_loss < initial * 0.35,
        "held-out loss did not show real learning: {initial} -> {final_loss}"
    );
    for pair in held_out_losses.windows(2) {
        assert!(
            pair[1] < pair[0],
            "held-out loss must decrease: {held_out_losses:?}"
        );
    }
    assert_eq!(
        optimizer.steps(),
        EPOCHS * train_loader.num_batches() as u64
    );

    // The whole typed training trajectory, not just an untrained forward pass,
    // must match the runtime peer that saw the same seeds and the same batches.
    assert_eq!(dynamic_optimizer.steps(), optimizer.steps());
    let dynamic_final_loss = dynamic_mean_loss(&mut dynamic, &eval_loader)?;
    assert!(
        (final_loss - dynamic_final_loss).abs() <= 1e-6,
        "typed and runtime trajectories diverged: {final_loss} != {dynamic_final_loss}"
    );
    let typed_trained: Tensor2<DYN, CLASSES> = model.forward(&batch_x, Mode::EVAL)?;
    let dynamic_trained = dynamic.forward(batch_x.as_dynamic(), Mode::EVAL)?;
    assert_close(&typed_trained.to_vec()?, &dynamic_trained.to_vec::<f32>()?);

    let paths = state_dict(&model)?
        .paths()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    assert_eq!(paths, ["0.bias", "0.weight", "2.bias", "2.weight"]);

    let checkpoints = CheckpointDir::new()?;
    let first_path = checkpoints.path("first");
    let replica_path = checkpoints.path("replica");
    let resumed_a_path = checkpoints.path("resumed-a");
    let resumed_b_path = checkpoints.path("resumed-b");
    let limits = Limits::defaults();
    save_adam_checkpoint(&optimizer, &mut model, &first_path, &limits)?;
    let first_bytes = std::fs::read(&first_path)?;
    assert!(!first_bytes.is_empty());
    let envelope = Envelope::load(&first_path, &limits)?;
    assert_eq!(envelope.section_names(), ["optimizer"]);
    assert_eq!(
        envelope
            .tensors()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        [
            "0.bias",
            "0.weight",
            "2.bias",
            "2.weight",
            "optim.0.bias.m",
            "optim.0.bias.v",
            "optim.0.weight.m",
            "optim.0.weight.v",
            "optim.2.bias.m",
            "optim.2.bias.v",
            "optim.2.weight.m",
            "optim.2.weight.v",
        ]
    );

    let mut replica = build_model(&ctx, 999)?;
    let mut replica_optimizer = Adam::new(0.5);
    load_adam_checkpoint(
        &mut replica_optimizer,
        &mut replica,
        &first_path,
        &LoadOptions::strict(),
    )?;
    save_adam_checkpoint(&replica_optimizer, &mut replica, &replica_path, &limits)?;
    assert_eq!(std::fs::read(&replica_path)?, first_bytes);
    assert_eq!(replica_optimizer.steps(), optimizer.steps());

    let loss_a = model
        .forward(&batch_x, Mode::TRAIN)?
        .cross_entropy(&batch_y)?;
    let loss_b = replica
        .forward(&batch_x, Mode::TRAIN)?
        .cross_entropy(&batch_y)?;
    assert!((loss_a.item()? - loss_b.item()?).abs() <= 1e-6);
    adam_step(&mut optimizer, &mut model, loss_a.backward()?)?;
    adam_step(&mut replica_optimizer, &mut replica, loss_b.backward()?)?;
    let output_a = model.forward(&batch_x, Mode::EVAL)?.to_vec()?;
    let output_b = replica.forward(&batch_x, Mode::EVAL)?.to_vec()?;
    assert_close(&output_a, &output_b);
    save_adam_checkpoint(&optimizer, &mut model, &resumed_a_path, &limits)?;
    save_adam_checkpoint(&replica_optimizer, &mut replica, &resumed_b_path, &limits)?;
    assert_eq!(
        std::fs::read(&resumed_a_path)?,
        std::fs::read(&resumed_b_path)?
    );

    // `checkpoints` removes the directory on drop, including the failure path.
    Ok(())
}

#[test]
fn public_boundaries_reject_width_and_placement_mutations() -> Result<()> {
    struct Auxiliary;
    impl Placement for Auxiliary {}

    let cpu = DeviceCtx::<Cpu>::cpu()?;
    let auxiliary = DeviceCtx::<Auxiliary>::bind(Device::Cpu)?;
    assert!(matches!(
        Linear::<{ FEATURES + 1 }, HIDDEN>::new(FEATURES, HIDDEN, &cpu, &mut Rng::seed(1),),
        Err(Error::InvalidArg {
            op: "typed::nn::Linear::new",
            ..
        })
    ));

    let source = build_model(&cpu, 3)?;
    let state = state_dict(&source)?;
    let mut wrong_placement = build_model(&auxiliary, 3)?;
    let loaded = rstorch::typed::persist::load_state_dict(&mut wrong_placement, &state);
    assert!(matches!(
        loaded,
        Err(Error::InvalidArg {
            op: "typed::nn::load_state_dict",
            ..
        })
    ));

    // A consuming placement conversion is the supported mutation and changes
    // the model's input/parameter placement types together.
    let converted: Mlp<Auxiliary> = source.to_device(&auxiliary)?;
    assert_eq!(state_dict(&converted)?.paths().count(), 4);
    Ok(())
}
