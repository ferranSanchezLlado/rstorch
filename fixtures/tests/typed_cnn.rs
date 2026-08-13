#![cfg(feature = "typed")]

//! Public-API-only typed CNN acceptance fixture. The data are generated from a
//! seeded three-pattern image problem, so training is deterministic and offline.

use rstorch::persist::{Envelope, Limits, LoadOptions};
use rstorch::typed::data::{DataLoader, Dataset, TensorDataset};
use rstorch::typed::nn::{
    BatchNorm2d, Forward, Linear, Mode, Relu, Sequential2, TypedModule, TypedParam, TypedStateDict,
    sequential, state_dict,
};
use rstorch::typed::optim::{
    Adam, adam_param_steps, adam_step, load_adam_checkpoint, save_adam_checkpoint,
};
use rstorch::typed::prelude::*;
use rstorch::{Result, Rng};
use std::path::PathBuf;

const HEIGHT: usize = 8;
const WIDTH: usize = 6;
const CHANNELS: usize = 4;
const CLASSES: usize = 3;
const FEATURES: usize = CHANNELS * (HEIGHT / 2) * (WIDTH / 2);
const TRAIN_ITEMS: usize = 51;
const EVAL_ITEMS: usize = 18;
const BATCH_SIZE: usize = 8;

type Images = Tensor4<DYN, 1, HEIGHT, WIDTH>;
type Labels = Tensor1<DYN, i64>;
type Image = Tensor3<1, HEIGHT, WIDTH>;
type Label = Tensor0<i64>;
type Split = TensorDataset<Image, Label>;
type ConvOutput = Tensor4<DYN, CHANNELS, DYN, DYN>;
type Trunk = Sequential2<ConvOutput, BatchNorm2d<CHANNELS>, Relu>;

#[derive(TypedModule)]
struct Cnn {
    conv: TypedParam<Tensor4<CHANNELS, 1, 3, 3>>,
    trunk: Trunk,
    head: Linear<FEATURES, CLASSES>,
}

impl Cnn {
    fn new(ctx: &DeviceCtx<Cpu>, seed: u64) -> Result<Self> {
        let mut rng = Rng::seed(seed);
        let bound = (6.0f64 / 9.0).sqrt();
        let weights = (0..CHANNELS * 9)
            .map(|_| rng.uniform(-bound, bound) as f32)
            .collect();
        Ok(Self {
            conv: TypedParam::new(Tensor4::from_vec(weights, [CHANNELS, 1, 3, 3], ctx)?)?,
            trunk: sequential::<ConvOutput, _>((BatchNorm2d::new(CHANNELS, ctx)?, Relu)),
            head: Linear::new(FEATURES, CLASSES, ctx, &mut rng)?,
        })
    }
}

impl Forward<Images> for Cnn {
    type Output = Tensor2<DYN, CLASSES>;

    fn forward(&mut self, input: &Images, mode: Mode) -> Result<Self::Output> {
        let batch = input.dims()[0];
        let weight = self.conv.get(mode)?;
        let features: ConvOutput = input.conv2d(&weight, (1, 1), (1, 1), (1, 1))?;
        let features = self.trunk.forward(&features, mode)?;
        let features: ConvOutput = features.max_pool2d((2, 2), (2, 2), (0, 0))?;
        let features = features.reshape::<Tensor2<DYN, FEATURES>>([batch, FEATURES])?;
        self.head.forward(&features, mode)
    }
}

fn generated_split(count: usize, seed: u64, ctx: &DeviceCtx<Cpu>) -> Result<Split> {
    let mut rng = Rng::seed(seed);
    let mut pixels = Vec::with_capacity(count * HEIGHT * WIDTH);
    let mut labels = Vec::with_capacity(count);
    for item in 0..count {
        let class = item % CLASSES;
        labels.push(class as i64);
        for row in 0..HEIGHT {
            for col in 0..WIDTH {
                let ink = match class {
                    0 => col == 1 || col == 2,
                    1 => col == WIDTH - 2 || col == WIDTH - 1,
                    _ => row == 3 || row == 4,
                };
                let base = if ink { 1.0 } else { 0.0 };
                pixels.push((base + rng.uniform(-0.08, 0.08)) as f32);
            }
        }
    }
    TensorDataset::new(
        Tensor4::<DYN, 1, HEIGHT, WIDTH>::from_vec(pixels, [count, 1, HEIGHT, WIDTH], ctx)?,
        Tensor1::<DYN, i64>::from_vec(labels, [count], ctx)?,
    )
}

fn metrics(model: &mut Cnn, loader: &DataLoader<&Split>) -> Result<(f64, f64)> {
    let mut loss = 0.0;
    let mut correct = 0usize;
    let mut seen = 0usize;
    for batch in loader.batches() {
        let (images, labels): (Images, Labels) = batch?;
        let logits = model.forward(&images, Mode::EVAL)?;
        let predicted = logits.argmax::<1>()?.to_vec()?;
        let expected = labels.to_vec()?;
        // `cross_entropy` returns a per-batch mean and the tail batch is short,
        // so weight each mean by its batch size to get the per-item mean.
        loss += logits.cross_entropy(&labels)?.item()? * expected.len() as f64;
        correct += predicted
            .iter()
            .zip(&expected)
            .filter(|(actual, expected)| actual == expected)
            .count();
        seen += expected.len();
    }
    Ok((loss / seen as f64, correct as f64 / seen as f64))
}

fn train_epoch(
    model: &mut Cnn,
    optimizer: &mut Adam,
    loader: &DataLoader<&Split>,
    epoch: u64,
) -> Result<f64> {
    let mut total = 0.0;
    let mut batches = 0usize;
    for batch in loader.batches_for_epoch(epoch) {
        let (images, labels): (Images, Labels) = batch?;
        let loss = model
            .forward(&images, Mode::TRAIN)?
            .cross_entropy(&labels)?;
        total += loss.item()?;
        batches += 1;
        adam_step(optimizer, model, loss.backward()?)?;
    }
    Ok(total / batches as f64)
}

fn checkpoint_path(tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "rstorch-typed-cnn-{}-{tag}.rst",
        std::process::id()
    ))
}

fn assert_paths(state: &TypedStateDict) {
    assert_eq!(
        state.paths().collect::<Vec<_>>(),
        [
            "conv",
            "head.bias",
            "head.weight",
            "trunk.0.bias",
            "trunk.0.running_mean",
            "trunk.0.running_var",
            "trunk.0.weight",
        ]
    );
}

/// Every extent here is distinct (input 5x4, kernel 2x3, pool 2x1) so a height
/// and width transposition anywhere in the typed shape algebra or in the
/// backend geometry changes the asserted output dims instead of cancelling out.
#[test]
fn typed_conv_pool_values_and_gradients_match_the_dynamic_view() -> Result<()> {
    let ctx = DeviceCtx::cpu()?;
    let dynamic_input = rstorch::Tensor::from_vec(
        (0..20).map(|n| n as f32 / 20.0).collect(),
        [1, 1, 5, 4],
        &rstorch::Device::Cpu,
    )?
    .traced()?;
    let dynamic_weight = rstorch::Tensor::from_vec(
        vec![0.25f32, -0.5, 0.75, 1.0, -0.125, 0.5],
        [1, 1, 2, 3],
        &rstorch::Device::Cpu,
    )?
    .traced()?;
    let input = Tensor4::<1, 1, 5, 4>::try_from_dynamic(dynamic_input.clone(), &ctx)?;
    let weight = Tensor4::<1, 1, 2, 3>::try_from_dynamic(dynamic_weight.clone(), &ctx)?;

    let typed_conv: Tensor4<1, 1, DYN, DYN> = input.conv2d(&weight, (1, 1), (0, 0), (1, 1))?;
    let dynamic_conv = dynamic_input.conv2d(&dynamic_weight, (1, 1), (0, 0), (1, 1))?;
    assert_eq!(typed_conv.dims(), [1, 1, 4, 2]);
    assert_eq!(typed_conv.to_vec()?, dynamic_conv.to_vec::<f32>()?);

    let typed_pool: Tensor4<1, 1, DYN, DYN> = typed_conv.avg_pool2d((2, 1), (1, 1), (0, 0))?;
    let dynamic_pool = dynamic_conv.avg_pool2d((2, 1), (1, 1), (0, 0))?;
    assert_eq!(typed_pool.dims(), [1, 1, 3, 2]);
    assert_eq!(typed_pool.to_vec()?, dynamic_pool.to_vec::<f32>()?);

    // Back-propagate the two graphs independently: reading one `Grads` twice
    // would only exercise the typed lookup adapter, not the gradients. Even so,
    // the typed op delegates to the same kernel as the dynamic one, so this can
    // never catch a wrong kernel - only a wrong typed wrapper or a broken
    // dynamic backward path. Kernel-level gradient correctness is covered by
    // the finite-difference checks in `src/typed/ops/conv.rs`.
    let typed_grads = typed_pool.sum_all()?.backward()?;
    let dynamic_grads = dynamic_pool.sum_all()?.backward()?;
    assert_eq!(
        typed_grads.wrt_typed_input(&input)?.to_vec()?,
        dynamic_grads.wrt_input(&dynamic_input)?.to_vec::<f32>()?
    );
    assert_eq!(
        typed_grads.wrt_typed_input(&weight)?.to_vec()?,
        dynamic_grads.wrt_input(&dynamic_weight)?.to_vec::<f32>()?
    );
    Ok(())
}

#[test]
fn typed_cnn_learns_and_resumes_bit_exactly_across_a_tail_batch() -> Result<()> {
    const EPOCHS: u64 = 8;

    let ctx = DeviceCtx::cpu()?;
    let train = generated_split(TRAIN_ITEMS, 11, &ctx)?;
    let held_out = generated_split(EVAL_ITEMS, 977, &ctx)?;
    let train_loader = DataLoader::new(&train, BATCH_SIZE).shuffle(23);
    let eval_loader = DataLoader::new(&held_out, BATCH_SIZE);

    let batch_sizes = train_loader
        .batches()
        .map(|batch| batch.map(|(images, _)| images.dims()[0]))
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(batch_sizes, [8, 8, 8, 8, 8, 8, 3]);

    let mut model = Cnn::new(&ctx, 1)?;
    let mut optimizer = Adam::new(2e-2);
    let initial_state = state_dict(&model)?;
    assert_paths(&initial_state);
    assert_eq!(initial_state.len(), 7);
    let (initial_loss, initial_accuracy) = metrics(&mut model, &eval_loader)?;

    let mut epoch_losses = Vec::new();
    for epoch in 0..EPOCHS {
        epoch_losses.push(train_epoch(
            &mut model,
            &mut optimizer,
            &train_loader,
            epoch,
        )?);
    }
    let (final_loss, final_accuracy) = metrics(&mut model, &eval_loader)?;
    assert!(
        epoch_losses.last().unwrap() < &(0.25 * epoch_losses[0]),
        "training loss did not materially fall: {epoch_losses:?}"
    );
    // Anchor the untrained model near chance, then demand a near-perfect fit:
    // a merely "improving" run would otherwise hide a badly scaled gradient.
    assert!(
        initial_accuracy < 0.5,
        "untrained accuracy is not near chance: {initial_accuracy}"
    );
    assert!(
        final_loss < 0.02 * initial_loss,
        "held-out loss did not fall: {initial_loss} -> {final_loss}"
    );
    assert!(
        final_accuracy == 1.0 && final_accuracy > initial_accuracy,
        "held-out accuracy did not improve: {initial_accuracy} -> {final_accuracy}"
    );
    assert_eq!(
        optimizer.steps(),
        EPOCHS * train_loader.num_batches() as u64
    );
    assert_eq!(
        adam_param_steps(&optimizer, &mut model, "conv")?,
        optimizer.steps()
    );

    // A model forward exposes the same traced parameter leaf through typed and
    // dynamic gradient lookup, while BatchNorm contributes persistent buffers.
    // Both sides read one `Grads` for one leaf, so this pins the typed lookup
    // adapter only - it is not a gradient-value check.
    let (images, labels): (Images, Labels) = train.batch(&[0, 1, 2])?;
    let traced_conv = model.conv.get(Mode::TRAIN)?;
    let loss = model
        .forward(&images, Mode::TRAIN)?
        .cross_entropy(&labels)?;
    let grads = loss.backward()?;
    assert_eq!(grads.len(), 5);
    assert_eq!(
        model.conv.grad_from(&grads)?.to_vec()?,
        grads.wrt_input(traced_conv.as_dynamic())?.to_vec::<f32>()?
    );

    // BatchNorm mode contract: EVAL must consume the running buffers and leave
    // them alone, TRAIN must use this batch's own statistics and age them.
    let eval_logits = model.forward(&images, Mode::EVAL)?.to_vec()?;
    assert_eq!(
        eval_logits,
        model.forward(&images, Mode::EVAL)?.to_vec()?,
        "a repeated EVAL forward must be unchanged"
    );
    assert_ne!(
        model.forward(&images, Mode::TRAIN)?.to_vec()?,
        eval_logits,
        "TRAIN must normalise with batch statistics, not the running buffers"
    );
    assert_ne!(
        model.forward(&images, Mode::EVAL)?.to_vec()?,
        eval_logits,
        "the preceding TRAIN forward must have aged the buffers EVAL consumes"
    );

    let saved = checkpoint_path("saved");
    let reference_after = checkpoint_path("reference-after");
    let resumed_after = checkpoint_path("resumed-after");
    save_adam_checkpoint(&optimizer, &mut model, &saved, &Limits::default())?;

    let mut resumed = Cnn::new(&ctx, 999)?;
    let mut resumed_optimizer = Adam::new(2e-2);
    load_adam_checkpoint(
        &mut resumed_optimizer,
        &mut resumed,
        &saved,
        &LoadOptions::default(),
    )?;
    assert_paths(&state_dict(&resumed)?);
    let probe = train.batch(&[4, 5, 6, 7, 8])?;
    let reference_logits = model.forward(&probe.0, Mode::EVAL)?.to_vec()?;
    assert_eq!(
        reference_logits,
        resumed.forward(&probe.0, Mode::EVAL)?.to_vec()?
    );
    assert_eq!(optimizer.steps(), resumed_optimizer.steps());

    let reference_loss = model
        .forward(&probe.0, Mode::TRAIN)?
        .cross_entropy(&probe.1)?;
    let resumed_loss = resumed
        .forward(&probe.0, Mode::TRAIN)?
        .cross_entropy(&probe.1)?;
    assert_eq!(reference_loss.item()?, resumed_loss.item()?);
    adam_step(&mut optimizer, &mut model, reference_loss.backward()?)?;
    adam_step(
        &mut resumed_optimizer,
        &mut resumed,
        resumed_loss.backward()?,
    )?;
    save_adam_checkpoint(&optimizer, &mut model, &reference_after, &Limits::default())?;
    save_adam_checkpoint(
        &resumed_optimizer,
        &mut resumed,
        &resumed_after,
        &Limits::default(),
    )?;
    let reference_envelope = Envelope::load(&reference_after, &Limits::default())?;
    let resumed_envelope = Envelope::load(&resumed_after, &Limits::default())?;
    assert_eq!(reference_envelope.tensors(), resumed_envelope.tensors());
    assert_eq!(
        reference_envelope.section("optimizer"),
        resumed_envelope.section("optimizer")
    );

    for path in [saved, reference_after, resumed_after] {
        let _ = std::fs::remove_file(path);
    }
    Ok(())
}
