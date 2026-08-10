//! **The m3 acceptance fixture**:
//! the flagship loop — an MLP over an MNIST split, trained end to end through
//! `DataLoader` + `Adam`, with the loss going down and the accuracy going up.
//!
//! Like every fixture this file is a downstream consumer: it imports the
//! first-hour prelude plus the two `data::hub` paths the dataset lives behind,
//! and nothing else. Every step below therefore has to be expressible in the
//! public vocabulary. What it pins:
//!
//! - The `#[derive(Module)]` + `Forward` model —
//!   `Linear`/`Dropout`/`Linear` with a `relu` between — reaches
//!   `Adam::step(&mut model, loss.backward()?)` with no glue types.
//! - `MnistDataset` with [`MnistLayout::Flat`] drops straight into
//!   `DataLoader::new(&ds, 32).shuffle(0)`, and its `(F32, I64)` batch is
//!   exactly what `cross_entropy` takes — no conversion, no collator.
//! - **Real learning, not "it ran"**: the epoch loss falls monotonically from
//!   near `ln 10` to a fraction of it, and held-out accuracy goes from chance
//!   to a floor no random model reaches.
//! - **Determinism**: everything is seeded (data, init, dropout, shuffling), so
//!   two runs produce bit-identical loss sequences. A flaky training test is
//!   worse than none. This is *structural*, not luck: the CPU kernels partition
//!   work by **output** element/chunk, so no float is accumulated across
//!   threads in a racing order. Verified by measurement — the 15-epoch loss
//!   sequence is bit-identical with the `rayon` feature off, and on at
//!   `RAYON_NUM_THREADS` of 1, 2 and 8.
//! - **The loudness net**: with every parameter read through
//!   `Param::get(mode)` under a recording `Mode`, every parameter gets a
//!   gradient and `MissingGrad` never fires — which is the *correct* outcome,
//!   asserted positively here (`Grads::len` equals the parameter count and the
//!   step succeeds). The last test then shows the net catching the one
//!   mistake it exists to catch.
//!
//! # Offline by construction
//!
//! CI has no network, so nothing here downloads: the split is a small
//! **synthetic** stand-in built from a 5×7 stroke font for the ten digits,
//! encoded into MNIST's own IDX bytes and parsed with
//! [`Mnist::from_idx_bytes`] — the same parser, wrapper, upload, and batching
//! path the real 28×28 archives take, only 10×10 and a few hundred items so
//! the whole file runs in seconds. It is honestly *not* MNIST: it is a
//! learnable ten-class image problem that exercises the identical code.
//!
//! Two consequences worth stating plainly rather than discovering later:
//!
//! - The problem is easy enough that the trained net **saturates** held-out
//!   accuracy at `1.000`. So the `> 0.9` floor proves learning happened but
//!   cannot measure how *well* — it has no resolution left. The loss-trend
//!   assertions are the sensitive ones; accuracy is the sanity check. What
//!   makes either meaningful is the asserted starting point: the untrained net
//!   scores `0.105` at a loss of `2.363`, so the floors are not reachable
//!   without training.
//! - No assertion here is a performance claim. T47 measured these CPU kernels
//!   at 4–15× slower than v2; this fixture would pass just as green once that
//!   is fixed. It gates *correctness of the assembled stack*, nothing else.

use rstorch::data::hub::{Mnist, MnistDataset, MnistLayout};
use rstorch::prelude::*;

// ---------------------------------------------------------------------------
// The offline split: a stroke font, jittered, in IDX bytes.
// ---------------------------------------------------------------------------

/// Image geometry of the synthetic split (the real dataset is 28×28).
const ROWS: usize = 10;
/// Image width; see [`ROWS`].
const COLS: usize = 10;
/// Pixels per image, and therefore the MLP's input width under
/// [`MnistLayout::Flat`].
const PIXELS: usize = ROWS * COLS;
/// The ten digit classes.
const CLASSES: usize = 10;

/// A 5×7 stroke font for the digits `0`–`9`: `#` is ink, `.` is paper.
/// Formatting is skipped so the glyphs stay legible as glyphs.
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

/// A uniform pick from `0..n` off the public [`Rng`] — the fixture's only
/// source of randomness, so the split is a pure function of its seed.
fn pick(rng: &mut Rng, n: usize) -> usize {
    (rng.uniform(0.0, n as f64) as usize).min(n - 1)
}

/// Render one class as a `ROWS * COLS` row-major `u8` image: the glyph is
/// placed at a jittered offset, inked at a random intensity, and every pixel
/// gets a little noise. The jitter and the noise are what keep the problem
/// from being a lookup table.
fn render(class: usize, rng: &mut Rng) -> Vec<u8> {
    let glyph = &GLYPHS[class];
    let row_offset = 1 + pick(rng, 2); // rows 1..=2, so 7 glyph rows fit in 10
    let col_offset = 2 + pick(rng, 2); // cols 2..=3, so 5 glyph cols fit in 10
    let ink = 255.0 * rng.uniform(0.65, 1.0);

    let mut image = vec![0u8; PIXELS];
    for (r, row) in glyph.iter().enumerate() {
        for (c, byte) in row.bytes().enumerate() {
            if byte == b'#' {
                image[(row_offset + r) * COLS + col_offset + c] = ink as u8;
            }
        }
    }
    for pixel in &mut image {
        let noisy = f64::from(*pixel) + rng.uniform(-20.0, 20.0);
        *pixel = noisy.clamp(0.0, 255.0) as u8;
    }
    image
}

/// The IDX3 (image) file format the real `train-images-idx3-ubyte` uses:
/// magic `2051`, then count/rows/cols as big-endian `u32`, then the pixels.
fn idx_images(count: usize, pixels: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16 + pixels.len());
    bytes.extend(2051u32.to_be_bytes());
    bytes.extend(u32::try_from(count).unwrap().to_be_bytes());
    bytes.extend(u32::try_from(ROWS).unwrap().to_be_bytes());
    bytes.extend(u32::try_from(COLS).unwrap().to_be_bytes());
    bytes.extend_from_slice(pixels);
    bytes
}

/// The IDX1 (label) file format: magic `2049`, count, then one byte per label.
fn idx_labels(labels: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(8 + labels.len());
    bytes.extend(2049u32.to_be_bytes());
    bytes.extend(u32::try_from(labels.len()).unwrap().to_be_bytes());
    bytes.extend_from_slice(labels);
    bytes
}

/// A balanced split of `per_class * CLASSES` items, class-interleaved so even
/// an unshuffled loader sees every class in every batch.
fn synthetic_split(per_class: usize, seed: u64) -> Result<Mnist> {
    let mut rng = Rng::seed(seed);
    let mut pixels = Vec::with_capacity(per_class * CLASSES * PIXELS);
    let mut labels = Vec::with_capacity(per_class * CLASSES);
    for _ in 0..per_class {
        for class in 0..CLASSES {
            pixels.extend(render(class, &mut rng));
            labels.push(class as u8);
        }
    }
    Mnist::from_idx_bytes(&idx_images(labels.len(), &pixels), &idx_labels(&labels))
}

/// Both halves of the problem, uploaded to `device` in the flat layout an MLP
/// eats: a training split and a *disjointly seeded* held-out split.
fn splits(device: &Device) -> Result<(MnistDataset, MnistDataset)> {
    let train = MnistDataset::new(&synthetic_split(60, 11)?, MnistLayout::Flat, device)?;
    let held_out = MnistDataset::new(&synthetic_split(20, 977)?, MnistLayout::Flat, device)?;
    Ok((train, held_out))
}

// ---------------------------------------------------------------------------
// The model, verbatim in shape.
// ---------------------------------------------------------------------------

/// The flagship MLP: `784 → 128 → 10` in the design's example, `100 → 64 → 10`
/// here so the fixture runs in seconds. `#[derive(Module)]` writes both
/// parameter walks; `Dropout` contributes no parameters but is still a visited
/// child, and leaving it out of the struct would be a compile error rather
/// than a silently untrained layer.
#[derive(Module)]
struct Mlp {
    fc1: Linear,
    drop: Dropout,
    fc2: Linear,
}

impl Mlp {
    /// Kaiming-uniform weights and zero biases off `rng`; the dropout layer
    /// splits its own stream from the same generator.
    fn new(device: &Device, rng: &mut Rng) -> Result<Mlp> {
        Ok(Mlp {
            fc1: Linear::new(PIXELS, 64, device, rng)?,
            drop: Dropout::new(0.1, rng)?,
            fc2: Linear::new(64, CLASSES, device, rng)?,
        })
    }
}

impl Forward for Mlp {
    fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        let h = self.fc1.forward(x, mode)?.relu()?;
        let h = self.drop.forward(&h, mode)?;
        self.fc2.forward(&h, mode)
    }
}

/// The same net as a `Sequential` chain, to pin that the container route works
/// for a downstream consumer too (`Relu` is a unit struct, pushed by value).
fn sequential_mlp(device: &Device, rng: &mut Rng) -> Result<Sequential> {
    Ok(Sequential::new()
        .push(Linear::new(PIXELS, 64, device, rng)?)
        .push(Relu)
        .push(Linear::new(64, CLASSES, device, rng)?))
}

// ---------------------------------------------------------------------------
// The loop.
// ---------------------------------------------------------------------------

/// What every dataset in this file yields: `(F32 pixels, I64 labels)`.
type Batch = (Tensor, Tensor);

/// Train one epoch and return the mean cross-entropy over its batches.
///
/// `step` is a closure rather than an `&mut impl Optimizer` because there is
/// no `Optimizer` trait (every optimizer is a concrete
/// type), which is exactly what a generic training helper has to work around.
/// The body is otherwise the design's four lines: forward under
/// [`Mode::TRAIN`], `cross_entropy`, `backward`, `step` — no `zero_grad`,
/// because the gradients are a value that the step consumes.
fn train_epoch<M, D>(
    model: &mut M,
    loader: &DataLoader<D>,
    epoch: u64,
    step: &mut dyn FnMut(&mut M, Grads) -> Result<()>,
) -> Result<f64>
where
    M: Forward + Module,
    D: Dataset<Batch = Batch>,
{
    let mut total = 0.0;
    let mut batches = 0usize;
    for batch in loader.batches_for_epoch(epoch) {
        let (x, y) = batch?;
        let loss = model.forward(&x, Mode::TRAIN)?.cross_entropy(&y)?;
        total += loss.item()?;
        batches += 1;
        step(model, loss.backward()?)?;
    }
    assert!(batches > 0, "an epoch must yield at least one batch");
    Ok(total / batches as f64)
}

/// Top-1 accuracy over a whole loader in [`Mode::EVAL`] — dropout is the
/// identity there and nothing is recorded, so no activations are retained.
fn accuracy<M, D>(model: &mut M, loader: &DataLoader<D>) -> Result<f64>
where
    M: Forward,
    D: Dataset<Batch = Batch>,
{
    let mut correct = 0usize;
    let mut seen = 0usize;
    for batch in loader.batches() {
        let (x, y) = batch?;
        let predicted = model.forward(&x, Mode::EVAL)?.argmax(-1)?.to_vec::<i64>()?;
        let labels = y.to_vec::<i64>()?;
        assert_eq!(predicted.len(), labels.len());
        correct += predicted
            .iter()
            .zip(&labels)
            .filter(|(p, label)| p == label)
            .count();
        seen += labels.len();
    }
    Ok(correct as f64 / seen as f64)
}

/// Mean cross-entropy of a model over a loader, without training it.
fn mean_loss<M, D>(model: &mut M, loader: &DataLoader<D>) -> Result<f64>
where
    M: Forward,
    D: Dataset<Batch = Batch>,
{
    let mut total = 0.0;
    let mut batches = 0usize;
    for batch in loader.batches() {
        let (x, y) = batch?;
        total += model.forward(&x, Mode::EVAL)?.cross_entropy(&y)?.item()?;
        batches += 1;
    }
    Ok(total / batches as f64)
}

/// The loss a model that has learned nothing pays: `ln 10 ≈ 2.303`, the
/// cross-entropy of the uniform distribution over ten classes.
const CHANCE_LOSS: f64 = std::f64::consts::LN_10;
/// Top-1 accuracy of guessing: one class in ten.
const CHANCE_ACCURACY: f64 = 1.0 / CLASSES as f64;

// ---------------------------------------------------------------------------
// The acceptance tests.
// ---------------------------------------------------------------------------

/// How many epochs the flagship trains for — chosen for the *headroom* of the
/// strictest assertion below, not for the best final number.
///
/// Measured at this setting (unoptimized, ~3 s): training loss
/// `2.271 → 0.335`, monotone every epoch; held-out loss `2.363 → 0.317`;
/// held-out accuracy `0.105 → 1.000`. The floors are `0.25 × first`, `< 0.5`
/// and `> 0.9`, so every margin is wide — except one.
///
/// ⚠️ **The binding constraint is the per-epoch monotonicity**, whose headroom
/// is the *smallest* consecutive drop: `0.046` here. That number shrinks as the
/// curve flattens, and the drops are already non-smooth at the tail (`0.046`
/// then `0.060`), so monotonicity is noise-limited, not trend-limited. At 20
/// epochs the loss is a prettier `0.192` but the smallest drop falls to
/// `0.020` — the same assertion riding 2.3× less margin. Raising `EPOCHS`
/// therefore makes this test *more* fragile while making its headline number
/// look better; re-measure the smallest drop before touching it.
const EPOCHS: u64 = 15;

/// **The m3 example.** The flagship MLP trains: the epoch loss falls from
/// chance to a fraction of it, monotonically, and held-out accuracy clears a
/// floor a random model cannot reach.
#[test]
fn the_flagship_mlp_learns_from_an_mnist_split() -> Result<()> {
    let device = Device::best_available();
    let (train, held_out) = splits(&device)?;
    assert_eq!(train.len(), 600);
    assert_eq!(held_out.len(), 200);
    assert_eq!(train.inputs().dims(), &[600, PIXELS]); // Flat: one row per image
    assert_eq!(train.inputs().dtype(), DType::F32);
    assert_eq!(train.targets().dtype(), DType::I64);

    let mut rng = Rng::seed(42);
    let mut model = Mlp::new(&device, &mut rng)?;
    let mut opt = Adam::new(1e-3);
    let train_loader = DataLoader::new(&train, 32).shuffle(0);
    let eval_loader = DataLoader::new(&held_out, 64);

    // Where an untrained net starts: chance accuracy, chance loss.
    let start_accuracy = accuracy(&mut model, &eval_loader)?;
    let start_loss = mean_loss(&mut model, &eval_loader)?;
    assert!(
        start_accuracy < 0.35,
        "an untrained net should be near chance, got {start_accuracy}"
    );
    assert!(
        (start_loss - CHANCE_LOSS).abs() < 0.6,
        "initial loss should be near ln 10 = {CHANCE_LOSS}, got {start_loss}"
    );

    let mut losses = Vec::new();
    for epoch in 0..EPOCHS {
        losses.push(train_epoch(
            &mut model,
            &train_loader,
            epoch,
            &mut |m, g| opt.step(m, g),
        )?);
    }
    let final_accuracy = accuracy(&mut model, &eval_loader)?;
    let final_loss = mean_loss(&mut model, &eval_loader)?;

    // The loss DECREASES — every epoch, not just on average.
    for (epoch, pair) in losses.windows(2).enumerate() {
        assert!(
            pair[1] < pair[0],
            "epoch {} loss must fall: {:?}",
            epoch + 1,
            losses
        );
    }
    assert!(
        losses[0] > 1.0 && *losses.last().unwrap() < 0.25 * losses[0],
        "loss must fall from near chance to a fraction of it: {losses:?}"
    );
    assert!(
        final_loss < 0.5 && final_loss < start_loss,
        "held-out loss must fall too: {start_loss} -> {final_loss}"
    );

    // Accuracy clears a floor guessing cannot: 90% of ten classes.
    assert!(
        final_accuracy > 0.9,
        "held-out accuracy {final_accuracy} must beat the floor \
         (chance is {CHANCE_ACCURACY}, start was {start_accuracy})"
    );

    // The optimizer really did move: one update per batch per epoch.
    assert_eq!(opt.steps(), EPOCHS * train_loader.num_batches() as u64);
    Ok(())
}

/// The `Sequential` + `Sgd` route over the same split: no `#[derive(Module)]`
/// struct, no hand-written `Forward`, three `push` calls.
#[test]
fn a_sequential_stack_trains_with_sgd() -> Result<()> {
    let device = Device::best_available();
    let (train, held_out) = splits(&device)?;
    let mut rng = Rng::seed(7);
    let mut model = sequential_mlp(&device, &mut rng)?;
    assert_eq!(model.len(), 3);

    let mut opt = Sgd::new(0.5);
    let train_loader = DataLoader::new(&train, 32).shuffle(1);
    let eval_loader = DataLoader::new(&held_out, 64);

    let first = train_epoch(&mut model, &train_loader, 0, &mut |m, g| opt.step(m, g))?;
    let mut last = first;
    for epoch in 1..12 {
        last = train_epoch(&mut model, &train_loader, epoch, &mut |m, g| opt.step(m, g))?;
    }

    // Measured: 1.761 → 0.020, accuracy 1.000. The floors are far below that.
    assert!(
        last < 0.5 * first,
        "Sgd must reduce the loss: {first} -> {last}"
    );
    let final_accuracy = accuracy(&mut model, &eval_loader)?;
    assert!(
        final_accuracy > 0.9,
        "held-out accuracy {final_accuracy} must beat chance {CHANCE_ACCURACY}"
    );
    Ok(())
}

/// Nothing here is ambient, so two runs of the same seeds agree **bitwise**:
/// the split, the initialization, the dropout masks, and the shuffle order are
/// all pure functions of their seeds. A training test that could not assert
/// this would not be worth having.
#[test]
fn training_is_reproducible_from_its_seeds() -> Result<()> {
    let device = Device::best_available();

    let run = || -> Result<Vec<f64>> {
        let train = MnistDataset::new(&synthetic_split(20, 11)?, MnistLayout::Flat, &device)?;
        let mut rng = Rng::seed(42);
        let mut model = Mlp::new(&device, &mut rng)?;
        let mut opt = Adam::new(1e-3);
        let loader = DataLoader::new(&train, 32).shuffle(0);
        (0..3)
            .map(|epoch| train_epoch(&mut model, &loader, epoch, &mut |m, g| opt.step(m, g)))
            .collect()
    };

    let first = run()?;
    assert_eq!(first, run()?, "identical seeds must give identical losses");
    Ok(())
}

/// The loudness net, from the *quiet* side: every parameter of the flagship is
/// read through `Param::get(mode)` under a recording mode, so `backward`
/// produces one gradient per parameter and `Adam::step` never raises
/// [`Error::MissingGrad`]. That is the correct outcome, and asserting it
/// positively is what makes the next test's failure meaningful.
#[test]
fn every_parameter_of_the_flagship_is_traced() -> Result<()> {
    let device = Device::best_available();
    let (train, _) = splits(&device)?;
    let mut rng = Rng::seed(42);
    let mut model = Mlp::new(&device, &mut rng)?;

    // The four leaves, by the dotted paths the derive emits.
    let paths: Vec<String> = rstorch::nn::state_dict(&model).into_keys().collect();
    assert_eq!(paths, ["fc1.bias", "fc1.weight", "fc2.bias", "fc2.weight"]);
    assert_eq!(
        rstorch::nn::num_params(&model),
        PIXELS * 64 + 64 + 64 * 10 + 10
    );

    let (x, y) = train.batch(&[0, 1, 2, 3])?;
    let grads = model
        .forward(&x, Mode::TRAIN)?
        .cross_entropy(&y)?
        .backward()?;
    assert_eq!(grads.len(), paths.len(), "one gradient per parameter");

    // Consuming the gradients therefore succeeds: no MissingGrad, and the
    // pre-pass has nothing to reject.
    Adam::new(1e-3).step(&mut model, grads)?;
    Ok(())
}

/// The loudness net catching the mistake it exists for: a layer that reads
/// `Param::value()` (the untraced accessor) instead of `Param::get(mode)`
/// contributes to the forward pass but not to the graph, so the next step
/// names it — rather than training everything else and leaving it frozen
/// forever.
#[test]
fn an_untraced_parameter_is_loud_at_the_next_step() -> Result<()> {
    /// `fc` is traced correctly; `head` is not.
    #[derive(Module)]
    struct HalfTraced {
        fc: Linear,
        head: Param,
    }

    impl Forward for HalfTraced {
        fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
            let h = self.fc.forward(x, mode)?.relu()?;
            // The bug: `value()` is the raw tensor, outside the graph.
            h.matmul(&self.head.value().transpose(-2, -1)?)
        }
    }

    let device = Device::best_available();
    let (train, _) = splits(&device)?;
    let mut rng = Rng::seed(3);
    let mut model = HalfTraced {
        fc: Linear::new(PIXELS, 16, &device, &mut rng)?,
        head: Param::new(Tensor::zeros([CLASSES, 16], DType::F32, &device)?),
    };

    let (x, y) = train.batch(&[0, 1, 2, 3])?;
    let grads = model
        .forward(&x, Mode::TRAIN)?
        .cross_entropy(&y)?
        .backward()?;
    // The traced half did produce gradients; only `head` is missing.
    assert_eq!(grads.len(), 2);

    let rejected = Sgd::new(0.1).step(&mut model, grads);
    match rejected {
        Err(Error::MissingGrad { path }) => assert_eq!(path, "head"),
        other => panic!("expected MissingGrad naming `head`, got {other:?}"),
    }
    Ok(())
}
