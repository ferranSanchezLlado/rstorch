//! A whole training loop: model, data, optimizer, loss going down.
//!
//! The task is synthetic so the example needs no download and runs in a
//! second: two interleaved spirals, which a linear model cannot separate and a
//! one-hidden-layer MLP can. Run it with `cargo run --example mlp`.

use rstorch::prelude::*;

const CLASSES: usize = 2;
const POINTS_PER_CLASS: usize = 300;
const EPOCHS: usize = 60;

/// Two spirals, one per class, as an `(inputs, labels)` dataset. Everything is
/// drawn from the seeded [`Rng`], so a run is reproducible.
fn spirals(device: &Device, rng: &mut Rng) -> Result<TensorDataset> {
    let mut xs = Vec::with_capacity(POINTS_PER_CLASS * CLASSES * 2);
    let mut ys = Vec::with_capacity(POINTS_PER_CLASS * CLASSES);
    for class in 0..CLASSES {
        for i in 0..POINTS_PER_CLASS {
            let radius = i as f64 / POINTS_PER_CLASS as f64;
            let angle = radius * 6.0 + class as f64 * std::f64::consts::PI + rng.normal(0.0, 0.1);
            xs.push((radius * angle.sin()) as f32);
            xs.push((radius * angle.cos()) as f32);
            ys.push(class as i64);
        }
    }
    let n = ys.len();
    TensorDataset::new(
        Tensor::from_vec(xs, [n, 2], device)?,
        Tensor::from_vec(ys, [n], device)?,
    )
}

/// `#[derive(Module)]` writes both parameter walks. Every field that owns
/// parameters must appear here — a layer left out of the struct is a compile
/// error rather than a layer that silently never trains.
#[derive(Module)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Mlp {
    fn new(device: &Device, rng: &mut Rng) -> Result<Mlp> {
        Ok(Mlp {
            fc1: Linear::new(2, 64, device, rng)?,
            fc2: Linear::new(64, CLASSES, device, rng)?,
        })
    }
}

impl Forward for Mlp {
    type Output = Tensor;

    fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        let hidden = self.fc1.forward(x, mode)?.relu()?;
        self.fc2.forward(&hidden, mode)
    }
}

fn accuracy(model: &mut Mlp, data: &TensorDataset) -> Result<f64> {
    let loader = DataLoader::new(data, 128);
    let (mut right, mut seen) = (0usize, 0usize);
    for batch in loader.batches() {
        let (x, y) = batch?;
        let predicted = model.forward(&x, Mode::EVAL)?.argmax(-1)?.to_vec::<i64>()?;
        for (p, t) in predicted.iter().zip(y.to_vec::<i64>()?) {
            right += usize::from(*p == t);
            seen += 1;
        }
    }
    Ok(right as f64 / seen as f64)
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let mut rng = Rng::seed(7);

    let data = spirals(&device, &mut rng)?;
    let mut model = Mlp::new(&device, &mut rng)?;
    let mut optimizer = Adam::new(5e-3);
    // The points are generated one class at a time, so shuffling is what keeps
    // a batch from being all one label. The seed makes the run reproducible.
    let loader = DataLoader::new(&data, 32).shuffle(0);

    println!(
        "accuracy before training: {:.3}",
        accuracy(&mut model, &data)?
    );

    for epoch in 0..EPOCHS {
        let mut total = 0.0;
        let mut batches = 0;
        // A fresh permutation per epoch, derived from the epoch number rather
        // than from wall-clock state, so the whole run stays reproducible.
        for batch in loader.batches_for_epoch(epoch as u64) {
            let (x, y) = batch?;
            let loss = model.forward(&x, Mode::TRAIN)?.cross_entropy(&y)?;
            total += loss.item()?;
            batches += 1;
            // `backward` hands over the one `Grads` for this step and `step`
            // consumes it. There is no `zero_grad`: gradients are never
            // accumulated in the parameters, so there is nothing to clear.
            optimizer.step(&mut model, loss.backward()?)?;
        }
        if epoch % 5 == 0 || epoch == EPOCHS - 1 {
            println!("epoch {epoch:>2}  loss {:.4}", total / batches as f64);
        }
    }

    println!(
        "accuracy after training:  {:.3}",
        accuracy(&mut model, &data)?
    );
    Ok(())
}
