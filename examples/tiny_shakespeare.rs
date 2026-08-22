//! Train a character-level decoder transformer on the real Tiny Shakespeare
//! corpus, measure held-out next-character loss, and generate a short sample.
//!
//! The first run downloads the corpus into rstorch's dataset cache. Release
//! mode is strongly recommended:
//!
//! ```text
//! cargo run --release --example tiny_shakespeare --features hub
//! ```
//!
//! `RSTORCH_LM_STEPS` (default 200) and `RSTORCH_LM_BATCH` (default 16) can be
//! used to change the training budget.

use rstorch::data::hub::{DatasetHub, TinyShakespeareDataset};
use rstorch::optim::schedule;
use rstorch::prelude::*;

const SEQ_LEN: usize = 64;
const LEARNING_RATE: f64 = 3e-3;
const MIN_LEARNING_RATE: f64 = 3e-4;

fn causal_loss(logits: Tensor, targets: &Tensor) -> Result<Tensor> {
    let vocab_size = logits.dims()[2];
    logits
        .reshape([targets.num_elements(), vocab_size])?
        .cross_entropy(&targets.reshape([targets.num_elements()])?)
}

fn sample_indices(rng: &mut Rng, start: usize, end: usize, count: usize) -> Vec<usize> {
    (0..count)
        .map(|_| start + rng.uniform(0.0, (end - start) as f64) as usize)
        .collect()
}

fn batch_loss(
    model: &mut DecoderTransformer,
    dataset: &TinyShakespeareDataset,
    indices: &[usize],
) -> Result<f64> {
    let (inputs, targets) = dataset.batch(indices)?;
    causal_loss(model.logits(&inputs, Mode::EVAL)?, &targets)?.item()
}

fn main() -> Result<()> {
    let steps = env_usize("RSTORCH_LM_STEPS", 200).max(1);
    let batch_size = env_usize("RSTORCH_LM_BATCH", 16).max(1);
    let device = Device::best_available();

    println!("loading Tiny Shakespeare on {device} (downloading it on the first run)...");
    let dataset = TinyShakespeareDataset::load(&DatasetHub::default_cache(), SEQ_LEN, &device)?;
    let train_end = dataset.len() * 9 / 10;
    let validation_start = train_end + SEQ_LEN;
    let validation_indices: Vec<usize> = (0..batch_size)
        .map(|i| validation_start + i * (dataset.len() - validation_start) / batch_size)
        .collect();

    let config =
        TransformerConfig::new(dataset.vocab_size(), SEQ_LEN, 32, 4, 1).with_feed_forward_dim(64);
    let mut rng = Rng::seed(42);
    let mut model = DecoderTransformer::new(config, &device, &mut rng)?;
    // Weight decay belongs on learned matrices, not additive biases or
    // normalization parameters. Dotted module paths make that policy explicit.
    let mut optimizer = AdamW::new(LEARNING_RATE, 0.1).group(
        |path| path.ends_with("bias") || path.contains("norm"),
        |group| group.weight_decay(0.0),
    );
    let warmup_steps = (steps as u64 / 20).max(1);
    let initial_validation = batch_loss(&mut model, &dataset, &validation_indices)?;

    println!(
        "{} characters, {}-character vocabulary, validation loss {initial_validation:.3}",
        dataset.num_tokens(),
        dataset.vocab_size(),
    );
    for step in 0..steps {
        let indices = sample_indices(&mut rng, 0, train_end, batch_size);
        let (inputs, targets) = dataset.batch(&indices)?;
        let loss = causal_loss(model.logits(&inputs, Mode::TRAIN)?, &targets)?;
        let value = loss.item()?;
        let completed = optimizer.steps();
        optimizer.set_lr(schedule::warmup(
            schedule::cosine(LEARNING_RATE, MIN_LEARNING_RATE, steps as u64, completed),
            warmup_steps,
            completed,
        ));
        // Gradient transforms consume and return the one linear `Grads`; the
        // optimizer then consumes it, so stale gradients cannot be reused.
        let grads = loss.backward()?.clip_norm(1.0)?;
        optimizer.step(&mut model, grads)?;

        if step == 0 || (step + 1) % 25 == 0 || step + 1 == steps {
            println!(
                "step {:>4}/{steps}  train loss {value:.3}  lr {:.2e}",
                step + 1,
                optimizer.lr(),
            );
        }
    }

    let final_validation = batch_loss(&mut model, &dataset, &validation_indices)?;
    let (prompt, _) = dataset.batch(&[validation_start])?;
    let prompt = prompt.narrow(1, 0, 16)?;
    let (generated, cache) = model.generate_with_cache(&prompt, SEQ_LEN - 16)?;
    let ids: Vec<usize> = generated
        .to_vec::<i64>()?
        .into_iter()
        .map(|id| id as usize)
        .collect();

    println!(
        "validation loss: {initial_validation:.3} -> {final_validation:.3} ({} cached positions)",
        cache.len(),
    );
    println!("\nsample:\n{}", dataset.tokenizer().decode(&ids)?);
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
