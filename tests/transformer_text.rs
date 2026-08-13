//! Does the language model actually *learn* real text?
//!
//! The sibling of `tests/resnet_mnist.rs`: that file asks the question for
//! convolutions and pixels, this one asks it for attention and tokens. Both
//! train a real network end to end and assert the loss falls, because a model
//! that builds, runs and produces plausible-looking numbers while learning
//! nothing is the failure mode worth catching.
//!
//! This is also the only place [`BpeTokenizer`] is driven the way a user would
//! drive it: trained on a corpus, used to encode that corpus into windows, and
//! used again to decode what the model generates.
//!
//! # What runs when
//!
//! - [`transformer_learns_a_repeating_corpus`] runs on every `cargo test`. It
//!   trains a tiny model on a few kilobytes of text in well under a second.
//! - [`transformer_learns_tiny_shakespeare`] is `#[ignore]`d because it needs
//!   the network and the `hub` feature, not because it is slow: it downloads
//!   the real corpus and takes about twelve seconds at the defaults.
//!
//! ```text
//! cargo test --release --features hub,rayon --test transformer_text \
//!     -- --ignored --nocapture
//! ```
//!
//! `--release` is not optional in practice. Note also that
//! [`BpeTokenizer::train`] rescans the whole symbol sequence once per merge,
//! so the vocabulary size and the corpus slice are chosen together to keep
//! tokenizer training under a second.
//!
//! The ignored test reads `RSTORCH_LM_STEPS` (default 600),
//! `RSTORCH_LM_VOCAB` (default 512) and `RSTORCH_LM_CORPUS_BYTES`
//! (default 131072, the slice BPE is trained on). At those defaults it lands
//! around 4.05 nats/token on held-out text against a 6.24 uniform baseline —
//! not a converged model, only a decisive one.

use rstorch::prelude::*;
use rstorch::text::{BpeTokenizer, Tokenizer};

/// The window length every test here trains on.
const SEQ_LEN: usize = 16;

fn config(vocab_size: usize, embed_dim: usize, num_layers: usize) -> TransformerConfig {
    TransformerConfig {
        vocab_size,
        max_seq_len: SEQ_LEN,
        embed_dim,
        num_heads: 2,
        num_layers,
        feed_forward_dim: embed_dim * 2,
    }
}

/// Next-token cross-entropy over a `[batch, sequence, vocab]` logit tensor.
fn causal_loss(logits: Tensor, targets: &Tensor) -> Result<Tensor> {
    let vocab = logits.dims()[2];
    logits
        .reshape([targets.num_elements(), vocab])?
        .cross_entropy(&targets.reshape([targets.num_elements()])?)
}

/// Encode `text` and cut it into as many non-overlapping `SEQ_LEN` windows as
/// it yields (at most `max`), each paired with itself shifted one token right.
///
/// Returns `(inputs, targets)`, both `[windows, SEQ_LEN]` and `I64` — the
/// shape [`DecoderTransformer::logits`] takes.
fn windows(
    tokenizer: &BpeTokenizer,
    text: &str,
    max: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let ids = tokenizer.encode(text, false)?;
    let count = max.min(ids.len().saturating_sub(1) / SEQ_LEN);
    assert!(
        count > 0,
        "corpus encodes to {} tokens, too few for one window of {SEQ_LEN}",
        ids.len()
    );

    let take = |offset: usize| -> Vec<i64> {
        (0..count * SEQ_LEN)
            .map(|i| ids[offset + i] as i64)
            .collect()
    };
    let dims = [count, SEQ_LEN];
    Ok((
        Tensor::from_vec(take(0), dims, device)?,
        Tensor::from_vec(take(1), dims, device)?,
    ))
}

/// One training step; returns the loss it was taken at.
fn train_step(
    model: &mut DecoderTransformer,
    optimizer: &mut Adam,
    inputs: &Tensor,
    targets: &Tensor,
) -> Result<f64> {
    let loss = causal_loss(model.logits(inputs, Mode::TRAIN)?, targets)?;
    let value = loss.item()?;
    optimizer.step(model, loss.backward()?)?;
    Ok(value)
}

/// Mean next-token cross-entropy with no recording, in nats per token.
#[cfg(feature = "hub")]
fn eval_loss(model: &mut DecoderTransformer, batches: &[(Tensor, Tensor)]) -> Result<f64> {
    let mut total = 0.0;
    for (inputs, targets) in batches {
        total += causal_loss(model.logits(inputs, Mode::EVAL)?, targets)?.item()?;
    }
    Ok(total / batches.len() as f64)
}

/// Learning a corpus with strong repetition is the cheapest end-to-end proof
/// that BPE encoding, the embeddings, the causal mask, attention, the LM head
/// and the optimizer are all wired together — and it runs offline in under a
/// second.
#[test]
fn transformer_learns_a_repeating_corpus() -> Result<()> {
    let device = Device::Cpu;
    let corpus = "To be, or not to be, that is the question.\n".repeat(48);

    // 260 is the floor (4 specials + 256 bytes); 24 merges is enough for BPE
    // to discover the repeated words without making training slow.
    let tokenizer = BpeTokenizer::train(&corpus, 284)?;
    assert!(
        !tokenizer.merges().is_empty(),
        "a corpus this repetitive must yield merges"
    );
    assert_eq!(
        tokenizer.decode(&tokenizer.encode("not to be", false)?)?,
        "not to be",
        "BPE must round-trip the text it was trained on"
    );

    let (inputs, targets) = windows(&tokenizer, &corpus, 4, &device)?;
    let vocab = tokenizer.vocab_size();
    let mut model = DecoderTransformer::new(config(vocab, 32, 1), &device, &mut Rng::seed(7))?;
    let mut optimizer = Adam::new(0.02);

    let mut losses = Vec::new();
    for _ in 0..15 {
        losses.push(train_step(&mut model, &mut optimizer, &inputs, &targets)?);
    }

    assert!(
        losses.iter().all(|loss| loss.is_finite()),
        "non-finite loss in {losses:?}"
    );
    let (first, last) = (losses[0], losses[losses.len() - 1]);
    assert!(
        last < first * 0.5,
        "15 steps on a repeating corpus did not halve the loss: {first:.4} -> {last:.4}"
    );

    // Generation runs off the trained weights and decodes back to text.
    let prompt = inputs.narrow(0, 0, 1)?.narrow(1, 0, 4)?;
    let generated = model.generate(&prompt, 4)?;
    assert_eq!(generated.dims(), &[1, 8]);
    let ids: Vec<usize> = generated
        .to_vec::<i64>()?
        .into_iter()
        .map(|id| id as usize)
        .collect();
    tokenizer.decode(&ids)?;
    Ok(())
}

/// Trains on real `TinyShakespeare` with a real BPE vocabulary and asserts the
/// model beats the only baseline that needs no learning at all: predicting
/// every token uniformly, which costs `ln(vocab)` nats.
#[cfg(feature = "hub")]
#[test]
#[ignore = "downloads a corpus: run with --release --features hub --ignored --nocapture"]
fn transformer_learns_tiny_shakespeare() -> Result<()> {
    use rstorch::data::hub::{DatasetHub, TinyShakespeare};
    use std::time::Instant;

    let steps = env_usize("RSTORCH_LM_STEPS", 600).max(4);
    let vocab_target = env_usize("RSTORCH_LM_VOCAB", 512).max(260);
    let corpus_bytes = env_usize("RSTORCH_LM_CORPUS_BYTES", 128 * 1024);
    let (batch, eval_batches) = (16, 4);

    let text = TinyShakespeare::load_text(&DatasetHub::default_cache())?;
    let slice = char_boundary_prefix(&text, corpus_bytes);

    let start = Instant::now();
    let tokenizer = BpeTokenizer::train(slice, vocab_target)?;
    let vocab = tokenizer.vocab_size();
    println!(
        "\ntransformer/tiny_shakespeare: BPE vocab {vocab} ({} merges) trained on {} KiB in {:.1}s",
        tokenizer.merges().len(),
        slice.len() / 1024,
        start.elapsed().as_secs_f64(),
    );

    // Held-out text comes after the slice BPE was trained on, so the eval
    // windows are text the tokenizer's merges were not fitted to either.
    let held_out = char_boundary_prefix(&text[slice.len()..], corpus_bytes / 4);
    let uniform_baseline = (vocab as f64).ln();

    for device in devices() {
        let (inputs, targets) = windows(&tokenizer, slice, usize::MAX, &device)?;
        let eval = split_batches(
            &windows(&tokenizer, held_out, eval_batches * batch, &device)?,
            batch,
            eval_batches,
        )?;
        // `steps` batches cycled over however many the corpus yields, so the
        // step count is a training budget rather than a corpus-size limit.
        let batches = inputs.dims()[0] / batch;

        let mut model =
            DecoderTransformer::new(config(vocab, 64, 2), &device, &mut Rng::seed(1234))?;
        let mut optimizer = Adam::new(3e-3);

        let start = Instant::now();
        let mut losses = Vec::with_capacity(steps);
        for step in 0..steps {
            let offset = (step % batches) * batch;
            let batch_inputs = inputs.narrow(0, offset, batch)?;
            let batch_targets = targets.narrow(0, offset, batch)?;
            losses.push(train_step(
                &mut model,
                &mut optimizer,
                &batch_inputs,
                &batch_targets,
            )?);
        }
        let elapsed = start.elapsed();
        let held_out_loss = eval_loss(&mut model, &eval)?;

        // The tail rather than the last single step: one batch can be unlucky.
        let tail = (losses.len() / 4).max(1);
        let first = mean(&losses[..tail]);
        let last = mean(&losses[losses.len() - tail..]);
        println!(
            "  {device:<10} {:>7.1} ms/step  {:>9.0} tok/s  {batches} batches × {:.1} epochs  \
             train {first:.3} -> {last:.3}  held-out {held_out_loss:.3} nats/token \
             (uniform {uniform_baseline:.3})",
            elapsed.as_secs_f64() * 1e3 / steps as f64,
            (steps * batch * SEQ_LEN) as f64 / elapsed.as_secs_f64(),
            steps as f64 / batches as f64,
        );

        assert!(
            losses.iter().all(|loss| loss.is_finite()),
            "{device} produced a non-finite loss: {losses:?}"
        );
        assert!(
            last < 0.8 * first,
            "{device} did not train: mean loss went {first:.3} -> {last:.3} over {steps} steps"
        );
        assert!(
            held_out_loss < 0.75 * uniform_baseline,
            "{device} reached {held_out_loss:.3} nats/token on held-out text after {steps} steps; \
             predicting uniformly over {vocab} tokens already costs {uniform_baseline:.3}"
        );

        // A trained model should still generate decodable text.
        let prompt = inputs.narrow(0, 0, 1)?.narrow(1, 0, 8)?;
        let ids: Vec<usize> = model
            .generate(&prompt, 8)?
            .to_vec::<i64>()?
            .into_iter()
            .map(|id| id as usize)
            .collect();
        println!("  {device:<10} sample: {:?}", tokenizer.decode(&ids)?);
    }
    println!();
    Ok(())
}

/// The longest prefix of `text` that is at most `bytes` long and ends on a
/// `char` boundary, so slicing it can never panic mid-codepoint.
#[cfg(feature = "hub")]
fn char_boundary_prefix(text: &str, bytes: usize) -> &str {
    let mut end = bytes.min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// Cut one `[count * batch, SEQ_LEN]` pair into `count` batches of `batch`.
#[cfg(feature = "hub")]
fn split_batches(
    (inputs, targets): &(Tensor, Tensor),
    batch: usize,
    count: usize,
) -> Result<Vec<(Tensor, Tensor)>> {
    (0..count)
        .map(|i| {
            Ok((
                inputs.narrow(0, i * batch, batch)?,
                targets.narrow(0, i * batch, batch)?,
            ))
        })
        .collect()
}

#[cfg(feature = "hub")]
fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

#[cfg(feature = "hub")]
fn env_usize(key: &str, default: usize) -> usize {
    match std::env::var(key) {
        Ok(raw) => raw
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("{key} must be a non-negative integer, got {raw:?}")),
        Err(_) => default,
    }
}

#[cfg(feature = "hub")]
fn devices() -> Vec<Device> {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        vec![Device::Cpu, Device::Metal(0)]
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        vec![Device::Cpu]
    }
}
