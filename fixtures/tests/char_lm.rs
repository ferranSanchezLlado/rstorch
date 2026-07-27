//! Offline decoder-LM acceptance fixture over TinyShakespeare-style text.

use std::path::PathBuf;

use rstorch::data::hub::TinyShakespeareDataset;
use rstorch::persist::{Envelope, Limits};
use rstorch::prelude::*;

const CORPUS: &str = "To be, or not to be, that is the question.\n\
Whether 'tis nobler in the mind to suffer.\n\
To be, or not to be, that is the question.\n\
Whether 'tis nobler in the mind to suffer.\n\
To be, or not to be, that is the question.\n";

fn config(vocab_size: usize) -> TransformerConfig {
    TransformerConfig {
        vocab_size,
        max_seq_len: 16,
        embed_dim: 8,
        num_heads: 2,
        num_layers: 1,
        feed_forward_dim: 16,
    }
}

fn checkpoint_path() -> PathBuf {
    std::env::temp_dir().join(format!(
        "rstorch-char-lm-{}-{:?}.rstorch",
        std::process::id(),
        std::thread::current().id()
    ))
}

fn causal_loss(logits: Tensor, targets: &Tensor, vocab_size: usize) -> Result<Tensor> {
    logits
        .reshape([targets.num_elements(), vocab_size])?
        .cross_entropy(&targets.reshape([targets.num_elements()])?)
}

#[test]
fn seeded_char_lm_trains_offline_and_generates_with_a_kv_cache() -> Result<()> {
    let device = Device::Cpu;
    let dataset = TinyShakespeareDataset::from_text(CORPUS, 8, &device)?;
    assert_eq!(
        dataset
            .tokenizer()
            .decode(&dataset.tokenizer().encode("To be", false)?)?,
        "To be"
    );

    let (inputs, targets) = dataset.batch(&[0, 8, 24, 48])?;
    let mut model =
        DecoderTransformer::new(config(dataset.vocab_size()), &device, &mut Rng::seed(7))?;
    let mut optimizer = Adam::new(0.02);
    let first = causal_loss(
        model.logits(&inputs, Mode::EVAL)?,
        &targets,
        dataset.vocab_size(),
    )?
    .item()?;
    for _ in 0..20 {
        let loss = causal_loss(
            model.logits(&inputs, Mode::TRAIN)?,
            &targets,
            dataset.vocab_size(),
        )?;
        optimizer.step(&mut model, loss.backward()?)?;
    }
    let last = causal_loss(
        model.logits(&inputs, Mode::EVAL)?,
        &targets,
        dataset.vocab_size(),
    )?
    .item()?;
    assert!(
        last < first * 0.65,
        "causal-LM loss did not fall enough: {first} -> {last}"
    );

    let prompt = inputs.narrow(0, 0, 1)?.narrow(1, 0, 4)?;
    let (generated, cache) = model.generate_with_cache(&prompt, 4)?;
    assert_eq!(generated.dims(), &[1, 8]);
    assert_eq!(cache.len(), 8);
    let generated_ids: Vec<usize> = generated
        .to_vec::<i64>()?
        .into_iter()
        .map(|id| id as usize)
        .collect();
    assert_eq!(
        dataset.tokenizer().decode(&generated_ids)?.chars().count(),
        8
    );
    Ok(())
}

#[test]
fn checkpoint_alone_reconstructs_model_and_exact_logits() -> Result<()> {
    let device = Device::Cpu;
    let dataset = TinyShakespeareDataset::from_text(CORPUS, 8, &device)?;
    let (inputs, _) = dataset.batch(&[0, 4])?;
    let model = DecoderTransformer::new(config(dataset.vocab_size()), &device, &mut Rng::seed(91))?;
    let mut original = model;
    let expected = original.logits(&inputs, Mode::EVAL)?.to_vec::<f32>()?;

    let path = checkpoint_path();
    let limits = Limits::defaults();
    original.save_checkpoint(&path, &limits)?;
    let envelope = Envelope::load(&path, &limits)?;
    assert!(envelope.section("config").is_some());
    assert!(
        envelope
            .tensor("blocks.0.attention.q_proj.weight")
            .is_some()
    );

    let mut reconstructed = DecoderTransformer::load_checkpoint(&path, &device, &limits)?;
    assert_eq!(reconstructed.config(), original.config());
    assert_eq!(
        reconstructed.logits(&inputs, Mode::EVAL)?.to_vec::<f32>()?,
        expected
    );
    std::fs::remove_file(path)?;
    Ok(())
}
