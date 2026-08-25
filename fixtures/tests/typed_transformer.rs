#![cfg(feature = "typed")]

//! Public-only, offline typed decoder/character-LM acceptance fixture.
//!
//! This file deliberately contains no `compile_fail` doctests. Cargo only
//! harvests doctests from library targets, so a `compile_fail` block written
//! in an integration test is never compiled: it would report success forever,
//! including after the contract it claims to cover had been deleted. The
//! executed coverage for the compile-time contracts this fixture relies on
//! lives in the library's own doctests and in the typed UI suite:
//!
//! - head geometry, `HEADS` must divide `EMBED`: `src/typed/nn/attention.rs:310`
//! - attention float-mask dtype: `src/typed/nn/attention.rs:459`
//! - attention mask placement: `src/typed/nn/attention.rs:472`
//! - rank-eight head split: `src/typed/nn/attention.rs:490`
//! - attention input width vs `EMBED`: `tests/ui/typed/nn/attention_input_width.rs`
//! - embedding float indices: `src/typed/nn/embedding.rs:109`
//! - embedding index placement: `src/typed/nn/embedding.rs:119`
//! - rank-eight embedding indices: `src/typed/nn/embedding.rs:132`
//! - `DYN` embedding vocabulary: `src/typed/nn/embedding.rs:69`
//! - zero embedding width: `src/typed/nn/embedding.rs:77`

use std::path::PathBuf;

use half::{bf16, f16};
use rstorch::models::{DecoderTransformer, TransformerConfig};
use rstorch::nn::{self as dynamic_nn, ModuleExt};
use rstorch::persist::{Envelope, Limits, LoadOptions};
use rstorch::prelude::{DType, Device, Result, Rng, Tensor};
use rstorch::typed::nn::{
    Embedding, Forward, LayerNorm, Linear, Mode, MultiHeadAttention, ToDType, TypedModule,
    scaled_dot_product_attention,
};
use rstorch::typed::optim::{
    Adam, Sgd, adam_param_steps, adam_step, load_adam_checkpoint, save_adam_state, sgd_step,
};
use rstorch::typed::persist::save_model_state;
use rstorch::typed::{
    Cpu, DYN, DeviceCtx, FloatElement, NumericElement, Tensor1, Tensor2, Tensor3, Tensor4,
};

const VOCAB: usize = 7;
const MAX_SEQUENCE: usize = 12;
// head_dim is EMBED / HEADS = 3, so it collides with neither HEADS (2) nor any
// batch or sequence extent used below. A head/head-dim or batch/head axis
// transposition therefore cannot line up by coincidence anywhere here.
const EMBED: usize = 6;
const HEADS: usize = 2;
const FF: usize = 8;
const LAYERS: usize = 2;
const CONFIG: &str =
    "vocab_size=7;max_seq_len=12;embed_dim=6;num_heads=2;num_layers=2;feed_forward_dim=8";
const CORPUS: &[u8] = b"ab cab dab cab ab dab cab ab cab dab ";

type Tokens = Tensor2<DYN, DYN, i64>;
type Hidden = Tensor3<DYN, DYN, EMBED>;
type Context = Tensor4<DYN, HEADS, DYN, DYN>;
type Mask = Tensor4<DYN, HEADS, DYN, DYN, bool>;
type Logits = Tensor3<DYN, DYN, VOCAB>;

fn config() -> TransformerConfig {
    TransformerConfig::new(VOCAB, MAX_SEQUENCE, EMBED, HEADS, LAYERS).with_feed_forward_dim(FF)
}

fn checkpoint_path() -> PathBuf {
    std::env::temp_dir().join(format!(
        "rstorch-typed-transformer-{}-{:?}.rstorch",
        std::process::id(),
        std::thread::current().id()
    ))
}

fn token(byte: u8) -> i64 {
    match byte {
        b' ' => 0,
        b'a' => 1,
        b'b' => 2,
        b'c' => 3,
        b'd' => 4,
        _ => 5,
    }
}

fn batch(ctx: &DeviceCtx<Cpu>, offsets: &[usize], sequence: usize) -> Result<(Tokens, Tokens)> {
    let mut inputs = Vec::with_capacity(offsets.len() * sequence);
    let mut targets = Vec::with_capacity(offsets.len() * sequence);
    for &offset in offsets {
        for step in 0..sequence {
            inputs.push(token(CORPUS[(offset + step) % CORPUS.len()]));
            targets.push(token(CORPUS[(offset + step + 1) % CORPUS.len()]));
        }
    }
    Ok((
        Tokens::from_vec(inputs, [offsets.len(), sequence], ctx)?,
        Tokens::from_vec(targets, [offsets.len(), sequence], ctx)?,
    ))
}

#[derive(TypedModule)]
struct TypedBlock {
    norm1: LayerNorm<Tensor1<EMBED>>,
    attention: MultiHeadAttention<EMBED, HEADS>,
    norm2: LayerNorm<Tensor1<EMBED>>,
    feed_forward1: Linear<EMBED, FF>,
    feed_forward2: Linear<FF, EMBED>,
}

impl TypedBlock {
    fn new(ctx: &DeviceCtx<Cpu>, rng: &mut Rng) -> Result<Self> {
        Ok(Self {
            norm1: LayerNorm::new([EMBED], ctx)?,
            attention: MultiHeadAttention::new_without_bias(ctx, rng)?,
            norm2: LayerNorm::new([EMBED], ctx)?,
            feed_forward1: Linear::new(EMBED, FF, ctx, rng)?,
            feed_forward2: Linear::new(FF, EMBED, ctx, rng)?,
        })
    }

    fn forward(&mut self, input: &Hidden, mask: &Mask, mode: Mode) -> Result<Hidden> {
        let normalized = self.norm1.forward(input, mode)?;
        let attended = self.attention.attend(&normalized, Some(mask), mode)?;
        let residual = input.add(&attended)?;
        let normalized = self.norm2.forward(&residual, mode)?;
        let hidden = self.feed_forward1.forward(&normalized, mode)?.gelu()?;
        residual.add(&self.feed_forward2.forward(&hidden, mode)?)
    }

    fn forward_cached(
        &mut self,
        input: &Hidden,
        cache: &mut Option<(Context, Context)>,
        mode: Mode,
    ) -> Result<Hidden> {
        let normalized = self.norm1.forward(input, mode)?;
        let query = self.attention.project_query(&normalized, mode)?;
        let (new_keys, new_values) = self.attention.project_keys_values(&normalized, mode)?;
        let (keys, values) = match cache.take() {
            Some((old_keys, old_values)) => (
                Context::cat::<2>(&[&old_keys, &new_keys])?,
                Context::cat::<2>(&[&old_values, &new_values])?,
            ),
            None => (new_keys, new_values),
        };
        let context = scaled_dot_product_attention::<EMBED, HEADS, f32, Cpu, _>(
            &query, &keys, &values, None,
        )?;
        let attended = self.attention.project_output(&context, mode)?;
        *cache = Some((keys, values));
        let residual = input.add(&attended)?;
        let normalized = self.norm2.forward(&residual, mode)?;
        let hidden = self.feed_forward1.forward(&normalized, mode)?.gelu()?;
        residual.add(&self.feed_forward2.forward(&hidden, mode)?)
    }
}

#[derive(TypedModule)]
struct TypedDecoder {
    token_embedding: Embedding<VOCAB, EMBED>,
    position_embedding: Embedding<MAX_SEQUENCE, EMBED>,
    blocks: Vec<TypedBlock>,
    final_norm: LayerNorm<Tensor1<EMBED>>,
    output: Linear<EMBED, VOCAB>,
}

struct TypedCache {
    layers: Vec<Option<(Context, Context)>>,
    len: usize,
}

impl TypedDecoder {
    fn new(ctx: &DeviceCtx<Cpu>, rng: &mut Rng) -> Result<Self> {
        Ok(Self {
            token_embedding: Embedding::new(ctx, rng)?,
            position_embedding: Embedding::new(ctx, rng)?,
            blocks: (0..LAYERS)
                .map(|_| TypedBlock::new(ctx, rng))
                .collect::<Result<_>>()?,
            final_norm: LayerNorm::new([EMBED], ctx)?,
            output: Linear::new(EMBED, VOCAB, ctx, rng)?,
        })
    }

    fn embed(
        &self,
        ids: &Tokens,
        position: usize,
        mode: Mode,
        ctx: &DeviceCtx<Cpu>,
    ) -> Result<Hidden> {
        let [batch, sequence] = ids.dims();
        let positions = Tensor1::<DYN, i64>::arange(
            i64::try_from(position).unwrap(),
            i64::try_from(position + sequence).unwrap(),
            ctx,
        )?;
        let tokens = self.token_embedding.lookup(ids, mode)?;
        let positions = self.position_embedding.lookup(&positions, mode)?;
        Hidden::try_from_dynamic(tokens.as_dynamic().add(positions.as_dynamic())?, ctx)
            .inspect(|hidden| assert_eq!(hidden.dims(), [batch, sequence, EMBED]))
    }

    fn logits(&mut self, ids: &Tokens, mode: Mode, ctx: &DeviceCtx<Cpu>) -> Result<Logits> {
        let [batch, sequence] = ids.dims();
        let mut hidden = self.embed(ids, 0, mode, ctx)?;
        let causal = Tensor2::<DYN, DYN, bool>::causal_mask(sequence, ctx)?;
        let mask: Mask = causal.broadcast_to([batch, HEADS, sequence, sequence])?;
        for block in &mut self.blocks {
            hidden = block.forward(&hidden, &mask, mode)?;
        }
        self.output
            .forward(&self.final_norm.forward(&hidden, mode)?, mode)
    }

    fn empty_cache(&self) -> TypedCache {
        TypedCache {
            layers: (0..self.blocks.len()).map(|_| None).collect(),
            len: 0,
        }
    }

    fn logits_cached(
        &mut self,
        ids: &Tokens,
        cache: &mut TypedCache,
        ctx: &DeviceCtx<Cpu>,
    ) -> Result<Logits> {
        assert_eq!(ids.dims()[1], 1);
        assert!(cache.len < MAX_SEQUENCE);
        let mut hidden = self.embed(ids, cache.len, Mode::EVAL, ctx)?;
        for (block, layer_cache) in self.blocks.iter_mut().zip(&mut cache.layers) {
            hidden = block.forward_cached(&hidden, layer_cache, Mode::EVAL)?;
        }
        cache.len += 1;
        self.output
            .forward(&self.final_norm.forward(&hidden, Mode::EVAL)?, Mode::EVAL)
    }

    fn generate(
        &mut self,
        prompt: &Tokens,
        new_tokens: usize,
        ctx: &DeviceCtx<Cpu>,
    ) -> Result<(Tokens, TypedCache)> {
        assert!(prompt.dims()[1] + new_tokens <= MAX_SEQUENCE);
        let mut cache = self.empty_cache();
        let mut tokens = prompt.clone();
        let mut logits = None;
        for position in 0..prompt.dims()[1] {
            logits =
                Some(self.logits_cached(&prompt.narrow::<1>(position, 1)?, &mut cache, ctx)?);
        }
        for _ in 0..new_tokens {
            let next = logits.as_ref().unwrap().argmax::<2>()?;
            tokens = Tokens::cat::<1>(&[&tokens, &next])?;
            logits = Some(self.logits_cached(&next, &mut cache, ctx)?);
        }
        Ok((tokens, cache))
    }

    /// Greedy decoding written the obvious, cache-free way: every step
    /// recomputes the logits over the whole prefix. This is the reference
    /// `generate` is checked against, so the check binds both the greediness
    /// of the selection rule and the correctness of the KV cache.
    fn generate_uncached(
        &mut self,
        prompt: &Tokens,
        new_tokens: usize,
        ctx: &DeviceCtx<Cpu>,
    ) -> Result<Tokens> {
        let mut tokens = prompt.clone();
        for _ in 0..new_tokens {
            let length = tokens.dims()[1];
            let logits = self.logits(&tokens, Mode::EVAL, ctx)?;
            let next = logits.narrow::<1>(length - 1, 1)?.argmax::<2>()?;
            tokens = Tokens::cat::<1>(&[&tokens, &next])?;
        }
        Ok(tokens)
    }
}

fn loss(
    model: &mut TypedDecoder,
    inputs: &Tokens,
    targets: &Tokens,
    ctx: &DeviceCtx<Cpu>,
) -> Result<Tensor> {
    let logits = model.logits(inputs, Mode::TRAIN, ctx)?;
    let rows = targets.as_dynamic().num_elements();
    let logits: Tensor2<DYN, VOCAB> = logits.reshape([rows, VOCAB])?;
    let targets: Tensor1<DYN, i64> = targets.reshape([rows])?;
    Ok(logits.cross_entropy(&targets)?.as_dynamic().clone())
}

fn train_steps(
    model: &mut TypedDecoder,
    optimizer: &mut Adam,
    inputs: &Tokens,
    targets: &Tokens,
    ctx: &DeviceCtx<Cpu>,
    steps: usize,
) -> Result<()> {
    for _ in 0..steps {
        let grads = loss(model, inputs, targets, ctx)?.backward()?;
        adam_step(optimizer, model, grads)?;
    }
    Ok(())
}

fn model_envelope(model: &mut TypedDecoder) -> Result<Envelope> {
    let mut envelope = Envelope::new();
    save_model_state(model, &mut envelope)?;
    Ok(envelope)
}

fn save_training_checkpoint(
    model: &mut TypedDecoder,
    optimizer: &Adam,
    path: &PathBuf,
) -> Result<()> {
    let mut envelope = model_envelope(model)?;
    save_adam_state(optimizer, model, &mut envelope)?;
    envelope.set_section("config", CONFIG)?;
    envelope.save(path, &Limits::defaults())
}

#[test]
fn seeded_dynamic_and_typed_logits_gradients_paths_and_variable_shapes_agree() -> Result<()> {
    let ctx = DeviceCtx::cpu()?;
    let (small, small_targets) = batch(&ctx, &[0], 2)?;
    // Batch 4, sequence 5, HEADS 2, head_dim 3: no two axes share an extent.
    let (inputs, targets) = batch(&ctx, &[0, 5, 11, 19], 5)?;
    let mut typed = TypedDecoder::new(&ctx, &mut Rng::seed(17))?;
    let mut dynamic = DecoderTransformer::new(config(), &Device::Cpu, &mut Rng::seed(17))?;

    assert_eq!(
        typed.logits(&small, Mode::EVAL, &ctx)?.dims(),
        [1, 2, VOCAB]
    );
    let typed_logits = typed.logits(&inputs, Mode::EVAL, &ctx)?;
    let dynamic_logits = dynamic.logits(inputs.as_dynamic(), dynamic_nn::Mode::EVAL)?;
    // The typed decoder must dispatch to the same kernels in the same order as
    // the dynamic one, so this is bit-exact rather than merely close.
    assert_eq!(typed_logits.to_vec()?, dynamic_logits.to_vec::<f32>()?);

    let typed_paths = rstorch::typed::nn::state_dict(&typed)?
        .paths()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let dynamic_paths = dynamic
        .state_dict()
        .unwrap()
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(typed_paths, dynamic_paths);
    assert!(typed_paths.contains(&"blocks.1.attention.k_proj.weight".to_string()));

    let typed_grads = loss(&mut typed, &inputs, &targets, &ctx)?.backward()?;
    let dynamic_loss = dynamic
        .logits(inputs.as_dynamic(), dynamic_nn::Mode::TRAIN)?
        .reshape([targets.as_dynamic().num_elements(), VOCAB])?
        .cross_entropy(
            &targets
                .as_dynamic()
                .reshape([targets.as_dynamic().num_elements()])?,
        )?;
    let dynamic_grads = dynamic_loss.backward()?;
    sgd_step(&mut Sgd::new(0.01), &mut typed, typed_grads)?;
    Sgd::new(0.01).step(&mut dynamic, dynamic_grads)?;

    let typed_after = model_envelope(&mut typed)?;
    let dynamic_path = checkpoint_path();
    dynamic.save_checkpoint(&dynamic_path, &Limits::defaults())?;
    let dynamic_after = Envelope::load(&dynamic_path, &Limits::defaults())?;
    assert_eq!(typed_after.tensors(), dynamic_after.tensors());
    std::fs::remove_file(dynamic_path)?;

    assert!(
        loss(&mut typed, &small, &small_targets, &ctx)?
            .item()?
            .is_finite()
    );
    Ok(())
}

#[test]
fn typed_cache_generation_training_and_checkpoint_resume_are_exact() -> Result<()> {
    let ctx = DeviceCtx::cpu()?;
    // Batch 6, sequence 4, narrowed to batch 5: the cached path then runs with
    // batch 5, HEADS 2, sequence 4 and head_dim 3, all four extents distinct,
    // so no axis transposition can survive by coincidence.
    let (inputs, targets) = batch(&ctx, &[0, 4, 9, 15, 21, 27], 4)?;
    let narrowed = inputs.narrow::<0>(0, 5)?;
    let mut cached_model = TypedDecoder::new(&ctx, &mut Rng::seed(23))?;
    let full = cached_model.logits(&narrowed, Mode::EVAL, &ctx)?;
    assert_eq!(full.dims(), [5, 4, VOCAB]);
    let mut cache = cached_model.empty_cache();
    for position in 0..4 {
        let step =
            cached_model.logits_cached(&narrowed.narrow::<1>(position, 1)?, &mut cache, &ctx)?;
        // Incremental decoding is an optimization, not an approximation.
        assert_eq!(step.to_vec()?, full.narrow::<1>(position, 1)?.to_vec()?);
    }
    assert_eq!(cache.len, 4);
    assert_eq!(cache.layers.len(), LAYERS);
    assert!(cache.layers.iter().all(Option::is_some));
    let prompt = narrowed.narrow::<1>(0, 3)?;
    let (generated, generated_cache) = cached_model.generate(&prompt, 3, &ctx)?;
    assert_eq!(generated.dims(), [5, 6]);
    assert_eq!(generated_cache.len, 6);
    // Shape alone accepts any continuation, including a non-greedy one. Pin the
    // actual token ids against the cache-free greedy reference.
    let expected_generated = cached_model.generate_uncached(&prompt, 3, &ctx)?;
    assert_eq!(generated.to_vec()?, expected_generated.to_vec()?);
    assert_eq!(generated.narrow::<1>(0, 3)?.to_vec()?, prompt.to_vec()?);
    assert!(
        generated
            .narrow::<1>(3, 3)?
            .to_vec()?
            .iter()
            .all(|&id| (0..VOCAB as i64).contains(&id))
    );

    let mut reference = TypedDecoder::new(&ctx, &mut Rng::seed(31))?;
    let mut reference_optimizer = Adam::new(0.025);
    let first = loss(&mut reference, &inputs, &targets, &ctx)?.item()?;
    train_steps(
        &mut reference,
        &mut reference_optimizer,
        &inputs,
        &targets,
        &ctx,
        10,
    )?;
    let last = loss(&mut reference, &inputs, &targets, &ctx)?.item()?;
    assert!(
        last < first * 0.8,
        "character-LM loss did not decrease: {first} -> {last}"
    );

    let mut staged = TypedDecoder::new(&ctx, &mut Rng::seed(31))?;
    let mut staged_optimizer = Adam::new(0.025);
    train_steps(
        &mut staged,
        &mut staged_optimizer,
        &inputs,
        &targets,
        &ctx,
        5,
    )?;
    let path = checkpoint_path();
    save_training_checkpoint(&mut staged, &staged_optimizer, &path)?;
    let saved = Envelope::load(&path, &Limits::defaults())?;
    assert_eq!(saved.section("config"), Some(CONFIG));
    assert!(saved.tensor("blocks.0.attention.q_proj.weight").is_some());
    assert!(saved.tensor("optim.output.weight.m").is_some());

    let reconstructed_config = saved.section("config").unwrap();
    assert_eq!(reconstructed_config, CONFIG);
    let mut reconstructed = TypedDecoder::new(&ctx, &mut Rng::seed(999))?;
    let mut resumed_optimizer = Adam::new(999.0);
    load_adam_checkpoint(
        &mut resumed_optimizer,
        &mut reconstructed,
        &path,
        &LoadOptions::strict(),
    )?;
    assert_eq!(
        adam_param_steps(&resumed_optimizer, &mut reconstructed, "output.weight")?,
        5
    );
    train_steps(
        &mut reconstructed,
        &mut resumed_optimizer,
        &inputs,
        &targets,
        &ctx,
        5,
    )?;

    let mut expected = model_envelope(&mut reference)?;
    save_adam_state(&reference_optimizer, &mut reference, &mut expected)?;
    let mut actual = model_envelope(&mut reconstructed)?;
    save_adam_state(&resumed_optimizer, &mut reconstructed, &mut actual)?;
    assert_eq!(actual, expected);
    assert_eq!(
        reconstructed.logits(&inputs, Mode::EVAL, &ctx)?.to_vec()?,
        reference.logits(&inputs, Mode::EVAL, &ctx)?.to_vec()?
    );
    std::fs::remove_file(path)?;
    Ok(())
}

/// One decoder layer's worth of typed modules at precision `E`.
///
/// Every module is built directly at `E` except the output projection, which
/// exercises the public [`ToDType`] conversion from an `f32` model instead.
/// Returns the logits widened to `f32` so callers can compare precisions.
fn reduced_precision_logits<E: FloatElement + NumericElement>(
    ids: &Tokens,
    expected: DType,
    ctx: &DeviceCtx<Cpu>,
) -> Result<Vec<f32>> {
    let mut rng = Rng::seed(23);
    let embedding = Embedding::<VOCAB, EMBED, E>::new(ctx, &mut rng)?;
    let attention = MultiHeadAttention::<EMBED, HEADS, E>::new(ctx, &mut rng)?;
    let mut norm = LayerNorm::<Tensor1<EMBED, E>>::new([EMBED], ctx)?;
    let wide_output = Linear::<EMBED, VOCAB>::new(EMBED, VOCAB, ctx, &mut rng)?;
    let mut output = ToDType::<E>::to_dtype(wide_output)?;

    let [rows, sequence] = ids.dims();
    let causal = Tensor2::<DYN, DYN, bool>::causal_mask(sequence, ctx)?;
    let mask: Tensor4<DYN, HEADS, DYN, DYN, bool> =
        causal.broadcast_to([rows, HEADS, sequence, sequence])?;

    let hidden = embedding.lookup(ids, Mode::EVAL)?;
    assert_eq!(hidden.dims(), [rows, sequence, EMBED]);
    assert_eq!(hidden.as_dynamic().dtype(), expected);
    let attended = attention.attend(&hidden, Some(&mask), Mode::EVAL)?;
    let normalized = norm.forward(&hidden.add(&attended)?, Mode::EVAL)?;
    let logits = output.forward(&normalized, Mode::EVAL)?;

    assert_eq!(logits.dims(), [rows, sequence, VOCAB]);
    assert_eq!(logits.as_dynamic().dtype(), expected);
    let widened = logits.as_dynamic().to_dtype(DType::F32)?.to_vec::<f32>()?;
    assert!(
        widened.iter().all(|value| value.is_finite()),
        "{expected} logits are not all finite: {widened:?}"
    );
    Ok(widened)
}

fn assert_tracks_wide(actual: &[f32], wide: &[f32], tolerance: f32, label: &str) {
    assert_eq!(actual.len(), wide.len());
    let worst = actual
        .iter()
        .zip(wide)
        .map(|(actual, wide)| (actual - wide).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst <= tolerance,
        "{label} logits drifted from f32 by {worst}, above {tolerance}"
    );
}

#[test]
fn typed_decoder_layer_runs_in_reduced_precision() -> Result<()> {
    let ctx = DeviceCtx::cpu()?;
    let (ids, _) = batch(&ctx, &[0, 4, 9, 15, 21], 4)?;

    // Both reduced formats are supported on CPU; neither is skipped.
    let wide = reduced_precision_logits::<f32>(&ids, DType::F32, &ctx)?;
    let half = reduced_precision_logits::<f16>(&ids, DType::F16, &ctx)?;
    let brain = reduced_precision_logits::<bf16>(&ids, DType::BF16, &ctx)?;

    // Every module is seeded identically across precisions, so the reduced runs
    // must track the f32 reference rather than merely be finite. Measured worst
    // drift is 3.4e-3 (f16) and 2.8e-2 (bf16); bf16 keeps f32's exponent but
    // only eight mantissa bits, so its tolerance is the looser of the two.
    assert_tracks_wide(&half, &wide, 5e-3, "f16");
    assert_tracks_wide(&brain, &wide, 4e-2, "bf16");

    // The reduced runs must actually lose precision, otherwise the dtype
    // assertions would also be satisfied by an f32 computation that was merely
    // cast at the very end.
    assert_ne!(half, wide);
    assert_ne!(brain, wide);
    assert_ne!(half, brain);
    Ok(())
}
