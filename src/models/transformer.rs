//! A config-driven decoder-only transformer with incremental KV caching.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use crate::device::Device;
use crate::error::{Error, Result};
use crate::nn::{
    Embedding, Forward, LayerNorm, Linear, Mode, MultiHeadAttention, scaled_dot_product_attention,
};
use crate::persist::{Envelope, Limits, LoadOptions};
use crate::rng::Rng;
use crate::tensor::Tensor;

/// Runtime dimensions for a [`DecoderTransformer`].
///
/// The complete value is stored in every model checkpoint, so loading a
/// checkpoint reconstructs the model without a separately supplied
/// architecture definition.
///
/// `#[non_exhaustive]`: the knob set is crate-owned and may grow in a minor
/// release, so build a config with [`TransformerConfig::new`] and the `with_*`
/// setters rather than a struct literal. That is what keeps a later field
/// addition non-breaking — a new field arrives with its own `with_*` setter
/// and a default, and existing builder chains keep compiling untouched.
///
/// # Examples
///
/// ```
/// use rstorch::models::TransformerConfig;
///
/// let config = TransformerConfig::new(16, 8, 4, 2, 1).with_num_layers(2);
/// assert_eq!(config.num_layers(), 2);
/// assert_eq!(config.feed_forward_dim(), 16); // 4 * embed_dim, the default
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct TransformerConfig {
    /// Number of tokens in the vocabulary.
    vocab_size: usize,
    /// Longest sequence accepted by the learned positional embedding.
    max_seq_len: usize,
    /// Width of each token representation.
    embed_dim: usize,
    /// Number of attention heads in each block.
    num_heads: usize,
    /// Number of decoder blocks.
    num_layers: usize,
    /// Width of each block's feed-forward hidden layer.
    feed_forward_dim: usize,
}

impl TransformerConfig {
    /// Assemble a config from the five dimensions that have no defensible
    /// default. `feed_forward_dim` defaults to `4 * embed_dim` — the standard
    /// convention — and is overridable with
    /// [`with_feed_forward_dim`](TransformerConfig::with_feed_forward_dim).
    ///
    /// That default exists to close a specific hazard, not for brevity: with
    /// all six dimensions positional and the same type, transposing
    /// `embed_dim` and `feed_forward_dim` at a call site compiles, and
    /// `validate` — which only checks
    /// non-zero and `embed_dim % num_heads == 0` — accepts the swap whenever
    /// the new `embed_dim` still happens to divide evenly, silently training
    /// and checkpointing the wrong architecture. Removing the field from the
    /// positional list removes the transposition. `num_heads` and
    /// `num_layers` stay required and adjacent (transposing *them* is usually
    /// invisible because the two are often equal, but neither has a default a
    /// caller would want silently applied) — an omitted dimension is a
    /// missing-argument compile error, which is the failure mode this
    /// signature is designed to convert transpositions into.
    ///
    /// Infallible: the dimensions are checked where they are used, by
    /// [`DecoderTransformer::new`] and by `decode`.
    /// The `4 * embed_dim` default saturates rather than overflowing, so a
    /// nonsense `embed_dim` — from a corrupt checkpoint, say — still reaches
    /// that validation instead of panicking here in a debug build.
    #[must_use]
    pub fn new(
        vocab_size: usize,
        max_seq_len: usize,
        embed_dim: usize,
        num_heads: usize,
        num_layers: usize,
    ) -> Self {
        Self {
            vocab_size,
            max_seq_len,
            embed_dim,
            num_heads,
            num_layers,
            feed_forward_dim: embed_dim.saturating_mul(4),
        }
    }

    /// Number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Longest sequence accepted by the learned positional embedding.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Width of each token representation.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// Number of attention heads in each block.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Number of decoder blocks.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Width of each block's feed-forward hidden layer.
    pub fn feed_forward_dim(&self) -> usize {
        self.feed_forward_dim
    }

    /// Replace the vocabulary size.
    #[must_use]
    pub fn with_vocab_size(mut self, vocab_size: usize) -> Self {
        self.vocab_size = vocab_size;
        self
    }

    /// Replace the longest accepted sequence length.
    #[must_use]
    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    /// Replace the token representation width.
    #[must_use]
    pub fn with_embed_dim(mut self, embed_dim: usize) -> Self {
        self.embed_dim = embed_dim;
        self
    }

    /// Replace the per-block attention head count.
    #[must_use]
    pub fn with_num_heads(mut self, num_heads: usize) -> Self {
        self.num_heads = num_heads;
        self
    }

    /// Replace the decoder block count.
    #[must_use]
    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Replace the feed-forward hidden width.
    #[must_use]
    pub fn with_feed_forward_dim(mut self, feed_forward_dim: usize) -> Self {
        self.feed_forward_dim = feed_forward_dim;
        self
    }

    fn validate(&self) -> Result<()> {
        if self.vocab_size == 0
            || self.max_seq_len == 0
            || self.embed_dim == 0
            || self.num_heads == 0
            || self.num_layers == 0
            || self.feed_forward_dim == 0
            || !self.embed_dim.is_multiple_of(self.num_heads)
        {
            return Err(Error::InvalidArg {
                op: "DecoderTransformer::new",
                msg: format!(
                    "all dimensions must be non-zero and num_heads must divide embed_dim: {self:?}"
                ),
            });
        }
        Ok(())
    }

    fn encode(&self) -> String {
        format!(
            "version=1;vocab_size={};max_seq_len={};embed_dim={};num_heads={};num_layers={};feed_forward_dim={}",
            self.vocab_size,
            self.max_seq_len,
            self.embed_dim,
            self.num_heads,
            self.num_layers,
            self.feed_forward_dim
        )
    }

    fn decode(value: &str) -> Result<Self> {
        let mut fields = BTreeMap::new();
        for part in value.split(';') {
            let (name, raw) = part.split_once('=').ok_or_else(|| {
                Error::persistence(format!("invalid transformer config field {part:?}"))
            })?;
            if fields.insert(name, raw).is_some() {
                return Err(Error::persistence(format!(
                    "duplicate transformer config field {name:?}"
                )));
            }
        }

        let version = fields
            .remove("version")
            .ok_or_else(|| Error::persistence("transformer config missing \"version\""))?
            .parse::<usize>()
            .map_err(|source| {
                Error::persistence_with("transformer config \"version\" is not a usize", source)
            })?;
        if version != 1 {
            return Err(Error::persistence(format!(
                "unsupported transformer config version {version} (expected 1)"
            )));
        }

        let mut parse = |name: &str| -> Result<usize> {
            fields
                .remove(name)
                .ok_or_else(|| Error::persistence(format!("transformer config missing {name:?}")))?
                .parse()
                .map_err(|source| {
                    Error::persistence_with(
                        format!("transformer config {name:?} is not a usize"),
                        source,
                    )
                })
        };
        let config = Self::new(
            parse("vocab_size")?,
            parse("max_seq_len")?,
            parse("embed_dim")?,
            parse("num_heads")?,
            parse("num_layers")?,
        )
        .with_feed_forward_dim(parse("feed_forward_dim")?);
        if let Some(name) = fields.keys().next() {
            return Err(Error::persistence(format!(
                "unknown transformer config field {name:?}"
            )));
        }
        config.validate().map_err(|error| {
            Error::persistence_with(
                "checkpoint's transformer config is internally inconsistent",
                error,
            )
        })?;
        Ok(config)
    }
}
#[derive(rstorch::Module)]
struct Block {
    norm1: LayerNorm,
    attention: MultiHeadAttention,
    norm2: LayerNorm,
    feed_forward1: Linear,
    feed_forward2: Linear,
}

impl Block {
    fn new(config: &TransformerConfig, device: &Device, rng: &mut Rng) -> Result<Self> {
        Ok(Self {
            norm1: LayerNorm::new([config.embed_dim], device)?,
            attention: MultiHeadAttention::new_without_bias(
                config.embed_dim,
                config.num_heads,
                device,
                rng,
            )?,
            norm2: LayerNorm::new([config.embed_dim], device)?,
            feed_forward1: Linear::new(config.embed_dim, config.feed_forward_dim, device, rng)?,
            feed_forward2: Linear::new(config.feed_forward_dim, config.embed_dim, device, rng)?,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: &Tensor, mode: Mode) -> Result<Tensor> {
        let normalized = self.norm1.forward(x, mode)?;
        let x = x.add(&self.attention.attend(&normalized, Some(mask), mode)?)?;
        let normalized = self.norm2.forward(&x, mode)?;
        let hidden = self.feed_forward1.forward(&normalized, mode)?.gelu()?;
        x.add(&self.feed_forward2.forward(&hidden, mode)?)
    }

    /// One cached decoding step for this block.
    ///
    /// Reads the previous prefix but does **not** write it back: the extended
    /// cache is returned so the caller can commit every layer at once. A
    /// rejected step must leave the cache exactly as it was — extending some
    /// layers and not others would silently corrupt every later step, and it is
    /// this function's fallible tail (`cat`, attention, the output projection)
    /// that makes that reachable.
    fn forward_cached(
        &mut self,
        x: &Tensor,
        cache: Option<&LayerCache>,
        mode: Mode,
    ) -> Result<(Tensor, LayerCache)> {
        let normalized = self.norm1.forward(x, mode)?;
        let query = self.attention.project_query(&normalized, mode)?;
        let (new_keys, new_values) = self.attention.project_keys_values(&normalized, mode)?;
        let (keys, values) = match cache {
            Some(previous) => (
                Tensor::cat(&[&previous.keys, &new_keys], -2)?,
                Tensor::cat(&[&previous.values, &new_values], -2)?,
            ),
            None => (new_keys, new_values),
        };
        let context = scaled_dot_product_attention(&query, &keys, &values, None)?;
        let attended = self.attention.project_output(&context, mode)?;

        let x = x.add(&attended)?;
        let normalized = self.norm2.forward(&x, mode)?;
        let hidden = self.feed_forward1.forward(&normalized, mode)?.gelu()?;
        let out = x.add(&self.feed_forward2.forward(&hidden, mode)?)?;
        Ok((out, LayerCache { keys, values }))
    }
}

struct LayerCache {
    keys: Tensor,
    values: Tensor,
}

/// Per-layer projected keys and values used for incremental decoding.
///
/// Construct a cache with [`DecoderTransformer::empty_cache`]. Each cached
/// step appends one token to every layer; the existing prefix is attended to
/// directly and is never projected again.
pub struct KvCache {
    layers: Vec<Option<LayerCache>>,
    len: usize,
    model_id: Arc<()>,
}

impl KvCache {
    /// Number of token positions currently cached in every layer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the cache contains no token positions.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A pre-normalized decoder-only transformer with learned token and position
/// embeddings, causal self-attention, GELU feed-forward blocks, and a linear
/// vocabulary head.
///
/// Parameter paths are stable dotted checkpoint keys. In particular, blocks
/// are stored in a derived `Vec<Block>` and therefore use paths such as
/// `blocks.0.attention.q_proj.weight`.
///
/// # Examples
///
/// ```
/// use rstorch::models::{DecoderTransformer, TransformerConfig};
/// use rstorch::{Device, Rng, Tensor};
///
/// # fn main() -> rstorch::Result<()> {
/// let config = TransformerConfig::new(16, 8, 4, 2, 1).with_feed_forward_dim(8);
/// let mut model = DecoderTransformer::new(config, &Device::Cpu, &mut Rng::seed(0))?;
///
/// let prompt = Tensor::from_vec(vec![1i64, 2, 3], [1, 3], &Device::Cpu)?;
/// let generated = model.generate(&prompt, 2)?;
/// assert_eq!(generated.dims(), &[1, 5]);
/// # Ok(())
/// # }
/// ```
#[derive(rstorch::Module)]
pub struct DecoderTransformer {
    token_embedding: Embedding,
    position_embedding: Embedding,
    blocks: Vec<Block>,
    final_norm: LayerNorm,
    output: Linear,
    #[module(skip)]
    config: TransformerConfig,
    #[module(skip)]
    cache_id: Arc<()>,
}

impl DecoderTransformer {
    /// Construct a model from `config`, drawing all random parameters from
    /// `rng` in a deterministic order.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArg`] if a dimension is zero or `num_heads` does not
    /// divide `embed_dim`, plus allocation errors from the constituent layers.
    pub fn new(config: TransformerConfig, device: &Device, rng: &mut Rng) -> Result<Self> {
        config.validate()?;
        let token_embedding = Embedding::new(config.vocab_size, config.embed_dim, device, rng)?;
        let position_embedding = Embedding::new(config.max_seq_len, config.embed_dim, device, rng)?;
        let blocks = (0..config.num_layers)
            .map(|_| Block::new(&config, device, rng))
            .collect::<Result<_>>()?;
        let final_norm = LayerNorm::new([config.embed_dim], device)?;
        let output = Linear::new(config.embed_dim, config.vocab_size, device, rng)?;
        Ok(Self {
            token_embedding,
            position_embedding,
            blocks,
            final_norm,
            output,
            config,
            cache_id: Arc::new(()),
        })
    }

    /// The architecture dimensions persisted with this model.
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }
    /// Create an empty key/value cache owned by this model.
    ///
    /// The returned cache is accepted by [`logits_cached`](Self::logits_cached)
    /// and is tied to this model instance.
    pub fn empty_cache(&self) -> KvCache {
        KvCache {
            layers: (0..self.blocks.len()).map(|_| None).collect(),
            len: 0,
            model_id: Arc::clone(&self.cache_id),
        }
    }

    /// Compute causal language-model logits for an `I64` token-id tensor of
    /// shape `[batch, sequence]`.
    ///
    /// # Errors
    ///
    /// Returns a rank/shape error for inputs other than a non-empty rank-2
    /// batch or for a sequence longer than `max_seq_len`; embedding, attention,
    /// and projection errors otherwise propagate.
    pub fn logits(&mut self, token_ids: &Tensor, mode: Mode) -> Result<Tensor> {
        let (_batch, sequence) = checked_tokens(
            token_ids,
            self.config.max_seq_len,
            "DecoderTransformer::logits",
        )?;
        let positions = Tensor::index_range(sequence, &token_ids.device())?;
        let mut hidden = self
            .token_embedding
            .lookup(token_ids, mode)?
            .add(&self.position_embedding.lookup(&positions, mode)?)?;
        let mask = Tensor::causal_mask(sequence, &token_ids.device())?;
        for block in &mut self.blocks {
            hidden = block.forward(&hidden, &mask, mode)?;
        }
        let hidden = self.final_norm.forward(&hidden, mode)?;
        self.output.forward(&hidden, mode)
    }

    /// Process one `[batch, 1]` token using and extending `cache`.
    ///
    /// The result is `[batch, 1, vocab_size]`. Keys and values for the new
    /// token are projected once per layer and appended to the stored prefix.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArg`] if the cache belongs to a different model,
    /// is full, or `token_ids` is not one token per batch.
    pub fn logits_cached(
        &mut self,
        token_ids: &Tensor,
        cache: &mut KvCache,
        mode: Mode,
    ) -> Result<Tensor> {
        let (_batch, sequence) = checked_tokens(token_ids, 1, "DecoderTransformer::logits_cached")?;
        if sequence != 1
            || cache.layers.len() != self.blocks.len()
            || !Arc::ptr_eq(&cache.model_id, &self.cache_id)
        {
            return Err(Error::InvalidArg {
                op: "DecoderTransformer::logits_cached",
                msg: "expected one token and a cache created by this model".to_string(),
            });
        }
        if cache.len >= self.config.max_seq_len {
            return Err(Error::InvalidArg {
                op: "DecoderTransformer::logits_cached",
                msg: format!("KV cache reached max_seq_len {}", self.config.max_seq_len),
            });
        }
        let position = Tensor::from_vec(
            vec![i64::try_from(cache.len).map_err(|_| Error::InvalidArg {
                op: "DecoderTransformer::logits_cached",
                msg: "cache position does not fit in i64".to_string(),
            })?],
            [1],
            &token_ids.device(),
        )?;
        let mut hidden = self
            .token_embedding
            .lookup(token_ids, mode)?
            .add(&self.position_embedding.lookup(&position, mode)?)?;
        // Stage every layer's extended cache, then commit once the whole step
        // has succeeded. Writing each layer as it is computed would leave the
        // cache half-advanced (and `cache.len` stale) on any error below, and
        // the next otherwise-valid step would return wrong logits with no error
        // to explain them.
        let mut staged = Vec::with_capacity(self.blocks.len());
        for (block, layer_cache) in self.blocks.iter_mut().zip(&cache.layers) {
            let (next, extended) = block.forward_cached(&hidden, layer_cache.as_ref(), mode)?;
            hidden = next;
            staged.push(extended);
        }
        let normalized = self.final_norm.forward(&hidden, mode)?;
        let logits = self.output.forward(&normalized, mode)?;

        for (slot, extended) in cache.layers.iter_mut().zip(staged) {
            *slot = Some(extended);
        }
        cache.len += 1;
        Ok(logits)
    }

    /// Greedily append `max_new_tokens` tokens using a real per-layer KV cache.
    ///
    /// The prompt is consumed one token at a time to populate the cache; each
    /// generated token then attends to those cached projections. The returned
    /// tensor has shape `[batch, prompt_len + max_new_tokens]`.
    ///
    /// # Errors
    ///
    /// As [`logits_cached`](Self::logits_cached), and
    /// [`Error::InvalidArg`] if the requested total exceeds `max_seq_len`.
    pub fn generate(&mut self, prompt: &Tensor, max_new_tokens: usize) -> Result<Tensor> {
        self.generate_with_cache_impl(prompt, max_new_tokens, "DecoderTransformer::generate")
            .map(|(tokens, _)| tokens)
    }

    /// The cached variant of [`generate`](Self::generate), returning the
    /// populated cache as evidence and for continued generation.
    ///
    /// # Errors
    ///
    /// As [`generate`](Self::generate).
    ///
    /// # Panics
    ///
    /// Never in practice: `prompt` is checked non-empty above, so the
    /// prompt-priming loop runs at least once before `logits` is read.
    pub fn generate_with_cache(
        &mut self,
        prompt: &Tensor,
        max_new_tokens: usize,
    ) -> Result<(Tensor, KvCache)> {
        self.generate_with_cache_impl(
            prompt,
            max_new_tokens,
            "DecoderTransformer::generate_with_cache",
        )
    }

    fn generate_with_cache_impl(
        &mut self,
        prompt: &Tensor,
        max_new_tokens: usize,
        op: &'static str,
    ) -> Result<(Tensor, KvCache)> {
        let (_batch, prompt_len) = checked_tokens(prompt, self.config.max_seq_len, op)?;
        let total = prompt_len
            .checked_add(max_new_tokens)
            .ok_or_else(|| Error::InvalidArg {
                op,
                msg: "generated sequence length overflow".to_string(),
            })?;
        if total > self.config.max_seq_len {
            return Err(Error::InvalidArg {
                op,
                msg: format!(
                    "prompt plus generation is {total}, above max_seq_len {}",
                    self.config.max_seq_len
                ),
            });
        }

        let mut cache = self.empty_cache();
        let mut tokens = prompt.clone();
        let mut logits = None;
        for position in 0..prompt_len {
            logits = Some(self.logits_cached(
                &prompt.narrow(1, position, 1)?,
                &mut cache,
                Mode::EVAL,
            )?);
        }
        for _ in 0..max_new_tokens {
            let next = logits
                .as_ref()
                .expect("checked_tokens rejects an empty prompt")
                .argmax(-1)?;
            tokens = Tensor::cat(&[&tokens, &next], 1)?;
            logits = Some(self.logits_cached(&next, &mut cache, Mode::EVAL)?);
        }
        Ok((tokens, cache))
    }

    /// Atomically save model configuration and model tensor state in the
    /// versioned [`Envelope`] format. Optimizer, RNG, and application sections
    /// are not included by this model-only convenience method.
    ///
    /// # Errors
    ///
    /// Host transfer, envelope validation, and filesystem errors propagate.
    ///
    /// # Examples
    ///
    /// ```
    /// use rstorch::models::{DecoderTransformer, TransformerConfig};
    /// use rstorch::persist::Limits;
    /// use rstorch::{Device, Rng};
    ///
    /// # fn main() -> rstorch::Result<()> {
    /// let config = TransformerConfig::new(16, 8, 4, 2, 1).with_feed_forward_dim(8);
    /// let model = DecoderTransformer::new(config, &Device::Cpu, &mut Rng::seed(0))?;
    ///
    /// let path = std::env::temp_dir().join(format!(
    ///     "rstorch-doctest-transformer-{}.safetensors",
    ///     std::process::id()
    /// ));
    /// model.save_checkpoint(&path, &Limits::default())?;
    ///
    /// let restored = DecoderTransformer::load_checkpoint(
    ///     &path,
    ///     &Device::Cpu,
    ///     &Limits::default(),
    /// )?;
    /// assert_eq!(restored.config(), model.config());
    /// # let _ = std::fs::remove_file(&path);
    /// # Ok(())
    /// # }
    /// ```
    pub fn save_checkpoint(&self, path: impl AsRef<Path>, limits: &Limits) -> Result<()> {
        let mut envelope = Envelope::new();
        envelope.set_section("config", self.config.encode())?;
        crate::persist::save_model_state(self, &mut envelope)?;
        envelope.save(path, limits)
    }

    /// Load an [`Envelope`] from disk, reconstruct the architecture from its
    /// config section, and load its exact model tensor state on `device`.
    ///
    /// # Errors
    ///
    /// Invalid/missing config, model state conversion, exact state-dict
    /// matching, reader-limit, and filesystem errors propagate.
    pub fn load_checkpoint(
        path: impl AsRef<Path>,
        device: &Device,
        limits: &Limits,
    ) -> Result<Self> {
        let envelope = Envelope::load(path, limits)?;
        let encoded = envelope
            .section("config")
            .ok_or_else(|| Error::persistence("checkpoint has no config section"))?;
        let config = TransformerConfig::decode(encoded)?;
        let mut model = Self::new(config, device, &mut Rng::seed(0))?;
        let options = LoadOptions::strict().with_limits(*limits);
        crate::persist::load_model_state(&mut model, &envelope, &options)?;
        Ok(model)
    }
}

impl Forward for DecoderTransformer {
    type Output = Tensor;

    fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        self.logits(x, mode)
    }
}

fn checked_tokens(
    token_ids: &Tensor,
    max_sequence: usize,
    op: &'static str,
) -> Result<(usize, usize)> {
    let (batch, sequence) = token_ids.dims2().map_err(|_| Error::RankMismatch {
        op,
        expected: 2,
        got: token_ids.rank(),
    })?;
    if batch == 0 || sequence == 0 || sequence > max_sequence {
        return Err(Error::InvalidArg {
            op,
            msg: format!(
                "expected non-empty [batch, sequence] with sequence <= {max_sequence}, got {}",
                token_ids.shape()
            ),
        });
    }

    Ok((batch, sequence))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::ModuleExt;

    const CPU: Device = Device::Cpu;

    fn config() -> TransformerConfig {
        TransformerConfig::new(11, 8, 8, 2, 2).with_feed_forward_dim(16)
    }

    #[test]
    fn config_v1_round_trips_and_requires_supported_version() {
        let config = config();
        let encoded = config.encode();
        assert!(encoded.starts_with("version=1;"));
        assert_eq!(TransformerConfig::decode(&encoded).unwrap(), config);

        let without_version = encoded.split_once(';').map(|(_, rest)| rest).unwrap();
        assert!(TransformerConfig::decode(without_version).is_err());
        assert!(TransformerConfig::decode(&encoded.replacen("version=1", "version=2", 1)).is_err());
    }

    #[test]
    fn config_v1_rejects_duplicate_and_unknown_fields() {
        let encoded = config().encode();
        assert!(TransformerConfig::decode(&format!("{encoded};num_layers=2")).is_err());
        assert!(TransformerConfig::decode(&format!("{encoded};future=1")).is_err());
    }

    #[test]
    fn public_token_operations_attribute_metadata_errors() {
        let mut model = DecoderTransformer::new(config(), &CPU, &mut Rng::seed(8)).unwrap();
        let rank_one = Tensor::from_vec(vec![1i64], [1], &CPU).unwrap();
        assert!(matches!(
            model.logits(&rank_one, Mode::EVAL),
            Err(Error::RankMismatch {
                op: "DecoderTransformer::logits",
                ..
            })
        ));
        let mut cache = model.empty_cache();
        assert!(matches!(
            model.logits_cached(&rank_one, &mut cache, Mode::EVAL),
            Err(Error::RankMismatch {
                op: "DecoderTransformer::logits_cached",
                ..
            })
        ));
        assert!(matches!(
            model.generate_with_cache(&rank_one, 0),
            Err(Error::RankMismatch {
                op: "DecoderTransformer::generate_with_cache",
                ..
            })
        ));
    }

    #[test]
    fn direct_generation_overflow_uses_the_public_operation_label() {
        let mut model = DecoderTransformer::new(config(), &CPU, &mut Rng::seed(9)).unwrap();
        let prompt = Tensor::from_vec(vec![1i64], [1, 1], &CPU).unwrap();
        assert!(matches!(
            model.generate(&prompt, usize::MAX),
            Err(Error::InvalidArg {
                op: "DecoderTransformer::generate",
                ..
            })
        ));
        assert!(matches!(
            model.generate_with_cache(&prompt, usize::MAX),
            Err(Error::InvalidArg {
                op: "DecoderTransformer::generate_with_cache",
                ..
            })
        ));
    }

    #[test]
    fn derived_block_paths_are_indexed() {
        let model = DecoderTransformer::new(config(), &CPU, &mut Rng::seed(3)).unwrap();
        let state = model.state_dict().unwrap();
        assert!(state.contains_key("blocks.0.attention.q_proj.weight"));
        assert!(state.contains_key("blocks.1.feed_forward2.bias"));
        assert!(!state.keys().any(|key| key.starts_with("0.")));
    }

    #[test]
    fn cached_steps_match_full_causal_logits() {
        let mut model = DecoderTransformer::new(config(), &CPU, &mut Rng::seed(4)).unwrap();
        let ids = Tensor::from_vec(vec![4i64, 5, 6, 7], [1, 4], &CPU).unwrap();
        let full = model.logits(&ids, Mode::EVAL).unwrap();
        let mut cache = model.empty_cache();
        assert!(cache.is_empty());
        for position in 0..4 {
            let step = model
                .logits_cached(&ids.narrow(1, position, 1).unwrap(), &mut cache, Mode::EVAL)
                .unwrap();
            let expected = full.narrow(1, position, 1).unwrap();
            let got = step.to_vec::<f32>().unwrap();
            let want = expected.to_vec::<f32>().unwrap();
            for (actual, expected) in got.iter().zip(want) {
                assert!((actual - expected).abs() < 2e-5, "{actual} vs {expected}");
            }
        }
        assert_eq!(cache.len(), 4);
    }

    #[test]
    fn a_cache_cannot_be_reused_with_another_model() {
        let first = DecoderTransformer::new(config(), &CPU, &mut Rng::seed(4)).unwrap();
        let mut second = DecoderTransformer::new(config(), &CPU, &mut Rng::seed(4)).unwrap();
        let ids = Tensor::from_vec(vec![4i64], [1, 1], &CPU).unwrap();
        let mut cache = first.empty_cache();

        let error = second
            .logits_cached(&ids, &mut cache, Mode::EVAL)
            .unwrap_err();
        assert!(matches!(
            error,
            Error::InvalidArg {
                op: "DecoderTransformer::logits_cached",
                ..
            }
        ));
    }

    /// A rejected cached step must leave the cache byte-for-byte usable.
    ///
    /// The step is rejected *after* the per-layer projections have been
    /// computed, which is exactly the window in which a half-committed cache
    /// used to survive: the next legitimate step then returned wrong logits
    /// with no error to explain them.
    #[test]
    fn a_rejected_cached_step_does_not_corrupt_the_cache() {
        let mut model = DecoderTransformer::new(config(), &CPU, &mut Rng::seed(4)).unwrap();
        let ids = Tensor::from_vec(vec![4i64, 5, 6, 7], [1, 4], &CPU).unwrap();

        // Reference: three clean steps.
        let mut clean = model.empty_cache();
        let mut expected = Vec::new();
        for position in 0..3 {
            let step = model
                .logits_cached(&ids.narrow(1, position, 1).unwrap(), &mut clean, Mode::EVAL)
                .unwrap();
            expected.push(step.to_vec::<f32>().unwrap());
        }

        // Same run, but with a rejected step wedged in after the second.
        let mut cache = model.empty_cache();
        let mut actual = Vec::new();
        for position in 0..3 {
            if position == 2 {
                // Batch 2, sequence 1: this passes every up-front guard, so it
                // reaches the per-layer loop and fails inside it, when the
                // batch-2 projection is concatenated onto the batch-1 prefix.
                // That is the only window in which a partially updated cache
                // can survive, so it is the case worth pinning.
                let wide = Tensor::from_vec(vec![4i64, 5], [2, 1], &CPU).unwrap();
                assert!(model.logits_cached(&wide, &mut cache, Mode::EVAL).is_err());
                assert_eq!(cache.len(), 2, "a rejected step must not advance the cache");
            }
            let step = model
                .logits_cached(&ids.narrow(1, position, 1).unwrap(), &mut cache, Mode::EVAL)
                .unwrap();
            actual.push(step.to_vec::<f32>().unwrap());
        }

        assert_eq!(cache.len(), clean.len());
        for (position, (got, want)) in actual.iter().zip(&expected).enumerate() {
            for (a, b) in got.iter().zip(want) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "step {position} diverged after a rejected step: {a} vs {b}"
                );
            }
        }
    }
}
