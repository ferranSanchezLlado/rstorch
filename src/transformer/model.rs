#![allow(clippy::type_complexity)]

use crate::backend::{Backend, Cpu};
use crate::data::Batch;
use crate::dtype::FloatDType;
use crate::error::{Error, Result, const_check};
use crate::nn::{
    CrossEntropyOpts, Embedding, Layer, LayerNorm, Linear, Module, MultiHeadAttention,
    PositionalEmbedding, Reduction, ensure_head_shape,
};
use crate::no_grad;
use crate::random::SmallRng;
use crate::shape::{AnyDim, C, D2, D3, Sym};
use crate::tensor::{Scalar, Tensor};
use crate::transformer::Tokenizer;

#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct TransformerConfig<E> {
    pub norm_eps: E,
    pub init_std: E,
}

impl<E> Default for TransformerConfig<E>
where
    E: FloatDType,
{
    fn default() -> Self {
        Self {
            norm_eps: E::from_f64(1e-5),
            init_std: E::from_f64(0.02),
        }
    }
}

/// Decoder transformer block.
///
/// Parameter names are part of the persistence contract. Child modules are
/// named `norm1`, `attention`, `norm2`, `fc1`, and `fc2`; their own parameter
/// names are appended, for example `attention.q_proj.weight` or `fc2.bias`.
pub struct TransformerBlock<
    const SEQ: usize,
    const EMBED: usize,
    const HEADS: usize,
    const HEAD_DIM: usize,
    const FF: usize,
    E = f32,
    B = Cpu,
> where
    E: FloatDType,
    B: Backend<E>,
{
    norm1: LayerNorm<EMBED, E, B>,
    attention: MultiHeadAttention<SEQ, EMBED, HEADS, HEAD_DIM, E, B>,
    norm2: LayerNorm<EMBED, E, B>,
    fc1: Linear<EMBED, FF, E, B>,
    fc2: Linear<FF, EMBED, E, B>,
}

impl<
    const SEQ: usize,
    const EMBED: usize,
    const HEADS: usize,
    const HEAD_DIM: usize,
    const FF: usize,
    E,
    B,
> TransformerBlock<SEQ, EMBED, HEADS, HEAD_DIM, FF, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn new(rng: &mut SmallRng) -> Result<Self> {
        Self::with_config(rng, TransformerConfig::default())
    }

    pub fn with_config(rng: &mut SmallRng, config: TransformerConfig<E>) -> Result<Self> {
        Ok(Self {
            norm1: LayerNorm::new(config.norm_eps)?,
            attention: MultiHeadAttention::xavier_uniform(rng)?,
            norm2: LayerNorm::new(config.norm_eps)?,
            fc1: Linear::xavier_uniform(rng)?,
            fc2: Linear::xavier_uniform(rng)?,
        })
    }

    pub fn forward(
        &self,
        input: &Tensor<D3<Sym<Batch>, C<SEQ>, C<EMBED>>, E, B>,
    ) -> Result<Tensor<D3<Sym<Batch>, C<SEQ>, C<EMBED>>, E, B>> {
        let attn_input = layer_norm_3d(&self.norm1, input)?;
        let attn = self.attention.forward_causal(&attn_input)?;
        let hidden = input.add(&attn)?;
        let ff_input = layer_norm_3d(&self.norm2, &hidden)?;
        let ff = feed_forward_3d(&self.fc1, &self.fc2, &ff_input)?;
        hidden.add(&ff)
    }
}

impl<
    const SEQ: usize,
    const EMBED: usize,
    const HEADS: usize,
    const HEAD_DIM: usize,
    const FF: usize,
    E,
    B,
> Layer<Tensor<D3<Sym<Batch>, C<SEQ>, C<EMBED>>, E, B>>
    for TransformerBlock<SEQ, EMBED, HEADS, HEAD_DIM, FF, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    type Output = Tensor<D3<Sym<Batch>, C<SEQ>, C<EMBED>>, E, B>;
}

impl<
    const SEQ: usize,
    const EMBED: usize,
    const HEADS: usize,
    const HEAD_DIM: usize,
    const FF: usize,
    E,
    B,
    Context,
> Module<Tensor<D3<Sym<Batch>, C<SEQ>, C<EMBED>>, E, B>, Context>
    for TransformerBlock<SEQ, EMBED, HEADS, HEAD_DIM, FF, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn forward(
        &self,
        input: &Tensor<D3<Sym<Batch>, C<SEQ>, C<EMBED>>, E, B>,
        _ctx: &mut Context,
    ) -> Result<Self::Output> {
        self.forward(input)
    }
}

crate::nn::has_parameters! {
    impl[
        const SEQ: usize,
        const EMBED: usize,
        const HEADS: usize,
        const HEAD_DIM: usize,
        const FF: usize,
        E,
        B,
    ] TransformerBlock<SEQ, EMBED, HEADS, HEAD_DIM, FF, E, B>
    where { }
    {
        params { }
        children { norm1, attention, norm2, fc1, fc2 }
        transparent_children { }
    }
}

/// Minimal decoder-only language model.
///
/// Parameter names are part of the persistence contract. Top-level names are
/// `token_embedding`, `position_embedding`, `blocks.{i}`, `final_norm`, and
/// `lm_head`; child module names are appended, for example
/// `blocks.0.attention.q_proj.weight` and `lm_head.bias`.
pub struct DecoderOnlyTransformer<
    const VOCAB: usize,
    const SEQ: usize,
    const EMBED: usize,
    const HEADS: usize,
    const HEAD_DIM: usize,
    const FF: usize,
    const LAYERS: usize = 1,
    E = f32,
    B = Cpu,
> where
    E: FloatDType,
    B: Backend<E>,
{
    token_embedding: Embedding<VOCAB, EMBED, E, B>,
    position_embedding: PositionalEmbedding<SEQ, EMBED, E, B>,
    blocks: [TransformerBlock<SEQ, EMBED, HEADS, HEAD_DIM, FF, E, B>; LAYERS],
    final_norm: LayerNorm<EMBED, E, B>,
    lm_head: Linear<EMBED, VOCAB, E, B>,
}

impl<
    const VOCAB: usize,
    const SEQ: usize,
    const EMBED: usize,
    const HEADS: usize,
    const HEAD_DIM: usize,
    const FF: usize,
    const LAYERS: usize,
    E,
    B,
> DecoderOnlyTransformer<VOCAB, SEQ, EMBED, HEADS, HEAD_DIM, FF, LAYERS, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn new(rng: &mut SmallRng) -> Result<Self> {
        Self::with_config(rng, TransformerConfig::default())
    }

    pub fn with_config(rng: &mut SmallRng, config: TransformerConfig<E>) -> Result<Self> {
        ensure_head_shape::<EMBED, HEADS, HEAD_DIM>()?;
        let emb_limit = config.init_std;
        let mut blocks = Vec::with_capacity(LAYERS);
        for _ in 0..LAYERS {
            blocks.push(TransformerBlock::with_config(rng, config)?);
        }
        let blocks = match blocks.try_into() {
            Ok(blocks) => blocks,
            // The vector is built with exactly one push for each value in 0..LAYERS.
            Err(_) => unreachable!("constructed exactly LAYERS transformer blocks"),
        };
        Ok(Self {
            token_embedding: Embedding::uniform(rng, -emb_limit, emb_limit)?,
            position_embedding: PositionalEmbedding::uniform(rng, -emb_limit, emb_limit)?,
            blocks,
            final_norm: LayerNorm::new(config.norm_eps)?,
            lm_head: Linear::xavier_uniform(rng)?,
        })
    }

    pub fn new_with_tokenizer<T>(rng: &mut SmallRng, tokenizer: &T) -> Result<Self>
    where
        T: Tokenizer,
    {
        Self::with_tokenizer_config(rng, tokenizer, TransformerConfig::default())
    }

    pub fn with_tokenizer_config<T>(
        rng: &mut SmallRng,
        tokenizer: &T,
        config: TransformerConfig<E>,
    ) -> Result<Self>
    where
        T: Tokenizer,
    {
        if tokenizer.vocab_size() != VOCAB {
            return Err(crate::error::ShapeError::LengthMismatch {
                expected: VOCAB,
                found: tokenizer.vocab_size(),
            }
            .into());
        }
        Self::with_config(rng, config)
    }

    pub fn forward(
        &self,
        input_ids: &[[usize; SEQ]],
    ) -> Result<Tensor<D3<Sym<Batch>, C<SEQ>, C<VOCAB>>, E, B>> {
        let batch = input_ids.len();
        let token = self.token_embedding.forward(input_ids)?;
        let pos = self.position_embedding.forward(batch)?;
        let mut hidden = token.add(&pos)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        let hidden = layer_norm_3d(&self.final_norm, &hidden)?;
        project_logits_3d(&self.lm_head, &hidden)
    }

    pub fn loss(
        &self,
        input_ids: &[[usize; SEQ]],
        targets: &[[usize; SEQ]],
    ) -> Result<Scalar<E, B>> {
        self.loss_ignore_index(input_ids, targets, usize::MAX)
    }

    pub fn loss_ignore_index(
        &self,
        input_ids: &[[usize; SEQ]],
        targets: &[[usize; SEQ]],
        ignore_index: usize,
    ) -> Result<Scalar<E, B>> {
        let logits = self.forward(input_ids)?;
        let targets: Vec<_> = targets.iter().flat_map(|row| row.iter().copied()).collect();
        logits
            .reshape_with_shape::<D2<AnyDim, C<VOCAB>>>([input_ids.len() * SEQ, VOCAB])?
            .cross_entropy_with(
                &targets,
                CrossEntropyOpts {
                    reduction: Reduction::Mean,
                    ignore_index: Some(ignore_index),
                },
            )
    }

    /// Greedy generation over a fixed `SEQ` context window. Each new token runs
    /// a full forward pass over the last `SEQ` ids; no KV cache is maintained.
    ///
    /// Prompts shorter than `SEQ` are right-padded with zeros so that real tokens
    /// occupy positions 0..len and receive correct learned positional embeddings.
    /// The logit is read at the last real position. Once the sequence grows to
    /// `SEQ` tokens the window slides left and all positions are real.
    pub fn generate(&self, prompt_ids: &[usize], max_new_tokens: usize) -> Result<Vec<usize>> {
        const { const_check::nonzero(SEQ, "generate", "SEQ") };

        if prompt_ids.is_empty() {
            return Err(Error::InvalidInput {
                op: "generate",
                reason: "prompt must not be empty",
            });
        }
        let _guard = no_grad();
        let mut ids = prompt_ids.to_vec();
        for _ in 0..max_new_tokens {
            let start = ids.len().saturating_sub(SEQ);
            let recent = &ids[start..];
            let read_pos = recent.len() - 1;
            let mut window = [0usize; SEQ];
            window[..recent.len()].copy_from_slice(recent);
            let batch = [window];
            let logits = self.forward(&batch)?;
            let predictions = logits
                .reshape_with_shape::<D2<AnyDim, C<VOCAB>>>([SEQ, VOCAB])?
                .argmax_last()?;
            ids.push(predictions[read_pos]);
        }
        Ok(ids)
    }
}

impl<
    const VOCAB: usize,
    const SEQ: usize,
    const EMBED: usize,
    const HEADS: usize,
    const HEAD_DIM: usize,
    const FF: usize,
    const LAYERS: usize,
    E,
    B,
> Layer<[[usize; SEQ]]>
    for DecoderOnlyTransformer<VOCAB, SEQ, EMBED, HEADS, HEAD_DIM, FF, LAYERS, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    type Output = Tensor<D3<Sym<Batch>, C<SEQ>, C<VOCAB>>, E, B>;
}

impl<
    const VOCAB: usize,
    const SEQ: usize,
    const EMBED: usize,
    const HEADS: usize,
    const HEAD_DIM: usize,
    const FF: usize,
    const LAYERS: usize,
    E,
    B,
    Context,
> Module<[[usize; SEQ]], Context>
    for DecoderOnlyTransformer<VOCAB, SEQ, EMBED, HEADS, HEAD_DIM, FF, LAYERS, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn forward(&self, input: &[[usize; SEQ]], _ctx: &mut Context) -> Result<Self::Output> {
        self.forward(input)
    }
}

crate::nn::has_parameters! {
    impl[
        const VOCAB: usize,
        const SEQ: usize,
        const EMBED: usize,
        const HEADS: usize,
        const HEAD_DIM: usize,
        const FF: usize,
        const LAYERS: usize,
        E,
        B,
    ] DecoderOnlyTransformer<VOCAB, SEQ, EMBED, HEADS, HEAD_DIM, FF, LAYERS, E, B>
    where { }
    {
        params { }
        children { token_embedding, position_embedding, blocks[], final_norm, lm_head }
        transparent_children { }
    }
}

fn project_3d<const SEQ: usize, const IN: usize, const OUT: usize, E, B>(
    layer: &Linear<IN, OUT, E, B>,
    input: &Tensor<D3<Sym<Batch>, C<SEQ>, C<IN>>, E, B>,
) -> Result<Tensor<D3<Sym<Batch>, C<SEQ>, C<OUT>>, E, B>>
where
    E: FloatDType,
    B: Backend<E>,
{
    let batch = input.shape().dims()[0];
    let flat = input.reshape_with_shape::<D2<AnyDim, C<IN>>>([batch * SEQ, IN])?;
    let mut ctx = ();
    layer
        .forward(&flat, &mut ctx)?
        .reshape_with_shape([batch, SEQ, OUT])
}

fn project_logits_3d<const SEQ: usize, const EMBED: usize, const VOCAB: usize, E, B>(
    layer: &Linear<EMBED, VOCAB, E, B>,
    input: &Tensor<D3<Sym<Batch>, C<SEQ>, C<EMBED>>, E, B>,
) -> Result<Tensor<D3<Sym<Batch>, C<SEQ>, C<VOCAB>>, E, B>>
where
    E: FloatDType,
    B: Backend<E>,
{
    project_3d(layer, input)
}

fn layer_norm_3d<const SEQ: usize, const FEATURES: usize, E, B>(
    norm: &LayerNorm<FEATURES, E, B>,
    input: &Tensor<D3<Sym<Batch>, C<SEQ>, C<FEATURES>>, E, B>,
) -> Result<Tensor<D3<Sym<Batch>, C<SEQ>, C<FEATURES>>, E, B>>
where
    E: FloatDType,
    B: Backend<E>,
{
    let batch = input.shape().dims()[0];
    let flat = input.reshape_with_shape::<D2<AnyDim, C<FEATURES>>>([batch * SEQ, FEATURES])?;
    let mut ctx = ();
    norm.forward(&flat, &mut ctx)?
        .reshape_with_shape([batch, SEQ, FEATURES])
}

fn feed_forward_3d<const SEQ: usize, const EMBED: usize, const FF: usize, E, B>(
    fc1: &Linear<EMBED, FF, E, B>,
    fc2: &Linear<FF, EMBED, E, B>,
    input: &Tensor<D3<Sym<Batch>, C<SEQ>, C<EMBED>>, E, B>,
) -> Result<Tensor<D3<Sym<Batch>, C<SEQ>, C<EMBED>>, E, B>>
where
    E: FloatDType,
    B: Backend<E>,
{
    let batch = input.shape().dims()[0];
    let flat = input.reshape_with_shape::<D2<AnyDim, C<EMBED>>>([batch * SEQ, EMBED])?;
    let mut ctx = ();
    let hidden = fc1.forward(&flat, &mut ctx)?.gelu()?;
    fc2.forward(&hidden, &mut ctx)?
        .reshape_with_shape([batch, SEQ, EMBED])
}
