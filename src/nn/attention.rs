//! [`MultiHeadAttention`] and the [`scaled_dot_product_attention`] kernel it
//! is built from.
//!
//! Everything here is **runtime-dimensioned** (`embed_dim`, `num_heads` and
//! every sequence length are ordinary `usize`s read off the tensors) and the
//! attention mask is an **argument, never stored state**. Those two choices
//! are what make one type serve self-attention, cross-attention, and
//! incremental decoding:
//!
//! - a stored mask has to be rebuilt (and cached, and invalidated) whenever
//!   the sequence length changes, which is exactly what generation does on
//!   every step, and the cache is shared mutable state in what is otherwise
//!   an immutable forward;
//! - a mask *argument* is a plain [`Bool`](crate::DType::Bool) tensor, so
//!   combining a causal mask with a padding mask is ordinary tensor
//!   arithmetic in the caller: `causal.where_cond(&causal, &padding)?` is
//!   their boolean "or" (blocked if either blocks) in one on-device kernel,
//!   with no host round-trip.
//!
//! `true` in a mask means **blocked** ([`Tensor::causal_mask`]'s polarity):
//! masked positions are filled with `-inf` *before* the softmax, so they
//! receive exactly zero weight and no gradient. A row that is masked in full
//! yields zeros rather than `NaN` — [`Tensor::softmax`] handles that case.
//!
//! # KV-cache-ready shape handling
//!
//! The query and the key/value source are separate arguments with
//! **independent sequence lengths**, and the four projection halves are
//! public:
//! [`project_query`](MultiHeadAttention::project_query),
//! [`project_keys`](MultiHeadAttention::project_keys),
//! [`project_values`](MultiHeadAttention::project_values) and
//! [`project_output`](MultiHeadAttention::project_output) each expose the
//! head-split `[.., heads, seq, head_dim]` form. An incremental decoder
//! projects only the *new* token, concatenates the result onto its cached
//! keys and values, and calls [`scaled_dot_product_attention`] directly — no
//! re-projection of the prefix, and no cache type inside this layer (the
//! cache lives with the transformer model).

use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::nn::{Forward, Mode, Param};
use crate::rng::Rng;
use crate::shape::Shape;
use crate::tensor::Tensor;

/// Scaled dot-product attention: `softmax(q @ kᵀ / √head_dim + mask) @ v`.
///
/// Shapes, with `..` any number of batch axes that broadcast between the
/// operands (`[batch, heads]` after
/// [`project_query`](MultiHeadAttention::project_query), or nothing at all for
/// one bare head):
///
/// | operand | shape |
/// |---|---|
/// | `q` | `[.., q_len, head_dim]` |
/// | `k` | `[.., kv_len, head_dim]` |
/// | `v` | `[.., kv_len, value_dim]` |
/// | `mask` | broadcasts **into** `[.., q_len, kv_len]` |
/// | result | `[.., q_len, value_dim]` |
///
/// `q_len` and `kv_len` are independent: one query against a long cached
/// prefix is the decoding step, and the same call with `q_len == kv_len` is
/// ordinary self-attention.
///
/// The scale is `1/√head_dim` — the variance correction that keeps the logits
/// (and therefore the softmax gradient) from saturating as the head widens.
/// `mask` is additive `-inf` on the blocked positions, applied **after** the
/// scale and **before** the softmax, so a blocked key contributes no weight
/// and no gradient. `true` means blocked.
///
/// # Errors
///
/// - [`Error::InvalidArg`] (`op: "scaled_dot_product_attention"`) if an
///   operand has rank < 2, or if `q_len`, `kv_len` or `head_dim` is zero —
///   attending over nothing has no defined softmax, and a zero-width head
///   would divide the logits by zero.
/// - [`Error::ShapeMismatch`] if `q` and `k` disagree on `head_dim`, if `k`
///   and `v` disagree on `kv_len`, if the batch axes do not broadcast, or if
///   `mask` does not broadcast *into* the score shape (a mask that would
///   *grow* the scores is a bug, not a broadcast).
/// - [`Error::DTypeMismatch`] if `mask` is not [`Bool`](crate::DType::Bool) or
///   the value operands disagree, and [`Error::DeviceMismatch`] across
///   devices.
/// - [`Error::Unsupported`] for a dtype the backend cannot multiply.
///
/// ```
/// # use rstorch::nn::scaled_dot_product_attention;
/// # use rstorch::{DType, Device, Tensor};
/// # fn main() -> rstorch::Result<()> {
/// let dev = Device::Cpu;
/// // Two positions, one head of width 2; keys and values are the same length.
/// let q = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], [2, 2], &dev)?;
/// let v = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2], &dev)?;
/// // Causal: position 0 may only see itself, so it returns v's first row.
/// let mask = Tensor::causal_mask(2, &dev)?;
/// let out = scaled_dot_product_attention(&q, &q, &v, Some(&mask))?;
/// assert_eq!(out.dims(), &[2, 2]);
/// assert_eq!(&out.to_vec::<f32>()?[..2], &[1.0, 2.0]);
/// # Ok(())
/// # }
/// ```
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    const OP: &str = "scaled_dot_product_attention";
    for (name, t) in [("q", q), ("k", k), ("v", v)] {
        if t.rank() < 2 {
            return Err(Error::InvalidArg {
                op: OP,
                msg: format!(
                    "{name} must be rank >= 2 ([.., seq, dim]), got shape {}",
                    t.shape()
                ),
            });
        }
    }
    let (q_len, head_dim) = last_two(q);
    let (kv_len, key_dim) = last_two(k);
    if head_dim != key_dim {
        return Err(Error::ShapeMismatch {
            op: OP,
            lhs: q.shape().clone(),
            rhs: k.shape().clone(),
        });
    }
    if last_two(v).0 != kv_len {
        return Err(Error::ShapeMismatch {
            op: OP,
            lhs: k.shape().clone(),
            rhs: v.shape().clone(),
        });
    }
    if q_len == 0 || kv_len == 0 || head_dim == 0 {
        return Err(Error::InvalidArg {
            op: OP,
            msg: format!(
                "q_len, kv_len and head_dim must all be non-zero, got {q_len}, {kv_len} and \
                 {head_dim}"
            ),
        });
    }

    // `[.., q_len, head_dim] @ [.., head_dim, kv_len]` — the transpose is a
    // free layout change, so no copy of the keys is made.
    let scores = q
        .matmul(&k.transpose(-2, -1)?)?
        .div_scalar((head_dim as f64).sqrt())?;
    let scores = match mask {
        Some(mask) => {
            check_mask(OP, mask, &scores)?;
            scores.masked_fill(mask, f64::NEG_INFINITY)?
        }
        None => scores,
    };
    // The softmax is over the *key* axis: each query's weights sum to one.
    scores.softmax(-1)?.matmul(v)
}

/// The last two dimensions of a rank ≥ 2 tensor (`seq`, `dim`).
fn last_two(t: &Tensor) -> (usize, usize) {
    let dims = t.dims();
    (dims[dims.len() - 2], dims[dims.len() - 1])
}

/// Reject a mask that is not a `Bool` tensor on the right device, or that
/// would *grow* the scores instead of broadcasting into them.
fn check_mask(op: &'static str, mask: &Tensor, scores: &Tensor) -> Result<()> {
    if mask.dtype() != DType::Bool {
        return Err(Error::DTypeMismatch {
            op,
            expected: DType::Bool,
            got: mask.dtype(),
        });
    }
    if mask.device() != scores.device() {
        return Err(Error::DeviceMismatch {
            op,
            expected: scores.device(),
            got: mask.device(),
        });
    }
    let mismatch = || Error::ShapeMismatch {
        op,
        lhs: scores.shape().clone(),
        rhs: mask.shape().clone(),
    };
    if mask.rank() > scores.rank() {
        return Err(mismatch());
    }
    // Right-aligned, as broadcasting is: every mask axis is either 1 or the
    // score axis it lines up with.
    let offset = scores.rank() - mask.rank();
    for (axis, &m) in mask.dims().iter().enumerate() {
        if m != 1 && m != scores.dims()[offset + axis] {
            return Err(mismatch());
        }
    }
    Ok(())
}

/// The permutation that swaps the two axes before the last — the only axis
/// motion head splitting and merging need
/// (`[.., a, b, c]` ⇄ `[.., b, a, c]`, its own inverse).
fn swap_before_last(rank: usize) -> Vec<isize> {
    let mut perm: Vec<isize> = (0..rank as isize).collect();
    perm.swap(rank - 3, rank - 2);
    perm
}

/// `[.., seq, embed_dim]` → `[.., num_heads, seq, head_dim]`.
///
/// Shared with the typed wrapper, which splits heads over the same runtime
/// tensor before re-wrapping the result in its marker type.
pub(crate) fn split_heads(x: &Tensor, num_heads: usize, head_dim: usize) -> Result<Tensor> {
    let rank = x.rank();
    let mut dims = x.dims()[..rank - 1].to_vec();
    dims.push(num_heads);
    dims.push(head_dim);
    // [.., seq, heads, head_dim], then swap `seq` and `heads`.
    let split = x.reshape(dims)?;
    split.permute(&swap_before_last(rank + 1))
}

/// The inverse: `[.., num_heads, seq, head_dim]` → `[.., seq, embed_dim]`,
/// the form the output projection consumes. Shared with the typed wrapper.
pub(crate) fn merge_heads(context: &Tensor, embed_dim: usize) -> Result<Tensor> {
    let rank = context.rank();
    let dims = context.dims();
    // [.., heads, seq, head_dim] -> [.., seq, heads, head_dim] -> [.., seq, embed]
    let merged = context.permute(&swap_before_last(rank))?;
    let mut flat = dims[..rank - 3].to_vec();
    flat.push(dims[rank - 2]);
    flat.push(embed_dim);
    merged.reshape(flat)
}

/// One affine projection: `x @ weightᵀ + bias`.
///
/// The crate-private stand-in for `nn::Linear` (the
/// two tasks are parallel). The field names — `weight` shaped
/// `[out_features, in_features]`, optional `bias` shaped `[out_features]` —
/// are deliberately `PyTorch`'s, so the `state_dict` paths this emits
/// (`q_proj.weight`, `q_proj.bias`) are the same ones a `Linear`-based
/// implementation would, and swapping the two later moves no checkpoint keys.
#[derive(rstorch::Module)]
struct Proj {
    weight: Param,
    bias: Option<Param>,
}

impl Proj {
    /// Glorot/Xavier-uniform weights on `U(-a, a)`, `a = √(6 / (in + out))`,
    /// and zero bias — `PyTorch`'s `MultiheadAttention` initialization.
    fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
        device: &Device,
        rng: &mut Rng,
    ) -> Result<Proj> {
        let limit = (6.0 / (in_features + out_features) as f64).sqrt();
        let weight = Tensor::rand([out_features, in_features], DType::F32, device, rng)?
            .mul_scalar(2.0 * limit)?
            .sub_scalar(limit)?;
        let bias = if bias {
            Some(Param::new(Tensor::zeros(
                [out_features],
                DType::F32,
                device,
            )?))
        } else {
            None
        };
        Ok(Proj {
            weight: Param::new(weight),
            bias,
        })
    }

    /// `[.., in_features] -> [.., out_features]`.
    fn apply(&self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        let out = x.matmul(&self.weight.get(mode).transpose(-2, -1)?)?;
        match &self.bias {
            Some(bias) => out.add(&bias.get(mode)),
            None => Ok(out),
        }
    }
}

/// Multi-head attention over runtime dimensions, with the mask as an
/// argument.
///
/// Four projections — `q_proj`, `k_proj`, `v_proj`, `out_proj`, each
/// `[embed_dim, embed_dim]` with an optional bias — so the `state_dict` paths
/// are `q_proj.weight`, `q_proj.bias`, …, `out_proj.bias`. Those names are the
/// persistence contract.
///
/// The forward pass is deliberately spelled as composable steps rather than
/// one closed method: [`attend`](MultiHeadAttention::attend) is
/// self-attention, [`attend_to`](MultiHeadAttention::attend_to) is
/// cross-attention (and the cached decoding step, since the query and the
/// key/value source have independent lengths), and the `project_*` methods
/// are those two taken apart for a KV cache.
///
/// [`Forward`] is implemented for [`AttentionInput`], not for a bare
/// [`Tensor`]: the mask has no place in a one-tensor signature, and an
/// attention layer that silently picks its own masking policy —
/// bidirectional when the model needed causal — is a bug that trains to a
/// plausible-looking loss. The mask stays required, `None` spelled out,
/// whichever spelling you call.
///
/// ```
/// # use rstorch::nn::{MultiHeadAttention, Mode};
/// # use rstorch::{DType, Device, Rng, Tensor};
/// # fn main() -> rstorch::Result<()> {
/// let dev = Device::Cpu;
/// let mut rng = Rng::seed(3);
/// let attn = MultiHeadAttention::new(8, 2, &dev, &mut rng)?;
///
/// // Causal self-attention over a [batch, seq, embed] batch.
/// let x = Tensor::rand([2, 5, 8], DType::F32, &dev, &mut rng)?;
/// let mask = Tensor::causal_mask(5, &dev)?;
/// let out = attn.attend(&x, Some(&mask), Mode::TRAIN)?;
/// assert_eq!(out.dims(), &[2, 5, 8]);
///
/// // One new token attending over a five-token prefix (the decoding shape).
/// let step = Tensor::rand([2, 1, 8], DType::F32, &dev, &mut rng)?;
/// assert_eq!(attn.attend_to(&step, &x, None, Mode::EVAL)?.dims(), &[2, 1, 8]);
/// # Ok(())
/// # }
/// ```
#[derive(rstorch::Module)]
pub struct MultiHeadAttention {
    q_proj: Proj,
    k_proj: Proj,
    v_proj: Proj,
    out_proj: Proj,
    /// How many heads `embed_dim` is split into. Configuration, not state: a
    /// whitelisted primitive, so the derive never visits it.
    num_heads: usize,
}

impl MultiHeadAttention {
    /// Four `[embed_dim, embed_dim]` projections with biases (`PyTorch`'s
    /// default), Xavier-uniform weights drawn from `rng`, zero biases,
    /// [`F32`](crate::DType::F32) on `device`.
    ///
    /// Weights are created in `F32` and converted afterwards
    /// ([`ModuleExt::to_dtype`](crate::nn::ModuleExt::to_dtype)), like every other layer.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArg`] (`op: "MultiHeadAttention::new"`) if `embed_dim`
    /// or `num_heads` is zero, or if `num_heads` does not divide `embed_dim` —
    /// the heads partition the embedding exactly, and a remainder is a
    /// configuration bug rather than a truncation to absorb.
    pub fn new(
        embed_dim: usize,
        num_heads: usize,
        device: &Device,
        rng: &mut Rng,
    ) -> Result<MultiHeadAttention> {
        Self::build(
            "MultiHeadAttention::new",
            embed_dim,
            num_heads,
            true,
            device,
            rng,
        )
    }

    /// [`new`](MultiHeadAttention::new) with the four biases omitted — the
    /// usual transformer-LM recipe. The `bias` paths are simply absent from
    /// the `state_dict`.
    ///
    /// # Errors
    ///
    /// As [`new`](MultiHeadAttention::new), with
    /// `op: "MultiHeadAttention::new_without_bias"`.
    pub fn new_without_bias(
        embed_dim: usize,
        num_heads: usize,
        device: &Device,
        rng: &mut Rng,
    ) -> Result<MultiHeadAttention> {
        Self::build(
            "MultiHeadAttention::new_without_bias",
            embed_dim,
            num_heads,
            false,
            device,
            rng,
        )
    }

    fn build(
        op: &'static str,
        embed_dim: usize,
        num_heads: usize,
        bias: bool,
        device: &Device,
        rng: &mut Rng,
    ) -> Result<MultiHeadAttention> {
        if embed_dim == 0 || num_heads == 0 || !embed_dim.is_multiple_of(num_heads) {
            return Err(Error::InvalidArg {
                op,
                msg: format!(
                    "num_heads must be non-zero and divide a non-zero embed_dim, got \
                     embed_dim {embed_dim} and num_heads {num_heads}"
                ),
            });
        }
        // Four independent draws from the one stream, in q/k/v/out order: the
        // initialization is reproducible from the seed.
        let q_proj = Proj::new(embed_dim, embed_dim, bias, device, rng)?;
        let k_proj = Proj::new(embed_dim, embed_dim, bias, device, rng)?;
        let v_proj = Proj::new(embed_dim, embed_dim, bias, device, rng)?;
        let out_proj = Proj::new(embed_dim, embed_dim, bias, device, rng)?;
        Ok(MultiHeadAttention {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
        })
    }

    /// The model width — the size of the last input axis, and of the output's.
    pub fn embed_dim(&self) -> usize {
        self.q_proj.weight.value().dims()[0]
    }

    /// How many heads the embedding is split into.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Width of one head: `embed_dim / num_heads`.
    pub fn head_dim(&self) -> usize {
        self.embed_dim() / self.num_heads
    }

    /// Self-attention: `x` supplies the queries, the keys **and** the values.
    ///
    /// `x` is `[.., seq, embed_dim]` (any number of leading batch axes,
    /// including none) and the result has the same shape. `mask` broadcasts
    /// into `[.., num_heads, seq, seq]` — a bare `[seq, seq]`
    /// [`causal_mask`](Tensor::causal_mask) does, unchanged, for every batch
    /// and head.
    ///
    /// # Errors
    ///
    /// As [`attend_to`](MultiHeadAttention::attend_to).
    pub fn attend(&self, x: &Tensor, mask: Option<&Tensor>, mode: Mode) -> Result<Tensor> {
        self.attend_to(x, x, mask, mode)
    }

    /// Attention of `query` over the keys and values projected from
    /// `keys_values` — cross-attention, and the shape an incremental decoder
    /// wants (`query` one token, `keys_values` the whole prefix).
    ///
    /// `query` is `[.., q_len, embed_dim]`, `keys_values` is
    /// `[.., kv_len, embed_dim]`, and the result is
    /// `[.., q_len, embed_dim]`; the two lengths are independent and the
    /// batch axes must broadcast. `mask` broadcasts into
    /// `[.., num_heads, q_len, kv_len]` — note the **head axis**: a
    /// per-sample padding mask of `[batch, kv_len]` must be reshaped to
    /// `[batch, 1, 1, kv_len]` first, which is one
    /// [`reshape`](Tensor::reshape).
    ///
    /// A cached decoder should project once and keep the halves instead:
    /// see [`project_keys`](MultiHeadAttention::project_keys).
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeMismatch`] if either input's last axis is not
    ///   `embed_dim`, or as [`scaled_dot_product_attention`] for the mask and
    ///   the batch axes.
    /// - [`Error::InvalidArg`] if either input has rank < 2 or an empty
    ///   sequence axis.
    /// - [`Error::DTypeMismatch`] / [`Error::DeviceMismatch`] if the inputs,
    ///   the mask and the parameters do not agree.
    pub fn attend_to(
        &self,
        query: &Tensor,
        keys_values: &Tensor,
        mask: Option<&Tensor>,
        mode: Mode,
    ) -> Result<Tensor> {
        let q = self.project_query(query, mode)?;
        let k = self.project_keys(keys_values, mode)?;
        let v = self.project_values(keys_values, mode)?;
        self.project_output(&scaled_dot_product_attention(&q, &k, &v, mask)?, mode)
    }

    /// Project queries and split the heads: `[.., q_len, embed_dim]` →
    /// `[.., num_heads, q_len, head_dim]`, ready for
    /// [`scaled_dot_product_attention`].
    ///
    /// # Errors
    ///
    /// [`Error::ShapeMismatch`] if the last axis is not `embed_dim`,
    /// [`Error::InvalidArg`] if the input has rank < 2.
    pub fn project_query(&self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        const OP: &str = "MultiHeadAttention::project_query";
        self.split_heads(&self.project(OP, &self.q_proj, x, mode)?)
    }

    /// Project keys and split the heads: `[.., kv_len, embed_dim]` →
    /// `[.., num_heads, kv_len, head_dim]`.
    ///
    /// This is the half a KV cache stores: concatenate the new token's keys
    /// onto the cached ones along the **second-to-last** axis
    /// (`Tensor::cat(&[&cached, &new], -2)`) and pass the result straight to
    /// [`scaled_dot_product_attention`].
    ///
    /// # Errors
    ///
    /// As [`project_query`](MultiHeadAttention::project_query).
    pub fn project_keys(&self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        const OP: &str = "MultiHeadAttention::project_keys";
        self.split_heads(&self.project(OP, &self.k_proj, x, mode)?)
    }

    /// Project values and split the heads: `[.., kv_len, embed_dim]` →
    /// `[.., num_heads, kv_len, head_dim]`. The other half a KV cache stores
    /// (see [`project_keys`](MultiHeadAttention::project_keys)).
    ///
    /// # Errors
    ///
    /// As [`project_query`](MultiHeadAttention::project_query).
    pub fn project_values(&self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        const OP: &str = "MultiHeadAttention::project_values";
        self.split_heads(&self.project(OP, &self.v_proj, x, mode)?)
    }

    /// Both cached halves of one key/value source in one call — the pair a
    /// decoder appends to its cache each step.
    ///
    /// # Errors
    ///
    /// As [`project_query`](MultiHeadAttention::project_query).
    pub fn project_keys_values(&self, x: &Tensor, mode: Mode) -> Result<(Tensor, Tensor)> {
        Ok((self.project_keys(x, mode)?, self.project_values(x, mode)?))
    }

    /// Merge the heads of an attention result and apply the output
    /// projection: `[.., num_heads, q_len, head_dim]` →
    /// `[.., q_len, embed_dim]`.
    ///
    /// The inverse of [`project_query`](MultiHeadAttention::project_query)'s
    /// head split, plus `out_proj`. The last step of a cached decoding step.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArg`] if `context` has rank < 3, and
    /// [`Error::ShapeMismatch`] if its head axis is not `num_heads` or its
    /// last axis is not `head_dim`.
    pub fn project_output(&self, context: &Tensor, mode: Mode) -> Result<Tensor> {
        const OP: &str = "MultiHeadAttention::project_output";
        let rank = context.rank();
        if rank < 3 {
            return Err(Error::InvalidArg {
                op: OP,
                msg: format!(
                    "context must be rank >= 3 ([.., heads, seq, head_dim]), got shape {}",
                    context.shape()
                ),
            });
        }
        let dims = context.dims();
        if dims[rank - 3] != self.num_heads || dims[rank - 1] != self.head_dim() {
            return Err(Error::ShapeMismatch {
                op: OP,
                lhs: Shape::from(vec![self.num_heads, dims[rank - 2], self.head_dim()]),
                rhs: context.shape().clone(),
            });
        }
        let merged = merge_heads(context, self.embed_dim())?;
        self.out_proj.apply(&merged, mode)
    }

    /// One projection with the input's last axis checked against `embed_dim`
    /// first, so the error names this layer rather than `matmul`.
    fn project(&self, op: &'static str, proj: &Proj, x: &Tensor, mode: Mode) -> Result<Tensor> {
        if x.rank() < 2 {
            return Err(Error::InvalidArg {
                op,
                msg: format!(
                    "input must be rank >= 2 ([.., seq, embed_dim]), got shape {}",
                    x.shape()
                ),
            });
        }
        let (seq, embed) = last_two(x);
        if embed != self.embed_dim() {
            return Err(Error::ShapeMismatch {
                op,
                lhs: Shape::from(vec![seq, self.embed_dim()]),
                rhs: x.shape().clone(),
            });
        }
        proj.apply(x, mode)
    }

    /// `[.., seq, embed_dim]` → `[.., num_heads, seq, head_dim]`.
    fn split_heads(&self, x: &Tensor) -> Result<Tensor> {
        split_heads(x, self.num_heads, self.head_dim())
    }
}

/// The [`Forward`] input of [`MultiHeadAttention`]: the sequence and its mask,
/// in one value.
///
/// This is what makes a multi-input layer fit the crate's central trait
/// without smuggling the mask through `&mut self`. The mask stays **required**
/// — `None` has to be spelled out — so a layer can never silently pick its own
/// masking policy, which is the bug that trains to a plausible-looking loss.
///
/// The fields are owned rather than borrowed because [`Forward::forward`]
/// takes `&Input`: a caller builds one of these per call from tensors it
/// already holds, and a `Tensor` clone is an `Arc` bump.
///
/// `#[non_exhaustive]` with a constructor, for the same reason
/// [`TransformerConfig`](crate::models::TransformerConfig) is: a struct
/// literal is the only downstream construction path a bare `pub`-field struct
/// offers, so adding a field in a 1.x release would break every caller. Sealed
/// and constructed through [`AttentionInput::new`], the shape can still grow —
/// a cross-attention memory tensor is the obvious addition, and today it is
/// reachable only through [`attend_to`](MultiHeadAttention::attend_to).
///
/// ```
/// # use rstorch::nn::{AttentionInput, Forward, Mode, MultiHeadAttention};
/// # use rstorch::{DType, Device, Rng, Tensor};
/// # fn main() -> rstorch::Result<()> {
/// let dev = Device::Cpu;
/// let mut rng = Rng::seed(3);
/// let mut attn = MultiHeadAttention::new(8, 2, &dev, &mut rng)?;
/// let x = Tensor::rand([2, 5, 8], DType::F32, &dev, &mut rng)?;
/// let input = AttentionInput::new(x, Some(Tensor::causal_mask(5, &dev)?));
/// assert_eq!(attn.forward(&input, Mode::TRAIN)?.dims(), &[2, 5, 8]);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AttentionInput {
    /// The sequence, `[.., seq, embed_dim]`. Supplies the queries, the keys
    /// **and** the values (self-attention); for cross-attention call
    /// [`attend_to`](MultiHeadAttention::attend_to) directly.
    pub x: Tensor,
    /// The mask, broadcasting into `[.., num_heads, seq, seq]`. `true` means
    /// **blocked** ([`Tensor::causal_mask`]'s polarity).
    pub mask: Option<Tensor>,
}

impl AttentionInput {
    /// One self-attention call's input. `mask` is positional rather than
    /// defaulted so it cannot be forgotten — see the type's own docs.
    #[must_use]
    pub fn new(x: Tensor, mask: Option<Tensor>) -> AttentionInput {
        AttentionInput { x, mask }
    }
}

impl Forward<AttentionInput> for MultiHeadAttention {
    type Output = Tensor;

    /// [`attend`](MultiHeadAttention::attend) over
    /// [`AttentionInput::x`] with [`AttentionInput::mask`].
    ///
    /// `attend`/`attend_to` remain the ergonomic spelling and share this
    /// implementation; this impl exists so the layer is reachable through the
    /// trait system (and therefore through generic code written against
    /// [`Forward`]).
    ///
    /// # Errors
    ///
    /// As [`attend`](MultiHeadAttention::attend).
    fn forward(&mut self, input: &AttentionInput, mode: Mode) -> Result<Tensor> {
        self.attend(&input.x, input.mask.as_ref(), mode)
    }
}

#[cfg(test)]
mod tests;
