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
use crate::nn::{Mode, Param};
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
/// are deliberately PyTorch's, so the `state_dict` paths this emits
/// (`q_proj.weight`, `q_proj.bias`) are the same ones a `Linear`-based
/// implementation would, and swapping the two later moves no checkpoint keys.
#[derive(rstorch::Module)]
struct Proj {
    weight: Param,
    bias: Option<Param>,
}

impl Proj {
    /// Glorot/Xavier-uniform weights on `U(-a, a)`, `a = √(6 / (in + out))`,
    /// and zero bias — PyTorch's `MultiheadAttention` initialization.
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
/// `forward` is *not* implemented: [`Forward`](crate::nn::Forward) has no
/// place for the mask, and an attention layer that silently picks its own
/// masking policy — bidirectional when the model needed causal — is a bug
/// that trains to a plausible-looking loss. The mask stays a required
/// argument, `None` spelled out.
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
    /// Four `[embed_dim, embed_dim]` projections with biases (PyTorch's
    /// default), Xavier-uniform weights drawn from `rng`, zero biases,
    /// [`F32`](crate::DType::F32) on `device`.
    ///
    /// Weights are created in `F32` and converted afterwards
    /// ([`nn::to_dtype`](crate::nn::to_dtype)), like every other layer.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn;
    use crate::testing::check_grad;

    const CPU: Device = Device::Cpu;

    fn seq(n: usize) -> Vec<f32> {
        // A deterministic, non-symmetric spread in [-1, 1): distinct values,
        // so no accidental cancellation hides a wrong axis.
        (0..n)
            .map(|i| ((i as f32 * 0.37).sin() * 0.9 + (i as f32) * 0.011).clamp(-1.0, 1.0))
            .collect()
    }

    fn t(shape: &[usize]) -> Tensor {
        Tensor::from_vec(seq(shape.iter().product()), shape.to_vec(), &CPU).unwrap()
    }

    fn v(t: &Tensor) -> Vec<f32> {
        t.to_vec::<f32>().unwrap()
    }

    fn mha(embed: usize, heads: usize) -> MultiHeadAttention {
        MultiHeadAttention::new(embed, heads, &CPU, &mut Rng::seed(17)).unwrap()
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length");
        for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() <= tol * (1.0 + x.abs().max(y.abs())),
                "element {i}: {x} vs {y}"
            );
        }
    }

    /// A fixed non-uniform weighting, so every output element contributes to
    /// the scalar objective the gradient tests differentiate.
    fn coefficients(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor::from_vec(
            (0..n).map(|i| 0.3 + i as f32 * 0.17).collect(),
            shape.to_vec(),
            &CPU,
        )
        .unwrap()
    }

    // ------------------------------------------------------------------
    // scaled_dot_product_attention
    // ------------------------------------------------------------------

    #[test]
    fn sdpa_shapes_are_query_length_by_value_width() {
        // Batch axes broadcast; q_len and kv_len are independent.
        let q = t(&[2, 3, 4, 5]); // [batch, heads, q_len, head_dim]
        let k = t(&[2, 3, 7, 5]); // [batch, heads, kv_len, head_dim]
        let val = t(&[2, 3, 7, 6]); // a value width of its own
        let out = scaled_dot_product_attention(&q, &k, &val, None).unwrap();
        assert_eq!(out.dims(), &[2, 3, 4, 6]);

        // Unbatched: one bare head.
        let out =
            scaled_dot_product_attention(&t(&[4, 5]), &t(&[7, 5]), &t(&[7, 5]), None).unwrap();
        assert_eq!(out.dims(), &[4, 5]);

        // One query against a cached prefix — the decoding step.
        let out = scaled_dot_product_attention(&t(&[2, 3, 1, 5]), &k, &val, None).unwrap();
        assert_eq!(out.dims(), &[2, 3, 1, 6]);
    }

    #[test]
    fn sdpa_is_the_scaled_softmax_weighted_average_of_the_values() {
        // Hand-rolled on paper: two queries, two keys of width 4.
        let q = t(&[2, 4]);
        let k = t(&[2, 4]);
        let val = t(&[2, 3]);
        let out = scaled_dot_product_attention(&q, &k, &val, None).unwrap();

        let (qv, kv, vv) = (v(&q), v(&k), v(&val));
        let scale = 4f32.sqrt();
        let mut expected = vec![0.0f32; 2 * 3];
        for i in 0..2 {
            let logits: Vec<f32> = (0..2)
                .map(|j| (0..4).map(|d| qv[i * 4 + d] * kv[j * 4 + d]).sum::<f32>() / scale)
                .collect();
            let peak = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = logits.iter().map(|l| (l - peak).exp()).collect();
            let denom: f32 = e.iter().sum();
            for c in 0..3 {
                expected[i * 3 + c] = (0..2).map(|j| e[j] / denom * vv[j * 3 + c]).sum();
            }
        }
        assert_close(&v(&out), &expected, 1e-6);
    }

    #[test]
    fn attention_weights_are_a_distribution_over_the_keys() {
        // Attending with the identity as values recovers the weights, whose
        // rows must each sum to one.
        let identity = Tensor::from_vec(
            (0..9).map(|i| f32::from(i % 4 == 0)).collect::<Vec<f32>>(),
            [3, 3],
            &CPU,
        )
        .unwrap();
        let weights =
            scaled_dot_product_attention(&t(&[3, 4]), &t(&[3, 4]), &identity, None).unwrap();
        for row in v(&weights).chunks(3) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6, "{row:?}");
        }
    }

    #[test]
    fn a_masked_key_gets_exactly_zero_weight() {
        // Blocking key 0 for every query must reproduce attention over the
        // remaining keys alone.
        let (q, k) = (t(&[2, 4]), t(&[3, 4]));
        let val = t(&[3, 5]);
        let blocked = Tensor::from_vec(vec![true, false, false], [1, 3], &CPU).unwrap();
        let masked = scaled_dot_product_attention(&q, &k, &val, Some(&blocked)).unwrap();

        let without = scaled_dot_product_attention(
            &q,
            &k.narrow(0, 1, 2).unwrap(),
            &val.narrow(0, 1, 2).unwrap(),
            None,
        )
        .unwrap();
        assert_close(&v(&masked), &v(&without), 1e-6);
    }

    #[test]
    fn a_fully_masked_row_is_zero_rather_than_nan() {
        // The degenerate case softmax is built to survive: a query with
        // no visible key at all. It must not poison the batch.
        let mask = Tensor::from_vec(vec![true, true, false, false], [2, 2], &CPU).unwrap();
        let out = scaled_dot_product_attention(&t(&[2, 3]), &t(&[2, 3]), &t(&[2, 3]), Some(&mask))
            .unwrap();
        let out = v(&out);
        assert_eq!(&out[0..3], &[0.0, 0.0, 0.0]);
        assert!(out.iter().all(|x| !x.is_nan()), "{out:?}");
        assert!(out[3..].iter().any(|x| *x != 0.0));
    }

    #[test]
    fn a_causal_mask_hides_the_future_from_every_position() {
        // The proof the gate asks for: perturbing position t+1 of the keys and
        // values cannot move the output at position t, while perturbing
        // position t-1 does.
        let (seq_len, dim) = (4, 3);
        let x = t(&[seq_len, dim]);
        let mask = Tensor::causal_mask(seq_len, &CPU).unwrap();
        let base = v(&scaled_dot_product_attention(&x, &x, &x, Some(&mask)).unwrap());

        for perturbed in 0..seq_len {
            let mut values = v(&x);
            values[perturbed * dim..(perturbed + 1) * dim]
                .iter_mut()
                .for_each(|value| *value += 10.0);
            let y = Tensor::from_vec(values, [seq_len, dim], &CPU).unwrap();
            let moved = v(&scaled_dot_product_attention(&y, &y, &y, Some(&mask)).unwrap());

            for query in 0..seq_len {
                let row = query * dim..(query + 1) * dim;
                let changed = base[row.clone()]
                    .iter()
                    .zip(&moved[row])
                    .any(|(a, b)| (a - b).abs() > 1e-6);
                if query < perturbed {
                    assert!(
                        !changed,
                        "query {query} saw the future at position {perturbed}"
                    );
                } else {
                    assert!(changed, "query {query} ignored position {perturbed}");
                }
            }
        }
    }

    #[test]
    fn sdpa_gradients_match_finite_differences() {
        // Unmasked, batched over two heads.
        let coef = coefficients(&[2, 3, 5]);
        let f = |xs: &[Tensor]| {
            scaled_dot_product_attention(&xs[0], &xs[1], &xs[2], None)?
                .mul(&coef)?
                .sum_all()
        };
        check_grad(
            f,
            &[t(&[2, 3, 4]), t(&[2, 4, 4]), t(&[2, 4, 5])],
            1e-2,
            1e-3,
        )
        .unwrap();

        // Masked: the causal case, where some scores are -inf. The cotangent
        // must not flow through them (and must not become NaN).
        let mask = Tensor::causal_mask(3, &CPU).unwrap();
        let coef = coefficients(&[2, 3, 4]);
        let g = move |xs: &[Tensor]| {
            scaled_dot_product_attention(&xs[0], &xs[1], &xs[2], Some(&mask))?
                .mul(&coef)?
                .sum_all()
        };
        check_grad(
            g,
            &[t(&[2, 3, 4]), t(&[2, 3, 4]), t(&[2, 3, 4])],
            1e-2,
            1e-3,
        )
        .unwrap();
    }

    #[test]
    fn sdpa_rejects_shapes_that_do_not_line_up() {
        let ok = t(&[3, 4]);
        // head_dim disagreement between q and k.
        assert!(matches!(
            scaled_dot_product_attention(&ok, &t(&[3, 5]), &ok, None),
            Err(Error::ShapeMismatch {
                op: "scaled_dot_product_attention",
                ..
            })
        ));
        // kv_len disagreement between k and v.
        assert!(matches!(
            scaled_dot_product_attention(&ok, &ok, &t(&[2, 4]), None),
            Err(Error::ShapeMismatch { .. })
        ));
        // Rank 1 is not a sequence of vectors.
        assert!(matches!(
            scaled_dot_product_attention(&t(&[4]), &ok, &ok, None),
            Err(Error::InvalidArg {
                op: "scaled_dot_product_attention",
                ..
            })
        ));
        // An empty axis has no softmax.
        let empty = Tensor::zeros([0, 4], DType::F32, &CPU).unwrap();
        assert!(matches!(
            scaled_dot_product_attention(&empty, &ok, &ok, None),
            Err(Error::InvalidArg { .. })
        ));
    }

    #[test]
    fn sdpa_rejects_a_mask_that_is_not_a_broadcastable_bool() {
        let x = t(&[3, 4]);
        let floats = t(&[3, 3]);
        assert!(matches!(
            scaled_dot_product_attention(&x, &x, &x, Some(&floats)),
            Err(Error::DTypeMismatch {
                op: "scaled_dot_product_attention",
                expected: DType::Bool,
                got: DType::F32
            })
        ));
        // A mask that would *grow* the scores is a bug, not a broadcast.
        let too_wide = Tensor::causal_mask(5, &CPU).unwrap();
        assert!(matches!(
            scaled_dot_product_attention(&x, &x, &x, Some(&too_wide)),
            Err(Error::ShapeMismatch {
                op: "scaled_dot_product_attention",
                ..
            })
        ));
        // …and so is one of higher rank than the scores.
        let too_deep = Tensor::causal_mask(3, &CPU)
            .unwrap()
            .reshape([1, 3, 3])
            .unwrap();
        assert!(matches!(
            scaled_dot_product_attention(&x, &x, &x, Some(&too_deep)),
            Err(Error::ShapeMismatch { .. })
        ));
    }

    // ------------------------------------------------------------------
    // MultiHeadAttention: shapes
    // ------------------------------------------------------------------

    #[test]
    fn attend_preserves_the_input_shape_across_batch_heads_and_seq() {
        for &heads in &[1usize, 2, 4, 8] {
            let attn = mha(8, heads);
            assert_eq!(attn.num_heads(), heads);
            assert_eq!(attn.head_dim(), 8 / heads);
            for shape in [vec![3usize, 8], vec![2, 3, 8], vec![2, 3, 5, 8]] {
                let out = attn.attend(&t(&shape), None, Mode::EVAL).unwrap();
                assert_eq!(out.dims(), &shape[..], "heads {heads}, shape {shape:?}");
            }
        }
    }

    #[test]
    fn attend_to_takes_independent_query_and_key_lengths() {
        let attn = mha(6, 3);
        // Cross-attention: 2 queries over 5 keys.
        let out = attn
            .attend_to(&t(&[2, 2, 6]), &t(&[2, 5, 6]), None, Mode::EVAL)
            .unwrap();
        assert_eq!(out.dims(), &[2, 2, 6]);

        // One new token over a cached prefix.
        let step = attn
            .attend_to(&t(&[2, 1, 6]), &t(&[2, 5, 6]), None, Mode::EVAL)
            .unwrap();
        assert_eq!(step.dims(), &[2, 1, 6]);
    }

    #[test]
    fn the_projection_halves_round_trip_the_head_split() {
        let attn = mha(6, 3);
        let x = t(&[2, 5, 6]);
        let q = attn.project_query(&x, Mode::EVAL).unwrap();
        assert_eq!(q.dims(), &[2, 3, 5, 2]);
        let (k, val) = attn.project_keys_values(&x, Mode::EVAL).unwrap();
        assert_eq!(k.dims(), &[2, 3, 5, 2]);
        assert_eq!(v(&val), v(&attn.project_values(&x, Mode::EVAL).unwrap()));

        // Assembling the halves by hand reproduces `attend` exactly.
        let context = scaled_dot_product_attention(&q, &k, &val, None).unwrap();
        let out = attn.project_output(&context, Mode::EVAL).unwrap();
        assert_eq!(v(&out), v(&attn.attend(&x, None, Mode::EVAL).unwrap()));
    }

    #[test]
    fn the_head_split_partitions_the_embedding_contiguously() {
        // The ordering no shape assertion can catch: head `h` of position `s`
        // must be the slice `[h * head_dim .. (h + 1) * head_dim]` of the
        // *unsplit* projection — PyTorch's
        // `reshape(batch, seq, heads, head_dim).transpose(1, 2)`. A permute
        // written the other way round has the same shape and different values.
        let attn = mha(6, 3);
        let (heads, head_dim, seq_len) = (3, 2, 4);
        let x = t(&[1, seq_len, 6]);
        let flat = v(&attn.q_proj.apply(&x, Mode::EVAL).unwrap());
        let split = v(&attn.project_query(&x, Mode::EVAL).unwrap());

        for h in 0..heads {
            for s in 0..seq_len {
                for d in 0..head_dim {
                    let from_split = split[(h * seq_len + s) * head_dim + d];
                    let from_flat = flat[s * (heads * head_dim) + h * head_dim + d];
                    assert_eq!(from_split, from_flat, "head {h}, position {s}, lane {d}");
                }
            }
        }
    }

    #[test]
    fn attending_over_a_concatenated_cache_equals_attending_over_the_whole_prefix() {
        // The KV-cache invariant, without a cache type: keys projected token by
        // token and concatenated are the keys projected in one go.
        let attn = mha(4, 2);
        let prefix = t(&[1, 3, 4]);
        let step = t(&[1, 1, 4]);
        let whole = Tensor::cat(&[&prefix, &step], 1).unwrap();

        let cached_k = Tensor::cat(
            &[
                &attn.project_keys(&prefix, Mode::EVAL).unwrap(),
                &attn.project_keys(&step, Mode::EVAL).unwrap(),
            ],
            -2,
        )
        .unwrap();
        let cached_v = Tensor::cat(
            &[
                &attn.project_values(&prefix, Mode::EVAL).unwrap(),
                &attn.project_values(&step, Mode::EVAL).unwrap(),
            ],
            -2,
        )
        .unwrap();
        let q = attn.project_query(&step, Mode::EVAL).unwrap();
        let cached = attn
            .project_output(
                &scaled_dot_product_attention(&q, &cached_k, &cached_v, None).unwrap(),
                Mode::EVAL,
            )
            .unwrap();

        // The same last position, attending over the same four tokens.
        let full = attn.attend(&whole, None, Mode::EVAL).unwrap();
        assert_close(&v(&cached), &v(&full.narrow(1, 3, 1).unwrap()), 1e-6);
    }

    // ------------------------------------------------------------------
    // MultiHeadAttention: the single-head equivalence
    // ------------------------------------------------------------------

    /// Single-head attention written out with plain tensors: the reference the
    /// module is measured against (and, in the gradient test below, the
    /// function finite differences are taken of).
    ///
    /// `xs` is `[x, w_q, w_k, w_v, w_out]`; biases are omitted, which is why
    /// the equivalence test uses `new_without_bias`.
    fn hand_rolled_single_head(xs: &[Tensor], mask: Option<&Tensor>) -> Result<Tensor> {
        let (x, embed) = (&xs[0], xs[0].dims()[xs[0].rank() - 1]);
        let project = |w: &Tensor| x.matmul(&w.transpose(0, 1)?);
        let (q, k, val) = (project(&xs[1])?, project(&xs[2])?, project(&xs[3])?);
        let scores = q
            .matmul(&k.transpose(-2, -1)?)?
            .div_scalar((embed as f64).sqrt())?;
        let scores = match mask {
            Some(mask) => scores.masked_fill(mask, f64::NEG_INFINITY)?,
            None => scores,
        };
        scores
            .softmax(-1)?
            .matmul(&val)?
            .matmul(&xs[4].transpose(0, 1)?)
    }

    /// `[x, w_q, w_k, w_v, w_out]` for a bias-free single-head module.
    fn single_head_inputs(attn: &MultiHeadAttention, x: &Tensor) -> Vec<Tensor> {
        vec![
            x.clone(),
            attn.q_proj.weight.value().clone(),
            attn.k_proj.weight.value().clone(),
            attn.v_proj.weight.value().clone(),
            attn.out_proj.weight.value().clone(),
        ]
    }

    #[test]
    fn single_head_attention_matches_a_hand_rolled_computation() {
        let attn = MultiHeadAttention::new_without_bias(4, 1, &CPU, &mut Rng::seed(5)).unwrap();
        let x = t(&[2, 3, 4]);
        let mask = Tensor::causal_mask(3, &CPU).unwrap();

        for m in [None, Some(&mask)] {
            let expected = hand_rolled_single_head(&single_head_inputs(&attn, &x), m).unwrap();
            let actual = attn.attend(&x, m, Mode::EVAL).unwrap();
            assert_eq!(actual.dims(), expected.dims());
            assert_close(&v(&actual), &v(&expected), 1e-6);
        }
    }

    // ------------------------------------------------------------------
    // MultiHeadAttention: gradients
    // ------------------------------------------------------------------

    #[test]
    fn attend_gradients_wrt_the_input_match_finite_differences() {
        let attn = mha(4, 2);
        let mask = Tensor::causal_mask(3, &CPU).unwrap();
        let coef = coefficients(&[2, 3, 4]);
        let f = |xs: &[Tensor]| {
            attn.attend(&xs[0], Some(&mask), Mode::TRAIN)?
                .mul(&coef)?
                .sum_all()
        };
        check_grad(f, &[t(&[2, 3, 4])], 1e-2, 1e-3).unwrap();
    }

    #[test]
    fn parameter_gradients_are_the_finite_differences_of_the_same_function() {
        // Two chained claims. First: the hand-rolled single-head function
        // agrees with central finite differences in *all five* of its inputs,
        // including the four projection weights.
        let attn = MultiHeadAttention::new_without_bias(4, 1, &CPU, &mut Rng::seed(9)).unwrap();
        let x = t(&[2, 3, 4]);
        let mask = Tensor::causal_mask(3, &CPU).unwrap();
        let coef = coefficients(&[2, 3, 4]);
        let inputs = single_head_inputs(&attn, &x);

        let objective = {
            let (mask, coef) = (mask.clone(), coef.clone());
            move |xs: &[Tensor]| {
                hand_rolled_single_head(xs, Some(&mask))?
                    .mul(&coef)?
                    .sum_all()
            }
        };
        check_grad(&objective, &inputs, 1e-2, 1e-3).unwrap();

        // Second: the module's own parameter gradients equal that function's,
        // weight for weight — so the finite-difference evidence carries over to
        // the `Param` path.
        let traced: Vec<Tensor> = inputs.iter().map(|t| t.traced().unwrap()).collect();
        let reference = objective(&traced).unwrap().backward().unwrap();

        let loss = attn
            .attend(&x, Some(&mask), Mode::TRAIN)
            .unwrap()
            .mul(&coef)
            .unwrap()
            .sum_all()
            .unwrap();
        let grads = loss.backward().unwrap();
        for (i, param) in [
            &attn.q_proj.weight,
            &attn.k_proj.weight,
            &attn.v_proj.weight,
            &attn.out_proj.weight,
        ]
        .into_iter()
        .enumerate()
        {
            let expected = reference.wrt_input(&traced[i + 1]).unwrap();
            let actual = grads.wrt(param).unwrap();
            assert_eq!(actual.dims(), expected.dims());
            assert_close(&v(&actual), &v(&expected), 1e-5);
        }
    }

    #[test]
    fn every_parameter_receives_a_gradient() {
        // The completeness property the optimizer enforces: with biases
        // on and several heads, all eight leaves are reached.
        let attn = mha(6, 3);
        let loss = attn
            .attend(&t(&[2, 4, 6]), None, Mode::TRAIN)
            .unwrap()
            .mul(&coefficients(&[2, 4, 6]))
            .unwrap()
            .sum_all()
            .unwrap();
        let grads = loss.backward().unwrap();
        assert_eq!(grads.len(), 8);
        for proj in [&attn.q_proj, &attn.k_proj, &attn.v_proj, &attn.out_proj] {
            for p in [Some(&proj.weight), proj.bias.as_ref()]
                .into_iter()
                .flatten()
            {
                let g = grads.wrt(p).unwrap();
                assert_eq!(g.dims(), p.value().dims());
                assert!(v(&g).iter().any(|x| *x != 0.0), "an all-zero gradient");
            }
        }
    }

    #[test]
    fn the_layer_trains_under_plain_gradient_descent() {
        // End to end, without an optimizer: 60 hand-written SGD steps over
        // every visited parameter must drive a real objective down. This is the
        // property all the gradient algebra exists for, and the one a
        // sign error or a mis-scaled head would break while every shape test
        // still passed.
        use crate::nn::visit::{LeafMut, visit_all_mut};

        let mut attn = mha(4, 2);
        let x = t(&[1, 4, 4]);
        let target = coefficients(&[1, 4, 4]).mul_scalar(0.05).unwrap();
        let mask = Tensor::causal_mask(4, &CPU).unwrap();

        let loss_now = |attn: &MultiHeadAttention| {
            attn.attend(&x, Some(&mask), Mode::TRAIN)
                .unwrap()
                .mse_loss(&target)
                .unwrap()
        };

        let first = loss_now(&attn).item().unwrap();
        for _ in 0..60 {
            let grads = loss_now(&attn).backward().unwrap();
            visit_all_mut(&mut attn, &mut |path, leaf| {
                let LeafMut::Param(p) = leaf else {
                    panic!("attention has no buffers, yet {path} is one");
                };
                let step = grads.wrt(p).unwrap().mul_scalar(0.5).unwrap();
                p.set(p.value().sub(&step).unwrap()).unwrap();
            });
        }
        let last = loss_now(&attn).item().unwrap();
        assert!(
            last < first * 0.1,
            "loss barely moved: {first} -> {last} in 60 steps"
        );
    }

    #[test]
    fn eval_records_nothing_and_agrees_with_train_on_the_values() {
        let attn = mha(4, 2);
        let x = t(&[1, 3, 4]);
        let train = attn.attend(&x, None, Mode::TRAIN).unwrap();
        let eval = attn.attend(&x, None, Mode::EVAL).unwrap();
        assert_eq!(v(&train), v(&eval));
        assert!(matches!(
            eval.sum_all().unwrap().backward(),
            Err(Error::NotTraced { .. })
        ));
    }

    // ------------------------------------------------------------------
    // MultiHeadAttention: module wiring and loud errors
    // ------------------------------------------------------------------

    #[test]
    fn the_state_dict_paths_are_the_four_projections() {
        let attn = mha(4, 2);
        assert_eq!(
            nn::state_dict(&attn).keys().collect::<Vec<_>>(),
            [
                "k_proj.bias",
                "k_proj.weight",
                "out_proj.bias",
                "out_proj.weight",
                "q_proj.bias",
                "q_proj.weight",
                "v_proj.bias",
                "v_proj.weight",
            ]
        );
        assert_eq!(nn::num_params(&attn), 4 * (4 * 4 + 4));

        // Bias-free: the `bias` paths are simply absent.
        let bare = MultiHeadAttention::new_without_bias(4, 2, &CPU, &mut Rng::seed(1)).unwrap();
        assert_eq!(nn::state_dict(&bare).len(), 4);
        assert_eq!(nn::num_params(&bare), 4 * 4 * 4);
    }

    #[test]
    fn a_checkpoint_round_trip_reproduces_the_outputs() {
        let trained = mha(4, 2);
        let mut fresh = MultiHeadAttention::new(4, 2, &CPU, &mut Rng::seed(999)).unwrap();
        let x = t(&[1, 3, 4]);
        assert_ne!(
            v(&trained.attend(&x, None, Mode::EVAL).unwrap()),
            v(&fresh.attend(&x, None, Mode::EVAL).unwrap())
        );
        nn::load_state_dict(&mut fresh, &nn::state_dict(&trained)).unwrap();
        assert_eq!(
            v(&trained.attend(&x, None, Mode::EVAL).unwrap()),
            v(&fresh.attend(&x, None, Mode::EVAL).unwrap())
        );
    }

    #[test]
    fn a_padding_mask_composes_with_a_causal_one() {
        // The recipe this module's docs promise: two blocked-position masks
        // OR-ed on device, no host round-trip. Key 2 is padding for the one
        // sample here, so no query may attend to it.
        let attn = mha(4, 2);
        let x = t(&[1, 3, 4]);
        let causal = Tensor::causal_mask(3, &CPU).unwrap();
        // [batch, kv_len] reshaped to broadcast over heads and queries.
        let padding = Tensor::from_vec(vec![false, false, true], [1, 1, 1, 3], &CPU).unwrap();
        let combined = causal.where_cond(&causal, &padding).unwrap();
        assert_eq!(combined.dims(), &[1, 1, 3, 3]);

        let masked = attn.attend(&x, Some(&combined), Mode::EVAL).unwrap();
        // Dropping the padded token entirely must give the same first two
        // positions, since nothing was allowed to see it.
        let short = attn
            .attend(
                &x.narrow(1, 0, 2).unwrap(),
                Some(&Tensor::causal_mask(2, &CPU).unwrap()),
                Mode::EVAL,
            )
            .unwrap();
        assert_close(&v(&masked)[..8], &v(&short), 1e-6);
    }

    #[test]
    fn a_head_count_that_does_not_divide_the_embedding_is_loud() {
        let mut rng = Rng::seed(1);
        for (embed, heads) in [(6usize, 4usize), (0, 1), (4, 0)] {
            assert!(
                matches!(
                    MultiHeadAttention::new(embed, heads, &CPU, &mut rng),
                    Err(Error::InvalidArg {
                        op: "MultiHeadAttention::new",
                        ..
                    })
                ),
                "embed {embed}, heads {heads}"
            );
        }
        assert!(matches!(
            MultiHeadAttention::new_without_bias(6, 4, &CPU, &mut rng),
            Err(Error::InvalidArg {
                op: "MultiHeadAttention::new_without_bias",
                ..
            })
        ));
    }

    #[test]
    fn a_wrongly_shaped_input_names_this_layer_not_matmul() {
        let attn = mha(4, 2);
        assert!(matches!(
            attn.attend(&t(&[3, 5]), None, Mode::EVAL),
            Err(Error::ShapeMismatch {
                op: "MultiHeadAttention::project_query",
                ..
            })
        ));
        assert!(matches!(
            attn.attend_to(&t(&[3, 4]), &t(&[3, 5]), None, Mode::EVAL),
            Err(Error::ShapeMismatch {
                op: "MultiHeadAttention::project_keys",
                ..
            })
        ));
        assert!(matches!(
            attn.project_query(&t(&[4]), Mode::EVAL),
            Err(Error::InvalidArg {
                op: "MultiHeadAttention::project_query",
                ..
            })
        ));
    }

    #[test]
    fn project_output_checks_the_head_axes() {
        let attn = mha(4, 2);
        // Rank 2 has no head axis at all.
        assert!(matches!(
            attn.project_output(&t(&[3, 2]), Mode::EVAL),
            Err(Error::InvalidArg {
                op: "MultiHeadAttention::project_output",
                ..
            })
        ));
        // Three heads where the layer has two.
        assert!(matches!(
            attn.project_output(&t(&[3, 3, 2]), Mode::EVAL),
            Err(Error::ShapeMismatch {
                op: "MultiHeadAttention::project_output",
                ..
            })
        ));
        // The right head count, the wrong head width.
        assert!(matches!(
            attn.project_output(&t(&[2, 3, 3]), Mode::EVAL),
            Err(Error::ShapeMismatch { .. })
        ));
    }
}
