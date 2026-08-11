//! Normalization layers: [`LayerNorm`], [`RMSNorm`], [`BatchNorm2d`].
//!
//! This module is private and its types are re-exported flat from
//! [`nn`](crate::nn), so rustdoc renders the *types'* documentation and not
//! this header — the user-facing statement of each convention lives on the
//! three types. What follows is the shared overview, for a reader of the file.
//!
//! All three share one shape: subtract a location statistic, divide by a scale
//! statistic, then apply a learnable per-feature affine. What differs is
//! **which axes the statistics are taken over** and **where they come from**:
//!
//! | layer | statistics over | source | learnable |
//! |---|---|---|---|
//! | [`LayerNorm`] | the trailing `normalized_shape` axes of each sample | always the current input | `weight`, `bias` |
//! | [`RMSNorm`] | the same axes, but no mean subtraction | always the current input | `weight` |
//! | [`BatchNorm2d`] | axes `(N, H, W)` — per channel | batch statistics in train, running buffers in eval | `weight`, `bias` |
//!
//! # The variance convention (`correction = 0`)
//!
//! Every normalization here divides the squared deviations by the **full**
//! element count `n`, not `n − 1`: the population (biased, `correction = 0`)
//! variance, which is what PyTorch's norm layers use. It is the only choice
//! that makes sense for a whitening transform — the statistic is a property of
//! the block being normalized, not an estimate of some wider population's
//! variance.
//!
//! This is deliberately **not** the convention of the standalone
//! [`Tensor::var`](crate::Tensor::var)/[`std`](crate::Tensor::std) ops, which
//! follow PyTorch's *other* default of `correction = 1` (Bessel's correction)
//! and are pinned that way by the reference-vector suite. The two are different
//! things with the same name and this crate matches PyTorch on both: the ops
//! estimate a population variance from a sample, the layers whiten a block.
//! Nothing here calls `Tensor::var`.
//!
//! [`BatchNorm2d`] then has the one exception that trips everybody, and it is
//! PyTorch's too: the value folded into the `running_var` **buffer** is the
//! *unbiased* (`correction = 1`) batch variance, because that buffer really is
//! an estimate of the data distribution's variance. Normalization uses the
//! biased form; the running estimate stores the unbiased one. See
//! [`BatchNorm2d`] for the arithmetic.
//!
//! # Mode
//!
//! [`LayerNorm`] and [`RMSNorm`] ignore [`Mode`]'s behavior axis entirely —
//! per-sample statistics need no train/eval distinction — and read only its
//! recording axis, through `Param::get`. [`BatchNorm2d`] reads **both**: see
//! its docs for the two-branch table.

use crate::autograd::{self, BackwardFn};
use crate::backend::{FusedOp, dispatch};
use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::nn::{Forward, Mode, Param};
use crate::shape::Shape;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// Shared arithmetic
//
// These are free functions over plain tensors, not methods on the layers, so
// the finite-difference suite can differentiate the formulas with respect to
// `weight`/`bias` directly (a `Param`'s traced leaf cannot be a `check_grad`
// input). The layers are thin `Mode` + state wrappers over them.
// ---------------------------------------------------------------------------

/// Mean over the last `axes` axes, each kept at size 1 so the result still
/// broadcasts against `x`.
///
/// A cascade of `mean_keepdim` calls is exactly the mean over the whole block:
/// every partial mean averages the same number of elements, so the outer means
/// are unweighted.
fn mean_last(x: &Tensor, axes: usize) -> Result<Tensor> {
    let mut m = x.clone();
    for k in 0..axes {
        m = m.mean_keepdim(-1 - k as isize)?;
    }
    Ok(m)
}

/// Mean over the batch and spatial axes of an `NCHW` tensor, keeping rank 4:
/// the per-channel statistic shaped `[1, C, 1, 1]`.
///
/// The axis set is the whole point of the layer — reducing `(0, 2, 3)` and
/// *not* the channel axis — so it is spelled once, here.
fn mean_nchw(x: &Tensor) -> Result<Tensor> {
    x.mean_keepdim(0)?.mean_keepdim(2)?.mean_keepdim(3)
}

/// `xhat * weight (+ bias)`, with `weight`/`bias` broadcasting against `xhat`.
fn affine(xhat: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let scaled = xhat.mul(weight)?;
    match bias {
        Some(b) => scaled.add(b),
        None => Ok(scaled),
    }
}

/// `sqrt(var + eps)`, the divisor shared by every layer here.
fn scale_from_var(var: &Tensor, eps: f64) -> Result<Tensor> {
    var.add_scalar(eps)?.sqrt()
}

/// The [`LayerNorm`] formula over plain tensors: normalize the last
/// `weight.rank()` axes of `x` with `correction = 0` statistics, then apply
/// `weight`/`bias` (which broadcast from the right).
fn composed_layer_norm(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    eps: f64,
) -> Result<Tensor> {
    let axes = weight.rank();
    let mu = mean_last(x, axes)?;
    let centered = x.sub(&mu)?;
    let var = mean_last(&centered.mul(&centered)?, axes)?;
    let xhat = centered.div(&scale_from_var(&var, eps)?)?;
    affine(&xhat, weight, bias)
}

/// The detached values needed by the single-node fused LayerNorm backward.
struct LayerNormBackward {
    xhat: Tensor,
    inv_std: Tensor,
    weight: Tensor,
    stat_dims: Vec<usize>,
    affine_dims: Vec<usize>,
    width: f64,
}

impl LayerNormBackward {
    fn grad(&self, g: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let state_dtype = self.xhat.dtype();
        let g_wide = if g.dtype() == state_dtype {
            g.clone()
        } else {
            g.to_dtype(state_dtype)?
        };
        let weight_wide = if self.weight.dtype() == state_dtype {
            self.weight.clone()
        } else {
            self.weight.to_dtype(state_dtype)?
        };
        let dx = match dispatch::backend(g.device()).fused(
            FusedOp::LayerNorm,
            &[
                g.view(),
                self.xhat.view(),
                self.inv_std.view(),
                self.weight.view(),
            ],
            &[],
        ) {
            Ok(mut outputs) if outputs.len() == 1 => {
                Tensor::from_parts(outputs.remove(0), Layout::contiguous(g.shape().clone())?)
            }
            Ok(outputs) => {
                return Err(Error::Backend {
                    op: "LayerNorm::backward",
                    msg: format!(
                        "fused LayerNorm backward returned {} outputs, expected exactly 1",
                        outputs.len()
                    ),
                });
            }
            Err(Error::Unsupported { .. }) => {
                let weighted = g_wide.mul(&weight_wide)?;
                let sum = weighted.sum_to(&self.stat_dims)?;
                let projected = weighted.mul(&self.xhat)?.sum_to(&self.stat_dims)?;
                let dx = weighted
                    .mul_scalar(self.width)?
                    .sub(&sum)?
                    .sub(&self.xhat.mul(&projected)?)?
                    .mul(&self.inv_std)?
                    .div_scalar(self.width)?;
                if dx.dtype() == g.dtype() {
                    dx
                } else {
                    dx.to_dtype(g.dtype())?
                }
            }
            Err(err) => return Err(err),
        };
        let dweight = g_wide
            .mul(&self.xhat)?
            .sum_to(&self.affine_dims)?
            .to_dtype(g.dtype())?;
        let dbias = g_wide.sum_to(&self.affine_dims)?.to_dtype(g.dtype())?;
        Ok((dx, dweight, dbias))
    }
}

/// Last-axis LayerNorm through the optional fused backend contract.
fn fused_layer_norm(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor> {
    const OP: &str = "LayerNorm::forward";
    let save_stats = [x, weight, bias].iter().any(|input| input.node().is_some());
    let scalars = [eps, 1.0];
    let mut outputs = dispatch::backend(x.device()).fused(
        FusedOp::LayerNorm,
        &[x.view(), weight.view(), bias.view()],
        &scalars[..if save_stats { 2 } else { 1 }],
    )?;
    let expected = if save_stats { 3 } else { 1 };
    if outputs.len() != expected {
        return Err(Error::InvalidArg {
            op: OP,
            msg: format!(
                "fused LayerNorm backend returned {} outputs, expected exactly {expected}",
                outputs.len(),
            ),
        });
    }
    let out = Tensor::from_parts(outputs.remove(0), Layout::contiguous(x.shape().clone())?);

    if !save_stats {
        return Ok(out);
    }

    let mut stat_dims = x.dims().to_vec();
    let width = stat_dims[stat_dims.len() - 1];
    *stat_dims.last_mut().expect("LayerNorm input has a suffix") = 1;
    let xhat = Tensor::from_parts(outputs.remove(0), Layout::contiguous(x.shape().clone())?);
    let inv_std = Tensor::from_parts(outputs.remove(0), Layout::contiguous(stat_dims.as_slice())?);
    let backward_state = LayerNormBackward {
        xhat,
        inv_std,
        weight: weight.detach(),
        stat_dims,
        affine_dims: weight.dims().to_vec(),
        width: width as f64,
    };
    let backward: BackwardFn = Box::new(move |g| {
        let (dx, dweight, dbias) = backward_state.grad(g)?;
        Ok(vec![Some(dx), Some(dweight), Some(dbias)])
    });
    Ok(autograd::record(OP, out, &[x, weight, bias], backward))
}

/// LayerNorm over caller-owned parameter tensors.
pub(crate) fn layer_norm_forward(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    eps: f64,
) -> Result<Tensor> {
    check_suffix("LayerNorm::forward", x, weight.shape())?;
    let Some(bias) = bias else {
        return composed_layer_norm(x, weight, None, eps);
    };
    if weight.rank() != 1 || !x.dtype().is_float() {
        return composed_layer_norm(x, weight, Some(bias), eps);
    }
    match fused_layer_norm(x, weight, bias, eps) {
        Ok(out) => Ok(out),
        Err(Error::Unsupported { .. }) => composed_layer_norm(x, weight, Some(bias), eps),
        Err(err) => Err(err),
    }
}

/// The [`RMSNorm`] formula over plain tensors: divide by the root mean square
/// of the last `weight.rank()` axes — no mean subtraction, no bias.
fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let ms = mean_last(&x.mul(x)?, weight.rank())?;
    let xhat = x.div(&scale_from_var(&ms, eps)?)?;
    affine(&xhat, weight, None)
}

/// RMSNorm over a caller-owned parameter tensor.
pub(crate) fn rms_norm_forward(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    check_suffix("RMSNorm::forward", x, weight.shape())?;
    rms_norm(x, weight, eps)
}

/// The [`BatchNorm2d`] formula over plain tensors. `mu` and `var` are
/// per-channel statistics shaped `[1, C, 1, 1]`, whatever their origin — the
/// batch's own (biased) statistics in train, the running buffers in eval;
/// `weight` and `bias` are shaped `[C]` and are reshaped here, because `[C]`
/// would otherwise broadcast against the **width** axis of an `NCHW` input.
fn batch_norm(
    x: &Tensor,
    mu: &Tensor,
    var: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    let channels = [1, weight.dims1()?, 1, 1];
    let xhat = x.sub(mu)?.div(&scale_from_var(var, eps)?)?;
    affine(
        &xhat,
        &weight.reshape(channels)?,
        Some(&bias.reshape(channels)?),
    )
}

/// Per-channel batch mean and **biased** variance of an `NCHW` tensor, both
/// shaped `[1, C, 1, 1]`, differentiable through `x`.
fn batch_stats(x: &Tensor) -> Result<(Tensor, Tensor)> {
    let mu = mean_nchw(x)?;
    let centered = x.sub(&mu)?;
    let var = mean_nchw(&centered.mul(&centered)?)?;
    Ok((mu, var))
}

/// BatchNorm2d over caller-owned parameters and running buffers.
///
/// Training returns detached replacement buffers; eval returns none. Both
/// replacements are computed before returning so the caller can update them
/// atomically.
#[allow(clippy::too_many_arguments)] // The arguments are the external state contract.
pub(crate) fn batch_norm2d_forward(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    running_mean: &Tensor,
    running_var: &Tensor,
    eps: f64,
    momentum: f64,
    mode: Mode,
) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
    const OP: &str = "BatchNorm2d::forward";
    let (n, c, h, w) = x.dims4()?;
    let channels = running_mean.dims1()?;
    if c != channels {
        return Err(Error::ShapeMismatch {
            op: OP,
            lhs: x.shape().clone(),
            rhs: Shape::from(vec![channels]),
        });
    }
    let count = n * h * w;
    let (mu, var) = if mode.is_training() {
        if count < 2 {
            return Err(Error::InvalidArg {
                op: OP,
                msg: format!(
                    "a training forward needs at least 2 elements per channel to \
                     estimate a variance, got N*H*W = {count} (input {})",
                    x.shape()
                ),
            });
        }
        batch_stats(x)?
    } else {
        let stat_shape = [1, c, 1, 1];
        (
            running_mean.reshape(stat_shape)?,
            running_var.reshape(stat_shape)?,
        )
    };

    // Build the output first: a rejected affine must not age the buffers.
    let out = batch_norm(x, &mu, &var, weight, bias, eps)?;
    if !mode.is_training() {
        return Ok((out, None));
    }

    let shape = [channels];
    let keep = 1.0 - momentum;
    let unbiased = count as f64 / (count - 1) as f64;
    let batch_mean = mu.detach().reshape(shape)?;
    let batch_var = var.detach().reshape(shape)?.mul_scalar(unbiased)?;
    let mean = running_mean
        .mul_scalar(keep)?
        .add(&batch_mean.mul_scalar(momentum)?)?
        .detach();
    let variance = running_var
        .mul_scalar(keep)?
        .add(&batch_var.mul_scalar(momentum)?)?
        .detach();
    Ok((out, Some((mean, variance))))
}

/// Validate a `weight`-shaped normalization spec at construction time.
pub(crate) fn check_normalized_shape(op: &'static str, shape: &Shape) -> Result<()> {
    if shape.rank() == 0 || shape.dims().contains(&0) {
        return Err(Error::InvalidArg {
            op,
            msg: format!(
                "normalized_shape must have rank >= 1 with every axis non-empty, got {shape}"
            ),
        });
    }
    Ok(())
}

/// Validate `eps`, which is added to a variance and square-rooted.
pub(crate) fn check_eps(op: &'static str, eps: f64) -> Result<()> {
    if !(eps.is_finite() && eps > 0.0) {
        return Err(Error::InvalidArg {
            op,
            msg: format!("eps must be finite and positive, got {eps}"),
        });
    }
    Ok(())
}

/// Validate `momentum`, the weight the running statistics give a fresh batch.
pub(crate) fn check_momentum(op: &'static str, momentum: f64) -> Result<()> {
    if !(momentum.is_finite() && (0.0..=1.0).contains(&momentum)) {
        return Err(Error::InvalidArg {
            op,
            msg: format!("momentum must lie in [0, 1], got {momentum}"),
        });
    }
    Ok(())
}

/// The trailing-axes rule shared by [`LayerNorm`] and [`RMSNorm`].
pub(crate) fn check_suffix(op: &'static str, x: &Tensor, normalized: &Shape) -> Result<()> {
    let (xd, nd) = (x.dims(), normalized.dims());
    if xd.len() < nd.len() || &xd[xd.len() - nd.len()..] != nd {
        return Err(Error::ShapeMismatch {
            op,
            lhs: x.shape().clone(),
            rhs: normalized.clone(),
        });
    }
    Ok(())
}

/// A `shape` tensor filled with `value`, F32 on `device` — how every parameter and
/// buffer here is initialized (constructors build F32; `nn::to_dtype` converts
/// afterwards).
fn filled(shape: impl Into<Shape>, value: f64, device: &Device) -> Result<Tensor> {
    Tensor::full(shape, value, DType::F32, device)
}

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

/// Layer normalization over the **trailing axes** of each sample
/// (`normalized_shape`), with a learnable per-feature scale and shift.
///
/// ```text
/// y = (x − mean(x)) / sqrt(var(x) + eps) * weight + bias
/// ```
///
/// where `mean`/`var` are taken over the last `normalized_shape.rank()` axes of
/// `x`, independently per remaining position. Every axis before the normalized
/// ones is a batch axis, so one `LayerNorm([d])` serves `[b, d]`, `[b, t, d]`,
/// and `[d]` alike — the transformer's usual call.
///
/// `weight` is initialized to ones and `bias` to zeros (F32; convert with
/// [`nn::to_dtype`](crate::nn::to_dtype)), so a fresh layer is a pure
/// whitening transform. The affine is not optional: the parameter names
/// `weight` and `bias` are part of the `state_dict` contract.
///
/// # The variance convention (`correction = 0`)
///
/// `var` divides the squared deviations by the **full** element count `n`, not
/// `n − 1`: the population (biased) variance, which is what PyTorch's norm
/// layers use and the only choice that makes sense for a whitening transform —
/// the statistic describes the block being normalized rather than estimating
/// some wider population's spread.
///
/// This is deliberately **not** the convention of the standalone
/// [`Tensor::var`](crate::Tensor::var)/[`std`](crate::Tensor::std) ops, which
/// follow PyTorch's *other* default of `correction = 1` (Bessel's correction).
/// The two are different things that share a name, and this crate matches
/// PyTorch on both; nothing in this layer calls `Tensor::var`.
///
/// [`Mode`]'s behavior axis is irrelevant here (the statistics are per-sample,
/// never running), so train and eval produce identical numbers; only recording
/// differs.
///
/// ```
/// use rstorch::nn::{Forward, LayerNorm, Mode};
/// use rstorch::{Device, Tensor};
///
/// let dev = Device::Cpu;
/// let mut norm = LayerNorm::new([4], &dev)?;
/// let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [1, 4], &dev)?;
/// let y = norm.forward(&x, Mode::EVAL)?;
/// // mean 2.5, population variance 1.25: the outer values land at ±1.5/√1.25.
/// let got = y.to_vec::<f32>()?;
/// assert!((got[0] - -1.341_640_8).abs() < 1e-4);
/// assert!((got[3] - 1.341_640_8).abs() < 1e-4);
/// # Ok::<(), rstorch::Error>(())
/// ```
#[derive(rstorch::Module)]
pub struct LayerNorm {
    weight: Param,
    bias: Param,
    eps: f64,
}

impl std::fmt::Debug for LayerNorm {
    /// The configuration, not the parameter values: `Param` is not `Debug`
    /// (its values are reached through `state_dict`), and a layer's shape is
    /// what a reader of a model dump wants.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerNorm")
            .field("normalized_shape", &self.normalized_shape().dims())
            .field("eps", &self.eps)
            .finish()
    }
}

impl LayerNorm {
    /// The default `eps`, matching PyTorch's `nn.LayerNorm`.
    pub const DEFAULT_EPS: f64 = 1e-5;

    /// A layer normalizing over the trailing `normalized_shape` axes, with
    /// [`DEFAULT_EPS`](LayerNorm::DEFAULT_EPS).
    ///
    /// # Errors
    /// As [`with_eps`](LayerNorm::with_eps).
    pub fn new(normalized_shape: impl Into<Shape>, device: &Device) -> Result<LayerNorm> {
        LayerNorm::with_eps(normalized_shape, Self::DEFAULT_EPS, device)
    }

    /// A layer normalizing over the trailing `normalized_shape` axes with an
    /// explicit `eps`.
    ///
    /// # Errors
    /// [`Error::InvalidArg`] (`op: "LayerNorm::new"`) if `normalized_shape` is
    /// rank 0 or has an empty axis, or if `eps` is not finite and positive
    /// (it is added to a variance and square-rooted). Allocation failures on
    /// `device` propagate.
    pub fn with_eps(
        normalized_shape: impl Into<Shape>,
        eps: f64,
        device: &Device,
    ) -> Result<LayerNorm> {
        const OP: &str = "LayerNorm::new";
        let shape: Shape = normalized_shape.into();
        check_normalized_shape(OP, &shape)?;
        check_eps(OP, eps)?;
        Ok(LayerNorm {
            weight: Param::new(filled(&shape, 1.0, device)?),
            bias: Param::new(filled(&shape, 0.0, device)?),
            eps,
        })
    }

    /// The shape this layer normalizes over (the shape of `weight`).
    pub fn normalized_shape(&self) -> &Shape {
        self.weight.value().shape()
    }

    /// The `eps` added to the variance before the square root.
    pub fn eps(&self) -> f64 {
        self.eps
    }
}

impl Forward for LayerNorm {
    /// # Errors
    /// [`Error::ShapeMismatch`] (`op: "LayerNorm::forward"`) if `x`'s trailing
    /// axes are not exactly [`normalized_shape`](LayerNorm::normalized_shape)
    /// (`lhs` is `x`'s shape, `rhs` the expected suffix), and
    /// [`Error::DTypeMismatch`]/[`Error::DeviceMismatch`] if `x` disagrees with
    /// the parameters — convert one side explicitly.
    fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        let weight = self.weight.get(mode);
        let bias = self.bias.get(mode);
        layer_norm_forward(x, &weight, Some(&bias), self.eps)
    }
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

/// Root-mean-square normalization over the trailing axes: [`LayerNorm`]
/// without the mean subtraction and without the bias.
///
/// ```text
/// y = x / sqrt(mean(x²) + eps) * weight
/// ```
///
/// with the mean over the last `normalized_shape.rank()` axes — a plain average
/// over all `n` of them, the same uncorrected convention [`LayerNorm`] uses for
/// its variance. Because nothing is centered, the layer is cheaper than
/// [`LayerNorm`] and re-centering-free — the reason modern decoder stacks prefer
/// it. `weight` is initialized to ones; there is no `bias` parameter at all, so
/// a `state_dict` has exactly one key per layer.
///
/// Like [`LayerNorm`], it ignores [`Mode`]'s behavior axis.
///
/// ```
/// use rstorch::nn::{Forward, Mode, RMSNorm};
/// use rstorch::{Device, Tensor};
///
/// let dev = Device::Cpu;
/// let mut norm = RMSNorm::new([4], &dev)?;
/// let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [4], &dev)?;
/// // mean(x²) = 7.5, so every element is divided by √7.5 ≈ 2.738613.
/// let got = norm.forward(&x, Mode::EVAL)?.to_vec::<f32>()?;
/// assert!((got[0] - 0.365_148_4).abs() < 1e-4);
/// # Ok::<(), rstorch::Error>(())
/// ```
#[derive(rstorch::Module)]
pub struct RMSNorm {
    weight: Param,
    eps: f64,
}

impl std::fmt::Debug for RMSNorm {
    /// The configuration, as for [`LayerNorm`].
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RMSNorm")
            .field("normalized_shape", &self.normalized_shape().dims())
            .field("eps", &self.eps)
            .finish()
    }
}

impl RMSNorm {
    /// The default `eps`. Smaller than [`LayerNorm::DEFAULT_EPS`] because the
    /// divisor is a raw second moment rather than a variance, so it is
    /// typically further from zero.
    ///
    /// Unlike [`LayerNorm::DEFAULT_EPS`] this is *not* PyTorch's default —
    /// `torch.nn.RMSNorm` defaults to the input dtype's machine epsilon
    /// (`~1.2e-7` in F32). `1e-6` is the value the published decoder stacks use,
    /// and it is dtype-independent, which the machine-epsilon rule is not. Pass
    /// [`with_eps`](RMSNorm::with_eps) to match a specific reference.
    pub const DEFAULT_EPS: f64 = 1e-6;

    /// A layer normalizing over the trailing `normalized_shape` axes, with
    /// [`DEFAULT_EPS`](RMSNorm::DEFAULT_EPS).
    ///
    /// # Errors
    /// As [`with_eps`](RMSNorm::with_eps).
    pub fn new(normalized_shape: impl Into<Shape>, device: &Device) -> Result<RMSNorm> {
        RMSNorm::with_eps(normalized_shape, Self::DEFAULT_EPS, device)
    }

    /// A layer normalizing over the trailing `normalized_shape` axes with an
    /// explicit `eps`.
    ///
    /// # Errors
    /// [`Error::InvalidArg`] (`op: "RMSNorm::new"`) on a rank-0 or empty
    /// `normalized_shape`, or an `eps` that is not finite and positive.
    pub fn with_eps(
        normalized_shape: impl Into<Shape>,
        eps: f64,
        device: &Device,
    ) -> Result<RMSNorm> {
        const OP: &str = "RMSNorm::new";
        let shape: Shape = normalized_shape.into();
        check_normalized_shape(OP, &shape)?;
        check_eps(OP, eps)?;
        Ok(RMSNorm {
            weight: Param::new(filled(&shape, 1.0, device)?),
            eps,
        })
    }

    /// The shape this layer normalizes over (the shape of `weight`).
    pub fn normalized_shape(&self) -> &Shape {
        self.weight.value().shape()
    }

    /// The `eps` added to the mean square before the square root.
    pub fn eps(&self) -> f64 {
        self.eps
    }
}

impl Forward for RMSNorm {
    /// # Errors
    /// As [`LayerNorm`]'s `forward` — [`Error::ShapeMismatch`] on a trailing
    /// shape that is not [`normalized_shape`](RMSNorm::normalized_shape)
    /// (`op: "RMSNorm::forward"`), or a dtype/device mismatch with `weight`.
    fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        rms_norm_forward(x, &self.weight.get(mode), self.eps)
    }
}

// ---------------------------------------------------------------------------
// BatchNorm2d
// ---------------------------------------------------------------------------

/// Batch normalization for `NCHW` images: per-**channel** statistics over the
/// batch and spatial axes, with running estimates kept in buffers for eval.
///
/// Four leaves, whose names are the `state_dict` contract: parameters `weight`
/// (ones) and `bias` (zeros), and non-trainable buffers `running_mean` (zeros)
/// and `running_var` (ones), each shaped `[C]`. Buffers land in
/// [`state_dict`](crate::nn::state_dict) and move under
/// [`to_device`](crate::nn::to_device)/[`to_dtype`](crate::nn::to_dtype), so a
/// checkpoint restores a usable eval model.
///
/// # The two branches
///
/// [`Mode`]'s **behavior** axis — not its recording axis — chooses the
/// statistics:
///
/// | `mode.is_training()` | normalizes with | touches the buffers |
/// |---|---|---|
/// | `true` (`Mode::TRAIN`, `Mode::TRAIN.frozen()`) | this batch's own mean/variance, differentiably | yes, an EMA update |
/// | `false` (`Mode::EVAL`, `Mode::EVAL.recorded()`) | `running_mean` / `running_var` | no |
///
/// So `Mode::EVAL.recorded()` is the frozen-statistics fine-tuning flow
/// (gradients reach `weight`/`bias` and the input, but the buffers stand
/// still), and `Mode::TRAIN.frozen()` still advances the buffers with no graph
/// — the two axes really are independent. An eval forward before any training
/// step normalizes with the initial mean 0 / variance 1, i.e. it only applies
/// the affine.
///
/// # The running-statistics arithmetic
///
/// After a training forward over `m = N·H·W` elements per channel:
///
/// ```text
/// running_mean ← (1 − momentum) · running_mean + momentum · batch_mean
/// running_var  ← (1 − momentum) · running_var  + momentum · batch_var_unbiased
/// batch_var_unbiased = batch_var_biased · m / (m − 1)
/// ```
///
/// The update is a plain exponential moving average with a constant
/// `momentum` (PyTorch's `momentum=None` cumulative-average mode is not
/// implemented, so there is no `num_batches_tracked` buffer). The values folded
/// in are detached, so the buffers never hold a graph.
///
/// Note the deliberate asymmetry, which is PyTorch's: the **normalization**
/// divides by the biased (`correction = 0`) batch variance — see
/// [`LayerNorm`]'s note on that convention — while the value folded into
/// `running_var` is the **unbiased** (`correction = 1`) one, because that buffer
/// really is an estimate of the data distribution's variance rather than a
/// whitening statistic. It is also why a converged eval forward does not
/// reproduce the train forward exactly: the two divisors differ by
/// `m / (m − 1)`.
///
/// ```
/// use rstorch::nn::{self, BatchNorm2d, Forward, Mode};
/// use rstorch::{Device, Tensor};
///
/// let dev = Device::Cpu;
/// let mut bn = BatchNorm2d::new(2, &dev)?;
/// let x = Tensor::from_vec((0..16).map(|i| i as f32).collect(), [2, 2, 2, 2], &dev)?;
/// let _ = bn.forward(&x, Mode::TRAIN)?;      // updates the buffers
/// let state = nn::state_dict(&bn);
/// // channel 0 holds 0,1,2,3, 8,9,10,11 -> mean 5.5; EMA from 0 with 0.1.
/// assert!((state["running_mean"].to_vec::<f32>()?[0] - 0.55).abs() < 1e-6);
/// # Ok::<(), rstorch::Error>(())
/// ```
#[derive(rstorch::Module)]
pub struct BatchNorm2d {
    weight: Param,
    bias: Param,
    running_mean: Tensor,
    running_var: Tensor,
    eps: f64,
    momentum: f64,
}

impl std::fmt::Debug for BatchNorm2d {
    /// The configuration, as for [`LayerNorm`]. The running buffers are
    /// tensors and print through [`running_mean`](BatchNorm2d::running_mean) /
    /// [`running_var`](BatchNorm2d::running_var) if wanted.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchNorm2d")
            .field("channels", &self.channels())
            .field("eps", &self.eps)
            .field("momentum", &self.momentum)
            .finish()
    }
}

impl BatchNorm2d {
    /// The default `eps`, matching PyTorch's `nn.BatchNorm2d`.
    pub const DEFAULT_EPS: f64 = 1e-5;

    /// The default EMA `momentum`, matching PyTorch's `nn.BatchNorm2d`. Note
    /// the convention: `momentum` weights the **new** batch, so larger means
    /// faster forgetting (the opposite of the optimizer's momentum).
    pub const DEFAULT_MOMENTUM: f64 = 0.1;

    /// A layer over `channels` channels with
    /// [`DEFAULT_EPS`](BatchNorm2d::DEFAULT_EPS) and
    /// [`DEFAULT_MOMENTUM`](BatchNorm2d::DEFAULT_MOMENTUM).
    ///
    /// # Errors
    /// As [`with_params`](BatchNorm2d::with_params).
    pub fn new(channels: usize, device: &Device) -> Result<BatchNorm2d> {
        BatchNorm2d::with_params(channels, Self::DEFAULT_EPS, Self::DEFAULT_MOMENTUM, device)
    }

    /// A layer over `channels` channels with an explicit `eps` and `momentum`.
    ///
    /// # Errors
    /// [`Error::InvalidArg`] (`op: "BatchNorm2d::new"`) if `channels` is zero,
    /// if `eps` is not finite and positive, or if `momentum` is outside
    /// `[0, 1]` (outside that range the EMA diverges or runs backwards).
    pub fn with_params(
        channels: usize,
        eps: f64,
        momentum: f64,
        device: &Device,
    ) -> Result<BatchNorm2d> {
        const OP: &str = "BatchNorm2d::new";
        if channels == 0 {
            return Err(Error::InvalidArg {
                op: OP,
                msg: "channels must be non-zero".to_string(),
            });
        }
        check_eps(OP, eps)?;
        check_momentum(OP, momentum)?;
        Ok(BatchNorm2d {
            weight: Param::new(filled([channels], 1.0, device)?),
            bias: Param::new(filled([channels], 0.0, device)?),
            running_mean: filled([channels], 0.0, device)?,
            running_var: filled([channels], 1.0, device)?,
            eps,
            momentum,
        })
    }

    /// The number of channels this layer normalizes.
    pub fn channels(&self) -> usize {
        self.running_mean.dims()[0]
    }

    /// The `eps` added to the variance before the square root.
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// The EMA weight given to each new batch (see the type docs).
    pub fn momentum(&self) -> f64 {
        self.momentum
    }

    /// The running mean buffer (`[C]`), as used by an eval forward.
    pub fn running_mean(&self) -> &Tensor {
        &self.running_mean
    }

    /// The running variance buffer (`[C]`), as used by an eval forward.
    pub fn running_var(&self) -> &Tensor {
        &self.running_var
    }
}

impl Forward for BatchNorm2d {
    /// # Errors
    /// [`Error::RankMismatch`] (`op: "dims4"`) if `x` is not rank 4,
    /// [`Error::ShapeMismatch`] (`op: "BatchNorm2d::forward"`) if `x`'s channel
    /// axis is not [`channels`](BatchNorm2d::channels) (`rhs` is `[C]`),
    /// [`Error::InvalidArg`] (`op: "BatchNorm2d::forward"`) if a **training**
    /// forward has fewer than two elements per channel (`N·H·W < 2`: no
    /// unbiased variance to store, exactly PyTorch's "expected more than 1
    /// value per channel" refusal), and
    /// [`Error::DTypeMismatch`]/[`Error::DeviceMismatch`] if `x` disagrees with
    /// the parameters and buffers.
    fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        let weight = self.weight.get(mode);
        let bias = self.bias.get(mode);
        let (out, replacements) = batch_norm2d_forward(
            x,
            &weight,
            &bias,
            &self.running_mean,
            &self.running_var,
            self.eps,
            self.momentum,
            mode,
        )?;
        if let Some((mean, variance)) = replacements {
            self.running_mean = mean;
            self.running_var = variance;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
