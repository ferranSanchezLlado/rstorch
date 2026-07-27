//! Reductions (T23): `sum`/`mean`/`max`/`min`/`var`/`std` in three spellings
//! each, the normalizations `softmax`/`log_softmax`, and the index reductions
//! `argmax`/`argmin`.
//!
//! # Three spellings, one meaning (exploration §4.2)
//!
//! Every value reduction comes as `op(axis)` (the axis is reduced away and
//! **dropped**), `op_keepdim(axis)` (the axis stays at size 1, so the result
//! still broadcasts against the input) and `op_all()` (every axis reduced, to
//! a rank-0 scalar). Axes are `isize` with negative indexing, so `-1` is
//! always the last axis.
//!
//! # Empty-reduction policy (exploration §3.1)
//!
//! Reducing an axis of size 0 has an answer only where the op has an identity
//! element. The rule here is one line:
//!
//! > **`sum`/`sum_all` return the identity (zeros); every other reduction
//! > over an empty axis is a loud [`Error::InvalidArg`] naming the op.**
//!
//! So `sum` of an empty axis is `0`, while `mean`, `max`, `min`, `var`,
//! `std`, `softmax`, `log_softmax`, `argmax` and `argmin` refuse rather than
//! return the `NaN`/`-inf`/arbitrary-index answers a silent implementation
//! would produce. `var`/`std` extend the same rule to `correction`: with
//! `correction = 1` an axis of length 1 has no unbiased variance, so it is
//! rejected instead of returning `NaN`.
//!
//! # Numerics
//!
//! - Accumulation happens in the wide [`Acc`](crate::dtype::Element::Acc)
//!   type inside the kernel, with a single cast at output (the fix for the v2
//!   f16 sum-saturation bug).
//! - `var`/`std` use **`correction = 1`** (Bessel's correction), PyTorch's
//!   default, and are float-only.
//! - `softmax`/`log_softmax` use the standard max-shifted formulas, so a row
//!   of large logits cannot overflow. The shift is **detached**: softmax is
//!   invariant to it, so no gradient flows through the `max`.
//! - **Mask-aware**: a row that is entirely `-inf` (the fully-masked attention
//!   row) yields all zeros from `softmax` and all `-inf` from `log_softmax` —
//!   the consistent `p = 0` / `log p = -inf` answer — instead of the `0/0`
//!   `NaN` the naive formula produces.
//!
//! # Backwards
//!
//! | forward | backward |
//! |---|---|
//! | `sum(axis)` | broadcast the cotangent back along `axis` |
//! | `mean(axis)` | the same, scaled by `1/n` |
//! | `max`/`min(axis)` | route to the winners, splitting ties evenly; a line whose extremum is `NaN` has no winner and gets an all-`NaN` cotangent |
//! | `var`/`std`, `log_softmax` | none of their own — they are *composed* from recorded ops (`mean`/`sub`/`mul`/`sum`/`exp`/`div`), so the engine differentiates the composition |
//! | `softmax` | the fused last-axis `F32`/`F64` path records one node with `y · (g − Σ(g · y))`; other variants retain the composed path |
//! | `argmax`/`argmin` | not differentiable ([`I64`](crate::DType::I64) output); they never reach the record seam |
//!
//! The `max`/`min` backward captures the **detached** input and output, while
//! fused softmax captures its **detached** output `y` (the detached-output
//! capture rule, exploration §4.3). Each is built before the traced output is
//! assembled. The extrema `NaN` rule is stated in full on `route_to_extrema`.

use crate::autograd::{BackwardFn, record};
use crate::backend::{ArgReduceOp, FusedOp, ReduceOp, dispatch};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// Shared plumbing
// ---------------------------------------------------------------------------

/// Re-label a backend error with the public method the caller used: the
/// kernels name their errors after the family (`"reduce"`/`"arg_reduce"`),
/// while the user called `sum`, `max`, `argmin`, ...
fn relabel(op: &'static str, e: Error) -> Error {
    match e {
        Error::Unsupported { device, dtype, .. } => Error::Unsupported { op, device, dtype },
        other => other,
    }
}

/// The empty-reduction policy for the axis spellings: only `sum` has an
/// identity element, so every other op refuses a size-0 axis.
fn require_non_empty(op: &'static str, x: &Tensor, axis: usize) -> Result<()> {
    if x.dims()[axis] == 0 {
        return Err(Error::InvalidArg {
            op,
            msg: format!(
                "{op} over axis {axis} of shape {} is undefined: the axis is empty \
                 (only `sum` has an identity element)",
                x.shape()
            ),
        });
    }
    Ok(())
}

/// The same policy for the `_all` spellings: an empty tensor is exactly one
/// with an empty axis.
fn require_non_empty_all(op: &'static str, x: &Tensor) -> Result<()> {
    if x.num_elements() == 0 {
        return Err(Error::InvalidArg {
            op,
            msg: format!(
                "{op} over an empty tensor (shape {}) is undefined \
                 (only `sum_all` has an identity element)",
                x.shape()
            ),
        });
    }
    Ok(())
}

/// `var`/`std`/`softmax`/`log_softmax` are float-only: an unbiased variance of
/// integers and an integer softmax are not meaningful, and there is no silent
/// promotion.
fn require_float(op: &'static str, x: &Tensor) -> Result<()> {
    if !x.dtype().is_float() {
        return Err(Error::Unsupported {
            op,
            device: x.device(),
            dtype: x.dtype(),
        });
    }
    Ok(())
}

/// `correction = 1` needs at least two samples along the reduced axis; one
/// sample would divide by zero.
fn require_correction(op: &'static str, n: usize) -> Result<()> {
    if n < 2 {
        return Err(Error::InvalidArg {
            op,
            msg: format!(
                "{op} uses correction=1 and needs at least 2 elements along the \
                 reduced axis, got {n}"
            ),
        });
    }
    Ok(())
}

/// The result dims of a reduction over `axis`: the axis is dropped, or held at
/// 1 for the `_keepdim` spellings.
fn out_dims(dims: &[usize], axis: usize, keepdim: bool) -> Vec<usize> {
    let mut out = dims.to_vec();
    if keepdim {
        out[axis] = 1;
    } else {
        out.remove(axis);
    }
    out
}

/// The untraced forward of one axis reduction: dispatch once, then describe
/// the (contiguous) kernel output with the reduced shape.
fn reduce_values(
    op: &'static str,
    kind: ReduceOp,
    x: &Tensor,
    axis: usize,
    keepdim: bool,
) -> Result<Tensor> {
    let storage = dispatch::backend(x.device())
        .reduce(kind, x.view(), axis)
        .map_err(|e| relabel(op, e))?;
    let layout = Layout::contiguous(out_dims(x.dims(), axis, keepdim))?;
    Ok(Tensor::from_parts(storage, layout))
}

/// Put the reduced axis back into a cotangent (or a reduction output) so it
/// lines up with the input again: a no-op for the `_keepdim` spellings, an
/// `unsqueeze` otherwise.
fn with_axis(t: &Tensor, axis: usize, keepdim: bool) -> Result<Tensor> {
    if keepdim {
        Ok(t.clone())
    } else {
        t.unsqueeze(axis as isize)
    }
}

/// The `sum`/`mean` backward: optionally scale the cotangent, then spread it
/// back over the reduced axis. Broadcasting is the transpose of summing, so
/// this is a stride-0 view — no elements are duplicated in memory.
fn spread(
    g: &Tensor,
    axis: usize,
    keepdim: bool,
    src_dims: &[usize],
    scale: Option<f64>,
) -> Result<Tensor> {
    let scaled = match scale {
        Some(s) => g.mul_scalar(s)?,
        None => g.clone(),
    };
    with_axis(&scaled, axis, keepdim)?.broadcast_to(src_dims.to_vec())
}

/// The `max`/`min` backward: the cotangent reaches exactly the elements that
/// achieved the extremum, split evenly when several tie (the rule
/// [`Tensor::maximum`] already uses element-wise, and PyTorch's `amax`/`amin`
/// rule).
///
/// `x` and `out` are the **detached** input and output; both are re-viewed
/// with the reduced axis present so they broadcast against each other.
///
/// # NaN lines (T31 decision, resolving T23's open question)
///
/// The forward **propagates** NaN: a line containing a NaN reduces to NaN
/// (`backend::cpu::reduce`). That makes the tie count degenerate here —
/// nothing compares equal to NaN, so a NaN line has *zero* winners and the
/// even split is `g / 0`.
///
/// The decision is that **NaN propagates through the backward too**: every
/// element of a line whose extremum is NaN receives a NaN cotangent,
/// whatever `g` was. Rationale:
///
/// - It is the only answer consistent with the forward. The loss that
///   consumed this output is already NaN; handing back a finite (or zero)
///   gradient would let a poisoned step look healthy, which is the failure
///   mode that is hardest to debug.
/// - It is what the even-split formula already *tried* to produce: with no
///   winners the share is `g / 0` and `hit · share` is `0 · ∞` = NaN. The
///   decision here is to keep that answer and make it deliberate, not to
///   overturn it. (PyTorch's `amax`/`amin` backward is the same
///   `mask · (g / mask.sum())` shape and so reaches `0 / 0` the same way,
///   but that was **not** re-measured for this decision — do not cite it as
///   verified parity.)
/// - The alternatives were rejected: routing the cotangent to the NaN's
///   position (`torch.max(dim)`'s index-based backward) contradicts the
///   even-split rule this op chose, and raising an error would make a
///   data-dependent NaN a hard failure deep inside `backward()`, unlike every
///   other op in the crate, which propagates NaN quietly.
///
/// The NaN is written **explicitly** rather than left to fall out of `0 · ∞`:
/// that accident holds under IEEE-754 but not under a backend free to fold
/// `0 · x` to `0`, and a numerics contract should not rest on that.
fn route_to_extrema(
    g: &Tensor,
    x: &Tensor,
    out: &Tensor,
    axis: usize,
    keepdim: bool,
) -> Result<Tensor> {
    let g_kd = with_axis(g, axis, keepdim)?;
    let out_kd = with_axis(out, axis, keepdim)?;
    let one = Tensor::ones((), x.dtype(), &x.device())?;
    let zero = Tensor::zeros((), x.dtype(), &x.device())?;
    // 1 at every element that ties the extremum, 0 elsewhere.
    let hit = x.eq(&out_kd)?.where_cond(&one, &zero)?;
    let winners = hit.sum_keepdim(axis as isize)?;
    if !x.dtype().is_float() {
        // An integer line always has a winner: the axis is non-empty and
        // every value compares equal to itself.
        return hit.mul(&g_kd.div(&winners)?);
    }
    // Zero winners means the extremum was NaN. Divide by 1 there so the share
    // stays finite, then overwrite the whole line with NaN.
    let starved = winners.eq(&zero)?;
    let share = g_kd.div(&starved.where_cond(&one, &winners)?)?;
    let nan = Tensor::full((), f64::NAN, x.dtype(), &x.device())?;
    starved.where_cond(&nan, &hit.mul(&share)?)
}

/// One axis reduction, forward plus its recorded backward. `axis` is already
/// resolved and the empty-axis policy already applied.
fn axis_reduce(
    op: &'static str,
    kind: ReduceOp,
    x: &Tensor,
    axis: usize,
    keepdim: bool,
) -> Result<Tensor> {
    if !matches!(kind, ReduceOp::Sum) {
        require_non_empty(op, x, axis)?;
    }
    let out = reduce_values(op, kind, x, axis, keepdim)?;
    let src_dims = x.dims().to_vec();
    let backward: BackwardFn = match kind {
        ReduceOp::Sum => Box::new(move |g| vec![spread(g, axis, keepdim, &src_dims, None).ok()]),
        ReduceOp::Mean => {
            let scale = 1.0 / src_dims[axis] as f64;
            Box::new(move |g| vec![spread(g, axis, keepdim, &src_dims, Some(scale)).ok()])
        }
        ReduceOp::Max | ReduceOp::Min => {
            let xd = x.detach();
            let od = out.detach();
            Box::new(move |g| vec![route_to_extrema(g, &xd, &od, axis, keepdim).ok()])
        }
    };
    Ok(record(op, out, &[x], backward))
}

/// Reduce every axis to a rank-0 scalar, one axis at a time from the last to
/// the first (so the axes still to be reduced keep their indices). A rank-0
/// input is already the answer.
fn fold_all(op: &'static str, kind: ReduceOp, x: &Tensor) -> Result<Tensor> {
    if !matches!(kind, ReduceOp::Sum) {
        require_non_empty_all(op, x)?;
    }
    let mut cur = x.clone();
    for axis in (0..x.rank()).rev() {
        cur = axis_reduce(op, kind, &cur, axis, false)?;
    }
    Ok(cur)
}

/// The shared body of `var`/`var_keepdim`/`std`/`std_keepdim`: the mean of the
/// squared deviations with `correction = 1`, composed from recorded ops (so
/// the gradient — including the path through the mean, which is what makes it
/// exactly `2(xᵢ − x̄)/(n−1)` — comes out of the engine, not a hand-written
/// closure).
fn variance(op: &'static str, x: &Tensor, axis: usize, keepdim: bool) -> Result<Tensor> {
    require_float(op, x)?;
    let n = x.dims()[axis];
    require_correction(op, n)?;
    let deviation = x.sub(&x.mean_keepdim(axis as isize)?)?;
    let squares = deviation.mul(&deviation)?;
    let summed = if keepdim {
        squares.sum_keepdim(axis as isize)?
    } else {
        squares.sum(axis as isize)?
    };
    summed.div_scalar((n - 1) as f64)
}

/// The whole-tensor variance, same contract as [`variance`].
fn variance_all(op: &'static str, x: &Tensor) -> Result<Tensor> {
    require_float(op, x)?;
    let n = x.num_elements();
    require_correction(op, n)?;
    let deviation = x.sub(&x.mean_all()?)?;
    deviation
        .mul(&deviation)?
        .sum_all()?
        .div_scalar((n - 1) as f64)
}

/// The shared, numerically stable core of `softmax`/`log_softmax`.
///
/// Returns `(z, e, denom)` where `z = x − max(x)` (the max **detached**, so no
/// gradient flows through it — softmax is invariant to the shift), `e =
/// exp(z)` and `denom = Σ e` held at size 1 on `axis`.
///
/// Fully-masked rows (every element `-inf`) are the reason this is not three
/// lines: their max is `-inf`, so the shift would be `-inf − -inf = NaN`. The
/// shift is replaced by 0 there — which leaves `z = -inf`, `e = 0` — and the
/// zero denominator is replaced by 1, so the row divides cleanly to zeros
/// (`softmax`) and `-inf` (`log_softmax`).
fn softmax_parts(op: &'static str, x: &Tensor, axis: usize) -> Result<(Tensor, Tensor, Tensor)> {
    require_float(op, x)?;
    let dtype = x.dtype();
    let device = x.device();
    let peak = axis_reduce(op, ReduceOp::Max, x, axis, true)?.detach();
    let dead = peak.eq(&Tensor::full((), f64::NEG_INFINITY, dtype, &device)?)?;
    let shift = dead.where_cond(&Tensor::zeros((), dtype, &device)?, &peak)?;
    let z = x.sub(&shift)?;
    let e = z.exp()?;
    let denom = dead.where_cond(
        &Tensor::ones((), dtype, &device)?,
        &e.sum_keepdim(axis as isize)?,
    )?;
    Ok((z, e, denom))
}

/// The existing composed softmax, kept as the exact fallback for unsupported
/// fused variants and for axes/dtypes outside the initial fused scope.
fn composed_softmax(op: &'static str, x: &Tensor, axis: usize) -> Result<Tensor> {
    let (_z, e, denom) = softmax_parts(op, x, axis)?;
    e.div(&denom)
}

/// Relabel every backend error carrying an operation name at the public seam.
fn relabel_fused_softmax(e: Error) -> Error {
    const OP: &str = "softmax";
    match e {
        Error::ShapeMismatch { lhs, rhs, .. } => Error::ShapeMismatch { op: OP, lhs, rhs },
        Error::RankMismatch { expected, got, .. } => Error::RankMismatch {
            op: OP,
            expected,
            got,
        },
        Error::InvalidAxis { axis, rank, .. } => Error::InvalidAxis { op: OP, axis, rank },
        Error::DTypeMismatch { expected, got, .. } => Error::DTypeMismatch {
            op: OP,
            expected,
            got,
        },
        Error::DeviceMismatch { expected, got, .. } => Error::DeviceMismatch {
            op: OP,
            expected,
            got,
        },
        Error::ReshapeMismatch { from, to, .. } => Error::ReshapeMismatch { op: OP, from, to },
        Error::IndexOutOfBounds {
            index, axis, size, ..
        } => Error::IndexOutOfBounds {
            op: OP,
            index,
            axis,
            size,
        },
        Error::Unsupported { device, dtype, .. } => Error::Unsupported {
            op: OP,
            device,
            dtype,
        },
        Error::NotTraced { .. } => Error::NotTraced { op: OP },
        Error::InvalidArg { msg, .. } => Error::InvalidArg { op: OP, msg },
        Error::Backend { msg, .. } => Error::Backend { op: OP, msg },
        other => other,
    }
}

/// Attempt the fused contract only for its initial production scope. `None`
/// means the caller must run the composed implementation unchanged.
fn try_fused_softmax(x: &Tensor, axis: usize) -> Result<Option<Tensor>> {
    if axis + 1 != x.rank() || !matches!(x.dtype(), DType::F32 | DType::F64) {
        return Ok(None);
    }

    let mut outputs = match dispatch::backend(x.device()).fused(FusedOp::Softmax, &[x.view()], &[])
    {
        Ok(outputs) => outputs,
        Err(Error::Unsupported { .. }) => return Ok(None),
        Err(e) => return Err(relabel_fused_softmax(e)),
    };
    if outputs.len() != 1 {
        return Err(Error::Backend {
            op: "softmax",
            msg: format!(
                "fused softmax returned {} outputs, expected exactly one",
                outputs.len()
            ),
        });
    }
    let storage = outputs.pop().expect("length checked");
    let layout = Layout::contiguous(x.dims())?;
    Ok(Some(Tensor::from_parts(storage, layout)))
}

/// `dx = y * (g - sum(g * y, axis, keepdim=true))`.
fn softmax_backward(g: &Tensor, y: &Tensor, axis: usize) -> Result<Tensor> {
    let projected = g.mul(y)?.sum_keepdim(axis as isize)?;
    y.mul(&g.sub(&projected)?)
}

/// One index reduction (`argmax`/`argmin`). Never differentiable: the result
/// is an [`I64`](crate::DType::I64) tensor of positions, so it does not go
/// through the record seam at all.
fn arg_reduce(
    op: &'static str,
    kind: ArgReduceOp,
    x: &Tensor,
    axis: isize,
    keepdim: bool,
) -> Result<Tensor> {
    let ax = x.shape().resolve_axis(axis, op)?;
    require_non_empty(op, x, ax)?;
    let storage = dispatch::backend(x.device())
        .arg_reduce(kind, x.view(), ax)
        .map_err(|e| relabel(op, e))?;
    let layout = Layout::contiguous(out_dims(x.dims(), ax, keepdim))?;
    Ok(Tensor::from_parts(storage, layout))
}

impl Tensor {
    // ---- sum -------------------------------------------------------------

    /// Sum over `axis`, dropping it (negative indexing allowed).
    ///
    /// Summing an **empty** axis returns zeros — the identity element, the one
    /// exception to the empty-reduction policy in the module docs.
    ///
    /// # Errors
    /// [`Error::InvalidAxis`] if `axis` is out of range, [`Error::Unsupported`]
    /// on a dtype without a sum (e.g. [`Bool`](crate::DType::Bool)).
    ///
    /// ```
    /// use rstorch::{Device, Tensor};
    /// let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], &Device::Cpu)?;
    /// assert_eq!(x.sum(1)?.to_vec::<f32>()?, vec![6.0, 15.0]);
    /// assert_eq!(x.sum(0)?.to_vec::<f32>()?, vec![5.0, 7.0, 9.0]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn sum(&self, axis: isize) -> Result<Tensor> {
        const OP: &str = "sum";
        let ax = self.shape().resolve_axis(axis, OP)?;
        axis_reduce(OP, ReduceOp::Sum, self, ax, false)
    }

    /// Sum over `axis`, keeping it at size 1 so the result still broadcasts
    /// against the input.
    ///
    /// # Errors
    /// As [`sum`](Tensor::sum).
    ///
    /// ```
    /// use rstorch::{Device, Tensor};
    /// let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], &Device::Cpu)?;
    /// let s = x.sum_keepdim(-1)?;
    /// assert_eq!(s.dims(), &[2, 1]);
    /// assert_eq!(x.div(&s)?.dims(), &[2, 3]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn sum_keepdim(&self, axis: isize) -> Result<Tensor> {
        const OP: &str = "sum_keepdim";
        let ax = self.shape().resolve_axis(axis, OP)?;
        axis_reduce(OP, ReduceOp::Sum, self, ax, true)
    }

    /// Sum every element into a rank-0 scalar. An empty tensor sums to zero.
    ///
    /// # Errors
    /// [`Error::Unsupported`] on a dtype without a sum.
    pub fn sum_all(&self) -> Result<Tensor> {
        fold_all("sum_all", ReduceOp::Sum, self)
    }

    // ---- mean ------------------------------------------------------------

    /// Arithmetic mean over `axis`, dropping it.
    ///
    /// # Errors
    /// [`Error::InvalidAxis`] out of range, [`Error::InvalidArg`] on an empty
    /// axis (no identity — see the module docs), [`Error::Unsupported`] on a
    /// dtype without arithmetic.
    pub fn mean(&self, axis: isize) -> Result<Tensor> {
        const OP: &str = "mean";
        let ax = self.shape().resolve_axis(axis, OP)?;
        axis_reduce(OP, ReduceOp::Mean, self, ax, false)
    }

    /// Arithmetic mean over `axis`, keeping it at size 1.
    ///
    /// # Errors
    /// As [`mean`](Tensor::mean).
    pub fn mean_keepdim(&self, axis: isize) -> Result<Tensor> {
        const OP: &str = "mean_keepdim";
        let ax = self.shape().resolve_axis(axis, OP)?;
        axis_reduce(OP, ReduceOp::Mean, self, ax, true)
    }

    /// Mean of every element, as a rank-0 scalar.
    ///
    /// Computed as `sum_all / n` (one division at the end) rather than as a
    /// cascade of per-axis means.
    ///
    /// # Errors
    /// [`Error::InvalidArg`] on an empty tensor, [`Error::Unsupported`] on a
    /// dtype without arithmetic.
    pub fn mean_all(&self) -> Result<Tensor> {
        const OP: &str = "mean_all";
        require_non_empty_all(OP, self)?;
        self.sum_all()?.div_scalar(self.num_elements() as f64)
    }

    // ---- max / min -------------------------------------------------------

    /// Maximum over `axis`, dropping it. The backward splits the cotangent
    /// evenly between tied maxima.
    ///
    /// For the *positions* of the maxima use [`argmax`](Tensor::argmax).
    ///
    /// # Errors
    /// [`Error::InvalidAxis`] out of range, [`Error::InvalidArg`] on an empty
    /// axis, [`Error::Unsupported`] on a dtype without an order (e.g.
    /// [`Bool`](crate::DType::Bool)).
    ///
    /// ```
    /// use rstorch::{Device, Tensor};
    /// let x = Tensor::from_vec(vec![1.0f32, 5.0, 3.0, 4.0], [2, 2], &Device::Cpu)?;
    /// assert_eq!(x.max(1)?.to_vec::<f32>()?, vec![5.0, 4.0]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn max(&self, axis: isize) -> Result<Tensor> {
        const OP: &str = "max";
        let ax = self.shape().resolve_axis(axis, OP)?;
        axis_reduce(OP, ReduceOp::Max, self, ax, false)
    }

    /// Maximum over `axis`, keeping it at size 1.
    ///
    /// # Errors
    /// As [`max`](Tensor::max).
    pub fn max_keepdim(&self, axis: isize) -> Result<Tensor> {
        const OP: &str = "max_keepdim";
        let ax = self.shape().resolve_axis(axis, OP)?;
        axis_reduce(OP, ReduceOp::Max, self, ax, true)
    }

    /// Maximum over every element, as a rank-0 scalar.
    ///
    /// # Errors
    /// As [`max`](Tensor::max) (an empty tensor is [`Error::InvalidArg`]).
    pub fn max_all(&self) -> Result<Tensor> {
        fold_all("max_all", ReduceOp::Max, self)
    }

    /// Minimum over `axis`, dropping it. The backward splits the cotangent
    /// evenly between tied minima.
    ///
    /// # Errors
    /// As [`max`](Tensor::max).
    pub fn min(&self, axis: isize) -> Result<Tensor> {
        const OP: &str = "min";
        let ax = self.shape().resolve_axis(axis, OP)?;
        axis_reduce(OP, ReduceOp::Min, self, ax, false)
    }

    /// Minimum over `axis`, keeping it at size 1.
    ///
    /// # Errors
    /// As [`max`](Tensor::max).
    pub fn min_keepdim(&self, axis: isize) -> Result<Tensor> {
        const OP: &str = "min_keepdim";
        let ax = self.shape().resolve_axis(axis, OP)?;
        axis_reduce(OP, ReduceOp::Min, self, ax, true)
    }

    /// Minimum over every element, as a rank-0 scalar.
    ///
    /// # Errors
    /// As [`max`](Tensor::max).
    pub fn min_all(&self) -> Result<Tensor> {
        fold_all("min_all", ReduceOp::Min, self)
    }

    // ---- var / std -------------------------------------------------------

    /// Variance over `axis` with **`correction = 1`** (Bessel's correction —
    /// PyTorch's default), dropping the axis.
    ///
    /// Float-only, and the axis must hold at least two elements: with
    /// `correction = 1` a single sample has no unbiased variance, and this
    /// returns a structured error instead of `NaN`.
    ///
    /// # Errors
    /// [`Error::InvalidAxis`] out of range, [`Error::Unsupported`] on a
    /// non-float dtype, [`Error::InvalidArg`] if the axis holds fewer than two
    /// elements.
    ///
    /// ```
    /// use rstorch::{Device, Tensor};
    /// let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [4], &Device::Cpu)?;
    /// // mean 2.5, Σ(x−x̄)² = 5, divided by n−1 = 3.
    /// assert!((x.var(0)?.item()? - 5.0 / 3.0).abs() < 1e-6);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn var(&self, axis: isize) -> Result<Tensor> {
        const OP: &str = "var";
        let ax = self.shape().resolve_axis(axis, OP)?;
        variance(OP, self, ax, false)
    }

    /// Variance over `axis` (`correction = 1`), keeping the axis at size 1.
    ///
    /// # Errors
    /// As [`var`](Tensor::var).
    pub fn var_keepdim(&self, axis: isize) -> Result<Tensor> {
        const OP: &str = "var_keepdim";
        let ax = self.shape().resolve_axis(axis, OP)?;
        variance(OP, self, ax, true)
    }

    /// Variance of every element (`correction = 1`), as a rank-0 scalar.
    ///
    /// # Errors
    /// As [`var`](Tensor::var) (fewer than two elements is
    /// [`Error::InvalidArg`]).
    pub fn var_all(&self) -> Result<Tensor> {
        variance_all("var_all", self)
    }

    /// Standard deviation over `axis` (`correction = 1`), dropping the axis —
    /// the square root of [`var`](Tensor::var).
    ///
    /// # Errors
    /// As [`var`](Tensor::var).
    pub fn std(&self, axis: isize) -> Result<Tensor> {
        const OP: &str = "std";
        let ax = self.shape().resolve_axis(axis, OP)?;
        variance(OP, self, ax, false)?.sqrt()
    }

    /// Standard deviation over `axis` (`correction = 1`), keeping the axis at
    /// size 1.
    ///
    /// # Errors
    /// As [`var`](Tensor::var).
    pub fn std_keepdim(&self, axis: isize) -> Result<Tensor> {
        const OP: &str = "std_keepdim";
        let ax = self.shape().resolve_axis(axis, OP)?;
        variance(OP, self, ax, true)?.sqrt()
    }

    /// Standard deviation of every element (`correction = 1`), as a rank-0
    /// scalar.
    ///
    /// # Errors
    /// As [`var`](Tensor::var).
    pub fn std_all(&self) -> Result<Tensor> {
        variance_all("std_all", self)?.sqrt()
    }

    // ---- softmax ---------------------------------------------------------

    /// Softmax over `axis`: `exp(xᵢ − max x) / Σ exp(x − max x)`.
    ///
    /// Numerically stable by construction (the max shift cannot overflow) and
    /// **mask-aware**: a line that is entirely `-inf` — the fully-masked
    /// attention row — produces zeros rather than `NaN`.
    ///
    /// # Errors
    /// [`Error::InvalidAxis`] out of range, [`Error::InvalidArg`] on an empty
    /// axis, [`Error::Unsupported`] on a non-float dtype.
    ///
    /// ```
    /// use rstorch::{Device, Tensor};
    /// let x = Tensor::from_vec(vec![1.0f32, 1.0, 1.0, 1.0], [2, 2], &Device::Cpu)?;
    /// assert_eq!(x.softmax(-1)?.to_vec::<f32>()?, vec![0.5, 0.5, 0.5, 0.5]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn softmax(&self, axis: isize) -> Result<Tensor> {
        const OP: &str = "softmax";
        let ax = self.shape().resolve_axis(axis, OP)?;
        require_float(OP, self)?;
        require_non_empty(OP, self, ax)?;
        let Some(out) = try_fused_softmax(self, ax)? else {
            return composed_softmax(OP, self, ax);
        };
        let y = out.detach();
        Ok(record(
            OP,
            out,
            &[self],
            Box::new(move |g| vec![softmax_backward(g, &y, ax).ok()]),
        ))
    }

    /// Log-softmax over `axis`: `(xᵢ − max x) − ln Σ exp(x − max x)`.
    ///
    /// The stable spelling — never `ln(softmax(x))`, which loses precision in
    /// the tail. A fully-masked (`-inf`) line yields `-inf`, the logarithm of
    /// the zeros [`softmax`](Tensor::softmax) gives for the same input.
    ///
    /// # Errors
    /// As [`softmax`](Tensor::softmax).
    pub fn log_softmax(&self, axis: isize) -> Result<Tensor> {
        const OP: &str = "log_softmax";
        let ax = self.shape().resolve_axis(axis, OP)?;
        let (z, _e, denom) = softmax_parts(OP, self, ax)?;
        z.sub(&denom.ln()?)
    }

    // ---- argmax / argmin -------------------------------------------------

    /// Positions of the maxima along `axis` as an [`I64`](crate::DType::I64)
    /// tensor, dropping the axis. The **first** occurrence wins a tie.
    ///
    /// Not differentiable — an index has no gradient — so it never enters the
    /// autograd graph. There is no `_all` spelling: flatten first
    /// (`x.reshape([x.num_elements()])?.argmax(0)`).
    ///
    /// `NaN` orders below every number, so it is only selected when the whole
    /// line is `NaN`.
    ///
    /// # Errors
    /// [`Error::InvalidAxis`] out of range, [`Error::InvalidArg`] on an empty
    /// axis, [`Error::Unsupported`] on a dtype without an order.
    ///
    /// ```
    /// use rstorch::{DType, Device, Tensor};
    /// let x = Tensor::from_vec(vec![1.0f32, 5.0, 3.0, 4.0], [2, 2], &Device::Cpu)?;
    /// let idx = x.argmax(1)?;
    /// assert_eq!(idx.dtype(), DType::I64);
    /// assert_eq!(idx.to_vec::<i64>()?, vec![1, 1]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn argmax(&self, axis: isize) -> Result<Tensor> {
        arg_reduce("argmax", ArgReduceOp::ArgMax, self, axis, false)
    }

    /// Positions of the maxima along `axis`, keeping the axis at size 1 — the
    /// shape [`gather`](Tensor::gather) wants.
    ///
    /// # Errors
    /// As [`argmax`](Tensor::argmax).
    pub fn argmax_keepdim(&self, axis: isize) -> Result<Tensor> {
        arg_reduce("argmax_keepdim", ArgReduceOp::ArgMax, self, axis, true)
    }

    /// Positions of the minima along `axis` as an [`I64`](crate::DType::I64)
    /// tensor, dropping the axis. The **first** occurrence wins a tie.
    ///
    /// `NaN` orders below every number, so a `NaN` in the line is selected.
    ///
    /// # Errors
    /// As [`argmax`](Tensor::argmax).
    pub fn argmin(&self, axis: isize) -> Result<Tensor> {
        arg_reduce("argmin", ArgReduceOp::ArgMin, self, axis, false)
    }

    /// Positions of the minima along `axis`, keeping the axis at size 1.
    ///
    /// # Errors
    /// As [`argmax`](Tensor::argmax).
    pub fn argmin_keepdim(&self, axis: isize) -> Result<Tensor> {
        arg_reduce("argmin_keepdim", ArgReduceOp::ArgMin, self, axis, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::dtype::DType;
    use crate::shape::Shape;
    use crate::testing::check_grad;

    const CPU: Device = Device::Cpu;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn t(data: &[f32], shape: impl Into<Shape>) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape, &CPU).unwrap()
    }

    fn v(x: &Tensor) -> Vec<f32> {
        x.to_vec::<f32>().unwrap()
    }

    fn close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length: {a:?} vs {b:?}");
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= tol, "{a:?} vs {b:?}");
        }
    }

    /// Re-view `x` through `layout` — the way this file's tests build
    /// non-contiguous inputs without depending on more than T21.
    fn re_view(x: &Tensor, layout: Layout) -> Tensor {
        Tensor::from_parts(x.storage().clone(), layout)
    }

    // ------------------------------------------------------------------
    // Forward: the three spellings
    // ------------------------------------------------------------------

    #[test]
    fn sum_mean_max_min_over_each_axis() {
        // [[1, 2, 3], [4, 5, 6]]
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);

        assert_eq!(v(&x.sum(0).unwrap()), vec![5.0, 7.0, 9.0]);
        assert_eq!(v(&x.sum(1).unwrap()), vec![6.0, 15.0]);
        assert_eq!(v(&x.mean(0).unwrap()), vec![2.5, 3.5, 4.5]);
        assert_eq!(v(&x.mean(1).unwrap()), vec![2.0, 5.0]);
        assert_eq!(v(&x.max(0).unwrap()), vec![4.0, 5.0, 6.0]);
        assert_eq!(v(&x.max(1).unwrap()), vec![3.0, 6.0]);
        assert_eq!(v(&x.min(0).unwrap()), vec![1.0, 2.0, 3.0]);
        assert_eq!(v(&x.min(1).unwrap()), vec![1.0, 4.0]);

        // The reduced axis is gone.
        assert_eq!(x.sum(0).unwrap().dims(), &[3]);
        assert_eq!(x.sum(1).unwrap().dims(), &[2]);
    }

    #[test]
    fn keepdim_holds_the_axis_at_one() {
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        for (kd, dropped) in [
            (x.sum_keepdim(1).unwrap(), x.sum(1).unwrap()),
            (x.mean_keepdim(1).unwrap(), x.mean(1).unwrap()),
            (x.max_keepdim(1).unwrap(), x.max(1).unwrap()),
            (x.min_keepdim(1).unwrap(), x.min(1).unwrap()),
        ] {
            assert_eq!(kd.dims(), &[2, 1]);
            assert_eq!(v(&kd), v(&dropped));
        }
        assert_eq!(x.sum_keepdim(0).unwrap().dims(), &[1, 3]);
        // The point of keepdim: the result broadcasts back against the input.
        assert_eq!(x.div(&x.sum_keepdim(-1).unwrap()).unwrap().dims(), &[2, 3]);
    }

    #[test]
    fn all_variants_reduce_to_a_rank_zero_scalar() {
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        for (out, expected) in [
            (x.sum_all().unwrap(), 21.0),
            (x.mean_all().unwrap(), 3.5),
            (x.max_all().unwrap(), 6.0),
            (x.min_all().unwrap(), 1.0),
        ] {
            assert_eq!(out.rank(), 0);
            assert_eq!(out.num_elements(), 1);
            assert_eq!(out.item().unwrap(), expected);
        }

        // A rank-0 input is already its own reduction.
        let s = t(&[7.0], ());
        assert_eq!(s.sum_all().unwrap().item().unwrap(), 7.0);
        assert_eq!(s.max_all().unwrap().item().unwrap(), 7.0);
        assert_eq!(s.mean_all().unwrap().item().unwrap(), 7.0);

        // Rank 3, so the fold runs more than twice.
        let x = t(&(0..24).map(|i| i as f32).collect::<Vec<_>>(), [2, 3, 4]);
        assert_eq!(x.sum_all().unwrap().item().unwrap(), 276.0);
        assert_eq!(x.max_all().unwrap().item().unwrap(), 23.0);
        assert_eq!(x.min_all().unwrap().item().unwrap(), 0.0);
        assert_eq!(x.mean_all().unwrap().item().unwrap(), 11.5);
    }

    #[test]
    fn negative_axes_count_from_the_end() {
        let x = t(&(0..24).map(|i| i as f32).collect::<Vec<_>>(), [2, 3, 4]);
        assert_eq!(v(&x.sum(-1).unwrap()), v(&x.sum(2).unwrap()));
        assert_eq!(v(&x.mean(-3).unwrap()), v(&x.mean(0).unwrap()));
        assert_eq!(
            v(&x.max_keepdim(-2).unwrap()),
            v(&x.max_keepdim(1).unwrap())
        );
        assert_eq!(
            x.argmin(-1).unwrap().to_vec::<i64>().unwrap(),
            x.argmin(2).unwrap().to_vec::<i64>().unwrap()
        );
    }

    #[test]
    fn rank_one_reductions_produce_scalars() {
        let x = t(&[2.0, 5.0, 1.0, 4.0], [4]);
        assert_eq!(x.sum(0).unwrap().rank(), 0);
        assert_eq!(x.sum(0).unwrap().item().unwrap(), 12.0);
        assert_eq!(x.max(0).unwrap().item().unwrap(), 5.0);
        assert_eq!(x.min(0).unwrap().item().unwrap(), 1.0);
        assert_eq!(x.mean(0).unwrap().item().unwrap(), 3.0);
        // keepdim keeps the rank instead.
        assert_eq!(x.sum_keepdim(0).unwrap().dims(), &[1]);
    }

    #[test]
    fn reductions_walk_strided_and_broadcast_views() {
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);

        // Transposed view: [[1,4],[2,5],[3,6]].
        let tr = re_view(&x, x.layout().transpose(0, 1).unwrap());
        assert!(!tr.is_contiguous());
        assert_eq!(v(&tr.sum(1).unwrap()), vec![5.0, 7.0, 9.0]);
        assert_eq!(v(&tr.max(1).unwrap()), vec![4.0, 5.0, 6.0]);

        // Narrowed view (non-zero offset).
        let mid = re_view(&x, x.layout().narrow(1, 1, 2).unwrap());
        assert_eq!(v(&mid.sum(1).unwrap()), vec![5.0, 11.0]);

        // Broadcast (stride-0) view: the repeats count.
        let row = t(&[1.0, 2.0, 3.0], [1, 3]);
        let b = re_view(
            &row,
            row.layout().broadcast_to(&Shape::from([4, 3])).unwrap(),
        );
        assert_eq!(v(&b.sum(0).unwrap()), vec![4.0, 8.0, 12.0]);
        assert_eq!(v(&b.mean(0).unwrap()), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn integer_reductions_stay_integer() {
        let x = Tensor::from_vec(vec![1i64, 2, 3, 4, 5, 6], [2, 3], &CPU).unwrap();
        let s = x.sum(1).unwrap();
        assert_eq!(s.dtype(), DType::I64);
        assert_eq!(s.to_vec::<i64>().unwrap(), vec![6, 15]);
        assert_eq!(x.max_all().unwrap().to_scalar::<i64>().unwrap(), 6);
        assert_eq!(x.argmax(1).unwrap().to_vec::<i64>().unwrap(), vec![2, 2]);
    }

    // ------------------------------------------------------------------
    // Forward: var / std
    // ------------------------------------------------------------------

    #[test]
    fn var_and_std_use_correction_one() {
        // mean 2.5; Σ(x−x̄)² = 2.25 + 0.25 + 0.25 + 2.25 = 5; /(4−1).
        let x = t(&[1.0, 2.0, 3.0, 4.0], [4]);
        let expected = 5.0f32 / 3.0;
        close(
            &[x.var(0).unwrap().item().unwrap() as f32],
            &[expected],
            1e-6,
        );
        close(
            &[x.std(0).unwrap().item().unwrap() as f32],
            &[expected.sqrt()],
            1e-6,
        );
        close(
            &[x.var_all().unwrap().item().unwrap() as f32],
            &[expected],
            1e-6,
        );
        close(
            &[x.std_all().unwrap().item().unwrap() as f32],
            &[expected.sqrt()],
            1e-6,
        );

        // Per row, and the keepdim spelling.
        let x = t(&[1.0, 2.0, 3.0, 10.0, 12.0, 14.0], [2, 3]);
        close(&v(&x.var(1).unwrap()), &[1.0, 4.0], 1e-6);
        close(&v(&x.std(1).unwrap()), &[1.0, 2.0], 1e-6);
        assert_eq!(x.var_keepdim(1).unwrap().dims(), &[2, 1]);
        assert_eq!(x.std_keepdim(1).unwrap().dims(), &[2, 1]);
        close(&v(&x.var_keepdim(1).unwrap()), &[1.0, 4.0], 1e-6);

        // A constant line has zero variance (and the sqrt of it).
        let c = t(&[3.0, 3.0, 3.0], [3]);
        assert_eq!(c.var(0).unwrap().item().unwrap(), 0.0);
        assert_eq!(c.std(0).unwrap().item().unwrap(), 0.0);
    }

    #[test]
    fn var_needs_two_samples_and_a_float_dtype() {
        let one = t(&[1.0, 2.0, 3.0], [3, 1]);
        // correction=1 over an axis of length 1: loud, not NaN.
        for r in [one.var(1), one.std(1), one.var_keepdim(1)] {
            assert!(matches!(r, Err(Error::InvalidArg { .. })), "expected err");
        }
        let single = t(&[1.0], [1]);
        assert!(matches!(single.var_all(), Err(Error::InvalidArg { .. })));

        let ints = Tensor::from_vec(vec![1i64, 2, 3], [3], &CPU).unwrap();
        assert!(matches!(
            ints.var(0),
            Err(Error::Unsupported { op: "var", .. })
        ));
        assert!(matches!(
            ints.std_all(),
            Err(Error::Unsupported { op: "std_all", .. })
        ));
    }

    // ------------------------------------------------------------------
    // Forward: softmax / log_softmax
    // ------------------------------------------------------------------

    #[test]
    fn softmax_normalizes_each_line() {
        let x = t(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], [2, 3]);
        let s = x.softmax(-1).unwrap();
        assert_eq!(s.dims(), &[2, 3]);
        // Hand-computed: exp shifted by the row max.
        let e = [(-2.0f32).exp(), (-1.0f32).exp(), 1.0];
        let z: f32 = e.iter().sum();
        close(&v(&s)[..3], &[e[0] / z, e[1] / z, e[2] / z], 1e-6);
        close(&v(&s)[3..], &[1.0 / 3.0; 3], 1e-6);
        // Every row sums to one.
        close(&v(&s.sum(-1).unwrap()), &[1.0, 1.0], 1e-6);

        // Softmax over the *leading* axis normalizes columns instead.
        let s0 = x.softmax(0).unwrap();
        close(&v(&s0.sum(0).unwrap()), &[1.0, 1.0, 1.0], 1e-6);
    }

    #[test]
    fn fused_softmax_scope_and_numerics_match_the_composed_path() {
        let x = t(&[1000.0, 1001.0, 1002.0, -3.0, 0.5, 7.0], [2, 3]);
        let expected = composed_softmax("softmax", &x, 1).unwrap();
        let fused = try_fused_softmax(&x, 1).unwrap().unwrap();
        assert!(fused.is_contiguous());
        assert_eq!(fused.dims(), x.dims());
        close(&v(&fused), &v(&expected), 1e-6);
        close(&v(&x.softmax(-1).unwrap()), &v(&expected), 1e-6);

        let x64 = Tensor::from_vec(vec![1000.0f64, 1001.0, 1002.0], [1, 3], &CPU).unwrap();
        let expected64 = composed_softmax("softmax", &x64, 1)
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        let fused64 = try_fused_softmax(&x64, 1)
            .unwrap()
            .unwrap()
            .to_vec::<f64>()
            .unwrap();
        for (got, expected) in fused64.iter().zip(expected64) {
            assert!((got - expected).abs() <= 1e-15, "{fused64:?}");
        }
    }

    #[test]
    fn non_last_axis_and_reduced_precision_keep_the_composed_fallback() {
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        assert!(try_fused_softmax(&x, 0).unwrap().is_none());
        close(
            &v(&x.softmax(0).unwrap()),
            &v(&composed_softmax("softmax", &x, 0).unwrap()),
            1e-6,
        );

        let f16 = Tensor::from_vec(
            [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
                .map(half::f16::from_f32)
                .to_vec(),
            [2, 3],
            &CPU,
        )
        .unwrap();
        assert!(try_fused_softmax(&f16, 1).unwrap().is_none());
        assert_eq!(
            f16.softmax(-1).unwrap().to_vec::<half::f16>().unwrap(),
            composed_softmax("softmax", &f16, 1)
                .unwrap()
                .to_vec::<half::f16>()
                .unwrap()
        );

        let bf16 = Tensor::from_vec(
            [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
                .map(half::bf16::from_f32)
                .to_vec(),
            [2, 3],
            &CPU,
        )
        .unwrap();
        assert!(try_fused_softmax(&bf16, 1).unwrap().is_none());
        assert_eq!(
            bf16.softmax(-1).unwrap().to_vec::<half::bf16>().unwrap(),
            composed_softmax("softmax", &bf16, 1)
                .unwrap()
                .to_vec::<half::bf16>()
                .unwrap()
        );
    }

    #[test]
    fn fused_softmax_reads_a_strided_input_and_returns_contiguous_output() {
        let base = t(&[1.0, 10.0, 2.0, 20.0, 3.0, 30.0], [3, 2]);
        let x = base.transpose(0, 1).unwrap();
        assert!(!x.is_contiguous());
        let expected = composed_softmax("softmax", &x, 1).unwrap();
        let got = x.softmax(-1).unwrap();
        assert!(got.is_contiguous());
        assert_eq!(got.dims(), &[2, 3]);
        close(&v(&got), &v(&expected), 1e-6);
    }

    #[test]
    fn softmax_is_stable_and_shift_invariant() {
        // The naive exp of these overflows to +inf; the max shift does not.
        let big = t(&[1000.0, 1000.0, 1000.0], [3]);
        close(&v(&big.softmax(0).unwrap()), &[1.0 / 3.0; 3], 1e-6);

        let x = t(&[-3.0, 0.5, 2.0, 7.0], [4]);
        let shifted = x.add_scalar(500.0).unwrap();
        close(
            &v(&x.softmax(0).unwrap()),
            &v(&shifted.softmax(0).unwrap()),
            1e-6,
        );
        // Very negative logits underflow to exactly zero, not to NaN.
        let tiny = t(&[-1000.0, 0.0], [2]);
        close(&v(&tiny.softmax(0).unwrap()), &[0.0, 1.0], 1e-6);
    }

    #[test]
    fn softmax_is_mask_aware() {
        // Row 0 is fully masked, row 1 keeps its first two logits.
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let mask =
            Tensor::from_vec(vec![true, true, true, false, false, true], [2, 3], &CPU).unwrap();
        let masked = x.masked_fill(&mask, f64::NEG_INFINITY).unwrap();

        let s = masked.softmax(-1).unwrap();
        // The fully-masked row is zeros, not NaN.
        assert_eq!(&v(&s)[..3], &[0.0, 0.0, 0.0]);
        // The partially-masked row renormalizes over its live entries
        // (logits 4 and 5, shifted by the row max of 5).
        let e = [(-1.0f32).exp(), 1.0f32];
        let z: f32 = e.iter().sum();
        close(&v(&s)[3..], &[e[0] / z, e[1] / z, 0.0], 1e-6);

        // log_softmax answers -inf where softmax answers 0.
        let l = masked.log_softmax(-1).unwrap();
        assert!(v(&l)[..3].iter().all(|p| *p == f32::NEG_INFINITY));
        close(&v(&l)[3..5], &[(e[0] / z).ln(), (e[1] / z).ln()], 1e-6);
        assert_eq!(v(&l)[5], f32::NEG_INFINITY);
    }

    #[test]
    fn fused_softmax_backward_is_exact_and_masked_rows_stay_zero() {
        let x = t(
            &[
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.3,
                -1.2,
                2.0,
            ],
            [2, 3],
        )
        .traced()
        .unwrap();
        let w = t(&[0.25, 0.75, -0.5, 1.0, -2.0, 0.5], [2, 3]);
        let y = x.softmax(-1).unwrap();
        assert!(y.node().is_some());
        assert_eq!(&v(&y)[..3], &[0.0, 0.0, 0.0]);

        let expected = softmax_backward(&w, &y.detach(), 1).unwrap();
        let loss = y.mul(&w).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let got = grads.wrt_input(&x).unwrap();
        close(&v(&got), &v(&expected), 1e-6);
        assert!(v(&got)[..3].iter().all(|value| *value == 0.0));
        assert!(v(&got).iter().all(|value| !value.is_nan()));

        let plain = t(&[1.0, 2.0, 3.0], [3]).softmax(-1).unwrap();
        assert!(plain.node().is_none());
    }

    #[test]
    fn log_softmax_is_the_log_of_softmax() {
        let x = t(&[1.0, 2.0, 3.0, -1.0, 0.0, 4.0], [2, 3]);
        let expected: Vec<f32> = v(&x.softmax(-1).unwrap()).iter().map(|p| p.ln()).collect();
        close(&v(&x.log_softmax(-1).unwrap()), &expected, 1e-6);

        // Stable in the tail, where ln(softmax) would have flushed to -inf.
        let far = t(&[0.0, -200.0], [2]);
        let l = v(&far.log_softmax(0).unwrap());
        close(&l, &[0.0, -200.0], 1e-4);
        assert!(l[1].is_finite());
    }

    #[test]
    fn softmax_requires_a_float_dtype() {
        let ints = Tensor::from_vec(vec![1i64, 2, 3], [3], &CPU).unwrap();
        assert!(matches!(
            ints.softmax(0),
            Err(Error::Unsupported { op: "softmax", .. })
        ));
        assert!(matches!(
            ints.log_softmax(0),
            Err(Error::Unsupported {
                op: "log_softmax",
                ..
            })
        ));
    }

    // ------------------------------------------------------------------
    // Forward: argmax / argmin
    // ------------------------------------------------------------------

    #[test]
    fn argmax_and_argmin_report_positions() {
        let x = t(&[1.0, 5.0, 3.0, 4.0, 2.0, 0.0], [2, 3]);
        let am = x.argmax(1).unwrap();
        assert_eq!(am.dtype(), DType::I64);
        assert_eq!(am.dims(), &[2]);
        assert_eq!(am.to_vec::<i64>().unwrap(), vec![1, 0]);
        assert_eq!(x.argmin(1).unwrap().to_vec::<i64>().unwrap(), vec![0, 2]);
        assert_eq!(x.argmax(0).unwrap().to_vec::<i64>().unwrap(), vec![1, 0, 0]);

        // keepdim keeps the rank (the shape `gather` wants).
        let kd = x.argmax_keepdim(1).unwrap();
        assert_eq!(kd.dims(), &[2, 1]);
        assert_eq!(kd.to_vec::<i64>().unwrap(), vec![1, 0]);
        assert_eq!(x.argmin_keepdim(-1).unwrap().dims(), &[2, 1]);

        // First occurrence wins a tie.
        let tied = t(&[2.0, 2.0, 1.0, 1.0], [4]);
        assert_eq!(tied.argmax(0).unwrap().to_vec::<i64>().unwrap(), vec![0]);
        assert_eq!(tied.argmin(0).unwrap().to_vec::<i64>().unwrap(), vec![2]);
    }

    #[test]
    fn argmax_points_at_the_value_max_reports() {
        let x = t(&[0.5, -2.0, 7.0, 3.0, 3.5, -1.0], [2, 3]);
        let idx = x.argmax(1).unwrap().to_vec::<i64>().unwrap();
        let peaks = v(&x.max(1).unwrap());
        let rows = v(&x);
        for (row, (&i, &peak)) in idx.iter().zip(peaks.iter()).enumerate() {
            assert_eq!(rows[row * 3 + i as usize], peak);
        }
    }

    // ------------------------------------------------------------------
    // The empty-reduction policy
    // ------------------------------------------------------------------

    #[test]
    fn sum_over_an_empty_axis_is_the_identity() {
        let empty = Tensor::zeros([2, 0], DType::F32, &CPU).unwrap();
        let s = empty.sum(1).unwrap();
        assert_eq!(s.dims(), &[2]);
        assert_eq!(v(&s), vec![0.0, 0.0]);
        assert_eq!(empty.sum_keepdim(1).unwrap().dims(), &[2, 1]);
        assert_eq!(empty.sum_all().unwrap().item().unwrap(), 0.0);

        // ...including when the *other* axis is the empty one.
        let empty = Tensor::zeros([0, 3], DType::F32, &CPU).unwrap();
        assert_eq!(v(&empty.sum(0).unwrap()), vec![0.0, 0.0, 0.0]);
        assert_eq!(empty.sum_all().unwrap().item().unwrap(), 0.0);
        // Reducing the *non-empty* axis of an empty tensor stays empty.
        assert_eq!(empty.sum(1).unwrap().dims(), &[0]);
    }

    #[test]
    fn every_other_reduction_refuses_an_empty_axis() {
        let empty = Tensor::zeros([2, 0], DType::F32, &CPU).unwrap();
        let cases: Vec<(&str, Result<Tensor>)> = vec![
            ("mean", empty.mean(1)),
            ("mean_keepdim", empty.mean_keepdim(1)),
            ("max", empty.max(1)),
            ("min", empty.min(1)),
            ("max_keepdim", empty.max_keepdim(1)),
            ("var", empty.var(1)),
            ("std", empty.std(1)),
            ("softmax", empty.softmax(1)),
            ("log_softmax", empty.log_softmax(1)),
            ("argmax", empty.argmax(1)),
            ("argmin_keepdim", empty.argmin_keepdim(1)),
            ("mean_all", empty.mean_all()),
            ("max_all", empty.max_all()),
            ("min_all", empty.min_all()),
            ("var_all", empty.var_all()),
        ];
        for (name, result) in cases {
            match result {
                Err(Error::InvalidArg { op, .. }) => assert_eq!(op, name),
                other => panic!("{name}: expected InvalidArg, got {:?}", other.is_ok()),
            }
        }
    }

    // ------------------------------------------------------------------
    // Loud failures
    // ------------------------------------------------------------------

    #[test]
    fn out_of_range_axes_name_the_op() {
        let x = t(&[1.0, 2.0], [2]);
        assert!(matches!(
            x.sum(1),
            Err(Error::InvalidAxis {
                op: "sum",
                axis: 1,
                rank: 1
            })
        ));
        assert!(matches!(
            x.mean_keepdim(-2),
            Err(Error::InvalidAxis {
                op: "mean_keepdim",
                ..
            })
        ));
        assert!(matches!(
            x.softmax(3),
            Err(Error::InvalidAxis { op: "softmax", .. })
        ));
        assert!(matches!(
            x.argmax(-9),
            Err(Error::InvalidAxis { op: "argmax", .. })
        ));
        // A rank-0 tensor has no axis to reduce at all.
        assert!(matches!(
            t(&[1.0], ()).sum(0),
            Err(Error::InvalidAxis {
                op: "sum",
                rank: 0,
                ..
            })
        ));
    }

    #[test]
    fn bool_reductions_are_unsupported_and_named_after_the_caller() {
        let b = Tensor::from_vec(vec![true, false, true, true], [2, 2], &CPU).unwrap();
        assert!(matches!(
            b.sum(0),
            Err(Error::Unsupported { op: "sum", .. })
        ));
        assert!(matches!(
            b.max_keepdim(1),
            Err(Error::Unsupported {
                op: "max_keepdim",
                ..
            })
        ));
        assert!(matches!(
            b.sum_all(),
            Err(Error::Unsupported { op: "sum_all", .. })
        ));
        assert!(matches!(
            b.argmax(1),
            Err(Error::Unsupported { op: "argmax", .. })
        ));
        // The route for counting a mask is an explicit cast.
        assert_eq!(
            b.to_dtype(DType::F32)
                .unwrap()
                .sum_all()
                .unwrap()
                .item()
                .unwrap(),
            3.0
        );
    }

    // ------------------------------------------------------------------
    // Backward: the value-level pieces that do not need the engine
    // ------------------------------------------------------------------
    //
    // These tests exercise the *helpers* the backward closures are built
    // from, which are ordinary value-level functions; the finite-difference
    // cases that check the closures end to end are further down.

    #[test]
    fn spread_broadcasts_the_cotangent_back_over_the_axis() {
        // Cotangent of `sum(axis=1)` on a [2, 3] input.
        let g = t(&[1.0, 2.0], [2]);
        let out = spread(&g, 1, false, &[2, 3], None).unwrap();
        assert_eq!(out.dims(), &[2, 3]);
        assert_eq!(v(&out), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

        // The keepdim spelling receives an already-shaped cotangent.
        let g = t(&[1.0, 2.0], [2, 1]);
        let out = spread(&g, 1, true, &[2, 3], None).unwrap();
        assert_eq!(v(&out), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

        // `mean` scales by 1/n on the way.
        let g = t(&[1.0, 2.0], [2]);
        let out = spread(&g, 1, false, &[2, 3], Some(1.0 / 3.0)).unwrap();
        close(
            &v(&out),
            &[
                1.0 / 3.0,
                1.0 / 3.0,
                1.0 / 3.0,
                2.0 / 3.0,
                2.0 / 3.0,
                2.0 / 3.0,
            ],
            1e-6,
        );

        // An empty axis takes the cotangent to an empty tensor, not an error.
        let g = t(&[1.0, 2.0], [2]);
        let out = spread(&g, 1, false, &[2, 0], None).unwrap();
        assert_eq!(out.dims(), &[2, 0]);
        assert!(v(&out).is_empty());
    }

    #[test]
    fn extremum_routing_splits_ties_evenly() {
        // Row 0 has a unique max, row 1 has two tied maxima.
        let x = t(&[1.0, 5.0, 3.0, 4.0, 4.0, 2.0], [2, 3]);
        let g = t(&[10.0, 6.0], [2]);
        let out = x.max(1).unwrap();
        let routed = route_to_extrema(&g, &x, &out, 1, false).unwrap();
        assert_eq!(routed.dims(), &[2, 3]);
        assert_eq!(v(&routed), vec![0.0, 10.0, 0.0, 3.0, 3.0, 0.0]);

        // Minima route the same way.
        let out = x.min(1).unwrap();
        let routed = route_to_extrema(&g, &x, &out, 1, false).unwrap();
        assert_eq!(v(&routed), vec![10.0, 0.0, 0.0, 0.0, 0.0, 6.0]);

        // ...and the keepdim spelling takes a keepdim cotangent.
        let g = t(&[10.0, 6.0], [2, 1]);
        let out = x.max_keepdim(1).unwrap();
        let routed = route_to_extrema(&g, &x, &out, 1, true).unwrap();
        assert_eq!(v(&routed), vec![0.0, 10.0, 0.0, 3.0, 3.0, 0.0]);
    }

    /// Pins the T31 NaN decision documented on `route_to_extrema`: the
    /// forward propagates NaN, so the backward does too — the *whole* line
    /// goes NaN, and the lines beside it are untouched.
    #[test]
    fn a_nan_line_propagates_nan_through_the_max_backward() {
        // Row 0 is clean, row 1 holds a NaN, row 2 is all NaN (the 0/0 case
        // T23 flagged: nothing compares equal to the NaN extremum).
        let x = t(
            &[
                1.0,
                5.0,
                3.0,
                4.0,
                f32::NAN,
                2.0,
                f32::NAN,
                f32::NAN,
                f32::NAN,
            ],
            [3, 3],
        );
        for reduced in [x.max(1).unwrap(), x.min(1).unwrap()] {
            // The forward is NaN on both poisoned rows to begin with.
            let forward = v(&reduced);
            assert!(forward[0].is_finite(), "{forward:?}");
            assert!(forward[1].is_nan() && forward[2].is_nan(), "{forward:?}");

            let g = t(&[10.0, 6.0, 7.0], [3]);
            let routed = v(&route_to_extrema(&g, &x, &reduced, 1, false).unwrap());
            // Row 0 is unaffected by its neighbours' NaNs.
            assert!(routed[..3].iter().all(|e| e.is_finite()), "{routed:?}");
            assert!((routed[..3].iter().sum::<f32>() - 10.0).abs() < 1e-6);
            // Rows 1 and 2 are NaN everywhere, not just at the NaN element,
            // and not zero.
            assert!(routed[3..].iter().all(|e| e.is_nan()), "{routed:?}");
        }
    }

    /// The same decision seen from the public surface: a NaN input poisons
    /// the gradient the engine hands back, rather than vanishing into a zero.
    #[test]
    fn a_nan_max_poisons_the_gradient_end_to_end() {
        let x = t(&[1.0, f32::NAN, 3.0], [3]).traced().unwrap();
        let grads = x.max_all().unwrap().backward().unwrap();
        let g = grads.wrt_input(&x).unwrap().to_vec::<f32>().unwrap();
        assert!(g.iter().all(|e| e.is_nan()), "{g:?}");
    }

    // ------------------------------------------------------------------
    // Backward: finite differences against the single `check_grad` harness,
    // activated by **T31** now that T30's engine is live.
    // ------------------------------------------------------------------

    const EPS: f64 = 1e-3;
    const TOL: f64 = 1e-4;

    /// A fixed non-constant weighting, so a reduction whose output is
    /// constant (softmax rows sum to 1) still has a non-trivial scalar
    /// objective.
    fn weights(dims: &[usize]) -> Tensor {
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n).map(|i| 0.25 + 0.5 * (i as f32)).collect();
        Tensor::from_vec(data, dims.to_vec(), &CPU).unwrap()
    }

    /// Finite-difference one single-input objective at the file's fixed
    /// `EPS`/`TOL`.
    fn fd(f: impl Fn(&[Tensor]) -> Result<Tensor>, x: &Tensor) {
        check_grad(f, std::slice::from_ref(x), EPS, TOL).unwrap();
    }

    #[test]
    fn grad_sum_and_mean() {
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let w = weights(&[2]);
        fd(|xs| xs[0].sum(1)?.mul(&w)?.sum_all(), &x);
        fd(|xs| xs[0].mean(1)?.mul(&w)?.sum_all(), &x);
        fd(|xs| xs[0].sum_all(), &x);
        fd(|xs| xs[0].mean_all(), &x);

        let wk = weights(&[2, 1]);
        fd(|xs| xs[0].sum_keepdim(1)?.mul(&wk)?.sum_all(), &x);
        fd(|xs| xs[0].mean_keepdim(-1)?.mul(&wk)?.sum_all(), &x);
    }

    #[test]
    fn grad_max_and_min_route_to_the_winners() {
        // Distinct values: the cotangent goes to exactly one element per line
        // (finite differences agree only away from ties).
        let x = t(&[1.0, 5.0, 3.0, 4.0, 0.5, 2.0], [2, 3]);
        let w = weights(&[2]);
        fd(|xs| xs[0].max(1)?.mul(&w)?.sum_all(), &x);
        fd(|xs| xs[0].min(1)?.mul(&w)?.sum_all(), &x);
        fd(|xs| xs[0].max_all(), &x);
        fd(|xs| xs[0].min_all(), &x);

        let wk = weights(&[2, 1]);
        fd(|xs| xs[0].max_keepdim(1)?.mul(&wk)?.sum_all(), &x);
    }

    #[test]
    fn grad_var_and_std() {
        let x = t(&[1.0, 2.0, 4.0, 8.0, 3.0, 5.0], [2, 3]);
        let w = weights(&[2]);
        fd(|xs| xs[0].var(1)?.mul(&w)?.sum_all(), &x);
        fd(|xs| xs[0].std(1)?.mul(&w)?.sum_all(), &x);
        fd(|xs| xs[0].var_all(), &x);
        fd(|xs| xs[0].std_all(), &x);
    }

    #[test]
    fn grad_softmax_and_log_softmax() {
        let x = t(&[0.3, -1.2, 2.0, 0.7, 1.1, -0.4], [2, 3]);
        let w = weights(&[2, 3]);
        fd(|xs| xs[0].softmax(-1)?.mul(&w)?.sum_all(), &x);
        fd(|xs| xs[0].log_softmax(-1)?.mul(&w)?.sum_all(), &x);
        // Along the leading axis too.
        fd(|xs| xs[0].softmax(0)?.mul(&w)?.sum_all(), &x);
    }

    #[test]
    fn grad_flows_through_a_reduction_chain() {
        // The shape a normalization layer has: subtract the mean, divide by
        // the standard deviation, reduce to a scalar.
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 7.0], [2, 3]);
        let w = weights(&[2, 3]);
        fd(
            |xs| {
                let centered = xs[0].sub(&xs[0].mean_keepdim(-1)?)?;
                let scaled = centered.div(&xs[0].std_keepdim(-1)?)?;
                scaled.mul(&w)?.sum_all()
            },
            &x,
        );
    }
}
