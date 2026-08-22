//! Reductions: `sum`/`mean`/`max`/`min`/`prod`/`var`/`std` in three spellings
//! each, the `p`-norm `norm`, the normalizations `softmax`/`log_softmax`, and
//! the index reductions `argmax`/`argmin`.
//!
//! # Three spellings, one meaning
//!
//! Every value reduction comes as `op(axis)` (the axis is reduced away and
//! **dropped**), `op_keepdim(axis)` (the axis stays at size 1, so the result
//! still broadcasts against the input) and `op_all()` (every axis reduced, to
//! a rank-0 scalar). Axes are `isize` with negative indexing, so `-1` is
//! always the last axis.
//!
//! # Empty-reduction policy
//!
//! Reducing an axis of size 0 has an answer only where the op has an identity
//! element, and only `sum` is spelled to return one. The rule here is one
//! line:
//!
//! > **`sum`/`sum_all` return the identity (zeros); every other reduction
//! > over an empty axis is a loud [`Error::InvalidArg`] naming the op.**
//!
//! So `sum` of an empty axis is `0`, while `mean`, `max`, `min`, `prod`,
//! `var`, `std`, `norm`, `softmax`, `log_softmax`, `argmax` and `argmin`
//! refuse rather than return the `NaN`/`-inf`/arbitrary-index answers a
//! silent implementation would produce. `prod` is in that list even though ∏
//! over nothing is conventionally `1`: the identity is only returned where a
//! caller reduces an empty axis on purpose, which `sum` covers, and an empty
//! `prod` is far more often a shape mistake than a request for ones.
//! `var`/`std` extend the same rule to `correction`: with `correction = 1` an
//! axis of length 1 has no unbiased variance, so it is rejected instead of
//! returning `NaN`.
//!
//! # Numerics
//!
//! - Accumulation happens in the wide [`Acc`](crate::dtype::Element::Acc)
//!   type inside the kernel, with a single cast at output, so a long `f16`
//!   sum cannot saturate to infinity on its way to an in-range total.
//! - `var`/`std` use **`correction = 1`** (Bessel's correction), `PyTorch`'s
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
//! | `prod(axis)` | the leave-one-out product ∏_{j≠k} xⱼ, built from the product over the *non-zero* elements so that a line containing a zero is finite and correct rather than `0/0` |
//! | `var`/`std`, `norm`, `log_softmax` | none of their own — they are *composed* from recorded ops (`mean`/`sub`/`mul`/`sum`/`sqrt`/`exp`/`div`), so the engine differentiates the composition |
//! | `softmax` | the fused last-axis `F32`/`F64` path records one node with `y · (g − Σ(g · y))`; other variants retain the composed path |
//! | `argmax`/`argmin` | not differentiable ([`I64`](crate::DType::I64) output); they never reach the record seam |
//!
//! The `max`/`min` backward captures the **detached** input and output, while
//! fused softmax captures its **detached** output `y` (the detached-output
//! capture rule). Each is built before the traced output is
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
///
/// The kernels name their errors after the family (`"reduce"`), while the
/// user called `sum`, `max`, `mean`, ... — hence the `with_op` at the seam.
fn reduce_values(
    op: &'static str,
    kind: ReduceOp,
    x: &Tensor,
    axis: usize,
    keepdim: bool,
) -> Result<Tensor> {
    let storage = dispatch::backend(x.device())
        .reduce(kind, x.view(), axis)
        .map_err(|e| e.with_op(op))?;
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
/// [`Tensor::maximum`] already uses element-wise, and `PyTorch`'s `amax`/`amin`
/// rule).
///
/// `x` and `out` are the **detached** input and output; both are re-viewed
/// with the reduced axis present so they broadcast against each other.
///
/// # NaN lines
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
///   overturn it. (`PyTorch`'s `amax`/`amin` backward is the same
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
        ReduceOp::Sum => {
            Box::new(move |g| Ok(vec![Some(spread(g, axis, keepdim, &src_dims, None)?)]))
        }
        ReduceOp::Mean => {
            let scale = 1.0 / src_dims[axis] as f64;
            Box::new(move |g| {
                Ok(vec![Some(spread(
                    g,
                    axis,
                    keepdim,
                    &src_dims,
                    Some(scale),
                )?)])
            })
        }
        ReduceOp::Max | ReduceOp::Min => {
            let xd = x.detach();
            let od = out.detach();
            Box::new(move |g| Ok(vec![Some(route_to_extrema(g, &xd, &od, axis, keepdim)?)]))
        }
        // ∂∏/∂xₖ = ∏_{j≠k} xⱼ. A product is multilinear, so that is finite
        // everywhere — including at a zero, where it is generally *not* zero.
        // Dividing the whole product by xₖ computes it only when the line
        // holds no zero; at a zero that division is 0/0. So the leave-one-out
        // product is built directly: multiply the non-zero elements, then
        // pick per element on how many zeros the line has — none (every
        // position gets ∏/xₖ), exactly one (only the zero position gets the
        // product of the others), or two or more (every position gets zero,
        // since every leave-one-out product still contains a zero).
        ReduceOp::Prod => {
            let xd = x.detach();
            Box::new(move |g| {
                let g_kd = with_axis(g, axis, keepdim)?;
                let ones = xd.ones_like()?;
                let zeros = xd.zeros_like()?;
                let is_zero = xd.eq(&zeros)?;
                // Zeros replaced by the multiplicative identity: the product
                // of this is ∏ over the non-zero elements, and dividing by it
                // is always safe.
                let safe = is_zero.where_cond(&ones, &xd)?;
                let nonzero_prod = reduce_values(op, ReduceOp::Prod, &safe, axis, true)?;
                // Counted in the accumulation dtype so a long f16/bf16 line
                // cannot round its own zero count.
                let wide = xd.to_dtype(xd.dtype().accumulation_dtype())?;
                let indicator = is_zero.where_cond(&wide.ones_like()?, &wide.zeros_like()?)?;
                let count = reduce_values(op, ReduceOp::Sum, &indicator, axis, true)?;
                let exactly_one = count.eq(&count.ones_like()?)?;
                let none = count.eq(&count.zeros_like()?)?;
                let keep = is_zero.where_cond(
                    &exactly_one.where_cond(&ones, &zeros)?,
                    &none.where_cond(&ones, &zeros)?,
                )?;
                Ok(vec![Some(g_kd.mul(&nonzero_prod)?.div(&safe)?.mul(&keep)?)])
            })
        }
    };
    Ok(record(op, out, &[x], backward))
}

/// Reduce every axis to a rank-0 scalar, one axis at a time from the last to
/// the first (so the axes still to be reduced keep their indices). A rank-0
/// input is already the answer.
fn fold_all_wide(op: &'static str, kind: ReduceOp, x: &Tensor) -> Result<Tensor> {
    if !matches!(kind, ReduceOp::Sum) {
        require_non_empty_all(op, x)?;
    }
    let accumulation_dtype = x.dtype().accumulation_dtype();
    let mut cur = if accumulation_dtype != x.dtype() {
        x.to_dtype(accumulation_dtype)?
    } else {
        x.clone()
    };
    for axis in (0..x.rank()).rev() {
        cur = axis_reduce(op, kind, &cur, axis, false)?;
    }
    Ok(cur)
}

fn narrow_all(value: Tensor, dtype: DType) -> Result<Tensor> {
    if value.dtype() == dtype {
        Ok(value)
    } else {
        value.to_dtype(dtype)
    }
}

fn fold_all(op: &'static str, kind: ReduceOp, x: &Tensor) -> Result<Tensor> {
    narrow_all(fold_all_wide(op, kind, x)?, x.dtype())
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
fn variance_all_wide(op: &'static str, x: &Tensor) -> Result<Tensor> {
    require_float(op, x)?;
    let n = x.num_elements();
    require_correction(op, n)?;
    let deviation = x.sub(&x.mean_all()?)?;
    fold_all_wide(op, ReduceOp::Sum, &deviation.mul(&deviation)?)?.div_scalar((n - 1) as f64)
}

fn variance_all(op: &'static str, x: &Tensor) -> Result<Tensor> {
    narrow_all(variance_all_wide(op, x)?, x.dtype())
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

/// Attempt the fused contract only for its initial production scope. `None`
/// means the caller must run the composed implementation unchanged.
fn try_fused_softmax(x: &Tensor, axis: usize) -> Result<Option<Tensor>> {
    if axis + 1 != x.rank() || !x.dtype().is_float() {
        return Ok(None);
    }

    let mut outputs = match dispatch::backend(x.device()).fused(FusedOp::Softmax, &[x.view()], &[])
    {
        Ok(outputs) => outputs,
        Err(Error::Unsupported { .. }) => return Ok(None),
        Err(e) => return Err(e.with_op("softmax")),
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
        .map_err(|e| e.with_op(op))?;
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
        narrow_all(
            fold_all_wide(OP, ReduceOp::Sum, self)?.div_scalar(self.num_elements() as f64)?,
            self.dtype(),
        )
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

    // ---- prod --------------------------------------------------------

    /// Product over `axis`, dropping it.
    ///
    /// # Errors
    /// [`Error::InvalidAxis`] out of range, [`Error::InvalidArg`] on an empty
    /// axis (see the module docs), [`Error::Unsupported`] on a dtype without
    /// arithmetic or on any non-CPU device: `ReduceOp::Prod` has no
    /// accelerator kernel, so Metal, CUDA and WGPU all decline it.
    ///
    /// # Gradient
    /// `∂∏/∂xᵢ = ∏_{j≠i} xⱼ`, the leave-one-out product — finite everywhere,
    /// including on a line containing a zero, where it is the product of the
    /// other elements at that one position and zero elsewhere. It is built
    /// from the product over the non-zero elements rather than as `prod / xᵢ`,
    /// which would be `0/0` exactly where the interesting answer is.
    ///
    /// ```
    /// use rstorch::{Device, Tensor};
    /// let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2], &Device::Cpu)?;
    /// assert_eq!(x.prod(1)?.to_vec::<f32>()?, vec![2.0, 12.0]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn prod(&self, axis: isize) -> Result<Tensor> {
        const OP: &str = "prod";
        let ax = self.shape().resolve_axis(axis, OP)?;
        axis_reduce(OP, ReduceOp::Prod, self, ax, false)
    }

    /// Product over `axis`, keeping it at size 1.
    ///
    /// # Errors
    /// As [`prod`](Tensor::prod).
    pub fn prod_keepdim(&self, axis: isize) -> Result<Tensor> {
        const OP: &str = "prod_keepdim";
        let ax = self.shape().resolve_axis(axis, OP)?;
        axis_reduce(OP, ReduceOp::Prod, self, ax, true)
    }

    /// Product of every element, as a rank-0 scalar.
    ///
    /// # Errors
    /// As [`prod`](Tensor::prod) (an empty tensor is [`Error::InvalidArg`]).
    pub fn prod_all(&self) -> Result<Tensor> {
        fold_all("prod_all", ReduceOp::Prod, self)
    }

    // ---- var / std -------------------------------------------------------

    /// Variance over `axis` with **`correction = 1`** (Bessel's correction —
    /// `PyTorch`'s default), dropping the axis.
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
        narrow_all(variance_all_wide("std_all", self)?.sqrt()?, self.dtype())
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
            Box::new(move |g| Ok(vec![Some(softmax_backward(g, &y, ax)?)])),
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

    // ---- norm --------------------------------------------------------

    /// The `p`-norm over `axis`, dropping it: `(Σ|x|ᵖ)^(1/p)` (`p = 2.0` is
    /// the Euclidean/L2 norm the crate's own `RMSNorm` and a manual RoPE need
    /// most).
    ///
    /// Composed from ops the engine already differentiates, so the gradient
    /// comes from that composition rather than a hand-written formula.
    /// `p == 1` is [`abs`](Tensor::abs) + [`sum`](Tensor::sum) and `p == 2` is
    /// [`mul`](Tensor::mul) + `sum` + [`sqrt`](Tensor::sqrt) — both available
    /// on every backend. Any other `p` goes through [`pow`](Tensor::pow),
    /// which today has a CPU kernel only.
    ///
    /// # Errors
    /// [`Error::InvalidAxis`] out of range, [`Error::InvalidArg`] on an empty
    /// axis or a `p` that is not finite and positive (there is no `p = ∞`
    /// spelling — use [`max`](Tensor::max) over [`abs`](Tensor::abs)),
    /// [`Error::Unsupported`] on a non-float dtype, or — for a `p` other than
    /// `1` or `2`, which route through [`pow`](Tensor::pow) — on any
    /// non-CPU device.
    ///
    /// ```
    /// use rstorch::{Device, Tensor};
    /// let x = Tensor::from_vec(vec![3.0f32, 4.0], [2], &Device::Cpu)?;
    /// assert!((x.norm(0, 2.0)?.item()? - 5.0).abs() < 1e-6);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn norm(&self, axis: isize, p: f64) -> Result<Tensor> {
        const OP: &str = "norm";
        self.norm_impl(OP, axis, p, false)
    }

    /// The `p`-norm over `axis`, keeping it at size 1.
    ///
    /// # Errors
    /// As [`norm`](Tensor::norm).
    pub fn norm_keepdim(&self, axis: isize, p: f64) -> Result<Tensor> {
        const OP: &str = "norm_keepdim";
        self.norm_impl(OP, axis, p, true)
    }

    /// The `p`-norm over every element, as a rank-0 scalar — the third
    /// spelling this module promises for every value reduction.
    ///
    /// Flattening first and reducing the one axis is what makes this the
    /// *whole-tensor* norm rather than a fold of per-axis norms: `‖x‖ₚ` over
    /// all elements is `(Σ|xᵢ|ᵖ)^(1/p)`, and folding `norm` axis by axis
    /// would raise the intermediate sums to `1/p` in between.
    ///
    /// # Errors
    /// As [`norm`](Tensor::norm) (an empty tensor is [`Error::InvalidArg`]).
    pub fn norm_all(&self, p: f64) -> Result<Tensor> {
        const OP: &str = "norm_all";
        let flat = self.reshape([self.num_elements()])?;
        flat.norm_impl(OP, 0, p, false)
    }

    fn norm_impl(&self, op: &'static str, axis: isize, p: f64, keepdim: bool) -> Result<Tensor> {
        let ax = self.shape().resolve_axis(axis, op)?;
        require_float(op, self)?;
        // `p` must be finite as well as positive. `p = ∞` would otherwise
        // sail through: `|x|^∞ ∈ {0, 1, ∞}` and the outer exponent is
        // `1/∞ = 0`, and `powf(_, 0) = 1`, so the max-norm spelling people
        // reach for would silently return all ones with a zero gradient.
        if !p.is_finite() || p <= 0.0 {
            return Err(Error::InvalidArg {
                op,
                msg: format!("norm requires a finite p > 0, got {p}"),
            });
        }
        require_non_empty(op, self, ax)?;
        // `p == 1` and `p == 2` are almost every call, and neither needs
        // `pow` — which has no accelerator kernel, and would make `norm`
        // unusable on a GPU for the two exponents people actually ask for.
        // `x * x` is also a shorter and no less accurate route to `|x|²` than
        // an `abs` plus a `powf` round trip.
        let reduce_axis = |t: &Tensor| {
            if keepdim {
                t.sum_keepdim(ax as isize)
            } else {
                t.sum(ax as isize)
            }
        };
        if p == 1.0 {
            return reduce_axis(&self.abs()?);
        }
        if p == 2.0 {
            return reduce_axis(&self.mul(self)?)?.sqrt();
        }
        reduce_axis(&self.abs()?.pow(p)?)?.pow(1.0 / p)
    }

    // ---- argmax / argmin -------------------------------------------------

    /// Positions of the maxima along `axis` as an [`I64`](crate::DType::I64)
    /// tensor, dropping the axis. The **first** occurrence wins a tie.
    ///
    /// Not differentiable — an index has no gradient — so it never enters the
    /// autograd graph. There is no `_all` spelling: flatten first
    /// (`x.reshape([x.num_elements()])?.argmax(0)`).
    ///
    /// `NaN` orders **above** every number, so a line containing one selects
    /// it (the first, on ties) — the only index that agrees with
    /// [`max`](Tensor::max), which propagates `NaN`.
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
    /// `NaN` orders **above** every number here too, so a line containing one
    /// selects it rather than the smallest finite element — again agreeing
    /// with [`min`](Tensor::min), which propagates `NaN`.
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
mod tests;
