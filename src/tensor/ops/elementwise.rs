//! Element-wise ops: arithmetic (`add`/`sub`/`mul`/`div`/`pow` and their
//! scalar spellings), `maximum`/`minimum`/`clamp`, the unary math family
//! (activations, `exp`/`ln`/`sqrt`, and the rounding/sign/reciprocal/`erf`
//! set), comparisons to [`Bool`](crate::DType::Bool), `masked_fill` and
//! `where_cond`.
//!
//! Every op here follows the same three-step shape:
//!
//! 1. **Validate** device and dtype in the op layer — there is no implicit
//!    promotion and no silent device hop, so mixing either is a structured
//!    [`Error`](crate::Error) naming the public method.
//! 2. **Broadcast in the layout**, never in a kernel: the operands are
//!    right-aligned with `Shape::broadcast_with` and re-viewed at the common
//!    shape with `Layout::broadcast_to` (stride-0 axes), so the backend always
//!    receives pre-broadcast, shape-identical views.
//! 3. **Dispatch once** through `backend::dispatch::backend` and wrap the
//!    result with the `autograd::record` seam.
//!
//! # Backward conventions
//!
//! - Broadcasting is undone in the backward pass by `Tensor::sum_to`, which
//!   is exactly the transpose of `Layout::broadcast_to`.
//! - Output-dependent formulas (`exp`, `sqrt`, `tanh`, `sigmoid`, `recip`,
//!   `div`, `gelu`) capture the op's output in **detached** form, built
//!   before the traced output is assembled (the detached-output capture
//!   rule) — capturing the traced output would create an `Arc` cycle through
//!   the closure.
//! - Comparisons produce [`Bool`](crate::DType::Bool) and are therefore not
//!   differentiable: they do not go through the record seam at all. For the
//!   same reason the [`Bool`](crate::DType::Bool) operand of `masked_fill`
//!   and `where_cond` is not listed as a graph input.

use super::{require_dtype, same_device, same_dtype};
use crate::autograd;
use crate::backend::{BinaryOp, CmpOp, UnaryOp, View, dispatch};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::tensor::Tensor;

/// `1/√(2π)` — the normalization of the standard normal pdf `φ`, which is the
/// second term of the exact-GELU derivative.
const INV_SQRT_2PI: f64 = 0.5 * std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2;

// ---------------------------------------------------------------------------
// Shared plumbing
// ---------------------------------------------------------------------------

/// The untraced forward of a broadcasting binary op.
fn binary_forward(op: &'static str, kind: BinaryOp, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    same_device(op, lhs, rhs)?;
    same_dtype(op, lhs, rhs)?;
    let out_shape = lhs.shape().broadcast_with(rhs.shape(), op)?;
    let ll = lhs.layout().broadcast_to(&out_shape)?;
    let rl = rhs.layout().broadcast_to(&out_shape)?;
    let storage = dispatch::backend(lhs.device()).binary(
        kind,
        View::new(lhs.storage(), &ll),
        View::new(rhs.storage(), &rl),
    )?;
    Ok(Tensor::from_parts(storage, Layout::contiguous(out_shape)?))
}

/// The untraced forward of a scalar binary op (`x <op> scalar`). The scalar
/// broadcasts trivially, so the output keeps `x`'s shape.
///
/// Kernels name their errors after the op *family* — every `BinaryOp::Add`
/// call reports `"add"`, whichever spelling reached it — so the scalar entry
/// points rewrite the name with `Error::with_op`. Everywhere else the two
/// already coincide.
fn binary_scalar_forward(
    op: &'static str,
    kind: BinaryOp,
    x: &Tensor,
    scalar: f64,
) -> Result<Tensor> {
    let storage = dispatch::backend(x.device())
        .binary_scalar(kind, x.view(), scalar)
        .map_err(|e| e.with_op(op))?;
    Ok(Tensor::from_parts(
        storage,
        Layout::contiguous(x.shape().clone())?,
    ))
}

/// The untraced forward of a unary op.
fn unary_forward(kind: UnaryOp, x: &Tensor) -> Result<Tensor> {
    let storage = dispatch::backend(x.device()).unary(kind, x.view())?;
    Ok(Tensor::from_parts(
        storage,
        Layout::contiguous(x.shape().clone())?,
    ))
}

/// The forward of a comparison: broadcast both sides, produce
/// [`Bool`](DType::Bool). Never traced.
fn compare_forward(op: &'static str, kind: CmpOp, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    same_device(op, lhs, rhs)?;
    same_dtype(op, lhs, rhs)?;
    let out_shape = lhs.shape().broadcast_with(rhs.shape(), op)?;
    let ll = lhs.layout().broadcast_to(&out_shape)?;
    let rl = rhs.layout().broadcast_to(&out_shape)?;
    let storage = dispatch::backend(lhs.device()).compare(
        kind,
        View::new(lhs.storage(), &ll),
        View::new(rhs.storage(), &rl),
    )?;
    Ok(Tensor::from_parts(storage, Layout::contiguous(out_shape)?))
}

/// One side of the `maximum`/`minimum` backward: the cotangent where this
/// operand strictly wins, plus half of it where the two tie (`PyTorch`'s
/// tie-splitting rule), reduced back to `dims`.
fn extremum_side(g: &Tensor, win: &Tensor, tie: &Tensor, dims: &[usize]) -> Result<Tensor> {
    let zero = g.zeros_like()?;
    let full = win.where_cond(g, &zero)?;
    let half = tie.where_cond(&g.mul_scalar(0.5)?, &zero)?;
    full.add(&half)?.sum_to(dims)
}

/// The whole `maximum`/`minimum` backward, over the detached operands.
///
/// Both operands always receive a gradient — zero where they lose, half where
/// they tie — so neither slot is ever a legitimate `None`.
fn extremum_backward(
    g: &Tensor,
    a: &Tensor,
    b: &Tensor,
    a_dims: &[usize],
    b_dims: &[usize],
    is_max: bool,
) -> Result<Vec<Option<Tensor>>> {
    let side = |wins: Result<Tensor>, dims: &[usize]| -> Result<Tensor> {
        let tie = a.eq(b)?;
        extremum_side(g, &wins?, &tie, dims)
    };
    let (a_wins, b_wins) = if is_max {
        (a.gt(b), b.gt(a))
    } else {
        (a.lt(b), b.lt(a))
    };
    Ok(vec![
        Some(side(a_wins, a_dims)?),
        Some(side(b_wins, b_dims)?),
    ])
}

impl Tensor {
    // ---- arithmetic ------------------------------------------------------

    /// Element-wise sum, broadcasting `self` and `rhs` to their common shape.
    ///
    /// # Errors
    /// [`Error::DeviceMismatch`](crate::Error::DeviceMismatch) /
    /// [`Error::DTypeMismatch`](crate::Error::DTypeMismatch) when the operands
    /// disagree (no implicit transfer, no implicit promotion),
    /// [`Error::ShapeMismatch`](crate::Error::ShapeMismatch) when the shapes do not broadcast, and
    /// [`Error::Unsupported`](crate::Error::Unsupported) on a dtype without arithmetic (e.g.
    /// [`Bool`](crate::DType::Bool)).
    pub fn add(&self, rhs: &Tensor) -> Result<Tensor> {
        let out = binary_forward("add", BinaryOp::Add, self, rhs)?;
        let (ld, rd) = (self.dims().to_vec(), rhs.dims().to_vec());
        Ok(autograd::record(
            "add",
            out,
            &[self, rhs],
            Box::new(move |g| Ok(vec![Some(g.sum_to(&ld)?), Some(g.sum_to(&rd)?)])),
        ))
    }

    /// Element-wise difference, broadcasting to the common shape.
    ///
    /// # Errors
    /// As [`add`](Tensor::add).
    pub fn sub(&self, rhs: &Tensor) -> Result<Tensor> {
        let out = binary_forward("sub", BinaryOp::Sub, self, rhs)?;
        let (ld, rd) = (self.dims().to_vec(), rhs.dims().to_vec());
        Ok(autograd::record(
            "sub",
            out,
            &[self, rhs],
            Box::new(move |g| Ok(vec![Some(g.sum_to(&ld)?), Some(g.neg()?.sum_to(&rd)?)])),
        ))
    }

    /// Element-wise product, broadcasting to the common shape.
    ///
    /// # Errors
    /// As [`add`](Tensor::add).
    pub fn mul(&self, rhs: &Tensor) -> Result<Tensor> {
        let out = binary_forward("mul", BinaryOp::Mul, self, rhs)?;
        let (ld, rd) = (self.dims().to_vec(), rhs.dims().to_vec());
        let (a, b) = (self.detach(), rhs.detach());
        Ok(autograd::record(
            "mul",
            out,
            &[self, rhs],
            Box::new(move |g| {
                Ok(vec![
                    Some(g.mul(&b)?.sum_to(&ld)?),
                    Some(g.mul(&a)?.sum_to(&rd)?),
                ])
            }),
        ))
    }

    /// Element-wise quotient, broadcasting to the common shape.
    ///
    /// Integer division by zero yields `0` rather than panicking (the kernel's
    /// panic-free contract); float division follows IEEE 754.
    ///
    /// # Errors
    /// As [`add`](Tensor::add).
    pub fn div(&self, rhs: &Tensor) -> Result<Tensor> {
        let out = binary_forward("div", BinaryOp::Div, self, rhs)?;
        let (ld, rd) = (self.dims().to_vec(), rhs.dims().to_vec());
        let b = rhs.detach();
        // d(a/b)/db = -a/b² = -(a/b)/b, so the detached output does the work.
        let out_d = out.detach();
        Ok(autograd::record(
            "div",
            out,
            &[self, rhs],
            Box::new(move |g| {
                Ok(vec![
                    Some(g.div(&b)?.sum_to(&ld)?),
                    Some(g.mul(&out_d)?.div(&b)?.neg()?.sum_to(&rd)?),
                ])
            }),
        ))
    }

    /// Add a scalar to every element (narrowed to this tensor's dtype).
    ///
    /// # Errors
    /// [`Error::Unsupported`](crate::Error::Unsupported) on a dtype without arithmetic.
    pub fn add_scalar(&self, scalar: f64) -> Result<Tensor> {
        let out = binary_scalar_forward("add_scalar", BinaryOp::Add, self, scalar)?;
        Ok(autograd::record(
            "add_scalar",
            out,
            &[self],
            Box::new(move |g| Ok(vec![Some(g.clone())])),
        ))
    }

    /// Subtract a scalar from every element (`self - scalar`).
    ///
    /// # Errors
    /// As [`add_scalar`](Tensor::add_scalar).
    pub fn sub_scalar(&self, scalar: f64) -> Result<Tensor> {
        let out = binary_scalar_forward("sub_scalar", BinaryOp::Sub, self, scalar)?;
        Ok(autograd::record(
            "sub_scalar",
            out,
            &[self],
            Box::new(move |g| Ok(vec![Some(g.clone())])),
        ))
    }

    /// Scale every element by a scalar.
    ///
    /// # Errors
    /// As [`add_scalar`](Tensor::add_scalar).
    pub fn mul_scalar(&self, scalar: f64) -> Result<Tensor> {
        let out = binary_scalar_forward("mul_scalar", BinaryOp::Mul, self, scalar)?;
        Ok(autograd::record(
            "mul_scalar",
            out,
            &[self],
            Box::new(move |g| Ok(vec![Some(g.mul_scalar(scalar)?)])),
        ))
    }

    /// Divide every element by a scalar (`self / scalar`).
    ///
    /// # Errors
    /// As [`add_scalar`](Tensor::add_scalar).
    pub fn div_scalar(&self, scalar: f64) -> Result<Tensor> {
        let out = binary_scalar_forward("div_scalar", BinaryOp::Div, self, scalar)?;
        Ok(autograd::record(
            "div_scalar",
            out,
            &[self],
            Box::new(move |g| Ok(vec![Some(g.div_scalar(scalar)?)])),
        ))
    }

    /// Element-wise maximum, broadcasting to the common shape.
    ///
    /// The backward splits the cotangent evenly on ties (`PyTorch`'s
    /// `torch.maximum` rule).
    ///
    /// # Errors
    /// As [`add`](Tensor::add).
    pub fn maximum(&self, rhs: &Tensor) -> Result<Tensor> {
        let out = binary_forward("maximum", BinaryOp::Maximum, self, rhs)?;
        let (ld, rd) = (self.dims().to_vec(), rhs.dims().to_vec());
        let (a, b) = (self.detach(), rhs.detach());
        Ok(autograd::record(
            "maximum",
            out,
            &[self, rhs],
            Box::new(move |g| extremum_backward(g, &a, &b, &ld, &rd, true)),
        ))
    }

    /// Element-wise minimum, broadcasting to the common shape.
    ///
    /// The backward splits the cotangent evenly on ties (`PyTorch`'s
    /// `torch.minimum` rule).
    ///
    /// # Errors
    /// As [`add`](Tensor::add).
    pub fn minimum(&self, rhs: &Tensor) -> Result<Tensor> {
        let out = binary_forward("minimum", BinaryOp::Minimum, self, rhs)?;
        let (ld, rd) = (self.dims().to_vec(), rhs.dims().to_vec());
        let (a, b) = (self.detach(), rhs.detach());
        Ok(autograd::record(
            "minimum",
            out,
            &[self, rhs],
            Box::new(move |g| extremum_backward(g, &a, &b, &ld, &rd, false)),
        ))
    }

    // ---- unary math ------------------------------------------------------

    /// Rectified linear unit, `max(x, 0)`.
    ///
    /// The subgradient at `0` is `0`.
    ///
    /// # Errors
    /// [`Error::Unsupported`](crate::Error::Unsupported) on a dtype the kernel does not implement
    /// (`relu` is float-only).
    pub fn relu(&self) -> Result<Tensor> {
        let out = unary_forward(UnaryOp::Relu, self)?;
        let x = self.detach();
        Ok(autograd::record(
            "relu",
            out,
            &[self],
            Box::new(move |g| {
                let zero = x.zeros_like()?;
                Ok(vec![Some(x.gt(&zero)?.where_cond(g, &zero)?)])
            }),
        ))
    }

    /// **Exact** Gaussian error linear unit, `0.5·x·(1 + erf(x/√2))` — not the
    /// tanh approximation (the familiar-semantics contract).
    ///
    /// # Errors
    /// [`Error::Unsupported`](crate::Error::Unsupported) on a non-float dtype.
    pub fn gelu(&self) -> Result<Tensor> {
        let out = unary_forward(UnaryOp::Gelu, self)?;
        let x = self.detach();
        let out_d = out.detach();
        Ok(autograd::record(
            "gelu",
            out,
            &[self],
            Box::new(move |g| {
                // d/dx [x·Φ(x)] = Φ(x) + x·φ(x). The forward already computed
                // x·Φ(x), so Φ(x) is a division — with the removable
                // singularity at x == 0 (where Φ(0) = 1/2) filled in
                // explicitly. `where_cond` evaluates both branches, so the
                // NaN produced by 0/0 is computed and then discarded.
                let zero = x.zeros_like()?;
                let half = x.full_like(0.5)?;
                let cdf = x.eq(&zero)?.where_cond(&half, &out_d.div(&x)?)?;
                let pdf = x
                    .mul(&x)?
                    .mul_scalar(-0.5)?
                    .exp()?
                    .mul_scalar(INV_SQRT_2PI)?;
                Ok(vec![Some(cdf.add(&x.mul(&pdf)?)?.mul(g)?)])
            }),
        ))
    }

    /// Element-wise `exp(x)`.
    ///
    /// # Errors
    /// [`Error::Unsupported`](crate::Error::Unsupported) on a non-float dtype.
    pub fn exp(&self) -> Result<Tensor> {
        let out = unary_forward(UnaryOp::Exp, self)?;
        let out_d = out.detach();
        Ok(autograd::record(
            "exp",
            out,
            &[self],
            Box::new(move |g| Ok(vec![Some(g.mul(&out_d)?)])),
        ))
    }

    /// Element-wise natural logarithm.
    ///
    /// # Errors
    /// [`Error::Unsupported`](crate::Error::Unsupported) on a non-float dtype.
    pub fn ln(&self) -> Result<Tensor> {
        let out = unary_forward(UnaryOp::Ln, self)?;
        let x = self.detach();
        Ok(autograd::record(
            "ln",
            out,
            &[self],
            Box::new(move |g| Ok(vec![Some(g.div(&x)?)])),
        ))
    }

    /// Element-wise square root.
    ///
    /// # Errors
    /// [`Error::Unsupported`](crate::Error::Unsupported) on a non-float dtype.
    pub fn sqrt(&self) -> Result<Tensor> {
        let out = unary_forward(UnaryOp::Sqrt, self)?;
        let out_d = out.detach();
        Ok(autograd::record(
            "sqrt",
            out,
            &[self],
            // d√x/dx = 1/(2√x), and √x is the output.
            Box::new(move |g| Ok(vec![Some(g.div(&out_d)?.mul_scalar(0.5)?)])),
        ))
    }

    /// Element-wise hyperbolic tangent.
    ///
    /// # Errors
    /// [`Error::Unsupported`](crate::Error::Unsupported) on a non-float dtype.
    pub fn tanh(&self) -> Result<Tensor> {
        let out = unary_forward(UnaryOp::Tanh, self)?;
        let out_d = out.detach();
        Ok(autograd::record(
            "tanh",
            out,
            &[self],
            // 1 - tanh(x)²
            Box::new(move |g| {
                Ok(vec![Some(
                    out_d.mul(&out_d)?.neg()?.add_scalar(1.0)?.mul(g)?,
                )])
            }),
        ))
    }

    /// Element-wise logistic sigmoid, `1/(1 + exp(-x))`.
    ///
    /// # Errors
    /// [`Error::Unsupported`](crate::Error::Unsupported) on a non-float dtype.
    pub fn sigmoid(&self) -> Result<Tensor> {
        let out = unary_forward(UnaryOp::Sigmoid, self)?;
        let out_d = out.detach();
        Ok(autograd::record(
            "sigmoid",
            out,
            &[self],
            // σ(x)·(1 - σ(x))
            Box::new(move |g| {
                Ok(vec![Some(
                    out_d.neg()?.add_scalar(1.0)?.mul(&out_d)?.mul(g)?,
                )])
            }),
        ))
    }

    /// Element-wise negation. Defined for floats and
    /// [`I64`](crate::DType::I64).
    ///
    /// # Errors
    /// [`Error::Unsupported`](crate::Error::Unsupported) on [`Bool`](crate::DType::Bool).
    pub fn neg(&self) -> Result<Tensor> {
        let out = unary_forward(UnaryOp::Neg, self)?;
        Ok(autograd::record(
            "neg",
            out,
            &[self],
            Box::new(move |g| Ok(vec![Some(g.neg()?)])),
        ))
    }

    /// Element-wise absolute value. Defined for floats and
    /// [`I64`](crate::DType::I64).
    ///
    /// The subgradient at `0` is `0`.
    ///
    /// # Errors
    /// [`Error::Unsupported`](crate::Error::Unsupported) on [`Bool`](crate::DType::Bool).
    pub fn abs(&self) -> Result<Tensor> {
        let out = unary_forward(UnaryOp::Abs, self)?;
        let x = self.detach();
        Ok(autograd::record(
            "abs",
            out,
            &[self],
            Box::new(move |g| {
                // sign(x)·g, with sign(0) = 0.
                let zero = x.zeros_like()?;
                let pos = x.gt(&zero)?.where_cond(g, &zero)?;
                let neg = x.lt(&zero)?.where_cond(g, &zero)?;
                Ok(vec![Some(pos.sub(&neg)?)])
            }),
        ))
    }

    /// `-1`/`0`/`+1` by the sign of each element, NaN preserved.
    ///
    /// The subgradient is `0` everywhere — `sign` is piecewise constant.
    ///
    /// # Errors
    /// [`Error::Unsupported`](crate::Error::Unsupported) on a non-float dtype.
    pub fn sign(&self) -> Result<Tensor> {
        let out = unary_forward(UnaryOp::Sign, self)?;
        Ok(autograd::record(
            "sign",
            out,
            &[self],
            Box::new(move |g| Ok(vec![Some(g.zeros_like()?)])),
        ))
    }

    /// Element-wise reciprocal `1/x`. IEEE: `1/±0` is an infinity, not an
    /// error.
    ///
    /// # Errors
    /// [`Error::Unsupported`](crate::Error::Unsupported) on a non-float dtype.
    pub fn recip(&self) -> Result<Tensor> {
        let out = unary_forward(UnaryOp::Recip, self)?;
        let out_d = out.detach();
        Ok(autograd::record(
            "recip",
            out,
            &[self],
            // d(1/x)/dx = -1/x² = -(1/x)².
            Box::new(move |g| Ok(vec![Some(out_d.mul(&out_d)?.neg()?.mul(g)?)])),
        ))
    }

    /// Round toward `-∞`. The subgradient is `0` everywhere.
    ///
    /// # Errors
    /// [`Error::Unsupported`](crate::Error::Unsupported) on a non-float dtype.
    pub fn floor(&self) -> Result<Tensor> {
        let out = unary_forward(UnaryOp::Floor, self)?;
        Ok(autograd::record(
            "floor",
            out,
            &[self],
            Box::new(move |g| Ok(vec![Some(g.zeros_like()?)])),
        ))
    }

    /// Round toward `+∞`. The subgradient is `0` everywhere.
    ///
    /// # Errors
    /// As [`floor`](Tensor::floor).
    pub fn ceil(&self) -> Result<Tensor> {
        let out = unary_forward(UnaryOp::Ceil, self)?;
        Ok(autograd::record(
            "ceil",
            out,
            &[self],
            Box::new(move |g| Ok(vec![Some(g.zeros_like()?)])),
        ))
    }

    /// Round to the nearest integer, **ties to even** (`PyTorch`'s `round`),
    /// not Rust's away-from-zero [`f64::round`]. The subgradient is `0`
    /// everywhere.
    ///
    /// # Errors
    /// As [`floor`](Tensor::floor).
    pub fn round(&self) -> Result<Tensor> {
        let out = unary_forward(UnaryOp::Round, self)?;
        Ok(autograd::record(
            "round",
            out,
            &[self],
            Box::new(move |g| Ok(vec![Some(g.zeros_like()?)])),
        ))
    }

    /// The Gaussian error function `erf(x)` — the primitive
    /// [`gelu`](Tensor::gelu) is built from, exposed on its own.
    ///
    /// # Errors
    /// As [`floor`](Tensor::floor).
    pub fn erf(&self) -> Result<Tensor> {
        let out = unary_forward(UnaryOp::Erf, self)?;
        let x = self.detach();
        Ok(autograd::record(
            "erf",
            out,
            &[self],
            // d(erf(x))/dx = (2/√π)·exp(-x²).
            Box::new(move |g| {
                const TWO_OVER_SQRT_PI: f64 = std::f64::consts::FRAC_2_SQRT_PI;
                Ok(vec![Some(
                    x.mul(&x)?
                        .neg()?
                        .exp()?
                        .mul_scalar(TWO_OVER_SQRT_PI)?
                        .mul(g)?,
                )])
            }),
        ))
    }

    /// Raise every element to the fixed scalar power `exponent`
    /// (`self^exponent`, `powf` under the hood).
    ///
    /// Float-only: a fractional or negative exponent has no integer meaning,
    /// so this declines rather than invent one.
    ///
    /// # Errors
    /// [`Error::Unsupported`](crate::Error::Unsupported) on a non-float
    /// dtype, or on any non-CPU device: `BinaryOp::Pow` has no accelerator
    /// kernel, so Metal, CUDA and WGPU all decline it.
    pub fn pow(&self, exponent: f64) -> Result<Tensor> {
        const OP: &str = "pow";
        if !self.dtype().is_float() {
            return Err(Error::Unsupported {
                op: OP,
                device: self.device(),
                dtype: self.dtype(),
            });
        }
        let out = binary_scalar_forward(OP, BinaryOp::Pow, self, exponent)?;
        let x = self.detach();
        Ok(autograd::record(
            OP,
            out,
            &[self],
            // d(xᵖ)/dx = p·xᵖ⁻¹.
            Box::new(move |g| {
                Ok(vec![Some(
                    x.pow(exponent - 1.0)?.mul_scalar(exponent)?.mul(g)?,
                )])
            }),
        ))
    }

    /// Clamp every element into `[min, max]`.
    ///
    /// # Errors
    /// [`Error::InvalidArg`](crate::Error::InvalidArg) if either bound is
    /// `NaN` or if `min > max` — the interval would be empty, and the
    /// unordered comparisons would quietly collapse the call to a `max`
    /// or to an all-`NaN` tensor with a zero gradient.
    /// [`Error::Unsupported`](crate::Error::Unsupported) on a dtype without
    /// arithmetic. An infinite bound is fine: `clamp(0.0, f64::INFINITY)` is
    /// the one-sided spelling.
    ///
    /// # Gradient
    /// `1` where `min <= x <= max`, `0` outside — the boundary itself
    /// receives a gradient (a closed interval), matching `PyTorch`.
    ///
    /// ```
    /// # use rstorch::{Device, Tensor};
    /// let x = Tensor::from_vec(vec![-1.0f32, 0.5, 2.0], [3], &Device::Cpu)?;
    /// assert_eq!(x.clamp(0.0, 1.0)?.to_vec::<f32>()?, vec![0.0, 0.5, 1.0]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn clamp(&self, min: f64, max: f64) -> Result<Tensor> {
        const OP: &str = "clamp";
        if min.is_nan() || max.is_nan() || min > max {
            return Err(Error::InvalidArg {
                op: OP,
                msg: format!("clamp requires min <= max and neither NaN, got [{min}, {max}]"),
            });
        }
        let lo = binary_scalar_forward(OP, BinaryOp::Maximum, self, min)?;
        let out = binary_scalar_forward(OP, BinaryOp::Minimum, &lo, max)?;
        let x = self.detach();
        Ok(autograd::record(
            OP,
            out,
            &[self],
            Box::new(move |g| {
                let zero = g.zeros_like()?;
                let ge_min = x.ge(&x.full_like(min)?)?;
                let le_max = x.le(&x.full_like(max)?)?;
                let inside = ge_min.where_cond(&le_max.where_cond(g, &zero)?, &zero)?;
                Ok(vec![Some(inside)])
            }),
        ))
    }

    // ---- composed activations ---------------------------------------------

    /// SiLU / swish: `x · σ(x)`. Composed from
    /// [`sigmoid`](Tensor::sigmoid) and [`mul`](Tensor::mul), so it inherits
    /// their backward exactly rather than a hand-written one.
    ///
    /// # Errors
    /// As [`sigmoid`](Tensor::sigmoid).
    pub fn silu(&self) -> Result<Tensor> {
        self.mul(&self.sigmoid()?)
    }

    /// Leaky ReLU: `x` where `x > 0`, `negative_slope * x` otherwise.
    ///
    /// # Errors
    /// [`Error::InvalidArg`](crate::Error::InvalidArg) if `negative_slope` is
    /// not finite, as [`mul_scalar`](Tensor::mul_scalar) otherwise.
    ///
    /// # Gradient
    /// `1` above zero and `negative_slope` at or below it — the one-sided
    /// convention `PyTorch` uses, and the one that keeps
    /// `leaky_relu(0.0)` agreeing with [`relu`](Tensor::relu) at the origin.
    ///
    /// Selecting with [`where_cond`](Tensor::where_cond) rather than composing
    /// `max(x, 0) + slope · min(x, 0)` is what buys that: `maximum`/`minimum`
    /// split the cotangent evenly on a tie, so the composed form returns
    /// `(1 + slope)/2` at exactly zero — `0.55` for the usual `0.1`, and `0.5`
    /// where `relu` returns `0`.
    pub fn leaky_relu(&self, negative_slope: f64) -> Result<Tensor> {
        const OP: &str = "leaky_relu";
        if !negative_slope.is_finite() {
            return Err(Error::InvalidArg {
                op: OP,
                msg: format!("negative_slope must be finite, got {negative_slope}"),
            });
        }
        let zero = self.zeros_like()?;
        self.gt(&zero)?
            .where_cond(self, &self.mul_scalar(negative_slope)?)
    }

    /// Numerically stable softplus: `ln(1 + exp(x))`, computed as
    /// `max(x, 0) + ln(1 + exp(-|x|))` so a large `x` never overflows `exp`.
    ///
    /// # Errors
    /// As [`exp`](Tensor::exp).
    pub fn softplus(&self) -> Result<Tensor> {
        let zero = self.zeros_like()?;
        let linear_part = self.maximum(&zero)?;
        let stable = self.abs()?.neg()?.exp()?.add_scalar(1.0)?.ln()?;
        linear_part.add(&stable)
    }

    /// Exponential linear unit: `x` where `x > 0`, `alpha * (exp(x) - 1)`
    /// otherwise.
    ///
    /// # Errors
    /// [`Error::InvalidArg`](crate::Error::InvalidArg) if `alpha` is not
    /// finite, as [`exp`](Tensor::exp) otherwise.
    ///
    /// # Gradient
    /// `1` above zero, `alpha · exp(x)` at or below it.
    ///
    /// The exponential is evaluated on `min(x, 0)`, not on `x`. Both branches
    /// of a [`where_cond`](Tensor::where_cond) stay in the graph, and the
    /// discarded one is handed a zero cotangent — so an unbounded `exp(x)`
    /// would saturate to infinity for `x` past the dtype's exponent range
    /// (about `88` in `F32`) and produce `0 · inf = NaN`, poisoning the
    /// accumulated gradient of a perfectly ordinary large activation. Masking
    /// the input first keeps the exponent at or below zero, so it cannot
    /// overflow, and leaves the selected branch's value and derivative
    /// unchanged.
    pub fn elu(&self, alpha: f64) -> Result<Tensor> {
        const OP: &str = "elu";
        if !alpha.is_finite() {
            return Err(Error::InvalidArg {
                op: OP,
                msg: format!("alpha must be finite, got {alpha}"),
            });
        }
        let zero = self.zeros_like()?;
        let positive = self.gt(&zero)?;
        let bounded = positive.where_cond(&zero, self)?;
        let negative_branch = bounded.exp()?.sub_scalar(1.0)?.mul_scalar(alpha)?;
        positive.where_cond(self, &negative_branch)
    }

    // ---- cumulative sum -----------------------------------------------

    /// Inclusive prefix sum along `axis`: element `i` of each line holds the
    /// running total of elements `0..=i`.
    ///
    /// Implemented as one [`matmul`](Tensor::matmul) against a lower
    /// triangular ones matrix built with [`tril`](Tensor::tril)
    /// (`out = x @ Lᵀ`), so the backward (a reverse cumulative sum) comes for
    /// free from `matmul`'s own recorded gradient rather than a hand-written
    /// scan.
    ///
    /// # Errors
    /// [`Error::InvalidAxis`](crate::Error::InvalidAxis) if `axis` is out of
    /// range, [`Error::Unsupported`](crate::Error::Unsupported) on a
    /// non-float dtype.
    ///
    /// ```
    /// # use rstorch::{Device, Tensor};
    /// let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [4], &Device::Cpu)?;
    /// assert_eq!(x.cumsum(0)?.to_vec::<f32>()?, vec![1.0, 3.0, 6.0, 10.0]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn cumsum(&self, axis: isize) -> Result<Tensor> {
        const OP: &str = "cumsum";
        let ax = self.shape().resolve_axis(axis, OP)?;
        if !self.dtype().is_float() {
            return Err(Error::Unsupported {
                op: OP,
                device: self.device(),
                dtype: self.dtype(),
            });
        }
        let n = self.dims()[ax];
        if n == 0 {
            return Ok(self.clone());
        }
        let rank = self.rank();
        let last = rank - 1;
        let moved = if ax == last {
            self.clone()
        } else {
            self.transpose(ax as isize, last as isize)?
        };
        let outer = moved.num_elements() / n;
        let flat = moved.reshape([outer, n])?;
        let lower = Tensor::ones([n, n], self.dtype(), &self.device())?.tril(0)?;
        let scanned = flat.matmul(&lower.transpose(0, 1)?)?;
        let restored = scanned.reshape(moved.dims().to_vec())?;
        if ax == last {
            Ok(restored)
        } else {
            restored.transpose(ax as isize, last as isize)
        }
    }

    // ---- comparisons -----------------------------------------------------

    /// Element-wise `self == rhs`, broadcasting to the common shape and
    /// producing a [`Bool`](crate::DType::Bool) tensor.
    ///
    /// Comparisons are not differentiable: the result carries no autograd
    /// graph.
    ///
    /// # Errors
    /// [`Error::DeviceMismatch`](crate::Error::DeviceMismatch),
    /// [`Error::DTypeMismatch`](crate::Error::DTypeMismatch) or
    /// [`Error::ShapeMismatch`](crate::Error::ShapeMismatch) as for [`add`](Tensor::add).
    pub fn eq(&self, rhs: &Tensor) -> Result<Tensor> {
        compare_forward("eq", CmpOp::Eq, self, rhs)
    }

    /// Element-wise `self != rhs` (see [`eq`](Tensor::eq)).
    ///
    /// # Errors
    /// As [`eq`](Tensor::eq).
    pub fn ne(&self, rhs: &Tensor) -> Result<Tensor> {
        compare_forward("ne", CmpOp::Ne, self, rhs)
    }

    /// Element-wise `self < rhs` (see [`eq`](Tensor::eq)).
    ///
    /// # Errors
    /// As [`eq`](Tensor::eq).
    pub fn lt(&self, rhs: &Tensor) -> Result<Tensor> {
        compare_forward("lt", CmpOp::Lt, self, rhs)
    }

    /// Element-wise `self <= rhs` (see [`eq`](Tensor::eq)).
    ///
    /// # Errors
    /// As [`eq`](Tensor::eq).
    pub fn le(&self, rhs: &Tensor) -> Result<Tensor> {
        compare_forward("le", CmpOp::Le, self, rhs)
    }

    /// Element-wise `self > rhs` (see [`eq`](Tensor::eq)).
    ///
    /// # Errors
    /// As [`eq`](Tensor::eq).
    pub fn gt(&self, rhs: &Tensor) -> Result<Tensor> {
        compare_forward("gt", CmpOp::Gt, self, rhs)
    }

    /// Element-wise `self >= rhs` (see [`eq`](Tensor::eq)).
    ///
    /// # Errors
    /// As [`eq`](Tensor::eq).
    pub fn ge(&self, rhs: &Tensor) -> Result<Tensor> {
        compare_forward("ge", CmpOp::Ge, self, rhs)
    }

    // ---- masking ---------------------------------------------------------

    /// Replace the elements where `mask` is true with `value` (narrowed to
    /// this tensor's dtype).
    ///
    /// `mask` is a [`Bool`](crate::DType::Bool) tensor broadcast against
    /// `self`; the result has their common shape, which is `self`'s shape
    /// whenever `mask` broadcasts *into* it (the usual attention-mask case).
    /// The mask is a constant as far as autograd is concerned — the cotangent
    /// simply does not flow through the filled positions.
    ///
    /// # Errors
    /// [`Error::DTypeMismatch`](crate::Error::DTypeMismatch) if `mask` is not
    /// [`Bool`](crate::DType::Bool),
    /// [`Error::DeviceMismatch`](crate::Error::DeviceMismatch) if it lives elsewhere, and
    /// [`Error::ShapeMismatch`](crate::Error::ShapeMismatch) if the shapes do not broadcast.
    pub fn masked_fill(&self, mask: &Tensor, value: f64) -> Result<Tensor> {
        const OP: &str = "masked_fill";
        require_dtype(OP, mask, DType::Bool)?;
        same_device(OP, self, mask)?;
        let out_shape = self.shape().broadcast_with(mask.shape(), OP)?;
        let xl = self.layout().broadcast_to(&out_shape)?;
        let ml = mask.layout().broadcast_to(&out_shape)?;
        let storage = dispatch::backend(self.device()).masked_fill(
            View::new(self.storage(), &xl),
            View::new(mask.storage(), &ml),
            value,
        )?;
        let out = Tensor::from_parts(storage, Layout::contiguous(out_shape)?);
        let dims = self.dims().to_vec();
        let m = mask.detach();
        Ok(autograd::record(
            OP,
            out,
            &[self],
            Box::new(move |g| Ok(vec![Some(g.masked_fill(&m, 0.0)?.sum_to(&dims)?)])),
        ))
    }

    /// Select from `on_true` where `self` (a [`Bool`](crate::DType::Bool)
    /// condition) is true and from `on_false` elsewhere — `torch.where` with
    /// the condition as the receiver.
    ///
    /// All three operands are broadcast to their common shape. The condition
    /// is a constant for autograd; the cotangent is routed to whichever value
    /// operand supplied each element.
    ///
    /// # Errors
    /// [`Error::DTypeMismatch`](crate::Error::DTypeMismatch) if `self` is not
    /// [`Bool`](crate::DType::Bool) or the two value operands disagree,
    /// [`Error::DeviceMismatch`](crate::Error::DeviceMismatch) across devices, and
    /// [`Error::ShapeMismatch`](crate::Error::ShapeMismatch) if the three shapes do not
    /// broadcast.
    pub fn where_cond(&self, on_true: &Tensor, on_false: &Tensor) -> Result<Tensor> {
        const OP: &str = "where";
        require_dtype(OP, self, DType::Bool)?;
        same_device(OP, self, on_true)?;
        same_device(OP, self, on_false)?;
        same_dtype(OP, on_true, on_false)?;
        let out_shape = self
            .shape()
            .broadcast_with(on_true.shape(), OP)?
            .broadcast_with(on_false.shape(), OP)?;
        let cl = self.layout().broadcast_to(&out_shape)?;
        let tl = on_true.layout().broadcast_to(&out_shape)?;
        let fl = on_false.layout().broadcast_to(&out_shape)?;
        let storage = dispatch::backend(self.device()).where_cond(
            View::new(self.storage(), &cl),
            View::new(on_true.storage(), &tl),
            View::new(on_false.storage(), &fl),
        )?;
        let out = Tensor::from_parts(storage, Layout::contiguous(out_shape)?);
        let cond = self.detach();
        let (td, fd) = (on_true.dims().to_vec(), on_false.dims().to_vec());
        Ok(autograd::record(
            OP,
            out,
            &[on_true, on_false],
            Box::new(move |g| {
                let split = |take_true: bool, dims: &[usize]| -> Result<Tensor> {
                    let zero = g.zeros_like()?;
                    let picked = if take_true {
                        cond.where_cond(g, &zero)?
                    } else {
                        cond.where_cond(&zero, g)?
                    };
                    picked.sum_to(dims)
                };
                Ok(vec![Some(split(true, &td)?), Some(split(false, &fd)?)])
            }),
        ))
    }
}

#[cfg(test)]
mod tests;
