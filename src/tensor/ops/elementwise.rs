//! Element-wise ops: arithmetic (`add`/`sub`/`mul`/`div` and their scalar
//! spellings), `maximum`/`minimum`, the unary math family, comparisons to
//! [`Bool`](crate::DType::Bool), `masked_fill` and `where_cond`.
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
//! - Output-dependent formulas (`exp`, `sqrt`, `tanh`, `sigmoid`, `div`,
//!   `gelu`) capture the op's output in **detached** form, built before the
//!   traced output is assembled (the detached-output capture rule) —
//!   capturing the traced output would create an `Arc`
//!   cycle through the closure.
//! - Comparisons produce [`Bool`](crate::DType::Bool) and are therefore not
//!   differentiable: they do not go through the record seam at all. For the
//!   same reason the [`Bool`](crate::DType::Bool) operand of `masked_fill`
//!   and `where_cond` is not listed as a graph input.

use super::{require_dtype, same_device, same_dtype};
use crate::autograd;
use crate::backend::{BinaryOp, CmpOp, UnaryOp, View, dispatch};
use crate::dtype::DType;
use crate::error::Result;
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

/// A zero tensor shaped, typed and placed like `like` — the "no gradient
/// here" branch of the mask-driven backwards.
fn zeros_like(like: &Tensor) -> Result<Tensor> {
    Tensor::zeros(like.dims(), like.dtype(), &like.device())
}

/// One side of the `maximum`/`minimum` backward: the cotangent where this
/// operand strictly wins, plus half of it where the two tie (PyTorch's
/// tie-splitting rule), reduced back to `dims`.
fn extremum_side(g: &Tensor, win: &Tensor, tie: &Tensor, dims: &[usize]) -> Result<Tensor> {
    let zero = zeros_like(g)?;
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
    /// The backward splits the cotangent evenly on ties (PyTorch's
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
    /// The backward splits the cotangent evenly on ties (PyTorch's
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
                let zero = zeros_like(&x)?;
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
                let zero = zeros_like(&x)?;
                let half = Tensor::full(x.dims(), 0.5, x.dtype(), &x.device())?;
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
                let zero = zeros_like(&x)?;
                let pos = x.gt(&zero)?.where_cond(g, &zero)?;
                let neg = x.lt(&zero)?.where_cond(g, &zero)?;
                Ok(vec![Some(pos.sub(&neg)?)])
            }),
        ))
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
                    let zero = zeros_like(g)?;
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
