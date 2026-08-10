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
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::error::Error;
    use crate::shape::Shape;
    use crate::testing::check_grad;

    const CPU: Device = Device::Cpu;

    fn t(data: &[f32], shape: impl Into<Shape>) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape, &CPU).unwrap()
    }

    fn v(t: &Tensor) -> Vec<f32> {
        t.to_vec::<f32>().unwrap()
    }

    /// Re-view `x` through `layout` over the same storage — the only way to
    /// build a non-contiguous tensor before T21's public view ops land.
    fn re_view(x: &Tensor, layout: Layout) -> Tensor {
        Tensor::from_parts(x.storage().clone(), layout)
    }

    fn close(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "length: {a:?} vs {b:?}");
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= tol, "{a:?} vs {b:?}");
        }
    }

    // ------------------------------------------------------------------
    // Arithmetic
    // ------------------------------------------------------------------

    #[test]
    fn same_shape_arithmetic() {
        let a = t(&[1.0, 2.0, 3.0, 4.0], [2, 2]);
        let b = t(&[10.0, 20.0, 30.0, 40.0], [2, 2]);
        assert_eq!(v(&a.add(&b).unwrap()), vec![11.0, 22.0, 33.0, 44.0]);
        assert_eq!(v(&a.sub(&b).unwrap()), vec![-9.0, -18.0, -27.0, -36.0]);
        assert_eq!(v(&a.mul(&b).unwrap()), vec![10.0, 40.0, 90.0, 160.0]);
        close(&v(&b.div(&a).unwrap()), &[10.0, 10.0, 10.0, 10.0], 1e-6);
        // Shape, dtype and device survive.
        let s = a.add(&b).unwrap();
        assert_eq!(s.dims(), &[2, 2]);
        assert_eq!(s.dtype(), DType::F32);
        assert_eq!(s.device(), CPU);
        assert!(s.is_contiguous());
    }

    #[test]
    fn broadcasting_follows_numpy_rules() {
        let m = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);

        // Row vector against a matrix.
        let row = t(&[10.0, 20.0, 30.0], [3]);
        let out = m.add(&row).unwrap();
        assert_eq!(out.dims(), &[2, 3]);
        assert_eq!(v(&out), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);

        // Column vector against a matrix.
        let col = t(&[100.0, 200.0], [2, 1]);
        assert_eq!(
            v(&m.add(&col).unwrap()),
            vec![101.0, 102.0, 103.0, 204.0, 205.0, 206.0]
        );

        // Outer-product style: [2, 1] × [1, 3] -> [2, 3].
        let a = t(&[1.0, 2.0], [2, 1]);
        let b = t(&[10.0, 20.0, 30.0], [1, 3]);
        let out = a.mul(&b).unwrap();
        assert_eq!(out.dims(), &[2, 3]);
        assert_eq!(v(&out), vec![10.0, 20.0, 30.0, 20.0, 40.0, 60.0]);

        // A rank-0 operand broadcasts against anything.
        let s = t(&[2.0], ());
        assert_eq!(v(&m.mul(&s).unwrap()), vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn scalar_variants() {
        let a = t(&[1.0, 2.0, 4.0], [3]);
        assert_eq!(v(&a.add_scalar(1.0).unwrap()), vec![2.0, 3.0, 5.0]);
        assert_eq!(v(&a.sub_scalar(1.0).unwrap()), vec![0.0, 1.0, 3.0]);
        assert_eq!(v(&a.mul_scalar(2.5).unwrap()), vec![2.5, 5.0, 10.0]);
        assert_eq!(v(&a.div_scalar(2.0).unwrap()), vec![0.5, 1.0, 2.0]);
        // The scalar is narrowed to the tensor's dtype.
        let i = Tensor::from_vec(vec![1i64, 2, 3], [3], &CPU).unwrap();
        let out = i.add_scalar(2.9).unwrap();
        assert_eq!(out.dtype(), DType::I64);
        assert_eq!(out.to_vec::<i64>().unwrap(), vec![3, 4, 5]);
    }

    #[test]
    fn scalar_variants_walk_strided_inputs_and_name_themselves_in_errors() {
        let base = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let transposed = re_view(&base, base.layout().transpose(0, 1).unwrap());
        let out = transposed.mul_scalar(10.0).unwrap();
        assert_eq!(out.dims(), &[3, 2]);
        assert!(out.is_contiguous());
        assert_eq!(v(&out), vec![10.0, 40.0, 20.0, 50.0, 30.0, 60.0]);

        // The kernel reports the op *family* (`"add"`); the op layer relabels
        // it with the public method the caller actually used.
        let b = Tensor::from_vec(vec![true, false], [2], &CPU).unwrap();
        assert!(matches!(
            b.add_scalar(1.0),
            Err(Error::Unsupported {
                op: "add_scalar",
                ..
            })
        ));
        assert!(matches!(
            b.div_scalar(1.0),
            Err(Error::Unsupported {
                op: "div_scalar",
                ..
            })
        ));
    }

    #[test]
    fn maximum_and_minimum() {
        let a = t(&[1.0, 5.0, -2.0], [3]);
        let b = t(&[3.0, 3.0, -3.0], [3]);
        assert_eq!(v(&a.maximum(&b).unwrap()), vec![3.0, 5.0, -2.0]);
        assert_eq!(v(&a.minimum(&b).unwrap()), vec![1.0, 3.0, -3.0]);
        // Broadcasting works here too.
        let s = t(&[0.0], ());
        assert_eq!(v(&a.maximum(&s).unwrap()), vec![1.0, 5.0, 0.0]);
    }

    /// `extremum_backward` is a pure function over already-detached operands,
    /// so unlike the closures it lives inside it can be driven before T30's
    /// engine exists. Ties and broadcast reduction are the parts worth
    /// pinning down early.
    #[test]
    fn extremum_backward_splits_ties_and_reduces_broadcasts() {
        let a = t(&[1.0, 3.0, 5.0], [3]);
        let b = t(&[2.0, 3.0, 4.0], [3]);
        let g = t(&[10.0, 10.0, 10.0], [3]);

        // b, tie, a wins: the tied element hands each side half the cotangent.
        let grads = extremum_backward(&g, &a, &b, a.dims(), b.dims(), true).unwrap();
        assert_eq!(v(grads[0].as_ref().unwrap()), vec![0.0, 5.0, 10.0]);
        assert_eq!(v(grads[1].as_ref().unwrap()), vec![10.0, 5.0, 0.0]);

        // `minimum` flips which side wins, tie handling unchanged.
        let grads = extremum_backward(&g, &a, &b, a.dims(), b.dims(), false).unwrap();
        assert_eq!(v(grads[0].as_ref().unwrap()), vec![10.0, 5.0, 0.0]);
        assert_eq!(v(grads[1].as_ref().unwrap()), vec![0.0, 5.0, 10.0]);

        // A broadcast operand's cotangent comes back at that operand's own
        // shape: max([[1,1],[4,4]], [[2,3],[2,3]]) takes the row in the top
        // half and the column in the bottom half.
        let col = t(&[1.0, 4.0], [2, 1]);
        let row = t(&[2.0, 3.0], [2]);
        let g = t(&[1.0, 1.0, 1.0, 1.0], [2, 2]);
        let grads = extremum_backward(&g, &col, &row, col.dims(), row.dims(), true).unwrap();
        assert_eq!(grads[0].as_ref().unwrap().dims(), &[2, 1]);
        assert_eq!(v(grads[0].as_ref().unwrap()), vec![0.0, 2.0]);
        assert_eq!(grads[1].as_ref().unwrap().dims(), &[2]);
        assert_eq!(v(grads[1].as_ref().unwrap()), vec![1.0, 1.0]);
    }

    #[test]
    fn integer_arithmetic_and_bool_rejection() {
        let a = Tensor::from_vec(vec![7i64, -7, 6], [3], &CPU).unwrap();
        let b = Tensor::from_vec(vec![2i64, 2, 0], [3], &CPU).unwrap();
        assert_eq!(a.add(&b).unwrap().to_vec::<i64>().unwrap(), vec![9, -5, 6]);
        // Integer division truncates; a zero divisor yields 0, never a panic.
        assert_eq!(a.div(&b).unwrap().to_vec::<i64>().unwrap(), vec![3, -3, 0]);

        let x = Tensor::from_vec(vec![true, false], [2], &CPU).unwrap();
        assert!(matches!(
            x.add(&x),
            Err(Error::Unsupported { op: "add", .. })
        ));
    }

    #[test]
    fn strided_operands_are_broadcast_through_the_layout() {
        let base = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let transposed = re_view(&base, base.layout().transpose(0, 1).unwrap()); // [3, 2]
        assert!(!transposed.is_contiguous());
        let other = t(&[10.0, 100.0], [2]);
        let out = transposed.add(&other).unwrap();
        assert_eq!(out.dims(), &[3, 2]);
        assert!(out.is_contiguous());
        // Transposed rows are [1,4], [2,5], [3,6].
        assert_eq!(v(&out), vec![11.0, 104.0, 12.0, 105.0, 13.0, 106.0]);
    }

    #[test]
    fn mismatched_operands_are_structured_errors() {
        let f = t(&[1.0, 2.0], [2]);
        let i = Tensor::from_vec(vec![1i64, 2], [2], &CPU).unwrap();
        assert!(matches!(
            f.add(&i),
            Err(Error::DTypeMismatch {
                op: "add",
                expected: DType::F32,
                got: DType::I64
            })
        ));
        let wide = t(&[1.0, 2.0, 3.0], [3]);
        match f.mul(&wide) {
            Err(Error::ShapeMismatch { op, lhs, rhs }) => {
                assert_eq!(op, "mul");
                assert_eq!(lhs, Shape::from([2]));
                assert_eq!(rhs, Shape::from([3]));
            }
            Err(e) => panic!("expected a ShapeMismatch, got {e}"),
            Ok(_) => panic!("expected a ShapeMismatch, got a tensor"),
        }
    }

    // ------------------------------------------------------------------
    // Unary math
    // ------------------------------------------------------------------

    #[test]
    fn unary_values() {
        let x = t(&[-2.0, -0.5, 0.0, 0.5, 2.0], [5]);
        assert_eq!(v(&x.relu().unwrap()), vec![0.0, 0.0, 0.0, 0.5, 2.0]);
        assert_eq!(v(&x.neg().unwrap()), vec![2.0, 0.5, -0.0, -0.5, -2.0]);
        assert_eq!(v(&x.abs().unwrap()), vec![2.0, 0.5, 0.0, 0.5, 2.0]);
        close(
            &v(&x.exp().unwrap()),
            &[0.135_335_28, 0.606_530_66, 1.0, 1.648_721_3, 7.389_056],
            1e-6,
        );
        close(
            &v(&x.tanh().unwrap()),
            &[-0.964_027_6, -0.462_117_16, 0.0, 0.462_117_16, 0.964_027_6],
            1e-6,
        );
        close(
            &v(&x.sigmoid().unwrap()),
            &[0.119_202_92, 0.377_540_67, 0.5, 0.622_459_33, 0.880_797_1],
            1e-6,
        );

        let p = t(&[1.0, 4.0, 9.0], [3]);
        close(&v(&p.sqrt().unwrap()), &[1.0, 2.0, 3.0], 1e-6);
        close(&v(&p.ln().unwrap()), &[0.0, 1.386_294_4, 2.197_224_6], 1e-6);
    }

    #[test]
    fn gelu_is_exact_not_the_tanh_approximation() {
        let x = t(&[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], [6]);
        // 0.5·x·(1 + erf(x/√2)) evaluated in f64.
        let expected = [
            -0.045_500_264,
            -0.158_655_25,
            0.0,
            0.841_344_8,
            1.954_499_7,
            2.995_950_2,
        ];
        close(&v(&x.gelu().unwrap()), &expected, 1e-6);

        // The tanh approximation `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`
        // deviates from the exact form by ~4e-4 around x = 3 — two orders of
        // magnitude outside the tolerance asserted above — so this test fails
        // loudly if a backend ever swaps the approximation in.
        let tanh_approx =
            |x: f32| 0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh());
        let worst = v(&x)
            .iter()
            .zip(&expected)
            .map(|(&xi, &e)| (tanh_approx(xi) - e).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst > 1e-4,
            "tanh-approximation deviation was only {worst}"
        );
    }

    #[test]
    fn unary_dtype_rules() {
        let i = Tensor::from_vec(vec![-3i64, 0, 4], [3], &CPU).unwrap();
        // Sign-preserving integer unaries are defined...
        assert_eq!(i.neg().unwrap().to_vec::<i64>().unwrap(), vec![3, 0, -4]);
        assert_eq!(i.abs().unwrap().to_vec::<i64>().unwrap(), vec![3, 0, 4]);
        // ...the float-only ones are loud.
        assert!(matches!(i.exp(), Err(Error::Unsupported { op: "exp", .. })));
        assert!(matches!(
            i.relu(),
            Err(Error::Unsupported { op: "relu", .. })
        ));
        let b = Tensor::from_vec(vec![true], [1], &CPU).unwrap();
        assert!(matches!(b.neg(), Err(Error::Unsupported { op: "neg", .. })));
    }

    #[test]
    fn unary_materializes_strided_inputs() {
        let base = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let transposed = re_view(&base, base.layout().transpose(0, 1).unwrap());
        let out = transposed.neg().unwrap();
        assert_eq!(out.dims(), &[3, 2]);
        assert!(out.is_contiguous());
        assert_eq!(v(&out), vec![-1.0, -4.0, -2.0, -5.0, -3.0, -6.0]);
    }

    // ------------------------------------------------------------------
    // Comparisons
    // ------------------------------------------------------------------

    #[test]
    fn comparisons_produce_bool() {
        let a = t(&[1.0, 2.0, 3.0], [3]);
        let b = t(&[3.0, 2.0, 1.0], [3]);
        let out = a.lt(&b).unwrap();
        assert_eq!(out.dtype(), DType::Bool);
        assert_eq!(out.to_vec::<bool>().unwrap(), vec![true, false, false]);
        assert_eq!(
            a.le(&b).unwrap().to_vec::<bool>().unwrap(),
            vec![true, true, false]
        );
        assert_eq!(
            a.gt(&b).unwrap().to_vec::<bool>().unwrap(),
            vec![false, false, true]
        );
        assert_eq!(
            a.ge(&b).unwrap().to_vec::<bool>().unwrap(),
            vec![false, true, true]
        );
        assert_eq!(
            a.eq(&b).unwrap().to_vec::<bool>().unwrap(),
            vec![false, true, false]
        );
        assert_eq!(
            a.ne(&b).unwrap().to_vec::<bool>().unwrap(),
            vec![true, false, true]
        );
    }

    #[test]
    fn comparisons_broadcast_and_stay_untraced() {
        let m = t(&[1.0, 2.0, 3.0, 4.0], [2, 2]);
        let s = t(&[2.5], ());
        let out = m.gt(&s).unwrap();
        assert_eq!(out.dims(), &[2, 2]);
        assert_eq!(
            out.to_vec::<bool>().unwrap(),
            vec![false, false, true, true]
        );
        assert!(out.node().is_none());
        // Bool operands compare fine against each other.
        let x = Tensor::from_vec(vec![true, false], [2], &CPU).unwrap();
        let y = Tensor::from_vec(vec![true, true], [2], &CPU).unwrap();
        assert_eq!(
            x.eq(&y).unwrap().to_vec::<bool>().unwrap(),
            vec![true, false]
        );
        // Mixed dtypes are still an error, not a promotion.
        let i = Tensor::from_vec(vec![1i64, 2], [2], &CPU).unwrap();
        assert!(matches!(
            i.eq(&t(&[1.0, 2.0], [2])),
            Err(Error::DTypeMismatch { op: "eq", .. })
        ));
    }

    // ------------------------------------------------------------------
    // masked_fill / where_cond
    // ------------------------------------------------------------------

    #[test]
    fn masked_fill_replaces_true_positions() {
        let x = t(&[1.0, 2.0, 3.0, 4.0], [2, 2]);
        let mask = Tensor::from_vec(vec![false, true, true, false], [2, 2], &CPU).unwrap();
        let out = x.masked_fill(&mask, -1.0).unwrap();
        assert_eq!(out.dims(), &[2, 2]);
        assert_eq!(v(&out), vec![1.0, -1.0, -1.0, 4.0]);

        // The mask broadcasts (the causal-attention shape).
        let row_mask = Tensor::from_vec(vec![true, false], [2], &CPU).unwrap();
        assert_eq!(
            v(&x.masked_fill(&row_mask, 0.0).unwrap()),
            vec![0.0, 2.0, 0.0, 4.0]
        );

        // A non-bool mask is a dtype error.
        assert!(matches!(
            x.masked_fill(&x, 0.0),
            Err(Error::DTypeMismatch {
                op: "masked_fill",
                expected: DType::Bool,
                ..
            })
        ));
    }

    #[test]
    fn where_cond_selects_per_element() {
        let cond = Tensor::from_vec(vec![true, false, true], [3], &CPU).unwrap();
        let a = t(&[1.0, 2.0, 3.0], [3]);
        let b = t(&[10.0, 20.0, 30.0], [3]);
        assert_eq!(v(&cond.where_cond(&a, &b).unwrap()), vec![1.0, 20.0, 3.0]);

        // All three operands broadcast: [2, 1] cond, [3] values.
        let cond = Tensor::from_vec(vec![true, false], [2, 1], &CPU).unwrap();
        let a = t(&[1.0, 2.0, 3.0], [3]);
        let b = t(&[-1.0], ());
        let out = cond.where_cond(&a, &b).unwrap();
        assert_eq!(out.dims(), &[2, 3]);
        assert_eq!(v(&out), vec![1.0, 2.0, 3.0, -1.0, -1.0, -1.0]);

        // A non-bool condition, and value operands of differing dtypes, are
        // both structured errors.
        assert!(matches!(
            a.where_cond(&a, &b),
            Err(Error::DTypeMismatch { op: "where", .. })
        ));
        let i = Tensor::from_vec(vec![1i64, 2, 3], [3], &CPU).unwrap();
        assert!(matches!(
            cond.where_cond(&a, &i),
            Err(Error::DTypeMismatch { op: "where", .. })
        ));
    }

    // ------------------------------------------------------------------
    // Backward — finite differences against the single `check_grad` harness.
    //
    // `check_grad` requires `f` to produce a **scalar**. T22 was written
    // before T23's reductions existed, so every case below used to be a
    // single-element tensor; **T31** widened them to multi-element shapes and
    // scalarizes with the weighted sum `wsum`.
    //
    // The weighting is load-bearing. A one-element input cannot distinguish a
    // correct backward from one that mixes elements up, and an *unweighted*
    // `sum_all` hands every element the same cotangent — which cannot tell a
    // gradient placed in the right slot from one transposed, reversed, or
    // broadcast-summed into the wrong slot. `wsum`'s weights are pairwise
    // distinct, so any misplacement shows up as a wrong number.
    // ------------------------------------------------------------------

    const EPS: f64 = 1e-3;
    const TOL: f64 = 1e-3;

    /// `Σ w ⊙ x` with pairwise-distinct constant weights: a scalar objective
    /// whose gradient w.r.t. `x` is `w` rather than a constant.
    fn wsum(x: &Tensor) -> Result<Tensor> {
        let w: Vec<f32> = (0..x.num_elements())
            .map(|i| 0.25 + 0.5 * (i as f32))
            .collect();
        x.mul(&Tensor::from_vec(w, x.dims().to_vec(), &CPU)?)?
            .sum_all()
    }

    /// The left operand of the binary cases: no element equals its `rhs`
    /// partner, so `maximum`/`minimum` are locally smooth (a tie is a kink
    /// finite differences cannot see through).
    fn lhs() -> Tensor {
        t(&[1.5, -0.75, 2.25, 0.5, -1.25, 3.0], [2, 3])
    }

    fn rhs() -> Tensor {
        t(&[-0.5, 2.0, 1.25, -2.5, 0.75, -1.5], [2, 3])
    }

    #[test]
    fn grad_binary_arithmetic() {
        type BinaryCase = fn(&[Tensor]) -> Result<Tensor>;
        let cases: [BinaryCase; 6] = [
            |i| wsum(&i[0].add(&i[1])?),
            |i| wsum(&i[0].sub(&i[1])?),
            |i| wsum(&i[0].mul(&i[1])?),
            |i| wsum(&i[0].div(&i[1])?),
            |i| wsum(&i[0].maximum(&i[1])?),
            |i| wsum(&i[0].minimum(&i[1])?),
        ];
        for f in cases {
            check_grad(f, &[lhs(), rhs()], EPS, TOL).unwrap();
        }
    }

    #[test]
    fn grad_binary_broadcast_reduces_through_sum_to() {
        // A rank-0 operand against a rank-1 one: the lhs cotangent has to be
        // summed back down over the padded leading axis.
        check_grad(
            |i: &[Tensor]| i[0].mul(&i[1]),
            &[t(&[2.0], ()), t(&[-3.0], [1])],
            EPS,
            TOL,
        )
        .unwrap();

        // A [3] operand against a [2, 3] one: the rhs cotangent is summed
        // over the *padded* axis only, keeping its per-column placement.
        check_grad(
            |i: &[Tensor]| wsum(&i[0].mul(&i[1])?),
            &[lhs(), t(&[-0.5, 2.0, 1.25], [3])],
            EPS,
            TOL,
        )
        .unwrap();

        // …and a [2, 1] operand, summed over the *existing* size-1 axis, so
        // the two reduction paths in `sum_to` are both exercised.
        check_grad(
            |i: &[Tensor]| wsum(&i[0].div(&i[1])?),
            &[lhs(), t(&[-2.5, 0.75], [2, 1])],
            EPS,
            TOL,
        )
        .unwrap();
    }

    #[test]
    fn grad_scalar_variants() {
        type ScalarCase = fn(&[Tensor]) -> Result<Tensor>;
        let cases: [ScalarCase; 4] = [
            |i| wsum(&i[0].add_scalar(2.0)?),
            |i| wsum(&i[0].sub_scalar(2.0)?),
            |i| wsum(&i[0].mul_scalar(-3.0)?),
            |i| wsum(&i[0].div_scalar(4.0)?),
        ];
        for f in cases {
            check_grad(f, &[lhs()], EPS, TOL).unwrap();
        }
    }

    #[test]
    fn grad_unary_family() {
        // Several points per op, both signs where the domain allows, all far
        // enough from a kink (`relu`/`abs` at 0) that `±EPS` stays on one side.
        type UnaryCase = (fn(&[Tensor]) -> Result<Tensor>, &'static [f32]);
        let cases: [UnaryCase; 9] = [
            (|i| wsum(&i[0].relu()?), &[0.7, -1.3, 2.5]),
            (|i| wsum(&i[0].gelu()?), &[0.7, -1.3, 2.5, -0.2]),
            (|i| wsum(&i[0].exp()?), &[0.3, -1.1, 1.4]),
            (|i| wsum(&i[0].ln()?), &[1.7, 0.4, 3.2]),
            (|i| wsum(&i[0].sqrt()?), &[2.3, 0.6, 4.1]),
            (|i| wsum(&i[0].tanh()?), &[0.4, -1.5, 2.2]),
            (|i| wsum(&i[0].sigmoid()?), &[0.4, -1.5, 2.2]),
            (|i| wsum(&i[0].neg()?), &[0.9, -2.0, 0.1]),
            (|i| wsum(&i[0].abs()?), &[-1.2, 0.8, 2.6]),
        ];
        for (f, at) in cases {
            check_grad(f, &[t(at, [at.len()])], EPS, TOL).unwrap();
        }
        // GELU's backward has a special case at exactly zero (Φ(0) = 1/2).
        check_grad(|i| i[0].gelu(), &[t(&[0.0], ())], EPS, TOL).unwrap();
    }

    #[test]
    fn grad_masking_ops() {
        // A mixed mask, so one call covers both the kept and the dropped
        // branch and the gradient has to land on the right elements.
        let mask =
            Tensor::from_vec(vec![false, true, true, false, true, false], [2, 3], &CPU).unwrap();

        let m = mask.clone();
        check_grad(
            move |i: &[Tensor]| wsum(&i[0].masked_fill(&m, 0.0)?),
            &[lhs()],
            EPS,
            TOL,
        )
        .unwrap();

        let c = mask;
        check_grad(
            move |i: &[Tensor]| wsum(&c.where_cond(&i[0], &i[1])?),
            &[lhs(), rhs()],
            EPS,
            TOL,
        )
        .unwrap();
    }
}
