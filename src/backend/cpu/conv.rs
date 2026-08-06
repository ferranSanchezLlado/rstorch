//! Convolution/pooling CPU kernels.
//!
//! Signature frozen by T01; **T26** fills the body. Per-variant input
//! contracts on [`ConvOp`](crate::backend::ConvOp); geometry in
//! [`Conv2dParams`](crate::backend::Conv2dParams); accumulation in `Acc`.
//!
//! # Layout and geometry
//!
//! Images are **NCHW** (`[batch, channels, height, width]`) and convolution
//! weights are **OIHW** (`[out_channels, in_channels, kernel_h, kernel_w]`),
//! matching PyTorch. Every operand is read stride-aware through its
//! [`Layout`], so transposed / narrowed / broadcast views work without a
//! pre-materialization; every result is a freshly allocated **contiguous**
//! buffer in the op layer's output shape.
//!
//! [`Conv2dGeometry`] is the one place the output-size formula
//! `out = (input + 2·padding − ((kernel − 1)·dilation + 1)) / stride + 1`
//! lives. The op layer resolves it to build the output [`Layout`] and hands
//! the same value back to the gradient kernels, so the two can never
//! disagree.
//!
//! # Semantics (PyTorch-familiar, exploration §3.1)
//!
//! - `Conv2d` is a **cross-correlation** (no kernel flip), like PyTorch's.
//!   Bias is not a kernel operand: the op layer adds it as a broadcast add.
//! - `MaxPool2d` picks the window maximum with a NaN-propagating,
//!   first-position-wins ordering, so the forward value and the position its
//!   gradient is routed to always agree.
//! - `AvgPool2d` divides by the **full window area** (`kernel_h · kernel_w`),
//!   i.e. PyTorch's `count_include_pad = true` default; positions that fall
//!   in the zero padding contribute `0` to the sum but still count in the
//!   divisor. (v2 divided by the valid-position count; v3 follows PyTorch.)
//! - Pooling ignores `dilation`, per
//!   [`Conv2dParams`](crate::backend::Conv2dParams).
//!
//! # Gradient kernels
//!
//! The four backward forms are routed through [`BackendOps::conv`](crate::backend::BackendOps::conv)
//! alongside their forward counterparts. Their saved forward operands make
//! the requested output shape and geometry explicit without exposing this
//! module's [`Conv2dGeometry`] through the backend contract.

use crate::backend::conv_geometry::Conv2dGeometry;
use crate::backend::{Conv2dParams, ConvOp, View};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::shape::Shape;
use crate::storage::{CpuStorage, Storage};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Wide accumulation
// ---------------------------------------------------------------------------

/// Wide-accumulator arithmetic for the conv/pool kernels.
///
/// Implemented for exactly the accumulator types the
/// [`Acc`](crate::dtype::Element::Acc) contract yields for a numeric element
/// (`f32` for `f16`/`bf16`/`f32`, `f64`, `i64`). `Bool` has `Acc = bool`,
/// which deliberately does not implement this trait, so a bool convolution is
/// rejected before any generic code is instantiated.
trait ConvAcc: Copy {
    /// Additive identity (window-accumulation seed).
    const ZERO: Self;
    /// Widening sum step.
    fn add(self, other: Self) -> Self;
    /// Widening product (integers wrap deterministically rather than aborting
    /// on a debug overflow, matching the matmul kernel).
    fn mul(self, other: Self) -> Self;
    /// Divide an accumulated window sum by the window area (`AvgPool2d`).
    fn div_count(self, count: usize) -> Self;
    /// Whether `self` strictly beats `other` as a running maximum.
    ///
    /// NaN beats every number but not another NaN, so a window containing a
    /// NaN pools to NaN (PyTorch's propagation) and the **first** NaN — or,
    /// with no NaN, the first occurrence of the maximum — owns the gradient.
    fn beats(self, other: Self) -> bool;
}

impl ConvAcc for f32 {
    const ZERO: Self = 0.0;
    fn add(self, other: Self) -> Self {
        self + other
    }
    fn mul(self, other: Self) -> Self {
        self * other
    }
    fn div_count(self, count: usize) -> Self {
        self / (count as f32)
    }
    fn beats(self, other: Self) -> bool {
        if self.is_nan() {
            !other.is_nan()
        } else {
            !other.is_nan() && self > other
        }
    }
}

impl ConvAcc for f64 {
    const ZERO: Self = 0.0;
    fn add(self, other: Self) -> Self {
        self + other
    }
    fn mul(self, other: Self) -> Self {
        self * other
    }
    fn div_count(self, count: usize) -> Self {
        self / (count as f64)
    }
    fn beats(self, other: Self) -> bool {
        if self.is_nan() {
            !other.is_nan()
        } else {
            !other.is_nan() && self > other
        }
    }
}

impl ConvAcc for i64 {
    const ZERO: Self = 0;
    fn add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
    fn mul(self, other: Self) -> Self {
        self.wrapping_mul(other)
    }
    fn div_count(self, count: usize) -> Self {
        // Integer division truncates toward zero: an integer average is
        // deliberately not rounded (there is no single obvious rule, and the
        // float dtypes are the ones nn code pools with).
        self / (count as i64)
    }
    fn beats(self, other: Self) -> bool {
        self > other
    }
}

/// The storage index of logical element `(a, b, c, d)` of a rank-4 view.
fn idx4(layout: &Layout, a: usize, b: usize, c: usize, d: usize) -> usize {
    let s = layout.strides();
    layout.offset() + a * s[0] + b * s[1] + c * s[2] + d * s[3]
}

/// Row-major flat index into a contiguous rank-4 output buffer.
fn flat4(dims: [usize; 4], a: usize, b: usize, c: usize, d: usize) -> usize {
    ((a * dims[1] + b) * dims[2] + c) * dims[3] + d
}

/// The single output cast of the accumulation contract, applied to a whole
/// scatter-accumulated buffer.
fn narrow_all<E: Element>(acc: Vec<E::Acc>) -> Vec<E> {
    acc.into_iter().map(E::from_acc).collect()
}

// ---------------------------------------------------------------------------
// Dtype dispatch
// ---------------------------------------------------------------------------

/// Borrow the [`CpuStorage`] behind a CPU view, or report the op as
/// unsupported on a non-CPU device (no silent host round-trip).
// `op` is only read by the `metal`-gated arm; on a CPU-only build it is unused.
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    allow(unused_variables)
)]
fn cpu_storage<'a>(op: &'static str, x: View<'a>) -> Result<&'a CpuStorage> {
    match x.storage() {
        Storage::Cpu(s) => Ok(s),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Storage::Metal(_) => Err(Error::Unsupported {
            op,
            device: x.device(),
            dtype: x.dtype(),
        }),
    }
}

/// Run `$body` over the numeric element slice of one CPU view, re-wrapping the
/// produced `Vec<E>` as storage of the same dtype. `Bool` is rejected with
/// `Error::Unsupported`.
macro_rules! numeric1 {
    ($op:expr, $x:expr, |$a:ident| $body:expr) => {{
        let x: View<'_> = $x;
        match cpu_storage($op, x)? {
            CpuStorage::F16($a) => {
                let $a: &[half::f16] = $a;
                Ok(Storage::Cpu(CpuStorage::F16(Arc::new($body))))
            }
            CpuStorage::BF16($a) => {
                let $a: &[half::bf16] = $a;
                Ok(Storage::Cpu(CpuStorage::BF16(Arc::new($body))))
            }
            CpuStorage::F32($a) => {
                let $a: &[f32] = $a;
                Ok(Storage::Cpu(CpuStorage::F32(Arc::new($body))))
            }
            CpuStorage::F64($a) => {
                let $a: &[f64] = $a;
                Ok(Storage::Cpu(CpuStorage::F64(Arc::new($body))))
            }
            CpuStorage::I64($a) => {
                let $a: &[i64] = $a;
                Ok(Storage::Cpu(CpuStorage::I64(Arc::new($body))))
            }
            CpuStorage::Bool(_) => Err(Error::Unsupported {
                op: $op,
                device: x.device(),
                dtype: DType::Bool,
            }),
        }
    }};
}

/// Two-operand form of `numeric1`. The operands must share a dtype
/// (`Error::DTypeMismatch` otherwise).
macro_rules! numeric2 {
    ($op:expr, $x:expr, $y:expr, |$a:ident, $b:ident| $body:expr) => {{
        let x: View<'_> = $x;
        let y: View<'_> = $y;
        if x.dtype() != y.dtype() {
            return Err(Error::DTypeMismatch {
                op: $op,
                expected: x.dtype(),
                got: y.dtype(),
            });
        }
        match (cpu_storage($op, x)?, cpu_storage($op, y)?) {
            (CpuStorage::F16($a), CpuStorage::F16($b)) => {
                let ($a, $b): (&[half::f16], &[half::f16]) = ($a, $b);
                Ok(Storage::Cpu(CpuStorage::F16(Arc::new($body))))
            }
            (CpuStorage::BF16($a), CpuStorage::BF16($b)) => {
                let ($a, $b): (&[half::bf16], &[half::bf16]) = ($a, $b);
                Ok(Storage::Cpu(CpuStorage::BF16(Arc::new($body))))
            }
            (CpuStorage::F32($a), CpuStorage::F32($b)) => {
                let ($a, $b): (&[f32], &[f32]) = ($a, $b);
                Ok(Storage::Cpu(CpuStorage::F32(Arc::new($body))))
            }
            (CpuStorage::F64($a), CpuStorage::F64($b)) => {
                let ($a, $b): (&[f64], &[f64]) = ($a, $b);
                Ok(Storage::Cpu(CpuStorage::F64(Arc::new($body))))
            }
            (CpuStorage::I64($a), CpuStorage::I64($b)) => {
                let ($a, $b): (&[i64], &[i64]) = ($a, $b);
                Ok(Storage::Cpu(CpuStorage::I64(Arc::new($body))))
            }
            (CpuStorage::Bool(_), _) => Err(Error::Unsupported {
                op: $op,
                device: x.device(),
                dtype: DType::Bool,
            }),
            // Dtype equality was checked above, so the remaining cross-variant
            // pairs are unreachable; report loudly rather than silently.
            _ => Err(Error::DTypeMismatch {
                op: $op,
                expected: x.dtype(),
                got: y.dtype(),
            }),
        }
    }};
}

// ---------------------------------------------------------------------------
// Forward entry point
// ---------------------------------------------------------------------------

/// See [`BackendOps::conv`](crate::backend::BackendOps::conv).
pub(crate) fn conv(op: ConvOp, inputs: &[View<'_>], params: &Conv2dParams) -> Result<Storage> {
    match op {
        ConvOp::Conv2d => {
            let [input, weight] = operands("conv2d", inputs)?;
            let geo = Conv2dGeometry::conv2d(
                "conv2d",
                input.layout().dims(),
                weight.layout().dims(),
                params,
            )?;
            numeric2!("conv2d", input, weight, |a, b| conv2d_forward_generic(
                a,
                input.layout(),
                b,
                weight.layout(),
                &geo
            ))
        }
        ConvOp::MaxPool2d => {
            let [input] = operands("max_pool2d", inputs)?;
            let geo = Conv2dGeometry::pool("max_pool2d", input.layout().dims(), params)?;
            numeric1!("max_pool2d", input, |a| max_pool2d_forward_generic(
                a,
                input.layout(),
                &geo
            ))
        }
        ConvOp::AvgPool2d => {
            let [input] = operands("avg_pool2d", inputs)?;
            let geo = Conv2dGeometry::pool("avg_pool2d", input.layout().dims(), params)?;
            numeric1!("avg_pool2d", input, |a| avg_pool2d_forward_generic(
                a,
                input.layout(),
                &geo
            ))
        }
        ConvOp::Conv2dInputGrad => {
            let [grad, weight, input] = operands("conv2d", inputs)?;
            let geo = Conv2dGeometry::conv2d(
                "conv2d",
                input.layout().dims(),
                weight.layout().dims(),
                params,
            )?;
            conv2d_input_grad(grad, weight, &geo)
        }
        ConvOp::Conv2dWeightGrad => {
            let [grad, input, weight] = operands("conv2d", inputs)?;
            let geo = Conv2dGeometry::conv2d(
                "conv2d",
                input.layout().dims(),
                weight.layout().dims(),
                params,
            )?;
            conv2d_weight_grad(grad, input, &geo)
        }
        ConvOp::MaxPool2dBackward => {
            let [grad, input] = operands("max_pool2d", inputs)?;
            let geo = Conv2dGeometry::pool("max_pool2d", input.layout().dims(), params)?;
            max_pool2d_backward(grad, input, &geo)
        }
        ConvOp::AvgPool2dBackward => {
            let [grad, input] = operands("avg_pool2d", inputs)?;
            let geo = Conv2dGeometry::pool("avg_pool2d", input.layout().dims(), params)?;
            avg_pool2d_backward(grad, &geo)
        }
    }
}

/// Destructure the operand list a [`ConvOp`](crate::backend::ConvOp) variant
/// requires, or report the arity mismatch as
/// [`Error::InvalidArg`](crate::error::Error::InvalidArg).
fn operands<'a, const N: usize>(op: &'static str, inputs: &[View<'a>]) -> Result<[View<'a>; N]> {
    if inputs.len() != N {
        return Err(Error::InvalidArg {
            op,
            msg: format!("expected {N} operand view(s), got {}", inputs.len()),
        });
    }
    let mut slots: [Option<View<'a>>; N] = [None; N];
    for (slot, view) in slots.iter_mut().zip(inputs) {
        *slot = Some(*view);
    }
    Ok(slots.map(|v| v.expect("arity checked above")))
}

// ---------------------------------------------------------------------------
// Gradient entry points
// ---------------------------------------------------------------------------

/// Gradient of `conv2d` with respect to its **input**, given the output
/// cotangent `grad` (NCHW, the resolved output shape) and the forward `weight`
/// (OIHW). The result is a contiguous buffer shaped like the forward input.
pub(crate) fn conv2d_input_grad(
    grad: View<'_>,
    weight: View<'_>,
    geo: &Conv2dGeometry,
) -> Result<Storage> {
    expect_dims(geo.op, grad.layout(), geo.output_dims())?;
    expect_dims(geo.op, weight.layout(), geo.weight_dims())?;
    numeric2!(geo.op, grad, weight, |a, b| conv2d_input_grad_generic(
        a,
        grad.layout(),
        b,
        weight.layout(),
        geo
    ))
}

/// Gradient of `conv2d` with respect to its **weight**, given the output
/// cotangent `grad` (NCHW) and the forward `input` (NCHW). The result is a
/// contiguous buffer shaped like the forward weight (OIHW).
pub(crate) fn conv2d_weight_grad(
    grad: View<'_>,
    input: View<'_>,
    geo: &Conv2dGeometry,
) -> Result<Storage> {
    expect_dims(geo.op, grad.layout(), geo.output_dims())?;
    expect_dims(geo.op, input.layout(), geo.input_dims())?;
    numeric2!(geo.op, grad, input, |a, b| conv2d_weight_grad_generic(
        a,
        grad.layout(),
        b,
        input.layout(),
        geo
    ))
}

/// Gradient of `max_pool2d`: route each output cotangent to the single input
/// position that won its window (NaN-propagating, first position wins a tie).
/// Needs the forward `input` to recompute the winners.
pub(crate) fn max_pool2d_backward(
    grad: View<'_>,
    input: View<'_>,
    geo: &Conv2dGeometry,
) -> Result<Storage> {
    expect_dims(geo.op, grad.layout(), geo.output_dims())?;
    expect_dims(geo.op, input.layout(), geo.input_dims())?;
    numeric2!(geo.op, grad, input, |a, b| max_pool2d_backward_generic(
        a,
        grad.layout(),
        b,
        input.layout(),
        geo
    ))
}

/// Gradient of `avg_pool2d`: spread each output cotangent over the
/// non-padding positions of its window, divided by the full window area
/// (`count_include_pad = true`). Independent of the forward input values.
pub(crate) fn avg_pool2d_backward(grad: View<'_>, geo: &Conv2dGeometry) -> Result<Storage> {
    expect_dims(geo.op, grad.layout(), geo.output_dims())?;
    numeric1!(geo.op, grad, |a| avg_pool2d_backward_generic(
        a,
        grad.layout(),
        geo
    ))
}

/// Check that a gradient-kernel operand has exactly the shape the resolved
/// geometry says it must.
fn expect_dims(op: &'static str, layout: &Layout, want: [usize; 4]) -> Result<()> {
    if layout.rank() != 4 {
        return Err(Error::RankMismatch {
            op,
            expected: 4,
            got: layout.rank(),
        });
    }
    if layout.dims() != want.as_slice() {
        return Err(Error::ShapeMismatch {
            op,
            lhs: layout.shape().clone(),
            rhs: Shape::from(want.to_vec()),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Generic kernels
// ---------------------------------------------------------------------------

/// `out[n, oc, oh, ow] = Σ input[n, ic, ih, iw] · weight[oc, ic, kh, kw]`,
/// accumulated in `Acc` and cast back once per output element.
/// Drive an output-indexed conv or pool kernel over its
/// `[batch, channels, out_h, out_w]` output, splitting whole output rows across
/// threads.
///
/// Only the **forward** kernels can use this. Each output element here reads
/// its own window and owns its own accumulator, so the split is a pure
/// partition. The backward kernels instead scatter each cotangent into a shared
/// input-shaped accumulator, where two output positions can land on the same
/// slot — those stay sequential rather than take a lock or reassociate a
/// gradient sum.
///
/// `fill(row, n, c, oh)` receives one output row's slice and the coordinates it
/// belongs to.
fn for_each_output_row<E, F>(
    geo: &Conv2dGeometry,
    zero: E,
    cost_per_element: usize,
    fill: F,
) -> Vec<E>
where
    E: Clone + Send,
    F: Fn(&mut [E], usize, usize, usize) + Send + Sync,
{
    let len = geo.batch * geo.out_channels * geo.out_h * geo.out_w;
    crate::backend::parallel::build(len, zero, geo.out_w, cost_per_element, |base, window| {
        // Reached only for a non-empty output, so `out_w`/`out_h` are non-zero
        // and the decode below is well defined.
        for (row, out_row) in (base / geo.out_w..).zip(window.chunks_exact_mut(geo.out_w)) {
            let oh = row % geo.out_h;
            let plane = row / geo.out_h;
            fill(
                out_row,
                plane / geo.out_channels,
                plane % geo.out_channels,
                oh,
            );
        }
    })
}

fn conv2d_forward_generic<E>(
    input: &[E],
    input_l: &Layout,
    weight: &[E],
    weight_l: &Layout,
    geo: &Conv2dGeometry,
) -> Vec<E>
where
    E: Element,
    E::Acc: ConvAcc,
{
    let cost = geo.kernel_h * geo.kernel_w * geo.in_channels;
    for_each_output_row(
        geo,
        E::from_acc(<E::Acc as ConvAcc>::ZERO),
        cost,
        |out_row, n, oc, oh| {
            for (ow, slot) in out_row.iter_mut().enumerate() {
                let mut acc = <E::Acc as ConvAcc>::ZERO;
                for kh in 0..geo.kernel_h {
                    let Some(ih) = geo.source_h(oh, kh) else {
                        continue;
                    };
                    for kw in 0..geo.kernel_w {
                        let Some(iw) = geo.source_w(ow, kw) else {
                            continue;
                        };
                        for ic in 0..geo.in_channels {
                            let x = input[idx4(input_l, n, ic, ih, iw)].to_acc();
                            let w = weight[idx4(weight_l, oc, ic, kh, kw)].to_acc();
                            acc = acc.add(x.mul(w));
                        }
                    }
                }
                *slot = E::from_acc(acc);
            }
        },
    )
}

/// `input_grad[n, ic, ih, iw] = Σ grad[n, oc, oh, ow] · weight[oc, ic, kh, kw]`
/// over every window that reads `(ih, iw)`.
fn conv2d_input_grad_generic<E>(
    grad: &[E],
    grad_l: &Layout,
    weight: &[E],
    weight_l: &Layout,
    geo: &Conv2dGeometry,
) -> Vec<E>
where
    E: Element,
    E::Acc: ConvAcc,
{
    let mut acc =
        vec![<E::Acc as ConvAcc>::ZERO; geo.batch * geo.in_channels * geo.in_h * geo.in_w];
    for n in 0..geo.batch {
        for oc in 0..geo.out_channels {
            for oh in 0..geo.out_h {
                for ow in 0..geo.out_w {
                    let g = grad[idx4(grad_l, n, oc, oh, ow)].to_acc();
                    for kh in 0..geo.kernel_h {
                        let Some(ih) = geo.source_h(oh, kh) else {
                            continue;
                        };
                        for kw in 0..geo.kernel_w {
                            let Some(iw) = geo.source_w(ow, kw) else {
                                continue;
                            };
                            for ic in 0..geo.in_channels {
                                let w = weight[idx4(weight_l, oc, ic, kh, kw)].to_acc();
                                let dst = flat4(geo.input_dims(), n, ic, ih, iw);
                                acc[dst] = acc[dst].add(g.mul(w));
                            }
                        }
                    }
                }
            }
        }
    }
    narrow_all::<E>(acc)
}

/// `weight_grad[oc, ic, kh, kw] = Σ grad[n, oc, oh, ow] · input[n, ic, ih, iw]`
/// over every batch element and output position.
fn conv2d_weight_grad_generic<E>(
    grad: &[E],
    grad_l: &Layout,
    input: &[E],
    input_l: &Layout,
    geo: &Conv2dGeometry,
) -> Vec<E>
where
    E: Element,
    E::Acc: ConvAcc,
{
    let mut acc = vec![
        <E::Acc as ConvAcc>::ZERO;
        geo.out_channels * geo.in_channels * geo.kernel_h * geo.kernel_w
    ];
    for n in 0..geo.batch {
        for oc in 0..geo.out_channels {
            for oh in 0..geo.out_h {
                for ow in 0..geo.out_w {
                    let g = grad[idx4(grad_l, n, oc, oh, ow)].to_acc();
                    for kh in 0..geo.kernel_h {
                        let Some(ih) = geo.source_h(oh, kh) else {
                            continue;
                        };
                        for kw in 0..geo.kernel_w {
                            let Some(iw) = geo.source_w(ow, kw) else {
                                continue;
                            };
                            for ic in 0..geo.in_channels {
                                let x = input[idx4(input_l, n, ic, ih, iw)].to_acc();
                                let dst = flat4(geo.weight_dims(), oc, ic, kh, kw);
                                acc[dst] = acc[dst].add(g.mul(x));
                            }
                        }
                    }
                }
            }
        }
    }
    narrow_all::<E>(acc)
}

/// The `(ih, iw)` position that wins the pooling window at `(n, c, oh, ow)`,
/// or `None` for an empty window (unreachable after geometry validation, but
/// the kernel does not assume it).
fn max_source<E>(
    input: &[E],
    input_l: &Layout,
    geo: &Conv2dGeometry,
    n: usize,
    c: usize,
    oh: usize,
    ow: usize,
) -> Option<(usize, usize)>
where
    E: Element,
    E::Acc: ConvAcc,
{
    let mut best: Option<((usize, usize), E::Acc)> = None;
    for kh in 0..geo.kernel_h {
        let Some(ih) = geo.source_h(oh, kh) else {
            continue;
        };
        for kw in 0..geo.kernel_w {
            let Some(iw) = geo.source_w(ow, kw) else {
                continue;
            };
            let value = input[idx4(input_l, n, c, ih, iw)].to_acc();
            let take = match best {
                Some((_, current)) => value.beats(current),
                None => true,
            };
            if take {
                best = Some(((ih, iw), value));
            }
        }
    }
    best.map(|(pos, _)| pos)
}

/// Window maximum per output position.
fn max_pool2d_forward_generic<E>(input: &[E], input_l: &Layout, geo: &Conv2dGeometry) -> Vec<E>
where
    E: Element,
    E::Acc: ConvAcc,
{
    let zero = E::from_acc(<E::Acc as ConvAcc>::ZERO);
    for_each_output_row(
        geo,
        zero,
        geo.kernel_h * geo.kernel_w,
        |out_row, n, c, oh| {
            for (ow, slot) in out_row.iter_mut().enumerate() {
                *slot = match max_source(input, input_l, geo, n, c, oh, ow) {
                    // The winning element is copied through unchanged, so
                    // `max_pool2d` is exact for every dtype.
                    Some((ih, iw)) => input[idx4(input_l, n, c, ih, iw)],
                    None => zero,
                };
            }
        },
    )
}

/// Route each output cotangent to its window's winning input position.
fn max_pool2d_backward_generic<E>(
    grad: &[E],
    grad_l: &Layout,
    input: &[E],
    input_l: &Layout,
    geo: &Conv2dGeometry,
) -> Vec<E>
where
    E: Element,
    E::Acc: ConvAcc,
{
    let mut acc =
        vec![<E::Acc as ConvAcc>::ZERO; geo.batch * geo.in_channels * geo.in_h * geo.in_w];
    for n in 0..geo.batch {
        for c in 0..geo.out_channels {
            for oh in 0..geo.out_h {
                for ow in 0..geo.out_w {
                    let Some((ih, iw)) = max_source(input, input_l, geo, n, c, oh, ow) else {
                        continue;
                    };
                    let g = grad[idx4(grad_l, n, c, oh, ow)].to_acc();
                    let dst = flat4(geo.input_dims(), n, c, ih, iw);
                    acc[dst] = acc[dst].add(g);
                }
            }
        }
    }
    narrow_all::<E>(acc)
}

/// Window mean per output position, divided by the full window area
/// (`count_include_pad = true`).
fn avg_pool2d_forward_generic<E>(input: &[E], input_l: &Layout, geo: &Conv2dGeometry) -> Vec<E>
where
    E: Element,
    E::Acc: ConvAcc,
{
    let zero = E::from_acc(<E::Acc as ConvAcc>::ZERO);
    for_each_output_row(
        geo,
        zero,
        geo.kernel_h * geo.kernel_w,
        |out_row, n, c, oh| {
            for (ow, slot) in out_row.iter_mut().enumerate() {
                let mut acc = <E::Acc as ConvAcc>::ZERO;
                for kh in 0..geo.kernel_h {
                    let Some(ih) = geo.source_h(oh, kh) else {
                        continue;
                    };
                    for kw in 0..geo.kernel_w {
                        let Some(iw) = geo.source_w(ow, kw) else {
                            continue;
                        };
                        acc = acc.add(input[idx4(input_l, n, c, ih, iw)].to_acc());
                    }
                }
                *slot = E::from_acc(acc.div_count(geo.window()));
            }
        },
    )
}

/// Spread each output cotangent over the non-padding positions of its window.
fn avg_pool2d_backward_generic<E>(grad: &[E], grad_l: &Layout, geo: &Conv2dGeometry) -> Vec<E>
where
    E: Element,
    E::Acc: ConvAcc,
{
    let mut acc =
        vec![<E::Acc as ConvAcc>::ZERO; geo.batch * geo.in_channels * geo.in_h * geo.in_w];
    for n in 0..geo.batch {
        for c in 0..geo.out_channels {
            for oh in 0..geo.out_h {
                for ow in 0..geo.out_w {
                    let share = grad[idx4(grad_l, n, c, oh, ow)]
                        .to_acc()
                        .div_count(geo.window());
                    for kh in 0..geo.kernel_h {
                        let Some(ih) = geo.source_h(oh, kh) else {
                            continue;
                        };
                        for kw in 0..geo.kernel_w {
                            let Some(iw) = geo.source_w(ow, kw) else {
                                continue;
                            };
                            let dst = flat4(geo.input_dims(), n, c, ih, iw);
                            acc[dst] = acc[dst].add(share);
                        }
                    }
                }
            }
        }
    }
    narrow_all::<E>(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: Conv2dParams = Conv2dParams {
        kernel: (2, 2),
        stride: (1, 1),
        padding: (0, 0),
        dilation: (1, 1),
    };

    fn params(
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Conv2dParams {
        Conv2dParams {
            kernel,
            stride,
            padding,
            dilation,
        }
    }

    fn f32_storage(data: Vec<f32>) -> Storage {
        Storage::Cpu(CpuStorage::F32(Arc::new(data)))
    }

    fn as_f32(s: &Storage) -> Vec<f32> {
        match s {
            Storage::Cpu(CpuStorage::F32(v)) => v.as_ref().clone(),
            _ => panic!("expected f32 storage"),
        }
    }

    /// A tiny xorshift PRNG so the cross-check tests are deterministic without
    /// depending on the crate `Rng`'s stream.
    struct Prng(u64);

    impl Prng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn f(&mut self) -> f32 {
            (self.next() % 2001) as f32 / 500.0 - 2.0
        }
        fn values(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.f()).collect()
        }
    }

    // ----- forward goldens ----------------------------------------------

    #[test]
    fn conv2d_3x3_by_2x2_hand_computed() {
        // input 1x1x3x3 = 1..9, weight 1x1x2x2 = [[1,0],[0,1]] (the diagonal
        // pick), so each output is the sum of a 2x2 window's diagonal.
        let input = f32_storage((1..=9).map(|v| v as f32).collect());
        let weight = f32_storage(vec![1.0, 0.0, 0.0, 1.0]);
        let il = Layout::contiguous([1, 1, 3, 3]).unwrap();
        let wl = Layout::contiguous([1, 1, 2, 2]).unwrap();
        let out = conv(
            ConvOp::Conv2d,
            &[View::new(&input, &il), View::new(&weight, &wl)],
            &IDENTITY,
        )
        .unwrap();
        // 1+5, 2+6, 4+8, 5+9
        assert_eq!(as_f32(&out), vec![6.0, 8.0, 12.0, 14.0]);
    }

    #[test]
    fn conv2d_multi_channel_hand_computed() {
        // input 1x2x2x2 = 1..8, weight 1x2x2x2 of ones -> the sum, 36.
        let input = f32_storage((1..=8).map(|v| v as f32).collect());
        let weight = f32_storage(vec![1.0; 8]);
        let il = Layout::contiguous([1, 2, 2, 2]).unwrap();
        let wl = Layout::contiguous([1, 2, 2, 2]).unwrap();
        let out = conv(
            ConvOp::Conv2d,
            &[View::new(&input, &il), View::new(&weight, &wl)],
            &IDENTITY,
        )
        .unwrap();
        assert_eq!(as_f32(&out), vec![36.0]);
    }

    #[test]
    fn conv2d_padding_and_stride_hand_computed() {
        // input 1x1x2x2 = [[1,2],[3,4]], 2x2 kernel of ones, padding 1,
        // stride 2 -> padded 4x4, output 2x2, each window seeing one corner.
        let input = f32_storage(vec![1.0, 2.0, 3.0, 4.0]);
        let weight = f32_storage(vec![1.0; 4]);
        let il = Layout::contiguous([1, 1, 2, 2]).unwrap();
        let wl = Layout::contiguous([1, 1, 2, 2]).unwrap();
        let p = params((2, 2), (2, 2), (1, 1), (1, 1));
        let out = conv(
            ConvOp::Conv2d,
            &[View::new(&input, &il), View::new(&weight, &wl)],
            &p,
        )
        .unwrap();
        assert_eq!(as_f32(&out), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn conv2d_dilation_hand_computed() {
        // input 1x1x3x3 = 1..9, 2x2 kernel of ones dilated by 2 -> a single
        // output summing the four corners 1 + 3 + 7 + 9 = 20.
        let input = f32_storage((1..=9).map(|v| v as f32).collect());
        let weight = f32_storage(vec![1.0; 4]);
        let il = Layout::contiguous([1, 1, 3, 3]).unwrap();
        let wl = Layout::contiguous([1, 1, 2, 2]).unwrap();
        let p = params((2, 2), (1, 1), (0, 0), (2, 2));
        let out = conv(
            ConvOp::Conv2d,
            &[View::new(&input, &il), View::new(&weight, &wl)],
            &p,
        )
        .unwrap();
        assert_eq!(as_f32(&out), vec![20.0]);
    }

    #[test]
    fn conv2d_reads_strided_input_views() {
        // Two images stored back to back; convolve only the second by
        // narrowing the batch axis (a non-zero-offset view).
        let input = f32_storage((1..=18).map(|v| v as f32).collect());
        let il = Layout::contiguous([2, 1, 3, 3]).unwrap();
        let second = il.narrow(0, 1, 1).unwrap();
        let weight = f32_storage(vec![1.0, 0.0, 0.0, 1.0]);
        let wl = Layout::contiguous([1, 1, 2, 2]).unwrap();
        let out = conv(
            ConvOp::Conv2d,
            &[View::new(&input, &second), View::new(&weight, &wl)],
            &IDENTITY,
        )
        .unwrap();
        // The second image is 10..18: 10+14, 11+15, 13+17, 14+18.
        assert_eq!(as_f32(&out), vec![24.0, 26.0, 30.0, 32.0]);
    }

    #[test]
    fn conv2d_accumulates_f16_in_f32() {
        // 4096 channels of ones dotted with ones: exact in f32, saturating if
        // the kernel accumulated in f16.
        let k = 4096usize;
        let input = Storage::Cpu(CpuStorage::F16(Arc::new(vec![half::f16::from_f32(1.0); k])));
        let weight = Storage::Cpu(CpuStorage::F16(Arc::new(vec![half::f16::from_f32(1.0); k])));
        let il = Layout::contiguous([1, k, 1, 1]).unwrap();
        let wl = Layout::contiguous([1, k, 1, 1]).unwrap();
        let p = params((1, 1), (1, 1), (0, 0), (1, 1));
        let out = conv(
            ConvOp::Conv2d,
            &[View::new(&input, &il), View::new(&weight, &wl)],
            &p,
        )
        .unwrap();
        let got = match &out {
            Storage::Cpu(CpuStorage::F16(v)) => v.as_ref().clone(),
            _ => panic!("expected f16"),
        };
        assert_eq!(got[0].to_f32(), 4096.0);
    }

    #[test]
    fn conv2d_accumulates_bf16_in_f32() {
        let k = 4096usize;
        let one = half::bf16::from_f32(1.0);
        let input = Storage::Cpu(CpuStorage::BF16(Arc::new(vec![one; k])));
        let weight = Storage::Cpu(CpuStorage::BF16(Arc::new(vec![one; k])));
        let il = Layout::contiguous([1, k, 1, 1]).unwrap();
        let wl = Layout::contiguous([1, k, 1, 1]).unwrap();
        let p = params((1, 1), (1, 1), (0, 0), (1, 1));
        let out = conv(
            ConvOp::Conv2d,
            &[View::new(&input, &il), View::new(&weight, &wl)],
            &p,
        )
        .unwrap();
        let Storage::Cpu(CpuStorage::BF16(got)) = out else {
            panic!("expected bf16")
        };
        assert_eq!(got[0].to_f32(), 4096.0);
    }

    #[test]
    fn max_pool2d_hand_computed() {
        // 1x1x4x4 = 1..16, 2x2 window, stride 2.
        let input = f32_storage((1..=16).map(|v| v as f32).collect());
        let il = Layout::contiguous([1, 1, 4, 4]).unwrap();
        let p = params((2, 2), (2, 2), (0, 0), (1, 1));
        let out = conv(ConvOp::MaxPool2d, &[View::new(&input, &il)], &p).unwrap();
        assert_eq!(as_f32(&out), vec![6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn max_pool2d_propagates_nan() {
        let input = f32_storage(vec![1.0, f32::NAN, 3.0, 4.0]);
        let il = Layout::contiguous([1, 1, 2, 2]).unwrap();
        let p = params((2, 2), (2, 2), (0, 0), (1, 1));
        let out = conv(ConvOp::MaxPool2d, &[View::new(&input, &il)], &p).unwrap();
        assert!(as_f32(&out)[0].is_nan());
    }

    #[test]
    fn avg_pool2d_hand_computed() {
        let input = f32_storage((1..=16).map(|v| v as f32).collect());
        let il = Layout::contiguous([1, 1, 4, 4]).unwrap();
        let p = params((2, 2), (2, 2), (0, 0), (1, 1));
        let out = conv(ConvOp::AvgPool2d, &[View::new(&input, &il)], &p).unwrap();
        assert_eq!(as_f32(&out), vec![3.5, 5.5, 11.5, 13.5]);
    }

    #[test]
    fn avg_pool2d_counts_padding_in_the_divisor() {
        // 1x1x2x2 = [[1,2],[3,4]], 2x2 window, stride 2, padding 1: each
        // output window holds exactly one real element, divided by 4.
        let input = f32_storage(vec![1.0, 2.0, 3.0, 4.0]);
        let il = Layout::contiguous([1, 1, 2, 2]).unwrap();
        let p = params((2, 2), (2, 2), (1, 1), (1, 1));
        let out = conv(ConvOp::AvgPool2d, &[View::new(&input, &il)], &p).unwrap();
        assert_eq!(as_f32(&out), vec![0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn pooling_ignores_dilation() {
        let input = f32_storage((1..=16).map(|v| v as f32).collect());
        let il = Layout::contiguous([1, 1, 4, 4]).unwrap();
        let dense = params((2, 2), (2, 2), (0, 0), (1, 1));
        let dilated = params((2, 2), (2, 2), (0, 0), (3, 3));
        let a = conv(ConvOp::MaxPool2d, &[View::new(&input, &il)], &dense).unwrap();
        let b = conv(ConvOp::MaxPool2d, &[View::new(&input, &il)], &dilated).unwrap();
        assert_eq!(as_f32(&a), as_f32(&b));
    }

    // ----- cross-check against a naive reference ------------------------

    /// Straightforward f64 reference convolution over contiguous NCHW/OIHW
    /// buffers, written independently of the kernel above.
    #[allow(clippy::too_many_arguments)]
    fn reference_conv2d(
        input: &[f32],
        weight: &[f32],
        dims: [usize; 4],
        wdims: [usize; 4],
        out: [usize; 2],
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Vec<f32> {
        let [n, ic, h, w] = dims;
        let [oc, _, kh, kw] = wdims;
        let mut result = Vec::new();
        for b in 0..n {
            for o in 0..oc {
                for oh in 0..out[0] {
                    for ow in 0..out[1] {
                        let mut acc = 0.0f64;
                        for c in 0..ic {
                            for i in 0..kh {
                                for j in 0..kw {
                                    let sh = (oh * stride.0 + i * dilation.0) as isize
                                        - padding.0 as isize;
                                    let sw = (ow * stride.1 + j * dilation.1) as isize
                                        - padding.1 as isize;
                                    if sh < 0 || sw < 0 {
                                        continue;
                                    }
                                    let (sh, sw) = (sh as usize, sw as usize);
                                    if sh >= h || sw >= w {
                                        continue;
                                    }
                                    let x = input[((b * ic + c) * h + sh) * w + sw] as f64;
                                    let k = weight[((o * ic + c) * kh + i) * kw + j] as f64;
                                    acc += x * k;
                                }
                            }
                        }
                        result.push(acc as f32);
                    }
                }
            }
        }
        result
    }

    #[test]
    fn conv2d_matches_naive_reference_over_a_parameter_grid() {
        let mut rng = Prng(0x5eed_1234_abcd_0001);
        for &stride in &[(1, 1), (2, 1), (2, 2)] {
            for &padding in &[(0, 0), (1, 0), (1, 1), (2, 2)] {
                for &dilation in &[(1, 1), (2, 1)] {
                    let dims = [2usize, 3, 5, 6];
                    let wdims = [4usize, 3, 2, 3];
                    let p = params((wdims[2], wdims[3]), stride, padding, dilation);
                    let geo = Conv2dGeometry::conv2d("conv2d", &dims, &wdims, &p).unwrap();
                    let input = rng.values(dims.iter().product());
                    let weight = rng.values(wdims.iter().product());
                    let expected = reference_conv2d(
                        &input,
                        &weight,
                        dims,
                        wdims,
                        [geo.out_h, geo.out_w],
                        stride,
                        padding,
                        dilation,
                    );
                    let si = f32_storage(input);
                    let sw = f32_storage(weight);
                    let il = Layout::contiguous(dims).unwrap();
                    let wl = Layout::contiguous(wdims).unwrap();
                    let got = as_f32(
                        &conv(
                            ConvOp::Conv2d,
                            &[View::new(&si, &il), View::new(&sw, &wl)],
                            &p,
                        )
                        .unwrap(),
                    );
                    assert_eq!(got.len(), expected.len());
                    for (g, e) in got.iter().zip(expected.iter()) {
                        assert!(
                            (g - e).abs() < 1e-3,
                            "conv mismatch {g} vs {e} at {stride:?}/{padding:?}/{dilation:?}"
                        );
                    }
                }
            }
        }
    }

    // ----- gradients ----------------------------------------------------

    /// Central finite differences of `⟨grad, forward(values)⟩` with respect to
    /// `values`, where `forward` re-runs a kernel with a perturbed buffer.
    fn finite_difference(
        values: &[f32],
        grad: &[f32],
        forward: impl Fn(&[f32]) -> Vec<f32>,
    ) -> Vec<f32> {
        const EPS: f32 = 1e-2;
        let dot = |v: &[f32]| -> f64 {
            forward(v)
                .iter()
                .zip(grad)
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum()
        };
        (0..values.len())
            .map(|i| {
                let mut plus = values.to_vec();
                plus[i] += EPS;
                let mut minus = values.to_vec();
                minus[i] -= EPS;
                ((dot(&plus) - dot(&minus)) / f64::from(2.0 * EPS)) as f32
            })
            .collect()
    }

    fn assert_close(got: &[f32], expected: &[f32], what: &str) {
        assert_eq!(got.len(), expected.len(), "{what}: length");
        for (i, (g, e)) in got.iter().zip(expected).enumerate() {
            assert!(
                (g - e).abs() <= 1e-2 * (1.0 + e.abs()),
                "{what}: element {i}: {g} vs {e}"
            );
        }
    }

    #[test]
    fn conv2d_input_grad_hand_computed() {
        // 1x1x3x3 input, 1x1x2x2 weight [[1,2],[3,4]], unit cotangent: each
        // input position accumulates the weights of every window reading it.
        let grad = f32_storage(vec![1.0; 4]);
        let weight = f32_storage(vec![1.0, 2.0, 3.0, 4.0]);
        let gl = Layout::contiguous([1, 1, 2, 2]).unwrap();
        let wl = Layout::contiguous([1, 1, 2, 2]).unwrap();
        let geo =
            Conv2dGeometry::conv2d("conv2d", &[1, 1, 3, 3], &[1, 1, 2, 2], &IDENTITY).unwrap();
        let got = as_f32(
            &conv2d_input_grad(View::new(&grad, &gl), View::new(&weight, &wl), &geo).unwrap(),
        );
        assert_eq!(got, vec![1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0]);
    }

    #[test]
    fn conv2d_weight_grad_hand_computed() {
        // 1x1x3x3 input 1..9, unit cotangent over the 2x2 output: each weight
        // gets the sum of the inputs it multiplies.
        let grad = f32_storage(vec![1.0; 4]);
        let input = f32_storage((1..=9).map(|v| v as f32).collect());
        let gl = Layout::contiguous([1, 1, 2, 2]).unwrap();
        let il = Layout::contiguous([1, 1, 3, 3]).unwrap();
        let geo =
            Conv2dGeometry::conv2d("conv2d", &[1, 1, 3, 3], &[1, 1, 2, 2], &IDENTITY).unwrap();
        let got = as_f32(
            &conv2d_weight_grad(View::new(&grad, &gl), View::new(&input, &il), &geo).unwrap(),
        );
        // w00 sees 1,2,4,5 = 12 ; w01: 2,3,5,6 = 16 ; w10: 4,5,7,8 = 24 ;
        // w11: 5,6,8,9 = 28.
        assert_eq!(got, vec![12.0, 16.0, 24.0, 28.0]);
    }

    #[test]
    fn conv2d_gradients_match_finite_differences() {
        let mut rng = Prng(0x5eed_1234_abcd_0002);
        for &stride in &[(1, 1), (2, 2)] {
            for &padding in &[(0, 0), (1, 1)] {
                for &dilation in &[(1, 1), (2, 1)] {
                    let dims = [2usize, 2, 5, 5];
                    let wdims = [3usize, 2, 2, 3];
                    let p = params((wdims[2], wdims[3]), stride, padding, dilation);
                    let geo = Conv2dGeometry::conv2d("conv2d", &dims, &wdims, &p).unwrap();
                    let input = rng.values(dims.iter().product());
                    let weight = rng.values(wdims.iter().product());
                    let out_len: usize = geo.output_dims().iter().product();
                    let cotangent = rng.values(out_len);

                    let il = Layout::contiguous(dims).unwrap();
                    let wl = Layout::contiguous(wdims).unwrap();
                    let gl = Layout::contiguous(geo.output_dims()).unwrap();
                    let sg = f32_storage(cotangent.clone());
                    let si = f32_storage(input.clone());
                    let sw = f32_storage(weight.clone());

                    let analytic = as_f32(
                        &conv2d_input_grad(View::new(&sg, &gl), View::new(&sw, &wl), &geo).unwrap(),
                    );
                    let numeric = finite_difference(&input, &cotangent, |x| {
                        let sx = f32_storage(x.to_vec());
                        as_f32(
                            &conv(
                                ConvOp::Conv2d,
                                &[View::new(&sx, &il), View::new(&sw, &wl)],
                                &p,
                            )
                            .unwrap(),
                        )
                    });
                    assert_close(&analytic, &numeric, "conv2d input grad");

                    let analytic = as_f32(
                        &conv2d_weight_grad(View::new(&sg, &gl), View::new(&si, &il), &geo)
                            .unwrap(),
                    );
                    let numeric = finite_difference(&weight, &cotangent, |w| {
                        let sww = f32_storage(w.to_vec());
                        as_f32(
                            &conv(
                                ConvOp::Conv2d,
                                &[View::new(&si, &il), View::new(&sww, &wl)],
                                &p,
                            )
                            .unwrap(),
                        )
                    });
                    assert_close(&analytic, &numeric, "conv2d weight grad");
                }
            }
        }
    }

    #[test]
    fn max_pool2d_backward_routes_to_the_winner() {
        // 2x2 window over 1x1x4x4 = 1..16: the winners are 6, 8, 14, 16.
        let input = f32_storage((1..=16).map(|v| v as f32).collect());
        let grad = f32_storage(vec![1.0, 2.0, 3.0, 4.0]);
        let il = Layout::contiguous([1, 1, 4, 4]).unwrap();
        let gl = Layout::contiguous([1, 1, 2, 2]).unwrap();
        let p = params((2, 2), (2, 2), (0, 0), (1, 1));
        let geo = Conv2dGeometry::pool("max_pool2d", &[1, 1, 4, 4], &p).unwrap();
        let got = as_f32(
            &max_pool2d_backward(View::new(&grad, &gl), View::new(&input, &il), &geo).unwrap(),
        );
        let mut expected = vec![0.0f32; 16];
        expected[5] = 1.0; // value 6
        expected[7] = 2.0; // value 8
        expected[13] = 3.0; // value 14
        expected[15] = 4.0; // value 16
        assert_eq!(got, expected);
    }

    #[test]
    fn max_pool2d_backward_breaks_ties_by_first_position() {
        let input = f32_storage(vec![5.0, 5.0, 5.0, 5.0]);
        let grad = f32_storage(vec![7.0]);
        let il = Layout::contiguous([1, 1, 2, 2]).unwrap();
        let gl = Layout::contiguous([1, 1, 1, 1]).unwrap();
        let p = params((2, 2), (2, 2), (0, 0), (1, 1));
        let geo = Conv2dGeometry::pool("max_pool2d", &[1, 1, 2, 2], &p).unwrap();
        let got = as_f32(
            &max_pool2d_backward(View::new(&grad, &gl), View::new(&input, &il), &geo).unwrap(),
        );
        assert_eq!(got, vec![7.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn avg_pool2d_backward_spreads_over_the_window() {
        let grad = f32_storage(vec![4.0]);
        let gl = Layout::contiguous([1, 1, 1, 1]).unwrap();
        let p = params((2, 2), (2, 2), (0, 0), (1, 1));
        let geo = Conv2dGeometry::pool("avg_pool2d", &[1, 1, 2, 2], &p).unwrap();
        let got = as_f32(&avg_pool2d_backward(View::new(&grad, &gl), &geo).unwrap());
        assert_eq!(got, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn pool_gradients_match_finite_differences() {
        let mut rng = Prng(0x5eed_1234_abcd_0003);
        for &stride in &[(1, 1), (2, 2)] {
            for &padding in &[(0, 0), (1, 1)] {
                let dims = [2usize, 2, 5, 5];
                let p = params((2, 2), stride, padding, (1, 1));
                let il = Layout::contiguous(dims).unwrap();
                // Distinct, well-separated values keep the window maximum
                // unambiguous under the finite-difference perturbation.
                let input: Vec<f32> = (0..dims.iter().product::<usize>())
                    .map(|i| i as f32 * 0.5 - 5.0)
                    .collect();
                let si = f32_storage(input.clone());

                let geo = Conv2dGeometry::pool("max_pool2d", &dims, &p).unwrap();
                let gl = Layout::contiguous(geo.output_dims()).unwrap();
                let out_len: usize = geo.output_dims().iter().product();
                let cotangent = rng.values(out_len);
                let sg = f32_storage(cotangent.clone());

                let analytic = as_f32(
                    &max_pool2d_backward(View::new(&sg, &gl), View::new(&si, &il), &geo).unwrap(),
                );
                let numeric = finite_difference(&input, &cotangent, |x| {
                    let sx = f32_storage(x.to_vec());
                    as_f32(&conv(ConvOp::MaxPool2d, &[View::new(&sx, &il)], &p).unwrap())
                });
                assert_close(&analytic, &numeric, "max_pool2d grad");

                let geo = Conv2dGeometry::pool("avg_pool2d", &dims, &p).unwrap();
                let analytic = as_f32(&avg_pool2d_backward(View::new(&sg, &gl), &geo).unwrap());
                let numeric = finite_difference(&input, &cotangent, |x| {
                    let sx = f32_storage(x.to_vec());
                    as_f32(&conv(ConvOp::AvgPool2d, &[View::new(&sx, &il)], &p).unwrap())
                });
                assert_close(&analytic, &numeric, "avg_pool2d grad");
            }
        }
    }

    #[test]
    fn bool_is_unsupported() {
        let input = Storage::Cpu(CpuStorage::Bool(Arc::new(vec![true; 4])));
        let il = Layout::contiguous([1, 1, 2, 2]).unwrap();
        let p = params((2, 2), (2, 2), (0, 0), (1, 1));
        assert!(matches!(
            conv(ConvOp::MaxPool2d, &[View::new(&input, &il)], &p),
            Err(Error::Unsupported {
                op: "max_pool2d",
                dtype: DType::Bool,
                ..
            })
        ));
    }

    #[test]
    fn dtype_mismatch_is_loud() {
        let input = f32_storage(vec![1.0; 9]);
        let weight = Storage::Cpu(CpuStorage::I64(Arc::new(vec![1; 4])));
        let il = Layout::contiguous([1, 1, 3, 3]).unwrap();
        let wl = Layout::contiguous([1, 1, 2, 2]).unwrap();
        assert!(matches!(
            conv(
                ConvOp::Conv2d,
                &[View::new(&input, &il), View::new(&weight, &wl)],
                &IDENTITY
            ),
            Err(Error::DTypeMismatch { op: "conv2d", .. })
        ));
    }

    #[test]
    fn operand_arity_is_checked() {
        let input = f32_storage(vec![1.0; 9]);
        let il = Layout::contiguous([1, 1, 3, 3]).unwrap();
        assert!(matches!(
            conv(ConvOp::Conv2d, &[View::new(&input, &il)], &IDENTITY),
            Err(Error::InvalidArg { op: "conv2d", .. })
        ));
    }

    #[test]
    fn gradient_kernels_check_operand_shapes() {
        let geo =
            Conv2dGeometry::conv2d("conv2d", &[1, 1, 3, 3], &[1, 1, 2, 2], &IDENTITY).unwrap();
        let grad = f32_storage(vec![1.0; 9]);
        let weight = f32_storage(vec![1.0; 4]);
        let bad = Layout::contiguous([1, 1, 3, 3]).unwrap();
        let wl = Layout::contiguous([1, 1, 2, 2]).unwrap();
        assert!(matches!(
            conv2d_input_grad(View::new(&grad, &bad), View::new(&weight, &wl), &geo),
            Err(Error::ShapeMismatch { op: "conv2d", .. })
        ));
    }

    #[test]
    fn i64_pooling_is_supported() {
        let input = Storage::Cpu(CpuStorage::I64(Arc::new(vec![1, 5, 3, 2])));
        let il = Layout::contiguous([1, 1, 2, 2]).unwrap();
        let p = params((2, 2), (2, 2), (0, 0), (1, 1));
        let out = conv(ConvOp::MaxPool2d, &[View::new(&input, &il)], &p).unwrap();
        match &out {
            Storage::Cpu(CpuStorage::I64(v)) => assert_eq!(v.as_ref(), &vec![5]),
            _ => panic!("expected i64 storage"),
        }
    }
}
