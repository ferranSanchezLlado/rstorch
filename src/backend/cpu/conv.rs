//! Convolution/pooling CPU kernels.
//!
//! Per-variant input
//! contracts on [`ConvOp`](crate::backend::ConvOp); geometry in
//! [`Conv2dParams`](crate::backend::Conv2dParams); accumulation in `Acc`.
//!
//! # Layout and geometry
//!
//! Images are **NCHW** (`[batch, channels, height, width]`) and convolution
//! weights are **OIHW** (`[out_channels, in_channels, kernel_h, kernel_w]`),
//! matching `PyTorch`. Every operand is read stride-aware through its
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
//! # Semantics (PyTorch-familiar)
//!
//! - `Conv2d` is a **cross-correlation** (no kernel flip), like `PyTorch`'s.
//!   Bias is not a kernel operand: the op layer adds it as a broadcast add.
//! - `MaxPool2d` picks the window maximum with a NaN-propagating,
//!   first-position-wins ordering, so the forward value and the position its
//!   gradient is routed to always agree.
//! - `AvgPool2d` divides by the **full window area** (`kernel_h · kernel_w`),
//!   i.e. `PyTorch`'s `count_include_pad = true` default; positions that fall
//!   in the zero padding contribute `0` to the sum but still count in the
//!   divisor.
//! - Pooling ignores `dilation`, per
//!   [`Conv2dParams`](crate::backend::Conv2dParams).
//!
//! # Gradient kernels
//!
//! The four backward forms are routed through [`BackendOps::conv`](crate::backend::BackendOps::conv)
//! alongside their forward counterparts. Their saved forward operands make
//! the requested output shape and geometry explicit without exposing this
//! module's [`Conv2dGeometry`] through the backend contract.

use super::cpu_storage;
use crate::backend::conv_geometry::Conv2dGeometry;
use crate::backend::cpu::acc::NumAcc;
use crate::backend::cpu::dispatch::{CpuElement, dispatch_numeric};
use crate::backend::{Conv2dParams, ConvOp, View};
use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::storage::Storage;

/// The storage index of logical element `(a, b, c, d)` of a rank-4 view.
fn idx4(layout: &Layout, a: usize, b: usize, c: usize, d: usize) -> usize {
    let s = layout.strides();
    layout.offset() + a * s[0] + b * s[1] + c * s[2] + d * s[3]
}

// ---------------------------------------------------------------------------
// Dtype dispatch
// ---------------------------------------------------------------------------

/// Guard that a two-operand kernel's operands share a dtype, so the single
/// [`dispatch_numeric!`] that follows addresses both of them.
fn require_same_dtype(op: &'static str, x: View<'_>, y: View<'_>) -> Result<()> {
    if x.dtype() != y.dtype() {
        return Err(Error::DTypeMismatch {
            op,
            expected: x.dtype(),
            got: y.dtype(),
        });
    }
    Ok(())
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
            require_same_dtype("conv2d", input, weight)?;
            let (a, b) = (cpu_storage(input), cpu_storage(weight));
            dispatch_numeric!(input.dtype(), "conv2d", input.device(), E => {
                Ok(E::storage(conv2d_forward_generic(
                    E::slice(a),
                    input.layout(),
                    E::slice(b),
                    weight.layout(),
                    &geo,
                )))
            })
        }
        ConvOp::MaxPool2d => {
            let [input] = operands("max_pool2d", inputs)?;
            let geo = Conv2dGeometry::pool("max_pool2d", input.layout().dims(), params)?;
            let a = cpu_storage(input);
            dispatch_numeric!(input.dtype(), "max_pool2d", input.device(), E => {
                Ok(E::storage(max_pool2d_forward_generic(
                    E::slice(a),
                    input.layout(),
                    &geo,
                )))
            })
        }
        ConvOp::AvgPool2d => {
            let [input] = operands("avg_pool2d", inputs)?;
            let geo = Conv2dGeometry::pool("avg_pool2d", input.layout().dims(), params)?;
            let a = cpu_storage(input);
            dispatch_numeric!(input.dtype(), "avg_pool2d", input.device(), E => {
                Ok(E::storage(avg_pool2d_forward_generic(
                    E::slice(a),
                    input.layout(),
                    &geo,
                )))
            })
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
    require_same_dtype(geo.op, grad, weight)?;
    let (a, b) = (cpu_storage(grad), cpu_storage(weight));
    dispatch_numeric!(grad.dtype(), geo.op, grad.device(), E => {
        Ok(E::storage(conv2d_input_grad_generic(
            E::slice(a),
            grad.layout(),
            E::slice(b),
            weight.layout(),
            geo,
        )))
    })
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
    require_same_dtype(geo.op, grad, input)?;
    let (a, b) = (cpu_storage(grad), cpu_storage(input));
    dispatch_numeric!(grad.dtype(), geo.op, grad.device(), E => {
        Ok(E::storage(conv2d_weight_grad_generic(
            E::slice(a),
            grad.layout(),
            E::slice(b),
            input.layout(),
            geo,
        )))
    })
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
    require_same_dtype(geo.op, grad, input)?;
    let (a, b) = (cpu_storage(grad), cpu_storage(input));
    dispatch_numeric!(grad.dtype(), geo.op, grad.device(), E => {
        Ok(E::storage(max_pool2d_backward_generic(
            E::slice(a),
            grad.layout(),
            E::slice(b),
            input.layout(),
            geo,
        )))
    })
}

/// Gradient of `avg_pool2d`: spread each output cotangent over the
/// non-padding positions of its window, divided by the full window area
/// (`count_include_pad = true`). Independent of the forward input values.
pub(crate) fn avg_pool2d_backward(grad: View<'_>, geo: &Conv2dGeometry) -> Result<Storage> {
    expect_dims(geo.op, grad.layout(), geo.output_dims())?;
    let a = cpu_storage(grad);
    dispatch_numeric!(grad.dtype(), geo.op, grad.device(), E => {
        Ok(E::storage(avg_pool2d_backward_generic(
            E::slice(a),
            grad.layout(),
            geo,
        )))
    })
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
        // `want` is the requirement the resolved geometry imposes, so it is
        // `lhs`; the operand's own shape is what failed to meet it.
        return Err(Error::shape_mismatch(op, want, layout.shape()));
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
/// Each output element reads its own window and owns its own accumulator, so
/// the split is a pure partition. The input-shaped gradient kernels get the
/// same treatment from [`for_each_input_row`], which is why they are written as
/// gathers over [`Conv2dGeometry::window_h`] rather than as scatters into a
/// shared accumulator.
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

/// [`for_each_output_row`]'s counterpart for `conv2d`'s **input** gradient,
/// which is a scatter and so cannot be split by output element.
///
/// It can be split by **batch item**, because every contribution to
/// `input_grad[n, …]` comes from `grad[n, …]`: one batch item's volume is a
/// private accumulator that no other item touches. The kernel body is
/// therefore the plain sequential scatter, and each task's result is
/// bit-identical to what a single thread would have written — `rayon` changes
/// the schedule, never the sum.
///
/// `scatter(acc, n)` receives one batch item's `[in_channels, in_h, in_w]`
/// accumulator, indexed `(ic · in_h + ih) · in_w + iw`.
fn for_each_batch_volume<E, F>(geo: &Conv2dGeometry, cost_per_element: usize, scatter: F) -> Vec<E>
where
    E: Element,
    E::Acc: NumAcc,
    F: Fn(&mut [E::Acc], usize) + Send + Sync,
{
    let volume = geo.in_channels * geo.in_h * geo.in_w;
    let len = geo.batch * volume;
    crate::backend::parallel::build(
        len,
        E::from_acc(<E::Acc as NumAcc>::ZERO),
        volume,
        cost_per_element,
        |base, window| {
            // One wide accumulator per task, reused across the batch items in
            // it: the reset is a memset, the allocation would not be.
            let mut acc = vec![<E::Acc as NumAcc>::ZERO; volume];
            for (n, out_volume) in (base / volume..).zip(window.chunks_exact_mut(volume)) {
                acc.fill(<E::Acc as NumAcc>::ZERO);
                scatter(&mut acc, n);
                for (slot, value) in out_volume.iter_mut().zip(&acc) {
                    *slot = E::from_acc(*value);
                }
            }
        },
    )
}

/// [`for_each_batch_volume`] for the pool gradients, which scatter within one
/// `[in_h, in_w]` **channel** plane rather than a whole batch volume — pooling
/// never mixes channels, so the split is `batch × channels`-way.
///
/// `scatter(acc, n, c)` receives that plane's accumulator, indexed
/// `ih · in_w + iw`.
fn for_each_channel_plane<E, F>(geo: &Conv2dGeometry, cost_per_element: usize, scatter: F) -> Vec<E>
where
    E: Element,
    E::Acc: NumAcc,
    F: Fn(&mut [E::Acc], usize, usize) + Send + Sync,
{
    let plane = geo.in_h * geo.in_w;
    let len = geo.batch * geo.in_channels * plane;
    crate::backend::parallel::build(
        len,
        E::from_acc(<E::Acc as NumAcc>::ZERO),
        plane,
        cost_per_element,
        |base, window| {
            let mut acc = vec![<E::Acc as NumAcc>::ZERO; plane];
            for (index, out_plane) in (base / plane..).zip(window.chunks_exact_mut(plane)) {
                acc.fill(<E::Acc as NumAcc>::ZERO);
                scatter(&mut acc, index / geo.in_channels, index % geo.in_channels);
                for (slot, value) in out_plane.iter_mut().zip(&acc) {
                    *slot = E::from_acc(*value);
                }
            }
        },
    )
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
    E::Acc: NumAcc,
{
    let cost = geo.kernel_h * geo.kernel_w * geo.in_channels;
    for_each_output_row(
        geo,
        E::from_acc(<E::Acc as NumAcc>::ZERO),
        cost,
        |out_row, n, oc, oh| {
            for (ow, slot) in out_row.iter_mut().enumerate() {
                let mut acc = <E::Acc as NumAcc>::ZERO;
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
    E::Acc: NumAcc,
{
    // Cost per output element: the scatter's total work spread over the result.
    let cost =
        geo.out_channels * geo.window() * geo.out_h * geo.out_w / (geo.in_h * geo.in_w).max(1);
    for_each_batch_volume(geo, cost, |acc: &mut [E::Acc], n| {
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
                                let dst = (ic * geo.in_h + ih) * geo.in_w + iw;
                                acc[dst] = acc[dst].add(g.mul(w));
                            }
                        }
                    }
                }
            }
        }
    })
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
    E::Acc: NumAcc,
{
    // Weight-shaped, so the batch split of [`for_each_batch_volume`] is out —
    // every batch item contributes to every weight. The **output channel**
    // works instead: `weight_grad[oc, …]` is fed only by `grad[:, oc, …]`, so
    // one channel's slab is a private accumulator and the body stays the plain
    // sequential scatter, bit-identical to what one thread would write.
    let slab = geo.in_channels * geo.kernel_h * geo.kernel_w;
    let len = geo.out_channels * slab;
    let cost = geo.batch * geo.out_h * geo.out_w;
    crate::backend::parallel::build(
        len,
        E::from_acc(<E::Acc as NumAcc>::ZERO),
        slab,
        cost,
        |base, window| {
            let mut acc = vec![<E::Acc as NumAcc>::ZERO; slab];
            for (oc, out_slab) in (base / slab..).zip(window.chunks_exact_mut(slab)) {
                acc.fill(<E::Acc as NumAcc>::ZERO);
                for n in 0..geo.batch {
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
                                        let dst = (ic * geo.kernel_h + kh) * geo.kernel_w + kw;
                                        acc[dst] = acc[dst].add(g.mul(x));
                                    }
                                }
                            }
                        }
                    }
                }
                for (slot, value) in out_slab.iter_mut().zip(&acc) {
                    *slot = E::from_acc(*value);
                }
            }
        },
    )
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
    E::Acc: NumAcc,
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
    E::Acc: NumAcc,
{
    let zero = E::from_acc(<E::Acc as NumAcc>::ZERO);
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
    E::Acc: NumAcc,
{
    let cost = geo.window() * geo.out_h * geo.out_w / (geo.in_h * geo.in_w).max(1);
    for_each_channel_plane(geo, cost.max(1), |acc: &mut [E::Acc], n, c| {
        for oh in 0..geo.out_h {
            for ow in 0..geo.out_w {
                let Some((ih, iw)) = max_source(input, input_l, geo, n, c, oh, ow) else {
                    continue;
                };
                let g = grad[idx4(grad_l, n, c, oh, ow)].to_acc();
                let dst = ih * geo.in_w + iw;
                acc[dst] = acc[dst].add(g);
            }
        }
    })
}

/// Window mean per output position, divided by the full window area
/// (`count_include_pad = true`).
fn avg_pool2d_forward_generic<E>(input: &[E], input_l: &Layout, geo: &Conv2dGeometry) -> Vec<E>
where
    E: Element,
    E::Acc: NumAcc,
{
    let zero = E::from_acc(<E::Acc as NumAcc>::ZERO);
    for_each_output_row(
        geo,
        zero,
        geo.kernel_h * geo.kernel_w,
        |out_row, n, c, oh| {
            for (ow, slot) in out_row.iter_mut().enumerate() {
                let mut acc = <E::Acc as NumAcc>::ZERO;
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
    E::Acc: NumAcc,
{
    let cost = geo.window() * geo.out_h * geo.out_w / (geo.in_h * geo.in_w).max(1);
    for_each_channel_plane(geo, cost.max(1), |acc: &mut [E::Acc], n, c| {
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
                        let dst = ih * geo.in_w + iw;
                        acc[dst] = acc[dst].add(share);
                    }
                }
            }
        }
    })
}

#[cfg(test)]
mod tests;
