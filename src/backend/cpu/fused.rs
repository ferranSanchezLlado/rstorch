//! Fused CPU kernels for last-axis softmax, layer normalization, and optimizer
//! updates. Recorded `LayerNorm` additionally returns its normalized values and
//! inverse standard deviations, and accepts them back for its input gradient.
//!
//! Optimizer variants use the multi-output encoding documented on
//! [`BackendOps::fused`](crate::backend::BackendOps::fused).

use super::cpu_storage;
use crate::backend::cpu::acc::{FloatAcc, NumAcc};
use crate::backend::cpu::dispatch::{CpuElement, CpuFloat, dispatch_float};
use crate::backend::{FusedOp, View};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::storage::Storage;

/// See [`BackendOps::fused`](crate::backend::BackendOps::fused).
///
/// Operand encoding:
///
/// - `Softmax`: `[x]`, no scalars; normalizes the last axis.
/// - `LayerNorm`: `[x, weight, bias]`, `[eps]`; `weight` and `bias` are
///   rank-one views matching `x`'s last axis. With `save_stats=1`, `y` keeps
///   the input dtype while `xhat` and `inv_std` use its accumulation dtype.
///   Backward accepts reduced `grad`/`weight` with F32 saved statistics.
pub(crate) fn fused(op: FusedOp, inputs: &[View<'_>], scalars: &[f64]) -> Result<Vec<Storage>> {
    match op {
        FusedOp::Softmax => softmax(inputs, scalars).map(|output| vec![output]),
        FusedOp::LayerNorm => layer_norm(inputs, scalars),
        FusedOp::SgdStep => sgd_step(inputs, scalars),
        FusedOp::AdamStep => adam_step(inputs, scalars),
    }
}

fn sgd_step(inputs: &[View<'_>], scalars: &[f64]) -> Result<Vec<Storage>> {
    const OP: &str = "fused_sgd_step";
    if !(inputs.len() == 2 || inputs.len() == 3) || scalars.len() != 3 {
        return invalid_encoding(OP, inputs, scalars, "2 or 3", 3);
    }
    let [lr, momentum, weight_decay] = scalars else {
        unreachable!("arity validated")
    };
    validate_optimizer_views(OP, inputs, 2)?;
    crate::optim::validate::sgd_scalars(OP, *lr, *momentum, *weight_decay, inputs[0].dtype())?;
    if inputs.len() == 3
        && crate::optim::validate::effective_scalar(*momentum, inputs[0].dtype()) == 0.0
    {
        return Err(Error::InvalidArg {
            op: OP,
            msg: "a velocity input requires non-zero momentum".to_owned(),
        });
    }

    let param = cpu_storage(inputs[0]);
    let grad = cpu_storage(inputs[1]);
    let velocity = inputs.get(2).map(|view| cpu_storage(*view));
    dispatch_float!(inputs[0].dtype(), E => {
        // The guard above rejects a velocity operand under zero momentum, so
        // whenever one is supplied the step returns a velocity as well.
        let velocity = velocity.map(|state| (E::acc_slice(state), inputs[2].layout()));
        let (param, velocity) = sgd_generic::<E>(
            E::slice(param),
            inputs[0].layout(),
            E::slice(grad),
            inputs[1].layout(),
            velocity,
            E::scalar(*lr),
            E::scalar(*momentum),
            E::scalar(*weight_decay),
        );
        let mut outputs = vec![E::storage(param)];
        if let Some(velocity) = velocity {
            outputs.push(E::acc_storage(velocity));
        }
        Ok(outputs)
    })
}

#[allow(clippy::too_many_arguments)]
fn sgd_generic<E: Element>(
    param: &[E],
    param_layout: &Layout,
    grad: &[E],
    grad_layout: &Layout,
    velocity: Option<(&[E::Acc], &Layout)>,
    lr: E::Acc,
    momentum: E::Acc,
    weight_decay: E::Acc,
) -> (Vec<E>, Option<Vec<E::Acc>>)
where
    E::Acc: FloatAcc,
{
    let use_momentum = momentum != E::Acc::ZERO;
    let len = param_layout.num_elements();
    let dense = param_layout.is_contiguous()
        && grad_layout.is_contiguous()
        && velocity.is_none_or(|(_, layout)| layout.is_contiguous());
    // Contiguous parameter/gradient/state buffers are the normal optimizer
    // case. Avoiding offset_for_linear here removes one div+rem walk per
    // operand while leaving the strided fallback and its cost model intact.
    let step = |logical: usize| -> (E, E::Acc) {
        let param_index = if dense {
            logical
        } else {
            offset_for_linear(param_layout, logical)
        };
        let grad_index = if dense {
            logical
        } else {
            offset_for_linear(grad_layout, logical)
        };
        let p = param[param_index].to_acc();
        let grad = grad[grad_index].to_acc();
        let g = if weight_decay != E::Acc::ZERO {
            grad + p * weight_decay
        } else {
            grad
        };
        let direction = if let Some((values, layout)) = velocity {
            let index = if dense {
                logical
            } else {
                offset_for_linear(layout, logical)
            };
            values[index] * momentum + g
        } else {
            g
        };
        (E::from_acc(p - direction * lr), direction)
    };

    let zero_param = E::from_acc(E::Acc::ZERO);
    if use_momentum {
        let mut next_param = vec![zero_param; len];
        let mut next_velocity = vec![E::Acc::ZERO; len];
        crate::backend::parallel::for_each_row_mut2(
            len,
            (&mut next_param, 1),
            (&mut next_velocity, 1),
            OPTIMIZER_STEP_COST,
            |base, params, velocities| {
                for ((slot, vel), logical) in
                    params.iter_mut().zip(velocities.iter_mut()).zip(base..)
                {
                    (*slot, *vel) = step(logical);
                }
            },
        );
        (next_param, Some(next_velocity))
    } else {
        let next_param = crate::backend::parallel::build(
            len,
            zero_param,
            1,
            OPTIMIZER_STEP_COST,
            |base, params| {
                for (slot, logical) in params.iter_mut().zip(base..) {
                    *slot = step(logical).0;
                }
            },
        );
        (next_param, None)
    }
}

fn adam_step(inputs: &[View<'_>], scalars: &[f64]) -> Result<Vec<Storage>> {
    const OP: &str = "fused_adam_step";
    require_encoding(OP, inputs, 4, scalars, 8)?;
    validate_optimizer_views(OP, inputs, 2)?;
    let [
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        correction1,
        correction2,
        decoupled,
    ] = scalars
    else {
        unreachable!("arity validated")
    };
    crate::optim::validate::adam_scalars(OP, scalars, inputs[0].dtype())?;
    let param = cpu_storage(inputs[0]);
    let grad = cpu_storage(inputs[1]);
    let first = cpu_storage(inputs[2]);
    let second = cpu_storage(inputs[3]);

    dispatch_float!(inputs[0].dtype(), E => {
        let (param, first, second) = adam_generic::<E>(
            E::slice(param),
            inputs[0].layout(),
            E::slice(grad),
            inputs[1].layout(),
            E::acc_slice(first),
            inputs[2].layout(),
            E::acc_slice(second),
            inputs[3].layout(),
            E::scalar(*lr),
            E::scalar(*beta1),
            E::scalar(1.0 - *beta1),
            E::scalar(*beta2),
            E::scalar(1.0 - *beta2),
            E::scalar(*eps),
            E::scalar(*weight_decay),
            E::scalar(*correction1),
            E::scalar(*correction2),
            E::scalar(1.0 - *lr * *weight_decay),
            *decoupled == 1.0,
        );
        Ok(vec![
            E::storage(param),
            E::acc_storage(first),
            E::acc_storage(second),
        ])
    })
}

#[allow(clippy::too_many_arguments)]
fn adam_generic<E: Element>(
    param: &[E],
    param_layout: &Layout,
    grad: &[E],
    grad_layout: &Layout,
    m: &[E::Acc],
    m_layout: &Layout,
    v: &[E::Acc],
    v_layout: &Layout,
    lr: E::Acc,
    beta1: E::Acc,
    one_minus_beta1: E::Acc,
    beta2: E::Acc,
    one_minus_beta2: E::Acc,
    eps: E::Acc,
    weight_decay: E::Acc,
    correction1: E::Acc,
    correction2: E::Acc,
    decoupled_scale: E::Acc,
    decoupled: bool,
) -> (Vec<E>, Vec<E::Acc>, Vec<E::Acc>)
where
    E::Acc: FloatAcc,
{
    let len = param_layout.num_elements();
    let mut next_param = vec![E::from_acc(E::Acc::ZERO); len];
    let mut next_m = vec![E::Acc::ZERO; len];
    let mut next_v = vec![E::Acc::ZERO; len];
    let dense = param_layout.is_contiguous()
        && grad_layout.is_contiguous()
        && m_layout.is_contiguous()
        && v_layout.is_contiguous();
    // Keep one loop body for dense and strided layouts. Dense rows use direct
    // indexing; the fallback preserves the existing offset walk.
    crate::backend::parallel::for_each_row_mut3(
        len,
        (&mut next_param, 1),
        (&mut next_m, 1),
        (&mut next_v, 1),
        OPTIMIZER_STEP_COST,
        |base, params, ms, vs| {
            for (((slot, m_slot), v_slot), logical) in params
                .iter_mut()
                .zip(ms.iter_mut())
                .zip(vs.iter_mut())
                .zip(base..)
            {
                let param_index = if dense {
                    logical
                } else {
                    offset_for_linear(param_layout, logical)
                };
                let grad_index = if dense {
                    logical
                } else {
                    offset_for_linear(grad_layout, logical)
                };
                let m_index = if dense {
                    logical
                } else {
                    offset_for_linear(m_layout, logical)
                };
                let v_index = if dense {
                    logical
                } else {
                    offset_for_linear(v_layout, logical)
                };
                let p = param[param_index].to_acc();
                let mut g = grad[grad_index].to_acc();
                if weight_decay != E::Acc::ZERO && !decoupled {
                    g = g + p * weight_decay;
                }
                let next_m_value = m[m_index] * beta1 + g * one_minus_beta1;
                let next_v_value = v[v_index] * beta2 + (g * g) * one_minus_beta2;
                let direction =
                    (next_m_value / correction1) / ((next_v_value / correction2).sqrt() + eps);
                let mut next = p;
                if weight_decay != E::Acc::ZERO && decoupled {
                    next = next * decoupled_scale;
                }
                next = next - direction * lr;
                *slot = E::from_acc(next);
                *m_slot = next_m_value;
                *v_slot = next_v_value;
            }
        },
    );
    (next_param, next_m, next_v)
}

/// What one parameter costs in a fused optimizer step, in the work units
/// [`crate::backend::parallel`] budgets in.
///
/// Dominated not by the arithmetic but by [`offset_for_linear`], which walks
/// the layout with an integer divide and remainder **per axis, per operand** —
/// and Adam has four operands. Integer division is tens of cycles, so a
/// parameter element here costs far more than a streaming one.
const OPTIMIZER_STEP_COST: usize = 32;

fn offset_for_linear(layout: &Layout, mut logical: usize) -> usize {
    let mut offset = layout.offset();
    for (&dim, &stride) in layout.dims().iter().zip(layout.strides()).rev() {
        if dim != 0 {
            offset += (logical % dim) * stride;
            logical /= dim;
        }
    }
    offset
}

fn softmax(inputs: &[View<'_>], scalars: &[f64]) -> Result<Storage> {
    const OP: &str = "fused_softmax";
    require_encoding(OP, inputs, 1, scalars, 0)?;
    let x = inputs[0];
    require_last_axis(OP, x.layout())?;
    require_float(OP, x)?;
    validate_view(OP, x)?;

    let values = cpu_storage(x);
    if x.dtype() == DType::F32 && x.layout().is_contiguous() {
        // The one dtype with a dedicated flat-slice kernel.
        return Ok(f32::storage(softmax_contiguous_f32(
            f32::slice(values),
            x.layout(),
        )));
    }
    Ok(dispatch_float!(x.dtype(), E => {
        E::storage(softmax_generic::<E>(E::slice(values), x.layout()))
    }))
}

fn softmax_contiguous_f32(values: &[f32], layout: &Layout) -> Vec<f32> {
    let width = layout.dims()[layout.rank() - 1];
    let mut output = vec![0.0; layout.num_elements()];
    // Rows are independent, so windows are whole rows and each row's three
    // passes stay in one task — nothing is reassociated across a split.
    crate::backend::parallel::for_each_window_mut(
        &mut output,
        width,
        ROW_PASS_COST,
        |base, window| {
            let input = &values[base..base + window.len()];
            for (input_row, output_row) in input
                .chunks_exact(width)
                .zip(window.chunks_exact_mut(width))
            {
                let mut peak = f32::NEG_INFINITY;
                let mut has_nan = false;
                for &value in input_row {
                    peak = peak.max(value);
                    has_nan |= value.is_nan();
                }
                if has_nan {
                    output_row.fill(f32::NAN);
                    continue;
                }

                if peak == f32::NEG_INFINITY {
                    continue;
                }

                let mut denominator = 0.0;
                for (output, &value) in output_row.iter_mut().zip(input_row) {
                    let exponent = (value - peak).exp();
                    denominator += exponent;
                    *output = exponent;
                }
                let scale = denominator.recip();
                for value in output_row {
                    *value *= scale;
                }
            }
        },
    );
    output
}

/// How many work units **one element** of a row-normalizing kernel costs:
/// softmax and layernorm each sweep their row about three times (statistics,
/// then the transform), and softmax's middle pass evaluates `exp`, which is far
/// dearer than the multiply-add the unit is calibrated to.
///
/// Measured rather than guessed. Sequentially, `softmax_last_f32/128x512` runs
/// 65,536 elements in ~136 µs and `layernorm/forward_f32/128x512` the same
/// count in ~255 µs — 2 ns and 3.9 ns per element against the ~0.067 ns of one
/// calibrated unit, so an element here is worth roughly 30–60 of them. An
/// earlier guess of 8 left these kernels just under the parallel threshold,
/// where they picked up only two tasks and lost 13% to the sequential form.
///
/// Per element, not per row. The element-indexed drivers
/// ([`build`](crate::backend::parallel::build),
/// [`for_each_window_mut`](crate::backend::parallel::for_each_window_mut)) take
/// a per-element cost and multiply by the output length themselves; only
/// [`for_each_row_mut3`](crate::backend::parallel::for_each_row_mut3) is quoted
/// per row and scales this by `width`. Passing the per-row figure to an
/// element-indexed driver overstates the job by a factor of `width`, which sent
/// a 32×128 softmax onto the thread pool and made it 4.6× slower.
const ROW_PASS_COST: usize = 32;

fn softmax_generic<E>(values: &[E], layout: &Layout) -> Vec<E>
where
    E: Element,
    E::Acc: FloatAcc,
{
    let width = layout.dims()[layout.rank() - 1];
    let stride = layout.strides()[layout.rank() - 1];
    crate::backend::parallel::build(
        layout.num_elements(),
        E::from_acc(E::Acc::ZERO),
        width,
        ROW_PASS_COST,
        |base, window| {
            // Per task, not per row: the exponent scratch is reused across
            // every row this window owns.
            let mut exponents: Vec<E::Acc> = Vec::with_capacity(width);
            for (row, output_row) in (base / width..).zip(window.chunks_exact_mut(width)) {
                let base = row_base(layout, row);
                let mut peak = E::Acc::NEG_INFINITY;
                for col in 0..width {
                    let value = values[base + col * stride].to_acc();
                    peak = if peak.is_nan() || value.is_nan() {
                        // Match the composed Max reduction's NaN propagation.
                        peak + value
                    } else if value > peak {
                        value
                    } else {
                        peak
                    };
                }

                if peak == E::Acc::NEG_INFINITY {
                    // The buffer arrives zeroed, which is this row's answer.
                    continue;
                }

                exponents.clear();
                let mut denominator = E::Acc::ZERO;
                for col in 0..width {
                    let exponent = (values[base + col * stride].to_acc() - peak).exp();
                    denominator = denominator + exponent;
                    exponents.push(exponent);
                }
                for (slot, &value) in output_row.iter_mut().zip(exponents.iter()) {
                    *slot = E::from_acc(value / denominator);
                }
            }
        },
    )
}

fn layer_norm(inputs: &[View<'_>], scalars: &[f64]) -> Result<Vec<Storage>> {
    const OP: &str = "fused_layer_norm";
    if inputs.len() == 4 {
        return layer_norm_backward_input(inputs, scalars);
    }
    if inputs.len() != 3 || !(scalars.len() == 1 || scalars.len() == 2) {
        return Err(Error::InvalidArg {
            op: OP,
            msg: format!(
                "expected 3 input(s) and 1 or 2 scalar(s), got {} and {}",
                inputs.len(),
                scalars.len()
            ),
        });
    }
    let [x, weight, bias] = inputs else {
        unreachable!("arity validated")
    };
    require_last_axis(OP, x.layout())?;
    require_float(OP, *x)?;
    require_same_dtype(OP, *x, *weight)?;
    require_same_dtype(OP, *x, *bias)?;
    validate_view(OP, *x)?;
    validate_view(OP, *weight)?;
    validate_view(OP, *bias)?;

    let width = x.layout().dims()[x.layout().rank() - 1];
    for affine in [weight, bias] {
        if affine.layout().rank() != 1 || affine.layout().dims()[0] != width {
            return Err(Error::ShapeMismatch {
                op: OP,
                lhs: x.layout().shape().clone(),
                rhs: affine.layout().shape().clone(),
            });
        }
    }
    let eps = scalars[0];
    if !(eps.is_finite() && eps > 0.0) {
        return Err(Error::InvalidArg {
            op: OP,
            msg: format!("eps must be finite and positive, got {eps}"),
        });
    }
    let save_stats = match scalars.get(1) {
        None => false,
        Some(&1.0) => true,
        Some(value) => {
            return Err(Error::InvalidArg {
                op: OP,
                msg: format!("save_stats must be encoded as 1, got {value}"),
            });
        }
    };

    let [x_values, weight_values, bias_values] =
        [cpu_storage(*x), cpu_storage(*weight), cpu_storage(*bias)];
    let (output, stats) = dispatch_float!(x.dtype(), E => {
        let (output, stats) = layer_norm_generic::<E>(
            E::slice(x_values),
            x.layout(),
            E::slice(weight_values),
            weight.layout(),
            E::slice(bias_values),
            bias.layout(),
            E::scalar(eps),
            save_stats,
        );
        (
            E::storage(output),
            stats.map(|(xhat, inv_std)| (E::acc_storage(xhat), E::acc_storage(inv_std))),
        )
    });
    let mut outputs = vec![output];
    if let Some((xhat, inv_std)) = stats {
        outputs.push(xhat);
        outputs.push(inv_std);
    }
    Ok(outputs)
}

fn layer_norm_backward_input(inputs: &[View<'_>], scalars: &[f64]) -> Result<Vec<Storage>> {
    const OP: &str = "fused_layer_norm_backward_input";
    require_encoding(OP, inputs, 4, scalars, 0)?;
    let [g, xhat, inv_std, weight] = inputs else {
        unreachable!("arity validated")
    };
    require_last_axis(OP, g.layout())?;
    require_float(OP, *g)?;
    require_same_dtype(OP, *g, *weight)?;
    let stats_dtype = match g.dtype() {
        DType::F16 | DType::BF16 => DType::F32,
        dtype => dtype,
    };
    for input in [xhat, inv_std] {
        if input.dtype() != stats_dtype {
            return Err(Error::DTypeMismatch {
                op: OP,
                expected: stats_dtype,
                got: input.dtype(),
            });
        }
    }
    for input in [xhat, inv_std, weight] {
        if input.device() != g.device() {
            return Err(Error::DeviceMismatch {
                op: OP,
                expected: g.device(),
                got: input.device(),
            });
        }
    }
    for input in [g, xhat, inv_std, weight] {
        validate_view(OP, *input)?;
    }
    let rank = g.layout().rank();
    let width = g.layout().dims()[rank - 1];
    if xhat.layout().shape() != g.layout().shape()
        || inv_std.layout().rank() != rank
        || inv_std.layout().dims()[..rank - 1] != g.layout().dims()[..rank - 1]
        || inv_std.layout().dims()[rank - 1] != 1
        || weight.layout().rank() != 1
        || weight.layout().dims()[0] != width
    {
        return Err(Error::ShapeMismatch {
            op: OP,
            lhs: g.layout().shape().clone(),
            rhs: xhat.layout().shape().clone(),
        });
    }

    let [g_values, xhat_values, inv_std_values, weight_values] = [
        cpu_storage(*g),
        cpu_storage(*xhat),
        cpu_storage(*inv_std),
        cpu_storage(*weight),
    ];
    let output = dispatch_float!(g.dtype(), E => {
        E::storage(layer_norm_backward_input_generic::<E>(
            E::slice(g_values),
            g.layout(),
            E::acc_slice(xhat_values),
            xhat.layout(),
            E::acc_slice(inv_std_values),
            inv_std.layout(),
            E::slice(weight_values),
            weight.layout(),
        ))
    });
    Ok(vec![output])
}

#[allow(clippy::too_many_arguments)]
fn layer_norm_backward_input_generic<E: Element>(
    gradients: &[E],
    gradient_layout: &Layout,
    xhat: &[E::Acc],
    xhat_layout: &Layout,
    inv_std: &[E::Acc],
    inv_std_layout: &Layout,
    weights: &[E],
    weight_layout: &Layout,
) -> Vec<E>
where
    E::Acc: FloatAcc,
{
    let width = gradient_layout.dims()[gradient_layout.rank() - 1];
    let width_acc = E::Acc::from_usize(width);
    crate::backend::parallel::build(
        gradient_layout.num_elements(),
        E::from_acc(E::Acc::ZERO),
        width,
        ROW_PASS_COST,
        |base, window| {
            // Per task, not per row.
            let mut weighted: Vec<E::Acc> = Vec::with_capacity(width);
            for (row, output_row) in (base / width..).zip(window.chunks_exact_mut(width)) {
                let g_base = row_base(gradient_layout, row);
                let x_base = row_base(xhat_layout, row);
                let stat_base = row_base(inv_std_layout, row);
                let mut sum = E::Acc::ZERO;
                let mut projected = E::Acc::ZERO;
                weighted.clear();
                for col in 0..width {
                    let g = gradients
                        [g_base + col * gradient_layout.strides()[gradient_layout.rank() - 1]];
                    let x = xhat[x_base + col * xhat_layout.strides()[xhat_layout.rank() - 1]];
                    let weight = weights[weight_layout.offset() + col * weight_layout.strides()[0]];
                    let value = g.to_acc() * weight.to_acc();
                    sum = sum + value;
                    projected = projected + value * x;
                    weighted.push(value);
                }
                let inverse = inv_std[stat_base];
                for (col, slot) in output_row.iter_mut().enumerate() {
                    let x = xhat[x_base + col * xhat_layout.strides()[xhat_layout.rank() - 1]];
                    let centered = weighted[col] * width_acc - sum - x * projected;
                    *slot = E::from_acc(centered * inverse / width_acc);
                }
            }
        },
    )
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn layer_norm_generic<E>(
    values: &[E],
    layout: &Layout,
    weights: &[E],
    weight_layout: &Layout,
    biases: &[E],
    bias_layout: &Layout,
    eps: E::Acc,
    save_stats: bool,
) -> (Vec<E>, Option<(Vec<E::Acc>, Vec<E::Acc>)>)
where
    E: Element,
    E::Acc: FloatAcc,
{
    let width = layout.dims()[layout.rank() - 1];
    let rows = layout.num_elements() / width;
    let stride = layout.strides()[layout.rank() - 1];

    // One row's forward, writing the normalized output and — when the backward
    // pass needs them — that row's saved state. Written once and driven by
    // either the stats or no-stats loop below, so the two cannot diverge.
    let row_forward = |row: usize,
                       output_row: &mut [E],
                       saved: Option<(&mut [E::Acc], &mut E::Acc)>| {
        let base = row_base(layout, row);
        let mut sum = E::Acc::ZERO;
        for col in 0..width {
            sum = sum + values[base + col * stride].to_acc();
        }
        let mean = sum / E::Acc::from_usize(width);
        let mut squared = E::Acc::ZERO;
        for col in 0..width {
            let centered = values[base + col * stride].to_acc() - mean;
            squared = squared + centered * centered;
        }
        let scale = (squared / E::Acc::from_usize(width) + eps).sqrt();
        for (col, slot) in output_row.iter_mut().enumerate() {
            let normalized = (values[base + col * stride].to_acc() - mean) / scale;
            let weight =
                weights[weight_layout.offset() + col * weight_layout.strides()[0]].to_acc();
            let bias = biases[bias_layout.offset() + col * bias_layout.strides()[0]].to_acc();
            *slot = E::from_acc(normalized * weight + bias);
        }
        let Some((xhat_row, inv_slot)) = saved else {
            return;
        };
        if matches!(E::DTYPE, DType::F16 | DType::BF16) {
            let inverse = E::Acc::from_usize(1) / scale;
            *inv_slot = inverse;
            for (col, slot) in xhat_row.iter_mut().enumerate() {
                let centered = values[base + col * stride].to_acc() - mean;
                *slot = centered * inverse;
            }
        } else {
            // Preserve the established F32/F64 state encoding exactly.
            let state_mean = E::from_acc(mean).to_acc();
            let mut state_squared = E::Acc::ZERO;
            for col in 0..width {
                let centered = E::from_acc(values[base + col * stride].to_acc() - state_mean);
                let centered_acc = centered.to_acc();
                state_squared = state_squared + E::from_acc(centered_acc * centered_acc).to_acc();
            }
            let variance = E::from_acc(state_squared / E::Acc::from_usize(width));
            let variance_eps = E::from_acc(variance.to_acc() + eps);
            let std = E::from_acc(variance_eps.to_acc().sqrt());
            let inverse = E::from_acc(E::Acc::from_usize(1) / std.to_acc());
            *inv_slot = inverse.to_acc();
            for (col, slot) in xhat_row.iter_mut().enumerate() {
                let centered = E::from_acc(values[base + col * stride].to_acc() - state_mean);
                *slot = E::from_acc(centered.to_acc() / std.to_acc()).to_acc();
            }
        }
    };

    let zero = E::from_acc(E::Acc::ZERO);
    let mut output = vec![zero; layout.num_elements()];
    if !save_stats {
        crate::backend::parallel::for_each_window_mut(
            &mut output,
            width,
            ROW_PASS_COST,
            |base, window| {
                for (row, output_row) in (base / width..).zip(window.chunks_exact_mut(width)) {
                    row_forward(row, output_row, None);
                }
            },
        );
        return (output, None);
    }

    let mut xhat = vec![E::Acc::ZERO; layout.num_elements()];
    let mut inv_std = vec![E::Acc::ZERO; rows];
    // `inv_std` carries one element per row against the others' `width`, which
    // is exactly what the row driver's per-output widths express.
    crate::backend::parallel::for_each_row_mut3(
        rows,
        (&mut output, width),
        (&mut xhat, width),
        (&mut inv_std, 1),
        width * ROW_PASS_COST,
        |base, outputs, xhats, invs| {
            for (row, ((output_row, xhat_row), inv_slot)) in (base..).zip(
                outputs
                    .chunks_exact_mut(width)
                    .zip(xhats.chunks_exact_mut(width))
                    .zip(invs.iter_mut()),
            ) {
                row_forward(row, output_row, Some((xhat_row, inv_slot)));
            }
        },
    );
    (output, Some((xhat, inv_std)))
}

fn row_base(layout: &Layout, row: usize) -> usize {
    let last = layout.rank() - 1;
    let mut base = layout.offset();
    let mut remainder = row;
    for axis in (0..last).rev() {
        let dim = layout.dims()[axis];
        base += (remainder % dim) * layout.strides()[axis];
        remainder /= dim;
    }
    base
}

fn validate_optimizer_views(
    op: &'static str,
    inputs: &[View<'_>],
    parameter_inputs: usize,
) -> Result<()> {
    let param = inputs[0];
    require_float(op, param)?;
    let acc_dtype = match param.dtype() {
        DType::F16 | DType::BF16 => DType::F32,
        dtype => dtype,
    };
    for (index, &input) in inputs.iter().enumerate() {
        if input.device() != param.device() {
            return Err(Error::DeviceMismatch {
                op,
                expected: param.device(),
                got: input.device(),
            });
        }
        let expected = if index < parameter_inputs {
            param.dtype()
        } else {
            acc_dtype
        };
        if input.dtype() != expected {
            return Err(Error::DTypeMismatch {
                op,
                expected,
                got: input.dtype(),
            });
        }
        if input.layout().shape() != param.layout().shape() {
            return Err(Error::ShapeMismatch {
                op,
                lhs: param.layout().shape().clone(),
                rhs: input.layout().shape().clone(),
            });
        }
        validate_view(op, input)?;
    }
    Ok(())
}

fn invalid_encoding<T>(
    op: &'static str,
    inputs: &[View<'_>],
    scalars: &[f64],
    input_count: &str,
    scalar_count: usize,
) -> Result<T> {
    Err(Error::InvalidArg {
        op,
        msg: format!(
            "expected {input_count} input(s) and {scalar_count} scalar(s), got {} and {}",
            inputs.len(),
            scalars.len()
        ),
    })
}

fn require_encoding(
    op: &'static str,
    inputs: &[View<'_>],
    input_count: usize,
    scalars: &[f64],
    scalar_count: usize,
) -> Result<()> {
    if inputs.len() != input_count || scalars.len() != scalar_count {
        return Err(Error::InvalidArg {
            op,
            msg: format!(
                "expected {input_count} input(s) and {scalar_count} scalar(s), got {} and {}",
                inputs.len(),
                scalars.len()
            ),
        });
    }
    Ok(())
}

fn require_last_axis(op: &'static str, layout: &Layout) -> Result<()> {
    if layout.rank() == 0 {
        return Err(Error::RankMismatch {
            op,
            expected: 1,
            got: 0,
        });
    }
    if layout.dims()[layout.rank() - 1] == 0 {
        return Err(Error::InvalidArg {
            op,
            msg: "last axis must be non-empty".to_owned(),
        });
    }
    Ok(())
}

fn require_float(op: &'static str, view: View<'_>) -> Result<()> {
    if !view.dtype().is_float() {
        return Err(Error::Unsupported {
            op,
            device: view.device(),
            dtype: view.dtype(),
        });
    }
    Ok(())
}

fn require_same_dtype(op: &'static str, expected: View<'_>, got: View<'_>) -> Result<()> {
    if expected.dtype() != got.dtype() {
        return Err(Error::DTypeMismatch {
            op,
            expected: expected.dtype(),
            got: got.dtype(),
        });
    }
    Ok(())
}

fn validate_view(op: &'static str, view: View<'_>) -> Result<()> {
    if view.layout().num_elements() == 0 {
        return Ok(());
    }
    let mut highest = view.layout().offset();
    for (&dim, &stride) in view.layout().dims().iter().zip(view.layout().strides()) {
        highest = highest
            .checked_add(
                (dim - 1)
                    .checked_mul(stride)
                    .ok_or_else(|| Error::InvalidArg {
                        op,
                        msg: "view address overflows usize".to_owned(),
                    })?,
            )
            .ok_or_else(|| Error::InvalidArg {
                op,
                msg: "view address overflows usize".to_owned(),
            })?;
    }
    if highest >= view.storage().len() {
        return Err(Error::InvalidArg {
            op,
            msg: format!(
                "view reaches storage index {highest}, but backing storage has length {}",
                view.storage().len()
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
