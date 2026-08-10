//! Fused CPU kernels for last-axis softmax, layer normalization, and optimizer
//! updates. Recorded LayerNorm additionally returns its normalized values and
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
    validate_sgd_scalars(OP, *lr, *momentum, *weight_decay, inputs[0].dtype())?;
    if inputs.len() == 3 && effective_scalar(*momentum, inputs[0].dtype()) == 0.0 {
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
    // One element's update, given its logical position. Written once and shared
    // by the momentum and no-momentum drivers below so the two spellings cannot
    // drift apart.
    let step = |logical: usize| -> (E, E::Acc) {
        let p = param[offset_for_linear(param_layout, logical)].to_acc();
        let grad = grad[offset_for_linear(grad_layout, logical)].to_acc();
        let g = if weight_decay != E::Acc::ZERO {
            grad + p * weight_decay
        } else {
            grad
        };
        let direction = if let Some((values, layout)) = velocity {
            values[offset_for_linear(layout, logical)] * momentum + g
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
    validate_adam_scalars(OP, scalars, inputs[0].dtype())?;
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
                let p = param[offset_for_linear(param_layout, logical)].to_acc();
                let mut g = grad[offset_for_linear(grad_layout, logical)].to_acc();
                if weight_decay != E::Acc::ZERO && !decoupled {
                    g = g + p * weight_decay;
                }
                let m = m[offset_for_linear(m_layout, logical)] * beta1 + g * one_minus_beta1;
                let v = v[offset_for_linear(v_layout, logical)] * beta2 + (g * g) * one_minus_beta2;
                let direction = (m / correction1) / ((v / correction2).sqrt() + eps);
                let mut next = p;
                if weight_decay != E::Acc::ZERO && decoupled {
                    next = next * decoupled_scale;
                }
                next = next - direction * lr;
                *slot = E::from_acc(next);
                *m_slot = m;
                *v_slot = v;
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

/// Range-check the SGD hyperparameters.
///
/// Also called by [`Sgd::step`](crate::optim::Sgd::step) *before* it mutates
/// anything: reaching this only from inside the kernel would mean an invalid
/// group hyperparameter is diagnosed part-way through the parameter walk,
/// leaving the step half-applied. Sharing one function keeps the up-front check
/// and the kernel's guard from drifting apart.
pub(crate) fn validate_sgd_scalars(
    op: &'static str,
    lr: f64,
    momentum: f64,
    weight_decay: f64,
    dtype: DType,
) -> Result<()> {
    validate_nonnegative(op, "lr", lr, dtype)?;
    validate_unit_interval(op, "momentum", momentum, false, dtype)?;
    validate_nonnegative(op, "weight_decay", weight_decay, dtype)
}

/// Range-check the Adam hyperparameters. Shared with
/// [`Adam::step`](crate::optim::Adam::step)'s up-front check for the reason
/// given on [`validate_sgd_scalars`].
pub(crate) fn validate_adam_scalars(op: &'static str, scalars: &[f64], dtype: DType) -> Result<()> {
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
    validate_nonnegative(op, "lr", *lr, dtype)?;
    validate_unit_interval(op, "beta1", *beta1, false, dtype)?;
    validate_unit_interval(op, "beta2", *beta2, false, dtype)?;
    validate_positive(op, "eps", *eps, dtype)?;
    validate_nonnegative(op, "weight_decay", *weight_decay, dtype)?;
    validate_unit_interval(op, "bias_correction1", *correction1, true, dtype)?;
    validate_unit_interval(op, "bias_correction2", *correction2, true, dtype)?;
    if !(*decoupled == 0.0 || *decoupled == 1.0) {
        return Err(Error::InvalidArg {
            op,
            msg: format!("decoupled must be encoded as 0 or 1, got {decoupled}"),
        });
    }
    if *decoupled == 1.0 {
        validate_effective(
            op,
            "1 - lr * weight_decay",
            1.0 - *lr * *weight_decay,
            dtype,
            |_| true,
            "finite",
        )?;
    }
    Ok(())
}

fn validate_nonnegative(op: &'static str, name: &str, value: f64, dtype: DType) -> Result<()> {
    validate_effective(
        op,
        name,
        value,
        dtype,
        |x| x >= 0.0,
        "finite and non-negative",
    )
}

fn validate_positive(op: &'static str, name: &str, value: f64, dtype: DType) -> Result<()> {
    validate_effective(op, name, value, dtype, |x| x > 0.0, "finite and positive")
}

fn validate_unit_interval(
    op: &'static str,
    name: &str,
    value: f64,
    include_zero: bool,
    dtype: DType,
) -> Result<()> {
    let valid = |x: f64| {
        if include_zero {
            x > 0.0 && x <= 1.0
        } else {
            (0.0..1.0).contains(&x)
        }
    };
    let expected = if include_zero {
        "in (0, 1]"
    } else {
        "in [0, 1)"
    };
    validate_effective(op, name, value, dtype, valid, expected)
}

fn validate_effective(
    op: &'static str,
    name: &str,
    value: f64,
    dtype: DType,
    valid: impl Fn(f64) -> bool,
    expected: &str,
) -> Result<()> {
    let effective = effective_scalar(value, dtype);
    if !value.is_finite() || !effective.is_finite() || !valid(effective) {
        return Err(Error::InvalidArg {
            op,
            msg: format!("{name} must be {expected} in the accumulation dtype, got {value}"),
        });
    }
    Ok(())
}

fn effective_scalar(value: f64, dtype: DType) -> f64 {
    if dtype == DType::F64 {
        value
    } else {
        f64::from(value as f32)
    }
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
mod tests {
    use super::*;
    use crate::storage::CpuStorage;
    use std::sync::Arc;

    fn storage(values: Vec<f32>) -> Storage {
        Storage::Cpu(CpuStorage::F32(Arc::new(values)))
    }

    fn values(storage: Storage) -> Vec<f32> {
        match storage {
            Storage::Cpu(CpuStorage::F32(values)) => Arc::unwrap_or_clone(values),
            _ => panic!("expected f32 CPU storage"),
        }
    }

    fn reduced_storage(dtype: DType, values: &[f32]) -> Storage {
        match dtype {
            DType::F16 => Storage::Cpu(CpuStorage::F16(Arc::new(
                values.iter().copied().map(half::f16::from_f32).collect(),
            ))),
            DType::BF16 => Storage::Cpu(CpuStorage::BF16(Arc::new(
                values.iter().copied().map(half::bf16::from_f32).collect(),
            ))),
            _ => panic!("expected reduced dtype"),
        }
    }

    fn reduced_values(storage: Storage) -> Vec<f32> {
        match storage {
            Storage::Cpu(CpuStorage::F16(values)) => {
                values.iter().map(|value| value.to_f32()).collect()
            }
            Storage::Cpu(CpuStorage::BF16(values)) => {
                values.iter().map(|value| value.to_f32()).collect()
            }
            _ => panic!("expected reduced CPU storage"),
        }
    }

    fn close(got: &[f32], expected: &[f32]) {
        assert_eq!(got.len(), expected.len());
        for (&got, &expected) in got.iter().zip(expected) {
            assert!((got - expected).abs() < 1e-6, "{got} != {expected}");
        }
    }

    fn one(mut outputs: Vec<Storage>) -> Storage {
        assert_eq!(outputs.len(), 1);
        outputs.pop().unwrap()
    }

    #[test]
    fn softmax_is_stable_and_handles_fully_masked_rows() {
        let x = storage(vec![
            1000.0,
            1001.0,
            1002.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ]);
        let layout = Layout::contiguous([2, 3]).unwrap();
        let composed = crate::tensor::Tensor::from_parts(x.clone(), layout.clone())
            .softmax(-1)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let got = values(one(
            fused(FusedOp::Softmax, &[View::new(&x, &layout)], &[]).unwrap()
        ));
        close(&got, &composed);
        close(&got[..3], &[0.090_030_57, 0.244_728_48, 0.665_240_94]);
        assert_eq!(&got[3..], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn reduced_softmax_matches_an_independent_f32_uniform_reference() {
        let width = 4096;
        let layout = Layout::contiguous([1, width]).unwrap();
        for dtype in [DType::F16, DType::BF16] {
            let x = reduced_storage(dtype, &vec![1.0; width]);
            let got = reduced_values(one(
                fused(FusedOp::Softmax, &[View::new(&x, &layout)], &[]).unwrap()
            ));
            let expected = match dtype {
                DType::F16 => half::f16::from_f32(1.0 / width as f32).to_f32(),
                DType::BF16 => half::bf16::from_f32(1.0 / width as f32).to_f32(),
                _ => unreachable!(),
            };
            assert!(got.iter().all(|&value| value == expected));
        }
    }

    #[test]
    fn softmax_reads_a_strided_last_axis_and_writes_contiguous_output() {
        let x = storage(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
        let base = Layout::contiguous([3, 2]).unwrap();
        let transposed = base.transpose(0, 1).unwrap();
        let got = values(one(fused(
            FusedOp::Softmax,
            &[View::new(&x, &transposed)],
            &[],
        )
        .unwrap()));
        close(&got[..3], &[0.090_030_57, 0.244_728_48, 0.665_240_94]);
        assert!(got[3] < 1e-8);
        assert!(got[4] < 1e-4);
        assert!(got[5] > 0.9999);
    }

    #[test]
    fn contiguous_f32_softmax_preserves_special_value_policy() {
        let layout = Layout::contiguous([3, 3]).unwrap();
        let x = storage(vec![
            1.0,
            2.0,
            3.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            4.0,
            f32::NAN,
            5.0,
        ]);
        let got = values(one(
            fused(FusedOp::Softmax, &[View::new(&x, &layout)], &[]).unwrap()
        ));

        close(&got[..3], &[0.090_030_57, 0.244_728_48, 0.665_240_94]);
        assert_eq!(&got[3..6], &[0.0, 0.0, 0.0]);
        assert!(got[6..].iter().all(|value| value.is_nan()));
    }

    #[test]
    fn layer_norm_applies_strided_affine_views() {
        let x = storage(vec![1.0, 2.0, 3.0, 4.0, 6.0, 8.0]);
        let x_layout = Layout::contiguous([2, 3]).unwrap();
        let weight = storage(vec![2.0, 99.0, 3.0, 99.0, 4.0]);
        let bias = storage(vec![1.0, 99.0, -1.0, 99.0, 0.5]);
        let affine_layout = Layout::from_parts(
            crate::shape::Shape::from([3]),
            vec![2].into_boxed_slice(),
            0,
        )
        .unwrap();
        let got = values(one(fused(
            FusedOp::LayerNorm,
            &[
                View::new(&x, &x_layout),
                View::new(&weight, &affine_layout),
                View::new(&bias, &affine_layout),
            ],
            &[1e-5],
        )
        .unwrap()));
        close(&got[..3], &[-1.449_471_2, -1.0, 5.398_942_5]);
        close(&got[3..], &[-1.449_485_3, -1.0, 5.398_970_6]);
    }

    #[test]
    fn layer_norm_saved_stats_match_the_composed_backward_state() {
        let x = storage(vec![0.5, -1.5, 2.0, 0.25, -0.75, 1.25]);
        let affine = storage(vec![1.0, 1.0, 1.0]);
        let bias = storage(vec![0.0, 0.0, 0.0]);
        let layout = Layout::contiguous([2, 3]).unwrap();
        let affine_layout = Layout::contiguous([3]).unwrap();
        let input = crate::tensor::Tensor::from_parts(x.clone(), layout.clone());
        let mean = input.mean_keepdim(-1).unwrap();
        let centered = input.sub(&mean).unwrap();
        let variance = centered.mul(&centered).unwrap().mean_keepdim(-1).unwrap();
        let std = variance.add_scalar(1e-5).unwrap().sqrt().unwrap();
        let expected_xhat = centered.div(&std).unwrap().to_vec::<f32>().unwrap();
        let expected_inv_std = crate::tensor::Tensor::ones([2, 1], DType::F32, &crate::Device::Cpu)
            .unwrap()
            .div(&std)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();

        let outputs = fused(
            FusedOp::LayerNorm,
            &[
                View::new(&x, &layout),
                View::new(&affine, &affine_layout),
                View::new(&bias, &affine_layout),
            ],
            &[1e-5, 1.0],
        )
        .unwrap();
        assert_eq!(outputs.len(), 3);
        assert_eq!(values(outputs[1].clone()), expected_xhat);
        assert_eq!(values(outputs[2].clone()), expected_inv_std);
    }

    #[test]
    fn layer_norm_fused_input_gradient_is_bit_exact_to_the_composed_formula() {
        let g = storage(vec![0.3, -0.7, 1.1, 0.25, 0.9, -1.3]);
        let xhat = storage(vec![-0.2, 1.4, -0.6, 0.8, -1.1, 0.35]);
        let inv_std = storage(vec![0.75, 1.25]);
        let weight = storage(vec![1.5, -0.5, 2.0]);
        let layout = Layout::contiguous([2, 3]).unwrap();
        let stat_layout = Layout::contiguous([2, 1]).unwrap();
        let weight_layout = Layout::contiguous([3]).unwrap();
        let tensor = |storage: &Storage, layout: &Layout| {
            crate::tensor::Tensor::from_parts(storage.clone(), layout.clone())
        };
        let (gt, xt, st, wt) = (
            tensor(&g, &layout),
            tensor(&xhat, &layout),
            tensor(&inv_std, &stat_layout),
            tensor(&weight, &weight_layout),
        );
        let weighted = gt.mul(&wt).unwrap();
        let sum = weighted.sum_to(&[2, 1]).unwrap();
        let projected = weighted.mul(&xt).unwrap().sum_to(&[2, 1]).unwrap();
        let expected = weighted
            .mul_scalar(3.0)
            .unwrap()
            .sub(&sum)
            .unwrap()
            .sub(&xt.mul(&projected).unwrap())
            .unwrap()
            .mul(&st)
            .unwrap()
            .div_scalar(3.0)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();

        let got = fused(
            FusedOp::LayerNorm,
            &[
                View::new(&g, &layout),
                View::new(&xhat, &layout),
                View::new(&inv_std, &stat_layout),
                View::new(&weight, &weight_layout),
            ],
            &[],
        )
        .unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(values(got[0].clone()), expected);
    }

    #[test]
    fn reduced_layer_norm_accumulates_mean_in_f32() {
        for dtype in [DType::F16, DType::BF16] {
            let values = [2048.0, 1.0, -2048.0, -1.0];
            let (x, weight, bias) = match dtype {
                DType::F16 => (
                    Storage::Cpu(CpuStorage::F16(Arc::new(
                        values.map(half::f16::from_f32).to_vec(),
                    ))),
                    Storage::Cpu(CpuStorage::F16(Arc::new(vec![half::f16::ONE; 4]))),
                    Storage::Cpu(CpuStorage::F16(Arc::new(vec![half::f16::ZERO; 4]))),
                ),
                DType::BF16 => (
                    Storage::Cpu(CpuStorage::BF16(Arc::new(
                        values.map(half::bf16::from_f32).to_vec(),
                    ))),
                    Storage::Cpu(CpuStorage::BF16(Arc::new(vec![half::bf16::ONE; 4]))),
                    Storage::Cpu(CpuStorage::BF16(Arc::new(vec![half::bf16::ZERO; 4]))),
                ),
                _ => unreachable!(),
            };
            let layout = Layout::contiguous([4]).unwrap();
            let result = one(fused(
                FusedOp::LayerNorm,
                &[
                    View::new(&x, &layout),
                    View::new(&weight, &layout),
                    View::new(&bias, &layout),
                ],
                &[1e-5],
            )
            .unwrap());
            let result: Vec<f32> = match result {
                Storage::Cpu(CpuStorage::F16(v)) => v.iter().map(|x| x.to_f32()).collect(),
                Storage::Cpu(CpuStorage::BF16(v)) => v.iter().map(|x| x.to_f32()).collect(),
                _ => panic!("expected reduced CPU storage"),
            };
            assert_eq!(result[0], -result[2]);
            assert_eq!(result[1], -result[3]);
        }
    }

    #[test]
    fn reduced_layer_norm_outputs_and_saved_stats_match_an_independent_f32_reference() {
        let input = [2048.0f32, 1.0, -2048.0, -1.0];
        let width = input.len();
        let mean = input.iter().sum::<f32>() / width as f32;
        let variance = input
            .iter()
            .map(|&value| {
                let centered = value - mean;
                centered * centered
            })
            .sum::<f32>()
            / width as f32;
        let inverse = 1.0 / (variance + 1e-5).sqrt();
        let expected_xhat: Vec<f32> = input
            .iter()
            .map(|&value| (value - mean) * inverse)
            .collect();
        let layout = Layout::contiguous([width]).unwrap();
        for dtype in [DType::F16, DType::BF16] {
            let x = reduced_storage(dtype, &input);
            let weight = reduced_storage(dtype, &vec![1.0; width]);
            let bias = reduced_storage(dtype, &vec![0.0; width]);
            let outputs = fused(
                FusedOp::LayerNorm,
                &[
                    View::new(&x, &layout),
                    View::new(&weight, &layout),
                    View::new(&bias, &layout),
                ],
                &[1e-5, 1.0],
            )
            .unwrap();
            assert_eq!(outputs[0].dtype(), dtype);
            assert_eq!(outputs[1].dtype(), DType::F32);
            assert_eq!(outputs[2].dtype(), DType::F32);
            let y = reduced_values(outputs[0].clone());
            let xhat = values(outputs[1].clone());
            let inv_std = values(outputs[2].clone());
            for ((&got_y, &got_xhat), &reference) in y.iter().zip(&xhat).zip(&expected_xhat) {
                let expected_y = match dtype {
                    DType::F16 => half::f16::from_f32(reference).to_f32(),
                    DType::BF16 => half::bf16::from_f32(reference).to_f32(),
                    _ => unreachable!(),
                };
                assert_eq!(got_y, expected_y);
                assert!((got_xhat - reference).abs() < 1e-6);
            }
            assert!((inv_std[0] - inverse).abs() < 1e-9);
        }
    }

    #[test]
    fn reduced_layer_norm_backward_uses_wide_stats_at_small_eps() {
        let layout = Layout::contiguous([1, 2]).unwrap();
        let stat_layout = Layout::contiguous([1, 1]).unwrap();
        let weight_layout = Layout::contiguous([2]).unwrap();
        for dtype in [DType::F16, DType::BF16] {
            let x = reduced_storage(dtype, &[1.0, 1.0]);
            let weight = reduced_storage(dtype, &[1.0, 1.0]);
            let bias = reduced_storage(dtype, &[0.0, 0.0]);
            let outputs = fused(
                FusedOp::LayerNorm,
                &[
                    View::new(&x, &layout),
                    View::new(&weight, &weight_layout),
                    View::new(&bias, &weight_layout),
                ],
                &[1e-12, 1.0],
            )
            .unwrap();
            assert_eq!(outputs[1].dtype(), DType::F32);
            assert_eq!(outputs[2].dtype(), DType::F32);
            assert_eq!(values(outputs[1].clone()), vec![0.0, 0.0]);
            let inverse = values(outputs[2].clone())[0];
            assert_eq!(inverse, 1_000_000.0);

            let g = reduced_storage(dtype, &[0.0, 0.001]);
            let backward = fused(
                FusedOp::LayerNorm,
                &[
                    View::new(&g, &layout),
                    View::new(&outputs[1], &layout),
                    View::new(&outputs[2], &stat_layout),
                    View::new(&weight, &weight_layout),
                ],
                &[],
            )
            .unwrap();
            let got = reduced_values(backward[0].clone());
            let quantized_g = match dtype {
                DType::F16 => half::f16::from_f32(0.001).to_f32(),
                DType::BF16 => half::bf16::from_f32(0.001).to_f32(),
                _ => unreachable!(),
            };
            let expected = [-0.5 * quantized_g * inverse, 0.5 * quantized_g * inverse];
            for (&got, expected) in got.iter().zip(expected) {
                let expected = match dtype {
                    DType::F16 => half::f16::from_f32(expected).to_f32(),
                    DType::BF16 => half::bf16::from_f32(expected).to_f32(),
                    _ => unreachable!(),
                };
                assert_eq!(got, expected);
            }

            let old_xhat = reduced_storage(dtype, &[0.0, 0.0]);
            let old_inv_std = reduced_storage(dtype, &[1.0]);
            assert!(matches!(
                fused(
                    FusedOp::LayerNorm,
                    &[
                        View::new(&g, &layout),
                        View::new(&old_xhat, &layout),
                        View::new(&old_inv_std, &stat_layout),
                        View::new(&weight, &weight_layout),
                    ],
                    &[],
                ),
                Err(Error::DTypeMismatch {
                    expected: DType::F32,
                    ..
                })
            ));
        }
    }

    #[test]
    fn sgd_matches_scalar_reference_for_strided_coupled_decay() {
        let param_values = Arc::new(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
        let grad_values = Arc::new(vec![0.5, 5.0, -1.0, 6.0, 2.0, 7.0]);
        let param = Storage::Cpu(CpuStorage::F32(param_values.clone()));
        let grad = Storage::Cpu(CpuStorage::F32(grad_values.clone()));
        let layout = Layout::contiguous([3, 2]).unwrap().transpose(0, 1).unwrap();
        let got = values(
            fused(
                FusedOp::SgdStep,
                &[View::new(&param, &layout), View::new(&grad, &layout)],
                &[0.1, 0.0, 0.2],
            )
            .unwrap()
            .remove(0),
        );
        let logical_param = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let logical_grad = [0.5, -1.0, 2.0, 5.0, 6.0, 7.0];
        let expected: Vec<_> = logical_param
            .iter()
            .zip(logical_grad)
            .map(|(&p, g)| p - (g + p * 0.2) * 0.1)
            .collect();
        close(&got, &expected);
        assert_eq!(param_values.as_slice(), &[1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
        assert_eq!(grad_values.as_slice(), &[0.5, 5.0, -1.0, 6.0, 2.0, 7.0]);
    }

    #[test]
    fn sgd_first_and_later_momentum_steps_preserve_wide_half_velocity() {
        let param = Storage::Cpu(CpuStorage::F16(Arc::new(vec![half::f16::from_f32(2048.0)])));
        let grad = Storage::Cpu(CpuStorage::F16(Arc::new(vec![half::f16::from_f32(1.0)])));
        let layout = Layout::contiguous([1]).unwrap();
        let first = fused(
            FusedOp::SgdStep,
            &[View::new(&param, &layout), View::new(&grad, &layout)],
            &[0.0, 0.999, 0.0],
        )
        .unwrap();
        assert_eq!(first.len(), 2);
        let Storage::Cpu(CpuStorage::F32(first_velocity)) = &first[1] else {
            panic!("expected wide f32 velocity")
        };
        assert_eq!(first_velocity.as_slice(), &[1.0]);

        let velocity = Storage::Cpu(CpuStorage::F32(Arc::new(vec![2048.0])));
        let later = fused(
            FusedOp::SgdStep,
            &[
                View::new(&param, &layout),
                View::new(&grad, &layout),
                View::new(&velocity, &layout),
            ],
            &[0.0, 0.999, 0.0],
        )
        .unwrap();
        let Storage::Cpu(CpuStorage::F32(next_velocity)) = &later[1] else {
            panic!("expected wide f32 velocity")
        };
        let expected = 2048.0f32 * 0.999 + 1.0;
        assert_eq!(next_velocity[0], expected);
        assert_ne!(next_velocity[0], half::f16::from_f32(expected).to_f32());
    }

    #[test]
    fn adam_and_adamw_match_scalar_reference_with_strided_state() {
        let param = storage(vec![1.0, 10.0, -2.0, 20.0]);
        let grad = storage(vec![0.5, 5.0, -0.25, 6.0]);
        let m = storage(vec![0.1, 1.0, -0.2, 2.0]);
        let v = storage(vec![0.3, 3.0, 0.4, 4.0]);
        let layout = Layout::contiguous([2, 2]).unwrap().transpose(0, 1).unwrap();
        let lr = 0.01f32;
        let beta1 = 0.9f32;
        let beta2 = 0.99f32;
        let eps = 1e-6f32;
        let decay = 0.1f32;
        let correction1 = 0.19f32;
        let correction2 = 0.0199f32;
        for decoupled in [false, true] {
            let outputs = fused(
                FusedOp::AdamStep,
                &[
                    View::new(&param, &layout),
                    View::new(&grad, &layout),
                    View::new(&m, &layout),
                    View::new(&v, &layout),
                ],
                &[
                    f64::from(lr),
                    f64::from(beta1),
                    f64::from(beta2),
                    f64::from(eps),
                    f64::from(decay),
                    f64::from(correction1),
                    f64::from(correction2),
                    f64::from(u8::from(decoupled)),
                ],
            )
            .unwrap();
            let got_p = values(outputs[0].clone());
            let got_m = values(outputs[1].clone());
            let got_v = values(outputs[2].clone());
            let ps = [1.0f32, -2.0, 10.0, 20.0];
            let gs = [0.5f32, -0.25, 5.0, 6.0];
            let ms = [0.1f32, -0.2, 1.0, 2.0];
            let vs = [0.3f32, 0.4, 3.0, 4.0];
            for i in 0..ps.len() {
                let g = if decoupled {
                    gs[i]
                } else {
                    gs[i] + ps[i] * decay
                };
                let next_m = ms[i] * beta1 + g * (1.0 - beta1);
                let next_v = vs[i] * beta2 + (g * g) * (1.0 - beta2);
                let direction = (next_m / correction1) / ((next_v / correction2).sqrt() + eps);
                let mut next_p = ps[i];
                if decoupled {
                    next_p *= 1.0 - lr * decay;
                }
                next_p -= direction * lr;
                assert_eq!(got_m[i], next_m);
                assert_eq!(got_v[i], next_v);
                assert_eq!(got_p[i], next_p);
            }
        }
    }

    #[test]
    fn optimizer_kernels_support_bf16_and_f64_parameters() {
        let layout = Layout::contiguous([2]).unwrap();
        let bf16 = |values: &[f32]| {
            Storage::Cpu(CpuStorage::BF16(Arc::new(
                values.iter().copied().map(half::bf16::from_f32).collect(),
            )))
        };
        let p = bf16(&[1.0, -2.0]);
        let g = bf16(&[0.5, -0.25]);
        let bf16_out = fused(
            FusedOp::SgdStep,
            &[View::new(&p, &layout), View::new(&g, &layout)],
            &[0.1, 0.0, 0.0],
        )
        .unwrap();
        assert!(matches!(bf16_out[0], Storage::Cpu(CpuStorage::BF16(_))));

        let f64_storage = |values: Vec<f64>| Storage::Cpu(CpuStorage::F64(Arc::new(values)));
        let p = f64_storage(vec![1.0, -2.0]);
        let g = f64_storage(vec![0.5, -0.25]);
        let m = f64_storage(vec![0.0, 0.0]);
        let v = f64_storage(vec![0.0, 0.0]);
        let f64_out = fused(
            FusedOp::AdamStep,
            &[
                View::new(&p, &layout),
                View::new(&g, &layout),
                View::new(&m, &layout),
                View::new(&v, &layout),
            ],
            &[0.1, 0.9, 0.999, 1e-8, 0.0, 0.1, 0.001, 0.0],
        )
        .unwrap();
        assert!(matches!(f64_out[0], Storage::Cpu(CpuStorage::F64(_))));
    }

    #[test]
    fn optimizer_invalid_encodings_hyperparameters_and_metadata_are_structured() {
        let x = storage(vec![1.0, 2.0]);
        let layout = Layout::contiguous([2]).unwrap();
        let view = View::new(&x, &layout);
        assert!(matches!(
            fused(FusedOp::Softmax, &[view], &[1.0]),
            Err(Error::InvalidArg {
                op: "fused_softmax",
                ..
            })
        ));
        assert!(matches!(
            fused(FusedOp::SgdStep, &[view], &[]),
            Err(Error::InvalidArg {
                op: "fused_sgd_step",
                ..
            })
        ));
        assert!(matches!(
            fused(FusedOp::AdamStep, &[view], &[]),
            Err(Error::InvalidArg {
                op: "fused_adam_step",
                ..
            })
        ));

        let wrong_shape = Layout::contiguous([1, 2]).unwrap();
        assert!(matches!(
            fused(
                FusedOp::SgdStep,
                &[view, View::new(&x, &wrong_shape)],
                &[0.1, 0.0, 0.0]
            ),
            Err(Error::ShapeMismatch { .. })
        ));
        let f64_grad = Storage::Cpu(CpuStorage::F64(Arc::new(vec![1.0, 2.0])));
        assert!(matches!(
            fused(
                FusedOp::SgdStep,
                &[view, View::new(&f64_grad, &layout)],
                &[0.1, 0.0, 0.0]
            ),
            Err(Error::DTypeMismatch { .. })
        ));
        assert!(matches!(
            fused(FusedOp::SgdStep, &[view, view], &[f64::NAN, 0.0, 0.0]),
            Err(Error::InvalidArg { .. })
        ));
        assert!(matches!(
            fused(FusedOp::SgdStep, &[view, view], &[0.1, 1.0, 0.0]),
            Err(Error::InvalidArg { .. })
        ));
        let half = Storage::Cpu(CpuStorage::F16(Arc::new(vec![half::f16::ZERO; 2])));
        let velocity = storage(vec![0.0, 0.0]);
        assert!(matches!(
            fused(
                FusedOp::SgdStep,
                &[
                    View::new(&half, &layout),
                    View::new(&half, &layout),
                    View::new(&velocity, &layout),
                ],
                &[0.1, f64::MIN_POSITIVE, 0.0]
            ),
            Err(Error::InvalidArg { .. })
        ));
        let zero = storage(vec![0.0, 0.0]);
        assert!(matches!(
            fused(
                FusedOp::AdamStep,
                &[
                    view,
                    view,
                    View::new(&zero, &layout),
                    View::new(&zero, &layout)
                ],
                &[0.1, 0.9, 0.999, 0.0, 0.0, 0.1, 0.001, 2.0]
            ),
            Err(Error::InvalidArg { .. })
        ));

        assert!(matches!(
            fused(
                FusedOp::AdamStep,
                &[
                    View::new(&half, &layout),
                    View::new(&half, &layout),
                    View::new(&half, &layout),
                    View::new(&half, &layout),
                ],
                &[0.1, 0.9, 0.999, 1e-8, 0.0, 0.1, 0.001, 0.0]
            ),
            Err(Error::DTypeMismatch {
                expected: DType::F32,
                got: DType::F16,
                ..
            })
        ));
    }

    #[test]
    fn non_float_and_out_of_bounds_views_are_rejected() {
        let integers = Storage::Cpu(CpuStorage::I64(Arc::new(vec![1, 2])));
        let layout = Layout::contiguous([2]).unwrap();
        assert!(matches!(
            fused(FusedOp::Softmax, &[View::new(&integers, &layout)], &[]),
            Err(Error::Unsupported {
                dtype: DType::I64,
                ..
            })
        ));

        let x = storage(vec![1.0]);
        assert!(matches!(
            fused(FusedOp::Softmax, &[View::new(&x, &layout)], &[]),
            Err(Error::InvalidArg {
                op: "fused_softmax",
                ..
            })
        ));
    }
}
