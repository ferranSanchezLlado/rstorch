//! Test utilities shared across the crate and the fixture suite.
//!
//! [`check_grad`] is the **single** finite-difference gradient harness
//! (implementation-plan §4): every W3 op family writes its backward tests as
//! cases against this one signature, so no task invents its own FD checker.
//! T01 froze the signature; **T30** filled it (and T31 removes the
//! `#[ignore]`s that kept those cases dormant while `record` was a no-op).

use crate::backend::dispatch;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::storage::CpuStorage;
use crate::tensor::Tensor;

/// Slack over the single-evaluation roundoff bound `ε·|f| / h` allowed by the
/// noise floor (see [`check_grad`]). One evaluation of `f` rounds once, but the
/// arithmetic *inside* it rounds too, and a differencing chain amplifies that;
/// measured across the crate's op suites the worst case sits under 10× the
/// single-evaluation bound, so 32× is a ~3× margin.
const ROUNDOFF_SLACK: f64 = 32.0;

/// The relative spacing of `dtype` — the largest relative error one rounding
/// of a value of that dtype can introduce.
fn dtype_epsilon(dtype: DType) -> f64 {
    match dtype {
        DType::F16 => f64::from(half::f16::EPSILON),
        DType::BF16 => f64::from(half::bf16::EPSILON),
        DType::F32 => f64::from(f32::EPSILON),
        DType::F64 => f64::EPSILON,
        // Only reachable through a non-float objective, which cannot be
        // differentiated at all; treat it as exact.
        DType::I64 | DType::Bool => 0.0,
    }
}

/// Read a tensor's elements in row-major logical order as `f64`, whatever its
/// dtype (the dtype-agnostic bulk sibling of [`Tensor::item`]).
fn to_f64_vec(t: &Tensor) -> Result<Vec<f64>> {
    let host = dispatch::backend(t.device()).transfer_out(t.view())?;
    Ok(match &host {
        CpuStorage::F16(a) => a.iter().map(|v| v.to_f64()).collect(),
        CpuStorage::BF16(a) => a.iter().map(|v| v.to_f64()).collect(),
        CpuStorage::F32(a) => a.iter().map(|&v| v as f64).collect(),
        CpuStorage::F64(a) => a.to_vec(),
        CpuStorage::I64(a) => a.iter().map(|&v| v as f64).collect(),
        CpuStorage::Bool(a) => a.iter().map(|&v| if v { 1.0 } else { 0.0 }).collect(),
    })
}

/// The inverse of [`to_f64_vec`]: a contiguous tensor with `like`'s shape,
/// float dtype and device, holding `values` narrowed to that dtype.
fn from_f64_vec(values: &[f64], like: &Tensor) -> Result<Tensor> {
    let dims = like.dims().to_vec();
    let device = like.device();
    match like.dtype() {
        DType::F16 => Tensor::from_vec(
            values.iter().map(|&v| half::f16::from_f64(v)).collect(),
            dims,
            &device,
        ),
        DType::BF16 => Tensor::from_vec(
            values.iter().map(|&v| half::bf16::from_f64(v)).collect(),
            dims,
            &device,
        ),
        DType::F32 => Tensor::from_vec(values.iter().map(|&v| v as f32).collect(), dims, &device),
        DType::F64 => Tensor::from_vec(values.to_vec(), dims, &device),
        other => Err(Error::InvalidArg {
            op: "check_grad",
            msg: format!("cannot perturb a {other} tensor: only float inputs have gradients"),
        }),
    }
}

/// Check the analytic gradients of `f` at `inputs` against central finite
/// differences.
///
/// `f` maps the input tensors to a **single-element** output. Each input is
/// wrapped in a fresh [`Tensor::traced`] leaf, `f` is run once on those leaves
/// and differentiated with [`Tensor::backward`]; then, for every element of
/// every float input, the element is perturbed by `±eps` and the numeric
/// gradient `(f(x+eps) − f(x−eps)) / (x₊ − x₋)` is compared to the analytic
/// one.
///
/// The denominator is the **realized** difference of the two perturbed values
/// read back out of the tensor, not `2·eps`, so a perturbation that rounds
/// under the input's dtype still divides by what actually changed.
///
/// An element passes when its absolute **or** relative error is within `tol`
/// (relative to the larger of the two magnitudes) — the usual "is-close" rule,
/// so one `tol` serves both tiny and large gradients. Non-float inputs are
/// skipped: they carry no gradient. An input the analytic pass produced no
/// gradient for is checked against zero rather than ignored.
///
/// # The roundoff-noise floor
///
/// A central difference cannot be more accurate than the values it differences.
/// `f` is evaluated in the objective's own dtype, so it is known only to about
/// `ε·|f|`, and dividing by a step of `h` inflates that to `ε·|f| / h` in the
/// derivative — for an `f32` objective of magnitude 100 stepped by `1e-3`, some
/// `6e-3` of unavoidable noise. `tol` is therefore widened by that floor (times
/// a small slack for the rounding inside `f` itself): below it a disagreement
/// carries no information about the backward formula. Tighten a check by
/// raising `eps` — the floor shrinks with `1 / eps` — or by moving the
/// objective to a wider dtype, not by lowering `tol` past the noise.
///
/// # Errors
///
/// [`Error::InvalidArg`] (`op: "check_grad"`) for a non-positive `eps`, a
/// negative `tol`, an output that is not a single element, an analytic
/// gradient whose shape disagrees with its input, or the first element whose
/// error exceeds `tol` (the message names the input index, the flat element
/// index, and both values). Anything `f`, [`Tensor::traced`] or
/// [`Tensor::backward`] reports is propagated unchanged.
pub fn check_grad<F>(f: F, inputs: &[Tensor], eps: f64, tol: f64) -> Result<()>
where
    F: Fn(&[Tensor]) -> Result<Tensor>,
{
    if !(eps.is_finite() && eps > 0.0) {
        return Err(Error::InvalidArg {
            op: "check_grad",
            msg: format!("eps must be finite and positive, got {eps}"),
        });
    }
    if !(tol.is_finite() && tol >= 0.0) {
        return Err(Error::InvalidArg {
            op: "check_grad",
            msg: format!("tol must be finite and non-negative, got {tol}"),
        });
    }

    // --- analytic pass: trace every float input, differentiate once --------
    let traced: Vec<Tensor> = inputs
        .iter()
        .map(|t| {
            if t.dtype().is_float() {
                t.traced()
            } else {
                Ok(t.clone())
            }
        })
        .collect::<Result<_>>()?;
    let out = f(&traced)?;
    if out.num_elements() != 1 {
        return Err(Error::InvalidArg {
            op: "check_grad",
            msg: format!(
                "f must produce a single-element tensor, got shape {}",
                out.shape()
            ),
        });
    }
    let out_dtype = out.dtype();
    let epsilon = dtype_epsilon(out_dtype);
    let grads = out.backward()?;

    // --- numeric pass: one central difference per input element ------------
    for (i, input) in inputs.iter().enumerate() {
        if !input.dtype().is_float() {
            continue;
        }
        let base = to_f64_vec(input)?;
        // A traced input that never reached the output has an all-zero
        // gradient; checking against zero is stricter than skipping it.
        let analytic = match grads.wrt_input(&traced[i]) {
            Ok(g) => to_f64_vec(&g)?,
            Err(_) => vec![0.0; base.len()],
        };
        if analytic.len() != base.len() {
            return Err(Error::InvalidArg {
                op: "check_grad",
                msg: format!(
                    "input {i}: gradient has {} elements but the input has {}",
                    analytic.len(),
                    base.len()
                ),
            });
        }

        let mut probe: Vec<Tensor> = inputs.to_vec();
        for j in 0..base.len() {
            let mut shifted = base.clone();

            shifted[j] = base[j] + eps;
            let plus = from_f64_vec(&shifted, input)?;
            let x_plus = to_f64_vec(&plus)?[j];
            probe[i] = plus;
            let f_plus = f(&probe)?.item()?;

            shifted[j] = base[j] - eps;
            let minus = from_f64_vec(&shifted, input)?;
            let x_minus = to_f64_vec(&minus)?[j];
            probe[i] = minus;
            let f_minus = f(&probe)?.item()?;

            let denominator = x_plus - x_minus;
            if denominator == 0.0 {
                return Err(Error::InvalidArg {
                    op: "check_grad",
                    msg: format!(
                        "input {i} element {j}: eps {eps} vanishes under {} \
                         (perturbing {} changed nothing)",
                        input.dtype(),
                        base[j]
                    ),
                });
            }
            let numeric = (f_plus - f_minus) / denominator;

            let absolute = (analytic[j] - numeric).abs();
            let scale = analytic[j].abs().max(numeric.abs());
            let relative = if scale > 0.0 { absolute / scale } else { 0.0 };
            // The roundoff-noise floor of a central difference: `f` is only
            // known to `ε·|f|`, and dividing by a step of `h` inflates that to
            // `ε·|f| / h` in the derivative. Below it, a disagreement says
            // nothing about the backward formula.
            let floor =
                ROUNDOFF_SLACK * epsilon * f_plus.abs().max(f_minus.abs()) / denominator.abs();
            if absolute > tol + floor && relative > tol {
                return Err(Error::InvalidArg {
                    op: "check_grad",
                    msg: format!(
                        "input {i} element {j}: analytic {} vs numeric {} \
                         (absolute error {absolute:e}, relative error \
                         {relative:e}, tol {tol:e}, {} noise floor {floor:e})",
                        analytic[j], numeric, out_dtype
                    ),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;

    const CPU: Device = Device::Cpu;

    fn t(data: &[f32], shape: impl Into<crate::shape::Shape>) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape, &CPU).unwrap()
    }

    #[test]
    fn accepts_a_correct_backward() {
        // sum(x²·w) — both operands, broadcasting, and a reduction.
        let x = t(&[0.5, -1.25, 2.0], [3]);
        let w = t(&[1.5, -0.5, 0.25], [3]);
        check_grad(
            |i| i[0].mul(&i[0])?.mul(&i[1])?.sum_all(),
            &[x, w],
            1e-3,
            1e-3,
        )
        .unwrap();
    }

    #[test]
    fn accepts_the_output_dependent_unaries() {
        let x = t(&[0.4, -0.9, 1.3], [3]);
        for f in [
            |i: &[Tensor]| i[0].sigmoid()?.sum_all(),
            |i: &[Tensor]| i[0].tanh()?.sum_all(),
            |i: &[Tensor]| i[0].exp()?.sum_all(),
        ] {
            check_grad(f, std::slice::from_ref(&x), 1e-3, 1e-3).unwrap();
        }
    }

    #[test]
    fn rejects_a_wrong_backward() {
        // `detach` cuts the gradient: the analytic answer is zero where the
        // numeric one is not.
        let x = t(&[1.0, 2.0], [2]);
        let err = check_grad(
            |i| i[0].mul(&i[0].detach())?.sum_all(),
            std::slice::from_ref(&x),
            1e-3,
            1e-3,
        )
        .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("check_grad"), "{msg}");
        assert!(msg.contains("analytic"), "{msg}");
    }

    #[test]
    fn a_non_scalar_output_is_rejected() {
        let x = t(&[1.0, 2.0], [2]);
        assert!(matches!(
            check_grad(|i| i[0].mul_scalar(2.0), &[x], 1e-3, 1e-3),
            Err(Error::InvalidArg {
                op: "check_grad",
                ..
            })
        ));
    }

    #[test]
    fn bad_eps_and_tol_are_rejected() {
        let x = t(&[1.0], [1]);
        for (eps, tol) in [(0.0, 1e-3), (-1e-3, 1e-3), (1e-3, -1.0), (f64::NAN, 1e-3)] {
            assert!(matches!(
                check_grad(|i| i[0].sum_all(), std::slice::from_ref(&x), eps, tol),
                Err(Error::InvalidArg {
                    op: "check_grad",
                    ..
                })
            ));
        }
    }

    #[test]
    fn non_float_inputs_are_skipped_not_traced() {
        // An i64 index tensor rides along untouched; only the float input is
        // perturbed and checked.
        let x = t(&[1.0, -2.0, 3.0], [3]);
        let idx = Tensor::from_vec(vec![2i64], [1], &CPU).unwrap();
        check_grad(
            |i| i[0].index_select(0, &i[1])?.sum_all(),
            &[x, idx],
            1e-3,
            1e-3,
        )
        .unwrap();
    }

    #[test]
    fn an_untraceable_objective_is_reported_as_not_traced() {
        let x = t(&[1.0], [1]);
        assert!(matches!(
            check_grad(|i| i[0].detach().sum_all(), &[x], 1e-3, 1e-3),
            Err(Error::NotTraced { .. })
        ));
    }
}
