//! Test utilities shared across the crate and the fixture suite.
//!
//! [`check_grad`] is the **single** finite-difference gradient harness
//! (implementation-plan §4): every W3 op family writes its backward tests as
//! `#[ignore]`d cases against this one signature, so no task invents its own
//! FD checker. T01 freezes the signature with a stub body; **T30** fills it
//! (and T31 removes the `#[ignore]`s once the engine is live).

use crate::error::Result;
use crate::tensor::Tensor;

/// Check the analytic gradients of `f` at `inputs` against central finite
/// differences.
///
/// `f` maps the input tensors to a **scalar** output. For each input, each
/// element is perturbed by `±eps`; the numeric gradient
/// `(f(x+eps) − f(x−eps)) / (2·eps)` is compared to the gradient
/// [`backward`](Tensor::backward) produces, and an element whose absolute or
/// relative error exceeds `tol` fails.
///
/// Returns `Ok(())` when every input gradient is within `tol`, else an
/// [`Error`](crate::Error) describing the first mismatch (offending input,
/// flat index, analytic vs numeric value).
///
/// # Panics
/// The T01 stub `todo!()`s; from T30 on it does not panic (mismatches are
/// returned as errors).
pub fn check_grad<F>(f: F, inputs: &[Tensor], eps: f64, tol: f64) -> Result<()>
where
    F: Fn(&[Tensor]) -> Result<Tensor>,
{
    let _ = (f, inputs, eps, tol);
    todo!("T30: finite-difference gradient check")
}
