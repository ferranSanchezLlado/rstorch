//! Backend-neutral validation for optimizer kernel scalar contracts.

use crate::dtype::DType;
use crate::error::{Error, Result};

/// Validate the scalars passed to an SGD update kernel.
pub(crate) fn sgd_scalars(
    op: &'static str,
    lr: f64,
    momentum: f64,
    weight_decay: f64,
    dtype: DType,
) -> Result<()> {
    nonnegative(op, "lr", lr, dtype)?;
    unit_interval(op, "momentum", momentum, false, dtype)?;
    nonnegative(op, "weight_decay", weight_decay, dtype)
}

/// Validate the scalars passed to an Adam/AdamW update kernel.
pub(crate) fn adam_scalars(op: &'static str, scalars: &[f64], dtype: DType) -> Result<()> {
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
        unreachable!("arity validated");
    };
    nonnegative(op, "lr", *lr, dtype)?;
    unit_interval(op, "beta1", *beta1, false, dtype)?;
    unit_interval(op, "beta2", *beta2, false, dtype)?;
    positive(op, "eps", *eps, dtype)?;
    nonnegative(op, "weight_decay", *weight_decay, dtype)?;
    unit_interval(op, "bias_correction1", *correction1, true, dtype)?;
    unit_interval(op, "bias_correction2", *correction2, true, dtype)?;
    if !(*decoupled == 0.0 || *decoupled == 1.0) {
        return Err(Error::InvalidArg {
            op,
            msg: format!("decoupled must be encoded as 0 or 1, got {decoupled}"),
        });
    }
    if *decoupled == 1.0 {
        effective(
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

fn nonnegative(op: &'static str, name: &str, value: f64, dtype: DType) -> Result<()> {
    effective(
        op,
        name,
        value,
        dtype,
        |x| x >= 0.0,
        "finite and non-negative",
    )
}

fn positive(op: &'static str, name: &str, value: f64, dtype: DType) -> Result<()> {
    effective(op, name, value, dtype, |x| x > 0.0, "finite and positive")
}

fn unit_interval(
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
    effective(op, name, value, dtype, valid, expected)
}

fn effective(
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

/// Convert a scalar to the accumulation precision used by a kernel.
pub(crate) fn effective_scalar(value: f64, dtype: DType) -> f64 {
    if dtype == DType::F64 {
        value
    } else {
        f64::from(value as f32)
    }
}
