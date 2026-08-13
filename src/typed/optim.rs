//! Typed-model adapters for the runtime optimizers.
//!
//! These functions borrow the crate-private runtime-module adapter only for the
//! delegated call. Optimizer values, groups, moment buffers, clocks, and state
//! encoding are the existing runtime implementations, not typed copies.
//!
//! A model must be reconstructed or consumed into its new typed form before a
//! dtype or device transition. Optimizer state is then either discarded by
//! constructing a fresh optimizer or explicitly restored with the matching
//! `load_*_state` function; it is never retained behind stale model markers.
//!
//! Typed modules must obey [`Module`]'s stable-walk contract. The adapter
//! compares repeated typed read-only walks before delegation; an implementation
//! whose paths vary between calls violates that safe semantic contract and no
//! finite preflight can make its future walks predictable.

use super::nn::{self, Module, RuntimeModuleAdapter, stable_state};
use crate::persist::{Envelope, Limits, LoadOptions};
use crate::{Error, Grads, Result};
use std::path::Path;

pub use crate::optim::{Adam, AdamGroup, AdamW, Sgd, SgdGroup};

fn validate_advancing_clocks(envelope: &Envelope) -> Result<()> {
    let Some(section) = envelope.section("optimizer") else {
        return Ok(());
    };
    for line in section.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if key == "steps" || key.starts_with("clock.") {
            let value = value.parse::<u64>().map_err(|_| Error::Persistence {
                msg: format!("optimizer state `{key}` is not a u64 clock: {value:?}"),
            })?;
            if value == u64::MAX {
                return Err(Error::Persistence {
                    msg: format!(
                        "optimizer state `{key}` is u64::MAX and cannot advance on another step"
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Applies one SGD update to a typed model, consuming `grads` by move.
///
/// Validation performed by SGD before its update and any backend failure keep
/// the runtime optimizer's documented semantics. In particular, this adapter
/// does not add rollback after a backend failure has begun applying updates.
///
/// Reusing the consumed gradients does not compile:
///
/// ```compile_fail
/// use rstorch::typed::{DeviceCtx, Tensor2};
/// use rstorch::typed::nn::{Forward, Linear, Mode};
/// use rstorch::typed::optim::{Sgd, sgd_step};
/// use rstorch::Rng;
///
/// let ctx = DeviceCtx::cpu().unwrap();
/// let mut model = Linear::<1, 1>::new(1, 1, &ctx, &mut Rng::seed(1)).unwrap();
/// let input = Tensor2::<1, 1>::from_vec(vec![1.0], [1, 1], &ctx).unwrap();
/// let loss = model.forward(&input, Mode::TRAIN).unwrap().sum_all().unwrap();
/// let grads = loss.backward().unwrap();
/// let mut optimizer = Sgd::new(0.1);
/// sgd_step(&mut optimizer, &mut model, grads).unwrap();
/// sgd_step(&mut optimizer, &mut model, grads).unwrap();
/// ```
///
/// # Errors
///
/// [`Error::InvalidArg`] if `model`'s walk is malformed (a stability-contract
/// violation); otherwise propagates [`Sgd::step`]'s own validation and
/// backend errors.
pub fn sgd_step<M: Module + ?Sized>(
    optimizer: &mut Sgd,
    model: &mut M,
    grads: Grads,
) -> Result<()> {
    stable_state(model, "typed::optim::sgd_step")?;
    optimizer.step(&mut RuntimeModuleAdapter::new(model), grads)
}

/// Applies one Adam or `AdamW` update to a typed model, consuming `grads` by move.
///
/// [`AdamW::new`] returns the same [`Adam`] implementation with decoupled
/// weight decay, so this one adapter covers both names exactly.
///
/// # Errors
///
/// As [`sgd_step`], against [`Adam::step`].
pub fn adam_step<M: Module + ?Sized>(
    optimizer: &mut Adam,
    model: &mut M,
    grads: Grads,
) -> Result<()> {
    stable_state(model, "typed::optim::adam_step")?;
    optimizer.step(&mut RuntimeModuleAdapter::new(model), grads)
}

/// Writes SGD state into an envelope using the typed model's exact paths.
///
/// # Errors
///
/// [`Error::InvalidArg`] if `model`'s walk is malformed; otherwise propagates
/// [`Sgd::save_state`]'s own errors.
pub fn save_sgd_state<M: Module + ?Sized>(
    optimizer: &Sgd,
    model: &mut M,
    envelope: &mut Envelope,
) -> Result<()> {
    stable_state(model, "typed::optim::save_sgd_state")?;
    optimizer.save_state(&RuntimeModuleAdapter::new(model), envelope)
}

/// Stages and restores SGD state against the target typed model.
///
/// The runtime loader checks every untrusted clock and wide buffer before
/// replacing any optimizer state. The model itself is not modified.
///
/// # Errors
///
/// [`Error::InvalidArg`] if `model`'s walk is malformed;
/// [`Error::Persistence`] if a clock in `envelope` is not a valid `u64` or
/// would overflow on the next step; otherwise propagates
/// [`Sgd::load_state`]'s own validation errors, none of which touch `model`.
pub fn load_sgd_state<M: Module + ?Sized>(
    optimizer: &mut Sgd,
    model: &mut M,
    envelope: &Envelope,
) -> Result<()> {
    stable_state(model, "typed::optim::load_sgd_state")?;
    validate_advancing_clocks(envelope)?;
    optimizer.load_state(&RuntimeModuleAdapter::new(model), envelope)
}

/// Writes Adam or `AdamW` state into an envelope using exact typed-model paths.
///
/// # Errors
///
/// As [`save_sgd_state`], against [`Adam::save_state`].
pub fn save_adam_state<M: Module + ?Sized>(
    optimizer: &Adam,
    model: &mut M,
    envelope: &mut Envelope,
) -> Result<()> {
    stable_state(model, "typed::optim::save_adam_state")?;
    optimizer.save_state(&RuntimeModuleAdapter::new(model), envelope)
}

/// Stages and restores Adam or `AdamW` state against the target typed model.
///
/// # Errors
///
/// As [`load_sgd_state`], against [`Adam::load_state`].
pub fn load_adam_state<M: Module + ?Sized>(
    optimizer: &mut Adam,
    model: &mut M,
    envelope: &Envelope,
) -> Result<()> {
    stable_state(model, "typed::optim::load_adam_state")?;
    validate_advancing_clocks(envelope)?;
    optimizer.load_state(&RuntimeModuleAdapter::new(model), envelope)
}

/// Reads one parameter's clock by locating its runtime `Param` and asking the
/// optimizer directly.
///
/// The obvious implementation — serialise the optimizer state and scan for the
/// `clock.<path>=` line — costs a full device-to-host copy of every moment
/// buffer (1x model bytes for SGD with momentum, 2x for Adam) to return a single
/// `u64`, and makes a read-only inspector fail for unrelated reasons: a
/// parameter path containing `=` or a newline is a `Persistence` error, as is
/// any backend transfer failure. Both runtime optimizers already answer in O(1)
/// from a `grad_key` lookup, so walk to the `Param` and delegate.
///
/// An unseen parameter reports zero, matching the runtime accessors, so a
/// parameter frozen before the first step reports 0 while its stepped sibling
/// reports 1.
fn param_steps<M, F>(model: &mut M, path: &str, op: &'static str, clock: F) -> Result<u64>
where
    M: Module + ?Sized,
    F: Fn(&crate::nn::Param) -> u64,
{
    let state = stable_state(model, op)?;
    if !state.is_param_path(path) {
        return Err(Error::InvalidArg {
            op,
            msg: format!("typed model has no parameter path {path:?}"),
        });
    }
    let adapter = RuntimeModuleAdapter::new(model);
    let mut found = 0;
    crate::nn::visit::visit_all(&adapter, &mut |leaf_path, leaf| {
        if leaf_path == path
            && let crate::nn::visit::Leaf::Param(param) = leaf
        {
            found = clock(param);
        }
    });
    Ok(found)
}

/// Returns SGD's saved update clock for the typed parameter at `path`.
///
/// As with runtime `Sgd::param_steps`, an unseen parameter reports zero. Typed
/// callers identify parameters by the same stable dotted paths used by groups
/// and checkpoints; malformed or path-varying module walks remain errors.
///
/// # Errors
///
/// [`Error::InvalidArg`] if `model`'s walk is malformed or if `path` is not
/// a parameter of `model`.
pub fn sgd_param_steps<M: Module + ?Sized>(
    optimizer: &Sgd,
    model: &mut M,
    path: &str,
) -> Result<u64> {
    param_steps(model, path, "typed::optim::sgd_param_steps", |param| {
        optimizer.param_steps(param)
    })
}

/// Returns Adam or `AdamW`'s saved bias-correction clock at `path`.
///
/// # Errors
///
/// As [`sgd_param_steps`].
pub fn adam_param_steps<M: Module + ?Sized>(
    optimizer: &Adam,
    model: &mut M,
    path: &str,
) -> Result<u64> {
    param_steps(model, path, "typed::optim::adam_param_steps", |param| {
        optimizer.param_steps(param)
    })
}

/// The message for the doubly-unlucky case: the optimizer half failed *and*
/// putting the model back failed too. Named rather than inlined so the test
/// suite can pin its wording — no reachable input produces it.
fn rollback_error(load: Error, model: Error) -> Error {
    Error::Persistence {
        msg: format!(
            "combined checkpoint optimizer load failed ({load}); model rollback failed ({model})"
        ),
    }
}

/// Load a combined model + optimizer checkpoint, restoring the model if the
/// optimizer half fails.
///
/// `load_optimizer` is the only part either caller supplies. The model is
/// snapshotted before mutation; if the optimizer load then fails, only the
/// model snapshot is restored — the optimizer is deliberately *not* reloaded,
/// which preserves all its prior state, including entries for other models and
/// clocks that could no longer be loaded.
fn load_checkpoint<M, F>(
    model: &mut M,
    path: impl AsRef<Path>,
    options: &LoadOptions,
    op: &'static str,
    load_optimizer: F,
) -> Result<()>
where
    M: Module + ?Sized,
    F: FnOnce(&RuntimeModuleAdapter<'_, M>, &Envelope) -> Result<()>,
{
    let envelope = Envelope::load(path, &options.limits)?;
    validate_advancing_clocks(&envelope)?;
    stable_state(model, op)?;
    let model_before = nn::state_dict(model)?;
    super::persist::load_combined_model_state(model, &envelope, options)?;
    if let Err(load) = load_optimizer(&RuntimeModuleAdapter::new(model), &envelope) {
        return match nn::load_state_dict(model, &model_before) {
            Ok(()) => Err(load),
            Err(model) => Err(rollback_error(load, model)),
        };
    }
    Ok(())
}

/// Loads model and SGD state from one checkpoint with checked rollback.
///
/// The checkpoint must carry an `optimizer` section. The bare tensor path
/// `optim` is invalid; `optim.*` tensors are reserved for and validated by the
/// transactional runtime SGD loader after model staging.
///
/// Model state is snapshotted before mutation. If the runtime optimizer load
/// fails, only the model snapshot is restored — the optimizer is not reloaded,
/// which preserves all its prior state, including entries for other models.
/// Optimizer *step* backend failures retain their separate, potentially
/// partial semantics.
///
/// # Errors
///
/// [`Envelope::load`]'s reader-limit/format/filesystem errors;
/// [`Error::Persistence`] for an invalid or overflowing clock; otherwise
/// propagates the model-state loader's (as [`load_model_state`](crate::typed::persist::load_model_state)) and
/// [`Sgd::load_state`]'s errors. If the optimizer half fails after the model
/// was staged, the model is rolled back to its pre-load snapshot; a failure
/// during that rollback itself is reported as a combined
/// [`Error::Persistence`] naming both failures.
pub fn load_sgd_checkpoint<M: Module + ?Sized>(
    optimizer: &mut Sgd,
    model: &mut M,
    path: impl AsRef<Path>,
    options: &LoadOptions,
) -> Result<()> {
    load_checkpoint(
        model,
        path,
        options,
        "typed::optim::load_sgd_checkpoint",
        |adapter, envelope| optimizer.load_state(adapter, envelope),
    )
}

/// Loads model and Adam or `AdamW` state from one checkpoint with checked rollback.
///
/// The checkpoint must carry an `optimizer` section. The bare tensor path
/// `optim` is invalid; `optim.*` tensors are reserved for and validated by the
/// transactional runtime Adam loader after model staging.
///
/// # Errors
///
/// As [`load_sgd_checkpoint`], against [`Adam::load_state`].
pub fn load_adam_checkpoint<M: Module + ?Sized>(
    optimizer: &mut Adam,
    model: &mut M,
    path: impl AsRef<Path>,
    options: &LoadOptions,
) -> Result<()> {
    load_checkpoint(
        model,
        path,
        options,
        "typed::optim::load_adam_checkpoint",
        |adapter, envelope| optimizer.load_state(adapter, envelope),
    )
}

/// Saves typed model and SGD state through [`Envelope::save`].
///
/// # Errors
///
/// As [`crate::typed::persist::save_model_state`] and [`save_sgd_state`];
/// otherwise propagates [`Envelope::save`]'s filesystem error.
pub fn save_sgd_checkpoint<M: Module + ?Sized>(
    optimizer: &Sgd,
    model: &mut M,
    path: impl AsRef<Path>,
    limits: &Limits,
) -> Result<()> {
    let mut envelope = Envelope::new();
    super::persist::save_model_state(model, &mut envelope)?;
    save_sgd_state(optimizer, model, &mut envelope)?;
    envelope.save(path, limits)
}

/// Saves typed model and Adam or `AdamW` state through [`Envelope::save`].
///
/// # Errors
///
/// As [`save_sgd_checkpoint`], against [`save_adam_state`].
pub fn save_adam_checkpoint<M: Module + ?Sized>(
    optimizer: &Adam,
    model: &mut M,
    path: impl AsRef<Path>,
    limits: &Limits,
) -> Result<()> {
    let mut envelope = Envelope::new();
    super::persist::save_model_state(model, &mut envelope)?;
    save_adam_state(optimizer, model, &mut envelope)?;
    envelope.save(path, limits)
}

#[cfg(test)]
mod tests;
