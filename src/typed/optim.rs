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
//! Typed modules must obey [`Module`]'s CT40 stable-walk contract. The adapter
//! compares repeated typed read-only walks before delegation; an implementation
//! whose paths vary between calls violates that safe semantic contract and no
//! finite preflight can make its future walks predictable.

use super::nn::{self, Module, RuntimeModuleAdapter};
use crate::persist::{Envelope, Limits, LoadOptions};
use crate::{Error, Grads, Result};
use std::path::Path;

pub use crate::optim::{Adam, AdamGroup, AdamW, Sgd, SgdGroup};

fn preflight<M: Module + ?Sized>(model: &M, op: &'static str) -> Result<nn::TypedStateDict> {
    let first = nn::state_dict(model)?;
    let second = nn::state_dict(model)?;
    if first.paths().ne(second.paths()) {
        return Err(Error::InvalidArg {
            op,
            msg: "typed Module paths changed between preflight walks; Module requires stable repeated walks"
                .into(),
        });
    }
    Ok(first)
}

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
pub fn sgd_step<M: Module + ?Sized>(
    optimizer: &mut Sgd,
    model: &mut M,
    grads: Grads,
) -> Result<()> {
    preflight(model, "typed::optim::sgd_step")?;
    optimizer.step(&mut RuntimeModuleAdapter::new(model), grads)
}

/// Applies one Adam or AdamW update to a typed model, consuming `grads` by move.
///
/// [`AdamW::new`] returns the same [`Adam`] implementation with decoupled
/// weight decay, so this one adapter covers both names exactly.
pub fn adam_step<M: Module + ?Sized>(
    optimizer: &mut Adam,
    model: &mut M,
    grads: Grads,
) -> Result<()> {
    preflight(model, "typed::optim::adam_step")?;
    optimizer.step(&mut RuntimeModuleAdapter::new(model), grads)
}

/// Writes SGD state into an envelope using the typed model's exact paths.
pub fn save_sgd_state<M: Module + ?Sized>(
    optimizer: &Sgd,
    model: &mut M,
    envelope: &mut Envelope,
) -> Result<()> {
    preflight(model, "typed::optim::save_sgd_state")?;
    optimizer.save_state(&RuntimeModuleAdapter::new(model), envelope)
}

/// Stages and restores SGD state against the target typed model.
///
/// The runtime loader checks every untrusted clock and wide buffer before
/// replacing any optimizer state. The model itself is not modified.
pub fn load_sgd_state<M: Module + ?Sized>(
    optimizer: &mut Sgd,
    model: &mut M,
    envelope: &Envelope,
) -> Result<()> {
    preflight(model, "typed::optim::load_sgd_state")?;
    validate_advancing_clocks(envelope)?;
    optimizer.load_state(&RuntimeModuleAdapter::new(model), envelope)
}

/// Writes Adam or AdamW state into an envelope using exact typed-model paths.
pub fn save_adam_state<M: Module + ?Sized>(
    optimizer: &Adam,
    model: &mut M,
    envelope: &mut Envelope,
) -> Result<()> {
    preflight(model, "typed::optim::save_adam_state")?;
    optimizer.save_state(&RuntimeModuleAdapter::new(model), envelope)
}

/// Stages and restores Adam or AdamW state against the target typed model.
pub fn load_adam_state<M: Module + ?Sized>(
    optimizer: &mut Adam,
    model: &mut M,
    envelope: &Envelope,
) -> Result<()> {
    preflight(model, "typed::optim::load_adam_state")?;
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
fn param_clock<M, F>(model: &mut M, path: &str, clock: F) -> u64
where
    M: Module + ?Sized,
    F: Fn(&crate::nn::Param) -> u64,
{
    let adapter = RuntimeModuleAdapter::new(model);
    let mut found = 0;
    crate::nn::visit::visit_all(&adapter, &mut |leaf_path, leaf| {
        if leaf_path == path
            && let crate::nn::visit::Leaf::Param(param) = leaf
        {
            found = clock(param);
        }
    });
    found
}

/// Returns SGD's saved update clock for the typed parameter at `path`.
///
/// As with runtime `Sgd::param_steps`, an unseen parameter reports zero. Typed
/// callers identify parameters by the same stable dotted paths used by groups
/// and checkpoints; malformed or path-varying module walks remain errors.
pub fn sgd_param_steps<M: Module + ?Sized>(
    optimizer: &Sgd,
    model: &mut M,
    path: &str,
) -> Result<u64> {
    let state = preflight(model, "typed::optim::sgd_param_steps")?;
    if !state.is_param_path(path) {
        return Err(Error::InvalidArg {
            op: "typed::optim::sgd_param_steps",
            msg: format!("typed model has no parameter path {path:?}"),
        });
    }
    Ok(param_clock(model, path, |param| {
        optimizer.param_steps(param)
    }))
}

/// Returns Adam or AdamW's saved bias-correction clock at `path`.
pub fn adam_param_steps<M: Module + ?Sized>(
    optimizer: &Adam,
    model: &mut M,
    path: &str,
) -> Result<u64> {
    let state = preflight(model, "typed::optim::adam_param_steps")?;
    if !state.is_param_path(path) {
        return Err(Error::InvalidArg {
            op: "typed::optim::adam_param_steps",
            msg: format!("typed model has no parameter path {path:?}"),
        });
    }
    Ok(param_clock(model, path, |param| {
        optimizer.param_steps(param)
    }))
}

fn rollback_error(load: Error, model: Error) -> Error {
    Error::Persistence {
        msg: format!(
            "combined checkpoint optimizer load failed ({load}); model rollback failed ({model})"
        ),
    }
}

/// Loads model and SGD state from one checkpoint with checked rollback.
///
/// The checkpoint must carry an `optimizer` section. The bare tensor path
/// `optim` is invalid; `optim.*` tensors are reserved for and validated by the
/// transactional runtime SGD loader after model staging.
///
/// Model state is snapshotted before mutation. If transactional runtime
/// optimizer loading fails, only the typed model snapshot is restored; the
/// optimizer is not reloaded, preserving all prior state including entries for
/// other models and clocks that could no longer be loaded. Optimizer *step*
/// backend failures retain their separate, potentially partial semantics.
pub fn load_sgd_checkpoint<M: Module + ?Sized>(
    optimizer: &mut Sgd,
    model: &mut M,
    path: impl AsRef<Path>,
    options: &LoadOptions,
) -> Result<()> {
    let envelope = Envelope::load(path, &options.limits)?;
    validate_advancing_clocks(&envelope)?;
    preflight(model, "typed::optim::load_sgd_checkpoint")?;
    let model_before = nn::state_dict(model)?;
    super::persist::load_combined_model_state(model, &envelope, options)?;
    if let Err(load) = optimizer.load_state(&RuntimeModuleAdapter::new(model), &envelope) {
        return match nn::load_state_dict(model, &model_before) {
            Ok(()) => Err(load),
            Err(model) => Err(rollback_error(load, model)),
        };
    }
    Ok(())
}

/// Loads model and Adam or AdamW state from one checkpoint with checked rollback.
///
/// The checkpoint must carry an `optimizer` section. The bare tensor path
/// `optim` is invalid; `optim.*` tensors are reserved for and validated by the
/// transactional runtime Adam loader after model staging.
pub fn load_adam_checkpoint<M: Module + ?Sized>(
    optimizer: &mut Adam,
    model: &mut M,
    path: impl AsRef<Path>,
    options: &LoadOptions,
) -> Result<()> {
    let envelope = Envelope::load(path, &options.limits)?;
    validate_advancing_clocks(&envelope)?;
    preflight(model, "typed::optim::load_adam_checkpoint")?;
    let model_before = nn::state_dict(model)?;
    super::persist::load_combined_model_state(model, &envelope, options)?;
    if let Err(load) = optimizer.load_state(&RuntimeModuleAdapter::new(model), &envelope) {
        return match nn::load_state_dict(model, &model_before) {
            Ok(()) => Err(load),
            Err(model) => Err(rollback_error(load, model)),
        };
    }
    Ok(())
}

/// Saves typed model and SGD state through [`Envelope::save`].
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

/// Saves typed model and Adam or AdamW state through [`Envelope::save`].
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
mod tests {
    use super::*;
    use crate::typed::nn::{Mode, TypedBuffer, TypedParam, TypedVisitor, TypedVisitorMut};
    use crate::typed::{Cpu, DeviceCtx, FloatElement, NumericElement, Tensor1};
    use crate::{Error, Tensor};
    use std::cell::Cell;

    struct Pair {
        first: TypedParam<Tensor1<1>>,
        second: TypedParam<Tensor1<1>>,
    }

    impl Module for Pair {
        fn visit(&self, visitor: &mut TypedVisitor<'_>) {
            visitor.param("first", &self.first);
            visitor.param("second", &self.second);
        }

        fn visit_mut(&mut self, visitor: &mut TypedVisitorMut<'_>) {
            visitor.param("first", &mut self.first);
            visitor.param("second", &mut self.second);
        }
    }

    fn pair() -> Pair {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        Pair {
            first: TypedParam::new(Tensor1::from_vec(vec![1.0], [1], &ctx).unwrap()).unwrap(),
            second: TypedParam::new(Tensor1::from_vec(vec![1.0], [1], &ctx).unwrap()).unwrap(),
        }
    }

    fn loss(model: &Pair, include_second: bool) -> Tensor {
        let first = model.first.get(Mode::TRAIN).unwrap();
        let mut loss = first.as_dynamic().sum_all().unwrap();
        if include_second {
            loss = loss
                .add(
                    &model
                        .second
                        .get(Mode::TRAIN)
                        .unwrap()
                        .as_dynamic()
                        .sum_all()
                        .unwrap(),
                )
                .unwrap();
        }
        loss
    }

    fn values(model: &Pair) -> [f32; 2] {
        [
            model.first.value().unwrap().to_vec().unwrap()[0],
            model.second.value().unwrap().to_vec().unwrap()[0],
        ]
    }

    #[test]
    fn sgd_groups_frozen_parameters_and_missing_grads_match_runtime_behavior() {
        let mut model = pair();
        let mut runtime_model = pair();
        let mut optimizer =
            Sgd::new(0.1).group(|path| path == "second", |group| group.lr_scale(2.0));
        let mut runtime_optimizer =
            Sgd::new(0.1).group(|path| path == "second", |group| group.lr_scale(2.0));
        let grads = loss(&model, true).backward().unwrap();
        sgd_step(&mut optimizer, &mut model, grads).unwrap();
        let runtime_grads = loss(&runtime_model, true).backward().unwrap();
        runtime_optimizer
            .step(
                &mut RuntimeModuleAdapter::new(&mut runtime_model),
                runtime_grads,
            )
            .unwrap();
        assert_eq!(values(&model), [0.9, 0.8]);
        assert_eq!(values(&model), values(&runtime_model));
        assert_eq!(sgd_param_steps(&optimizer, &mut model, "first").unwrap(), 1);
        assert_eq!(
            sgd_param_steps(&optimizer, &mut model, "second").unwrap(),
            1
        );

        let mut typed_state = Envelope::new();
        save_sgd_state(&optimizer, &mut model, &mut typed_state).unwrap();
        let mut runtime_state = Envelope::new();
        runtime_optimizer
            .save_state(
                &RuntimeModuleAdapter::new(&mut runtime_model),
                &mut runtime_state,
            )
            .unwrap();
        assert_eq!(typed_state, runtime_state);

        model.second.freeze();
        let grads = loss(&model, false).backward().unwrap();
        sgd_step(&mut optimizer, &mut model, grads).unwrap();
        assert_eq!(values(&model), [0.79999995, 0.8]);

        model.second.unfreeze();
        let before = values(&model);
        let grads = loss(&model, false).backward().unwrap();
        let error = sgd_step(&mut optimizer, &mut model, grads).unwrap_err();
        assert!(matches!(error, Error::MissingGrad { path } if path == "second"));
        assert_eq!(values(&model), before);
    }

    fn resumed_adam(mut optimizer: Adam) {
        use crate::persist::{Limits, LoadOptions};

        let mut reference_model = pair();
        let mut reference = if optimizer.is_decoupled() {
            AdamW::new(0.05, 0.02)
        } else {
            Adam::new(0.05).weight_decay(0.02)
        };
        let mut staged_model = pair();
        for _ in 0..4 {
            let grads = loss(&reference_model, true).backward().unwrap();
            reference
                .step(&mut RuntimeModuleAdapter::new(&mut reference_model), grads)
                .unwrap();
        }
        for _ in 0..2 {
            let grads = loss(&staged_model, true).backward().unwrap();
            adam_step(&mut optimizer, &mut staged_model, grads).unwrap();
        }
        let mut envelope = Envelope::new();
        save_adam_state(&optimizer, &mut staged_model, &mut envelope).unwrap();
        let mut runtime_envelope = Envelope::new();
        optimizer
            .save_state(
                &RuntimeModuleAdapter::new(&mut staged_model),
                &mut runtime_envelope,
            )
            .unwrap();
        assert_eq!(envelope, runtime_envelope);
        assert_eq!(envelope.tensors().len(), 4);
        assert!(
            envelope
                .tensors()
                .values()
                .all(|payload| payload.dtype() == crate::DType::F32)
        );
        let section = envelope.section("optimizer").unwrap();
        assert!(section.contains("clock.first=2"));
        assert!(section.contains("clock.second=2"));

        let mut checkpoint = Envelope::new();
        crate::typed::persist::save_model_state(&mut staged_model, &mut checkpoint).unwrap();
        save_adam_state(&optimizer, &mut staged_model, &mut checkpoint).unwrap();
        checkpoint.set_section("rng", "state=17").unwrap();
        let path = std::env::temp_dir().join(format!(
            "rstorch-typed-optim-{}-{}.rstorch",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        checkpoint.save(&path, &Limits::defaults()).unwrap();
        let loaded = Envelope::load(&path, &Limits::defaults()).unwrap();
        assert_eq!(loaded, checkpoint);
        assert_eq!(loaded.section("rng"), Some("state=17"));

        let mut reconstructed = pair();
        let mut disk_resumed = if optimizer.is_decoupled() {
            AdamW::new(999.0, 0.0)
        } else {
            Adam::new(999.0)
        };
        load_adam_checkpoint(
            &mut disk_resumed,
            &mut reconstructed,
            &path,
            &LoadOptions::strict(),
        )
        .unwrap();

        let mut resumed = if optimizer.is_decoupled() {
            AdamW::new(999.0, 0.0)
        } else {
            Adam::new(999.0)
        };
        load_adam_state(&mut resumed, &mut staged_model, &envelope).unwrap();
        assert_eq!(
            adam_param_steps(&resumed, &mut staged_model, "first").unwrap(),
            2
        );
        for _ in 0..2 {
            let grads = loss(&staged_model, true).backward().unwrap();
            adam_step(&mut resumed, &mut staged_model, grads).unwrap();
        }
        assert_eq!(values(&staged_model), values(&reference_model));
        for _ in 0..2 {
            let grads = loss(&reconstructed, true).backward().unwrap();
            adam_step(&mut disk_resumed, &mut reconstructed, grads).unwrap();
        }
        assert_eq!(values(&reconstructed), values(&reference_model));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn adam_and_adamw_resume_with_wide_moments_and_per_parameter_clocks() {
        resumed_adam(Adam::new(0.05).weight_decay(0.02));
        resumed_adam(AdamW::new(0.05, 0.02));
    }

    #[test]
    fn sgd_momentum_load_resumes_values_and_clocks() {
        let mut reference_model = pair();
        let mut reference = Sgd::new(0.05).momentum(0.8);
        for _ in 0..4 {
            let grads = loss(&reference_model, true).backward().unwrap();
            sgd_step(&mut reference, &mut reference_model, grads).unwrap();
        }

        let mut staged_model = pair();
        let mut staged = Sgd::new(0.05).momentum(0.8);
        for _ in 0..2 {
            let grads = loss(&staged_model, true).backward().unwrap();
            sgd_step(&mut staged, &mut staged_model, grads).unwrap();
        }
        let path = std::env::temp_dir().join(format!(
            "rstorch-typed-sgd-{}-{}.rstorch",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        save_sgd_checkpoint(&staged, &mut staged_model, &path, &Limits::defaults()).unwrap();
        let mut reconstructed = pair();
        let mut resumed = Sgd::new(999.0);
        load_sgd_checkpoint(
            &mut resumed,
            &mut reconstructed,
            &path,
            &LoadOptions::strict(),
        )
        .unwrap();
        assert_eq!(
            sgd_param_steps(&resumed, &mut reconstructed, "first").unwrap(),
            2
        );
        for _ in 0..2 {
            let grads = loss(&reconstructed, true).backward().unwrap();
            sgd_step(&mut resumed, &mut reconstructed, grads).unwrap();
        }
        assert_eq!(values(&reconstructed), values(&reference_model));
        std::fs::remove_file(path).unwrap();
    }

    struct Reduced<E: FloatElement + NumericElement> {
        weight: TypedParam<Tensor1<1, E>>,
    }

    impl<E: FloatElement + NumericElement> Module for Reduced<E> {
        fn visit(&self, visitor: &mut TypedVisitor<'_>) {
            visitor.param("weight", &self.weight);
        }

        fn visit_mut(&mut self, visitor: &mut TypedVisitorMut<'_>) {
            visitor.param("weight", &mut self.weight);
        }
    }

    fn reduced<E: FloatElement + NumericElement>(value: E) -> Reduced<E> {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        Reduced {
            weight: TypedParam::new(Tensor1::from_vec(vec![value], [1], &ctx).unwrap()).unwrap(),
        }
    }

    fn reduced_loss<E: FloatElement + NumericElement>(model: &Reduced<E>) -> Tensor {
        model
            .weight
            .get(Mode::TRAIN)
            .unwrap()
            .as_dynamic()
            .sum_all()
            .unwrap()
    }

    fn reduced_precision_case<E: FloatElement + NumericElement + Copy>(one: E) {
        let mut reference = reduced(one);
        let mut reference_optimizer = Adam::new(0.02);
        for _ in 0..4 {
            let grads = reduced_loss(&reference).backward().unwrap();
            adam_step(&mut reference_optimizer, &mut reference, grads).unwrap();
        }

        let mut staged = reduced(one);
        let mut optimizer = Adam::new(0.02);
        for _ in 0..2 {
            let grads = reduced_loss(&staged).backward().unwrap();
            adam_step(&mut optimizer, &mut staged, grads).unwrap();
        }
        let mut envelope = Envelope::new();
        save_adam_state(&optimizer, &mut staged, &mut envelope).unwrap();
        assert!(
            envelope
                .tensors()
                .values()
                .all(|payload| payload.dtype() == crate::DType::F32)
        );
        let state = nn::state_dict(&staged).unwrap();
        let mut reconstructed = reduced(one);
        nn::load_state_dict(&mut reconstructed, &state).unwrap();
        let mut resumed = Adam::new(999.0);
        load_adam_state(&mut resumed, &mut reconstructed, &envelope).unwrap();
        for _ in 0..2 {
            let grads = reduced_loss(&reconstructed).backward().unwrap();
            adam_step(&mut resumed, &mut reconstructed, grads).unwrap();
        }
        assert_eq!(
            reconstructed.weight.value().unwrap().item().unwrap(),
            reference.weight.value().unwrap().item().unwrap()
        );

        let mut sgd_model = reduced(one);
        let mut sgd = Sgd::new(0.02).momentum(0.8);
        let grads = reduced_loss(&sgd_model).backward().unwrap();
        sgd_step(&mut sgd, &mut sgd_model, grads).unwrap();
        let mut sgd_state = Envelope::new();
        save_sgd_state(&sgd, &mut sgd_model, &mut sgd_state).unwrap();
        assert!(
            sgd_state
                .tensors()
                .values()
                .all(|payload| payload.dtype() == crate::DType::F32)
        );
    }

    #[test]
    fn f16_and_bf16_optimizer_moments_are_wide_and_resume() {
        reduced_precision_case(half::f16::from_f32(1.0));
        reduced_precision_case(half::bf16::from_f32(1.0));
    }

    #[test]
    fn max_clocks_are_rejected_before_optimizer_mutation() {
        let mut model = pair();
        let mut optimizer = Sgd::new(0.1);
        let grads = loss(&model, true).backward().unwrap();
        sgd_step(&mut optimizer, &mut model, grads).unwrap();
        let before = sgd_param_steps(&optimizer, &mut model, "first").unwrap();
        let mut envelope = Envelope::new();
        save_sgd_state(&optimizer, &mut model, &mut envelope).unwrap();
        let section = envelope
            .section("optimizer")
            .unwrap()
            .replace("steps=1", &format!("steps={}", u64::MAX));
        envelope.set_section("optimizer", section).unwrap();
        assert!(load_sgd_state(&mut optimizer, &mut model, &envelope).is_err());
        assert_eq!(
            sgd_param_steps(&optimizer, &mut model, "first").unwrap(),
            before
        );
    }

    /// One optimizer's combined-checkpoint rollback contract, exercised through
    /// the public typed entry points so both `load_*_checkpoint` bodies are
    /// covered rather than only Adam's.
    fn combined_rollback_case<O>(
        source_optimizer: O,
        target_optimizer: O,
        step: fn(&mut O, &mut Pair, Grads) -> Result<()>,
        save: fn(&O, &mut Pair, &mut Envelope) -> Result<()>,
        load: impl Fn(&mut O, &mut Pair, &Path, &LoadOptions) -> Result<()>,
    ) {
        let mut source_optimizer = source_optimizer;
        let mut source = pair();
        for _ in 0..2 {
            let grads = loss(&source, true).backward().unwrap();
            step(&mut source_optimizer, &mut source, grads).unwrap();
        }
        let mut checkpoint = Envelope::new();
        crate::typed::persist::save_model_state(&mut source, &mut checkpoint).unwrap();
        save(&source_optimizer, &mut source, &mut checkpoint).unwrap();
        let malformed = format!(
            "{}unknown.field=1\n",
            checkpoint.section("optimizer").unwrap()
        );
        checkpoint.set_section("optimizer", malformed).unwrap();
        let path = std::env::temp_dir().join(format!(
            "rstorch-typed-rollback-{}-{}.rstorch",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        checkpoint.save(&path, &Limits::defaults()).unwrap();

        let mut target_optimizer = target_optimizer;
        let mut target = pair();
        let mut unrelated = pair();
        let grads = loss(&unrelated, true).backward().unwrap();
        step(&mut target_optimizer, &mut unrelated, grads).unwrap();
        let model_before = values(&target);
        let mut optimizer_before = Envelope::new();
        save(&target_optimizer, &mut unrelated, &mut optimizer_before).unwrap();
        // The checkpoint's model half differs from the target, so a rollback
        // that silently did nothing would leave `values(&target)` changed.
        assert_ne!(model_before, values(&source));

        let error = load(
            &mut target_optimizer,
            &mut target,
            &path,
            &LoadOptions::strict(),
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "persistence: unknown optimizer state key `unknown.field`"
        );
        assert_eq!(values(&target), model_before);
        let mut optimizer_after = Envelope::new();
        save(&target_optimizer, &mut unrelated, &mut optimizer_after).unwrap();
        assert_eq!(optimizer_after, optimizer_before);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn malformed_optimizer_rolls_back_a_valid_combined_model_load() {
        combined_rollback_case(
            Adam::new(0.03),
            Adam::new(0.07),
            adam_step,
            save_adam_state,
            |optimizer, model, path, options| load_adam_checkpoint(optimizer, model, path, options),
        );
        combined_rollback_case(
            Sgd::new(0.03).momentum(0.8),
            Sgd::new(0.07).momentum(0.8),
            sgd_step,
            save_sgd_state,
            |optimizer, model, path, options| load_sgd_checkpoint(optimizer, model, path, options),
        );
    }

    /// `rollback_error` is only rendered when a rollback *also* fails, which no
    /// reachable input produces today, so its wording is pinned directly rather
    /// than left as the one message in this module that nothing ever formats.
    #[test]
    fn a_failed_rollback_reports_both_causes() {
        let rendered = rollback_error(
            Error::Persistence {
                msg: "optimizer half broke".into(),
            },
            Error::Persistence {
                msg: "model half broke".into(),
            },
        )
        .to_string();
        assert_eq!(
            rendered,
            "persistence: combined checkpoint optimizer load failed \
             (persistence: optimizer half broke); model rollback failed \
             (persistence: model half broke)"
        );
    }

    #[test]
    fn clock_inspection_rejects_buffer_paths() {
        struct WithBuffer {
            weight: TypedParam<Tensor1<1>>,
            running: TypedBuffer<Tensor1<1>>,
        }
        impl Module for WithBuffer {
            fn visit(&self, visitor: &mut TypedVisitor<'_>) {
                visitor.param("weight", &self.weight);
                visitor.buffer("running", &self.running);
            }
            fn visit_mut(&mut self, visitor: &mut TypedVisitorMut<'_>) {
                visitor.param("weight", &mut self.weight);
                visitor.buffer("running", &mut self.running);
            }
        }
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let mut model = WithBuffer {
            weight: TypedParam::new(Tensor1::from_vec(vec![1.0], [1], &ctx).unwrap()).unwrap(),
            running: TypedBuffer::new(Tensor1::from_vec(vec![0.0], [1], &ctx).unwrap()).unwrap(),
        };
        assert!(sgd_param_steps(&Sgd::new(0.1), &mut model, "running").is_err());
        assert!(adam_param_steps(&Adam::new(0.1), &mut model, "running").is_err());
    }

    /// Both `*_param_steps` document that "an unseen parameter reports zero".
    /// A parameter frozen before the first step is the reachable case: the
    /// optimizer never creates state for it, so no `clock.<path>` line is
    /// written and the lookup must fall back to zero — while its updated sibling
    /// reports one, so a fallback that returned any other value would show up.
    #[test]
    fn a_parameter_the_optimizer_never_updated_reports_a_zero_clock() {
        let mut sgd_model = pair();
        sgd_model.second.freeze();
        let mut sgd = Sgd::new(0.1);
        let grads = loss(&sgd_model, false).backward().unwrap();
        sgd_step(&mut sgd, &mut sgd_model, grads).unwrap();
        assert_eq!(sgd_param_steps(&sgd, &mut sgd_model, "first").unwrap(), 1);
        assert_eq!(sgd_param_steps(&sgd, &mut sgd_model, "second").unwrap(), 0);

        let mut adam_model = pair();
        adam_model.second.freeze();
        let mut adam = Adam::new(0.1);
        let grads = loss(&adam_model, false).backward().unwrap();
        adam_step(&mut adam, &mut adam_model, grads).unwrap();
        assert_eq!(
            adam_param_steps(&adam, &mut adam_model, "first").unwrap(),
            1
        );
        assert_eq!(
            adam_param_steps(&adam, &mut adam_model, "second").unwrap(),
            0
        );
    }

    fn save_dynamic_model(model: &mut Pair, envelope: &mut Envelope) {
        let adapter = RuntimeModuleAdapter::new(model);
        for (path, value) in crate::nn::state_dict(&adapter) {
            envelope.insert_tensor(path, crate::checkpoint::to_host_tensor(&value).unwrap());
        }
    }

    #[test]
    fn typed_and_dynamic_sgd_and_adam_checkpoint_files_are_byte_identical() {
        let dir = std::env::temp_dir().join(format!(
            "rstorch-typed-byte-parity-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        let mut sgd_model = pair();
        let mut sgd = Sgd::new(0.1).momentum(0.8);
        let grads = loss(&sgd_model, true).backward().unwrap();
        sgd_step(&mut sgd, &mut sgd_model, grads).unwrap();
        let typed_sgd = dir.join("typed-sgd.rstorch");
        let dynamic_sgd = dir.join("dynamic-sgd.rstorch");
        save_sgd_checkpoint(&sgd, &mut sgd_model, &typed_sgd, &Limits::defaults()).unwrap();
        let mut dynamic = Envelope::new();
        save_dynamic_model(&mut sgd_model, &mut dynamic);
        sgd.save_state(&RuntimeModuleAdapter::new(&mut sgd_model), &mut dynamic)
            .unwrap();
        dynamic.save(&dynamic_sgd, &Limits::defaults()).unwrap();
        assert_eq!(
            std::fs::read(&typed_sgd).unwrap(),
            std::fs::read(&dynamic_sgd).unwrap()
        );

        let mut adam_model = pair();
        let mut adam = Adam::new(0.1);
        let grads = loss(&adam_model, true).backward().unwrap();
        adam_step(&mut adam, &mut adam_model, grads).unwrap();
        let typed_adam = dir.join("typed-adam.rstorch");
        let dynamic_adam = dir.join("dynamic-adam.rstorch");
        save_adam_checkpoint(&adam, &mut adam_model, &typed_adam, &Limits::defaults()).unwrap();
        let mut dynamic = Envelope::new();
        save_dynamic_model(&mut adam_model, &mut dynamic);
        adam.save_state(&RuntimeModuleAdapter::new(&mut adam_model), &mut dynamic)
            .unwrap();
        dynamic.save(&dynamic_adam, &Limits::defaults()).unwrap();
        assert_eq!(
            std::fs::read(&typed_adam).unwrap(),
            std::fs::read(&dynamic_adam).unwrap()
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn path_varying_modules_are_rejected_by_preflight() {
        struct Varying {
            calls: Cell<usize>,
            value: TypedParam<Tensor1<1>>,
        }
        impl Module for Varying {
            fn visit(&self, visitor: &mut TypedVisitor<'_>) {
                let call = self.calls.get();
                self.calls.set(call + 1);
                visitor.param(if call.is_multiple_of(2) { "a" } else { "b" }, &self.value);
            }
            fn visit_mut(&mut self, visitor: &mut TypedVisitorMut<'_>) {
                visitor.param("a", &mut self.value);
            }
        }
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let mut model = Varying {
            calls: Cell::new(0),
            value: TypedParam::new(Tensor1::from_vec(vec![1.0], [1], &ctx).unwrap()).unwrap(),
        };
        let error = sgd_param_steps(&Sgd::new(0.1), &mut model, "a").unwrap_err();
        assert!(error.to_string().contains("stable repeated walks"));
    }
}
