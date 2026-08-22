//! Persistence adapters for typed model state.
//!
//! In-memory state retains the full typed contracts and canonical binding
//! identity through [`TypedStateDict`]. On-disk tensors carry no such authority:
//! loads derive their schema and destination device from the target typed model,
//! stage all untrusted host tensors first, and only then delegate the existing
//! transactional runtime state loader through the private short-lived adapter.
//!
//! File parsing, writer limits, metadata decoding, and temp-file replacement are
//! delegated to [`crate::persist`]. This adapter adds no claims about duplicate
//! raw JSON/metadata keys, writer-limit symmetry, symlink handling, metadata
//! byte ordering, or **tensor payload integrity** beyond that runtime
//! persistence contract.
//!
//! Payload integrity is worth stating explicitly, because "transactionally
//! loads" and "versioned checkpoint" invite the opposite reading: the container
//! carries no checksum, so a corrupted data region loads as valid values. This
//! was verified by flipping one bit in the last byte of a checkpoint written by
//! [`crate::typed::optim::save_sgd_checkpoint`] and observing `Ok(())` from
//! [`crate::typed::optim::load_sgd_checkpoint`] with a silently different model.
//! *Structural* damage is caught: truncation, a wrong dtype on a model tensor or
//! on an `optim.*` buffer, a duplicated optimizer key, and a missing `optimizer`
//! section all fail the load and leave both halves untouched.

use super::nn::{self, Module, RuntimeModuleAdapter, TypedStateDict, stable_state};
use crate::nn::ModuleExt as _;
use crate::persist::{Envelope, Expected, Limits, LoadOptions};
use crate::{Error, Result};
use std::collections::BTreeMap;
use std::path::Path;

/// Collects an opaque in-memory state dictionary with exact typed metadata.
///
/// # Errors
///
/// [`Error::InvalidArg`] if `model`'s walk is malformed, or if any path
/// collides with the reserved `optim`/`optim.*` namespace.
pub fn state_dict<M: Module + ?Sized>(model: &M) -> Result<TypedStateDict> {
    let state = stable_state(model, "typed::persist::state_dict")?;
    reject_reserved_paths(state.paths(), "typed::persist::state_dict")?;
    Ok(state)
}

/// Transactionally loads an in-memory typed state dictionary.
///
/// Dtype/device changes require a reconstructed or consuming-retyped target;
/// this function never changes markers in place.
///
/// # Errors
///
/// [`Error::InvalidArg`] if `model`'s or `state`'s paths collide with the
/// reserved `optim`/`optim.*` namespace; otherwise propagates
/// [`nn::load_state_dict`]'s own contract/dimension/binding errors, which
/// leave `model` untouched.
pub fn load_state_dict<M: Module + ?Sized>(model: &mut M, state: &TypedStateDict) -> Result<()> {
    let target = stable_state(model, "typed::persist::load_state_dict")?;
    reject_reserved_paths(target.paths(), "typed::persist::load_state_dict")?;
    reject_reserved_paths(state.paths(), "typed::persist::load_state_dict")?;
    nn::load_state_dict(model, state)
}

fn reserved(path: &str) -> bool {
    path == "optim" || path.starts_with("optim.")
}

fn reject_reserved_paths<'a>(paths: impl Iterator<Item = &'a str>, op: &'static str) -> Result<()> {
    if let Some(path) = paths.into_iter().find(|path| reserved(path)) {
        return Err(Error::InvalidArg {
            op,
            msg: format!(
                "model path {path:?} conflicts with the reserved optimizer checkpoint namespace `optim.*`"
            ),
        });
    }
    Ok(())
}

/// Adds every model parameter and persistent buffer to `envelope`.
///
/// Existing sections and optimizer tensors are preserved. A model key already
/// present in the envelope is replaced, matching [`Envelope::insert_tensor`].
///
/// # Errors
///
/// [`Error::InvalidArg`] if a path collides with the reserved
/// `optim`/`optim.*` namespace; otherwise propagates the host-transfer
/// error for any tensor.
pub fn save_model_state<M: Module + ?Sized>(model: &mut M, envelope: &mut Envelope) -> Result<()> {
    let state = stable_state(model, "typed::persist::save_model_state")?;
    reject_reserved_paths(state.paths(), "typed::persist::save_model_state")?;
    let adapter = RuntimeModuleAdapter::new(model);
    let staged = adapter
        .state_dict()
        .into_iter()
        .map(|(path, tensor)| Ok((path, crate::checkpoint::to_host_tensor(&tensor)?)))
        .collect::<Result<Vec<_>>>()?;
    for (path, tensor) in staged {
        envelope.insert_tensor(path, tensor);
    }
    Ok(())
}

/// Stages envelope model tensors against `model`, then transactionally loads.
///
/// This is deliberately a model-only loader: an `optimizer` section, the bare
/// tensor path `optim`, and every `optim.*` tensor are rejected even when
/// `options` allows unexpected model paths. Combined files must use
/// [`crate::typed::optim::load_sgd_checkpoint`] or
/// [`crate::typed::optim::load_adam_checkpoint`], which validate and load both
/// halves together. Missing and other unexpected model paths obey `options`;
/// shape and dtype mismatches are always errors.
///
/// # Errors
///
/// [`Error::Persistence`] if `envelope` carries an `optimizer` section or any
/// `optim`/`optim.*` tensor; [`Error::InvalidArg`] if a model path collides
/// with that reserved namespace; otherwise propagates
/// [`crate::persist::stage`]'s missing/unexpected/shape/dtype errors and the
/// host-tensor conversion error, both under `options`.
pub fn load_model_state<M: Module + ?Sized>(
    model: &mut M,
    envelope: &Envelope,
    options: &LoadOptions,
) -> Result<()> {
    if envelope.section("optimizer").is_some() {
        return Err(Error::persistence(
            "model-only load rejects an `optimizer` section; use a combined typed optimizer checkpoint loader",
        ));
    }
    if let Some(path) = envelope.tensors().keys().find(|path| reserved(path)) {
        return Err(Error::persistence(format!(
            "model-only load rejects reserved optimizer tensor {path:?}; use a combined typed optimizer checkpoint loader"
        )));
    }
    load_model_state_impl(model, envelope, options, false)
}

pub(in crate::typed) fn load_combined_model_state<M: Module + ?Sized>(
    model: &mut M,
    envelope: &Envelope,
    options: &LoadOptions,
) -> Result<()> {
    if envelope.section("optimizer").is_none() {
        return Err(Error::persistence(
            "combined checkpoint has no `optimizer` section",
        ));
    }
    if envelope.tensor("optim").is_some() {
        return Err(Error::persistence(
            "combined checkpoint contains bare reserved tensor `optim`; optimizer tensors require `optim.<path>.<buffer>`",
        ));
    }
    load_model_state_impl(model, envelope, options, true)
}

fn load_model_state_impl<M: Module + ?Sized>(
    model: &mut M,
    envelope: &Envelope,
    options: &LoadOptions,
    combined: bool,
) -> Result<()> {
    let state = stable_state(model, "typed::persist::load_model_state")?;
    reject_reserved_paths(state.paths(), "typed::persist::load_model_state")?;
    let current = {
        let adapter = RuntimeModuleAdapter::new(model);
        adapter.state_dict()
    };
    let schema = current
        .iter()
        .map(|(path, tensor)| Expected::new(path, tensor.dtype(), tensor.dims().to_vec()))
        .collect::<Vec<_>>();
    let model_tensors = envelope
        .tensors()
        .iter()
        .filter(|(path, _)| !(combined && path.starts_with("optim.")))
        .map(|(path, tensor)| (path.clone(), tensor.clone()))
        .collect::<BTreeMap<_, _>>();
    let staged = crate::persist::stage(&schema, &model_tensors, options)?;

    let mut replacements = current;
    for (path, host) in staged.into_entries() {
        let device = replacements[&path].device();
        replacements.insert(path, crate::checkpoint::from_host_tensor(&host, &device)?);
    }
    RuntimeModuleAdapter::new(model).load_state_dict(&replacements)
}

/// Saves only typed model state through [`Envelope::save`].
///
/// To include optimizer or application sections, use [`save_model_state`], add
/// them to the same [`Envelope`], and call [`Envelope::save`].
///
/// # Errors
///
/// As [`save_model_state`]; otherwise propagates [`Envelope::save`]'s
/// filesystem error.
///
/// # Examples
///
/// ```
/// use rstorch::Rng;
/// use rstorch::persist::{Limits, LoadOptions};
/// use rstorch::typed::DeviceCtx;
/// use rstorch::typed::nn::Linear;
/// use rstorch::typed::persist::{load_checkpoint, save_checkpoint};
///
/// # fn main() -> rstorch::Result<()> {
/// let ctx = DeviceCtx::cpu()?;
/// let mut model = Linear::<3, 2>::new(3, 2, &ctx, &mut Rng::seed(0))?;
///
/// let path = std::env::temp_dir().join(format!(
///     "rstorch-doctest-typed-checkpoint-{}.safetensors",
///     std::process::id()
/// ));
/// save_checkpoint(&mut model, &path, &Limits::default())?;
///
/// let mut restored = Linear::<3, 2>::new(3, 2, &ctx, &mut Rng::seed(1))?;
/// load_checkpoint(&mut restored, &path, &LoadOptions::default())?;
/// assert_eq!(restored.weight().value()?.dims(), model.weight().value()?.dims());
/// # let _ = std::fs::remove_file(&path);
/// # Ok(())
/// # }
/// ```
pub fn save_checkpoint<M: Module + ?Sized>(
    model: &mut M,
    path: impl AsRef<Path>,
    limits: &Limits,
) -> Result<()> {
    let mut envelope = Envelope::new();
    save_model_state(model, &mut envelope)?;
    envelope.save(path, limits)
}

/// Loads a versioned checkpoint and stages its model state into `model`.
///
/// # Errors
///
/// [`Envelope::load`]'s reader-limit, format, and filesystem errors; otherwise
/// as [`load_model_state`].
pub fn load_checkpoint<M: Module + ?Sized>(
    model: &mut M,
    path: impl AsRef<Path>,
    options: &LoadOptions,
) -> Result<()> {
    let envelope = Envelope::load(path, &options.limits)?;
    load_model_state(model, &envelope, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed::nn::{TypedBuffer, TypedModule, TypedParam, TypedVisitor, TypedVisitorMut};
    use crate::typed::{Cpu, DeviceCtx, Placement, Tensor1};
    use crate::{DType, Device};

    #[derive(TypedModule)]
    struct Model<P: Placement = Cpu> {
        weight: TypedParam<Tensor1<2, f32, P>>,
        running: TypedBuffer<Tensor1<1, f32, P>>,
    }

    fn model<P: Placement>(base: f32, ctx: &DeviceCtx<P>) -> Model<P> {
        Model {
            weight: TypedParam::new(Tensor1::from_vec(vec![base, base + 1.0], [2], ctx).unwrap())
                .unwrap(),
            running: TypedBuffer::new(Tensor1::from_vec(vec![base + 2.0], [1], ctx).unwrap())
                .unwrap(),
        }
    }

    fn values<P: Placement>(model: &Model<P>) -> (Vec<f32>, Vec<f32>) {
        (
            model.weight.value().unwrap().to_vec().unwrap(),
            model.running.value().unwrap().to_vec().unwrap(),
        )
    }

    #[test]
    fn in_memory_state_retains_logical_binding_identity_and_load_is_transactional() {
        struct Main;
        impl Placement for Main {}
        struct Other;
        impl Placement for Other {}

        let main = DeviceCtx::<Main>::bind(Device::Cpu).unwrap();
        let other = DeviceCtx::<Other>::bind(Device::Cpu).unwrap();
        let source = model(10.0, &main);
        let state = state_dict(&source).unwrap();
        let mut target = model(1.0, &other);
        let before = values(&target);
        assert!(load_state_dict(&mut target, &state).is_err());
        assert_eq!(values(&target), before);
    }

    #[test]
    fn envelope_load_stages_every_value_and_preserves_checkpoint_sections() {
        let ctx = DeviceCtx::cpu().unwrap();
        let source = model(10.0, &ctx);
        let mut envelope = Envelope::new();
        envelope.set_section("rng", "seed=7").unwrap();
        save_model_state(&mut model(10.0, &ctx), &mut envelope).unwrap();
        let mut target = model(1.0, &ctx);
        load_model_state(&mut target, &envelope, &LoadOptions::strict()).unwrap();
        assert_eq!(values(&target), values(&source));
        assert_eq!(envelope.section("rng"), Some("seed=7"));

        let mut bad = envelope.clone();
        bad.insert_tensor(
            "running",
            crate::persist::HostTensor::from_bytes(DType::F32, vec![2], vec![0; 8]).unwrap(),
        );
        let before = values(&target);
        assert!(load_model_state(&mut target, &bad, &LoadOptions::strict()).is_err());
        assert_eq!(values(&target), before);
    }

    #[test]
    fn dtype_transition_requires_a_reconstructed_typed_target() {
        let ctx = DeviceCtx::cpu().unwrap();
        let mut source = model(4.0, &ctx);
        let mut envelope = Envelope::new();
        save_model_state(&mut source, &mut envelope).unwrap();

        #[derive(TypedModule)]
        struct F64Model {
            weight: TypedParam<Tensor1<2, f64>>,
        }
        let mut target = F64Model {
            weight: TypedParam::new(Tensor1::from_vec(vec![0.0f64, 0.0], [2], &ctx).unwrap())
                .unwrap(),
        };
        assert!(load_model_state(&mut target, &envelope, &LoadOptions::strict()).is_err());
        assert_eq!(
            target.weight.value().unwrap().to_vec().unwrap(),
            vec![0.0, 0.0]
        );
    }

    struct Reserved {
        value: TypedParam<Tensor1<1>>,
        dotted: bool,
    }

    impl Module for Reserved {
        fn visit(&self, visitor: &mut TypedVisitor<'_>) {
            visitor.param(
                if self.dotted { "optim.weight" } else { "optim" },
                &self.value,
            );
        }

        fn visit_mut(&mut self, visitor: &mut TypedVisitorMut<'_>) {
            visitor.param(
                if self.dotted { "optim.weight" } else { "optim" },
                &mut self.value,
            );
        }
    }

    #[test]
    fn optimizer_namespace_is_reserved_at_typed_state_and_save_boundaries() {
        let ctx = DeviceCtx::cpu().unwrap();
        let reserved = |dotted| Reserved {
            value: TypedParam::new(Tensor1::from_vec(vec![1.0], [1], &ctx).unwrap()).unwrap(),
            dotted,
        };
        let expect = |op: &str, path: &str| {
            format!(
                "{op}: invalid argument: model path {path:?} conflicts with the reserved \
                 optimizer checkpoint namespace `optim.*`"
            )
        };

        for (dotted, path) in [(false, "optim"), (true, "optim.weight")] {
            let mut offender = reserved(dotted);
            // `TypedStateDict` is deliberately opaque (no `Debug`), so the error
            // is taken through `err()` rather than `unwrap_err()`.
            assert_eq!(
                state_dict(&offender).err().unwrap().to_string(),
                expect("typed::persist::state_dict", path)
            );
            assert_eq!(
                save_model_state(&mut offender, &mut Envelope::new())
                    .unwrap_err()
                    .to_string(),
                expect("typed::persist::save_model_state", path)
            );
            // The load side rejects the reserved namespace on the *target* as
            // well; the state here is a well-formed unrelated model's, so the
            // target check is the only guard that can be doing the rejecting.
            let mut target = reserved(dotted);
            let unrelated = state_dict(&model(1.0, &ctx)).unwrap();
            assert_eq!(
                load_state_dict(&mut target, &unrelated)
                    .unwrap_err()
                    .to_string(),
                expect("typed::persist::load_state_dict", path)
            );
            assert_eq!(
                load_model_state(&mut target, &Envelope::new(), &LoadOptions::strict())
                    .unwrap_err()
                    .to_string(),
                expect("typed::persist::load_model_state", path)
            );
        }

        // ...and on the incoming *state*, reachable because the public
        // `typed::nn::state_dict` does not itself reject reserved paths. The
        // target is clean here, so only the state-side check can reject.
        let mut clean = model(1.0, &ctx);
        let smuggled = nn::state_dict(&reserved(true)).unwrap();
        assert_eq!(
            load_state_dict(&mut clean, &smuggled)
                .unwrap_err()
                .to_string(),
            expect("typed::persist::load_state_dict", "optim.weight")
        );
    }

    /// A combined checkpoint is the only route that may carry `optim.*`, so the
    /// combined loader must refuse a model-only file rather than load the model
    /// half and rely on the caller's optimizer failing afterwards.
    #[test]
    fn combined_load_rejects_a_model_only_checkpoint_and_leaves_the_model_untouched() {
        let ctx = DeviceCtx::cpu().unwrap();
        let mut envelope = Envelope::new();
        save_model_state(&mut model(9.0, &ctx), &mut envelope).unwrap();

        let mut target = model(1.0, &ctx);
        let before = values(&target);
        assert_eq!(
            load_combined_model_state(&mut target, &envelope, &LoadOptions::strict())
                .unwrap_err()
                .to_string(),
            "persistence: combined checkpoint has no `optimizer` section"
        );
        assert_eq!(values(&target), before);
    }

    #[test]
    fn model_only_load_never_ignores_optimizer_prefixed_tensors() {
        let ctx = DeviceCtx::cpu().unwrap();
        let mut source = model(5.0, &ctx);
        let mut envelope = Envelope::new();
        save_model_state(&mut source, &mut envelope).unwrap();
        envelope.insert_tensor(
            "optim.attacker.m",
            crate::persist::HostTensor::from_bytes(DType::F32, vec![1], vec![0; 4]).unwrap(),
        );
        let mut target = model(1.0, &ctx);
        let before = values(&target);
        assert!(load_model_state(&mut target, &envelope, &LoadOptions::strict()).is_err());
        assert!(
            load_model_state(
                &mut target,
                &envelope,
                &LoadOptions::strict().allow_unexpected()
            )
            .is_err()
        );
        assert_eq!(values(&target), before);

        let mut plain_sgd = Envelope::new();
        save_model_state(&mut source, &mut plain_sgd).unwrap();
        plain_sgd
            .set_section(
                "optimizer",
                "version=1\nkind=sgd\nsteps=0\nhyper.lr=0.1\nhyper.momentum=0\nhyper.weight_decay=0\n",
            )
            .unwrap();
        assert!(load_model_state(&mut target, &plain_sgd, &LoadOptions::strict()).is_err());
        assert_eq!(values(&target), before);
    }

    #[test]
    fn combined_load_rejects_bare_optim_even_when_unexpected_paths_are_allowed() {
        let ctx = DeviceCtx::cpu().unwrap();
        let mut source = model(5.0, &ctx);
        let mut envelope = Envelope::new();
        save_model_state(&mut source, &mut envelope).unwrap();
        envelope
            .set_section(
                "optimizer",
                "version=1\nkind=sgd\nsteps=0\nhyper.lr=0.1\nhyper.momentum=0\nhyper.weight_decay=0\n",
            )
            .unwrap();
        envelope.insert_tensor(
            "optim",
            crate::persist::HostTensor::from_bytes(DType::F32, vec![1], vec![0; 4]).unwrap(),
        );
        let mut target = model(1.0, &ctx);
        let before = values(&target);
        assert!(
            load_combined_model_state(
                &mut target,
                &envelope,
                &LoadOptions::strict().allow_unexpected()
            )
            .is_err()
        );
        assert_eq!(values(&target), before);
    }
}
