//! Persistence: safetensors read/write — atomic temp-and-rename saves, staged
//! all-or-nothing restore, reader limits, and a versioned envelope for
//! non-tensor state.
//!
//! # Scope: host data only
//!
//! This layer speaks in [`HostTensor`]s — dtype + dims + contiguous
//! little-endian bytes — and never touches a live runtime tensor, its storage,
//! or its layout. The crate's checkpoint runtime bridges tensors
//! to and from `HostTensor`s and drives the transactional swaps; this module
//! owns the fallible, on-disk half. Keeping the two apart lets the
//! file format and the tensor core evolve independently.
//!
//! # What lives here
//!
//! - [`HostTensor`]: the storage-independent tensor payload.
//! - [`save_safetensors`] / [`load_safetensors`]: a map of named
//!   `HostTensor`s to and from an ecosystem-compatible safetensors file,
//!   written atomically and read under [`Limits`].
//! - [`Envelope`]: a minimal versioned checkpoint carrying tensors plus
//!   opaque `config`/`optimizer`/`rng` sections.
//! - [`stage`] + [`StagedTensors`]: the all-or-nothing restore surface —
//!   validate a loaded map against an [`Expected`] schema under
//!   [`LoadOptions`], then hand the caller a value whose mere existence proves
//!   every swap it is about to make will succeed.
//! - [`Limits`] / [`LoadOptions`] / [`MissingPolicy`] / [`UnexpectedPolicy`]:
//!   the safe-default reader limits and load policies for untrusted files.
//!
//! These safeguards are structural, not cryptographic. Files are not
//! checksummed or authenticated, so a bit-flipped tensor payload can still
//! decode as valid, different values. Add an application-level integrity check
//! before loading checkpoints received from an untrusted or unreliable source.

// Also the install step for cached hub downloads (`crate::data::hub`), so
// every file this crate creates goes through one atomic-save path.
pub(crate) mod atomic;
mod envelope;
mod host_tensor;
mod options;
mod restore;
mod safetensors_io;

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use crate::error::{Error, Result};
use crate::nn::{Module, ModuleExt};

pub use envelope::{Envelope, FORMAT_MAJOR, FORMAT_MINOR};
pub use host_tensor::HostTensor;
pub use options::{Limits, LoadOptions, MissingPolicy, UnexpectedPolicy};
pub use restore::{Expected, StagedTensors, stage};

/// Atomically write a map of named [`HostTensor`]s to `path` as an
/// ecosystem-compatible safetensors file.
///
/// The write is atomic (temp-and-rename, see the [module docs](self)) and the
/// writer validates against the *same* `limits` a reader enforces, so this can
/// never emit a file [`load_safetensors`] with the same limits would reject.
///
/// # Errors
///
/// [`Error::Persistence`] if the tensor map exceeds `limits`, or [`Error::Io`]
/// on a filesystem failure.
///
/// # Examples
///
/// ```
/// use rstorch::persist::{HostTensor, Limits, load_safetensors, save_safetensors};
/// use rstorch::DType;
/// use std::collections::BTreeMap;
///
/// # fn main() -> rstorch::Result<()> {
/// let path = std::env::temp_dir().join(format!(
///     "rstorch-doctest-safetensors-{}.safetensors",
///     std::process::id()
/// ));
///
/// let mut tensors = BTreeMap::new();
/// tensors.insert(
///     "w".to_string(),
///     HostTensor::from_bytes(DType::F32, vec![2], 1.0f32.to_le_bytes().repeat(2))?,
/// );
/// save_safetensors(&path, &tensors, &Limits::default())?;
///
/// let (loaded, _metadata) = load_safetensors(&path, &Limits::default())?;
/// assert_eq!(loaded["w"].dims(), &[2]);
/// # let _ = std::fs::remove_file(&path);
/// # Ok(())
/// # }
/// ```
pub fn save_safetensors(
    path: impl AsRef<Path>,
    tensors: &BTreeMap<String, HostTensor>,
    limits: &Limits,
) -> Result<()> {
    safetensors_io::save_tensors(path.as_ref(), tensors, None, limits)
}

/// Read a safetensors file at `path` into a map of named [`HostTensor`]s,
/// returning the tensors and the file's free-form string metadata map.
///
/// Every self-declared size in the file is checked against `limits` *before*
/// any tensor data is allocated (see [`Limits`]), so a file that lies about
/// its sizes is rejected rather than allowed to exhaust memory.
///
/// # Errors
///
/// [`Error::Persistence`] if the file is not valid safetensors, exceeds
/// `limits`, or carries a dtype the crate does not model; [`Error::Io`] on a
/// filesystem failure.
pub fn load_safetensors(
    path: impl AsRef<Path>,
    limits: &Limits,
) -> Result<(BTreeMap<String, HostTensor>, HashMap<String, String>)> {
    safetensors_io::load_tensors(path.as_ref(), limits)
}

fn reserved_optimizer_path(path: &str) -> bool {
    path == "optim" || path.starts_with("optim.")
}

fn reject_reserved_model_paths<'a>(
    paths: impl Iterator<Item = &'a str>,
    op: &'static str,
) -> Result<()> {
    if let Some(path) = paths.into_iter().find(|path| reserved_optimizer_path(path)) {
        return Err(Error::invalid_arg(
            op,
            format!(
                "model path {path:?} conflicts with the reserved optimizer \
                 checkpoint namespace `optim.*`"
            ),
        ));
    }
    Ok(())
}

/// Add a dynamic model's tensor state to an [`Envelope`].
///
/// This function writes model tensors only. Configuration, optimizer state,
/// RNG state, and application sections remain caller-owned and can be added to
/// the same envelope before [`Envelope::save`].
///
/// # Errors
///
/// Returns an error for a malformed module walk, a model path that collides
/// with the reserved `optim.*` namespace, or a host transfer failure. All
/// conversions complete before the envelope is mutated; existing tensor keys
/// are replaced as documented by [`Envelope::insert_tensor`].
pub fn save_model_state<M: Module + ?Sized>(model: &M, envelope: &mut Envelope) -> Result<()> {
    let state = model.state_dict()?;
    reject_reserved_model_paths(state.paths(), "persist::save_model_state")?;
    let staged = state
        .into_iter()
        .map(|(path, tensor)| Ok((path, crate::checkpoint::to_host_tensor(&tensor)?)))
        .collect::<Result<Vec<_>>>()?;
    for (path, tensor) in staged {
        envelope.insert_tensor(path, tensor);
    }
    Ok(())
}

/// Stage and load dynamic model tensor state from an [`Envelope`].
///
/// The target model supplies each tensor's expected shape, dtype, and device.
/// Missing and unexpected **model** paths follow `options`; shape and dtype
/// mismatches are always errors. This helper selects model state from a
/// composable envelope, ignoring its reserved optimizer section and
/// `optim.*` tensors. Optimizer restoration remains a separate caller
/// operation; this load does not provide a combined rollback.
///
/// # Errors
///
/// Returns the envelope/schema, host-transfer, or model-load error. No model
/// leaf is replaced unless the complete staged load is valid.
pub fn load_model_state<M: Module + ?Sized>(
    model: &mut M,
    envelope: &Envelope,
    options: &LoadOptions,
) -> Result<()> {
    let model_tensors = envelope
        .tensors()
        .iter()
        .filter(|(path, _)| !reserved_optimizer_path(path))
        .map(|(path, tensor)| (path.clone(), tensor.clone()))
        .collect::<BTreeMap<_, _>>();

    let current = model.state_dict()?;
    reject_reserved_model_paths(current.paths(), "persist::load_model_state")?;
    let schema = current
        .iter()
        .map(|(path, tensor)| Expected::new(path, tensor.dtype(), tensor.dims().to_vec()))
        .collect::<Vec<_>>();
    let staged = stage(&schema, &model_tensors, options)?;

    let mut replacements = current;
    for (path, host) in staged.into_entries() {
        let device = replacements
            .get(&path)
            .expect("staged path came from the target schema")
            .device();
        replacements.insert(path, crate::checkpoint::from_host_tensor(&host, &device)?)?;
    }
    model.load_state_dict(&replacements)
}

/// Save a model-only dynamic checkpoint to `path`.
///
/// The file contains only model tensors. Use [`Envelope`] directly when a
/// checkpoint also needs configuration, optimizer, RNG, or application state.
pub fn save_checkpoint<M: Module + ?Sized>(
    model: &M,
    path: impl AsRef<Path>,
    limits: &Limits,
) -> Result<()> {
    let mut envelope = Envelope::new();
    save_model_state(model, &mut envelope)?;
    envelope.save(path, limits)
}

/// Load model state from a checkpoint envelope at `path` into `model`.
///
/// The checkpoint may be a composable envelope carrying configuration,
/// optimizer, RNG, or application state in addition to model tensors.
/// Optimizer restoration remains a separate caller operation; this helper does
/// not provide a combined rollback.
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
    use crate::dtype::DType;
    use std::path::PathBuf;

    fn tmpdir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rstorch-persist-mod-{}-{}-{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn public_save_load_round_trip() {
        let dir = tmpdir("rt");
        let path = dir.join("m.safetensors");
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "w".to_string(),
            HostTensor::from_bytes(DType::F32, vec![2], 1.0f32.to_le_bytes().repeat(2)).unwrap(),
        );

        save_safetensors(&path, &tensors, &Limits::defaults()).unwrap();
        let (back, meta) = load_safetensors(&path, &Limits::defaults()).unwrap();
        assert_eq!(back, tensors);
        assert!(meta.is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[derive(crate::Module)]
    struct Model {
        weight: crate::nn::Param,
        running: crate::Tensor,
    }

    fn model(weight: f32, running: f32) -> Model {
        Model {
            weight: crate::nn::Param::new(
                crate::Tensor::from_vec(vec![weight; 2], [2], &crate::Device::Cpu).unwrap(),
            ),
            running: crate::Tensor::from_vec(vec![running; 2], [2], &crate::Device::Cpu).unwrap(),
        }
    }

    #[test]
    fn dynamic_model_checkpoint_round_trips_parameters_and_buffers() {
        let dir = tmpdir("model");
        let path = dir.join("model.safetensors");
        let source = model(3.0, 7.0);
        save_checkpoint(&source, &path, &Limits::defaults()).unwrap();

        let mut target = model(0.0, 0.0);
        load_checkpoint(&mut target, &path, &LoadOptions::strict()).unwrap();
        let state = target.state_dict().unwrap();
        assert_eq!(state["weight"].to_vec::<f32>().unwrap(), vec![3.0; 2]);
        assert_eq!(state["running"].to_vec::<f32>().unwrap(), vec![7.0; 2]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn model_load_selects_model_state_from_composable_optimizer_envelope() {
        let mut source = model(2.0, 7.0);
        let mut optimizer = crate::optim::Sgd::new(0.1).momentum(0.9);
        let loss = source
            .weight
            .get(crate::nn::Mode::TRAIN)
            .mul(&source.weight.get(crate::nn::Mode::TRAIN))
            .unwrap()
            .sum_all()
            .unwrap();
        optimizer
            .step(&mut source, loss.backward().unwrap())
            .unwrap();

        let mut envelope = Envelope::new();
        save_model_state(&source, &mut envelope).unwrap();
        optimizer.save_state(&source, &mut envelope).unwrap();
        let optimizer_section = envelope.section("optimizer").map(str::to_owned);
        let optimizer_tensor = envelope.tensor("optim.weight.velocity").cloned();
        assert!(optimizer_section.is_some());
        assert!(optimizer_tensor.is_some());

        let mut target = model(0.0, 0.0);
        load_model_state(&mut target, &envelope, &LoadOptions::strict()).unwrap();
        let state = target.state_dict().unwrap();
        assert_eq!(
            state["weight"].to_vec::<f32>().unwrap(),
            source.state_dict().unwrap()["weight"]
                .to_vec::<f32>()
                .unwrap()
        );
        assert_eq!(state["running"].to_vec::<f32>().unwrap(), vec![7.0; 2]);
        assert_eq!(
            envelope.section("optimizer").map(str::to_owned),
            optimizer_section
        );
        assert_eq!(
            envelope.tensor("optim.weight.velocity"),
            optimizer_tensor.as_ref()
        );
    }
}
