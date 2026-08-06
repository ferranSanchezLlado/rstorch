//! Persistence: safetensors read/write with the 16.14 behaviors — atomic
//! temp-and-rename saves, staged all-or-nothing restore, reader limits,
//! and a versioned envelope for non-tensor state.
//!
//! # Scope: host data only
//!
//! This layer speaks in [`HostTensor`]s — dtype + dims + contiguous
//! little-endian bytes — and never touches a live runtime tensor, its storage,
//! or its layout. The crate's checkpoint runtime bridges tensors
//! to and from `HostTensor`s and drives the transactional swaps; this module
//! owns the fallible, on-disk half. Keeping the two apart lets the
//! file format and the tensor core evolve independently (exploration §4.6).
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

use crate::error::Result;

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
/// [`Error::Persistence`](crate::Error::Persistence) if the tensor map exceeds
/// `limits`, or [`Error::Io`](crate::Error::Io) on a filesystem failure.
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
/// [`Error::Persistence`](crate::Error::Persistence) if the file is not valid
/// safetensors, exceeds `limits`, or carries a dtype the crate does not model;
/// [`Error::Io`](crate::Error::Io) on a filesystem failure.
pub fn load_safetensors(
    path: impl AsRef<Path>,
    limits: &Limits,
) -> Result<(BTreeMap<String, HostTensor>, HashMap<String, String>)> {
    safetensors_io::load_tensors(path.as_ref(), limits)
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
}
