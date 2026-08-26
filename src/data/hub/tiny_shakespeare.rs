//! Raw `TinyShakespeare` loading: download, cache, and text access.
//!
//! This module is `Tensor`- and tokenizer-free: it only downloads and returns
//! the corpus as a `String`. Tokenizing it into causal-LM windows is the job of
//! the `Dataset` wrappers built on the [`crate::text`] tokenizers.

use super::{DatasetHub, DatasetResource};
use crate::error::Result;

const TINY_SHAKESPEARE_DATASET: &str = "tiny_shakespeare";

/// The single-file `TinyShakespeare` corpus resource.
pub const TINY_SHAKESPEARE: DatasetResource = DatasetResource::new(
    "TinyShakespeare input.txt",
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "input.txt",
    Some("86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"),
    Some(2_000_000),
);

/// The `TinyShakespeare` corpus source.
///
/// This is a thin handle over [`DatasetHub`]; the corpus itself is returned
/// as an owned `String` from `load_text` (with the `hub` feature) or
/// [`read_cached_text`](TinyShakespeare::read_cached_text).
///
/// # Examples
///
/// ```no_run
/// # #[cfg(feature = "hub")] {
/// use rstorch::data::hub::{DatasetHub, TinyShakespeare};
///
/// let hub = DatasetHub::default_cache();
/// let text = TinyShakespeare::load_text(&hub)?;
/// assert!(!text.is_empty());
/// # }
/// # Ok::<(), rstorch::Error>(())
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TinyShakespeare;

impl TinyShakespeare {
    /// The cache path the corpus is (or would be) stored at.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Data`](crate::Error::Data) if the fixed cache key is
    /// ever changed to an invalid path component.
    pub fn cache_path(hub: &DatasetHub) -> Result<std::path::PathBuf> {
        hub.resource_path(TINY_SHAKESPEARE_DATASET, &TINY_SHAKESPEARE)
    }

    /// Downloads the corpus into `hub`'s cache if missing.
    ///
    /// Requires the `hub` feature (network access).
    ///
    /// # Errors
    ///
    /// As [`DatasetHub::ensure_resource`].
    #[cfg(feature = "hub")]
    pub fn download(hub: &DatasetHub) -> Result<()> {
        hub.ensure_resource(TINY_SHAKESPEARE_DATASET, &TINY_SHAKESPEARE)?;
        Ok(())
    }

    /// Loads the corpus as UTF-8 text, downloading it first if necessary.
    ///
    /// Requires the `hub` feature (network access).
    ///
    /// # Errors
    ///
    /// As [`DatasetHub::ensure_resource`], plus [`Error::Io`](crate::Error::Io)
    /// if the downloaded file is not valid UTF-8.
    #[cfg(feature = "hub")]
    pub fn load_text(hub: &DatasetHub) -> Result<String> {
        let path = hub.ensure_resource(TINY_SHAKESPEARE_DATASET, &TINY_SHAKESPEARE)?;
        Ok(std::fs::read_to_string(path)?)
    }

    /// Reads the already-cached corpus as verified UTF-8 text without any
    /// network access. Returns an [`crate::Error::Io`] if it has not been
    /// downloaded, and [`crate::Error::Data`] if its fixed size or checksum
    /// does not match [`TINY_SHAKESPEARE`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Io`](crate::Error::Io) if the corpus has not been
    /// downloaded into `hub`'s cache, [`Error::Data`](crate::Error::Data) if
    /// the cached bytes fail verification, or [`Error::Io`](crate::Error::Io)
    /// if the file is not valid UTF-8.
    pub fn read_cached_text(hub: &DatasetHub) -> Result<String> {
        let path = hub.verified_cached_path(TINY_SHAKESPEARE_DATASET, &TINY_SHAKESPEARE)?;
        Ok(std::fs::read_to_string(path)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use std::path::PathBuf;

    fn scratch(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "rstorch-tinyshakes-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    #[test]
    fn resource_metadata_is_stable() {
        assert_eq!(TINY_SHAKESPEARE.file_name, "input.txt");
        assert_eq!(TINY_SHAKESPEARE.name, "TinyShakespeare input.txt");
        assert!(TINY_SHAKESPEARE.sha256.is_some());
        assert_eq!(TINY_SHAKESPEARE.max_bytes, Some(2_000_000));
    }

    #[test]
    fn cache_path_uses_dataset_subdir() {
        let hub = DatasetHub::new("/tmp/rstorch-ts-root");
        let path = TinyShakespeare::cache_path(&hub).unwrap();
        assert!(path.ends_with("tiny_shakespeare/input.txt"));
    }

    #[test]
    fn read_cached_text_rejects_unverified_file() {
        let root = scratch("read");
        let _ = std::fs::remove_dir_all(&root);
        let hub = DatasetHub::new(&root);
        let path = TinyShakespeare::cache_path(&hub).unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "To be, or not to be").unwrap();

        assert!(matches!(
            TinyShakespeare::read_cached_text(&hub),
            Err(Error::Data { msg, .. }) if msg.contains("checksum mismatch")
        ));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn read_cached_text_missing_file_is_io_error() {
        let hub = DatasetHub::new(scratch("missing"));
        assert!(matches!(
            TinyShakespeare::read_cached_text(&hub),
            Err(Error::Io(_))
        ));
    }
}
