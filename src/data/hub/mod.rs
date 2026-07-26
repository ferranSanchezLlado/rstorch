//! Dataset hub: download, caching, checksum verification, and parsing for
//! the bundled datasets (MNIST, TinyShakespeare).
//!
//! This is the **raw** layer (implementation-plan §4, task T15): it produces
//! plain `Vec`s and small metadata structs and has **no `Tensor`
//! dependency**. The [`crate::data`] `Dataset` wrappers that turn these into
//! trainable batches are layered on top separately (task T46).
//!
//! ## Feature gating
//!
//! Network access (`ureq`) and gzip decompression (`flate2`) live behind
//! the `hub` feature. The checksum and the IDX / text parsers compile and are
//! tested on the default feature set, so a downstream crate that provides its
//! own bytes never has to enable networking. Every method that fetches or
//! decompresses is either `#[cfg(feature = "hub")]` or takes a
//! caller-supplied fetch closure so it can be exercised entirely offline.
//!
//! ```no_run
//! # #[cfg(feature = "hub")] {
//! use rstorch::data::hub::{DatasetHub, Mnist, MnistSplit};
//!
//! let hub = DatasetHub::default_cache();
//! let train = Mnist::load(&hub, MnistSplit::Train)?;
//! assert_eq!(train.image_shape(), (28, 28));
//! # }
//! # Ok::<(), rstorch::Error>(())
//! ```

use crate::error::{Error, Result};
use std::fs;
use std::path::{Path, PathBuf};

pub mod mnist;
pub mod mnist_dataset;
mod sha256;
pub mod tiny_shakespeare;
pub mod tiny_shakespeare_dataset;

pub use mnist::{Mnist, MnistSplit, RawImages};
pub use tiny_shakespeare::{TINY_SHAKESPEARE, TinyShakespeare};

// The `Dataset` wrappers over the raw layer above (T46).
pub use mnist_dataset::{MnistDataset, MnistLayout};
pub use tiny_shakespeare_dataset::TinyShakespeareDataset;

/// A local cache directory for downloaded dataset files.
///
/// Files are stored under `<root>/<dataset>/<file_name>`. Downloads are
/// atomic (write to a `.tmp` sibling, verify, then rename) so a partial or
/// corrupt fetch never leaves a bad file at the final path.
#[derive(Debug, Clone)]
pub struct DatasetHub {
    root: PathBuf,
}

/// Static metadata describing one downloadable dataset file.
///
/// A resource is verified against its optional
/// [`sha256`](DatasetResource::sha256) digest and rejected if it exceeds its
/// optional [`max_bytes`](DatasetResource::max_bytes) size cap (an
/// untrusted-download guard).
#[derive(Debug, Clone, Copy)]
pub struct DatasetResource {
    /// Human-readable name, used in diagnostics.
    pub name: &'static str,
    /// Source URL fetched by the default network fetcher.
    pub url: &'static str,
    /// File name used inside the dataset's cache directory.
    pub file_name: &'static str,
    /// Expected lowercase-hex SHA-256 digest, if integrity is checked.
    pub sha256: Option<&'static str>,
    /// Maximum accepted payload size in bytes, if a cap is enforced.
    pub max_bytes: Option<u64>,
}

impl DatasetHub {
    /// Creates a hub rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The default cache location.
    ///
    /// Resolution order (keeping v2's convention): the `RSTORCH_DATA`
    /// environment variable, then `$HOME/.cache/rstorch`, then the
    /// project-local `data/` directory (which is git-ignored).
    pub fn default_cache() -> Self {
        let root = std::env::var_os("RSTORCH_DATA")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/rstorch"))
            })
            .unwrap_or_else(|| PathBuf::from("data"));
        Self::new(root)
    }

    /// The cache root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The directory holding files for `dataset`.
    pub fn dataset_dir(&self, dataset: &str) -> PathBuf {
        self.root.join(dataset)
    }

    /// The full cache path for `resource` within `dataset`.
    pub fn resource_path(&self, dataset: &str, resource: &DatasetResource) -> PathBuf {
        self.dataset_dir(dataset).join(resource.file_name)
    }

    /// Whether `resource` is already present in the cache.
    pub fn is_cached(&self, dataset: &str, resource: &DatasetResource) -> bool {
        self.resource_path(dataset, resource).is_file()
    }

    /// Ensures every resource is cached (downloading missing ones over the
    /// network), returning their paths in order.
    ///
    /// Available only with the `hub` feature; use
    /// [`ensure_cached_with`](Self::ensure_cached_with) for an offline,
    /// caller-supplied fetcher.
    #[cfg(feature = "hub")]
    pub fn ensure_cached(
        &self,
        dataset: &str,
        resources: &[DatasetResource],
    ) -> Result<Vec<PathBuf>> {
        self.ensure_cached_with(dataset, resources, fetch_resource)
    }

    /// Ensures every resource is cached, fetching missing ones with the
    /// caller-supplied `fetch` closure. Returns their paths in order.
    pub fn ensure_cached_with<F>(
        &self,
        dataset: &str,
        resources: &[DatasetResource],
        mut fetch: F,
    ) -> Result<Vec<PathBuf>>
    where
        F: FnMut(&DatasetResource) -> Result<Vec<u8>>,
    {
        resources
            .iter()
            .map(|resource| self.ensure_resource_with(dataset, resource, &mut fetch))
            .collect()
    }

    /// Ensures a single resource is cached, downloading it if missing.
    ///
    /// Available only with the `hub` feature; use
    /// [`ensure_resource_with`](Self::ensure_resource_with) for an offline,
    /// caller-supplied fetcher.
    #[cfg(feature = "hub")]
    pub fn ensure_resource(&self, dataset: &str, resource: &DatasetResource) -> Result<PathBuf> {
        self.ensure_resource_with(dataset, resource, fetch_resource)
    }

    /// Ensures a single resource is cached, fetching it with `fetch` if it is
    /// missing.
    pub fn ensure_resource_with<F>(
        &self,
        dataset: &str,
        resource: &DatasetResource,
        fetch: F,
    ) -> Result<PathBuf>
    where
        F: FnMut(&DatasetResource) -> Result<Vec<u8>>,
    {
        let path = self.resource_path(dataset, resource);
        if path.is_file() {
            return Ok(path);
        }
        self.download_with(dataset, resource, fetch)
    }

    /// Downloads `resource` over the network, replacing any cached copy.
    ///
    /// Available only with the `hub` feature; use
    /// [`download_with`](Self::download_with) for an offline, caller-supplied
    /// fetcher.
    #[cfg(feature = "hub")]
    pub fn download(&self, dataset: &str, resource: &DatasetResource) -> Result<PathBuf> {
        self.download_with(dataset, resource, fetch_resource)
    }

    /// Fetches `resource` with `fetch`, verifies it, and atomically installs
    /// it into the cache, returning its path.
    ///
    /// The bytes are written to a `.tmp` sibling first, then the size cap and
    /// SHA-256 checksum (if any) are checked, and only on success is the file
    /// renamed into place. A failing check removes the `.tmp` file and
    /// returns an error, so the final path is only ever a verified file.
    pub fn download_with<F>(
        &self,
        dataset: &str,
        resource: &DatasetResource,
        mut fetch: F,
    ) -> Result<PathBuf>
    where
        F: FnMut(&DatasetResource) -> Result<Vec<u8>>,
    {
        let dir = self.dataset_dir(dataset);
        fs::create_dir_all(&dir)?;
        let path = dir.join(resource.file_name);
        let tmp = path.with_extension("tmp");

        let bytes = fetch(resource)?;
        if let Some(max) = resource.max_bytes
            && bytes.len() as u64 > max
        {
            return Err(Error::Data {
                msg: format!(
                    "resource `{}` exceeds size cap of {max} bytes (got {})",
                    resource.name,
                    bytes.len()
                ),
            });
        }
        fs::write(&tmp, &bytes)?;
        if let Some(expected) = resource.sha256 {
            let found = sha256::sha256_hex(&bytes);
            if found != expected {
                let _ = fs::remove_file(&tmp);
                return Err(Error::Data {
                    msg: format!(
                        "checksum mismatch for `{}`: expected {expected}, found {found}",
                        resource.name
                    ),
                });
            }
        }
        fs::rename(&tmp, &path)?;
        Ok(path)
    }
}

/// The default network fetcher: an HTTP GET reading the full body into memory.
#[cfg(feature = "hub")]
fn fetch_resource(resource: &DatasetResource) -> Result<Vec<u8>> {
    use std::io::Read as _;
    let mut reader = ureq::get(resource.url)
        .call()
        .map_err(|source| Error::Data {
            msg: format!("failed to fetch `{}`: {source}", resource.name),
        })?
        .into_reader();
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "rstorch-hub-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    #[test]
    fn download_with_uses_supplied_fetcher() {
        let root = scratch("fetch");
        let _ = fs::remove_dir_all(&root);
        let hub = DatasetHub::new(&root);
        let resource = DatasetResource {
            name: "test",
            url: "https://example.invalid/test",
            file_name: "test.bin",
            sha256: None,
            max_bytes: None,
        };

        let path = hub
            .download_with("fixture", &resource, |_| Ok(vec![1, 2, 3]))
            .unwrap();

        assert_eq!(fs::read(&path).unwrap(), vec![1, 2, 3]);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn checksum_mismatch_rejects_wrong_bytes_and_cleans_tmp() {
        let root = scratch("sha");
        let _ = fs::remove_dir_all(&root);
        let hub = DatasetHub::new(&root);
        let resource = DatasetResource {
            name: "test",
            url: "https://example.invalid/test",
            file_name: "sha.bin",
            sha256: Some("0000000000000000000000000000000000000000000000000000000000000000"),
            max_bytes: None,
        };
        let result = hub.download_with("fixture", &resource, |_| Ok(vec![1, 2, 3]));
        assert!(
            matches!(result, Err(Error::Data { .. })),
            "download must fail on checksum mismatch"
        );
        assert!(!root.join("fixture/sha.bin.tmp").exists());
        assert!(!root.join("fixture/sha.bin").exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn matching_checksum_is_accepted() {
        let root = scratch("sha-ok");
        let _ = fs::remove_dir_all(&root);
        let hub = DatasetHub::new(&root);
        let bytes = b"abc".to_vec();
        let resource = DatasetResource {
            name: "test",
            url: "https://example.invalid/test",
            file_name: "ok.bin",
            sha256: Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
            max_bytes: None,
        };
        let path = hub
            .download_with("fixture", &resource, |_| Ok(bytes.clone()))
            .unwrap();
        assert_eq!(fs::read(path).unwrap(), bytes);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn size_cap_rejects_large_payload() {
        let root = scratch("cap");
        let _ = fs::remove_dir_all(&root);
        let hub = DatasetHub::new(&root);
        let resource = DatasetResource {
            name: "test",
            url: "https://example.invalid/test",
            file_name: "cap.bin",
            sha256: None,
            max_bytes: Some(2),
        };
        let result = hub.download_with("fixture", &resource, |_| Ok(vec![1, 2, 3]));
        assert!(matches!(result, Err(Error::Data { .. })));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn ensure_resource_returns_cached_without_fetching() {
        let root = scratch("cached");
        let _ = fs::remove_dir_all(&root);
        let hub = DatasetHub::new(&root);
        let resource = DatasetResource {
            name: "test",
            url: "https://example.invalid/test",
            file_name: "c.bin",
            sha256: None,
            max_bytes: None,
        };
        // First call fetches.
        hub.ensure_resource_with("fixture", &resource, |_| Ok(vec![9]))
            .unwrap();
        assert!(hub.is_cached("fixture", &resource));
        // Second call must not invoke the fetcher.
        let path = hub
            .ensure_resource_with("fixture", &resource, |_| {
                panic!("fetcher must not run for a cached resource")
            })
            .unwrap();
        assert_eq!(fs::read(path).unwrap(), vec![9]);
        let _ = fs::remove_dir_all(root);
    }
}
