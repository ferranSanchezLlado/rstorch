//! Dataset hub: download, caching, checksum verification, and parsing for
//! the bundled datasets (MNIST, `TinyShakespeare`).
//!
//! This is the **raw** layer: it produces plain `Vec`s and small metadata
//! structs and has **no `Tensor` dependency**. The [`crate::data`] `Dataset`
//! wrappers that turn these into trainable batches are layered on top.
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
use std::path::{Component, Path, PathBuf};

pub mod mnist;
pub mod mnist_dataset;
mod sha256;
pub mod tiny_shakespeare;
pub mod tiny_shakespeare_dataset;

pub use mnist::{Mnist, MnistSplit, RawImages};
pub use tiny_shakespeare::{TINY_SHAKESPEARE, TinyShakespeare};

// The `Dataset` wrappers over the raw layer above.
pub use mnist_dataset::{MnistDataset, MnistLayout};
pub use tiny_shakespeare_dataset::TinyShakespeareDataset;

/// A local cache directory for downloaded dataset files.
///
/// Files are stored under `<root>/<dataset>/<file_name>`. A download is
/// verified against its checksum and size cap *before* it is written, and then
/// installed atomically, so a partial or corrupt fetch never leaves a bad file
/// at the final path.
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
#[non_exhaustive]
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

impl DatasetResource {
    /// Creates static metadata for one downloadable dataset file.
    pub const fn new(
        name: &'static str,
        url: &'static str,
        file_name: &'static str,
        sha256: Option<&'static str>,
        max_bytes: Option<u64>,
    ) -> Self {
        Self {
            name,
            url,
            file_name,
            sha256,
            max_bytes,
        }
    }
}

impl DatasetHub {
    /// Creates a hub rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The default cache location.
    ///
    /// Resolution order: the `RSTORCH_DATA`
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
    ///
    /// # Errors
    ///
    /// Returns [`Error::Data`] unless `dataset` is one ordinary path
    /// component. This keeps the returned path rooted below this hub.
    pub fn dataset_dir(&self, dataset: &str) -> Result<PathBuf> {
        check_cache_component("dataset", dataset)?;
        Ok(self.dataset_dir_unchecked(dataset))
    }

    /// The full cache path for `resource` within `dataset`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Data`] unless both names are one ordinary path
    /// component. This keeps the returned path rooted below this hub.
    pub fn resource_path(&self, dataset: &str, resource: &DatasetResource) -> Result<PathBuf> {
        check_cache_key(dataset, resource.file_name)?;
        Ok(self.resource_path_unchecked(dataset, resource))
    }

    /// Whether `resource` is already present in the cache.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Data`] for a dataset or file name that is not one
    /// ordinary path component.
    pub fn is_cached(&self, dataset: &str, resource: &DatasetResource) -> Result<bool> {
        Ok(self.resource_path(dataset, resource)?.is_file())
    }

    fn dataset_dir_unchecked(&self, dataset: &str) -> PathBuf {
        self.root.join(dataset)
    }

    fn resource_path_unchecked(&self, dataset: &str, resource: &DatasetResource) -> PathBuf {
        self.dataset_dir_unchecked(dataset).join(resource.file_name)
    }

    /// Returns an existing cache path after re-checking its size and digest.
    ///
    /// This is intentionally private: callers that only need a path must use
    /// the fallible, name-validating public helpers, while offline readers use
    /// this stronger cache-hit boundary.
    fn verified_cached_path(&self, dataset: &str, resource: &DatasetResource) -> Result<PathBuf> {
        check_cache_key(dataset, resource.file_name)?;
        let path = self.resource_path_unchecked(dataset, resource);
        if !path.is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("cached resource `{}` is not present", resource.name),
            )
            .into());
        }
        resource.verify_cached(&path)?;
        Ok(path)
    }
}

impl DatasetHub {
    /// Ensures every resource is cached (downloading missing ones over the
    /// network), returning their paths in order.
    ///
    /// Available only with the `hub` feature; use
    /// [`ensure_cached_with`](Self::ensure_cached_with) for an offline,
    /// caller-supplied fetcher.
    ///
    /// # Errors
    ///
    /// As [`ensure_cached_with`](Self::ensure_cached_with).
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
    ///
    /// # Errors
    ///
    /// As [`ensure_resource_with`](Self::ensure_resource_with), for the first
    /// resource that fails.
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
    ///
    /// # Errors
    ///
    /// As [`ensure_resource_with`](Self::ensure_resource_with).
    #[cfg(feature = "hub")]
    pub fn ensure_resource(&self, dataset: &str, resource: &DatasetResource) -> Result<PathBuf> {
        self.ensure_resource_with(dataset, resource, fetch_resource)
    }

    /// Ensures a single resource is cached, fetching it with `fetch` if it is
    /// missing.
    ///
    /// A cached file is **re-verified** against the resource's size cap and
    /// SHA-256 digest before it is handed back. The download path verifies
    /// bytes before installing them, but nothing stops another writer from
    /// replacing a cached file afterwards, and everything downstream
    /// (gzip decompression, IDX parsing) treats these bytes as trusted. The
    /// digest is only recomputed for resources that declare one; a resource
    /// with neither a digest nor a cap is checked only for existence, because
    /// there is nothing to check it against.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Data`] if `dataset` or `resource.file_name` is not a
    /// single ordinary path component, or if a cached file fails its size or
    /// checksum re-verification. As [`download_with`](Self::download_with) if
    /// the resource is not yet cached.
    pub fn ensure_resource_with<F>(
        &self,
        dataset: &str,
        resource: &DatasetResource,
        fetch: F,
    ) -> Result<PathBuf>
    where
        F: FnMut(&DatasetResource) -> Result<Vec<u8>>,
    {
        check_cache_key(dataset, resource.file_name)?;
        let path = self.resource_path_unchecked(dataset, resource);
        if path.is_file() {
            resource.verify_cached(&path)?;
            return Ok(path);
        }
        self.download_with(dataset, resource, fetch)
    }

    /// Downloads `resource` over the network, replacing any cached copy.
    ///
    /// Available only with the `hub` feature; use
    /// [`download_with`](Self::download_with) for an offline, caller-supplied
    /// fetcher.
    ///
    /// # Errors
    ///
    /// As [`download_with`](Self::download_with).
    #[cfg(feature = "hub")]
    pub fn download(&self, dataset: &str, resource: &DatasetResource) -> Result<PathBuf> {
        self.download_with(dataset, resource, fetch_resource)
    }

    /// Fetches `resource` with `fetch`, verifies it, and atomically installs
    /// it into the cache, returning its path.
    ///
    /// The size cap and SHA-256 checksum (if any) are checked **before**
    /// anything touches the filesystem, so unverified bytes are never written.
    /// The install itself uses the same atomic-save path as the rest of the
    /// crate: a uniquely named temporary sibling, fsynced and renamed into
    /// place, and removed on any failure. So this function never *writes* an
    /// unverified file, and concurrent downloads of the same resource cannot
    /// collide on a shared temporary name.
    ///
    /// That is a guarantee about this call, not about the cache over time —
    /// anything else with write access to the cache directory can replace an
    /// installed file later, which is why
    /// [`ensure_resource_with`](Self::ensure_resource_with) re-verifies on a
    /// cache hit rather than trusting the path's mere existence.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Data`] if `dataset` or `resource.file_name` is not a
    /// single ordinary path component, if `fetch` fails, or if the fetched
    /// bytes fail their size cap or checksum check. Propagates any
    /// filesystem I/O error from creating the cache directory or installing
    /// the file.
    pub fn download_with<F>(
        &self,
        dataset: &str,
        resource: &DatasetResource,
        mut fetch: F,
    ) -> Result<PathBuf>
    where
        F: FnMut(&DatasetResource) -> Result<Vec<u8>>,
    {
        check_cache_key(dataset, resource.file_name)?;
        let bytes = fetch(resource)?;
        resource.verify(&bytes)?;

        let dir = self.dataset_dir_unchecked(dataset);
        fs::create_dir_all(&dir)?;
        let path = self.resource_path_unchecked(dataset, resource);
        crate::persist::atomic::write_atomic(&path, &bytes)?;
        Ok(path)
    }
}

impl DatasetResource {
    /// Checks `bytes` against this resource's size cap and expected digest.
    ///
    /// Note that the cap is also applied by the network fetcher while
    /// streaming, so an oversized body is refused before it is fully buffered;
    /// this re-check covers caller-supplied fetchers and keeps the error
    /// identical either way.
    fn verify(&self, bytes: &[u8]) -> Result<()> {
        self.check_size(bytes.len() as u64)?;
        if let Some(expected) = self.sha256 {
            let found = sha256::sha256_hex(bytes);
            if found != expected {
                return Err(Error::data(format!(
                    "checksum mismatch for `{}`: expected {expected}, found {found}",
                    self.name
                )));
            }
        }
        Ok(())
    }

    /// The size half of [`verify`](Self::verify), split out so the cached-file
    /// check can reject an oversized file from its length alone, without
    /// reading it.
    fn check_size(&self, len: u64) -> Result<()> {
        if let Some(max) = self.max_bytes
            && len > max
        {
            return Err(Error::data(format!(
                "resource `{}` exceeds size cap of {max} bytes (got {len})",
                self.name
            )));
        }
        Ok(())
    }

    /// Re-checks an already-installed cache file against this resource.
    ///
    /// The length is checked first and from metadata alone, so an oversized
    /// file is rejected without being read. The digest, when the resource
    /// declares one, then requires the bytes — bounded by the cap that just
    /// passed, or unbounded for a resource that declares no cap (in which case
    /// the caller has asked for no bound).
    fn verify_cached(&self, path: &Path) -> Result<()> {
        self.check_size(fs::metadata(path)?.len())?;
        if self.sha256.is_some() {
            self.verify(&fs::read(path)?)?;
        }
        Ok(())
    }
}

/// Rejects a `dataset` or `file_name` that is not a single ordinary path
/// component.
///
/// Both are joined onto the cache root, and every field of
/// [`DatasetResource`] is public, so a resource built from a manifest, a CLI
/// argument, or an index file could otherwise carry `..` or an absolute path
/// and place a downloaded file anywhere the process can write. Requiring
/// exactly one `Component::Normal` keeps every cache path inside the root.
fn check_cache_component(label: &str, value: &str) -> Result<()> {
    let mut components = Path::new(value).components();
    let single_normal =
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none();
    if !single_normal {
        return Err(Error::data(format!(
            "{label} `{value}` must be a single path component \
                 (no separators, `..`, or absolute paths)"
        )));
    }
    Ok(())
}

fn check_cache_key(dataset: &str, file_name: &str) -> Result<()> {
    check_cache_component("dataset", dataset)?;
    check_cache_component("file name", file_name)
}

/// How many bytes a fetch may read for a resource capped at `max_bytes`: one
/// past the cap, so an over-cap body is *detected* by
/// [`DatasetResource::verify`] rather than silently truncated to a valid-looking
/// prefix. Uncapped resources are unbounded.
///
/// Kept separate from the fetcher so the bound is testable without a network:
/// compiled for the `hub` feature (its caller) and for test builds (which
/// exercise it directly on a plain reader).
#[cfg(any(feature = "hub", test))]
fn read_limit(max_bytes: Option<u64>) -> u64 {
    // `saturating_add` keeps a `u64::MAX` cap from wrapping to 0, which would
    // otherwise turn the guard into a read of nothing.
    max_bytes.map_or(u64::MAX, |max| max.saturating_add(1))
}

/// The default network fetcher: an HTTP GET reading the body into memory.
///
/// The read is bounded by [`DatasetResource::max_bytes`] (plus one byte, so an
/// over-cap body is still detected rather than silently truncated). Without
/// that bound the cap could only be applied *after* an arbitrarily large
/// response had already been buffered, which would let a misbehaving or
/// compromised mirror exhaust memory before any check ran.
#[cfg(feature = "hub")]
fn fetch_resource(resource: &DatasetResource) -> Result<Vec<u8>> {
    use std::io::Read as _;
    let reader = ureq::get(resource.url)
        .call()
        .map_err(|source| Error::data_with(format!("failed to fetch `{}`", resource.name), source))?
        .into_reader();
    let mut bytes = Vec::new();
    reader
        .take(read_limit(resource.max_bytes))
        .read_to_end(&mut bytes)?;
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
        assert!(!root.join("fixture/sha.bin").exists());
        // Verification precedes any filesystem work, so a rejected download
        // must not even have created the dataset directory. Asserting on
        // `read_dir(...).unwrap_or_default()` here would be vacuous — the
        // directory does not exist, so the listing is empty no matter what the
        // code does. Assert the stronger, non-vacuous property instead.
        assert!(
            !root.join("fixture").exists(),
            "a rejected download must not create the dataset directory"
        );
        let _ = fs::remove_dir_all(root);
    }

    /// The leftover check that *is* meaningful: a download whose bytes verify
    /// but whose install fails must leave no temporary behind. The install is
    /// made to fail by occupying the destination path with a directory.
    #[test]
    fn a_failed_install_leaves_no_temporary_behind() {
        let root = scratch("install-fail");
        let _ = fs::remove_dir_all(&root);
        let hub = DatasetHub::new(&root);
        let resource = DatasetResource {
            name: "test",
            url: "https://example.invalid/test",
            file_name: "blocked.bin",
            sha256: None,
            max_bytes: None,
        };
        // A directory at the destination makes the final rename fail after the
        // bytes have already verified and been staged.
        fs::create_dir_all(root.join("fixture/blocked.bin")).unwrap();

        let result = hub.download_with("fixture", &resource, |_| Ok(vec![1, 2, 3]));
        assert!(result.is_err(), "install over a directory must fail");

        let leftovers: Vec<String> = fs::read_dir(root.join("fixture"))
            .expect("dataset directory exists on this path")
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name != "blocked.bin")
            .collect();
        assert!(
            leftovers.is_empty(),
            "temporary files left behind: {leftovers:?}"
        );
        let _ = fs::remove_dir_all(root);
    }

    /// `..` in a dataset name or file name would place a cache file outside the
    /// cache root. Every `DatasetResource` field is public, so this is
    /// reachable from a resource built out of a manifest or CLI argument.
    #[test]
    fn path_traversal_in_cache_keys_is_rejected() {
        let root = scratch("traversal");
        let _ = fs::remove_dir_all(&root);
        let hub = DatasetHub::new(&root);

        let escaping = DatasetResource {
            name: "test",
            url: "https://example.invalid/test",
            file_name: "../../escaped.bin",
            sha256: None,
            max_bytes: None,
        };
        assert!(
            matches!(
                hub.download_with("ds", &escaping, |_| Ok(vec![1])),
                Err(Error::Data { .. })
            ),
            "a traversing file name must be rejected"
        );
        assert!(
            matches!(
                hub.ensure_resource_with("ds", &escaping, |_| Ok(vec![1])),
                Err(Error::Data { .. })
            ),
            "ensure_resource_with must reject it too"
        );

        let ok = DatasetResource {
            file_name: "fine.bin",
            ..escaping
        };
        for bad in ["..", "a/b", "/abs", ""] {
            assert!(
                matches!(hub.dataset_dir(bad), Err(Error::Data { .. })),
                "dataset_dir must reject dataset name {bad:?}"
            );
            assert!(
                matches!(hub.resource_path(bad, &ok), Err(Error::Data { .. })),
                "resource_path must reject dataset name {bad:?}"
            );
            assert!(
                matches!(hub.is_cached(bad, &ok), Err(Error::Data { .. })),
                "is_cached must reject dataset name {bad:?}"
            );
            assert!(
                matches!(
                    hub.download_with(bad, &ok, |_| Ok(vec![1])),
                    Err(Error::Data { .. })
                ),
                "dataset name {bad:?} must be rejected"
            );
        }
        assert!(matches!(
            hub.resource_path("ds", &escaping),
            Err(Error::Data { .. })
        ));
        assert!(matches!(
            hub.is_cached("ds", &escaping),
            Err(Error::Data { .. })
        ));
        // The ordinary case still works.
        assert!(hub.download_with("ds", &ok, |_| Ok(vec![1])).is_ok());
        assert!(!root.join("../../escaped.bin").exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn dataset_resource_constructor_preserves_metadata() {
        let resource = DatasetResource::new(
            "name",
            "https://example.invalid/name",
            "name.bin",
            Some("digest"),
            Some(12),
        );
        assert_eq!(resource.name, "name");
        assert_eq!(resource.url, "https://example.invalid/name");
        assert_eq!(resource.file_name, "name.bin");
        assert_eq!(resource.sha256, Some("digest"));
        assert_eq!(resource.max_bytes, Some(12));
    }

    /// A cached file must be re-verified, not trusted for merely existing:
    /// anything with write access to the cache could otherwise substitute
    /// arbitrary bytes for a checksummed resource.
    #[test]
    fn a_tampered_cache_file_is_rejected_on_the_cache_hit_path() {
        let root = scratch("tamper");
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
        // A cache hit on the untouched file is fine and does not re-fetch.
        assert_eq!(
            hub.ensure_resource_with("fixture", &resource, |_| panic!("must not fetch"))
                .unwrap(),
            path
        );

        // Substitute different bytes at the installed path.
        fs::write(&path, b"tampered").unwrap();
        let err = hub
            .ensure_resource_with("fixture", &resource, |_| panic!("must not fetch"))
            .expect_err("a tampered cache file must be rejected");
        assert!(
            matches!(&err, Error::Data { msg, .. } if msg.contains("checksum mismatch")),
            "expected a checksum error, got {err:?}"
        );
        let _ = fs::remove_dir_all(root);
    }

    /// An oversized cached file is rejected from its length alone, so the cap
    /// still applies to bytes that arrived by some other route.
    #[test]
    fn an_oversized_cache_file_is_rejected_without_being_read() {
        let root = scratch("tamper-cap");
        let _ = fs::remove_dir_all(&root);
        let hub = DatasetHub::new(&root);
        let resource = DatasetResource {
            name: "test",
            url: "https://example.invalid/test",
            file_name: "capped.bin",
            sha256: None,
            max_bytes: Some(4),
        };
        let dir = root.join("fixture");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("capped.bin"), vec![0u8; 64]).unwrap();

        let err = hub
            .ensure_resource_with("fixture", &resource, |_| panic!("must not fetch"))
            .expect_err("an over-cap cache file must be rejected");
        assert!(
            matches!(&err, Error::Data { msg, .. } if msg.contains("size cap")),
            "expected a size-cap error, got {err:?}"
        );
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
        assert!(!root.join("fixture/cap.bin").exists());
        let _ = fs::remove_dir_all(root);
    }

    /// The streaming cap in `fetch_resource` reads at most `max_bytes + 1`, so
    /// an over-cap body is detected without buffering all of it. This pins the
    /// arithmetic on a plain reader, with no network involved.
    #[test]
    fn streaming_cap_reads_one_byte_past_the_limit() {
        use std::io::Read as _;

        let body = vec![7u8; 4096];
        let max: u64 = 8;
        let mut got = Vec::new();
        body.as_slice()
            .take(read_limit(Some(max)))
            .read_to_end(&mut got)
            .unwrap();

        assert_eq!(got.len() as u64, max + 1, "must stop just past the cap");
        let resource = DatasetResource {
            name: "test",
            url: "https://example.invalid/test",
            file_name: "s.bin",
            sha256: None,
            max_bytes: Some(max),
        };
        assert!(
            matches!(resource.verify(&got), Err(Error::Data { .. })),
            "the extra byte must make the cap check fail"
        );
        // A body exactly at the cap is still accepted.
        assert!(resource.verify(&body[..max as usize]).is_ok());
    }

    /// A `u64::MAX` cap must not wrap the `+ 1` to zero (which would read
    /// nothing at all), and an uncapped resource stays unbounded.
    #[test]
    fn read_limit_is_saturating_and_unbounded_when_uncapped() {
        assert_eq!(read_limit(Some(u64::MAX)), u64::MAX);
        assert_eq!(read_limit(None), u64::MAX);
        assert_eq!(read_limit(Some(0)), 1);
        assert_eq!(read_limit(Some(10)), 11);
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
        assert!(hub.is_cached("fixture", &resource).unwrap());
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
