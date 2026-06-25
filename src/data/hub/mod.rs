use crate::error::{DataError, Result};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

pub mod mnist;

pub use mnist::{Mnist, MnistCollate, MnistImageBatch, MnistImageCollate, MnistSample, MnistSplit};

#[derive(Debug, Clone)]
pub struct DatasetHub {
    root: PathBuf,
}

#[derive(Debug, Clone, Copy)]
pub struct DatasetResource {
    pub name: &'static str,
    pub url: &'static str,
    pub file_name: &'static str,
}

impl DatasetHub {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn default_cache() -> Self {
        let root = std::env::var_os("RSTORCH_DATA")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/rstorch"))
            })
            .unwrap_or_else(|| PathBuf::from("data"));
        Self::new(root)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn dataset_dir(&self, dataset: &str) -> PathBuf {
        self.root.join(dataset)
    }

    pub fn resource_path(&self, dataset: &str, resource: &DatasetResource) -> PathBuf {
        self.dataset_dir(dataset).join(resource.file_name)
    }

    pub fn is_cached(&self, dataset: &str, resource: &DatasetResource) -> bool {
        self.resource_path(dataset, resource).is_file()
    }

    pub fn ensure_cached(
        &self,
        dataset: &str,
        resources: &[DatasetResource],
    ) -> Result<Vec<PathBuf>> {
        self.ensure_cached_with(dataset, resources, fetch_resource)
    }

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

    pub fn ensure_resource(&self, dataset: &str, resource: &DatasetResource) -> Result<PathBuf> {
        self.ensure_resource_with(dataset, resource, fetch_resource)
    }

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

    pub fn download(&self, dataset: &str, resource: &DatasetResource) -> Result<PathBuf> {
        self.download_with(dataset, resource, fetch_resource)
    }

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
        fs::create_dir_all(&dir).map_err(|source| DataError::Io { source })?;
        let path = dir.join(resource.file_name);
        let tmp = path.with_extension("tmp");

        let bytes = fetch(resource)?;
        fs::write(&tmp, bytes).map_err(|source| DataError::Io { source })?;
        fs::rename(&tmp, &path).map_err(|source| DataError::Io { source })?;
        Ok(path)
    }
}

fn fetch_resource(resource: &DatasetResource) -> Result<Vec<u8>> {
    let mut response = ureq::get(resource.url)
        .call()
        .map_err(|source| DataError::Parse {
            source: Box::new(source),
        })?
        .into_reader();
    let mut bytes = Vec::new();
    response
        .read_to_end(&mut bytes)
        .map_err(|source| DataError::Io { source })?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn download_with_uses_supplied_fetcher() {
        let root = std::env::temp_dir().join(format!("rstorch-hub-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        let hub = DatasetHub::new(&root);
        let resource = DatasetResource {
            name: "test",
            url: "https://example.invalid/test",
            file_name: "test.bin",
        };

        let path = hub
            .download_with("fixture", &resource, |_| Ok(vec![1, 2, 3]))
            .unwrap();

        assert_eq!(fs::read(path).unwrap(), vec![1, 2, 3]);
        let _ = fs::remove_dir_all(root);
    }
}
