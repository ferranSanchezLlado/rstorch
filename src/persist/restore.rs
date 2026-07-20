//! Staged, all-or-nothing restore surface (16.14 WS5).
//!
//! Loading a checkpoint into a live model must be transactional: if *any*
//! parameter fails to match, the model, optimizer, RNG, and caches must be
//! left exactly as they were. The failure mode this prevents is a half-loaded
//! model — some weights swapped, some not — which silently produces wrong
//! results.
//!
//! This host-layer type does the fallible half: it validates a loaded tensor
//! map against a caller-declared **schema** (expected path → dtype + dims)
//! under a [`LoadOptions`] policy, and either fails wholesale or hands back a
//! [`StagedTensors`] value in which *every* expected tensor is present and
//! shape/dtype-checked. The caller (the `nn`/`optim` runtime, T40/T44) then
//! performs the actual, infallible swaps from that staged value — the
//! "commit through no-fail swaps only after all staging succeeds" step.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::persist::host_tensor::HostTensor;
use crate::persist::options::{LoadOptions, MissingPolicy, UnexpectedPolicy};
use std::collections::BTreeMap;

/// One entry of the target's expected schema: its path and the dtype/dims a
/// loaded tensor must match to be accepted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expected {
    /// The dotted path the target parameter/buffer lives at.
    pub path: String,
    /// The dtype the loaded tensor must have.
    pub dtype: DType,
    /// The dimensions the loaded tensor must have.
    pub dims: Vec<usize>,
}

impl Expected {
    /// Convenience constructor.
    pub fn new(path: impl Into<String>, dtype: DType, dims: Vec<usize>) -> Self {
        Self {
            path: path.into(),
            dtype,
            dims,
        }
    }
}

/// The fully-validated result of staging: exactly the expected paths that were
/// present in the file, each paired with a shape/dtype-checked [`HostTensor`],
/// in the schema's declared order.
///
/// Holding a `StagedTensors` is the guarantee that validation *fully*
/// succeeded — so the caller's subsequent swaps cannot fail on a mismatch and
/// leave the target half-updated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StagedTensors {
    entries: Vec<(String, HostTensor)>,
}

impl StagedTensors {
    /// The staged `(path, tensor)` pairs, in the schema's declared order.
    pub fn entries(&self) -> &[(String, HostTensor)] {
        &self.entries
    }

    /// Number of staged tensors.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing was staged.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Consume into the owned `(path, tensor)` pairs for committing.
    pub fn into_entries(self) -> Vec<(String, HostTensor)> {
        self.entries
    }
}

/// Validate `loaded` against `schema` under `options`, staging every match.
///
/// Returns [`StagedTensors`] only if **all** validation passes:
///
/// - every present tensor whose path is expected matches the expected dtype
///   and dims (a mismatch is always an error, regardless of policy);
/// - paths expected but absent obey [`MissingPolicy`];
/// - paths present but unexpected obey [`UnexpectedPolicy`].
///
/// On any failure it returns an [`Error::Persistence`] and stages nothing —
/// the caller has not been handed a partial result, so it commits nothing.
pub fn stage(
    schema: &[Expected],
    loaded: &BTreeMap<String, HostTensor>,
    options: &LoadOptions,
) -> Result<StagedTensors> {
    // Detect unexpected paths first (cheap, and the loudest signal of a schema
    // mismatch) before touching any data.
    if options.unexpected == UnexpectedPolicy::Reject {
        let expected_paths: std::collections::BTreeSet<&str> =
            schema.iter().map(|e| e.path.as_str()).collect();
        for name in loaded.keys() {
            if !expected_paths.contains(name.as_str()) {
                return Err(Error::Persistence {
                    msg: format!("unexpected tensor `{name}` in file (not in target schema)"),
                });
            }
        }
    }

    let mut entries = Vec::with_capacity(schema.len());
    for exp in schema {
        match loaded.get(&exp.path) {
            Some(tensor) => {
                if tensor.dtype() != exp.dtype {
                    return Err(Error::Persistence {
                        msg: format!(
                            "tensor `{}` dtype mismatch: file has {}, target expects {}",
                            exp.path,
                            tensor.dtype(),
                            exp.dtype
                        ),
                    });
                }
                if tensor.dims() != exp.dims.as_slice() {
                    return Err(Error::Persistence {
                        msg: format!(
                            "tensor `{}` shape mismatch: file has {:?}, target expects {:?}",
                            exp.path,
                            tensor.dims(),
                            exp.dims
                        ),
                    });
                }
                entries.push((exp.path.clone(), tensor.clone()));
            }
            None => {
                if options.missing == MissingPolicy::Reject {
                    return Err(Error::Persistence {
                        msg: format!("missing tensor `{}` (expected by target)", exp.path),
                    });
                }
                // allow-missing: leave the target's current value; do not stage.
            }
        }
    }

    Ok(StagedTensors { entries })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ht(dtype: DType, dims: Vec<usize>) -> HostTensor {
        let n: usize = dims.iter().product::<usize>() * dtype.size_in_bytes();
        HostTensor::from_bytes(dtype, dims, vec![0u8; n]).unwrap()
    }

    fn loaded() -> BTreeMap<String, HostTensor> {
        let mut m = BTreeMap::new();
        m.insert("fc.weight".to_string(), ht(DType::F32, vec![4, 3]));
        m.insert("fc.bias".to_string(), ht(DType::F32, vec![4]));
        m
    }

    fn schema() -> Vec<Expected> {
        vec![
            Expected::new("fc.weight", DType::F32, vec![4, 3]),
            Expected::new("fc.bias", DType::F32, vec![4]),
        ]
    }

    #[test]
    fn full_match_stages_everything_in_order() {
        let staged = stage(&schema(), &loaded(), &LoadOptions::strict()).unwrap();
        assert_eq!(staged.len(), 2);
        let paths: Vec<_> = staged.entries().iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(paths, vec!["fc.weight", "fc.bias"]);
    }

    #[test]
    fn missing_rejected_by_default_stages_nothing() {
        let mut l = loaded();
        l.remove("fc.bias");
        let err = stage(&schema(), &l, &LoadOptions::strict());
        assert!(matches!(err, Err(Error::Persistence { .. })));
    }

    #[test]
    fn missing_allowed_skips_absent() {
        let mut l = loaded();
        l.remove("fc.bias");
        let staged = stage(&schema(), &l, &LoadOptions::strict().allow_missing()).unwrap();
        assert_eq!(staged.len(), 1);
        assert_eq!(staged.entries()[0].0, "fc.weight");
    }

    #[test]
    fn unexpected_rejected_by_default() {
        let mut l = loaded();
        l.insert("extra".to_string(), ht(DType::I64, vec![2]));
        let err = stage(&schema(), &l, &LoadOptions::strict());
        assert!(matches!(err, Err(Error::Persistence { .. })));
    }

    #[test]
    fn unexpected_allowed_ignores_extra() {
        let mut l = loaded();
        l.insert("extra".to_string(), ht(DType::I64, vec![2]));
        let staged = stage(&schema(), &l, &LoadOptions::strict().allow_unexpected()).unwrap();
        assert_eq!(staged.len(), 2);
    }

    #[test]
    fn dtype_mismatch_is_always_an_error() {
        let mut l = loaded();
        l.insert("fc.bias".to_string(), ht(DType::I64, vec![4]));
        // Even with the most permissive policies, a dtype mismatch fails.
        let opts = LoadOptions::strict().allow_missing().allow_unexpected();
        let err = stage(&schema(), &l, &opts);
        assert!(matches!(err, Err(Error::Persistence { .. })));
    }

    #[test]
    fn shape_mismatch_is_always_an_error() {
        let mut l = loaded();
        l.insert("fc.weight".to_string(), ht(DType::F32, vec![4, 5]));
        let err = stage(&schema(), &l, &LoadOptions::strict());
        assert!(matches!(err, Err(Error::Persistence { .. })));
    }
}
