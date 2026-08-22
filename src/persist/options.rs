//! Reader limits and load policies for untrusted files.
//!
//! Persistence readers must never trust a file's self-declared sizes: a
//! hostile or corrupt file can claim a petabyte tensor to make the reader
//! allocate itself to death. [`Limits`] carries the safe defaults every read
//! enforces *before* allocating, and [`LoadOptions`] pairs them with the
//! path-matching policies for the staged restore [`stage`](crate::persist::stage).
//!
//! Loading policies are expressed with enums, not booleans: a
//! call site reads `MissingPolicy::Allow` far more clearly than `true`.

/// What to do when the file is **missing** a tensor the target expects.
///
/// `#[non_exhaustive]`: the policy set is crate-owned and may grow in a minor
/// release (a warn-and-continue policy is the obvious next one), so downstream
/// `match` arms must carry a `_` catch-all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MissingPolicy {
    /// Every expected path must be present, else [`Error::Persistence`](crate::Error::Persistence).
    Reject,
    /// Absent paths are left at their current value.
    Allow,
}

/// What to do when the file carries a tensor the target does **not** expect.
///
/// `#[non_exhaustive]`: the policy set is crate-owned and may grow in a minor
/// release, so downstream `match` arms must carry a `_` catch-all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnexpectedPolicy {
    /// Any extra path is an error ([`Error::Persistence`](crate::Error::Persistence)).
    Reject,
    /// Extra paths are ignored.
    Allow,
}

/// Safe-default size caps enforced on every read of an untrusted file.
///
/// Every field is a hard upper bound the reader checks against the file's
/// self-declared sizes **before** allocating the corresponding buffer.
/// Defaults are intentionally generous enough for
/// real models yet finite. Writers validate against the *same* limits (the
/// writer-side check in [`save_safetensors`](crate::persist::save_safetensors))
/// so the library can never write a file its own default reader would reject
/// (the reader and the writer enforce the same limits).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// Maximum bytes of safetensors JSON header (metadata section).
    pub max_metadata_bytes: u64,
    /// Maximum number of tensor records in one file.
    pub max_records: u64,
    /// Maximum rank (number of dimensions) of any one tensor.
    pub max_rank: u64,
    /// Maximum bytes of any single tensor's data.
    pub max_tensor_bytes: u64,
    /// Maximum bytes summed across *all* tensor data (the total allocation
    /// budget).
    pub max_total_bytes: u64,
    /// Maximum length in bytes of any tensor name / path string.
    pub max_string_bytes: u64,
}

impl Limits {
    const DEFAULT_MAX_METADATA_BYTES: u64 = 16 * 1024 * 1024;
    const DEFAULT_MAX_RECORDS: u64 = 1_000_000;
    const DEFAULT_MAX_RANK: u64 = 64;
    const DEFAULT_MAX_TENSOR_BYTES: u64 = 4 * 1024 * 1024 * 1024;
    const DEFAULT_MAX_TOTAL_BYTES: u64 = 16 * 1024 * 1024 * 1024;
    const DEFAULT_MAX_STRING_BYTES: u64 = 1 << 20;

    /// The safe defaults (see field docs).
    pub fn defaults() -> Self {
        Self {
            max_metadata_bytes: Self::DEFAULT_MAX_METADATA_BYTES,
            max_records: Self::DEFAULT_MAX_RECORDS,
            max_rank: Self::DEFAULT_MAX_RANK,
            max_tensor_bytes: Self::DEFAULT_MAX_TENSOR_BYTES,
            max_total_bytes: Self::DEFAULT_MAX_TOTAL_BYTES,
            max_string_bytes: Self::DEFAULT_MAX_STRING_BYTES,
        }
    }

    /// Check a single declared value against a bound, producing a uniform
    /// [`Error::Persistence`](crate::Error::Persistence) naming the field.
    pub(crate) fn check(field: &str, value: u64, max: u64) -> crate::error::Result<()> {
        if value > max {
            return Err(crate::error::Error::persistence(format!(
                "{field} {value} exceeds limit {max}"
            )));
        }
        Ok(())
    }
}

impl Default for Limits {
    fn default() -> Self {
        Self::defaults()
    }
}

/// Options controlling how a persisted file is read into a target.
///
/// Combines the untrusted-input [`Limits`] with the path-matching policies for
/// the staged restore [`stage`](crate::persist::stage). Constructed via
/// [`LoadOptions::strict`] (the recommended default: reject missing *and*
/// unexpected paths) then relaxed field-by-field with the builder methods.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadOptions {
    /// Untrusted-input size caps.
    pub limits: Limits,
    /// Policy for expected-but-absent paths.
    pub missing: MissingPolicy,
    /// Policy for present-but-unexpected paths.
    pub unexpected: UnexpectedPolicy,
}

impl LoadOptions {
    /// The strictest, safest options: default limits, and both a missing and
    /// an unexpected path are errors. This is the recommended default for
    /// loading your own checkpoints where the schema must match exactly.
    pub fn strict() -> Self {
        Self {
            limits: Limits::defaults(),
            missing: MissingPolicy::Reject,
            unexpected: UnexpectedPolicy::Reject,
        }
    }

    /// Allow paths the target expects but the file omits.
    #[must_use]
    pub fn allow_missing(mut self) -> Self {
        self.missing = MissingPolicy::Allow;
        self
    }

    /// Allow paths present in the file but not expected by the target.
    #[must_use]
    pub fn allow_unexpected(mut self) -> Self {
        self.unexpected = UnexpectedPolicy::Allow;
        self
    }

    /// Replace the untrusted-input limits (rarely needed; the defaults are
    /// generous). Use to *tighten* limits for genuinely untrusted files.
    #[must_use]
    pub fn with_limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self::strict()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_rejects_both() {
        let o = LoadOptions::strict();
        assert_eq!(o.missing, MissingPolicy::Reject);
        assert_eq!(o.unexpected, UnexpectedPolicy::Reject);
        assert_eq!(o.limits, Limits::defaults());
    }

    #[test]
    fn builders_relax_policies() {
        let o = LoadOptions::strict().allow_missing().allow_unexpected();
        assert_eq!(o.missing, MissingPolicy::Allow);
        assert_eq!(o.unexpected, UnexpectedPolicy::Allow);
    }

    #[test]
    fn check_enforces_bounds() {
        assert!(Limits::check("x", 5, 10).is_ok());
        assert!(Limits::check("x", 11, 10).is_err());
        // exactly at the bound is allowed.
        assert!(Limits::check("x", 10, 10).is_ok());
    }

    #[test]
    fn default_is_strict() {
        assert_eq!(LoadOptions::default(), LoadOptions::strict());
    }

    #[test]
    fn with_limits_tightens_without_touching_policies() {
        let tight = Limits {
            max_tensor_bytes: 1024,
            ..Limits::defaults()
        };
        let o = LoadOptions::strict().allow_missing().with_limits(tight);
        assert_eq!(o.limits, tight);
        assert_eq!(o.limits.max_tensor_bytes, 1024);
        // The other caps come from the base the caller built on, and the
        // policies set before it are untouched.
        assert_eq!(o.limits.max_records, Limits::defaults().max_records);
        assert_eq!(o.missing, MissingPolicy::Allow);
        assert_eq!(o.unexpected, UnexpectedPolicy::Reject);
    }
}
