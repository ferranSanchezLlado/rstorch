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
#[non_exhaustive]
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

    /// Construct limits with the safe defaults (see field docs).
    pub fn new() -> Self {
        Self::defaults()
    }

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

    /// Set the maximum metadata/header size.
    #[must_use]
    pub fn with_max_metadata_bytes(mut self, value: u64) -> Self {
        self.max_metadata_bytes = value;
        self
    }

    /// Set the maximum number of tensor records.
    #[must_use]
    pub fn with_max_records(mut self, value: u64) -> Self {
        self.max_records = value;
        self
    }

    /// Set the maximum tensor rank.
    #[must_use]
    pub fn with_max_rank(mut self, value: u64) -> Self {
        self.max_rank = value;
        self
    }

    /// Set the maximum size of an individual tensor.
    #[must_use]
    pub fn with_max_tensor_bytes(mut self, value: u64) -> Self {
        self.max_tensor_bytes = value;
        self
    }

    /// Set the maximum total tensor-data budget.
    #[must_use]
    pub fn with_max_total_bytes(mut self, value: u64) -> Self {
        self.max_total_bytes = value;
        self
    }

    /// Set the maximum tensor-name/path string size.
    #[must_use]
    pub fn with_max_string_bytes(mut self, value: u64) -> Self {
        self.max_string_bytes = value;
        self
    }

    /// Return the maximum metadata/header size.
    pub fn max_metadata_bytes(&self) -> u64 {
        self.max_metadata_bytes
    }

    /// Return the maximum number of tensor records.
    pub fn max_records(&self) -> u64 {
        self.max_records
    }

    /// Return the maximum tensor rank.
    pub fn max_rank(&self) -> u64 {
        self.max_rank
    }

    /// Return the maximum size of an individual tensor.
    pub fn max_tensor_bytes(&self) -> u64 {
        self.max_tensor_bytes
    }

    /// Return the maximum total tensor-data budget.
    pub fn max_total_bytes(&self) -> u64 {
        self.max_total_bytes
    }

    /// Return the maximum tensor-name/path string size.
    pub fn max_string_bytes(&self) -> u64 {
        self.max_string_bytes
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
#[non_exhaustive]
pub struct LoadOptions {
    /// Untrusted-input size caps.
    pub limits: Limits,
    /// Policy for expected-but-absent paths.
    pub missing: MissingPolicy,
    /// Policy for present-but-unexpected paths.
    pub unexpected: UnexpectedPolicy,
}

impl LoadOptions {
    /// Construct options from explicit limits and path policies.
    pub fn new(limits: Limits, missing: MissingPolicy, unexpected: UnexpectedPolicy) -> Self {
        Self {
            limits,
            missing,
            unexpected,
        }
    }

    /// The strictest, safest options: default limits, and both a missing and
    /// an unexpected path are errors. This is the recommended default for
    /// loading your own checkpoints where the schema must match exactly.
    pub fn strict() -> Self {
        Self::new(
            Limits::defaults(),
            MissingPolicy::Reject,
            UnexpectedPolicy::Reject,
        )
    }

    /// Set the untrusted-input limits.
    #[must_use]
    pub fn with_limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    /// Set the policy for expected-but-absent paths.
    #[must_use]
    pub fn with_missing(mut self, missing: MissingPolicy) -> Self {
        self.missing = missing;
        self
    }

    /// Set the policy for present-but-unexpected paths.
    #[must_use]
    pub fn with_unexpected(mut self, unexpected: UnexpectedPolicy) -> Self {
        self.unexpected = unexpected;
        self
    }

    /// Allow paths the target expects but the file omits.
    #[must_use]
    pub fn allow_missing(self) -> Self {
        self.with_missing(MissingPolicy::Allow)
    }

    /// Allow paths present in the file but not expected by the target.
    #[must_use]
    pub fn allow_unexpected(self) -> Self {
        self.with_unexpected(UnexpectedPolicy::Allow)
    }

    /// Return the configured untrusted-input limits.
    pub fn limits(&self) -> Limits {
        self.limits
    }

    /// Return the configured missing-path policy.
    pub fn missing(&self) -> MissingPolicy {
        self.missing
    }

    /// Return the configured unexpected-path policy.
    pub fn unexpected(&self) -> UnexpectedPolicy {
        self.unexpected
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
        assert_eq!(o.missing(), MissingPolicy::Reject);
        assert_eq!(o.unexpected(), UnexpectedPolicy::Reject);
        assert_eq!(o.limits(), Limits::defaults());
    }

    #[test]
    fn builders_relax_policies() {
        let o = LoadOptions::strict().allow_missing().allow_unexpected();
        assert_eq!(o.missing(), MissingPolicy::Allow);
        assert_eq!(o.unexpected(), UnexpectedPolicy::Allow);
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
    fn limits_builders_customize_each_cap() {
        let limits = Limits::new()
            .with_max_metadata_bytes(1)
            .with_max_records(2)
            .with_max_rank(3)
            .with_max_tensor_bytes(4)
            .with_max_total_bytes(5)
            .with_max_string_bytes(6);
        assert_eq!(limits.max_metadata_bytes(), 1);
        assert_eq!(limits.max_records(), 2);
        assert_eq!(limits.max_rank(), 3);
        assert_eq!(limits.max_tensor_bytes(), 4);
        assert_eq!(limits.max_total_bytes(), 5);
        assert_eq!(limits.max_string_bytes(), 6);
    }

    #[test]
    fn with_limits_tightens_without_touching_policies() {
        let tight = Limits::defaults().with_max_tensor_bytes(1024);
        let o = LoadOptions::strict().allow_missing().with_limits(tight);
        assert_eq!(o.limits(), tight);
        assert_eq!(o.limits().max_tensor_bytes(), 1024);
        // The other caps come from the base the caller built on, and the
        // policies set before it are untouched.
        assert_eq!(o.limits().max_records(), Limits::defaults().max_records());
        assert_eq!(o.missing(), MissingPolicy::Allow);
        assert_eq!(o.unexpected(), UnexpectedPolicy::Reject);
    }

    #[test]
    fn load_options_constructor_and_policy_builders_replace_policies() {
        let options =
            LoadOptions::new(Limits::new(), MissingPolicy::Allow, UnexpectedPolicy::Allow);
        assert_eq!(options.limits(), Limits::defaults());
        assert_eq!(options.missing(), MissingPolicy::Allow);
        assert_eq!(options.unexpected(), UnexpectedPolicy::Allow);

        let options = LoadOptions::strict()
            .with_missing(MissingPolicy::Allow)
            .with_unexpected(UnexpectedPolicy::Allow);
        assert_eq!(options.missing(), MissingPolicy::Allow);
        assert_eq!(options.unexpected(), UnexpectedPolicy::Allow);
    }
}
