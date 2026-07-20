//! A minimal versioned checkpoint envelope (16.14 WS2).
//!
//! A full training checkpoint is more than a bag of tensors: it also carries
//! model **config**, **optimizer** hyperparameters/step clocks, and **RNG**
//! state so a run can be reconstructed and resumed. Rather than invent a
//! second binary format, the envelope rides *inside* a safetensors file:
//!
//! - tensor payloads (model weights, optimizer moment buffers) are ordinary
//!   safetensors tensors, keyed by their dotted path (a `String`);
//! - the non-tensor sections live as string entries in the safetensors
//!   free-form `__metadata__` map, under a reserved `rstorch.*` namespace;
//! - a format-family magic and a major/minor version let readers dispatch by
//!   version and reject unknown families/majors loudly.
//!
//! The magic is independent of any internal epoch number, and the reader is
//! permanent for every shipped major version (16.14 WS2). Section *values* are
//! opaque strings here — the `optim`/`nn` layers decide their own encoding
//! (T40/T44); the envelope only guarantees they round-trip verbatim.

use crate::error::{Error, Result};
use crate::persist::host_tensor::HostTensor;
use crate::persist::options::Limits;
use crate::persist::safetensors_io::{load_tensors, save_tensors};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

/// Format-family magic, independent of internal epoch numbers.
const MAGIC: &str = "rstorch-checkpoint";
/// Current envelope major version. Bumped only on incompatible changes; a
/// permanent reader is kept for every shipped major.
pub const FORMAT_MAJOR: u32 = 1;
/// Current envelope minor version. Bumped for additive, back-compatible
/// changes a `FORMAT_MAJOR` reader can ignore.
pub const FORMAT_MINOR: u32 = 0;

const KEY_MAGIC: &str = "rstorch.magic";
const KEY_MAJOR: &str = "rstorch.format_major";
const KEY_MINOR: &str = "rstorch.format_minor";
const SECTION_PREFIX: &str = "rstorch.section.";

/// A versioned checkpoint: a map of tensors plus named non-tensor sections.
///
/// Build one with [`Envelope::new`], attach tensors and sections, then
/// [`Envelope::save`] atomically. [`Envelope::load`] validates the magic and
/// dispatches on the major version before returning the parsed envelope.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Envelope {
    tensors: BTreeMap<String, HostTensor>,
    sections: BTreeMap<String, String>,
    major: u32,
    minor: u32,
}

impl Envelope {
    /// A new, empty envelope stamped with the current format version.
    pub fn new() -> Self {
        Self {
            tensors: BTreeMap::new(),
            sections: BTreeMap::new(),
            major: FORMAT_MAJOR,
            minor: FORMAT_MINOR,
        }
    }

    /// The envelope's format `(major, minor)` version.
    pub fn version(&self) -> (u32, u32) {
        (self.major, self.minor)
    }

    /// Insert or replace a tensor payload under `path`.
    pub fn insert_tensor(&mut self, path: impl Into<String>, tensor: HostTensor) {
        self.tensors.insert(path.into(), tensor);
    }

    /// The tensor payloads (model + optimizer buffers), keyed by path.
    pub fn tensors(&self) -> &BTreeMap<String, HostTensor> {
        &self.tensors
    }

    /// Look up a tensor payload by path.
    pub fn tensor(&self, path: &str) -> Option<&HostTensor> {
        self.tensors.get(path)
    }

    /// Insert or replace a non-tensor section (e.g. `"config"`, `"optimizer"`,
    /// `"rng"`). The value is an opaque string the caller owns the meaning of.
    ///
    /// Errors with [`Error::Persistence`] if `name` contains a `.` (which
    /// would collide with the reserved key namespace) or is empty.
    pub fn set_section(&mut self, name: impl Into<String>, value: impl Into<String>) -> Result<()> {
        let name = name.into();
        if name.is_empty() || name.contains('.') {
            return Err(Error::Persistence {
                msg: format!(
                    "invalid section name `{name}` (must be non-empty and contain no `.`)"
                ),
            });
        }
        self.sections.insert(name, value.into());
        Ok(())
    }

    /// The value of a named section, if present.
    pub fn section(&self, name: &str) -> Option<&str> {
        self.sections.get(name).map(String::as_str)
    }

    /// All section names, sorted.
    pub fn section_names(&self) -> Vec<&str> {
        self.sections.keys().map(String::as_str).collect()
    }

    /// Assemble the safetensors `__metadata__` map: magic, version, and each
    /// section under the reserved prefix.
    fn metadata_map(&self) -> HashMap<String, String> {
        let mut m = HashMap::new();
        m.insert(KEY_MAGIC.to_string(), MAGIC.to_string());
        m.insert(KEY_MAJOR.to_string(), self.major.to_string());
        m.insert(KEY_MINOR.to_string(), self.minor.to_string());
        for (name, value) in &self.sections {
            m.insert(format!("{SECTION_PREFIX}{name}"), value.clone());
        }
        m
    }

    /// Atomically save the envelope to `path`, validating writer `limits`.
    pub fn save(&self, path: impl AsRef<Path>, limits: &Limits) -> Result<()> {
        save_tensors(
            path.as_ref(),
            &self.tensors,
            Some(self.metadata_map()),
            limits,
        )
    }

    /// Load and version-dispatch an envelope from `path`, enforcing `limits`.
    ///
    /// Rejects a file whose magic is not this format family, or whose major
    /// version this reader does not understand ([`Error::Persistence`]).
    pub fn load(path: impl AsRef<Path>, limits: &Limits) -> Result<Self> {
        let (tensors, meta) = load_tensors(path.as_ref(), limits)?;

        let magic = meta.get(KEY_MAGIC).map(String::as_str);
        if magic != Some(MAGIC) {
            return Err(Error::Persistence {
                msg: format!(
                    "not an rstorch checkpoint: magic {magic:?} (expected {MAGIC:?}); \
                     use load_tensors for a plain safetensors file"
                ),
            });
        }

        let major = parse_version(&meta, KEY_MAJOR)?;
        let minor = parse_version(&meta, KEY_MINOR)?;
        // Version dispatch: this reader understands exactly FORMAT_MAJOR. A
        // higher major means an incompatible future format; refuse it loudly
        // rather than misread it.
        if major != FORMAT_MAJOR {
            return Err(Error::Persistence {
                msg: format!(
                    "unsupported checkpoint format major {major} (this build reads major {FORMAT_MAJOR})"
                ),
            });
        }

        let mut sections = BTreeMap::new();
        for (key, value) in &meta {
            if let Some(name) = key.strip_prefix(SECTION_PREFIX) {
                sections.insert(name.to_string(), value.clone());
            }
        }

        Ok(Self {
            tensors,
            sections,
            major,
            minor,
        })
    }
}

/// Parse a reserved `u32` version key, erroring on absence or non-numeric.
fn parse_version(meta: &HashMap<String, String>, key: &str) -> Result<u32> {
    let raw = meta.get(key).ok_or_else(|| Error::Persistence {
        msg: format!("checkpoint missing required `{key}`"),
    })?;
    raw.parse::<u32>().map_err(|_| Error::Persistence {
        msg: format!("checkpoint `{key}` is not a number: {raw:?}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use std::path::PathBuf;

    fn tmpdir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rstorch-persist-env-{}-{}-{}",
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

    fn envelope() -> Envelope {
        let mut e = Envelope::new();
        e.insert_tensor(
            "model.weight",
            HostTensor::from_bytes(DType::F32, vec![2], 1.5f32.to_le_bytes().repeat(2)).unwrap(),
        );
        e.set_section("config", "{\"hidden\":16}").unwrap();
        e.set_section("optimizer", "adamw;lr=0.001;step=42")
            .unwrap();
        e.set_section("rng", "seed=7;state=abcdef").unwrap();
        e
    }

    #[test]
    fn round_trip_tensors_and_sections() {
        let dir = tmpdir("rt");
        let path = dir.join("run.rstorch");
        let e = envelope();
        e.save(&path, &Limits::defaults()).unwrap();

        let back = Envelope::load(&path, &Limits::defaults()).unwrap();
        assert_eq!(back, e);
        assert_eq!(back.version(), (FORMAT_MAJOR, FORMAT_MINOR));
        assert_eq!(back.section("config"), Some("{\"hidden\":16}"));
        assert_eq!(back.section("optimizer"), Some("adamw;lr=0.001;step=42"));
        assert_eq!(back.section("rng"), Some("seed=7;state=abcdef"));
        assert_eq!(back.section_names(), vec!["config", "optimizer", "rng"]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn plain_safetensors_is_not_a_checkpoint() {
        let dir = tmpdir("plain");
        let path = dir.join("m.safetensors");
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "w".to_string(),
            HostTensor::from_bytes(DType::F32, vec![1], 0.0f32.to_le_bytes().to_vec()).unwrap(),
        );
        save_tensors(&path, &tensors, None, &Limits::defaults()).unwrap();

        assert!(matches!(
            Envelope::load(&path, &Limits::defaults()),
            Err(Error::Persistence { .. })
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn unknown_major_is_rejected() {
        let dir = tmpdir("major");
        let path = dir.join("run.rstorch");
        // Hand-build metadata with a future major version.
        let mut meta = HashMap::new();
        meta.insert(KEY_MAGIC.to_string(), MAGIC.to_string());
        meta.insert(KEY_MAJOR.to_string(), (FORMAT_MAJOR + 1).to_string());
        meta.insert(KEY_MINOR.to_string(), "0".to_string());
        let tensors: BTreeMap<String, HostTensor> = BTreeMap::new();
        save_tensors(&path, &tensors, Some(meta), &Limits::defaults()).unwrap();

        let err = Envelope::load(&path, &Limits::defaults());
        assert!(matches!(err, Err(Error::Persistence { .. })));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn newer_minor_still_loads() {
        let dir = tmpdir("minor");
        let path = dir.join("run.rstorch");
        let mut meta = HashMap::new();
        meta.insert(KEY_MAGIC.to_string(), MAGIC.to_string());
        meta.insert(KEY_MAJOR.to_string(), FORMAT_MAJOR.to_string());
        meta.insert(KEY_MINOR.to_string(), (FORMAT_MINOR + 5).to_string());
        meta.insert(
            format!("{SECTION_PREFIX}config"),
            "future-config".to_string(),
        );
        let tensors: BTreeMap<String, HostTensor> = BTreeMap::new();
        save_tensors(&path, &tensors, Some(meta), &Limits::defaults()).unwrap();

        let back = Envelope::load(&path, &Limits::defaults()).unwrap();
        assert_eq!(back.version(), (FORMAT_MAJOR, FORMAT_MINOR + 5));
        assert_eq!(back.section("config"), Some("future-config"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn invalid_section_name_rejected() {
        let mut e = Envelope::new();
        assert!(e.set_section("has.dot", "x").is_err());
        assert!(e.set_section("", "x").is_err());
        assert!(e.set_section("ok", "x").is_ok());
    }
}
