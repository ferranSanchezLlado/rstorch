use crate::dtype::DTypeId;
use std::error;
use std::fmt;

#[derive(Debug)]
#[non_exhaustive]
pub enum PersistenceError {
    Io {
        source: std::io::Error,
    },
    InvalidMagic {
        expected: &'static str,
        found: Vec<u8>,
    },
    UnsupportedVersion {
        version: u32,
    },
    InvalidUtf8 {
        source: std::string::FromUtf8Error,
    },
    InvalidDType {
        code: u8,
    },
    DuplicateTensor {
        name: String,
    },
    MissingTensor {
        name: String,
    },
    UnexpectedTensor {
        name: String,
    },
    DTypeMismatch {
        name: String,
        expected: DTypeId,
        found: DTypeId,
    },
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    LengthMismatch {
        name: String,
        expected: usize,
        found: usize,
    },
    AllocationTooLarge {
        field: &'static str,
        len: u64,
        max: u64,
    },
    SizeOverflow {
        name: String,
    },
    InvalidFormat {
        reason: &'static str,
    },
    UnknownOptimizer {
        code: u8,
    },
    OptimizerMismatch {
        expected: &'static str,
        found: &'static str,
    },
    MissingOptimizerState,
    UnexpectedTrailingBytes,
}

impl fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { source } => write!(f, "io error: {source}"),
            Self::InvalidMagic { expected, found } => {
                write!(f, "invalid magic: expected {expected}, found {found:?}")
            }
            Self::UnsupportedVersion { version } => {
                write!(f, "unsupported persistence version {version}")
            }
            Self::InvalidUtf8 { source } => write!(f, "invalid utf-8: {source}"),
            Self::InvalidDType { code } => write!(f, "invalid dtype code {code}"),
            Self::DuplicateTensor { name } => write!(f, "duplicate tensor name {name}"),
            Self::MissingTensor { name } => write!(f, "missing tensor {name}"),
            Self::UnexpectedTensor { name } => write!(f, "unexpected tensor {name}"),
            Self::DTypeMismatch {
                name,
                expected,
                found,
            } => write!(
                f,
                "dtype mismatch for {name}: expected {expected:?}, found {found:?}"
            ),
            Self::ShapeMismatch {
                name,
                expected,
                found,
            } => write!(
                f,
                "shape mismatch for {name}: expected {expected:?}, found {found:?}"
            ),
            Self::LengthMismatch {
                name,
                expected,
                found,
            } => write!(
                f,
                "length mismatch for {name}: expected {expected}, found {found}"
            ),
            Self::AllocationTooLarge { field, len, max } => {
                write!(f, "{field} length {len} exceeds maximum {max}")
            }
            Self::SizeOverflow { name } => write!(f, "size overflow while reading {name}"),
            Self::InvalidFormat { reason } => write!(f, "invalid persistence format: {reason}"),
            Self::UnknownOptimizer { code } => write!(f, "unknown optimizer code {code}"),
            Self::OptimizerMismatch { expected, found } => {
                write!(f, "optimizer mismatch: expected {expected}, found {found}")
            }
            Self::MissingOptimizerState => write!(f, "checkpoint does not contain optimizer state"),
            Self::UnexpectedTrailingBytes => write!(f, "unexpected trailing bytes"),
        }
    }
}

impl error::Error for PersistenceError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Io { source } => Some(source),
            Self::InvalidUtf8 { source } => Some(source),
            _ => None,
        }
    }
}
