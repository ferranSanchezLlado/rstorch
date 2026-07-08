use std::error;
use std::fmt;

#[derive(Debug)]
#[non_exhaustive]
pub enum DataError {
    IndexOutOfBounds {
        index: usize,
        len: usize,
    },
    NegativeIndex {
        index: i64,
    },
    EmptyBatch,
    InvalidBatchSize {
        batch_size: usize,
    },
    WrongBatchSize {
        expected: usize,
        found: usize,
    },
    InconsistentSampleShape {
        index: usize,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    InvalidTensorDataset {
        reason: &'static str,
    },
    MissingSpecialToken {
        token: &'static str,
    },
    Io {
        source: std::io::Error,
    },
    Parse {
        source: Box<dyn error::Error + Send + Sync + 'static>,
    },
    ChecksumMismatch {
        expected: String,
        found: String,
    },
    DownloadSizeCap {
        max_bytes: u64,
    },
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IndexOutOfBounds { index, len } => {
                write!(f, "index {index} out of bounds for dataset of length {len}")
            }
            Self::NegativeIndex { index } => write!(f, "negative index {index} is invalid"),
            Self::EmptyBatch => write!(f, "cannot collate an empty batch"),
            Self::InvalidBatchSize { batch_size } => {
                write!(f, "invalid batch size {batch_size}")
            }
            Self::WrongBatchSize { expected, found } => {
                write!(f, "wrong batch size: expected {expected}, found {found}")
            }
            Self::InconsistentSampleShape {
                index,
                expected,
                found,
            } => write!(
                f,
                "inconsistent sample shape at index {index}: expected {expected:?}, found {found:?}"
            ),
            Self::InvalidTensorDataset { reason } => write!(f, "invalid tensor dataset: {reason}"),
            Self::MissingSpecialToken { token } => {
                write!(f, "tokenizer does not define required {token} token")
            }
            Self::Io { source } => write!(f, "io error: {source}"),
            Self::Parse { source } => write!(f, "parse error: {source}"),
            Self::ChecksumMismatch { expected, found } => {
                write!(f, "checksum mismatch: expected {expected}, found {found}")
            }
            Self::DownloadSizeCap { max_bytes } => {
                write!(f, "download exceeds size cap of {max_bytes} bytes")
            }
        }
    }
}

impl error::Error for DataError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Io { source } => Some(source),
            Self::Parse { source } => Some(source.as_ref()),
            _ => None,
        }
    }
}
