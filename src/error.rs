use crate::dtype::DTypeId;
use std::convert::Infallible;
use std::error;
use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Shape(ShapeError),
    Backend(Box<dyn error::Error + Send + Sync + 'static>),
    Device(DeviceError),
    DType(DTypeError),
    Data(DataError),
}

impl Error {
    pub(crate) fn backend<E>(err: E) -> Self
    where
        E: error::Error + Send + Sync + 'static,
    {
        Self::Backend(Box::new(err))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(err) => write!(f, "shape error: {err}"),
            Self::Backend(err) => write!(f, "backend error: {err}"),
            Self::Device(err) => write!(f, "device error: {err}"),
            Self::DType(err) => write!(f, "dtype error: {err}"),
            Self::Data(err) => write!(f, "data error: {err}"),
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Backend(err) => Some(err.as_ref()),
            Self::Data(err) => err.source(),
            _ => None,
        }
    }
}

impl From<ShapeError> for Error {
    fn from(err: ShapeError) -> Self {
        Self::Shape(err)
    }
}

impl From<DeviceError> for Error {
    fn from(err: DeviceError) -> Self {
        Self::Device(err)
    }
}

impl From<DTypeError> for Error {
    fn from(err: DTypeError) -> Self {
        Self::DType(err)
    }
}

impl From<DataError> for Error {
    fn from(err: DataError) -> Self {
        Self::Data(err)
    }
}

impl From<Infallible> for Error {
    fn from(err: Infallible) -> Self {
        match err {}
    }
}

#[derive(Debug)]
pub enum DataError {
    IndexOutOfBounds {
        index: usize,
        len: usize,
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
    Io {
        source: std::io::Error,
    },
    Parse {
        source: Box<dyn error::Error + Send + Sync + 'static>,
    },
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IndexOutOfBounds { index, len } => {
                write!(f, "index {index} out of bounds for dataset of length {len}")
            }
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
            Self::Io { source } => write!(f, "io error: {source}"),
            Self::Parse { source } => write!(f, "parse error: {source}"),
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeError {
    LengthMismatch {
        expected: usize,
        found: usize,
    },
    RankMismatch {
        expected: usize,
        found: usize,
    },
    DimMismatch {
        op: &'static str,
        operand: usize,
        axis: usize,
        expected: usize,
        found: usize,
    },
    SymbolMismatch {
        op: &'static str,
        symbol: &'static str,
        lhs: (usize, usize),
        rhs: (usize, usize),
        lhs_size: usize,
        rhs_size: usize,
    },
    NumelOverflow {
        dims: Box<[usize]>,
    },
    InvalidOffset {
        offset: usize,
        storage_len: usize,
    },
    LayoutOutOfBounds {
        offset: usize,
        storage_len: usize,
    },
    ViewIncompatible {
        op: &'static str,
        reason: &'static str,
    },
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { expected, found } => {
                write!(f, "length mismatch: expected {expected}, found {found}")
            }
            Self::RankMismatch { expected, found } => {
                write!(f, "rank mismatch: expected {expected}, found {found}")
            }
            Self::DimMismatch {
                op,
                operand,
                axis,
                expected,
                found,
            } => write!(
                f,
                "{op} dimension mismatch at operand {operand}, axis {axis}: expected {expected}, found {found}"
            ),
            Self::SymbolMismatch {
                op,
                symbol,
                lhs,
                rhs,
                lhs_size,
                rhs_size,
            } => write!(
                f,
                "{op} symbol mismatch for {symbol}: {:?} has {lhs_size}, {:?} has {rhs_size}",
                lhs, rhs
            ),
            Self::NumelOverflow { dims } => write!(f, "shape element count overflow for {dims:?}"),
            Self::InvalidOffset {
                offset,
                storage_len,
            } => write!(
                f,
                "invalid layout offset {offset} for storage length {storage_len}"
            ),
            Self::LayoutOutOfBounds {
                offset,
                storage_len,
            } => write!(
                f,
                "layout addresses element {offset} outside storage length {storage_len}"
            ),
            Self::ViewIncompatible { op, reason } => {
                write!(f, "{op} view incompatible: {reason}")
            }
        }
    }
}

impl error::Error for ShapeError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceError {
    Mismatch {
        op: &'static str,
        lhs: String,
        rhs: String,
    },
    Unavailable {
        requested: String,
    },
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mismatch { op, lhs, rhs } => write!(f, "{op} device mismatch: {lhs} != {rhs}"),
            Self::Unavailable { requested } => write!(f, "device unavailable: {requested}"),
        }
    }
}

impl error::Error for DeviceError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DTypeError {
    Mismatch {
        op: &'static str,
        expected: DTypeId,
        found: DTypeId,
    },
}

impl fmt::Display for DTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mismatch {
                op,
                expected,
                found,
            } => {
                write!(
                    f,
                    "{op} dtype mismatch: expected {expected:?}, found {found:?}"
                )
            }
        }
    }
}

impl error::Error for DTypeError {}
