use crate::dtype::DTypeId;
use std::error;
use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Shape(ShapeError),
    Backend(Box<dyn error::Error + Send + Sync + 'static>),
    Device(DeviceError),
    DType(DTypeError),
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
        }
    }
}

impl error::Error for Error {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Backend(err) => Some(err.as_ref()),
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
