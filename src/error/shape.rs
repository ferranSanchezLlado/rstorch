use std::error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
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
    ZeroDimension {
        op: &'static str,
        axis: usize,
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
    InvalidSpatialParam {
        op: &'static str,
        param: &'static str,
        value: usize,
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
            Self::ZeroDimension { op, axis } => {
                write!(f, "{op} requires non-zero dimension at axis {axis}")
            }
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
            Self::InvalidSpatialParam {
                op,
                param,
                value,
                reason,
            } => write!(
                f,
                "{op} invalid spatial parameter {param}={value}: {reason}"
            ),
        }
    }
}

impl error::Error for ShapeError {}
