use std::error;
use std::fmt;

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
