use crate::dtype::DTypeId;
use std::error;
use std::fmt;

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
