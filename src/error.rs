//! The single error type for the crate.
//!
//! Design contract: one `thiserror` enum; every variant
//! carries the operation name (`op`) where one exists and the offending
//! values, so a failure is loud and self-describing without a backtrace.
//!
//! Fallibility is two-tier: every *named* method returns
//! [`Result`]; operator sugar (`+`, `-`, `*`, `/`) panics with the identical
//! structured message under `#[track_caller]`.
//!
//! Variant set ownership: this initial set is
//! deliberately small; later additions are appends. The enum is
//! `#[non_exhaustive]` for that reason.

use crate::device::Device;
use crate::dtype::DType;
use crate::shape::Shape;

/// Crate-wide result alias; the error type defaults to [`enum@Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The error type for every fallible operation in the crate.
///
/// Every variant that corresponds to a tensor operation carries the
/// operation name in `op` (a `&'static str` such as `"matmul"` or
/// `"reshape"`), naming the *public* method the user called.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Two shapes were incompatible for the operation (e.g. broadcast
    /// failure on a binary op, inner-dimension mismatch on `matmul`,
    /// mismatched non-`cat` axes on `cat`).
    #[error("{op}: shape mismatch: lhs {lhs} vs rhs {rhs}")]
    ShapeMismatch {
        /// Public operation that failed.
        op: &'static str,
        /// Left-hand / first shape involved.
        lhs: Shape,
        /// Right-hand / second shape involved.
        rhs: Shape,
    },

    /// The tensor's rank differs from what the operation requires
    /// (e.g. `dims3()` on a rank-2 tensor, `conv2d` on a rank-3 input).
    #[error("{op}: rank mismatch: expected rank {expected}, got rank {got}")]
    RankMismatch {
        /// Public operation that failed.
        op: &'static str,
        /// Rank the operation requires.
        expected: usize,
        /// Rank the tensor actually has.
        got: usize,
    },

    /// An axis argument was outside `[-rank, rank)` (or `[-rank-1, rank]`
    /// for axis-inserting ops such as `unsqueeze`).
    #[error("{op}: invalid axis {axis} for rank {rank}")]
    InvalidAxis {
        /// Public operation that failed.
        op: &'static str,
        /// The axis as given by the caller (negative axes allowed).
        axis: isize,
        /// Rank of the tensor the axis was resolved against.
        rank: usize,
    },

    /// Dtypes were incompatible. There is **no implicit dtype promotion**:
    /// mixing dtypes is an error telling you to cast
    /// explicitly with `to_dtype`.
    #[error(
        "{op}: dtype mismatch: expected {expected}, got {got} (no implicit promotion; cast explicitly with to_dtype)"
    )]
    DTypeMismatch {
        /// Public operation that failed.
        op: &'static str,
        /// Dtype the operation expected.
        expected: DType,
        /// Dtype it received.
        got: DType,
    },

    /// Operands live on different devices; move one with `to_device`.
    #[error("{op}: device mismatch: expected {expected}, got {got}")]
    DeviceMismatch {
        /// Public operation that failed.
        op: &'static str,
        /// Device the operation expected.
        expected: Device,
        /// Device it received.
        got: Device,
    },

    /// `reshape` target has a different element count than the source (or
    /// is otherwise impossible).
    #[error("{op}: cannot reshape {from} into {to}")]
    ReshapeMismatch {
        /// Public operation that failed.
        op: &'static str,
        /// Source shape.
        from: Shape,
        /// Requested target shape.
        to: Shape,
    },

    /// An index (e.g. in `narrow`, `index_select`, `gather`) was out of
    /// bounds for the indexed axis.
    #[error("{op}: index {index} out of bounds for axis {axis} of size {size}")]
    IndexOutOfBounds {
        /// Public operation that failed.
        op: &'static str,
        /// The offending index value.
        index: i64,
        /// The axis being indexed.
        axis: usize,
        /// The size of that axis.
        size: usize,
    },

    /// The operation has no kernel for this device/dtype combination.
    /// There are **no silent host round-trips or CPU fallbacks**:
    /// a missing GPU kernel is this loud error.
    #[error("{op}: unsupported on {device} for dtype {dtype}")]
    Unsupported {
        /// Public operation that failed.
        op: &'static str,
        /// Device the operation ran on.
        device: Device,
        /// Dtype involved.
        dtype: DType,
    },

    /// `backward()` was called on a tensor that carries no autograd graph
    /// (nothing in its history was traced), or `traced()` misuse was
    /// detected. A graph-less `backward()` is an error, **not** an empty
    /// `Grads`.
    #[error(
        "{op}: tensor is not traced (no autograd graph; did you forget Mode::TRAIN or Param::get?)"
    )]
    NotTraced {
        /// Public operation that failed.
        op: &'static str,
    },

    /// The optimizer visited a non-frozen parameter that has no gradient in
    /// the `Grads` it was given. This is the loud answer to a silently
    /// untrained parameter: an untraced weight access is
    /// an error at the very next `step`, not silent non-training.
    #[error(
        "step: missing gradient for non-frozen parameter `{path}` (was it used under a recording Mode?)"
    )]
    MissingGrad {
        /// Dotted visitor path of the parameter (e.g. `"fc1.weight"`).
        path: String,
    },

    /// A scalar/argument precondition failed (negative probability,
    /// zero batch size, empty reduction without a policy, ...).
    #[error("{op}: invalid argument: {msg}")]
    InvalidArg {
        /// Public operation that failed.
        op: &'static str,
        /// Human-readable description of the violated precondition.
        msg: String,
    },

    /// Dataset / data-pipeline error (download, checksum, parse, collate).
    #[error("data: {msg}")]
    Data {
        /// Human-readable description.
        msg: String,
    },

    /// Tokenizer error (invalid UTF-8, unknown token, vocab validation).
    #[error("tokenizer: {msg}")]
    Tokenizer {
        /// Human-readable description.
        msg: String,
    },

    /// Persistence error (safetensors envelope, versioning, reader limits,
    /// staged-restore validation).
    #[error("persistence: {msg}")]
    Persistence {
        /// Human-readable description.
        msg: String,
    },

    /// An underlying I/O error.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// A backend-internal failure (device API error, allocation failure).
    #[error("{op}: backend error: {msg}")]
    Backend {
        /// Public operation that failed.
        op: &'static str,
        /// Backend-provided description.
        msg: String,
    },
}

impl Error {
    /// Rename the operation this error blames, keeping everything else.
    ///
    /// An inner layer names an error after whatever it knows: a kernel reports
    /// the op *family* (`"reduce"`, `"add"`), and a composed implementation
    /// reports the primitive it happened to call (`gather` inside a loss).
    /// The public seam rewrites that to the method the user actually called.
    ///
    /// The match is deliberately exhaustive — no `_` arm — so that adding a
    /// variant to this `#[non_exhaustive]` enum is a compile error here rather
    /// than a silently mis-labelled error at every seam.
    pub(crate) fn with_op(mut self, op: &'static str) -> Error {
        match &mut self {
            Error::ShapeMismatch { op: slot, .. }
            | Error::RankMismatch { op: slot, .. }
            | Error::InvalidAxis { op: slot, .. }
            | Error::DTypeMismatch { op: slot, .. }
            | Error::DeviceMismatch { op: slot, .. }
            | Error::ReshapeMismatch { op: slot, .. }
            | Error::IndexOutOfBounds { op: slot, .. }
            | Error::Unsupported { op: slot, .. }
            | Error::NotTraced { op: slot }
            | Error::InvalidArg { op: slot, .. }
            | Error::Backend { op: slot, .. } => *slot = op,
            // No operation name to rewrite.
            Error::MissingGrad { .. }
            | Error::Data { .. }
            | Error::Tokenizer { .. }
            | Error::Persistence { .. }
            | Error::Io(_) => {}
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_op_renames_the_operation() {
        let e = Error::Unsupported {
            op: "reduce",
            device: Device::Cpu,
            dtype: DType::Bool,
        }
        .with_op("sum");
        assert_eq!(e.to_string(), "sum: unsupported on cpu for dtype bool");

        // Variants without an `op` pass through untouched.
        let e = Error::MissingGrad {
            path: "fc1.weight".to_string(),
        }
        .with_op("sum");
        assert!(matches!(e, Error::MissingGrad { path } if path == "fc1.weight"));
    }

    #[test]
    fn messages_carry_op_and_values() {
        let e = Error::ShapeMismatch {
            op: "matmul",
            lhs: Shape::from(vec![2, 3]),
            rhs: Shape::from(vec![4, 5]),
        };
        assert_eq!(
            e.to_string(),
            "matmul: shape mismatch: lhs [2, 3] vs rhs [4, 5]"
        );

        let e = Error::InvalidAxis {
            op: "sum",
            axis: -3,
            rank: 2,
        };
        assert_eq!(e.to_string(), "sum: invalid axis -3 for rank 2");

        let e = Error::MissingGrad {
            path: "fc1.weight".to_string(),
        };
        assert!(e.to_string().contains("`fc1.weight`"));
    }

    #[test]
    fn io_error_converts() {
        fn fails() -> Result<()> {
            Err(std::io::Error::new(std::io::ErrorKind::NotFound, "gone"))?;
            Ok(())
        }
        assert!(matches!(fails(), Err(Error::Io(_))));
    }
}
