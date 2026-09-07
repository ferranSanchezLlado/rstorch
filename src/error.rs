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
//! Timing is a contract too, and it splits the variant set. The *argument*
//! errors — [`Error::ShapeMismatch`], [`Error::RankMismatch`],
//! [`Error::InvalidAxis`], [`Error::DTypeMismatch`], [`Error::DeviceMismatch`],
//! [`Error::ReshapeMismatch`] and [`Error::InvalidArg`] — are decided from
//! metadata (shapes, ranks, dtypes, devices, axes) before any kernel runs, so
//! they always surface at the call site. That is what makes the
//! `#[track_caller]` location on the operator sugar mean something, and it
//! stays true if execution is ever deferred. The *execution* errors —
//! [`Error::Unsupported`] and [`Error::Backend`] — surface at or before the
//! next materialization instead: backends batch work (Metal encodes dispatches
//! into a command buffer and flushes on a threshold), so a kernel failure can
//! be observed at a later `to_vec`, device transfer, save, or explicit
//! [`Device::synchronize`](crate::Device::synchronize) rather than at the
//! operation that queued it. Both carry `op`, so the failing operation is named
//! even when the reported location belongs to the transfer.
//!
//! [`Error::IndexOutOfBounds`] falls in either half depending on where the bad
//! index lives. An out-of-range *argument* is metadata and reports at the call
//! site. An out-of-range *element of an index tensor* can only be found by
//! looking at the data: the CPU backend reads it on the host and reports
//! immediately, while Metal, CUDA and WGPU validate on device and collect the
//! verdict at the next host boundary, so the error arrives from the following
//! transfer. No wrong value is observable in between — the host cannot see a
//! result without passing the check.
//!
//! Variant set ownership: this initial set is
//! deliberately small; later additions are appends. The enum is
//! `#[non_exhaustive]` for that reason.

use crate::device::Device;
use crate::dtype::DType;
use crate::shape::Shape;

/// Crate-wide result alias; the error type defaults to [`enum@Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The boxed cause the three message-carrying variants can wrap.
///
/// Spelled out once so both the variant fields and the constructor signatures
/// stay readable.
type Cause = Box<dyn std::error::Error + Send + Sync + 'static>;

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
    #[non_exhaustive]
    ShapeMismatch {
        /// Public operation that failed.
        op: &'static str,
        /// The shape the operation *requires*. `Display` renders only
        /// "lhs X vs rhs Y", so the order is the only thing that says which
        /// side is the requirement, and the rule is: for an asymmetric check
        /// — against a weight, a resolved geometry, a destination — `lhs` is
        /// always the requirement, matching `expected` on
        /// [`Error::RankMismatch`] and its siblings. For a symmetric check
        /// between two operands, such as a broadcast failure on `a + b`,
        /// `lhs` is the first operand.
        lhs: Shape,
        /// The shape the operation actually received, or the second operand
        /// of a symmetric check.
        rhs: Shape,
    },

    /// The tensor's rank differs from what the operation requires
    /// (e.g. `dims3()` on a rank-2 tensor, `conv2d` on a rank-3 input).
    #[error("{op}: rank mismatch: expected rank {expected}, got rank {got}")]
    #[non_exhaustive]
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
    #[non_exhaustive]
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
    #[non_exhaustive]
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
    #[non_exhaustive]
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
    #[non_exhaustive]
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
    #[non_exhaustive]
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
    #[non_exhaustive]
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
    #[non_exhaustive]
    NotTraced {
        /// Public operation that failed.
        op: &'static str,
    },

    /// The optimizer found no gradient for a non-frozen parameter.
    ///
    /// `op` is the public method that reported the missing gradient.
    #[error(
        "{op}: missing gradient for non-frozen parameter `{path}` (was it used under a recording Mode?)"
    )]
    #[non_exhaustive]
    MissingGrad {
        /// Public operation that failed.
        op: &'static str,
        /// Dotted visitor path of the parameter (e.g. `"fc1.weight"`).
        path: String,
    },

    /// A scalar/argument precondition failed (negative probability,
    /// zero batch size, empty reduction without a policy, ...).
    #[error("{op}: invalid argument: {msg}")]
    #[non_exhaustive]
    InvalidArg {
        /// Public operation that failed.
        op: &'static str,
        /// Human-readable description of the violated precondition.
        msg: String,
    },

    /// Dataset / data-pipeline error (download, checksum, parse, collate).
    ///
    /// Build it with [`Error::data`], or with [`Error::data_with`] when a
    /// lower-level cause (a `ureq` transport failure, a `flate2` decode
    /// failure) is in hand and worth keeping.
    #[error("data: {msg}")]
    #[non_exhaustive]
    Data {
        /// Human-readable description.
        msg: String,
        /// The underlying failure, when there was one, reachable through
        /// [`std::error::Error::source`].
        #[source]
        source: Option<Cause>,
    },

    /// Tokenizer error (invalid UTF-8, unknown token, vocab validation).
    ///
    /// Build it with [`Error::tokenizer`], or with [`Error::tokenizer_with`]
    /// when a lower-level cause is in hand and worth keeping.
    #[error("tokenizer: {msg}")]
    #[non_exhaustive]
    Tokenizer {
        /// Human-readable description.
        msg: String,
        /// The underlying failure, when there was one, reachable through
        /// [`std::error::Error::source`].
        #[source]
        source: Option<Cause>,
    },

    /// Persistence error (safetensors envelope, versioning, reader limits,
    /// staged-restore validation).
    ///
    /// Build it with [`Error::persistence`], or with
    /// [`Error::persistence_with`] when a lower-level cause (a
    /// `safetensors` or `serde_json` failure, a failed integer parse) is in
    /// hand and worth keeping.
    #[error("persistence: {msg}")]
    #[non_exhaustive]
    Persistence {
        /// Human-readable description.
        msg: String,
        /// The underlying failure, when there was one, reachable through
        /// [`std::error::Error::source`].
        #[source]
        source: Option<Cause>,
    },

    /// An underlying I/O error.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// A backend-internal failure (device API error, allocation failure).
    #[error("{op}: backend error: {msg}")]
    #[non_exhaustive]
    Backend {
        /// Public operation that failed.
        op: &'static str,
        /// Backend-provided description.
        msg: String,
    },
}

/// [`enum@Error`] crosses thread boundaries: a worker thread in a data
/// pipeline must be able to hand its failure back to the caller. Nothing in
/// the enum may quietly give that up, so it is asserted rather than assumed.
///
/// The witness lives *inside* the `const` block on purpose: as a sibling item
/// it reads as dead code to the MSRV compiler, which does not count a call
/// from a `const _` initializer as a use, and the warning would invite
/// deleting the assertion that is the whole point.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Error>();
};

/// Constructors.
///
/// Every struct variant is `#[non_exhaustive]`, so a struct literal is a
/// crate-only spelling and code outside `rstorch` builds errors through these.
/// That is deliberate: the public traits — [`Forward`](crate::nn::Forward),
/// [`Module`](crate::nn::Module), [`Dataset`](crate::data::Dataset),
/// [`Tokenizer`](crate::text::Tokenizer) — all return [`Result`], so a
/// third-party layer validating its own input has to be able to speak this
/// vocabulary rather than inventing a parallel one.
///
/// The set covered is every variant a third-party implementation can decide
/// for itself: the *argument* half of the timing contract in the module docs,
/// [`Error::Unsupported`] and [`Error::IndexOutOfBounds`] — which the module
/// docs place in the *execution* half and straddling both respectively, but
/// which a layer declining a dtype or rejecting an index argument raises just
/// as directly — plus the three message-carrying wrappers.
/// [`Error::NotTraced`], [`Error::MissingGrad`] and [`Error::Backend`] have no
/// constructor on purpose: they report on the autograd engine and on backend
/// internals, which only this crate owns. [`Error::Io`] is built with
/// `From<std::io::Error>`.
impl Error {
    /// Two shapes were incompatible for `op`.
    pub fn shape_mismatch(op: &'static str, lhs: impl Into<Shape>, rhs: impl Into<Shape>) -> Error {
        Error::ShapeMismatch {
            op,
            lhs: lhs.into(),
            rhs: rhs.into(),
        }
    }

    /// `op` requires rank `expected` and received rank `got`.
    pub fn rank_mismatch(op: &'static str, expected: usize, got: usize) -> Error {
        Error::RankMismatch { op, expected, got }
    }

    /// `axis` is outside the range `op` accepts for a tensor of rank `rank`.
    pub fn invalid_axis(op: &'static str, axis: isize, rank: usize) -> Error {
        Error::InvalidAxis { op, axis, rank }
    }

    /// `op` expected dtype `expected` and received `got`. There is no implicit
    /// promotion anywhere in the crate; a layer that mixes dtypes reports this.
    pub fn dtype_mismatch(op: &'static str, expected: DType, got: DType) -> Error {
        Error::DTypeMismatch { op, expected, got }
    }

    /// `op`'s operands are on different devices.
    pub fn device_mismatch(op: &'static str, expected: Device, got: Device) -> Error {
        Error::DeviceMismatch { op, expected, got }
    }

    /// `from` cannot be reshaped into `to` under `op`.
    pub fn reshape_mismatch(
        op: &'static str,
        from: impl Into<Shape>,
        to: impl Into<Shape>,
    ) -> Error {
        Error::ReshapeMismatch {
            op,
            from: from.into(),
            to: to.into(),
        }
    }

    /// `index` is out of bounds for an axis of length `size`.
    pub fn index_out_of_bounds(op: &'static str, index: i64, axis: usize, size: usize) -> Error {
        Error::IndexOutOfBounds {
            op,
            index,
            axis,
            size,
        }
    }

    /// `op` has no kernel for this device/dtype pair. Report this rather than
    /// copying through the host: a missing kernel is loud, never silent.
    pub fn unsupported(op: &'static str, device: Device, dtype: DType) -> Error {
        Error::Unsupported { op, device, dtype }
    }

    /// A scalar or argument precondition of `op` was violated. The catch-all
    /// of the argument bucket, for a failure the structured variants above do
    /// not describe.
    pub fn invalid_arg(op: &'static str, msg: impl Into<String>) -> Error {
        Error::InvalidArg {
            op,
            msg: msg.into(),
        }
    }
}

impl Error {
    /// Build [`Error::Data`] from a message, with no underlying cause.
    pub fn data(msg: impl Into<String>) -> Error {
        Error::Data {
            msg: msg.into(),
            source: None,
        }
    }

    /// Build [`Error::Data`] from a message and the failure that caused it.
    ///
    /// The cause stays reachable through [`std::error::Error::source`], so a
    /// caller can inspect (say) the `ureq` transport error behind a failed
    /// download instead of only reading about it in the message.
    pub fn data_with(msg: impl Into<String>, source: impl Into<Cause>) -> Error {
        Error::Data {
            msg: msg.into(),
            source: Some(source.into()),
        }
    }

    /// Build [`Error::Tokenizer`] from a message, with no underlying cause.
    pub fn tokenizer(msg: impl Into<String>) -> Error {
        Error::Tokenizer {
            msg: msg.into(),
            source: None,
        }
    }

    /// Build [`Error::Tokenizer`] from a message and the failure that caused
    /// it; the cause stays reachable through [`std::error::Error::source`].
    pub fn tokenizer_with(msg: impl Into<String>, source: impl Into<Cause>) -> Error {
        Error::Tokenizer {
            msg: msg.into(),
            source: Some(source.into()),
        }
    }

    /// Build [`Error::Persistence`] from a message, with no underlying cause.
    pub fn persistence(msg: impl Into<String>) -> Error {
        Error::Persistence {
            msg: msg.into(),
            source: None,
        }
    }

    /// Build [`Error::Persistence`] from a message and the failure that caused
    /// it.
    ///
    /// The cause stays reachable through [`std::error::Error::source`], so a
    /// caller can inspect the `safetensors` or `serde_json` error behind a
    /// rejected checkpoint header rather than re-parsing the message.
    pub fn persistence_with(msg: impl Into<String>, source: impl Into<Cause>) -> Error {
        Error::Persistence {
            msg: msg.into(),
            source: Some(source.into()),
        }
    }

    /// Replace the operation name while keeping the rest of the error.
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
            | Error::MissingGrad { op: slot, .. }
            | Error::InvalidArg { op: slot, .. }
            | Error::Backend { op: slot, .. } => *slot = op,
            // No operation name to rewrite.
            Error::Data { .. }
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

    /// The exact `Display` text of every variant is part of the crate's
    /// observable surface — logs and downstream tests match on it — so each
    /// message is pinned here rather than merely spot-checked.
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

        let e = Error::RankMismatch {
            op: "dims3",
            expected: 3,
            got: 2,
        };
        assert_eq!(
            e.to_string(),
            "dims3: rank mismatch: expected rank 3, got rank 2"
        );

        let e = Error::InvalidAxis {
            op: "sum",
            axis: -3,
            rank: 2,
        };
        assert_eq!(e.to_string(), "sum: invalid axis -3 for rank 2");

        let e = Error::DTypeMismatch {
            op: "add",
            expected: DType::F32,
            got: DType::I64,
        };
        assert_eq!(
            e.to_string(),
            "add: dtype mismatch: expected f32, got i64 \
             (no implicit promotion; cast explicitly with to_dtype)"
        );

        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            let e = Error::DeviceMismatch {
                op: "add",
                expected: Device::Cpu,
                got: Device::Metal(0),
            };
            assert_eq!(
                e.to_string(),
                "add: device mismatch: expected cpu, got metal:0"
            );
        }

        let e = Error::ReshapeMismatch {
            op: "reshape",
            from: Shape::from(vec![2, 3]),
            to: Shape::from(vec![5]),
        };
        assert_eq!(e.to_string(), "reshape: cannot reshape [2, 3] into [5]");

        let e = Error::IndexOutOfBounds {
            op: "index_select",
            index: 3,
            axis: 0,
            size: 3,
        };
        assert_eq!(
            e.to_string(),
            "index_select: index 3 out of bounds for axis 0 of size 3"
        );

        let e = Error::Unsupported {
            op: "tanh",
            device: Device::Cpu,
            dtype: DType::Bool,
        };
        assert_eq!(e.to_string(), "tanh: unsupported on cpu for dtype bool");

        let e = Error::NotTraced { op: "backward" };
        assert_eq!(
            e.to_string(),
            "backward: tensor is not traced \
             (no autograd graph; did you forget Mode::TRAIN or Param::get?)"
        );

        let e = Error::MissingGrad {
            op: "step",
            path: "fc1.weight".to_string(),
        };
        assert_eq!(
            e.to_string(),
            "step: missing gradient for non-frozen parameter `fc1.weight` \
             (was it used under a recording Mode?)"
        );

        let e = Error::InvalidArg {
            op: "dropout",
            msg: "p must be in [0, 1)".to_string(),
        };
        assert_eq!(
            e.to_string(),
            "dropout: invalid argument: p must be in [0, 1)"
        );

        let e = Error::Backend {
            op: "matmul",
            msg: "device lost".to_string(),
        };
        assert_eq!(e.to_string(), "matmul: backend error: device lost");

        // The three message-carrying variants render the message alone; the
        // optional cause is reachable through `source`, never spliced into the
        // text, so adding one cannot change what a log line says.
        assert_eq!(
            Error::data("bad idx magic").to_string(),
            "data: bad idx magic"
        );
        assert_eq!(
            Error::data_with("bad idx magic", "transport closed").to_string(),
            "data: bad idx magic"
        );
        assert_eq!(
            Error::tokenizer("unknown token id 7").to_string(),
            "tokenizer: unknown token id 7"
        );
        assert_eq!(
            Error::tokenizer_with("unknown token id 7", "vocab truncated").to_string(),
            "tokenizer: unknown token id 7"
        );
        assert_eq!(
            Error::persistence("header too short").to_string(),
            "persistence: header too short"
        );
        assert_eq!(
            Error::persistence_with("header too short", "eof").to_string(),
            "persistence: header too short"
        );

        let e = Error::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No such file or directory",
        ));
        assert_eq!(e.to_string(), "io: No such file or directory");
    }

    #[test]
    fn io_error_converts() {
        fn fails() -> Result<()> {
            Err(std::io::Error::new(std::io::ErrorKind::NotFound, "gone"))?;
            Ok(())
        }
        assert!(matches!(fails(), Err(Error::Io(_))));
    }

    /// The point of the boxed cause is `source()`, not the message — which is
    /// why `Display` is asserted to be unchanged above. Both halves are pinned:
    /// present when a cause was threaded, absent when there was none.
    #[test]
    fn wrapped_causes_are_reachable_through_source() {
        use std::error::Error as _;

        for cause_less in [
            Error::data("checksum mismatch"),
            Error::tokenizer("unknown token"),
            Error::persistence("header too short"),
        ] {
            assert!(cause_less.source().is_none(), "{cause_less}");
        }

        let inner = std::io::Error::new(std::io::ErrorKind::InvalidData, "bad header");
        let e = Error::persistence_with("invalid safetensors JSON header", inner);
        let cause = e.source().expect("cause is reachable");
        assert_eq!(cause.to_string(), "bad header");
        assert!(cause.downcast_ref::<std::io::Error>().is_some());

        // `Io` already exposed a cause before the three wrappers did; that
        // must not have regressed.
        let io = Error::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "No such file or directory",
        ));
        assert!(io.source().is_some());
    }

    /// `with_op` has two halves and the pass-through half is what the
    /// deliberately wildcard-free match protects: a variant that carries no
    /// `op` must come back untouched rather than silently mislabeled.
    #[test]
    fn with_op_renames_only_the_variants_that_carry_an_op() {
        let renamed = Error::invalid_axis("reduce", 3, 2).with_op("sum");
        assert!(matches!(renamed, Error::InvalidAxis { op: "sum", .. }));
        assert_eq!(renamed.to_string(), "sum: invalid axis 3 for rank 2");

        // `MissingGrad` carries an `op` too, so a custom optimizer's seam can
        // claim the error instead of inheriting the built-in `"step"`.
        let renamed = Error::MissingGrad {
            op: "step",
            path: "fc1.weight".to_string(),
        }
        .with_op("MyOptimizer::update");
        assert_eq!(
            renamed.to_string(),
            "MyOptimizer::update: missing gradient for non-frozen parameter \
             `fc1.weight` (was it used under a recording Mode?)"
        );

        let untouched = Error::persistence("header too short").with_op("step");
        assert_eq!(untouched.to_string(), "persistence: header too short");
    }

    /// Every constructor in the argument bucket must produce exactly the
    /// variant and text a struct literal did, because downstream code has no
    /// other spelling once the variants are `#[non_exhaustive]`.
    #[test]
    fn argument_constructors_match_the_literal_spelling() {
        assert_eq!(
            Error::shape_mismatch("matmul", [2, 3], [4, 5]).to_string(),
            "matmul: shape mismatch: lhs [2, 3] vs rhs [4, 5]"
        );
        assert_eq!(
            Error::rank_mismatch("conv2d", 4, 3).to_string(),
            "conv2d: rank mismatch: expected rank 4, got rank 3"
        );
        assert_eq!(
            Error::dtype_mismatch("add", DType::F32, DType::I64).to_string(),
            "add: dtype mismatch: expected f32, got i64 \
             (no implicit promotion; cast explicitly with to_dtype)"
        );
        // Two `Device::Cpu`s make this constructor symmetric, so `to_string`
        // alone would pass with `expected` and `got` swapped. Pin the mapping
        // structurally, and again against two distinct devices where a second
        // one exists.
        let e = Error::device_mismatch("add", Device::Cpu, Device::Cpu);
        assert_eq!(e.to_string(), "add: device mismatch: expected cpu, got cpu");
        assert!(matches!(
            e,
            Error::DeviceMismatch {
                op: "add",
                expected: Device::Cpu,
                got: Device::Cpu,
            }
        ));
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            let e = Error::device_mismatch("add", Device::Metal(1), Device::Cpu);
            assert_eq!(
                e.to_string(),
                "add: device mismatch: expected metal:1, got cpu"
            );
            assert!(matches!(
                e,
                Error::DeviceMismatch {
                    op: "add",
                    expected: Device::Metal(1),
                    got: Device::Cpu,
                }
            ));
        }
        assert_eq!(
            Error::reshape_mismatch("reshape", [2, 3], [4]).to_string(),
            "reshape: cannot reshape [2, 3] into [4]"
        );
        assert_eq!(
            Error::index_out_of_bounds("gather", 7, 0, 3).to_string(),
            "gather: index 7 out of bounds for axis 0 of size 3"
        );
        assert_eq!(
            Error::unsupported("erf", Device::Cpu, DType::Bool).to_string(),
            "erf: unsupported on cpu for dtype bool"
        );
        assert_eq!(
            Error::invalid_arg("dropout", "p must be in [0, 1)").to_string(),
            "dropout: invalid argument: p must be in [0, 1)"
        );
    }
}
