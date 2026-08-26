//! Shared table-driven backend conformance tests against the CPU reference.
//!
//! Each case uploads host-visible operands, runs one backend entry point, and
//! compares the downloaded outputs. Targeted integration tests supplement this
//! table with public graphs and model-level behavior.
//!
//! # How it works
//!
//! A `Case` is a `Call` (one coarse [`BackendOps`] entry point plus its
//! non-view arguments) applied to a list of `Operand`s (contiguous host
//! data + the [`Layout`] to view it through). Running a case on a backend
//! means: upload every operand with
//! [`transfer_in`](BackendOps::transfer_in), invoke the entry point, then
//! download the result with [`transfer_out`](BackendOps::transfer_out) —
//! so the comparison is between *host-observable* values and never touches
//! backend internals. `run` does that twice (candidate and CPU) and diffs
//! the two host buffers.
//!
//! # Scope and policy
//!
//! - **Dtypes**: `F16`, `BF16`, `F32`, `F64`, `I64`, `Bool`, with
//!   dtype-appropriate tolerances. Each dtype is a separate row, so an
//!   accelerator can honestly report BF16 as unsupported without hiding its
//!   F16 coverage.
//! - **Layouts**: cases deliberately include transposed and broadcast
//!   views, because the kernel contract is stride-aware.
//! - **`Unsupported` is not a mismatch.** A backend that reports
//!   [`Error::Unsupported`] has declined the case; there are no silent
//!   fallbacks. Declared out-of-scope rows land in
//!   `Report::expected_unsupported`; any other decline lands in
//!   `Report::skipped`, which a promotion gate requires to be empty.
//! - **Fused ops are in the table.** The runner compares a list of downloaded
//!   outputs, so multi-output encodings such as saved normalization statistics
//!   and optimizer state are ordinary rows.
//!
//! CPU is the reference. Metal, CUDA, and WGPU reuse `run` for the rows their
//! capability declarations cover; the CPU self-check keeps the full table
//! runnable on every target.

use crate::backend::{
    ArgReduceOp, BackendOps, BinaryOp, CmpOp, Conv2dParams, ConvOp, FusedOp, ReduceOp, UnaryOp,
    View, conv_geometry::Conv2dGeometry, dispatch,
};
use crate::device::Device;
use crate::dtype::{DType, HostConv};
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::shape::Shape;
use crate::storage::{CpuStorage, Storage};

/// Default relative tolerance for float comparisons. Exact for CPU-vs-CPU;
/// the slack exists for accelerators whose fused-multiply-add ordering
/// differs from the reference loops.
const DEFAULT_TOL: f64 = 1e-5;
const F16_TOL: f64 = 2e-3;
const BF16_TOL: f64 = 2e-2;

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

/// One kernel operand: contiguous host data plus the [`Layout`] the kernel
/// sees it through (which may be transposed, narrowed, or broadcast).
pub(crate) struct Operand {
    host: CpuStorage,
    layout: Layout,
}

impl Operand {
    /// An operand viewed through the contiguous layout for `dims`.
    ///
    /// # Panics
    /// If `dims` does not describe `host` (a malformed table entry).
    pub(crate) fn new(host: CpuStorage, dims: &[usize]) -> Operand {
        let layout = Layout::contiguous(dims).expect("conformance table: valid dims");
        assert_eq!(
            layout.num_elements(),
            host.len(),
            "conformance table: operand data does not fill {dims:?}"
        );
        Operand { host, layout }
    }

    /// An operand viewed through an arbitrary (strided/broadcast) layout.
    pub(crate) fn strided(host: CpuStorage, layout: Layout) -> Operand {
        Operand { host, layout }
    }
}

/// One [`BackendOps`] entry point plus the arguments that are not views.
///
/// This mirrors the trait one-for-one, which is the point: the table can
/// only exercise the coarse enum-routed surface, so it stays valid as
/// kernels are added behind it.
pub(crate) enum Call {
    /// [`BackendOps::full`].
    Full {
        /// Element count to allocate.
        len: usize,
        /// Dtype of the allocation.
        dtype: DType,
        /// Fill value, narrowed to `dtype`.
        value: f64,
    },
    /// [`BackendOps::cast`] to the given dtype.
    Cast(DType),
    /// [`BackendOps::copy_strided`].
    CopyStrided,
    /// [`BackendOps::copy_into`]: allocate a `dst_dims` destination filled with
    /// `fill`, copy the single operand into the region `narrow(axis, start,
    /// len)` of it, and hand the **whole** destination back so the untouched
    /// surroundings are compared too.
    ///
    /// This is the only entry point whose result is a mutated buffer rather
    /// than a returned one, which is exactly why it needs a row: a kernel that
    /// ignores the destination layout writes plausible values into the wrong
    /// slots and nothing else in the table would notice.
    CopyInto {
        /// Shape of the freshly allocated destination.
        dst_dims: Vec<usize>,
        /// Value the destination is filled with before the copy.
        fill: f64,
        /// Axis the destination region is narrowed on.
        axis: usize,
        /// First index of the destination region.
        start: usize,
    },
    /// [`BackendOps::binary`].
    Binary(BinaryOp),
    /// [`BackendOps::binary_scalar`].
    BinaryScalar(BinaryOp, f64),
    /// [`BackendOps::unary`].
    Unary(UnaryOp),
    /// [`BackendOps::compare`].
    Compare(CmpOp),
    /// [`BackendOps::where_cond`] over `(cond, on_true, on_false)`.
    WhereCond,
    /// [`BackendOps::masked_fill`] over `(x, mask)` with the fill value.
    MaskedFill(f64),
    /// [`BackendOps::reduce`] along the axis.
    Reduce(ReduceOp, usize),
    /// [`BackendOps::arg_reduce`] along the axis.
    ArgReduce(ArgReduceOp, usize),
    /// [`BackendOps::matmul`].
    Matmul,
    /// [`BackendOps::index_select`] over `(x, indices)` along the axis.
    IndexSelect(usize),
    /// [`BackendOps::index_add`] over `(x, indices, src)` along the axis.
    IndexAdd(usize),
    /// [`BackendOps::gather`] over `(x, indices)` along the axis.
    Gather(usize),
    /// [`BackendOps::scatter_add`] over `(x, indices, src)` along the axis.
    ScatterAdd(usize),
    /// [`BackendOps::arg_sort`] along the axis. The only entry point whose
    /// result is a *permutation*: the output is the source position of each
    /// element in sorted order, so it is `I64` whatever the sorted dtype was.
    ArgSort {
        /// Axis whose lines are sorted.
        axis: usize,
        /// Sort direction; NaN orders above every number either way.
        descending: bool,
    },
    /// [`BackendOps::conv`] with the given geometry.
    Conv(ConvOp, Conv2dParams),
    /// [`BackendOps::fused`] with the given scalar tail. Multi-output by
    /// nature, which is why [`Call::apply`] returns a list.
    Fused(FusedOp, Vec<f64>),
}

impl Call {
    /// Invoke the entry point on `backend` with `views` bound in table order.
    ///
    /// Returns a **list** of outputs: every entry point but
    /// [`BackendOps::fused`] produces exactly one, and the fused encodings
    /// produce one, two, or three depending on the variant. The runner
    /// compares the lists element-wise, so an output *count* divergence is a
    /// reported failure rather than a silently ignored one.
    fn apply(&self, backend: &dyn BackendOps, views: &[View<'_>]) -> Result<Vec<Storage>> {
        let arity = |want: usize| Error::InvalidArg {
            op: "conformance",
            msg: format!("expected {want} operand view(s), got {}", views.len()),
        };
        let one = |storage: Result<Storage>| storage.map(|value| vec![value]);
        match (self, views) {
            (Call::Full { len, dtype, value }, []) => one(backend.full(*len, *dtype, *value)),
            (Call::Cast(to), [x]) => one(backend.cast(*x, *to)),
            (Call::CopyStrided, [x]) => one(backend.copy_strided(*x)),
            (
                Call::CopyInto {
                    dst_dims,
                    fill,
                    axis,
                    start,
                },
                [src],
            ) => {
                let dst_layout = Layout::contiguous(dst_dims.clone())?;
                let region = dst_layout.narrow(*axis, *start, src.layout().dims()[*axis])?;
                let mut dst = backend.full(dst_layout.num_elements(), src.dtype(), *fill)?;
                backend.copy_into(*src, &mut dst, &region)?;
                Ok(vec![dst])
            }
            (Call::Binary(op), [a, b]) => one(backend.binary(*op, *a, *b)),
            (Call::BinaryScalar(op, s), [x]) => one(backend.binary_scalar(*op, *x, *s)),
            (Call::Unary(op), [x]) => one(backend.unary(*op, *x)),
            (Call::Compare(op), [a, b]) => one(backend.compare(*op, *a, *b)),
            (Call::WhereCond, [c, t, f]) => one(backend.where_cond(*c, *t, *f)),
            (Call::MaskedFill(v), [x, m]) => one(backend.masked_fill(*x, *m, *v)),
            (Call::Reduce(op, axis), [x]) => one(backend.reduce(*op, *x, *axis)),
            (Call::ArgReduce(op, axis), [x]) => one(backend.arg_reduce(*op, *x, *axis)),
            (Call::Matmul, [a, b]) => one(backend.matmul(*a, *b)),
            (Call::IndexSelect(axis), [x, i]) => one(backend.index_select(*x, *axis, *i)),
            (Call::IndexAdd(axis), [x, i, s]) => one(backend.index_add(*x, *axis, *i, *s)),
            (Call::Gather(axis), [x, i]) => one(backend.gather(*x, *axis, *i)),
            (Call::ScatterAdd(axis), [x, i, s]) => one(backend.scatter_add(*x, *axis, *i, *s)),
            (Call::ArgSort { axis, descending }, [x]) => {
                one(backend.arg_sort(*x, *axis, *descending))
            }
            (Call::Conv(op, params), inputs) => one(backend.conv(*op, inputs, params)),
            (Call::Fused(op, scalars), inputs) => backend.fused(*op, inputs, scalars),
            (Call::Full { .. }, _) => Err(arity(0)),
            (
                Call::Cast(_)
                | Call::CopyStrided
                | Call::BinaryScalar(..)
                | Call::CopyInto { .. }
                | Call::Unary(_)
                | Call::Reduce(..)
                | Call::ArgReduce(..)
                | Call::ArgSort { .. },
                _,
            ) => Err(arity(1)),
            (
                Call::Binary(_)
                | Call::Compare(_)
                | Call::Matmul
                | Call::MaskedFill(_)
                | Call::IndexSelect(_)
                | Call::Gather(_),
                _,
            ) => Err(arity(2)),
            (Call::WhereCond | Call::IndexAdd(_) | Call::ScatterAdd(_), _) => Err(arity(3)),
        }
    }
}

/// Expected result category for one conformance row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum Outcome {
    /// The operation must return values.
    Value,
    /// Both backends must reject before producing a value.
    Rejected {
        /// The structured error variant expected from both backends.
        variant: &'static str,
    },
    /// Both backends are deliberately expected to decline the row.
    Declined,
}

/// One row of the conformance table.
pub(crate) struct Case {
    /// Human-readable identity, e.g. `"binary.Add.f32"`; appears verbatim in
    /// failure reports.
    pub(crate) name: String,
    call: Call,
    operands: Vec<Operand>,
    tol: f64,
    outcome: Outcome,
}

impl Case {
    /// A case with the default float tolerance and a successful result.
    fn new(name: String, call: Call, operands: Vec<Operand>) -> Case {
        Self::with_outcome(name, call, operands, Outcome::Value)
    }

    /// A case whose operation must reject with one named error variant.
    fn rejected(name: String, call: Call, operands: Vec<Operand>, variant: &'static str) -> Case {
        Self::with_outcome(name, call, operands, Outcome::Rejected { variant })
    }

    fn with_outcome(name: String, call: Call, operands: Vec<Operand>, outcome: Outcome) -> Case {
        let output_dtype = match &call {
            Call::Full { dtype, .. } | Call::Cast(dtype) => *dtype,
            Call::Compare(_) => DType::Bool,
            Call::ArgReduce(..) | Call::ArgSort { .. } => DType::I64,
            Call::WhereCond => operands[1].host.dtype(),
            Call::MaskedFill(_)
            | Call::CopyStrided
            | Call::CopyInto { .. }
            | Call::BinaryScalar(..)
            | Call::Unary(_)
            | Call::Reduce(..)
            | Call::IndexSelect(_)
            | Call::IndexAdd(..)
            | Call::Gather(..)
            | Call::ScatterAdd(..)
            | Call::Conv(..)
            | Call::Binary(_)
            | Call::Matmul
            | Call::Fused(..) => operands[0].host.dtype(),
        };
        let exact_output = matches!(
            call,
            Call::Full { .. }
                | Call::Cast(_)
                | Call::CopyStrided
                | Call::CopyInto { .. }
                | Call::Compare(_)
                | Call::WhereCond
                | Call::MaskedFill(_)
                | Call::ArgReduce(..)
                | Call::IndexSelect(_)
                | Call::Gather(..)
                | Call::ArgSort { .. }
        );
        let tol = if exact_output {
            0.0
        } else {
            dtype_tolerance(output_dtype)
        };
        Case {
            name,
            call,
            operands,
            tol,
            outcome,
        }
    }
}

fn dtype_tolerance(dtype: DType) -> f64 {
    match dtype {
        DType::F16 => F16_TOL,
        DType::BF16 => BF16_TOL,
        DType::F32 | DType::F64 => DEFAULT_TOL,
        DType::I64 | DType::Bool => 0.0,
    }
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

/// The outcome of a conformance run.
pub(crate) struct Report {
    /// Cases whose candidate output matched the CPU reference.
    pub(crate) matched: Vec<String>,
    /// Cases one of the two backends declined with [`Error::Unsupported`].
    /// Not a failure — but a promotion gate should require this to be empty.
    pub(crate) skipped: Vec<String>,
    /// Rows outside the candidate's declared capability, such as BF16 on
    /// Metal, or operation/dtype pairs the common backend contract excludes.
    pub(crate) expected_unsupported: Vec<String>,
    /// Cases where one backend rejected differently from the other.
    pub(crate) rejection_mismatches: Vec<String>,
    /// Cases that diverged or errored, one rendered message each.
    pub(crate) failures: Vec<String>,
}

impl Report {
    /// Turn the report into a single [`Result`], failing loudly with every
    /// divergence listed. `device` names the backend under test.
    pub(crate) fn into_result(self, device: Device) -> Result<()> {
        if self.failures.is_empty() && self.rejection_mismatches.is_empty() {
            return Ok(());
        }
        let mut failures = self.failures;
        failures.extend(self.rejection_mismatches);
        Err(Error::Backend {
            op: "conformance",
            msg: format!(
                "{} of {} case(s) diverged from the cpu reference on {device}:\n  {}",
                failures.len(),
                self.matched.len()
                    + self.skipped.len()
                    + self.expected_unsupported.len()
                    + failures.len(),
                failures.join("\n  ")
            ),
        })
    }
}

/// Return the stable variant label used by rejection-parity rows.
fn outcome_variant(result: &Result<Vec<CpuStorage>>) -> &'static str {
    let Err(error) = result else {
        return "Value";
    };
    if matches!(error, Error::ShapeMismatch { .. }) {
        "ShapeMismatch"
    } else if matches!(error, Error::RankMismatch { .. }) {
        "RankMismatch"
    } else if matches!(error, Error::InvalidAxis { .. }) {
        "InvalidAxis"
    } else if matches!(error, Error::DTypeMismatch { .. }) {
        "DTypeMismatch"
    } else if matches!(error, Error::DeviceMismatch { .. }) {
        "DeviceMismatch"
    } else if matches!(error, Error::ReshapeMismatch { .. }) {
        "ReshapeMismatch"
    } else if matches!(error, Error::IndexOutOfBounds { .. }) {
        "IndexOutOfBounds"
    } else if matches!(error, Error::Unsupported { .. }) {
        "Unsupported"
    } else if matches!(error, Error::InvalidArg { .. }) {
        "InvalidArg"
    } else if matches!(error, Error::Backend { .. }) {
        "Backend"
    } else {
        "Other"
    }
}

/// Run the standard [`suite`] on `candidate` (labelled `device`) against the
/// CPU reference backend.
pub(crate) fn run(candidate: &dyn BackendOps, device: Device) -> Report {
    let reference = dispatch::backend(Device::Cpu);
    let mut report = Report {
        matched: Vec::new(),
        skipped: Vec::new(),
        expected_unsupported: Vec::new(),
        rejection_mismatches: Vec::new(),
        failures: Vec::new(),
    };
    for case in suite() {
        let reference_result = evaluate(reference, &case);
        let candidate_result = evaluate(candidate, &case);
        match case.outcome {
            Outcome::Rejected { variant } => {
                let reference_variant = outcome_variant(&reference_result);
                let candidate_variant = outcome_variant(&candidate_result);
                if reference_variant == variant && candidate_variant == variant {
                    report.matched.push(case.name);
                } else if candidate_variant == "Unsupported" && expected_unsupported(device, &case)
                {
                    report.expected_unsupported.push(case.name);
                } else {
                    report.rejection_mismatches.push(format!(
                        "{}: expected rejection {variant}, cpu={reference_variant}, \
                         {device}={candidate_variant}",
                        case.name
                    ));
                }
            }
            Outcome::Declined => {
                let reference_variant = outcome_variant(&reference_result);
                let candidate_variant = outcome_variant(&candidate_result);
                if reference_variant == "Unsupported" && candidate_variant == "Unsupported" {
                    report.matched.push(case.name);
                } else {
                    report.rejection_mismatches.push(format!(
                        "{}: expected Declined, cpu={reference_variant}, \
                         {device}={candidate_variant}",
                        case.name
                    ));
                }
            }
            Outcome::Value => match (reference_result, candidate_result) {
                (Err(Error::Unsupported { .. }), _) | (_, Err(Error::Unsupported { .. })) => {
                    if expected_unsupported(device, &case) {
                        report.expected_unsupported.push(case.name);
                    } else {
                        report.skipped.push(case.name);
                    }
                }
                (Err(e), _) => report
                    .failures
                    .push(format!("{}: cpu reference failed: {e}", case.name)),
                (_, Err(e)) => report
                    .failures
                    .push(format!("{}: {device} backend failed: {e}", case.name)),
                (Ok(want), Ok(got)) => match compare_all(&want, &got, case.tol) {
                    Ok(()) => report.matched.push(case.name),
                    Err(diff) => report.failures.push(format!("{}: {diff}", case.name)),
                },
            },
        }
    }
    report
}

/// The ops that exist **only** in the CPU kernel set.
///
/// Not a contract and not a capability declaration: a hole, written down.
/// Metal, WGPU, and CUDA implement none of these and return
/// [`Error::Unsupported`], so every row naming one would otherwise land in
/// `Report::skipped` and fail all three lane tests, which require that set to
/// be empty.
///
/// This list *is* the record of that hole — it is the only place the crate
/// admits it, so it is deliberately one explicit list of named variants rather
/// than arms scattered through [`expected_unsupported`] or folded into the
/// common-contract tier, where it would read as a rule instead of a debt.
/// Nothing here is out of scope for an accelerator; every entry is a kernel
/// nobody has written yet.
///
/// **An entry must be deleted the moment its kernel lands.** That deletion is
/// the promotion gate: it is what starts the row being compared against the
/// CPU reference for real. An entry left behind after the kernel exists goes
/// on silently excusing a kernel that could be checked.
fn has_no_accelerator_kernel(call: &Call) -> bool {
    matches!(
        call,
        Call::Binary(BinaryOp::Pow)
            | Call::BinaryScalar(BinaryOp::Pow, _)
            | Call::Unary(
                UnaryOp::Sign
                    | UnaryOp::Recip
                    | UnaryOp::Floor
                    | UnaryOp::Ceil
                    | UnaryOp::Round
                    | UnaryOp::Erf
            )
            | Call::Reduce(ReduceOp::Prod, _)
            | Call::ArgSort { .. }
    )
}

/// Whether a declined case is *declared* out of contract rather than a hole.
///
/// Three tiers, and the distinction matters. The first is the **common**
/// backend contract — combinations the op enums themselves put out of scope,
/// which every backend including the CPU reference declines, so the row exists
/// to prove the decline is loud and universal. The second is the
/// **accelerator gap** of [`has_no_accelerator_kernel`], which is the one tier
/// that *is* a hole rather than a rule, and is written as a single named list
/// for exactly that reason. The third is a **per-backend** capability
/// declaration (Metal has no BF16 and no F64 storage at all).
///
/// Outside that middle tier, nothing here may name a combination one backend
/// implements and another merely has not got round to: that is the entry that
/// would silently excuse a real gap, so every arm below is justified by a rule
/// stated in [`crate::backend`] or in the kernel module that enforces it.
fn expected_unsupported(device: Device, case: &Case) -> bool {
    let input_dtype = case.operands.first().map(|operand| operand.host.dtype());
    // Left un-nested on purpose: each top-level `|` group below is one rule
    // from the doc comment above, with its own justifying comment; flattening
    // into one mega or-pattern (as clippy suggests) would erase that mapping.
    #[allow(clippy::unnested_or_patterns)]
    let outside_common_contract =
        matches!(
            (&case.call, input_dtype),
            // `UnaryOp`'s doc comment: the transcendental unaries are float-only,
            // and so are the rounding and reciprocal ones — `Neg`/`Abs` are the
            // whole of the integer unary surface.
            (
            Call::Unary(
                UnaryOp::Relu
                    | UnaryOp::Gelu
                    | UnaryOp::Exp
                    | UnaryOp::Ln
                    | UnaryOp::Sqrt
                    | UnaryOp::Tanh
                    | UnaryOp::Sigmoid
                    | UnaryOp::Sign
                    | UnaryOp::Recip
                    | UnaryOp::Floor
                    | UnaryOp::Ceil
                    | UnaryOp::Round
                    | UnaryOp::Erf
            ),
            Some(DType::I64)
        )
        // `BinaryOp::Pow`'s doc comment: an integer lane would have to invent a
        // meaning for a fractional or negative exponent, so it declines rather
        // than answer differently per device.
            | (
                Call::Binary(BinaryOp::Pow) | Call::BinaryScalar(BinaryOp::Pow, _),
                Some(DType::I64)
            )
        // `bool` has no `NumAcc`, so it has no accumulation and hence no
        // reduction, arg-reduction, sort, matmul, conv, or accumulating scatter
        // (see `cpu::acc`'s module docs). `Bool` also has no arithmetic:
        // comparisons live in `compare`, and `where`/`masked_fill` carry it.
            | (
                Call::Reduce(..) | Call::ArgReduce(..) | Call::ArgSort { .. },
                Some(DType::Bool)
            )
            | (Call::Matmul | Call::Conv(..), Some(DType::Bool))
            | (Call::IndexAdd(_) | Call::ScatterAdd(_), Some(DType::Bool))
            | (Call::Binary(_) | Call::BinaryScalar(..) | Call::Unary(_), Some(DType::Bool))
        // `i64` has no `FloatAcc`, which is what makes softmax, LayerNorm, and
        // the optimizer steps decline an integer dtype instead of computing
        // something numerically meaningless.
            | (Call::Fused(..), Some(DType::I64 | DType::Bool))
        // `BackendOps::cast`'s doc comment: the F64 conversion lanes are
        // deferred, and declining is required to be loud rather than a silent
        // reinterpretation.
            | (Call::Cast(DType::F64), _)
        ) || matches!((&case.call, input_dtype), (Call::Cast(_), Some(DType::F64)));
    if outside_common_contract {
        return true;
    }
    // Tier two. Gated on the candidate being an accelerator: the CPU reference
    // implements every one of these, so the self-check must keep comparing
    // them for real and would go blind the moment this list covered it too.
    if !matches!(device, Device::Cpu) && has_no_accelerator_kernel(&case.call) {
        return true;
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if matches!(device, Device::Metal(_)) {
        return case
            .operands
            .iter()
            .any(|operand| matches!(operand.host.dtype(), DType::BF16 | DType::F64))
            || matches!(
                case.call,
                Call::Full {
                    dtype: DType::BF16 | DType::F64,
                    ..
                } | Call::Cast(DType::BF16 | DType::F64)
            );
    }
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    if matches!(device, Device::Cuda(_)) {
        return case
            .operands
            .iter()
            .any(|operand| matches!(operand.host.dtype(), DType::BF16 | DType::F64))
            || matches!(
                case.call,
                Call::Full {
                    dtype: DType::BF16 | DType::F64,
                    ..
                } | Call::Cast(DType::BF16 | DType::F64)
            );
    }
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    if matches!(device, Device::Wgpu(_)) {
        let f16 = crate::backend::wgpu::supports_f16(device);
        let float_supported = |dtype| dtype == DType::F32 || (dtype == DType::F16 && f16);
        let supported = match (&case.call, input_dtype) {
            (Call::Full { dtype, .. }, _) => {
                matches!(dtype, DType::F32 | DType::I64 | DType::Bool)
                    || (*dtype == DType::F16 && f16)
            }
            (Call::CopyStrided | Call::CopyInto { .. }, Some(dtype)) => {
                matches!(dtype, DType::F32 | DType::I64 | DType::Bool)
                    || (dtype == DType::F16 && f16)
            }
            (Call::IndexSelect(_), Some(dtype)) => {
                matches!(dtype, DType::F32 | DType::I64 | DType::Bool)
                    || (dtype == DType::F16 && f16)
            }
            (Call::Cast(to), Some(from)) => {
                matches!(
                    (from, to),
                    (DType::F32, DType::Bool) | (DType::Bool, DType::F32)
                ) || (f16
                    && (from == DType::F16 || *to == DType::F16)
                    && matches!(from, DType::F16 | DType::F32 | DType::I64 | DType::Bool)
                    && matches!(to, DType::F16 | DType::F32 | DType::I64 | DType::Bool))
            }
            (Call::WhereCond, _) => case
                .operands
                .get(1)
                .is_some_and(|o| float_supported(o.host.dtype())),
            (Call::Fused(FusedOp::Softmax, scalars), Some(dtype)) => {
                scalars.is_empty() && float_supported(dtype)
            }
            (Call::Fused(FusedOp::LayerNorm, scalars), Some(dtype)) => {
                scalars.len() == 1 && case.operands.len() == 3 && float_supported(dtype)
            }
            // `Fused(..)` must stay ahead of the `Some`/`None` catch-alls: any
            // other fused op is unsupported regardless of dtype, which the
            // catch-alls' `float_supported` would get wrong if reordered.
            #[allow(clippy::match_same_arms)]
            (Call::Fused(..), _) => false,
            (_, Some(dtype)) => float_supported(dtype),
            (_, None) => false,
        };
        return !supported;
    }
    false
}

/// Run the suite against the backend registered for `device`.
pub(crate) fn run_device(device: Device) -> Report {
    run(dispatch::backend(device), device)
}

/// Upload the case's operands to `backend`, invoke the entry point, and
/// download the result as host data.
///
/// The output is downloaded through a flat contiguous layout over the whole
/// returned buffer: kernels return dense row-major storage, so this observes
/// every element (and any length divergence) without the table having to
/// restate each op's output shape.
fn evaluate(backend: &dyn BackendOps, case: &Case) -> Result<Vec<CpuStorage>> {
    let storages = case
        .operands
        .iter()
        .map(|o| backend.transfer_in(o.host.clone()))
        .collect::<Result<Vec<Storage>>>()?;
    let views: Vec<View<'_>> = storages
        .iter()
        .zip(&case.operands)
        .map(|(s, o)| View::new(s, &o.layout))
        .collect();
    case.call
        .apply(backend, &views)?
        .iter()
        .map(|out| {
            let layout = Layout::contiguous([out.len()])?;
            backend.transfer_out(View::new(out, &layout))
        })
        .collect()
}

/// Compare two output *lists*, naming the diverging output when there is more
/// than one. A differing output count is itself a divergence.
fn compare_all(
    want: &[CpuStorage],
    got: &[CpuStorage],
    tol: f64,
) -> std::result::Result<(), String> {
    if want.len() != got.len() {
        return Err(format!("output count {} != {}", want.len(), got.len()));
    }
    for (index, (a, b)) in want.iter().zip(got).enumerate() {
        compare(a, b, tol).map_err(|diff| {
            if want.len() == 1 {
                diff
            } else {
                format!("output {index}: {diff}")
            }
        })?;
    }
    Ok(())
}

/// Compare two downloaded buffers, describing the first divergence.
fn compare(want: &CpuStorage, got: &CpuStorage, tol: f64) -> std::result::Result<(), String> {
    if want.dtype() != got.dtype() {
        return Err(format!("dtype {} != {}", want.dtype(), got.dtype()));
    }
    if want.len() != got.len() {
        return Err(format!("length {} != {}", want.len(), got.len()));
    }
    match (want, got) {
        (CpuStorage::F32(a), CpuStorage::F32(b)) => diff_float(
            a.iter().map(|&v| f64::from(v)),
            b.iter().map(|&v| f64::from(v)),
            tol,
        ),
        (CpuStorage::F64(a), CpuStorage::F64(b)) => {
            diff_float(a.iter().copied(), b.iter().copied(), tol)
        }
        (CpuStorage::F16(a), CpuStorage::F16(b)) => diff_float(
            a.iter().map(|v| v.to_f64()),
            b.iter().map(|v| v.to_f64()),
            tol,
        ),
        (CpuStorage::BF16(a), CpuStorage::BF16(b)) => diff_float(
            a.iter().map(|v| v.to_f64()),
            b.iter().map(|v| v.to_f64()),
            tol,
        ),
        // Integers and booleans are exact: any difference is a bug, never
        // rounding.
        (CpuStorage::I64(a), CpuStorage::I64(b)) => diff_exact(a, b),
        (CpuStorage::Bool(a), CpuStorage::Bool(b)) => diff_exact(a, b),
        // Unreachable: the dtype tags were compared above.
        _ => Err("mismatched storage variants".to_string()),
    }
}

/// Element-wise float scan reporting the first position outside `tol`.
fn diff_float<I, J>(want: I, got: J, tol: f64) -> std::result::Result<(), String>
where
    I: Iterator<Item = f64>,
    J: Iterator<Item = f64>,
{
    for (i, (a, b)) in want.zip(got).enumerate() {
        if !close(a, b, tol) {
            return Err(format!("element {i}: expected {a}, got {b}"));
        }
    }
    Ok(())
}

/// Element-wise exact scan for the dtypes rounding cannot excuse.
fn diff_exact<T>(want: &[T], got: &[T]) -> std::result::Result<(), String>
where
    T: PartialEq + std::fmt::Debug,
{
    for (i, (a, b)) in want.iter().zip(got).enumerate() {
        if a != b {
            return Err(format!("element {i}: expected {a:?}, got {b:?}"));
        }
    }
    Ok(())
}

/// Mixed absolute/relative closeness with NaN treated as equal to NaN (a
/// kernel that must produce NaN — `ln(-1)`, `0/0` — should keep producing
/// it on every backend).
fn close(a: f64, b: f64, tol: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    if a == b {
        return true; // exact, including matching infinities
    }
    if !a.is_finite() || !b.is_finite() {
        return false;
    }
    (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
}

// ---------------------------------------------------------------------------
// The table
// ---------------------------------------------------------------------------

/// Host data helpers. Each builds a contiguous operand of the named dtype.
fn f32s(dims: &[usize], data: &[f32]) -> Operand {
    Operand::new(HostConv::into_cpu_storage(data.to_vec()), dims)
}

fn reduceds(dtype: DType, dims: &[usize], data: &[f32]) -> Operand {
    let host = match dtype {
        DType::F16 => {
            HostConv::into_cpu_storage(data.iter().copied().map(half::f16::from_f32).collect())
        }
        DType::BF16 => {
            HostConv::into_cpu_storage(data.iter().copied().map(half::bf16::from_f32).collect())
        }
        _ => panic!("conformance reduced operand requires F16 or BF16"),
    };
    Operand::new(host, dims)
}

fn reduceds_strided(dtype: DType, data: &[f32], layout: Layout) -> Operand {
    let host = reduceds(dtype, &[data.len()], data).host;
    Operand::strided(host, layout)
}

fn reduceds_transposed(dtype: DType, data: &[f32]) -> Operand {
    let layout = Layout::contiguous([2, 3])
        .and_then(|layout| layout.transpose(0, 1))
        .expect("conformance table: transposable reduced layout");
    reduceds_strided(dtype, data, layout)
}

fn reduceds_broadcast(dtype: DType, data: &[f32]) -> Operand {
    let layout = Layout::contiguous([3, 1])
        .and_then(|layout| layout.broadcast_to(&Shape::from([3, 2])))
        .expect("conformance table: broadcastable reduced layout");
    reduceds_strided(dtype, data, layout)
}

/// A float operand of **any** float dtype, built from one `f32` source array.
///
/// `reduceds` above covers only the two reduced dtypes; this covers the four,
/// so a table row can be written once and swept over `[F16, BF16, F32, F64]`.
/// That sweep is what puts `F64` — a dtype the CPU kernels implement in full
/// and the table previously never touched — under the harness.
///
/// # Panics
/// On a non-float dtype (a malformed table entry).
fn floats(dtype: DType, dims: &[usize], data: &[f32]) -> Operand {
    Operand::new(float_host(dtype, data), dims)
}

/// [`floats`] viewed through an arbitrary layout.
fn floats_strided(dtype: DType, data: &[f32], layout: Layout) -> Operand {
    Operand::strided(float_host(dtype, data), layout)
}

fn float_host(dtype: DType, data: &[f32]) -> CpuStorage {
    match dtype {
        DType::F16 => {
            HostConv::into_cpu_storage(data.iter().copied().map(half::f16::from_f32).collect())
        }
        DType::BF16 => {
            HostConv::into_cpu_storage(data.iter().copied().map(half::bf16::from_f32).collect())
        }
        DType::F32 => HostConv::into_cpu_storage(data.to_vec()),
        DType::F64 => {
            HostConv::into_cpu_storage(data.iter().map(|&v| f64::from(v)).collect::<Vec<f64>>())
        }
        other => panic!("conformance float operand requires a float dtype, got {other}"),
    }
}

/// Every float dtype the CPU reference implements. Metal declares BF16 and F64
/// outside its capability, so those rows land in `expected_unsupported` there
/// and still hold the CPU reference to account.
const FLOATS: [DType; 4] = [DType::F16, DType::BF16, DType::F32, DType::F64];

fn i64s(dims: &[usize], data: &[i64]) -> Operand {
    Operand::new(HostConv::into_cpu_storage(data.to_vec()), dims)
}

fn bools(dims: &[usize], data: &[bool]) -> Operand {
    Operand::new(HostConv::into_cpu_storage(data.to_vec()), dims)
}

fn bools_broadcast(data: &[bool]) -> Operand {
    let layout = Layout::contiguous([3, 1])
        .and_then(|layout| layout.broadcast_to(&Shape::from([3, 2])))
        .expect("conformance table: broadcastable bool layout");
    Operand::strided(HostConv::into_cpu_storage(data.to_vec()), layout)
}

/// A `[2, 3]` f32 operand viewed transposed to `[3, 2]` — the stride-aware
/// path every kernel must honor.
fn f32s_transposed(data: &[f32]) -> Operand {
    let layout = Layout::contiguous([2, 3])
        .and_then(|l| l.transpose(0, 1))
        .expect("conformance table: transposable layout");
    Operand::strided(HostConv::into_cpu_storage(data.to_vec()), layout)
}

/// A `[3, 1]` f32 operand broadcast to `[3, 2]` (stride-0 axis).
fn f32s_broadcast(data: &[f32]) -> Operand {
    let layout = Layout::contiguous([3, 1])
        .and_then(|l| l.broadcast_to(&Shape::from([3, 2])))
        .expect("conformance table: broadcastable layout");
    Operand::strided(HostConv::into_cpu_storage(data.to_vec()), layout)
}

/// Signed f32 sample data, shape `[2, 3]`.
const A_F32: [f32; 6] = [1.0, -2.0, 3.0, -4.0, 5.0, 6.5];
/// Second f32 operand, shape `[2, 3]`; no zeros, so `div` stays finite.
const B_F32: [f32; 6] = [0.5, 2.0, -1.0, 4.0, 0.25, -3.0];
/// Strictly positive f32 data for the `ln`/`sqrt` domain, shape `[2, 3]`.
const P_F32: [f32; 6] = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0];
/// Signed i64 sample data, shape `[2, 3]`.
const A_I64: [i64; 6] = [1, -2, 3, -4, 5, 6];
/// Second i64 operand; the zero exercises the guarded integer division.
const B_I64: [i64; 6] = [2, 3, -1, 4, 0, 5];
/// Bool sample data, shape `[2, 3]`.
const A_BOOL: [bool; 6] = [true, false, true, false, true, false];
/// Second bool operand, shape `[2, 3]`.
const B_BOOL: [bool; 6] = [true, true, false, false, true, true];

/// The full op × dtype table (see the module docs for scope).
///
/// # Panics
/// If a table entry is malformed (data that does not fill its declared
/// dims, or an impossible layout) — a bug in this file, caught by the
/// CPU self-check test.
pub(crate) fn suite() -> Vec<Case> {
    let mut cases = Vec::new();
    push_host(&mut cases);
    push_elementwise(&mut cases);
    push_reduce(&mut cases);
    push_matmul(&mut cases);
    push_index(&mut cases);
    push_conv(&mut cases);
    push_fused(&mut cases);
    push_edges(&mut cases);
    push_rejection_cases(&mut cases);
    cases
}

/// Softmax, `LayerNorm` (plain, recorded, and its input gradient), and the two
/// optimizer steps, over every float dtype and both layout paths.
///
/// These are the rows that used to be missing entirely. Nothing else in the
/// suite reaches [`BackendOps::fused`], so before this section a backend could
/// return anything at all from any fused variant — or, worse, return *nearly*
/// the right thing for a dtype no bespoke test happened to cover — and the
/// conformance gate would still be green.
fn push_fused(cases: &mut Vec<Case>) {
    // Two rows of four, deliberately asymmetric and signed: a symmetric row
    // hides a reversed traversal of the normalized axis.
    const X: [f32; 8] = [0.5, -1.5, 2.0, 0.25, -0.75, 1.25, -2.5, 3.0];
    const G: [f32; 8] = [1.0, -0.5, 0.25, 2.0, -1.25, 0.75, 1.5, -2.0];
    const W: [f32; 4] = [1.5, -0.5, 2.0, 0.75];
    const B: [f32; 4] = [0.25, -1.0, 0.5, 2.0];
    // Saved statistics for the backward encoding, in the accumulation dtype.
    const XHAT: [f32; 8] = [-0.25, 1.5, -0.75, 0.5, 2.0, -1.25, 0.25, -1.0];
    const INV_STD: [f32; 2] = [1.25, 0.5];
    // Optimizer state. `V` is strictly positive: Adam takes its square root.
    const P: [f32; 6] = [0.5, -1.5, 2.0, -0.25, 1.75, -3.0];
    const PG: [f32; 6] = [0.25, 1.0, -0.5, 2.0, -1.25, 0.75];
    const M: [f32; 6] = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6];
    const V: [f32; 6] = [0.04, 0.09, 0.16, 0.25, 0.36, 0.49];

    for dtype in FLOATS {
        let acc = dtype.accumulation_dtype();

        cases.push(Case::new(
            format!("fused.Softmax.{dtype}"),
            Call::Fused(FusedOp::Softmax, vec![]),
            vec![floats(dtype, &[2, 4], &X)],
        ));
        // A transposed source makes the normalized axis strided, which is the
        // only thing separating a stride-aware row walk from `base + c`.
        let transposed = Layout::contiguous([2, 4])
            .and_then(|layout| layout.transpose(0, 1))
            .expect("conformance table: transposable softmax layout");
        cases.push(Case::new(
            format!("fused.Softmax.{dtype}.strided"),
            Call::Fused(FusedOp::Softmax, vec![]),
            vec![floats_strided(dtype, &X, transposed)],
        ));
        // An offset view: the row walk must start from `layout.offset()`, not 0.
        let offset = Layout::from_parts(Shape::from([2, 3]), Box::from([3usize, 1]), 2)
            .expect("conformance table: offset softmax layout");
        cases.push(Case::new(
            format!("fused.Softmax.{dtype}.offset"),
            Call::Fused(FusedOp::Softmax, vec![]),
            vec![floats_strided(dtype, &X, offset)],
        ));

        for (label, scalars) in [
            ("fused.LayerNorm", vec![1e-3]),
            ("fused.LayerNorm.saved", vec![1e-3, 1.0]),
        ] {
            cases.push(Case::new(
                format!("{label}.{dtype}"),
                Call::Fused(FusedOp::LayerNorm, scalars.clone()),
                vec![
                    floats(dtype, &[2, 4], &X),
                    floats(dtype, &[4], &W),
                    floats(dtype, &[4], &B),
                ],
            ));
            // Strided `x` with strided rank-1 affine views: `weight`/`bias` are
            // read through their own strides, which a kernel that indexes them
            // as `w[c]` gets wrong without ever producing a NaN.
            let strided_x = Layout::contiguous([4, 2])
                .and_then(|layout| layout.transpose(0, 1))
                .expect("conformance table: transposable layer-norm layout");
            let stride2 = Layout::from_parts(Shape::from([4]), Box::from([2usize]), 0)
                .expect("conformance table: strided affine layout");
            cases.push(Case::new(
                format!("{label}.{dtype}.strided"),
                Call::Fused(FusedOp::LayerNorm, scalars),
                vec![
                    floats_strided(dtype, &X, strided_x),
                    floats_strided(dtype, &G, stride2.clone()),
                    floats_strided(dtype, &X, stride2),
                ],
            ));
        }

        cases.push(Case::new(
            format!("fused.LayerNormBackward.{dtype}"),
            Call::Fused(FusedOp::LayerNorm, vec![]),
            vec![
                floats(dtype, &[2, 4], &G),
                floats(acc, &[2, 4], &XHAT),
                floats(acc, &[2, 1], &INV_STD),
                floats(dtype, &[4], &W),
            ],
        ));

        for (label, momentum) in [("plain", 0.0), ("momentum", 0.9)] {
            cases.push(Case::new(
                format!("fused.SgdStep.{label}.{dtype}"),
                Call::Fused(FusedOp::SgdStep, vec![0.1, momentum, 0.01]),
                vec![floats(dtype, &[2, 3], &P), floats(dtype, &[2, 3], &PG)],
            ));
        }
        // The three-input form: a velocity carried in the accumulation dtype.
        cases.push(Case::new(
            format!("fused.SgdStep.velocity.{dtype}"),
            Call::Fused(FusedOp::SgdStep, vec![0.1, 0.9, 0.01]),
            vec![
                floats(dtype, &[2, 3], &P),
                floats(dtype, &[2, 3], &PG),
                floats(acc, &[2, 3], &M),
            ],
        ));
        // Strided parameter and gradient: the Metal optimizer kernels
        // densify with `copy_strided` first, a path no other test reaches.
        let strided_p = Layout::contiguous([3, 2])
            .and_then(|layout| layout.transpose(0, 1))
            .expect("conformance table: transposable optimizer layout");
        cases.push(Case::new(
            format!("fused.SgdStep.strided.{dtype}"),
            Call::Fused(FusedOp::SgdStep, vec![0.1, 0.9, 0.01]),
            vec![
                floats_strided(dtype, &P, strided_p.clone()),
                floats_strided(dtype, &PG, strided_p.clone()),
                floats_strided(acc, &M, strided_p.clone()),
            ],
        ));

        for (label, decoupled) in [("coupled", 0.0), ("decoupled", 1.0)] {
            cases.push(Case::new(
                format!("fused.AdamStep.{label}.{dtype}"),
                Call::Fused(
                    FusedOp::AdamStep,
                    vec![0.1, 0.9, 0.999, 1e-8, 0.01, 0.19, 0.002, decoupled],
                ),
                vec![
                    floats(dtype, &[2, 3], &P),
                    floats(dtype, &[2, 3], &PG),
                    floats(acc, &[2, 3], &M),
                    floats(acc, &[2, 3], &V),
                ],
            ));
        }
        cases.push(Case::new(
            format!("fused.AdamStep.strided.{dtype}"),
            Call::Fused(
                FusedOp::AdamStep,
                vec![0.1, 0.9, 0.999, 1e-8, 0.0, 0.19, 0.002, 0.0],
            ),
            vec![
                floats_strided(dtype, &P, strided_p.clone()),
                floats_strided(dtype, &PG, strided_p.clone()),
                floats_strided(acc, &M, strided_p.clone()),
                floats_strided(acc, &V, strided_p),
            ],
        ));
    }

    // The non-float dtypes: every fused variant must decline them *loudly*,
    // which is the `FloatAcc` gate working. A backend that quietly computed
    // something here would be caught as an unexpected match, not a skip.
    cases.push(Case::new(
        "fused.Softmax.i64".to_string(),
        Call::Fused(FusedOp::Softmax, vec![]),
        vec![i64s(&[2, 3], &A_I64)],
    ));
    cases.push(Case::new(
        "fused.Softmax.bool".to_string(),
        Call::Fused(FusedOp::Softmax, vec![]),
        vec![bools(&[2, 3], &A_BOOL)],
    ));
    cases.push(Case::new(
        "fused.LayerNorm.i64".to_string(),
        Call::Fused(FusedOp::LayerNorm, vec![1e-3]),
        vec![
            i64s(&[2, 3], &A_I64),
            i64s(&[3], &[1, 2, 3]),
            i64s(&[3], &[0, 1, 0]),
        ],
    ));
    cases.push(Case::new(
        "fused.SgdStep.i64".to_string(),
        Call::Fused(FusedOp::SgdStep, vec![0.1, 0.0, 0.0]),
        vec![i64s(&[2, 3], &A_I64), i64s(&[2, 3], &B_I64)],
    ));
    cases.push(Case::new(
        "fused.AdamStep.i64".to_string(),
        Call::Fused(
            FusedOp::AdamStep,
            vec![0.1, 0.9, 0.999, 1e-8, 0.0, 0.19, 0.002, 0.0],
        ),
        vec![
            i64s(&[2, 3], &A_I64),
            i64s(&[2, 3], &B_I64),
            i64s(&[2, 3], &A_I64),
            i64s(&[2, 3], &B_I64),
        ],
    ));
}

/// Allocation, cast, and strided-materialization cases.
fn push_host(cases: &mut Vec<Case>) {
    for (dtype, value) in [
        (DType::F16, -1.5),
        (DType::BF16, -1.5),
        (DType::F32, -1.5),
        (DType::I64, 7.0),
        (DType::Bool, 1.0),
    ] {
        cases.push(Case::new(
            format!("full.{dtype}"),
            Call::Full {
                len: 6,
                dtype,
                value,
            },
            vec![],
        ));
    }
    let sources = [
        ("f32", f32s(&[2, 3], &A_F32) as Operand, DType::F32),
        ("i64", i64s(&[2, 3], &A_I64), DType::I64),
        ("bool", bools(&[2, 3], &A_BOOL), DType::Bool),
    ];
    for (label, operand, from) in sources {
        for to in [DType::F32, DType::I64, DType::Bool] {
            if to == from {
                continue;
            }
            cases.push(Case::new(
                format!("cast.{label}_to_{to}"),
                Call::Cast(to),
                vec![clone_operand(&operand)],
            ));
        }
    }
    for dtype in [DType::F16, DType::BF16] {
        let label = dtype.to_string();
        let reduced = reduceds(dtype, &[2, 3], &A_F32);
        for to in [DType::F32, DType::I64, DType::Bool] {
            cases.push(Case::new(
                format!("cast.{label}_to_{to}"),
                Call::Cast(to),
                vec![clone_operand(&reduced)],
            ));
        }
        for from in [DType::F32, DType::I64, DType::Bool] {
            let operand = match from {
                DType::F32 => f32s(&[2, 3], &A_F32),
                DType::I64 => i64s(&[2, 3], &A_I64),
                DType::Bool => bools(&[2, 3], &A_BOOL),
                _ => unreachable!(),
            };
            cases.push(Case::new(
                format!("cast.{from}_to_{label}"),
                Call::Cast(dtype),
                vec![operand],
            ));
        }
        let other = if dtype == DType::F16 {
            DType::BF16
        } else {
            DType::F16
        };
        cases.push(Case::new(
            format!("cast.{label}_to_{other}"),
            Call::Cast(other),
            vec![reduced],
        ));
        cases.push(Case::new(
            format!("cast.{label}_to_f32.transposed"),
            Call::Cast(DType::F32),
            vec![reduceds_transposed(dtype, &A_F32)],
        ));
        cases.push(Case::new(
            format!("cast.{label}_to_f32.broadcast"),
            Call::Cast(DType::F32),
            vec![reduceds_broadcast(dtype, &[1.0, -2.0, 3.0])],
        ));
        cases.push(Case::new(
            format!("copy_strided.{label}.transposed"),
            Call::CopyStrided,
            vec![reduceds_transposed(dtype, &A_F32)],
        ));
        cases.push(Case::new(
            format!("copy_strided.{label}.broadcast"),
            Call::CopyStrided,
            vec![reduceds_broadcast(dtype, &[1.0, -2.0, 3.0])],
        ));
    }
    cases.push(Case::new(
        "copy_strided.f32.transposed".to_string(),
        Call::CopyStrided,
        vec![f32s_transposed(&A_F32)],
    ));
    cases.push(Case::new(
        "copy_strided.f32.broadcast".to_string(),
        Call::CopyStrided,
        vec![f32s_broadcast(&[1.0, 2.0, 3.0])],
    ));
    cases.push(Case::new(
        "copy_strided.i64.contiguous".to_string(),
        Call::CopyStrided,
        vec![i64s(&[2, 3], &A_I64)],
    ));
}

/// Cheap operand duplication (host buffers are `Arc`-shared).
fn clone_operand(o: &Operand) -> Operand {
    Operand {
        host: o.host.clone(),
        layout: o.layout.clone(),
    }
}

/// Every variant of each op enum, hoisted so the edge-case sections below
/// sweep exactly the same set the main sections do — a variant added to an
/// enum and forgotten in one list would then be missing from both, which is a
/// compile-time-visible omission rather than a silent hole.
const BINARY_OPS: [BinaryOp; 7] = [
    BinaryOp::Add,
    BinaryOp::Sub,
    BinaryOp::Mul,
    BinaryOp::Div,
    BinaryOp::Maximum,
    BinaryOp::Minimum,
    BinaryOp::Pow,
];
const UNARY_OPS: [UnaryOp; 15] = [
    UnaryOp::Relu,
    UnaryOp::Gelu,
    UnaryOp::Exp,
    UnaryOp::Ln,
    UnaryOp::Sqrt,
    UnaryOp::Tanh,
    UnaryOp::Sigmoid,
    UnaryOp::Neg,
    UnaryOp::Abs,
    UnaryOp::Sign,
    UnaryOp::Recip,
    UnaryOp::Floor,
    UnaryOp::Ceil,
    UnaryOp::Round,
    UnaryOp::Erf,
];
const CMP_OPS: [CmpOp; 6] = [
    CmpOp::Eq,
    CmpOp::Ne,
    CmpOp::Lt,
    CmpOp::Le,
    CmpOp::Gt,
    CmpOp::Ge,
];
const REDUCE_OPS: [ReduceOp; 5] = [
    ReduceOp::Sum,
    ReduceOp::Mean,
    ReduceOp::Max,
    ReduceOp::Min,
    ReduceOp::Prod,
];

/// Binary, scalar-binary, unary, comparison, `where`, and `masked_fill`.
fn push_elementwise(cases: &mut Vec<Case>) {
    const BINARY: [BinaryOp; 7] = BINARY_OPS;
    for op in BINARY {
        cases.push(Case::new(
            format!("binary.{op:?}.f32"),
            Call::Binary(op),
            vec![f32s(&[2, 3], &A_F32), f32s(&[2, 3], &B_F32)],
        ));
        for dtype in [DType::F16, DType::BF16] {
            cases.push(Case::new(
                format!("binary.{op:?}.{dtype}"),
                Call::Binary(op),
                vec![
                    reduceds(dtype, &[2, 3], &A_F32),
                    reduceds(dtype, &[2, 3], &B_F32),
                ],
            ));
            cases.push(Case::new(
                format!("binary.{op:?}.{dtype}.strided_broadcast"),
                Call::Binary(op),
                vec![
                    reduceds_transposed(dtype, &A_F32),
                    reduceds_broadcast(dtype, &[1.5, -2.5, 4.0]),
                ],
            ));
            cases.push(Case::new(
                format!("binary_scalar.{op:?}.{dtype}"),
                Call::BinaryScalar(op, 2.5),
                vec![reduceds(dtype, &[2, 3], &A_F32)],
            ));
        }
        cases.push(Case::new(
            format!("binary.{op:?}.i64"),
            Call::Binary(op),
            vec![i64s(&[2, 3], &A_I64), i64s(&[2, 3], &B_I64)],
        ));
        // Transposed lhs against a broadcast rhs: both cursors non-trivial.
        cases.push(Case::new(
            format!("binary.{op:?}.f32.strided"),
            Call::Binary(op),
            vec![f32s_transposed(&A_F32), f32s_broadcast(&[1.5, -2.5, 4.0])],
        ));
        cases.push(Case::new(
            format!("binary_scalar.{op:?}.f32"),
            Call::BinaryScalar(op, 2.5),
            vec![f32s(&[2, 3], &A_F32)],
        ));
        cases.push(Case::new(
            format!("binary_scalar.{op:?}.i64"),
            Call::BinaryScalar(op, 3.0),
            vec![i64s(&[2, 3], &A_I64)],
        ));
    }

    const UNARY: [UnaryOp; 15] = UNARY_OPS;
    for op in UNARY {
        // Strictly positive data keeps `ln`/`sqrt` in-domain; the signed
        // sample below covers the sign-sensitive unaries.
        cases.push(Case::new(
            format!("unary.{op:?}.f32"),
            Call::Unary(op),
            vec![f32s(&[2, 3], &P_F32)],
        ));
        for dtype in [DType::F16, DType::BF16] {
            cases.push(Case::new(
                format!("unary.{op:?}.{dtype}"),
                Call::Unary(op),
                vec![reduceds(dtype, &[2, 3], &P_F32)],
            ));
        }
        if matches!(
            op,
            UnaryOp::Relu | UnaryOp::Gelu | UnaryOp::Tanh | UnaryOp::Sigmoid | UnaryOp::Neg
        ) {
            cases.push(Case::new(
                format!("unary.{op:?}.f32.signed"),
                Call::Unary(op),
                vec![f32s(&[2, 3], &A_F32)],
            ));
        }
        // `UnaryOp` defines only the sign-preserving integer unaries; the
        // rest report `Unsupported` and land in `Report::skipped`.
        cases.push(Case::new(
            format!("unary.{op:?}.i64"),
            Call::Unary(op),
            vec![i64s(&[2, 3], &A_I64)],
        ));
    }

    const CMP: [CmpOp; 6] = CMP_OPS;
    for op in CMP {
        cases.push(Case::new(
            format!("compare.{op:?}.f32"),
            Call::Compare(op),
            vec![f32s(&[2, 3], &A_F32), f32s(&[2, 3], &B_F32)],
        ));
        for dtype in [DType::F16, DType::BF16] {
            cases.push(Case::new(
                format!("compare.{op:?}.{dtype}"),
                Call::Compare(op),
                vec![
                    reduceds(dtype, &[2, 3], &A_F32),
                    reduceds(dtype, &[2, 3], &B_F32),
                ],
            ));
        }
        cases.push(Case::new(
            format!("compare.{op:?}.i64"),
            Call::Compare(op),
            vec![i64s(&[2, 3], &A_I64), i64s(&[2, 3], &B_I64)],
        ));
        cases.push(Case::new(
            format!("compare.{op:?}.bool"),
            Call::Compare(op),
            vec![bools(&[2, 3], &A_BOOL), bools(&[2, 3], &B_BOOL)],
        ));
    }

    for (label, on_true, on_false) in [
        ("f32", f32s(&[2, 3], &A_F32), f32s(&[2, 3], &B_F32)),
        ("i64", i64s(&[2, 3], &A_I64), i64s(&[2, 3], &B_I64)),
    ] {
        cases.push(Case::new(
            format!("where_cond.{label}"),
            Call::WhereCond,
            vec![bools(&[2, 3], &A_BOOL), on_true, on_false],
        ));
    }
    for (label, x) in [
        ("f32", f32s(&[2, 3], &A_F32)),
        ("i64", i64s(&[2, 3], &A_I64)),
        ("bool", bools(&[2, 3], &B_BOOL)),
    ] {
        cases.push(Case::new(
            format!("masked_fill.{label}"),
            Call::MaskedFill(-7.0),
            vec![x, bools(&[2, 3], &A_BOOL)],
        ));
    }
    for dtype in [DType::F16, DType::BF16] {
        cases.push(Case::new(
            format!("where_cond.{dtype}"),
            Call::WhereCond,
            vec![
                bools(&[2, 3], &A_BOOL),
                reduceds(dtype, &[2, 3], &A_F32),
                reduceds(dtype, &[2, 3], &B_F32),
            ],
        ));
        cases.push(Case::new(
            format!("masked_fill.{dtype}"),
            Call::MaskedFill(-7.0),
            vec![reduceds(dtype, &[2, 3], &A_F32), bools(&[2, 3], &A_BOOL)],
        ));
        cases.push(Case::new(
            format!("where_cond.{dtype}.strided_broadcast"),
            Call::WhereCond,
            vec![
                bools_broadcast(&[true, false, true]),
                reduceds_transposed(dtype, &A_F32),
                reduceds_broadcast(dtype, &[1.5, -2.5, 4.0]),
            ],
        ));
        cases.push(Case::new(
            format!("masked_fill.{dtype}.strided_broadcast"),
            Call::MaskedFill(-7.0),
            vec![
                reduceds_transposed(dtype, &A_F32),
                bools_broadcast(&[true, false, true]),
            ],
        ));
    }
}

/// Axis reductions and index-producing reductions, over dense and strided
/// views, on every axis.
fn push_reduce(cases: &mut Vec<Case>) {
    const REDUCE: [ReduceOp; 5] = REDUCE_OPS;
    for op in REDUCE {
        for axis in [0usize, 1] {
            cases.push(Case::new(
                format!("reduce.{op:?}.f32.axis{axis}"),
                Call::Reduce(op, axis),
                vec![f32s(&[2, 3], &A_F32)],
            ));
            for dtype in [DType::F16, DType::BF16] {
                cases.push(Case::new(
                    format!("reduce.{op:?}.{dtype}.axis{axis}"),
                    Call::Reduce(op, axis),
                    vec![reduceds(dtype, &[2, 3], &A_F32)],
                ));
                cases.push(Case::new(
                    format!("reduce.{op:?}.{dtype}.strided.axis{axis}"),
                    Call::Reduce(op, axis),
                    vec![reduceds_transposed(dtype, &A_F32)],
                ));
            }
            cases.push(Case::new(
                format!("reduce.{op:?}.i64.axis{axis}"),
                Call::Reduce(op, axis),
                vec![i64s(&[2, 3], &A_I64)],
            ));
            cases.push(Case::new(
                format!("reduce.{op:?}.f32.strided.axis{axis}"),
                Call::Reduce(op, axis),
                vec![f32s_transposed(&A_F32)],
            ));
        }
        // Reducing over a stride-0 (broadcast) axis must count the repeats.
        cases.push(Case::new(
            format!("reduce.{op:?}.f32.broadcast"),
            Call::Reduce(op, 1),
            vec![f32s_broadcast(&[1.0, 2.0, 3.0])],
        ));
        for dtype in [DType::F16, DType::BF16] {
            cases.push(Case::new(
                format!("reduce.{op:?}.{dtype}.broadcast"),
                Call::Reduce(op, 1),
                vec![reduceds_broadcast(dtype, &[1.0, 2.0, 3.0])],
            ));
        }
        cases.push(Case::new(
            format!("reduce.{op:?}.bool"),
            Call::Reduce(op, 0),
            vec![bools(&[2, 3], &A_BOOL)],
        ));
    }
    for op in [ArgReduceOp::ArgMax, ArgReduceOp::ArgMin] {
        for axis in [0usize, 1] {
            cases.push(Case::new(
                format!("arg_reduce.{op:?}.f32.axis{axis}"),
                Call::ArgReduce(op, axis),
                vec![f32s(&[2, 3], &A_F32)],
            ));
            for dtype in [DType::F16, DType::BF16] {
                cases.push(Case::new(
                    format!("arg_reduce.{op:?}.{dtype}.axis{axis}"),
                    Call::ArgReduce(op, axis),
                    vec![reduceds(dtype, &[2, 3], &A_F32)],
                ));
                cases.push(Case::new(
                    format!("arg_reduce.{op:?}.{dtype}.strided.axis{axis}"),
                    Call::ArgReduce(op, axis),
                    vec![reduceds_transposed(dtype, &A_F32)],
                ));
            }
            cases.push(Case::new(
                format!("arg_reduce.{op:?}.i64.axis{axis}"),
                Call::ArgReduce(op, axis),
                vec![i64s(&[2, 3], &A_I64)],
            ));
        }
        // Ties: the contract is "first occurrence wins".
        cases.push(Case::new(
            format!("arg_reduce.{op:?}.f32.ties"),
            Call::ArgReduce(op, 1),
            vec![f32s(&[2, 3], &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0])],
        ));
    }
}

/// Plain, batched, broadcast-batch, and strided matmul.
fn push_matmul(cases: &mut Vec<Case>) {
    cases.push(Case::new(
        "matmul.f32.2d".to_string(),
        Call::Matmul,
        vec![f32s(&[2, 3], &A_F32), f32s(&[3, 2], &B_F32)],
    ));
    for dtype in [DType::F16, DType::BF16] {
        cases.push(Case::new(
            format!("matmul.{dtype}.2d"),
            Call::Matmul,
            vec![
                reduceds(dtype, &[2, 3], &A_F32),
                reduceds(dtype, &[3, 2], &B_F32),
            ],
        ));
        cases.push(Case::new(
            format!("matmul.{dtype}.transposed_lhs"),
            Call::Matmul,
            vec![
                reduceds_transposed(dtype, &A_F32),
                reduceds(dtype, &[2, 2], &[1.0, 2.0, 3.0, 4.0]),
            ],
        ));
        let lhs_layout = Layout::contiguous([1, 2, 3])
            .and_then(|layout| layout.broadcast_to(&Shape::from([2, 2, 3])))
            .expect("conformance table: broadcastable reduced matmul lhs");
        cases.push(Case::new(
            format!("matmul.{dtype}.stride0_batch"),
            Call::Matmul,
            vec![
                reduceds_strided(dtype, &A_F32, lhs_layout),
                reduceds(
                    dtype,
                    &[2, 3, 2],
                    &[
                        0.5, 2.0, -1.0, 4.0, 0.25, -3.0, 1.0, -0.5, 2.0, 0.75, -2.0, 3.0,
                    ],
                ),
            ],
        ));
    }
    cases.push(Case::new(
        "matmul.i64.2d".to_string(),
        Call::Matmul,
        vec![i64s(&[2, 3], &A_I64), i64s(&[3, 2], &B_I64)],
    ));
    cases.push(Case::new(
        "matmul.f32.batched_broadcast".to_string(),
        Call::Matmul,
        vec![
            f32s(
                &[2, 2, 3],
                &[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, 0.5, 1.5, 2.5,
                ],
            ),
            f32s(&[1, 3, 2], &B_F32),
        ],
    ));
    // A transposed lhs is the shape every `x @ w.T` produces.
    cases.push(Case::new(
        "matmul.f32.transposed_lhs".to_string(),
        Call::Matmul,
        vec![
            f32s_transposed(&A_F32),
            f32s(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
        ],
    ));
}

/// `index_select` / `gather` and their accumulating backwards.
fn push_index(cases: &mut Vec<Case>) {
    // Per dtype: the source, a `[1, 3]` slice for `index_add`, and a
    // `[2, 2]` grid for `scatter_add`.
    let rows = [
        (
            "f32",
            f32s(&[2, 3], &A_F32),
            f32s(&[1, 3], &[1.0, 2.0, 3.0]),
            f32s(&[2, 2], &[10.0, 20.0, 30.0, 40.0]),
        ),
        (
            "i64",
            i64s(&[2, 3], &A_I64),
            i64s(&[1, 3], &[1, 2, 3]),
            i64s(&[2, 2], &[10, 20, 30, 40]),
        ),
    ];
    for (label, x, slice_src, grid_src) in rows {
        // Repeated and out-of-order positions, and an index count that
        // differs from the axis length.
        cases.push(Case::new(
            format!("index_select.{label}.axis1"),
            Call::IndexSelect(1),
            vec![clone_operand(&x), i64s(&[4], &[2, 0, 2, 1])],
        ));
        cases.push(Case::new(
            format!("index_add.{label}.axis0"),
            Call::IndexAdd(0),
            vec![clone_operand(&x), i64s(&[1], &[1]), slice_src],
        ));
        cases.push(Case::new(
            format!("gather.{label}.axis1"),
            Call::Gather(1),
            vec![clone_operand(&x), i64s(&[2, 2], &[0, 2, 1, 1])],
        ));
        cases.push(Case::new(
            format!("scatter_add.{label}.axis1"),
            Call::ScatterAdd(1),
            vec![x, i64s(&[2, 2], &[0, 2, 1, 1]), grid_src],
        ));
    }
    cases.push(Case::new(
        "index_select.f32.strided".to_string(),
        Call::IndexSelect(0),
        vec![f32s_transposed(&A_F32), i64s(&[2], &[2, 0])],
    ));
    for dtype in [DType::F16, DType::BF16] {
        let x = reduceds(dtype, &[2, 3], &A_F32);
        cases.push(Case::new(
            format!("index_select.{dtype}.axis1"),
            Call::IndexSelect(1),
            vec![clone_operand(&x), i64s(&[4], &[2, 0, 2, 1])],
        ));
        cases.push(Case::new(
            format!("index_select.{dtype}.strided.axis0"),
            Call::IndexSelect(0),
            vec![reduceds_transposed(dtype, &A_F32), i64s(&[2], &[2, 0])],
        ));
        cases.push(Case::new(
            format!("index_select.{dtype}.broadcast.axis1"),
            Call::IndexSelect(1),
            vec![
                reduceds_broadcast(dtype, &[1.0, 2.0, 3.0]),
                i64s(&[3], &[1, 0, 1]),
            ],
        ));
        cases.push(Case::new(
            format!("index_add.{dtype}.axis0"),
            Call::IndexAdd(0),
            vec![
                clone_operand(&x),
                i64s(&[1], &[1]),
                reduceds(dtype, &[1, 3], &[1.0, 2.0, 3.0]),
            ],
        ));
        cases.push(Case::new(
            format!("gather.{dtype}.axis1"),
            Call::Gather(1),
            vec![clone_operand(&x), i64s(&[2, 2], &[0, 2, 1, 1])],
        ));
        cases.push(Case::new(
            format!("scatter_add.{dtype}.axis1"),
            Call::ScatterAdd(1),
            vec![
                x,
                i64s(&[2, 2], &[0, 2, 1, 1]),
                reduceds(dtype, &[2, 2], &[10.0, 20.0, 30.0, 40.0]),
            ],
        ));
        cases.push(Case::new(
            format!("index_add.{dtype}.repeated.strided.axis0"),
            Call::IndexAdd(0),
            vec![
                reduceds_transposed(dtype, &A_F32),
                i64s(&[3], &[2, 0, 2]),
                reduceds_transposed(dtype, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
            ],
        ));
        cases.push(Case::new(
            format!("scatter_add.{dtype}.repeated.strided.axis0"),
            Call::ScatterAdd(0),
            vec![
                reduceds_transposed(dtype, &A_F32),
                i64s(&[3, 2], &[2, 1, 2, 1, 0, 1]),
                reduceds_transposed(dtype, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
            ],
        ));
    }

    // `arg_sort` is the index-*producing* kernel, so its output is a
    // permutation and every property worth pinning is an ordering rule rather
    // than a number: direction, tie stability, where NaN lands, and that the
    // positions it reports are positions along the *axis* of a strided view
    // rather than into flat storage.
    cases.push(Case::new(
        "arg_sort.ascending.f32".to_string(),
        Call::ArgSort {
            axis: 1,
            descending: false,
        },
        vec![f32s(&[2, 3], &A_F32)],
    ));
    cases.push(Case::new(
        "arg_sort.descending.f32".to_string(),
        Call::ArgSort {
            axis: 1,
            descending: true,
        },
        vec![f32s(&[2, 3], &A_F32)],
    ));
    // Equal values: the sort is documented stable, so ties keep their source
    // order. A kernel that reaches for an unstable sort still returns a valid
    // permutation and only diverges on a line like this one.
    cases.push(Case::new(
        "arg_sort.ties.f32".to_string(),
        Call::ArgSort {
            axis: 1,
            descending: false,
        },
        vec![f32s(&[2, 3], &[2.0, 1.0, 2.0, 1.0, 1.0, 2.0])],
    ));
    // NaN has a *position* here, unlike in `arg_reduce` where it wins both
    // directions: `cpu::index::total_order` puts it above every number, so an
    // ascending line ends in NaN and a descending one starts with it. Both
    // directions are rows because a kernel that sorts NaN to the bottom
    // matches the ascending row's shape and only diverges on one of them.
    for (label, descending) in [("non_finite", false), ("non_finite_descending", true)] {
        cases.push(Case::new(
            format!("arg_sort.{label}.f32"),
            Call::ArgSort {
                axis: 1,
                descending,
            },
            vec![f32s(&[2, 4], &NAN_A)],
        ));
    }
    // A transposed view sorted along its outer axis: the reported positions
    // are coordinates along that axis, so a kernel that hands back storage
    // indices produces plausible-looking garbage here and nowhere else.
    cases.push(Case::new(
        "arg_sort.strided.f32".to_string(),
        Call::ArgSort {
            axis: 0,
            descending: false,
        },
        vec![f32s_transposed(&A_F32)],
    ));
    // Comparison happens in the wide `Acc`, so a reduced line is ordered
    // exactly rather than through a lossy round trip.
    cases.push(Case::new(
        "arg_sort.ascending.f16".to_string(),
        Call::ArgSort {
            axis: 1,
            descending: false,
        },
        vec![reduceds(DType::F16, &[2, 3], &A_F32)],
    ));
    cases.push(Case::new(
        "arg_sort.ascending.i64".to_string(),
        Call::ArgSort {
            axis: 1,
            descending: false,
        },
        vec![i64s(&[2, 3], &A_I64)],
    ));
}

/// Convolution and pooling geometry, including stride/padding/dilation.
fn push_conv(cases: &mut Vec<Case>) {
    let input: Vec<f32> = (0..16).map(|i| i as f32 * 0.5 - 3.0).collect();
    let weight: Vec<f32> = vec![1.0, -0.5, 0.25, 2.0, 0.0, 1.5, -1.0, 0.75];
    // The last four are *combinations* with the two spatial axes deliberately
    // disagreeing. That is where a gradient kernel breaks: mapping an output
    // position back to an input position is correct under a stride alone and
    // under a dilation alone far more often than under both, and an (h, w)
    // transposition survives every symmetric geometry above it.
    // Pooling caps padding at half the window, so no padding here exceeds 1.
    let geoms = [
        ("dense", (1, 1), (0, 0), (1, 1)),
        ("strided", (2, 2), (0, 0), (1, 1)),
        ("padded", (1, 1), (1, 1), (1, 1)),
        ("dilated", (1, 1), (0, 0), (2, 2)),
        ("stride_pad", (2, 1), (1, 0), (1, 1)),
        ("stride_dilate", (2, 1), (0, 0), (1, 2)),
        ("pad_dilate", (1, 2), (1, 1), (2, 1)),
        ("all_three", (2, 1), (1, 1), (2, 1)),
    ];
    for (label, stride, padding, dilation) in geoms {
        let params = Conv2dParams {
            kernel: (2, 2),
            stride,
            padding,
            dilation,
        };
        cases.push(Case::new(
            format!("conv.Conv2d.f32.{label}"),
            Call::Conv(ConvOp::Conv2d, params),
            vec![
                f32s(&[1, 1, 4, 4], &input),
                // Two output channels over one input channel.
                f32s(&[2, 1, 2, 2], &weight),
            ],
        ));
        for dtype in [DType::F16, DType::BF16] {
            cases.push(Case::new(
                format!("conv.Conv2d.{dtype}.{label}"),
                Call::Conv(ConvOp::Conv2d, params),
                vec![
                    reduceds(dtype, &[1, 1, 4, 4], &input),
                    reduceds(dtype, &[2, 1, 2, 2], &weight),
                ],
            ));
        }
        // Multi-batch, multi-channel: the only shape in which a kernel that
        // decodes its flat thread index in the wrong axis order is visible.
        let batched: Vec<f32> = (0..2 * 2 * 4 * 4).map(|i| i as f32 * 0.25 - 4.0).collect();
        let wide_weight: Vec<f32> = (0..3 * 2 * 2 * 2).map(|i| 0.5 - i as f32 * 0.125).collect();
        cases.push(Case::new(
            format!("conv.Conv2d.f32.{label}.batched"),
            Call::Conv(ConvOp::Conv2d, params),
            vec![
                f32s(&[2, 2, 4, 4], &batched),
                f32s(&[3, 2, 2, 2], &wide_weight),
            ],
        ));
        // Integer convolution and pooling: `i64` has a `NumAcc`, so it is in
        // contract, and its accumulation wraps rather than saturating.
        cases.push(Case::new(
            format!("conv.Conv2d.i64.{label}"),
            Call::Conv(ConvOp::Conv2d, params),
            vec![
                i64s(&[1, 1, 4, 4], &(0..16).map(|i| i - 7).collect::<Vec<i64>>()),
                i64s(&[2, 1, 2, 2], &[1, -2, 3, -4, 5, -6, 7, -8]),
            ],
        ));
        for op in [ConvOp::MaxPool2d, ConvOp::AvgPool2d] {
            cases.push(Case::new(
                format!("conv.{op:?}.f32.{label}"),
                Call::Conv(op, params),
                vec![f32s(&[1, 1, 4, 4], &input)],
            ));
            for dtype in [DType::F16, DType::BF16] {
                cases.push(Case::new(
                    format!("conv.{op:?}.{dtype}.{label}"),
                    Call::Conv(op, params),
                    vec![reduceds(dtype, &[1, 1, 4, 4], &input)],
                ));
            }
            cases.push(Case::new(
                format!("conv.{op:?}.i64.{label}"),
                Call::Conv(op, params),
                vec![i64s(
                    &[1, 1, 4, 4],
                    &(0..16).map(|i| i - 7).collect::<Vec<i64>>(),
                )],
            ));
            cases.push(Case::new(
                format!("conv.{op:?}.f32.{label}.batched"),
                Call::Conv(op, params),
                vec![f32s(&[2, 2, 4, 4], &batched)],
            ));
        }
        push_conv_backward(cases, label, &params, &input, &weight);
    }
}

/// The backward half of [`push_conv`], for one geometry.
///
/// Kept separate only for size. It matters as much as the forward half: a
/// gradient kernel that maps output positions back to input positions
/// incorrectly under a stride or a dilation still produces finite, plausible
/// numbers, so nothing short of a reference diff catches it — the model just
/// trains badly. The cotangent is deliberately non-uniform, because a uniform
/// one is invariant under any permutation of output positions and would hide
/// exactly the class of error these cases exist to find.
fn push_conv_backward(
    cases: &mut Vec<Case>,
    label: &str,
    params: &Conv2dParams,
    input: &[f32],
    weight: &[f32],
) {
    const INPUT_DIMS: [usize; 4] = [1, 1, 4, 4];
    const WEIGHT_DIMS: [usize; 4] = [2, 1, 2, 2];

    // Ask the geometry for the cotangent shape rather than restating the
    // output formula here: a hand-written shape that disagrees would surface
    // as a shape error in every backend at once, which proves nothing.
    let conv = Conv2dGeometry::conv2d("conv2d", &INPUT_DIMS, &WEIGHT_DIMS, params)
        .expect("conformance table: valid conv geometry");
    let grad = cotangent(conv.output_dims());
    cases.push(Case::new(
        format!("conv.Conv2dInputGrad.f32.{label}"),
        Call::Conv(ConvOp::Conv2dInputGrad, *params),
        vec![
            f32s(&conv.output_dims(), &grad),
            f32s(&WEIGHT_DIMS, weight),
            f32s(&INPUT_DIMS, input),
        ],
    ));
    cases.push(Case::new(
        format!("conv.Conv2dWeightGrad.f32.{label}"),
        Call::Conv(ConvOp::Conv2dWeightGrad, *params),
        vec![
            f32s(&conv.output_dims(), &grad),
            f32s(&INPUT_DIMS, input),
            f32s(&WEIGHT_DIMS, weight),
        ],
    ));

    // The reduced dtypes have their own accumulation contract, and before this
    // no backward row exercised it: a gradient kernel that accumulates in F16
    // rather than in `Acc` still produces plausible numbers, just wrong ones.
    for dtype in [DType::F16, DType::BF16] {
        cases.push(Case::new(
            format!("conv.Conv2dInputGrad.{dtype}.{label}"),
            Call::Conv(ConvOp::Conv2dInputGrad, *params),
            vec![
                reduceds(dtype, &conv.output_dims(), &grad),
                reduceds(dtype, &WEIGHT_DIMS, weight),
                reduceds(dtype, &INPUT_DIMS, input),
            ],
        ));
        cases.push(Case::new(
            format!("conv.Conv2dWeightGrad.{dtype}.{label}"),
            Call::Conv(ConvOp::Conv2dWeightGrad, *params),
            vec![
                reduceds(dtype, &conv.output_dims(), &grad),
                reduceds(dtype, &INPUT_DIMS, input),
                reduceds(dtype, &WEIGHT_DIMS, weight),
            ],
        ));
    }

    // Multi-batch, multi-channel backward. The single-batch single-channel
    // shape above cannot distinguish a kernel that decodes `(b, c)` in the

    // wrong order from a correct one, because both indices are always 0.
    const BATCH_DIMS: [usize; 4] = [2, 2, 4, 4];
    const WIDE_WEIGHT_DIMS: [usize; 4] = [3, 2, 2, 2];
    let batched: Vec<f32> = (0..2 * 2 * 4 * 4).map(|i| i as f32 * 0.25 - 4.0).collect();
    let wide_weight: Vec<f32> = (0..3 * 2 * 2 * 2).map(|i| 0.5 - i as f32 * 0.125).collect();
    let wide = Conv2dGeometry::conv2d("conv2d", &BATCH_DIMS, &WIDE_WEIGHT_DIMS, params)
        .expect("conformance table: valid batched conv geometry");
    let wide_grad = cotangent(wide.output_dims());
    cases.push(Case::new(
        format!("conv.Conv2dInputGrad.f32.{label}.batched"),
        Call::Conv(ConvOp::Conv2dInputGrad, *params),
        vec![
            f32s(&wide.output_dims(), &wide_grad),
            f32s(&WIDE_WEIGHT_DIMS, &wide_weight),
            f32s(&BATCH_DIMS, &batched),
        ],
    ));
    cases.push(Case::new(
        format!("conv.Conv2dWeightGrad.f32.{label}.batched"),
        Call::Conv(ConvOp::Conv2dWeightGrad, *params),
        vec![
            f32s(&wide.output_dims(), &wide_grad),
            f32s(&BATCH_DIMS, &batched),
            f32s(&WIDE_WEIGHT_DIMS, &wide_weight),
        ],
    ));

    let pool = Conv2dGeometry::pool("pool2d", &INPUT_DIMS, params)
        .expect("conformance table: valid pool geometry");
    let pool_grad = cotangent(pool.output_dims());
    let wide_pool = Conv2dGeometry::pool("pool2d", &BATCH_DIMS, params)
        .expect("conformance table: valid batched pool geometry");
    let wide_pool_grad = cotangent(wide_pool.output_dims());
    for op in [ConvOp::MaxPool2dBackward, ConvOp::AvgPool2dBackward] {
        cases.push(Case::new(
            format!("conv.{op:?}.f32.{label}"),
            Call::Conv(op, *params),
            vec![
                f32s(&pool.output_dims(), &pool_grad),
                f32s(&INPUT_DIMS, input),
            ],
        ));
        for dtype in [DType::F16, DType::BF16] {
            cases.push(Case::new(
                format!("conv.{op:?}.{dtype}.{label}"),
                Call::Conv(op, *params),
                vec![
                    reduceds(dtype, &pool.output_dims(), &pool_grad),
                    reduceds(dtype, &INPUT_DIMS, input),
                ],
            ));
        }
        cases.push(Case::new(
            format!("conv.{op:?}.f32.{label}.batched"),
            Call::Conv(op, *params),
            vec![
                f32s(&wide_pool.output_dims(), &wide_pool_grad),
                f32s(&BATCH_DIMS, &batched),
            ],
        ));
        // Repeated values inside the window make max-pool tie-breaking
        // observable: the *first* maximum owns the whole gradient.
        let ties: Vec<f32> = (0..16).map(|i| ((i % 4) / 2) as f32).collect();
        cases.push(Case::new(
            format!("conv.{op:?}.f32.{label}.ties"),
            Call::Conv(op, *params),
            vec![
                f32s(&pool.output_dims(), &pool_grad),
                f32s(&INPUT_DIMS, &ties),
            ],
        ));
    }
}

/// Invalid-argument rows that exercise validators in the fused entry points.
///
/// These are intentionally table rows rather than backend-specific patches:
/// a backend that drops the scalar guard must disagree with the CPU reference
/// and is reported through `rejection_mismatches`.
fn push_rejection_cases(cases: &mut Vec<Case>) {
    cases.push(Case::rejected(
        "fused.SgdStep.invalid_lr".to_string(),
        Call::Fused(FusedOp::SgdStep, vec![-1.0, 0.0, 0.0]),
        vec![
            f32s(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            f32s(&[2, 3], &[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        ],
        "InvalidArg",
    ));
    cases.push(Case::rejected(
        "fused.AdamStep.invalid_eps".to_string(),
        Call::Fused(
            FusedOp::AdamStep,
            vec![0.1, 0.9, 0.999, 0.0, 0.0, 1.0, 1.0, 0.0],
        ),
        vec![
            f32s(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            f32s(&[2, 3], &[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            f32s(&[2, 3], &[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            f32s(&[2, 3], &[0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
        ],
        "InvalidArg",
    ));
}

/// The dimensions kernels actually break in: values (NaN, infinities, signed
/// zero, subnormals, integer extremes), degenerate shapes (empty axes,
/// single-element axes), offset views, `copy_into`, and the `F64` lane the
/// table never touched.
///
/// Every row here is a *differential* row like any other, so it costs one
/// table entry and covers every present and future backend at once.
fn push_edges(cases: &mut Vec<Case>) {
    push_non_finite(cases);
    push_integer_extremes(cases);
    push_degenerate_shapes(cases);
    push_offset_views(cases);
    push_copy_into(cases);
    push_f64(cases);
    push_bool_declines(cases);
    push_extreme_ranks(cases);
}

/// Rank 0 and rank 5: the two ends of the rank range.
///
/// A rank-0 view has one element and *no* axes, so every per-axis loop runs
/// zero times and the storage index is the offset alone — which is exactly the
/// field a kernel can drop unnoticed, since there is no stride left to get
/// wrong. Rank 5 is past the four spatial axes conv works in, so it catches a
/// kernel that sized a fixed per-axis buffer.
fn push_extreme_ranks(cases: &mut Vec<Case>) {
    let scalar = || f32s(&[], &[-1.5]);
    // A rank-0 view of element 3 of a longer buffer: one element, no axes, and
    // a non-zero offset that only the offset field can express.
    let offset_scalar = || {
        Operand::strided(
            HostConv::into_cpu_storage(vec![9.0f32, -9.0, 8.0, -1.5, 7.0]),
            Layout::from_parts(Shape::from([]), Box::from([]), 3)
                .expect("conformance table: rank-0 offset layout"),
        )
    };
    cases.push(Case::new(
        "copy_strided.f32.rank0".to_string(),
        Call::CopyStrided,
        vec![offset_scalar()],
    ));
    cases.push(Case::new(
        "cast.f32_to_i64.rank0".to_string(),
        Call::Cast(DType::I64),
        vec![offset_scalar()],
    ));
    cases.push(Case::new(
        "binary.Add.f32.rank0".to_string(),
        Call::Binary(BinaryOp::Add),
        vec![offset_scalar(), scalar()],
    ));
    cases.push(Case::new(
        "binary_scalar.Mul.f32.rank0".to_string(),
        Call::BinaryScalar(BinaryOp::Mul, 3.0),
        vec![offset_scalar()],
    ));
    cases.push(Case::new(
        "unary.Relu.f32.rank0".to_string(),
        Call::Unary(UnaryOp::Relu),
        vec![offset_scalar()],
    ));
    cases.push(Case::new(
        "compare.Lt.f32.rank0".to_string(),
        Call::Compare(CmpOp::Lt),
        vec![offset_scalar(), scalar()],
    ));
    cases.push(Case::new(
        "where_cond.f32.rank0".to_string(),
        Call::WhereCond,
        vec![bools(&[], &[true]), offset_scalar(), scalar()],
    ));
    cases.push(Case::new(
        "masked_fill.f32.rank0".to_string(),
        Call::MaskedFill(-7.0),
        vec![offset_scalar(), bools(&[], &[true])],
    ));

    // Rank 5, with two axes transposed so the walk is non-trivial.
    let wide: Vec<f32> = (0..24).map(|i| i as f32 * 0.5 - 6.0).collect();
    let rank5 = || {
        Layout::contiguous([2, 1, 3, 2, 2])
            .and_then(|layout| layout.transpose(0, 2))
            .expect("conformance table: rank-5 layout")
    };
    let rank5_dense =
        || Layout::contiguous([3, 1, 2, 2, 2]).expect("conformance table: dense rank-5 layout");
    let strided5 = || Operand::strided(HostConv::into_cpu_storage(wide.clone()), rank5());
    cases.push(Case::new(
        "copy_strided.f32.rank5".to_string(),
        Call::CopyStrided,
        vec![strided5()],
    ));
    cases.push(Case::new(
        "binary.Sub.f32.rank5".to_string(),
        Call::Binary(BinaryOp::Sub),
        vec![
            strided5(),
            Operand::strided(HostConv::into_cpu_storage(wide.clone()), rank5_dense()),
        ],
    ));
    for axis in [0usize, 2, 4] {
        cases.push(Case::new(
            format!("reduce.Sum.f32.rank5.axis{axis}"),
            Call::Reduce(ReduceOp::Sum, axis),
            vec![strided5()],
        ));
    }
    cases.push(Case::new(
        "arg_reduce.ArgMax.f32.rank5".to_string(),
        Call::ArgReduce(ArgReduceOp::ArgMax, 2),
        vec![strided5()],
    ));
    // Three batch axes on the left against a bare rank-2 right-hand side.
    cases.push(Case::new(
        "matmul.f32.rank5_batch".to_string(),
        Call::Matmul,
        vec![
            strided5(),
            f32s(&[2, 3], &[1.0, -2.0, 0.5, 2.0, -1.0, 0.25]),
        ],
    ));
    cases.push(Case::new(
        "index_select.f32.rank5".to_string(),
        Call::IndexSelect(2),
        vec![strided5(), i64s(&[3], &[1, 0, 1])],
    ));
    cases.push(Case::new(
        "gather.f32.rank5".to_string(),
        Call::Gather(4),
        vec![
            strided5(),
            i64s(
                &[3, 1, 2, 2, 2],
                &[
                    1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1,
                ],
            ),
        ],
    ));
}

/// The arithmetic entry points `Bool` has no answer for.
///
/// `bool` has no [`NumAcc`](crate::backend::cpu::acc::NumAcc), so it has no
/// accumulation and therefore no reduction, matmul, convolution, or
/// accumulating scatter; and it has no arithmetic at all, comparisons living
/// in [`BackendOps::compare`] instead. These rows exist to prove the decline
/// keeps happening: a backend that started quietly computing `true + true`
/// would move from `expected_unsupported` to `matched`, and the self-check's
/// exact expected-skip list would fail.
fn push_bool_declines(cases: &mut Vec<Case>) {
    cases.push(Case::new(
        "binary.Add.bool".to_string(),
        Call::Binary(BinaryOp::Add),
        vec![bools(&[2, 3], &A_BOOL), bools(&[2, 3], &B_BOOL)],
    ));
    cases.push(Case::new(
        "binary_scalar.Add.bool".to_string(),
        Call::BinaryScalar(BinaryOp::Add, 1.0),
        vec![bools(&[2, 3], &A_BOOL)],
    ));
    cases.push(Case::new(
        "unary.Neg.bool".to_string(),
        Call::Unary(UnaryOp::Neg),
        vec![bools(&[2, 3], &A_BOOL)],
    ));
    cases.push(Case::new(
        "arg_reduce.ArgMax.bool".to_string(),
        Call::ArgReduce(ArgReduceOp::ArgMax, 1),
        vec![bools(&[2, 3], &A_BOOL)],
    ));
    // `arg_sort` compares in `Acc`, so it routes through `dispatch_numeric!`
    // and declines `Bool` for the same reason the reductions do — not because
    // booleans have no order, but because they have no wide accumulator to be
    // ordered in.
    cases.push(Case::new(
        "arg_sort.bool".to_string(),
        Call::ArgSort {
            axis: 1,
            descending: false,
        },
        vec![bools(&[2, 3], &A_BOOL)],
    ));
    cases.push(Case::new(
        "matmul.bool".to_string(),
        Call::Matmul,
        vec![bools(&[2, 3], &A_BOOL), bools(&[3, 2], &B_BOOL)],
    ));
    cases.push(Case::new(
        "index_add.bool".to_string(),
        Call::IndexAdd(0),
        vec![
            bools(&[2, 3], &A_BOOL),
            i64s(&[1], &[1]),
            bools(&[1, 3], &[true, false, true]),
        ],
    ));
    cases.push(Case::new(
        "scatter_add.bool".to_string(),
        Call::ScatterAdd(1),
        vec![
            bools(&[2, 3], &A_BOOL),
            i64s(&[2, 2], &[0, 2, 1, 1]),
            bools(&[2, 2], &[true, false, true, true]),
        ],
    ));
    cases.push(Case::new(
        "conv.Conv2d.bool".to_string(),
        Call::Conv(ConvOp::Conv2d, CONV_DENSE),
        vec![
            bools(&[1, 1, 4, 4], &[true; 16]),
            bools(&[2, 1, 2, 2], &[true; 8]),
        ],
    ));
    cases.push(Case::new(
        "conv.MaxPool2d.bool".to_string(),
        Call::Conv(ConvOp::MaxPool2d, CONV_DENSE),
        vec![bools(&[1, 1, 4, 4], &[true; 16])],
    ));
}

/// Shape `[2, 4]`: NaN, both infinities, both zeros, a subnormal, and ordinary
/// values, so no lane is accidentally uniform.
const NAN_A: [f32; 8] = [
    f32::NAN,
    f32::INFINITY,
    f32::NEG_INFINITY,
    -0.0,
    0.0,
    1.5,
    -2.5,
    3.0,
];
/// The partner array, arranged so every interesting pair occurs somewhere:
/// NaN-vs-number, inf-vs-inf, inf-vs-(-inf), zero-vs-zero.
const NAN_B: [f32; 8] = [
    2.0,
    f32::INFINITY,
    f32::INFINITY,
    0.0,
    -0.0,
    f32::NAN,
    -2.5,
    // The smallest positive subnormal `f32`: it survives the host conversion
    // to F16/BF16 as a zero, identically on both sides, so the row stays a
    // fair comparison while the F32 lane really does carry a subnormal.
    1.0e-45,
];

/// Non-finite and signed-zero inputs to every element-wise family.
///
/// `maximum`/`minimum`/`relu`/`max`/`argmax` are the ops with a genuine choice
/// to make here, and the fixture data everywhere else in the table is finite,
/// so without these rows a backend could pick the opposite convention for all
/// of them and stay green.
fn push_non_finite(cases: &mut Vec<Case>) {
    for op in BINARY_OPS {
        cases.push(Case::new(
            format!("binary.{op:?}.f32.non_finite"),
            Call::Binary(op),
            vec![f32s(&[2, 4], &NAN_A), f32s(&[2, 4], &NAN_B)],
        ));
        cases.push(Case::new(
            format!("binary_scalar.{op:?}.f32.infinite"),
            Call::BinaryScalar(op, f64::INFINITY),
            vec![f32s(&[2, 4], &NAN_A)],
        ));
    }
    for op in UNARY_OPS {
        cases.push(Case::new(
            format!("unary.{op:?}.f32.non_finite"),
            Call::Unary(op),
            vec![f32s(&[2, 4], &NAN_A)],
        ));
    }
    for op in CMP_OPS {
        // Every comparison against NaN is false, including `Ne`… which is
        // true. A kernel that folds `Ne` into `!Eq` gets this wrong.
        cases.push(Case::new(
            format!("compare.{op:?}.f32.non_finite"),
            Call::Compare(op),
            vec![f32s(&[2, 4], &NAN_A), f32s(&[2, 4], &NAN_B)],
        ));
    }
    for op in REDUCE_OPS {
        for axis in [0usize, 1] {
            cases.push(Case::new(
                format!("reduce.{op:?}.f32.non_finite.axis{axis}"),
                Call::Reduce(op, axis),
                vec![f32s(&[2, 4], &NAN_A)],
            ));
        }
    }
    for op in [ArgReduceOp::ArgMax, ArgReduceOp::ArgMin] {
        // A NaN must win *both* directions, so `max`/`argmax` name the same
        // element. Both axes, because the NaN is in a different position
        // relative to the walk in each.
        for axis in [0usize, 1] {
            cases.push(Case::new(
                format!("arg_reduce.{op:?}.f32.non_finite.axis{axis}"),
                Call::ArgReduce(op, axis),
                vec![f32s(&[2, 4], &NAN_A)],
            ));
        }
    }
    cases.push(Case::new(
        "matmul.f32.non_finite".to_string(),
        Call::Matmul,
        vec![f32s(&[2, 4], &NAN_A), f32s(&[4, 2], &NAN_B)],
    ));
    cases.push(Case::new(
        "where_cond.f32.non_finite".to_string(),
        Call::WhereCond,
        vec![
            bools(
                &[2, 4],
                &[true, false, true, false, true, false, true, false],
            ),
            f32s(&[2, 4], &NAN_A),
            f32s(&[2, 4], &NAN_B),
        ],
    ));
    cases.push(Case::new(
        "masked_fill.f32.non_finite".to_string(),
        Call::MaskedFill(f64::NEG_INFINITY),
        vec![
            f32s(&[2, 4], &NAN_A),
            bools(
                &[2, 4],
                &[true, false, true, false, true, false, true, false],
            ),
        ],
    ));
    // Float→int saturation is a documented conversion, not UB: `NaN as i64` is
    // 0 and the infinities clamp to the extremes.
    cases.push(Case::new(
        "cast.f32_to_i64.non_finite".to_string(),
        Call::Cast(DType::I64),
        vec![f32s(&[2, 4], &NAN_A)],
    ));
    cases.push(Case::new(
        "cast.f32_to_bool.non_finite".to_string(),
        Call::Cast(DType::Bool),
        vec![f32s(&[2, 4], &NAN_A)],
    ));
    // The reduced dtypes carry the same policy at their own precision.
    for dtype in [DType::F16, DType::BF16] {
        cases.push(Case::new(
            format!("binary.Maximum.{dtype}.non_finite"),
            Call::Binary(BinaryOp::Maximum),
            vec![
                reduceds(dtype, &[2, 4], &NAN_A),
                reduceds(dtype, &[2, 4], &NAN_B),
            ],
        ));
        cases.push(Case::new(
            format!("reduce.Max.{dtype}.non_finite"),
            Call::Reduce(ReduceOp::Max, 1),
            vec![reduceds(dtype, &[2, 4], &NAN_A)],
        ));
        cases.push(Case::new(
            format!("arg_reduce.ArgMax.{dtype}.non_finite"),
            Call::ArgReduce(ArgReduceOp::ArgMax, 1),
            vec![reduceds(dtype, &[2, 4], &NAN_A)],
        ));
        cases.push(Case::new(
            format!("unary.Relu.{dtype}.non_finite"),
            Call::Unary(UnaryOp::Relu),
            vec![reduceds(dtype, &[2, 4], &NAN_A)],
        ));
    }
    // A fully `-inf` softmax row is the masked-attention case; a NaN row must
    // stay NaN rather than being laundered into a uniform distribution.
    for dtype in [DType::F16, DType::F32] {
        cases.push(Case::new(
            format!("fused.Softmax.{dtype}.non_finite"),
            Call::Fused(FusedOp::Softmax, vec![]),
            vec![floats(
                dtype,
                &[3, 3],
                &[
                    f32::NEG_INFINITY,
                    f32::NEG_INFINITY,
                    f32::NEG_INFINITY,
                    1.0,
                    f32::NAN,
                    2.0,
                    0.5,
                    f32::NEG_INFINITY,
                    1.5,
                ],
            )],
        ));
    }
}

/// `i64` at the edges: wrapping arithmetic, the two division special cases,
/// and the extremes under `Neg`/`Abs` and accumulation.
///
/// The contract is *wrapping*, and wrapping is exactly what a kernel gets
/// wrong silently: nothing panics, the answer is simply a different large
/// number. The only division-by-zero and `i64::MIN / -1` values pinned before
/// this lived behind the `metal` feature, so a default `cargo test` never
/// checked them at all.
fn push_integer_extremes(cases: &mut Vec<Case>) {
    const EXTREME_A: [i64; 6] = [i64::MIN, i64::MAX, i64::MIN, i64::MAX, -7, i64::MIN];
    const EXTREME_B: [i64; 6] = [-1, 1, -1, 2, 0, i64::MIN];
    for op in BINARY_OPS {
        cases.push(Case::new(
            format!("binary.{op:?}.i64.extremes"),
            Call::Binary(op),
            vec![i64s(&[2, 3], &EXTREME_A), i64s(&[2, 3], &EXTREME_B)],
        ));
        for (label, scalar) in [("zero", 0.0), ("minus_one", -1.0)] {
            cases.push(Case::new(
                format!("binary_scalar.{op:?}.i64.{label}"),
                Call::BinaryScalar(op, scalar),
                vec![i64s(&[2, 3], &EXTREME_A)],
            ));
        }
    }
    for op in [UnaryOp::Neg, UnaryOp::Abs] {
        cases.push(Case::new(
            format!("unary.{op:?}.i64.extremes"),
            Call::Unary(op),
            vec![i64s(&[2, 3], &EXTREME_A)],
        ));
    }
    for op in REDUCE_OPS {
        cases.push(Case::new(
            format!("reduce.{op:?}.i64.extremes"),
            Call::Reduce(op, 1),
            vec![i64s(&[2, 3], &EXTREME_A)],
        ));
    }
    for op in [ArgReduceOp::ArgMax, ArgReduceOp::ArgMin] {
        cases.push(Case::new(
            format!("arg_reduce.{op:?}.i64.extremes"),
            Call::ArgReduce(op, 1),
            vec![i64s(&[2, 3], &EXTREME_A)],
        ));
    }
    cases.push(Case::new(
        "matmul.i64.extremes".to_string(),
        Call::Matmul,
        vec![i64s(&[2, 3], &EXTREME_A), i64s(&[3, 2], &EXTREME_B)],
    ));
    cases.push(Case::new(
        "index_add.i64.extremes".to_string(),
        Call::IndexAdd(0),
        vec![
            i64s(&[2, 3], &EXTREME_A),
            i64s(&[2], &[0, 0]),
            i64s(&[2, 3], &EXTREME_A),
        ],
    ));
    cases.push(Case::new(
        "cast.i64_to_f32.extremes".to_string(),
        Call::Cast(DType::F32),
        vec![i64s(&[2, 3], &EXTREME_A)],
    ));
}

/// Empty axes and single-element axes.
///
/// `sum` over an empty axis is legal (the identity), and `narrow(axis, i, 0)`
/// or a zero batch makes it reachable, so every entry point has to survive a
/// zero-length operand without dividing by the axis length or dispatching a
/// zero-thread grid it then reads the output of.
fn push_degenerate_shapes(cases: &mut Vec<Case>) {
    let empty_f32 = || f32s(&[2, 0], &[]);
    let empty_i64 = || i64s(&[2, 0], &[]);
    let empty_bool = || bools(&[2, 0], &[]);

    cases.push(Case::new(
        "copy_strided.f32.empty".to_string(),
        Call::CopyStrided,
        vec![empty_f32()],
    ));
    cases.push(Case::new(
        "cast.f32_to_i64.empty".to_string(),
        Call::Cast(DType::I64),
        vec![empty_f32()],
    ));
    cases.push(Case::new(
        "full.f32.empty".to_string(),
        Call::Full {
            len: 0,
            dtype: DType::F32,
            value: 1.0,
        },
        vec![],
    ));
    for op in BINARY_OPS {
        cases.push(Case::new(
            format!("binary.{op:?}.f32.empty"),
            Call::Binary(op),
            vec![empty_f32(), empty_f32()],
        ));
    }
    cases.push(Case::new(
        "binary_scalar.Add.f32.empty".to_string(),
        Call::BinaryScalar(BinaryOp::Add, 2.0),
        vec![empty_f32()],
    ));
    cases.push(Case::new(
        "unary.Relu.f32.empty".to_string(),
        Call::Unary(UnaryOp::Relu),
        vec![empty_f32()],
    ));
    cases.push(Case::new(
        "compare.Lt.f32.empty".to_string(),
        Call::Compare(CmpOp::Lt),
        vec![empty_f32(), empty_f32()],
    ));
    cases.push(Case::new(
        "where_cond.f32.empty".to_string(),
        Call::WhereCond,
        vec![empty_bool(), empty_f32(), empty_f32()],
    ));
    cases.push(Case::new(
        "masked_fill.f32.empty".to_string(),
        Call::MaskedFill(1.0),
        vec![empty_f32(), empty_bool()],
    ));
    // Summing over the empty axis keeps the *other* axis, so the output is a
    // non-empty buffer of identities: the one shape where a backend that sizes
    // its grid as `num_elements` writes nothing at all.
    cases.push(Case::new(
        "reduce.Sum.f32.empty_axis".to_string(),
        Call::Reduce(ReduceOp::Sum, 1),
        vec![empty_f32()],
    ));
    cases.push(Case::new(
        "reduce.Sum.i64.empty_axis".to_string(),
        Call::Reduce(ReduceOp::Sum, 1),
        vec![empty_i64()],
    ));
    // Reducing the *non-empty* axis of an empty tensor stays empty.
    cases.push(Case::new(
        "reduce.Sum.f32.empty_output".to_string(),
        Call::Reduce(ReduceOp::Sum, 0),
        vec![empty_f32()],
    ));
    cases.push(Case::new(
        "matmul.f32.empty_inner".to_string(),
        Call::Matmul,
        vec![f32s(&[2, 0], &[]), f32s(&[0, 3], &[])],
    ));
    cases.push(Case::new(
        "matmul.f32.empty_batch".to_string(),
        Call::Matmul,
        vec![f32s(&[0, 2, 3], &[]), f32s(&[0, 3, 2], &[])],
    ));
    cases.push(Case::new(
        "index_select.f32.empty_indices".to_string(),
        Call::IndexSelect(1),
        vec![f32s(&[2, 3], &A_F32), i64s(&[0], &[])],
    ));
    cases.push(Case::new(
        "index_add.f32.empty_indices".to_string(),
        Call::IndexAdd(0),
        vec![f32s(&[2, 3], &A_F32), i64s(&[0], &[]), f32s(&[0, 3], &[])],
    ));
    cases.push(Case::new(
        "gather.f32.empty_indices".to_string(),
        Call::Gather(1),
        vec![f32s(&[2, 3], &A_F32), i64s(&[2, 0], &[])],
    ));
    cases.push(Case::new(
        "scatter_add.f32.empty_indices".to_string(),
        Call::ScatterAdd(1),
        vec![
            f32s(&[2, 3], &A_F32),
            i64s(&[2, 0], &[]),
            f32s(&[2, 0], &[]),
        ],
    ));
    // A zero batch through conv and pooling: the geometry is valid, the output
    // is empty, and nothing may divide by the batch count.
    let empty_conv = Conv2dParams {
        kernel: (2, 2),
        stride: (1, 1),
        padding: (0, 0),
        dilation: (1, 1),
    };
    cases.push(Case::new(
        "conv.Conv2d.f32.empty_batch".to_string(),
        Call::Conv(ConvOp::Conv2d, empty_conv),
        vec![
            f32s(&[0, 1, 4, 4], &[]),
            f32s(&[2, 1, 2, 2], &[1.0, -0.5, 0.25, 2.0, 0.0, 1.5, -1.0, 0.75]),
        ],
    ));
    cases.push(Case::new(
        "conv.MaxPool2d.f32.empty_batch".to_string(),
        Call::Conv(ConvOp::MaxPool2d, empty_conv),
        vec![f32s(&[0, 1, 4, 4], &[])],
    ));

    // Single-element axes: `[1, 1]` exercises every divisor-of-one path, and a
    // width-1 softmax/LayerNorm row must normalize to exactly 1 / to the bias.
    cases.push(Case::new(
        "reduce.Mean.f32.unit_axis".to_string(),
        Call::Reduce(ReduceOp::Mean, 1),
        vec![f32s(&[3, 1], &[1.5, -2.5, 0.0])],
    ));
    cases.push(Case::new(
        "matmul.f32.unit_inner".to_string(),
        Call::Matmul,
        vec![
            f32s(&[2, 1], &[1.5, -2.5]),
            f32s(&[1, 3], &[2.0, -1.0, 0.5]),
        ],
    ));
    for dtype in [DType::F16, DType::F32] {
        cases.push(Case::new(
            format!("fused.Softmax.{dtype}.unit_width"),
            Call::Fused(FusedOp::Softmax, vec![]),
            vec![floats(dtype, &[3, 1], &[1.5, -2.5, 0.0])],
        ));
        cases.push(Case::new(
            format!("fused.LayerNorm.{dtype}.unit_width"),
            Call::Fused(FusedOp::LayerNorm, vec![1e-3]),
            vec![
                floats(dtype, &[3, 1], &[1.5, -2.5, 0.0]),
                floats(dtype, &[1], &[2.0]),
                floats(dtype, &[1], &[-0.5]),
            ],
        ));
    }
}

/// Views with a non-zero storage offset.
///
/// `narrow` produces them constantly, and an offset is the one layout field a
/// kernel can drop without any shape check noticing: the result has the right
/// size and plausible values, just read from the wrong place.
fn push_offset_views(cases: &mut Vec<Case>) {
    // A `[2, 3]` window starting at element 2 of an 8-element buffer.
    let offset = || {
        Layout::from_parts(Shape::from([2, 3]), Box::from([3usize, 1]), 2)
            .expect("conformance table: offset layout")
    };
    // The same window, transposed: offset *and* non-canonical strides.
    let offset_t = || {
        Layout::from_parts(Shape::from([3, 2]), Box::from([1usize, 3]), 2)
            .expect("conformance table: transposed offset layout")
    };
    const WIDE: [f32; 8] = [9.0, -9.0, 1.0, -2.0, 3.0, -4.0, 5.0, 6.5];
    const WIDE_B: [f32; 8] = [-8.0, 8.0, 0.5, 2.0, -1.0, 4.0, 0.25, -3.0];
    let wide = |data: &[f32]| Operand::strided(HostConv::into_cpu_storage(data.to_vec()), offset());
    let wide_t =
        |data: &[f32]| Operand::strided(HostConv::into_cpu_storage(data.to_vec()), offset_t());

    cases.push(Case::new(
        "copy_strided.f32.offset".to_string(),
        Call::CopyStrided,
        vec![wide(&WIDE)],
    ));
    cases.push(Case::new(
        "cast.f32_to_i64.offset".to_string(),
        Call::Cast(DType::I64),
        vec![wide(&WIDE)],
    ));
    cases.push(Case::new(
        "binary.Sub.f32.offset".to_string(),
        Call::Binary(BinaryOp::Sub),
        vec![wide(&WIDE), wide(&WIDE_B)],
    ));
    cases.push(Case::new(
        "binary_scalar.Sub.f32.offset".to_string(),
        Call::BinaryScalar(BinaryOp::Sub, 2.5),
        vec![wide(&WIDE)],
    ));
    cases.push(Case::new(
        "unary.Neg.f32.offset".to_string(),
        Call::Unary(UnaryOp::Neg),
        vec![wide(&WIDE)],
    ));
    cases.push(Case::new(
        "compare.Lt.f32.offset".to_string(),
        Call::Compare(CmpOp::Lt),
        vec![wide(&WIDE), wide(&WIDE_B)],
    ));
    cases.push(Case::new(
        "where_cond.f32.offset".to_string(),
        Call::WhereCond,
        vec![
            Operand::strided(
                HostConv::into_cpu_storage(vec![
                    false, false, true, false, true, false, true, true,
                ]),
                offset(),
            ),
            wide(&WIDE),
            wide(&WIDE_B),
        ],
    ));
    cases.push(Case::new(
        "masked_fill.f32.offset".to_string(),
        Call::MaskedFill(-7.0),
        vec![
            wide(&WIDE),
            Operand::strided(
                HostConv::into_cpu_storage(vec![
                    false, false, true, false, true, false, true, true,
                ]),
                offset(),
            ),
        ],
    ));
    for op in REDUCE_OPS {
        cases.push(Case::new(
            format!("reduce.{op:?}.f32.offset"),
            Call::Reduce(op, 1),
            vec![wide(&WIDE)],
        ));
    }
    cases.push(Case::new(
        "arg_reduce.ArgMax.f32.offset".to_string(),
        Call::ArgReduce(ArgReduceOp::ArgMax, 0),
        vec![wide(&WIDE)],
    ));
    cases.push(Case::new(
        "matmul.f32.offset".to_string(),
        Call::Matmul,
        vec![wide(&WIDE), wide_t(&WIDE_B)],
    ));
    cases.push(Case::new(
        "index_select.f32.offset".to_string(),
        Call::IndexSelect(1),
        vec![wide(&WIDE), i64s(&[3], &[2, 0, 1])],
    ));
    cases.push(Case::new(
        "gather.f32.offset".to_string(),
        Call::Gather(1),
        vec![wide(&WIDE), i64s(&[2, 2], &[2, 0, 1, 1])],
    ));
    cases.push(Case::new(
        "index_add.f32.offset".to_string(),
        Call::IndexAdd(0),
        vec![wide(&WIDE), i64s(&[2], &[1, 1]), wide(&WIDE_B)],
    ));
    cases.push(Case::new(
        "scatter_add.f32.offset".to_string(),
        Call::ScatterAdd(1),
        vec![wide(&WIDE), i64s(&[2, 2], &[2, 0, 1, 1]), wide(&WIDE_B)],
    ));
    // A conv whose input is an offset window of a larger buffer.
    let conv_offset =
        Layout::from_parts(Shape::from([1, 1, 4, 4]), Box::from([16usize, 16, 4, 1]), 4)
            .expect("conformance table: offset conv layout");
    let conv_input: Vec<f32> = (0..20).map(|i| i as f32 * 0.5 - 3.0).collect();
    cases.push(Case::new(
        "conv.Conv2d.f32.offset".to_string(),
        Call::Conv(ConvOp::Conv2d, CONV_DENSE),
        vec![
            Operand::strided(HostConv::into_cpu_storage(conv_input), conv_offset),
            f32s(&[2, 1, 2, 2], &[1.0, -0.5, 0.25, 2.0, 0.0, 1.5, -1.0, 0.75]),
        ],
    ));
}

/// The dense 2×2 geometry, shared by the offset-conv and F64 conv rows.
const CONV_DENSE: Conv2dParams = Conv2dParams {
    kernel: (2, 2),
    stride: (1, 1),
    padding: (0, 0),
    dilation: (1, 1),
};

/// [`BackendOps::copy_into`], the entry point no test reached at all.
///
/// It is the device-side assembly primitive behind `cat`/`stack`/`pad`, and
/// unlike every other entry point its result is a *mutated destination*, so
/// the row hands the whole destination back: a kernel that writes the region
/// contiguously instead of through `dst_layout` corrupts the surroundings, and
/// only comparing the untouched slots catches it.
fn push_copy_into(cases: &mut Vec<Case>) {
    for (label, src) in [
        ("f32", f32s(&[2, 3], &A_F32)),
        ("i64", i64s(&[2, 3], &A_I64)),
        ("bool", bools(&[2, 3], &A_BOOL)),
        ("f16", reduceds(DType::F16, &[2, 3], &A_F32)),
        ("bf16", reduceds(DType::BF16, &[2, 3], &A_F32)),
        ("f64", floats(DType::F64, &[2, 3], &A_F32)),
    ] {
        // Copy into rows 1..3 of a [4, 3] destination.
        cases.push(Case::new(
            format!("copy_into.{label}.axis0"),
            Call::CopyInto {
                dst_dims: vec![4, 3],
                fill: -1.0,
                axis: 0,
                start: 1,
            },
            vec![src],
        ));
    }
    // A destination region on the *inner* axis is the one `cat` uses that is
    // not a straight run of storage.
    cases.push(Case::new(
        "copy_into.f32.axis1".to_string(),
        Call::CopyInto {
            dst_dims: vec![2, 7],
            fill: -1.0,
            axis: 1,
            start: 2,
        },
        vec![f32s(&[2, 3], &A_F32)],
    ));
    // …and a strided source into that inner region: both cursors non-trivial.
    cases.push(Case::new(
        "copy_into.f32.strided_src".to_string(),
        Call::CopyInto {
            dst_dims: vec![3, 5],
            fill: -1.0,
            axis: 1,
            start: 1,
        },
        vec![f32s_transposed(&A_F32)],
    ));
    cases.push(Case::new(
        "copy_into.f32.broadcast_src".to_string(),
        Call::CopyInto {
            dst_dims: vec![3, 4],
            fill: -1.0,
            axis: 1,
            start: 1,
        },
        vec![f32s_broadcast(&[1.0, 2.0, 3.0])],
    ));
    cases.push(Case::new(
        "copy_into.f32.empty".to_string(),
        Call::CopyInto {
            dst_dims: vec![2, 3],
            fill: -1.0,
            axis: 1,
            start: 1,
        },
        vec![f32s(&[2, 0], &[])],
    ));
}

/// The `F64` lane, which the table never touched.
///
/// The CPU kernels implement `F64` in full — every match arm is live — but
/// before this section nothing in the suite instantiated one, so a broken
/// `F64` arm in any kernel was invisible to the conformance gate. Metal
/// declares `F64` outside its capability, so these rows land in
/// `expected_unsupported` there and hold only the reference to account, which
/// is exactly what a per-backend capability declaration is for.
fn push_f64(cases: &mut Vec<Case>) {
    const D: DType = DType::F64;
    cases.push(Case::new(
        "full.f64".to_string(),
        Call::Full {
            len: 6,
            dtype: D,
            value: -1.5,
        },
        vec![],
    ));
    // Both F64 cast directions are the deferred lane: they must decline
    // loudly, and the row is what proves the decline still happens.
    cases.push(Case::new(
        "cast.f32_to_f64".to_string(),
        Call::Cast(D),
        vec![f32s(&[2, 3], &A_F32)],
    ));
    cases.push(Case::new(
        "cast.f64_to_f32".to_string(),
        Call::Cast(DType::F32),
        vec![floats(D, &[2, 3], &A_F32)],
    ));
    cases.push(Case::new(
        "copy_strided.f64.transposed".to_string(),
        Call::CopyStrided,
        vec![floats_strided(
            D,
            &A_F32,
            Layout::contiguous([2, 3])
                .and_then(|layout| layout.transpose(0, 1))
                .expect("conformance table: transposable f64 layout"),
        )],
    ));
    for op in BINARY_OPS {
        cases.push(Case::new(
            format!("binary.{op:?}.f64"),
            Call::Binary(op),
            vec![floats(D, &[2, 3], &A_F32), floats(D, &[2, 3], &B_F32)],
        ));
        cases.push(Case::new(
            format!("binary_scalar.{op:?}.f64"),
            Call::BinaryScalar(op, 2.5),
            vec![floats(D, &[2, 3], &A_F32)],
        ));
    }
    for op in UNARY_OPS {
        cases.push(Case::new(
            format!("unary.{op:?}.f64"),
            Call::Unary(op),
            vec![floats(D, &[2, 3], &P_F32)],
        ));
    }
    for op in CMP_OPS {
        cases.push(Case::new(
            format!("compare.{op:?}.f64"),
            Call::Compare(op),
            vec![floats(D, &[2, 3], &A_F32), floats(D, &[2, 3], &B_F32)],
        ));
    }
    cases.push(Case::new(
        "where_cond.f64".to_string(),
        Call::WhereCond,
        vec![
            bools(&[2, 3], &A_BOOL),
            floats(D, &[2, 3], &A_F32),
            floats(D, &[2, 3], &B_F32),
        ],
    ));
    cases.push(Case::new(
        "masked_fill.f64".to_string(),
        Call::MaskedFill(-7.0),
        vec![floats(D, &[2, 3], &A_F32), bools(&[2, 3], &A_BOOL)],
    ));
    for op in REDUCE_OPS {
        for axis in [0usize, 1] {
            cases.push(Case::new(
                format!("reduce.{op:?}.f64.axis{axis}"),
                Call::Reduce(op, axis),
                vec![floats(D, &[2, 3], &A_F32)],
            ));
        }
    }
    for op in [ArgReduceOp::ArgMax, ArgReduceOp::ArgMin] {
        cases.push(Case::new(
            format!("arg_reduce.{op:?}.f64"),
            Call::ArgReduce(op, 1),
            vec![floats(D, &[2, 3], &A_F32)],
        ));
    }
    cases.push(Case::new(
        "matmul.f64.2d".to_string(),
        Call::Matmul,
        vec![floats(D, &[2, 3], &A_F32), floats(D, &[3, 2], &B_F32)],
    ));
    cases.push(Case::new(
        "index_select.f64.axis1".to_string(),
        Call::IndexSelect(1),
        vec![floats(D, &[2, 3], &A_F32), i64s(&[4], &[2, 0, 2, 1])],
    ));
    cases.push(Case::new(
        "index_add.f64.axis0".to_string(),
        Call::IndexAdd(0),
        vec![
            floats(D, &[2, 3], &A_F32),
            i64s(&[1], &[1]),
            floats(D, &[1, 3], &[1.0, 2.0, 3.0]),
        ],
    ));
    cases.push(Case::new(
        "gather.f64.axis1".to_string(),
        Call::Gather(1),
        vec![floats(D, &[2, 3], &A_F32), i64s(&[2, 2], &[0, 2, 1, 1])],
    ));
    cases.push(Case::new(
        "scatter_add.f64.axis1".to_string(),
        Call::ScatterAdd(1),
        vec![
            floats(D, &[2, 3], &A_F32),
            i64s(&[2, 2], &[0, 2, 1, 1]),
            floats(D, &[2, 2], &[10.0, 20.0, 30.0, 40.0]),
        ],
    ));
    let conv_input: Vec<f32> = (0..16).map(|i| i as f32 * 0.5 - 3.0).collect();
    let conv_weight = [1.0f32, -0.5, 0.25, 2.0, 0.0, 1.5, -1.0, 0.75];
    cases.push(Case::new(
        "conv.Conv2d.f64.dense".to_string(),
        Call::Conv(ConvOp::Conv2d, CONV_DENSE),
        vec![
            floats(D, &[1, 1, 4, 4], &conv_input),
            floats(D, &[2, 1, 2, 2], &conv_weight),
        ],
    ));
    for op in [ConvOp::MaxPool2d, ConvOp::AvgPool2d] {
        cases.push(Case::new(
            format!("conv.{op:?}.f64.dense"),
            Call::Conv(op, CONV_DENSE),
            vec![floats(D, &[1, 1, 4, 4], &conv_input)],
        ));
    }
}

/// A non-uniform, deterministic cotangent for `dims`.
fn cotangent(dims: [usize; 4]) -> Vec<f32> {
    let len: usize = dims.iter().product();
    (0..len).map(|i| 0.75 - (i % 7) as f32 * 0.25).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The self-check: the reference backend against itself. It proves the
    /// whole table is *runnable* (no malformed rows, no hard errors) and
    /// exactly matched, which is what another backend is diffed against.
    #[test]
    fn cpu_is_conformant_with_itself() {
        let report = run_device(Device::Cpu);
        let matched = report.matched.len();
        let skipped = report.expected_unsupported.clone();
        assert!(
            report.skipped.is_empty(),
            "unexpected skips: {:?}",
            report.skipped
        );
        report.into_result(Device::Cpu).expect("cpu self-check");
        assert!(matched > 100, "suite is too thin: {matched} matched cases");
        // Exactly the rows the op enums put out of contract — nothing else may
        // appear here, because anything else means a kernel silently lost a
        // dtype and `expected_unsupported` quietly excused it. Note that the
        // accelerator-gap tier is invisible from here by construction: it is
        // gated on a non-CPU candidate, so every op in it is still compared
        // against the reference for real on this lane.
        //
        // - `BinaryOp::Pow`'s integer lanes, which it declines rather than
        //   invent a meaning for a fractional exponent;
        // - the float-only unaries on `I64` (`UnaryOp`'s contract);
        // - every accumulating or arithmetic entry point on `Bool`, which has
        //   no `NumAcc` and no arithmetic — `arg_sort` included, since it
        //   orders in `Acc`;
        // - every fused variant on `I64`/`Bool`, which have no `FloatAcc`;
        // - both `F64` cast lanes, which `BackendOps::cast` defers.
        let expected: Vec<String> = ["binary.Pow.i64", "binary_scalar.Pow.i64"]
            .iter()
            .map(|name| (*name).to_string())
            .chain(
                [
                    "Relu", "Gelu", "Exp", "Ln", "Sqrt", "Tanh", "Sigmoid", "Sign", "Recip",
                    "Floor", "Ceil", "Round", "Erf",
                ]
                .iter()
                .map(|op| format!("unary.{op}.i64")),
            )
            .chain(
                ["Sum", "Mean", "Max", "Min", "Prod"]
                    .iter()
                    .map(|op| format!("reduce.{op}.bool")),
            )
            .chain(
                [
                    "fused.Softmax.i64",
                    "fused.Softmax.bool",
                    "fused.LayerNorm.i64",
                    "fused.SgdStep.i64",
                    "fused.AdamStep.i64",
                    "binary.Pow.i64.extremes",
                    "binary_scalar.Pow.i64.zero",
                    "binary_scalar.Pow.i64.minus_one",
                    "cast.f32_to_f64",
                    "cast.f64_to_f32",
                    "binary.Add.bool",
                    "binary_scalar.Add.bool",
                    "unary.Neg.bool",
                    "arg_reduce.ArgMax.bool",
                    "arg_sort.bool",
                    "matmul.bool",
                    "index_add.bool",
                    "scatter_add.bool",
                    "conv.Conv2d.bool",
                    "conv.MaxPool2d.bool",
                ]
                .iter()
                .map(|name| (*name).to_string()),
            )
            .collect();
        assert_eq!(skipped, expected);
    }

    /// Every case name is unique, so a failure report identifies exactly one
    /// table row.
    #[test]
    fn case_names_are_unique() {
        let mut names: Vec<String> = suite().into_iter().map(|c| c.name).collect();
        let total = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), total, "duplicate case name in the table");
    }

    #[test]
    fn tolerance_follows_the_output_not_the_operands() {
        let widening = Case::new(
            "cast.f16_to_f32.exact".to_string(),
            Call::Cast(DType::F32),
            vec![reduceds(DType::F16, &[2], &[1.0, -2.0])],
        );
        assert_eq!(widening.tol, 0.0);

        let comparison = Case::new(
            "compare.bf16.exact".to_string(),
            Call::Compare(CmpOp::Eq),
            vec![
                reduceds(DType::BF16, &[2], &[1.0, -2.0]),
                reduceds(DType::BF16, &[2], &[1.0, -2.0]),
            ],
        );
        assert_eq!(comparison.tol, 0.0);

        let arithmetic = Case::new(
            "binary.bf16.output".to_string(),
            Call::Binary(BinaryOp::Add),
            vec![
                reduceds(DType::BF16, &[2], &[1.0, -2.0]),
                reduceds(DType::BF16, &[2], &[1.0, -2.0]),
            ],
        );
        assert_eq!(arithmetic.tol, BF16_TOL);

        let selection = Case::new(
            "index_select.bf16.exact".to_string(),
            Call::IndexSelect(0),
            vec![reduceds(DType::BF16, &[2], &[1.0, -2.0]), i64s(&[1], &[1])],
        );
        assert_eq!(selection.tol, 0.0);
    }

    /// The comparator has teeth: it is the part that would silently pass a
    /// broken backend if it were wrong.
    #[test]
    fn comparator_detects_divergence() {
        let a = HostConv::into_cpu_storage(vec![1.0f32, 2.0, 3.0]);
        let b = HostConv::into_cpu_storage(vec![1.0f32, 2.0, 3.5]);
        let err = compare(&a, &b, DEFAULT_TOL).expect_err("must diverge");
        assert!(err.contains("element 2"), "{err}");

        // Within tolerance is not a divergence.
        let near = HostConv::into_cpu_storage(vec![1.0f32, 2.0, 3.000_001]);
        compare(&a, &near, DEFAULT_TOL).expect("within tolerance");

        // Integers and bools are exact, and dtype/length differ loudly.
        let i = HostConv::into_cpu_storage(vec![1i64, 2]);
        let j = HostConv::into_cpu_storage(vec![1i64, 3]);
        assert!(compare(&i, &j, DEFAULT_TOL).is_err());
        assert!(compare(&a, &i, DEFAULT_TOL).unwrap_err().contains("dtype"));
        let short = HostConv::into_cpu_storage(vec![1.0f32, 2.0]);
        assert!(
            compare(&a, &short, DEFAULT_TOL)
                .unwrap_err()
                .contains("length")
        );
    }

    /// NaN matches NaN, infinities match by sign, and a sign flip does not.
    #[test]
    fn closeness_handles_non_finite_values() {
        assert!(close(f64::NAN, f64::NAN, 0.0));
        assert!(!close(f64::NAN, 0.0, 1.0));
        assert!(close(f64::INFINITY, f64::INFINITY, 0.0));
        assert!(!close(f64::INFINITY, f64::NEG_INFINITY, 1.0));
        assert!(!close(f64::INFINITY, 1e300, 1.0));
    }

    /// An arity error is reported, not a panic, if a row is malformed.
    #[test]
    fn wrong_arity_is_an_error() {
        let case = Case::new(
            "bad.binary".to_string(),
            Call::Binary(BinaryOp::Add),
            vec![f32s(&[2], &[1.0, 2.0])],
        );
        let Err(err) = evaluate(dispatch::backend(Device::Cpu), &case) else {
            panic!("a one-operand `binary` row must be an arity error");
        };
        assert!(matches!(
            err,
            Error::InvalidArg {
                op: "conformance",
                ..
            }
        ));
    }
}
