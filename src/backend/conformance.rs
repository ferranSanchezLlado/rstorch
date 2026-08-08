//! The table-driven backend conformance harness: one op × dtype table, run
//! against the CPU reference (exploration §4.5, implementation-plan §4 T19).
//!
//! CPU is *the* reference implementation. Every other backend is validated
//! against it by this one suite rather than by a per-backend pile of
//! bespoke tests, so adding backend #5 costs "one module + a conformance
//! run" (exploration §7) instead of a new test corpus.
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
//! - **Dtypes**: `F16`, `BF16`, `F32`, `I64`, `Bool`, with dtype-appropriate
//!   tolerances. Each dtype is a separate row, so a future Metal backend can
//!   honestly report BF16 as unsupported without hiding its F16 coverage.
//! - **Layouts**: cases deliberately include transposed and broadcast
//!   views, because the kernel contract is stride-aware.
//! - **`Unsupported` is not a mismatch.** A backend that reports
//!   [`Error::Unsupported`] for a case has *loudly* declined it (there are
//!   no silent fallbacks). Declared out-of-scope rows land in
//!   `Report::expected_unsupported`; any other decline lands in
//!   `Report::skipped`, which a promotion gate requires to be empty.
//! - **Fused ops are absent from the table.** Their multi-output encodings do
//!   not fit this single-output harness; softmax, LayerNorm, and optimizer
//!   kernels instead have direct dtype-specific CPU tests. T61 may extend the
//!   harness when accelerator fused parity lands.
//!
//! Today the only backend is CPU, so the shipped test is the self-check
//! (CPU vs CPU): it proves the whole table is runnable and exactly matched
//! on the reference. T61 reuses `run` unchanged for Metal.

use crate::backend::{
    ArgReduceOp, BackendOps, BinaryOp, CmpOp, Conv2dParams, ConvOp, ReduceOp, UnaryOp, View,
    conv_geometry::Conv2dGeometry, dispatch,
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
    /// [`BackendOps::conv`] with the given geometry.
    Conv(ConvOp, Conv2dParams),
}

impl Call {
    /// Invoke the entry point on `backend` with `views` bound in table order.
    fn apply(&self, backend: &dyn BackendOps, views: &[View<'_>]) -> Result<Storage> {
        let arity = |want: usize| Error::InvalidArg {
            op: "conformance",
            msg: format!("expected {want} operand view(s), got {}", views.len()),
        };
        match (self, views) {
            (Call::Full { len, dtype, value }, []) => backend.full(*len, *dtype, *value),
            (Call::Cast(to), [x]) => backend.cast(*x, *to),
            (Call::CopyStrided, [x]) => backend.copy_strided(*x),
            (Call::Binary(op), [a, b]) => backend.binary(*op, *a, *b),
            (Call::BinaryScalar(op, s), [x]) => backend.binary_scalar(*op, *x, *s),
            (Call::Unary(op), [x]) => backend.unary(*op, *x),
            (Call::Compare(op), [a, b]) => backend.compare(*op, *a, *b),
            (Call::WhereCond, [c, t, f]) => backend.where_cond(*c, *t, *f),
            (Call::MaskedFill(v), [x, m]) => backend.masked_fill(*x, *m, *v),
            (Call::Reduce(op, axis), [x]) => backend.reduce(*op, *x, *axis),
            (Call::ArgReduce(op, axis), [x]) => backend.arg_reduce(*op, *x, *axis),
            (Call::Matmul, [a, b]) => backend.matmul(*a, *b),
            (Call::IndexSelect(axis), [x, i]) => backend.index_select(*x, *axis, *i),
            (Call::IndexAdd(axis), [x, i, s]) => backend.index_add(*x, *axis, *i, *s),
            (Call::Gather(axis), [x, i]) => backend.gather(*x, *axis, *i),
            (Call::ScatterAdd(axis), [x, i, s]) => backend.scatter_add(*x, *axis, *i, *s),
            (Call::Conv(op, params), inputs) => backend.conv(*op, inputs, params),
            (Call::Full { .. }, _) => Err(arity(0)),
            (Call::Cast(_) | Call::CopyStrided | Call::BinaryScalar(..), _) => Err(arity(1)),
            (Call::Unary(_) | Call::Reduce(..) | Call::ArgReduce(..), _) => Err(arity(1)),
            (Call::Binary(_) | Call::Compare(_) | Call::Matmul, _) => Err(arity(2)),
            (Call::MaskedFill(_) | Call::IndexSelect(_) | Call::Gather(_), _) => Err(arity(2)),
            (Call::WhereCond | Call::IndexAdd(_) | Call::ScatterAdd(_), _) => Err(arity(3)),
        }
    }
}

/// One row of the conformance table.
pub(crate) struct Case {
    /// Human-readable identity, e.g. `"binary.Add.f32"`; appears verbatim in
    /// failure reports.
    pub(crate) name: String,
    call: Call,
    operands: Vec<Operand>,
    tol: f64,
}

impl Case {
    /// A case with the default float tolerance.
    fn new(name: String, call: Call, operands: Vec<Operand>) -> Case {
        let output_dtype = match &call {
            Call::Full { dtype, .. } | Call::Cast(dtype) => *dtype,
            Call::Compare(_) => DType::Bool,
            Call::ArgReduce(..) => DType::I64,
            Call::WhereCond => operands[1].host.dtype(),
            Call::MaskedFill(_)
            | Call::CopyStrided
            | Call::BinaryScalar(..)
            | Call::Unary(_)
            | Call::Reduce(..)
            | Call::IndexSelect(_)
            | Call::IndexAdd(_)
            | Call::Gather(_)
            | Call::ScatterAdd(_)
            | Call::Conv(..) => operands[0].host.dtype(),
            Call::Binary(_) | Call::Matmul => operands[0].host.dtype(),
        };
        // Casts are deterministic representation conversions. In particular,
        // widening a reduced value to F32 must reproduce it exactly; input
        // dtype never grants slack to an exact output.
        let exact_output = matches!(
            call,
            Call::Full { .. }
                | Call::Cast(_)
                | Call::CopyStrided
                | Call::Compare(_)
                | Call::WhereCond
                | Call::MaskedFill(_)
                | Call::ArgReduce(..)
                | Call::IndexSelect(_)
                | Call::Gather(_)
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
    /// Cases that diverged or errored, one rendered message each.
    pub(crate) failures: Vec<String>,
}

impl Report {
    /// Turn the report into a single [`Result`], failing loudly with every
    /// divergence listed. `device` names the backend under test.
    pub(crate) fn into_result(self, device: Device) -> Result<()> {
        if self.failures.is_empty() {
            return Ok(());
        }
        Err(Error::Backend {
            op: "conformance",
            msg: format!(
                "{} of {} case(s) diverged from the cpu reference on {device}:\n  {}",
                self.failures.len(),
                self.matched.len()
                    + self.skipped.len()
                    + self.expected_unsupported.len()
                    + self.failures.len(),
                self.failures.join("\n  ")
            ),
        })
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
        failures: Vec::new(),
    };
    for case in suite() {
        match (evaluate(reference, &case), evaluate(candidate, &case)) {
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
            (Ok(want), Ok(got)) => match compare(&want, &got, case.tol) {
                Ok(()) => report.matched.push(case.name),
                Err(diff) => report.failures.push(format!("{}: {diff}", case.name)),
            },
        }
    }
    report
}

fn expected_unsupported(_device: Device, case: &Case) -> bool {
    let input_dtype = case.operands.first().map(|operand| operand.host.dtype());
    let outside_common_contract = matches!(
        (&case.call, input_dtype),
        (
            Call::Unary(
                UnaryOp::Relu
                    | UnaryOp::Gelu
                    | UnaryOp::Exp
                    | UnaryOp::Ln
                    | UnaryOp::Sqrt
                    | UnaryOp::Tanh
                    | UnaryOp::Sigmoid
            ),
            Some(DType::I64)
        ) | (Call::Reduce(..), Some(DType::Bool))
    );
    if outside_common_contract {
        return true;
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if matches!(_device, Device::Metal(_)) {
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
fn evaluate(backend: &dyn BackendOps, case: &Case) -> Result<CpuStorage> {
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
    let out = case.call.apply(backend, &views)?;
    let layout = Layout::contiguous([out.len()])?;
    backend.transfer_out(View::new(&out, &layout))
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
    cases
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

/// Binary, scalar-binary, unary, comparison, `where`, and `masked_fill`.
fn push_elementwise(cases: &mut Vec<Case>) {
    const BINARY: [BinaryOp; 6] = [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Maximum,
        BinaryOp::Minimum,
    ];
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

    const UNARY: [UnaryOp; 9] = [
        UnaryOp::Relu,
        UnaryOp::Gelu,
        UnaryOp::Exp,
        UnaryOp::Ln,
        UnaryOp::Sqrt,
        UnaryOp::Tanh,
        UnaryOp::Sigmoid,
        UnaryOp::Neg,
        UnaryOp::Abs,
    ];
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

    const CMP: [CmpOp; 6] = [
        CmpOp::Eq,
        CmpOp::Ne,
        CmpOp::Lt,
        CmpOp::Le,
        CmpOp::Gt,
        CmpOp::Ge,
    ];
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
    const REDUCE: [ReduceOp; 4] = [ReduceOp::Sum, ReduceOp::Mean, ReduceOp::Max, ReduceOp::Min];
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
}

/// Convolution and pooling geometry, including stride/padding/dilation.
fn push_conv(cases: &mut Vec<Case>) {
    let input: Vec<f32> = (0..16).map(|i| i as f32 * 0.5 - 3.0).collect();
    let weight: Vec<f32> = vec![1.0, -0.5, 0.25, 2.0, 0.0, 1.5, -1.0, 0.75];
    let geoms = [
        ("dense", (1, 1), (0, 0), (1, 1)),
        ("strided", (2, 2), (0, 0), (1, 1)),
        ("padded", (1, 1), (1, 1), (1, 1)),
        ("dilated", (1, 1), (0, 0), (2, 2)),
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

    let pool = Conv2dGeometry::pool("pool2d", &INPUT_DIMS, params)
        .expect("conformance table: valid pool geometry");
    let pool_grad = cotangent(pool.output_dims());
    for op in [ConvOp::MaxPool2dBackward, ConvOp::AvgPool2dBackward] {
        cases.push(Case::new(
            format!("conv.{op:?}.f32.{label}"),
            Call::Conv(op, *params),
            vec![
                f32s(&pool.output_dims(), &pool_grad),
                f32s(&INPUT_DIMS, input),
            ],
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
    /// exactly matched, which is what T61 will diff Metal against.
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
        // Exactly the rows the op enums put out of contract: the float-only
        // unaries on `I64`, and reductions over `Bool`. Anything else
        // appearing here means a kernel silently lost a dtype.
        let expected: Vec<String> = ["Relu", "Gelu", "Exp", "Ln", "Sqrt", "Tanh", "Sigmoid"]
            .iter()
            .map(|op| format!("unary.{op}.i64"))
            .chain(
                ["Sum", "Mean", "Max", "Min"]
                    .iter()
                    .map(|op| format!("reduce.{op}.bool")),
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
