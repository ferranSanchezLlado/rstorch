//! Crate-private backend layer: the [`BackendOps`] trait, the op-family
//! enums, the borrowed [`View`] passed to kernels, and the single dispatch
//! point ([`dispatch::backend`]). No public surface here;
//! the public face of a backend is the [`Device`](crate::Device) enum.
//!
//! # Design
//!
//! - **Coarse, enum-routed entry points.** Adding an element-wise op is a
//!   new [`BinaryOp`]/[`UnaryOp`] variant plus kernel cases; the trait does
//!   not grow. Adding a backend is one module + one [`Device`] variant.
//! - **Stride-aware.** Every kernel receives [`View`]s — a `(Storage,
//!   Layout)` pair — never a bare contiguous slice. A backend may
//!   materialize a contiguous copy internally as a private choice.
//! - **Dtype is runtime.** Views carry a [`DType`]; CPU kernels are
//!   internally generic over the sealed [`Element`](crate::dtype::Element)
//!   with the [`Acc`](crate::dtype::Element::Acc) wide-accumulation
//!   contract. A backend that cannot honor the contract for a dtype returns
//!   [`Error::Unsupported`](crate::Error::Unsupported).
//!
//! # Asynchronous result semantics (a first-class requirement, not an
//! optimization)
//!
//! The methods below are specified as **device-ordered**, not
//! synchronous. A returned [`Storage`] may denote a computation still in
//! flight on the device queue; subsequent kernel calls that consume it are
//! ordered after it by the backend. **Synchronization happens only at host
//! boundaries** — [`transfer_out`](BackendOps::transfer_out) (and the
//! `to_vec`/`to_scalar`/`item` tensor methods built on it). This is what
//! lets a GPU backend batch command-buffer encoding and defer the sync that
//! would otherwise make a GPU path train slower than CPU. The CPU backend is trivially
//! synchronous and satisfies the contract vacuously.

// The table-driven op × dtype harness that validates any backend
// against the CPU reference (`conformance::run_device`). Test-only: its sole
// caller is the Metal backend's conformance test, so it is not compiled into
// release builds.
#[cfg(test)]
pub(crate) mod conformance;
pub(crate) mod conv_geometry;
pub(crate) mod cpu;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) mod metal;
// The parallelism switch. Always compiled — the `rayon` feature is
// consulted *inside* it, so a kernel writes one loop body and never carries a
// `#[cfg]` arm of its own (see the module docs for why that matters).
pub(crate) mod parallel;

use crate::device::Device;
use crate::dtype::DType;
use crate::error::Result;
use crate::storage::{CpuStorage, Storage};

/// A borrowed, stride-aware view into a tensor's storage: the `(Storage,
/// Layout)` pair every kernel operates on.
///
/// Views are cheap to copy (two references). A kernel reads elements by
/// walking [`Layout::strides`](crate::layout::Layout::strides) from
/// [`Layout::offset`](crate::layout::Layout::offset); it must not assume
/// contiguity (use [`copy_strided`](BackendOps::copy_strided) or an internal
/// materialization if it needs a flat buffer).
#[derive(Clone, Copy)]
pub(crate) struct View<'a> {
    storage: &'a Storage,
    layout: &'a crate::layout::Layout,
}

impl<'a> View<'a> {
    /// Build a view from a storage buffer and a layout over it.
    pub(crate) fn new(storage: &'a Storage, layout: &'a crate::layout::Layout) -> View<'a> {
        View { storage, layout }
    }

    /// The underlying (possibly-shared, possibly-oversized) storage buffer.
    pub(crate) fn storage(&self) -> &'a Storage {
        self.storage
    }

    /// The layout describing this view over [`storage`](View::storage).
    pub(crate) fn layout(&self) -> &'a crate::layout::Layout {
        self.layout
    }

    /// The element dtype.
    pub(crate) fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    /// The device the storage lives on.
    pub(crate) fn device(&self) -> Device {
        self.storage.device()
    }
}

/// Element-wise binary ops (broadcasting handled in the op layer, which
/// passes pre-broadcast, shape-identical views).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BinaryOp {
    /// `lhs + rhs`.
    Add,
    /// `lhs - rhs`.
    Sub,
    /// `lhs * rhs`.
    Mul,
    /// `lhs / rhs`.
    Div,
    /// Element-wise maximum.
    Maximum,
    /// Element-wise minimum.
    Minimum,
}

/// Element-wise unary ops. Float-only variants error with
/// [`Error::Unsupported`](crate::Error::Unsupported) on integer/bool dtypes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UnaryOp {
    /// `max(x, 0)`.
    Relu,
    /// Gaussian error linear unit. **Exact** GELU
    /// (`0.5·x·(1+erf(x/√2))`), not the tanh approximation
    /// (the familiar-semantics contract).
    Gelu,
    /// `exp(x)`.
    Exp,
    /// Natural logarithm.
    Ln,
    /// Square root.
    Sqrt,
    /// Hyperbolic tangent.
    Tanh,
    /// Logistic sigmoid `1/(1+exp(-x))`.
    Sigmoid,
    /// Negation `-x` (also valid for I64).
    Neg,
    /// Absolute value (valid for float and I64).
    Abs,
}

/// Element-wise comparisons; every variant produces a [`Bool`](DType::Bool)
/// result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CmpOp {
    /// `lhs == rhs`.
    Eq,
    /// `lhs != rhs`.
    Ne,
    /// `lhs < rhs`.
    Lt,
    /// `lhs <= rhs`.
    Le,
    /// `lhs > rhs`.
    Gt,
    /// `lhs >= rhs`.
    Ge,
}

/// Axis reductions. The result drops the reduced axis (the op layer
/// re-inserts it for the `_keepdim` spellings). Accumulation is in
/// [`Element::Acc`](crate::dtype::Element::Acc); `Mean` divides the wide
/// accumulator by the (wide) count and casts once at output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReduceOp {
    /// Sum along the axis.
    Sum,
    /// Arithmetic mean along the axis.
    Mean,
    /// Maximum along the axis.
    Max,
    /// Minimum along the axis.
    Min,
}

/// Index-producing reductions (`argmax`/`argmin`), returning an
/// [`I64`](DType::I64) tensor of positions along the reduced axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ArgReduceOp {
    /// Position of the maximum (first on ties).
    ArgMax,
    /// Position of the minimum (first on ties).
    ArgMin,
}

/// Convolution/pooling ops routed through one entry point
/// ([`BackendOps::conv`]); geometry travels in [`Conv2dParams`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConvOp {
    /// 2-D cross-correlation. Inputs: `[input, weight]` (+ optional bias
    /// folded in by the op layer). Accumulates in `Acc`.
    Conv2d,
    /// 2-D max pooling. Input: `[input]`.
    MaxPool2d,
    /// 2-D average pooling. Input: `[input]`. Accumulates in `Acc`.
    AvgPool2d,
    /// Input gradient of 2-D cross-correlation. Inputs:
    /// `[grad, weight, original_input]`; the original input supplies the
    /// requested output shape and completes the forward geometry.
    Conv2dInputGrad,
    /// Weight gradient of 2-D cross-correlation. Inputs:
    /// `[grad, original_input, original_weight]`; the original weight
    /// supplies the requested output shape and completes the geometry.
    Conv2dWeightGrad,
    /// Input gradient of max pooling. Inputs: `[grad, original_input]`.
    MaxPool2dBackward,
    /// Input gradient of average pooling. Inputs: `[grad, original_input]`.
    AvgPool2dBackward,
}

/// Geometry for [`ConvOp`] kernels. All pairs are `(height, width)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Conv2dParams {
    /// Kernel spatial size `(kh, kw)`. For pooling this is the window.
    pub(crate) kernel: (usize, usize),
    /// Stride `(sh, sw)`.
    pub(crate) stride: (usize, usize),
    /// Zero-padding `(ph, pw)` added to both sides of each spatial axis.
    pub(crate) padding: (usize, usize),
    /// Dilation `(dh, dw)` (1 = dense). Pooling ignores dilation.
    pub(crate) dilation: (usize, usize),
}

/// Fused kernels: softmax, layernorm, and the
/// optimizer updates that reclaim the fixed per-parameter allocation
/// hotspot. Optional — [`BackendOps::fused`] returns
/// [`Error::Unsupported`](crate::Error::Unsupported) until it is implemented for a
/// given variant, and the op layer composes the unfused form (the only
/// sanctioned fallback: same device, no host round-trip).
///
/// Encodings are crate-private but frozen across the kernel and its callers:
/// `Softmax` takes `[x]`/`[]` and returns `[y]`. `LayerNorm` forward takes
/// `[x, weight, bias]`/`[eps]` and returns `[y]`, or accepts
/// `[eps, save_stats=1]` and returns `[y, xhat, inv_std]`. `y` has the input
/// dtype; saved `xhat`/`inv_std` use its accumulation dtype (F32 for
/// F16/BF16). Its input-gradient form takes
/// `[grad, xhat_acc, inv_std_acc, weight]`/`[]` and returns `[grad_x]`, with
/// `grad`/`weight` in parameter dtype and saved statistics in accumulation
/// dtype.
/// `SgdStep` takes
/// `[param, grad]` or `[param, grad, velocity]` plus
/// `[lr, momentum, weight_decay]` and returns `[next_param]` or
/// `[next_param, next_velocity]`; `AdamStep` takes `[param, grad, m, v]` plus
/// `[lr, beta1, beta2, eps, weight_decay, bias_correction1,
/// bias_correction2, decoupled]` and returns `[next_param, next_m, next_v]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FusedOp {
    /// Numerically stable softmax over the last view axis.
    Softmax,
    /// Layer normalization over the last view axis.
    LayerNorm,
    /// In-one-pass SGD (+momentum) parameter update.
    SgdStep,
    /// In-one-pass Adam/AdamW parameter update.
    AdamStep,
}

/// The crate-private backend contract: coarse, stride-aware, enum-routed
/// entry points with device-ordered results (see the module docs for the
/// asynchronous-result and `Acc` contracts). One implementation per
/// [`Device`]; the CPU one is the conformance reference.
///
/// Object-safe by construction (no generics; dtype is carried at runtime by
/// the [`View`]s), so [`dispatch::backend`] can hand back a
/// `&'static dyn BackendOps`.
///
/// # Ops composed in the op layer, not here (so the trait does not grow)
///
/// `cat`/`stack` are **not** trait entry points: they allocate a
/// contiguous output ([`full`](BackendOps::full)) and assemble it from
/// [`narrow`](crate::layout::Layout::narrow) views of the inputs via
/// per-region [`copy_strided`](BackendOps::copy_strided). On CPU this is
/// exact and allocation-minimal. A future GPU backend that cannot express a
/// device-side region copy through the existing entry points would add one
/// crate-private `copy_into` primitive at that point — a lock-free,
/// non-semver change, since this trait is crate-private — rather than a host
/// round-trip, which the no-silent-fallback policy
/// forbids.
pub(crate) trait BackendOps: Send + Sync {
    /// Upload a host buffer, producing device storage. For CPU this wraps
    /// the buffer unchanged.
    fn transfer_in(&self, host: CpuStorage) -> Result<Storage>;

    /// Download `x` to host memory as a **contiguous** [`CpuStorage`],
    /// materializing the strided view. This is a host boundary: the backend
    /// synchronizes here.
    fn transfer_out(&self, x: View<'_>) -> Result<CpuStorage>;

    /// Materialize `x` into a fresh contiguous device buffer of the same
    /// dtype and shape (the public `contiguous()` and the copy branch of
    /// `reshape`).
    fn copy_strided(&self, x: View<'_>) -> Result<Storage>;

    /// Copy `src` into an equally shaped region of an existing destination.
    /// This construction-only primitive is the device-resident assembly path
    /// for `cat` and `stack`; callers do not publish `dst` until all copies
    /// have been encoded.
    fn copy_into(
        &self,
        src: View<'_>,
        dst: &mut Storage,
        dst_layout: &crate::layout::Layout,
    ) -> Result<()>;

    /// Allocate `len` elements of `dtype` filled with `value` (cast to the
    /// dtype; used by `zeros`/`ones`/`full`). `value` is a scalar carried as
    /// `f64` and narrowed by the kernel.
    fn full(&self, len: usize, dtype: DType, value: f64) -> Result<Storage>;

    /// Cast `x` to dtype `to`. Implemented lanes are `F32↔I64`, `F32↔Bool`,
    /// `I64↔Bool`, plus F16/BF16 with F32, each other, I64, and Bool. F64
    /// conversion remains outside the current cast scope. An unimplemented
    /// lane is [`Error::Unsupported`](crate::Error::Unsupported), never a
    /// silent reinterpretation.
    fn cast(&self, x: View<'_>, to: DType) -> Result<Storage>;

    /// Element-wise binary op over two **pre-broadcast, shape-identical**
    /// views (the op layer broadcasts via
    /// [`Layout::broadcast_to`](crate::layout::Layout::broadcast_to)).
    fn binary(&self, op: BinaryOp, lhs: View<'_>, rhs: View<'_>) -> Result<Storage>;

    /// Element-wise binary op against an `f64` scalar (narrowed to the
    /// view's dtype), for the `+ 1.0`-style scalar spellings.
    fn binary_scalar(&self, op: BinaryOp, x: View<'_>, scalar: f64) -> Result<Storage>;

    /// Element-wise unary op (see [`UnaryOp`] for the exact-GELU and
    /// float-only contracts).
    fn unary(&self, op: UnaryOp, x: View<'_>) -> Result<Storage>;

    /// Element-wise comparison over two pre-broadcast views, producing a
    /// [`Bool`](DType::Bool) result.
    fn compare(&self, op: CmpOp, lhs: View<'_>, rhs: View<'_>) -> Result<Storage>;

    /// `where(cond, on_true, on_false)` over three pre-broadcast views;
    /// `cond` is [`Bool`](DType::Bool), the value views share a dtype.
    fn where_cond(&self, cond: View<'_>, on_true: View<'_>, on_false: View<'_>) -> Result<Storage>;

    /// Replace elements of `x` where `mask` (broadcast, [`Bool`](DType::Bool))
    /// is true with `value` (narrowed to `x`'s dtype).
    fn masked_fill(&self, x: View<'_>, mask: View<'_>, value: f64) -> Result<Storage>;

    /// Reduce `x` along `axis` (pre-resolved to `[0, rank)`), dropping that
    /// axis. Accumulates in [`Element::Acc`](crate::dtype::Element::Acc).
    /// The empty-reduction policy is enforced by the op
    /// layer before calling.
    fn reduce(&self, op: ReduceOp, x: View<'_>, axis: usize) -> Result<Storage>;

    /// Index-producing reduction (`argmax`/`argmin`) along `axis`, returning
    /// an [`I64`](DType::I64) tensor.
    fn arg_reduce(&self, op: ArgReduceOp, x: View<'_>, axis: usize) -> Result<Storage>;

    /// Matrix multiply. Both views are rank ≥ 2; leading axes are batch
    /// dims that broadcast against each other. Inner dims must agree (the op
    /// layer raises [`Error::ShapeMismatch`](crate::Error::ShapeMismatch)
    /// otherwise). Inner products accumulate in
    /// [`Element::Acc`](crate::dtype::Element::Acc).
    fn matmul(&self, lhs: View<'_>, rhs: View<'_>) -> Result<Storage>;

    /// Select slices of `x` along `axis` at the positions in the 1-D
    /// [`I64`](DType::I64) `indices` view. Indices are bounds-checked
    /// ([`Error::IndexOutOfBounds`](crate::Error::IndexOutOfBounds)). Its
    /// backward is [`index_add`](BackendOps::index_add) (slice-wise), the hot
    /// path for [`Embedding`](crate::nn) — do not route it through the
    /// same-rank [`scatter_add`](BackendOps::scatter_add), which would force
    /// the op layer to materialize a full same-shape index grid.
    fn index_select(&self, x: View<'_>, axis: usize, indices: View<'_>) -> Result<Storage>;

    /// Accumulate `src` slices into a copy of `x` along `axis` at the
    /// positions in the 1-D [`I64`](DType::I64) `indices` view (the backward
    /// of [`index_select`](BackendOps::index_select); PyTorch `index_add`
    /// semantics — whole slices, 1-D index). Accumulates in
    /// [`Acc`](crate::dtype::Element::Acc). Bounds-checked.
    fn index_add(
        &self,
        x: View<'_>,
        axis: usize,
        indices: View<'_>,
        src: View<'_>,
    ) -> Result<Storage>;

    /// Gather along `axis` using a same-rank [`I64`](DType::I64) `indices`
    /// view (PyTorch `gather` semantics). Bounds-checked.
    fn gather(&self, x: View<'_>, axis: usize, indices: View<'_>) -> Result<Storage>;

    /// Scatter-add `src` into a copy of `x` along `axis` at a **same-rank**
    /// `indices` view (the backward of [`gather`](BackendOps::gather),
    /// per-element placement); accumulates in
    /// [`Acc`](crate::dtype::Element::Acc). Bounds-checked. For the 1-D
    /// slice-wise case use [`index_add`](BackendOps::index_add).
    fn scatter_add(
        &self,
        x: View<'_>,
        axis: usize,
        indices: View<'_>,
        src: View<'_>,
    ) -> Result<Storage>;

    /// Convolution/pooling (see [`ConvOp`]/[`Conv2dParams`]). `inputs` holds
    /// the per-variant operand views; accumulation is in `Acc`.
    fn conv(&self, op: ConvOp, inputs: &[View<'_>], params: &Conv2dParams) -> Result<Storage>;

    /// A fused kernel (see [`FusedOp`]). Returns
    /// [`Error::Unsupported`](crate::Error::Unsupported) for any variant not
    /// yet implemented, so the op layer falls back to the composed form.
    fn fused(&self, op: FusedOp, inputs: &[View<'_>], scalars: &[f64]) -> Result<Vec<Storage>>;
}

/// The CPU reference backend: a zero-sized dispatcher that delegates each
/// entry point to its per-family kernel module (`cpu::host`, `cpu::elementwise`,
/// `cpu::reduce`, `cpu::matmul`, `cpu::index`, `cpu::conv`, `cpu::fused`).
pub(crate) struct CpuBackend;

impl BackendOps for CpuBackend {
    fn transfer_in(&self, host: CpuStorage) -> Result<Storage> {
        cpu::host::transfer_in(host)
    }
    fn transfer_out(&self, x: View<'_>) -> Result<CpuStorage> {
        cpu::host::transfer_out(x)
    }
    fn copy_strided(&self, x: View<'_>) -> Result<Storage> {
        cpu::host::copy_strided(x)
    }
    fn copy_into(
        &self,
        src: View<'_>,
        dst: &mut Storage,
        dst_layout: &crate::layout::Layout,
    ) -> Result<()> {
        cpu::host::copy_into(src, dst, dst_layout)
    }
    fn full(&self, len: usize, dtype: DType, value: f64) -> Result<Storage> {
        cpu::host::full(len, dtype, value)
    }
    fn cast(&self, x: View<'_>, to: DType) -> Result<Storage> {
        cpu::host::cast(x, to)
    }
    fn binary(&self, op: BinaryOp, lhs: View<'_>, rhs: View<'_>) -> Result<Storage> {
        cpu::elementwise::binary(op, lhs, rhs)
    }
    fn binary_scalar(&self, op: BinaryOp, x: View<'_>, scalar: f64) -> Result<Storage> {
        cpu::elementwise::binary_scalar(op, x, scalar)
    }
    fn unary(&self, op: UnaryOp, x: View<'_>) -> Result<Storage> {
        cpu::elementwise::unary(op, x)
    }
    fn compare(&self, op: CmpOp, lhs: View<'_>, rhs: View<'_>) -> Result<Storage> {
        cpu::elementwise::compare(op, lhs, rhs)
    }
    fn where_cond(&self, cond: View<'_>, on_true: View<'_>, on_false: View<'_>) -> Result<Storage> {
        cpu::elementwise::where_cond(cond, on_true, on_false)
    }
    fn masked_fill(&self, x: View<'_>, mask: View<'_>, value: f64) -> Result<Storage> {
        cpu::elementwise::masked_fill(x, mask, value)
    }
    fn reduce(&self, op: ReduceOp, x: View<'_>, axis: usize) -> Result<Storage> {
        cpu::reduce::reduce(op, x, axis)
    }
    fn arg_reduce(&self, op: ArgReduceOp, x: View<'_>, axis: usize) -> Result<Storage> {
        cpu::reduce::arg_reduce(op, x, axis)
    }
    fn matmul(&self, lhs: View<'_>, rhs: View<'_>) -> Result<Storage> {
        cpu::matmul::matmul(lhs, rhs)
    }
    fn index_select(&self, x: View<'_>, axis: usize, indices: View<'_>) -> Result<Storage> {
        cpu::index::index_select(x, axis, indices)
    }
    fn index_add(
        &self,
        x: View<'_>,
        axis: usize,
        indices: View<'_>,
        src: View<'_>,
    ) -> Result<Storage> {
        cpu::index::index_add(x, axis, indices, src)
    }
    fn gather(&self, x: View<'_>, axis: usize, indices: View<'_>) -> Result<Storage> {
        cpu::index::gather(x, axis, indices)
    }
    fn scatter_add(
        &self,
        x: View<'_>,
        axis: usize,
        indices: View<'_>,
        src: View<'_>,
    ) -> Result<Storage> {
        cpu::index::scatter_add(x, axis, indices, src)
    }
    fn conv(&self, op: ConvOp, inputs: &[View<'_>], params: &Conv2dParams) -> Result<Storage> {
        cpu::conv::conv(op, inputs, params)
    }
    fn fused(&self, op: FusedOp, inputs: &[View<'_>], scalars: &[f64]) -> Result<Vec<Storage>> {
        cpu::fused::fused(op, inputs, scalars)
    }
}

/// The single dispatch point: every tensor op routes
/// through here to obtain the backend for its device, then calls one coarse
/// [`BackendOps`] entry point. Backends are zero-sized and long-lived, so a
/// `&'static` reference is handed back.
pub(crate) mod dispatch {
    use super::{BackendOps, CpuBackend};
    use crate::device::Device;

    static CPU: CpuBackend = CpuBackend;

    /// Return the backend implementation for `device`.
    ///
    /// Both arms are total: the CPU backend is always available, and the
    /// Metal arm is compiled in only when the `metal` feature is enabled on
    /// macOS, which is also the only configuration in which
    /// [`Device::Metal`] can be constructed.
    pub(crate) fn backend(device: Device) -> &'static dyn BackendOps {
        match device {
            Device::Cpu => &CPU,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Device::Metal(ordinal) => super::metal::backend(ordinal),
        }
    }
}
