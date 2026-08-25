//! The [`Tensor`] type and its op surface: one concrete tensor, one spelling
//! per operation.
//!
//! This file holds [`Tensor`]/[`Inner`], the accessors, the crate-internal
//! plumbing (`from_parts`, `view`), the autograd delegation
//! ([`traced`](Tensor::traced)/[`backward`](Tensor::backward)), the
//! constructors and host-transfer/movement bodies, and the crate-internal
//! broadcast-gradient reducer `Tensor::sum_to`, which every
//! binary/broadcast/reduction backward funnels through. The op families add
//! their methods in `tensor/ops/*` behind the
//! [`record`](crate::autograd) seam.

mod fmt;
pub(crate) mod ops;

use crate::autograd::{self, Grads};
use crate::backend::{ReduceOp, View, dispatch};
use crate::device::Device;
use crate::dtype::{DType, Element, HostConv};
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::rng::Rng;
use crate::shape::Shape;
use crate::storage::{CpuStorage, Storage};
use std::sync::Arc;

/// The reference-counted body of a [`Tensor`]: a storage buffer, a strided
/// [`Layout`] over it, and an optional autograd [`Node`](crate::autograd).
///
/// Immutable after construction at the tensor API: a pending storage may
/// fill its private write-once computation cache, but no logical value or
/// backing buffer is mutated. This is what makes [`Tensor::clone`] an `Arc`
/// bump and lets views and detached captures share storage safely.
pub(crate) struct Inner {
    storage: Storage,
    layout: Layout,
    node: Option<Arc<crate::autograd::Node>>,
}

/// The one and only tensor type: an immutable value with **zero generic
/// parameters**. `Clone` is an `Arc` bump; `Send + Sync` because its `Inner`
/// body has no observable mutation and its parts are `Send + Sync`.
///
/// Shapes, dtype, and device are runtime data. Rank assumptions are made
/// explicit and loud through [`dims2`](Tensor::dims2)/[`dims3`](Tensor::dims3)/
/// [`dims4`](Tensor::dims4).
#[derive(Clone)]
pub struct Tensor(Arc<Inner>);

impl Tensor {
    // ---- crate-internal construction / views -----------------------------

    /// Build an **untraced** tensor from a storage buffer and a layout over
    /// it. The entry point every kernel result flows through before autograd
    /// wrapping.
    pub(crate) fn from_parts(storage: Storage, layout: Layout) -> Tensor {
        Tensor(Arc::new(Inner {
            storage,
            layout,
            node: None,
        }))
    }

    /// Build a tensor from parts **with** an autograd node (used by the
    /// [`record`](crate::autograd::record) engine).
    pub(crate) fn from_parts_traced(
        storage: Storage,
        layout: Layout,
        node: Arc<crate::autograd::Node>,
    ) -> Tensor {
        Tensor(Arc::new(Inner {
            storage,
            layout,
            node: Some(node),
        }))
    }

    /// A borrowed view over storage that is known to be ready for a backend.
    #[allow(dead_code)]
    pub(crate) fn view(&self) -> View<'_> {
        View::new(&self.0.storage, &self.0.layout)
    }

    /// Resolve a pending value before constructing a backend view.
    pub(crate) fn ready_view(&self) -> Result<View<'_>> {
        let storage = self.0.storage.ready()?;
        Ok(View::new(storage, &self.0.layout))
    }

    /// The storage buffer (crate-internal).
    pub(crate) fn storage(&self) -> &Storage {
        &self.0.storage
    }

    /// The layout (crate-internal).
    pub(crate) fn layout(&self) -> &Layout {
        &self.0.layout
    }

    /// The autograd node, if this tensor is traced (crate-internal).
    pub(crate) fn node(&self) -> Option<&Arc<crate::autograd::Node>> {
        self.0.node.as_ref()
    }

    /// Whether two handles name the same storage/layout/autograd body. Used by
    /// the typed wrappers' tests to assert that erasing/refining a marker is a
    /// zero-copy handle reuse.
    #[cfg(all(test, feature = "typed"))]
    pub(crate) fn ptr_eq(&self, other: &Tensor) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    // ---- accessors -------------------------------------------------------

    /// The logical shape.
    pub fn shape(&self) -> &Shape {
        self.0.layout.shape()
    }

    /// Dimension sizes, outermost first.
    pub fn dims(&self) -> &[usize] {
        self.0.layout.dims()
    }

    /// Number of axes (0 for a scalar).
    pub fn rank(&self) -> usize {
        self.0.layout.rank()
    }

    /// Total number of elements.
    pub fn num_elements(&self) -> usize {
        self.0.layout.num_elements()
    }

    /// The element dtype.
    pub fn dtype(&self) -> DType {
        self.0.storage.dtype()
    }

    /// The device the tensor lives on.
    pub fn device(&self) -> Device {
        self.0.storage.device()
    }

    /// Whether the recorded layout is canonical row-major at offset 0
    /// (crate-internal).
    ///
    /// Deliberately not public: an infallible layout predicate would let
    /// callers branch on how a tensor is stored, which commits every tensor to
    /// a decided physical layout before execution and forecloses a fusion pass
    /// picking one (NCHW↔NHWC, a transposed operand folded into a GEMM). What
    /// this reports is the logical layout as recorded, not a promise about
    /// physical storage — so the only place it surfaces outside the crate is
    /// the `contiguous` field of [`Debug`](std::fmt::Debug), as a debugging
    /// aid rather than a contract.
    pub(crate) fn is_contiguous(&self) -> bool {
        self.0.layout.is_contiguous()
    }

    /// Assert rank 1, returning the single dimension.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RankMismatch`] if `self` is not rank 1.
    pub fn dims1(&self) -> Result<usize> {
        match self.dims() {
            &[a] => Ok(a),
            d => Err(Error::rank_mismatch("dims1", 1, d.len())),
        }
    }

    /// Assert rank 2, returning `(d0, d1)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RankMismatch`] if `self` is not rank 2.
    pub fn dims2(&self) -> Result<(usize, usize)> {
        match self.dims() {
            &[a, b] => Ok((a, b)),
            d => Err(Error::rank_mismatch("dims2", 2, d.len())),
        }
    }

    /// Assert rank 3, returning `(d0, d1, d2)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RankMismatch`] if `self` is not rank 3.
    pub fn dims3(&self) -> Result<(usize, usize, usize)> {
        match self.dims() {
            &[a, b, c] => Ok((a, b, c)),
            d => Err(Error::rank_mismatch("dims3", 3, d.len())),
        }
    }

    /// Assert rank 4, returning `(d0, d1, d2, d3)`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::RankMismatch`] if `self` is not rank 4.
    pub fn dims4(&self) -> Result<(usize, usize, usize, usize)> {
        match self.dims() {
            &[a, b, c, d] => Ok((a, b, c, d)),
            d => Err(Error::rank_mismatch("dims4", 4, d.len())),
        }
    }

    // ---- autograd (delegated to the engine) ------------------------------

    /// Turn this tensor into a traced leaf for grad-wrt-input (exploration
    /// §4.3). The **returned** binding must be used in both the computation
    /// and the [`Grads::wrt_input`](crate::Grads::wrt_input) lookup.
    /// Errors with [`Error::InvalidArg`](crate::Error::InvalidArg)
    /// (`op: "traced"`) if `self` already carries a graph — double-tracing is
    /// a bug. (`NotTraced` is reserved for the opposite condition: a
    /// `backward()`/lookup on a tensor with *no* graph.)
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArg`](crate::Error::InvalidArg) if `self`
    /// already carries a graph.
    pub fn traced(&self) -> Result<Tensor> {
        autograd::traced(self)
    }

    /// Reverse-mode autodiff from this tensor, returning a fresh, linear
    /// [`Grads`]. [`Error::NotTraced`](crate::Error::NotTraced) if this
    /// tensor carries no autograd graph.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotTraced`](crate::Error::NotTraced) if `self`
    /// carries no autograd graph.
    pub fn backward(&self) -> Result<Grads> {
        autograd::backward(self)
    }

    /// A detached copy: shares this tensor's storage and layout but carries
    /// no autograd node. Differentiable ops treat it as a constant.
    ///
    /// The returned tensor is a fresh `Inner` with `node: None`; its storage
    /// and layout are cloned (both are `Arc`/cheap, so this is a refcount
    /// bump, not an element copy). It is the public spelling of the
    /// crate-internal `detach_shallow`; the two are identical because `Inner`
    /// already keeps storage and layout out of the autograd node.
    #[must_use]
    pub fn detach(&self) -> Tensor {
        self.detach_shallow()
    }

    /// Internal shallow detach used by the detached-output capture rule:
    /// a fresh `Inner` sharing storage, `node: None`,
    /// built before the traced output is assembled. A backward closure that
    /// needs the op's output value (sigmoid/tanh/softmax) captures *this* form
    /// so the closure never holds an `Arc` back to its own output node.
    pub(crate) fn detach_shallow(&self) -> Tensor {
        Tensor::from_parts(self.0.storage.clone(), self.0.layout.clone())
    }

    // ---- constructors ----------------------------------------------------

    /// A tensor of zeros.
    ///
    /// # Errors
    ///
    /// As [`full`](Self::full).
    pub fn zeros(shape: impl Into<Shape>, dtype: DType, device: &Device) -> Result<Tensor> {
        Tensor::full(shape, 0.0, dtype, device)
    }

    /// A tensor of ones.
    ///
    /// # Errors
    ///
    /// As [`full`](Self::full).
    pub fn ones(shape: impl Into<Shape>, dtype: DType, device: &Device) -> Result<Tensor> {
        Tensor::full(shape, 1.0, dtype, device)
    }

    /// A tensor filled with `value` (narrowed to `dtype`).
    ///
    /// The single overflow-validation point for constructed tensors is the
    /// contiguous layout, which errors with
    /// [`Error::InvalidArg`](crate::Error::InvalidArg) (`op: "layout"`) if the
    /// element count overflows `usize`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArg`](crate::Error::InvalidArg) if `shape`'s
    /// element count overflows `usize`.
    pub fn full(
        shape: impl Into<Shape>,
        value: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let layout = Layout::contiguous(shape)?;
        let storage = if crate::lazy::enabled() {
            crate::lazy::constant("full", value, dtype, *device, layout.dims().to_vec())?
        } else {
            dispatch::backend(*device).full(layout.num_elements(), dtype, value)?
        };
        Ok(Tensor::from_parts(storage, layout))
    }

    /// A tensor of zeros shaped, typed and placed like `self`.
    ///
    /// # Errors
    ///
    /// As [`full`](Self::full).
    pub fn zeros_like(&self) -> Result<Tensor> {
        self.full_like(0.0)
    }

    /// A tensor of ones shaped, typed and placed like `self`.
    ///
    /// # Errors
    ///
    /// As [`full`](Self::full).
    pub fn ones_like(&self) -> Result<Tensor> {
        self.full_like(1.0)
    }

    /// A tensor filled with `value`, inheriting `self`'s shape, dtype **and**
    /// device — the three things a companion tensor otherwise respells at
    /// every call site.
    ///
    /// Like [`full`](Self::full) this is an untraced constructor: the result
    /// is a fresh constant with `node: None`, so `self`'s autograd graph is
    /// never propagated into it. Calling it on a traced tensor is how the
    /// crate's own backwards spell "no gradient here"; the result is
    /// deliberately not differentiable with respect to `self`.
    ///
    /// # Errors
    ///
    /// As [`full`](Self::full).
    pub fn full_like(&self, value: f64) -> Result<Tensor> {
        Tensor::full(self.dims(), value, self.dtype(), &self.device())
    }

    /// A tensor from a host vector; the dtype is `T`'s. `data.len()` must
    /// equal the shape's element count
    /// ([`Error::ShapeMismatch`](crate::Error::ShapeMismatch)).
    ///
    /// # Errors
    ///
    /// Returns [`Error::ShapeMismatch`](crate::Error::ShapeMismatch) if
    /// `data.len()` does not equal `shape`'s element count, or
    /// [`Error::InvalidArg`](crate::Error::InvalidArg) if `shape`'s element
    /// count overflows `usize`.
    ///
    /// # Examples
    ///
    /// ```
    /// use rstorch::{Device, Tensor};
    ///
    /// let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2], &Device::Cpu)?;
    /// assert_eq!(x.dims(), &[2, 2]);
    /// assert_eq!(x.to_vec::<f32>()?, vec![1.0, 2.0, 3.0, 4.0]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn from_vec<T: Element>(
        data: Vec<T>,
        shape: impl Into<Shape>,
        device: &Device,
    ) -> Result<Tensor> {
        let layout = Layout::contiguous(shape)?;
        let expected = layout.num_elements();
        if data.len() != expected {
            return Err(Error::ShapeMismatch {
                op: "from_vec",
                lhs: Shape::from(vec![data.len()]),
                rhs: layout.shape().clone(),
            });
        }
        let host = <T as HostConv>::into_cpu_storage(data);
        let storage = dispatch::backend(*device).transfer_in(host)?;
        Ok(Tensor::from_parts(storage, layout))
    }

    /// Uniform samples in `[0, 1)` (float `dtype` only). Consumes randomness
    /// from `rng`.
    ///
    /// A non-float `dtype` is [`Error::InvalidArg`](crate::Error::InvalidArg):
    /// only floating-point tensors participate in autograd, and a "random
    /// integer tensor" has no single obvious meaning here.
    ///
    /// A given seed reproduces a given tensor for a fixed shape, dtype,
    /// backend, feature set and build. Reproducibility is scoped to that
    /// tuple: values are not promised to match across backends or across
    /// versions, because whether samples are drawn on the host and uploaded or
    /// generated on the device is an implementation detail.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArg`](crate::Error::InvalidArg) if `dtype` is
    /// not a float dtype, or if `shape`'s element count overflows `usize`.
    pub fn rand(
        shape: impl Into<Shape>,
        dtype: DType,
        device: &Device,
        rng: &mut Rng,
    ) -> Result<Tensor> {
        Tensor::sampled("rand", shape, dtype, device, rng, |r| r.uniform(0.0, 1.0))
    }

    /// Standard-normal samples (float `dtype` only). Consumes randomness from
    /// `rng`.
    ///
    /// A non-float `dtype` is [`Error::InvalidArg`](crate::Error::InvalidArg)
    /// (see [`rand`](Tensor::rand)).
    ///
    /// # Errors
    ///
    /// As [`rand`](Self::rand).
    pub fn randn(
        shape: impl Into<Shape>,
        dtype: DType,
        device: &Device,
        rng: &mut Rng,
    ) -> Result<Tensor> {
        Tensor::sampled("randn", shape, dtype, device, rng, |r| r.normal(0.0, 1.0))
    }

    /// Shared body of [`rand`](Tensor::rand)/[`randn`](Tensor::randn): draw
    /// `num_elements` host samples as `f64` (each via `draw`), narrow them into
    /// the float `dtype`'s host buffer, and upload. Samples are drawn in
    /// row-major order, so a given `rng` state reproduces a given tensor for a
    /// fixed (seed, shape, dtype, backend, build) — that tuple, and no wider.
    /// Reproducibility across backends is deliberately **not** promised: it
    /// would freeze host-draw-then-upload as the only legal strategy and
    /// foreclose a counter-based on-device generator, which is what makes
    /// sampling fusible into a surrounding elementwise chain.
    fn sampled(
        op: &'static str,
        shape: impl Into<Shape>,
        dtype: DType,
        device: &Device,
        rng: &mut Rng,
        mut draw: impl FnMut(&mut Rng) -> f64,
    ) -> Result<Tensor> {
        if !dtype.is_float() {
            return Err(Error::InvalidArg {
                op,
                msg: format!("{op} requires a float dtype, got {dtype}"),
            });
        }
        let layout = Layout::contiguous(shape)?;
        let len = layout.num_elements();
        let host = match dtype {
            DType::F16 => CpuStorage::F16(Arc::new(
                (0..len).map(|_| half::f16::from_f64(draw(rng))).collect(),
            )),
            DType::BF16 => CpuStorage::BF16(Arc::new(
                (0..len).map(|_| half::bf16::from_f64(draw(rng))).collect(),
            )),
            DType::F32 => CpuStorage::F32(Arc::new((0..len).map(|_| draw(rng) as f32).collect())),
            DType::F64 => CpuStorage::F64(Arc::new((0..len).map(|_| draw(rng)).collect())),
            // Guarded by the `is_float` check above; the two integer/bool
            // dtypes never reach here.
            DType::I64 | DType::Bool => unreachable!("non-float dtype passed is_float check"),
        };
        let storage = dispatch::backend(*device).transfer_in(host)?;
        Ok(Tensor::from_parts(storage, layout))
    }

    /// A 1-D range `[start, end)` stepped by `step` (the sole owner of
    /// `arange`; the indexing set builds on it).
    ///
    /// `step` must be non-zero and every bound finite
    /// ([`Error::InvalidArg`](crate::Error::InvalidArg) otherwise). The value
    /// count is `ceil((end - start) / step)`, or zero when `step` points away
    /// from `end` (an empty range, matching `PyTorch` rather than erroring).
    /// Values are generated as `f64` (`start + i * step`) and narrowed to
    /// `dtype`; a [`Bool`](crate::DType::Bool) `dtype` is
    /// [`Error::InvalidArg`](crate::Error::InvalidArg) since a stepped range is
    /// not a boolean sequence, and so is a range whose element count would not
    /// fit in `usize`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArg`](crate::Error::InvalidArg) if any of
    /// `start`/`end`/`step` is not finite, if `step` is zero, if `dtype` is
    /// [`Bool`](crate::DType::Bool), or if the resulting element count would
    /// not fit in `usize`.
    pub fn arange(
        start: f64,
        end: f64,
        step: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        // Non-finite bounds first: a NaN `step` is not caught by `== 0.0`, and
        // an infinite span would silently saturate the `as usize` cast below
        // into a nonsense element count.
        if !start.is_finite() || !end.is_finite() || !step.is_finite() {
            return Err(Error::InvalidArg {
                op: "arange",
                msg: format!("start, end and step must be finite (got {start}, {end}, {step})"),
            });
        }
        if step == 0.0 {
            return Err(Error::InvalidArg {
                op: "arange",
                msg: "step must be non-zero".to_string(),
            });
        }
        if matches!(dtype, DType::Bool) {
            return Err(Error::InvalidArg {
                op: "arange",
                msg: "arange does not support the bool dtype".to_string(),
            });
        }
        // Count = ceil((end - start) / step), clamped at 0 for an empty range
        // (wrong-direction step yields a negative quotient -> no elements,
        // matching PyTorch).
        let span = (end - start) / step;
        let count = if span > 0.0 {
            // `span` can still be infinite here (a tiny `step` over a huge
            // range); `>=` against `usize::MAX` catches that as well.
            let ceil = span.ceil();
            if ceil >= usize::MAX as f64 {
                return Err(Error::InvalidArg {
                    op: "arange",
                    msg: format!("range {start}..{end} by {step} has too many elements"),
                });
            }
            ceil as usize
        } else {
            0
        };
        let values: Vec<f64> = (0..count).map(|i| start + (i as f64) * step).collect();
        let host = match dtype {
            DType::F16 => CpuStorage::F16(Arc::new(
                values.iter().map(|&v| half::f16::from_f64(v)).collect(),
            )),
            DType::BF16 => CpuStorage::BF16(Arc::new(
                values.iter().map(|&v| half::bf16::from_f64(v)).collect(),
            )),
            DType::F32 => CpuStorage::F32(Arc::new(values.iter().map(|&v| v as f32).collect())),
            DType::F64 => CpuStorage::F64(Arc::new(values)),
            DType::I64 => CpuStorage::I64(Arc::new(values.iter().map(|&v| v as i64).collect())),
            DType::Bool => unreachable!("bool rejected above"),
        };
        let layout = Layout::contiguous([count])?;
        let storage = dispatch::backend(*device).transfer_in(host)?;
        Ok(Tensor::from_parts(storage, layout))
    }

    // ---- host transfer / movement ----------------------------------------

    /// Copy the tensor's elements to a host `Vec` in row-major order (a host
    /// boundary: synchronizes the backend). Dtype must be `T`, else
    /// [`Error::DTypeMismatch`](crate::Error::DTypeMismatch).
    ///
    /// # Errors
    ///
    /// Returns [`Error::DTypeMismatch`](crate::Error::DTypeMismatch) if
    /// `self`'s dtype is not `T`.
    pub fn to_vec<T: Element>(&self) -> Result<Vec<T>> {
        let host = dispatch::backend(self.device()).transfer_out(self.ready_view()?)?;
        <T as HostConv>::try_from_cpu_storage(&host, "to_vec")
    }

    /// Read a single-element tensor as a scalar of type `T`. The tensor must
    /// have exactly one element ([`Error::InvalidArg`](crate::Error::InvalidArg))
    /// and dtype `T` ([`Error::DTypeMismatch`](crate::Error::DTypeMismatch)).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArg`](crate::Error::InvalidArg) if `self` does
    /// not have exactly one element, or
    /// [`Error::DTypeMismatch`](crate::Error::DTypeMismatch) if its dtype is
    /// not `T`.
    pub fn to_scalar<T: Element>(&self) -> Result<T> {
        if self.num_elements() != 1 {
            return Err(Error::InvalidArg {
                op: "to_scalar",
                msg: format!(
                    "to_scalar requires a single-element tensor, got shape {}",
                    self.shape()
                ),
            });
        }
        Ok(self.to_vec::<T>()?[0])
    }

    /// Read a single-element tensor as `f64` regardless of dtype (the
    /// dtype-agnostic convenience behind loss/accuracy reads). The tensor must
    /// have exactly one element ([`Error::InvalidArg`](crate::Error::InvalidArg)).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArg`](crate::Error::InvalidArg) if `self` does
    /// not have exactly one element.
    pub fn item(&self) -> Result<f64> {
        if self.num_elements() != 1 {
            return Err(Error::InvalidArg {
                op: "item",
                msg: format!(
                    "item requires a single-element tensor, got shape {}",
                    self.shape()
                ),
            });
        }
        let host = dispatch::backend(self.device()).transfer_out(self.ready_view()?)?;
        let v = match &host {
            CpuStorage::F16(a) => a[0].to_f64(),
            CpuStorage::BF16(a) => a[0].to_f64(),
            CpuStorage::F32(a) => a[0] as f64,
            CpuStorage::F64(a) => a[0],
            CpuStorage::I64(a) => a[0] as f64,
            CpuStorage::Bool(a) => {
                if a[0] {
                    1.0
                } else {
                    0.0
                }
            }
        };
        Ok(v)
    }

    /// Force this tensor's value to be materialized, then flush its device.
    ///
    /// Deferred element-wise expressions are realized up to this tensor before
    /// the backend-wide synchronization. With eager execution the first step
    /// is already complete, so this retains the existing device flush.
    ///
    /// The backend synchronization remains **device-wide**: it waits for every
    /// operation already submitted on this device, not only work this tensor
    /// depends on. CPU kernels are synchronous and therefore have no queued
    /// work to wait for.
    ///
    /// Nothing is returned — reading values is [`to_vec`](Tensor::to_vec),
    /// [`to_scalar`](Tensor::to_scalar) or [`item`](Tensor::item), each of
    /// which is a host boundary that synchronizes on its own.
    ///
    /// # Errors
    ///
    /// Whatever the flush turns up:
    /// [`Error::Backend`](crate::Error::Backend) if the drained work failed,
    /// or, on Metal and CUDA,
    /// [`Error::IndexOutOfBounds`](crate::Error::IndexOutOfBounds) for an
    /// earlier indexing op whose bounds check was encoded but not yet read.
    /// Because the wait is device-wide, an error here need not come from this
    /// tensor's own dependencies.
    ///
    /// WGPU keeps a pending bounds verdict on the storage that carries it
    /// rather than on the device, so this flush cannot report one; it surfaces
    /// at the host read of the tensor that carries it instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use rstorch::{DType, Device, Tensor};
    ///
    /// let device = Device::best_available();
    /// let x = Tensor::ones([128, 128], DType::F32, &device)?;
    /// let y = x.matmul(&x)?;
    /// y.realize()?; // the matmul has run, and nothing was copied to the host
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn realize(&self) -> Result<()> {
        self.storage().ready()?;
        self.device().synchronize()
    }

    /// Move to `device`. A differentiable op: the backward pass moves the
    /// cotangent back to the source device. Equal devices are returned without
    /// a copy; different devices use the explicit host transfer path.
    ///
    /// # Errors
    ///
    pub fn to_device(&self, device: &Device) -> Result<Tensor> {
        if self.device() == *device {
            return Ok(self.clone());
        }
        let host = dispatch::backend(self.device()).transfer_out(self.ready_view()?)?;
        let storage = dispatch::backend(*device).transfer_in(host)?;
        let layout = Layout::contiguous(self.shape().clone())?;
        let out = Tensor::from_parts(storage, layout);
        let src = self.device();
        Ok(autograd::record(
            "to_device",
            out,
            &[self],
            Box::new(move |g| Ok(vec![Some(g.to_device(&src)?)])),
        ))
    }

    /// Cast to `dtype` via the backend Cast kernel. A differentiable op: the
    /// backward pass casts the cotangent back to the input dtype (meaningful
    /// for float↔float; the engine drops gradients for integer/bool inputs).
    ///
    /// # Errors
    ///
    /// dtype cannot be cast to `dtype` on the current backend.
    pub fn to_dtype(&self, dtype: DType) -> Result<Tensor> {
        if self.dtype() == dtype {
            return Ok(self.clone());
        }
        let storage = if crate::lazy::enabled() {
            crate::lazy::cast(
                "to_dtype",
                self.storage(),
                self.layout(),
                dtype,
                self.device(),
            )?
        } else {
            dispatch::backend(self.device()).cast(self.ready_view()?, dtype)?
        };
        let layout = Layout::contiguous(self.shape().clone())?;
        let out = Tensor::from_parts(storage, layout);
        let src = self.dtype();
        Ok(autograd::record(
            "to_dtype",
            out,
            &[self],
            Box::new(move |g| Ok(vec![Some(g.to_dtype(src)?)])),
        ))
    }

    /// A copy whose recorded layout is canonical row-major, or `self` unchanged
    /// when its layout already is. Value-identity, so it is transparent to
    /// autograd (the cotangent flows straight through).
    ///
    /// That is a statement about the logical layout as recorded, not a promise
    /// about physical storage: a backend stays free to hold the elements
    /// however it likes, so long as they read back in row-major logical order.
    /// Call this when an op wants a dense operand, not to reason about bytes.
    ///
    /// # Errors
    /// Propagates any backend copy error from `self`'s device.
    pub fn contiguous(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        let storage = if crate::lazy::enabled() {
            crate::lazy::copy(
                "contiguous",
                self.storage(),
                self.layout(),
                self.dtype(),
                self.device(),
            )?
        } else {
            dispatch::backend(self.device()).copy_strided(self.ready_view()?)?
        };
        let layout = Layout::contiguous(self.shape().clone())?;
        let out = Tensor::from_parts(storage, layout);
        Ok(autograd::record(
            "contiguous",
            out,
            &[self],
            Box::new(move |g| Ok(vec![Some(g.clone())])),
        ))
    }

    // ---- broadcast-gradient reduction ------------------------------------

    /// Sum this tensor — the *broadcast* side of some op — back down to
    /// `target_dims`, the shape of the operand it was broadcast from. Every
    /// binary, broadcasting, and reduction backward in the op layer ends in a
    /// call to this.
    ///
    /// # Contract
    ///
    /// `self.dims()` must be a shape that `target_dims` broadcasts *to* under
    /// the NumPy/PyTorch rules: align the two to the right, leaving
    /// `pad = self.rank() - target_dims.len()` leading source axes unmatched,
    /// and require of every aligned pair `(t, s)` that `t == s` or `t == 1`.
    /// The reduction is then exactly the transpose of that broadcast:
    ///
    /// - each of the `pad` unmatched **leading** axes is summed away and
    ///   dropped (it was created by right-alignment), and
    /// - each aligned axis with `t == 1 < s` is summed away and re-inserted
    ///   at size 1 (it was expanded from a size-1 axis).
    ///
    /// Aligned axes with `t == s` are left untouched, including `t == s == 1`.
    ///
    /// The result has **exactly** `target_dims` — not merely a
    /// broadcast-compatible shape — is contiguous, and keeps `self`'s dtype
    /// and device. When `self.dims()` already equals `target_dims` this is the
    /// identity and `self` is returned unchanged (an `Arc` bump, no kernel
    /// call), which is the common case for the non-broadcast operand of a
    /// binary op.
    ///
    /// Summation goes through the backend `reduce` entry point with
    /// `ReduceOp::Sum`, one axis per call, highest axis index first so the
    /// indices of the axes still to be reduced stay valid as each reduction
    /// drops one. F16/BF16 inputs are widened before the first reduction and
    /// remain F32 across every reduced axis, then narrow once at the final
    /// requested shape.
    ///
    /// # Errors
    ///
    /// - [`Error::ShapeMismatch`](crate::Error::ShapeMismatch) with
    ///   `op: "sum_to"`, `lhs` = `self.shape()` and `rhs` = `target_dims`,
    ///   when `target_dims` has a higher rank than `self` or an aligned axis
    ///   pair is neither equal nor `1` on the target side. That is the "these
    ///   two shapes never broadcast" case, i.e. a bug in the calling backward
    ///   rather than bad user input.
    /// - Whatever the backend reduce reports — notably
    ///   [`Error::Unsupported`](crate::Error::Unsupported) on a
    ///   [`Bool`](crate::DType::Bool) tensor, which has no sum.
    ///
    /// # Not a differentiable op
    ///
    /// `sum_to` is a plain value-level reduction: it deliberately does **not**
    /// go through the `record` seam, because it exists to be called *inside*
    /// backward closures, on an already-computed cotangent. Do not use it on a
    /// forward path — gradients would silently stop there. (Forward code wants
    /// `sum`/`sum_keepdim`.)
    pub(crate) fn sum_to(&self, target_dims: &[usize]) -> Result<Tensor> {
        let src = self.dims();
        let mismatch = || Error::ShapeMismatch {
            op: "sum_to",
            lhs: self.shape().clone(),
            rhs: Shape::from(target_dims.to_vec()),
        };
        if target_dims.len() > src.len() {
            return Err(mismatch());
        }
        let pad = src.len() - target_dims.len();

        // Which source axes vanish into the sum: every unmatched leading axis,
        // plus every aligned axis the target holds at 1 while the source
        // expanded it. Ascending by construction.
        let mut reduce_axes: Vec<usize> = (0..pad).collect();
        for (i, &t) in target_dims.iter().enumerate() {
            let s = src[pad + i];
            if t == s {
                continue;
            }
            if t != 1 {
                return Err(mismatch());
            }
            reduce_axes.push(pad + i);
        }

        if reduce_axes.is_empty() {
            // Nothing was broadcast: the shapes are already identical.
            debug_assert_eq!(src, target_dims);
            return Ok(self.clone());
        }

        let dtype = self.dtype();
        let accumulation_dtype = dtype.accumulation_dtype();
        let mut cur = self.clone();
        if accumulation_dtype != dtype {
            cur = cur.to_dtype(accumulation_dtype)?;
        }
        // Highest axis first: dropping axis `k` leaves every axis below `k`
        // at its original index.
        let backend = dispatch::backend(self.device());
        for &axis in reduce_axes.iter().rev() {
            let storage = backend.reduce(ReduceOp::Sum, cur.ready_view()?, axis)?;
            let dims: Vec<usize> = cur
                .dims()
                .iter()
                .enumerate()
                .filter(|&(a, _)| a != axis)
                .map(|(_, &d)| d)
                .collect();
            cur = Tensor::from_parts(storage, Layout::contiguous(dims)?);
        }
        if cur.dtype() != dtype {
            cur = cur.to_dtype(dtype)?;
        }

        // What survives is `target_dims` with its summed-away size-1 axes
        // deleted, so the element counts agree and re-viewing the (contiguous)
        // reduction output as `target_dims` simply puts them back.
        let layout = Layout::contiguous(Shape::from(target_dims.to_vec()))?;
        debug_assert_eq!(layout.num_elements(), cur.num_elements());
        Ok(Tensor::from_parts(cur.storage().clone(), layout))
    }
}

// `Tensor` must be `Send + Sync`. This fails to compile if
// any field of `Inner` ever loses those bounds.
const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    fn check() {
        assert_send_sync::<Tensor>();
    }
    let _ = check;
};

#[cfg(test)]
mod tests;
