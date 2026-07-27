//! The [`Tensor`] type and its op surface: one concrete tensor, one spelling
//! per operation (exploration §4.1–4.2).
//!
//! **Contract file** (T01²/T20). T01 defines [`Tensor`]/[`Inner`], the
//! accessors, the crate-internal plumbing (`from_parts`, `view`), and the
//! autograd delegation ([`traced`](Tensor::traced)/[`backward`](Tensor::backward)).
//! **T20** fills the constructor and host-transfer/movement `todo!()` bodies
//! and adds the crate-internal broadcast-gradient reducer `Tensor::sum_to`,
//! which every binary/broadcast/reduction backward funnels through; op
//! families (T21–T27) add methods in `tensor/ops/*` behind the frozen
//! [`record`](crate::autograd) seam. No signature here changes after T01.

// The crate-internal plumbing (`from_parts*`, `view`, `storage`, `layout`,
// `node`, `detach_shallow`) is consumed by the wave-2/3 kernel, op, and
// autograd tasks; the integrator removes this allow at v3-m1. Public
// constructors/accessors are unaffected (they are never dead).
#![allow(dead_code)]

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
/// Immutable after construction — this is what makes [`Tensor::clone`] an
/// `Arc` bump and lets views and detached captures share storage safely.
pub(crate) struct Inner {
    storage: Storage,
    layout: Layout,
    node: Option<Arc<crate::autograd::Node>>,
}

/// The one and only tensor type: an immutable value with **zero generic
/// parameters** (exploration §4.1). `Clone` is an `Arc` bump; `Send + Sync`
/// because its `Inner` body is immutable and its parts are `Send + Sync`.
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
    /// [`record`](crate::autograd::record) engine, T30).
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

    /// A borrowed, stride-aware [`View`] for backend dispatch.
    pub(crate) fn view(&self) -> View<'_> {
        View::new(&self.0.storage, &self.0.layout)
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

    /// Whether the layout is contiguous (row-major, offset 0).
    pub fn is_contiguous(&self) -> bool {
        self.0.layout.is_contiguous()
    }

    /// Assert rank 1, returning the single dimension.
    pub fn dims1(&self) -> Result<usize> {
        match self.dims() {
            &[a] => Ok(a),
            d => Err(Self::rank_err("dims1", 1, d.len())),
        }
    }

    /// Assert rank 2, returning `(d0, d1)`.
    pub fn dims2(&self) -> Result<(usize, usize)> {
        match self.dims() {
            &[a, b] => Ok((a, b)),
            d => Err(Self::rank_err("dims2", 2, d.len())),
        }
    }

    /// Assert rank 3, returning `(d0, d1, d2)`.
    pub fn dims3(&self) -> Result<(usize, usize, usize)> {
        match self.dims() {
            &[a, b, c] => Ok((a, b, c)),
            d => Err(Self::rank_err("dims3", 3, d.len())),
        }
    }

    /// Assert rank 4, returning `(d0, d1, d2, d3)`.
    pub fn dims4(&self) -> Result<(usize, usize, usize, usize)> {
        match self.dims() {
            &[a, b, c, d] => Ok((a, b, c, d)),
            d => Err(Self::rank_err("dims4", 4, d.len())),
        }
    }

    fn rank_err(op: &'static str, expected: usize, got: usize) -> Error {
        Error::RankMismatch { op, expected, got }
    }

    // ---- autograd (delegated to the engine; T30 fills the bodies) --------

    /// Turn this tensor into a traced leaf for grad-wrt-input (exploration
    /// §4.3). The **returned** binding must be used in both the computation
    /// and the [`Grads::wrt_input`](crate::Grads::wrt_input) lookup.
    /// Errors with [`Error::InvalidArg`](crate::Error::InvalidArg)
    /// (`op: "traced"`) if `self` already carries a graph — double-tracing is
    /// a bug. (`NotTraced` is reserved for the opposite condition: a
    /// `backward()`/lookup on a tensor with *no* graph.)
    pub fn traced(&self) -> Result<Tensor> {
        autograd::traced(self)
    }

    /// Reverse-mode autodiff from this tensor, returning a fresh, linear
    /// [`Grads`]. [`Error::NotTraced`](crate::Error::NotTraced) if this
    /// tensor carries no autograd graph.
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
    pub fn detach(&self) -> Tensor {
        self.detach_shallow()
    }

    /// Internal shallow detach used by the detached-output capture rule
    /// (exploration §4.3): a fresh `Inner` sharing storage, `node: None`,
    /// built before the traced output is assembled. A backward closure that
    /// needs the op's output value (sigmoid/tanh/softmax) captures *this* form
    /// so the closure never holds an `Arc` back to its own output node.
    pub(crate) fn detach_shallow(&self) -> Tensor {
        Tensor::from_parts(self.0.storage.clone(), self.0.layout.clone())
    }

    // ---- constructors (T20 fills the bodies) -----------------------------

    /// A tensor of zeros.
    pub fn zeros(shape: impl Into<Shape>, dtype: DType, device: &Device) -> Result<Tensor> {
        Tensor::full(shape, 0.0, dtype, device)
    }

    /// A tensor of ones.
    pub fn ones(shape: impl Into<Shape>, dtype: DType, device: &Device) -> Result<Tensor> {
        Tensor::full(shape, 1.0, dtype, device)
    }

    /// A tensor filled with `value` (narrowed to `dtype`).
    ///
    /// The single overflow-validation point for constructed tensors is the
    /// contiguous layout, which errors with
    /// [`Error::InvalidArg`](crate::Error::InvalidArg) (`op: "layout"`) if the
    /// element count overflows `usize`.
    pub fn full(
        shape: impl Into<Shape>,
        value: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let layout = Layout::contiguous(shape)?;
        let storage = dispatch::backend(*device).full(layout.num_elements(), dtype, value)?;
        Ok(Tensor::from_parts(storage, layout))
    }

    /// A tensor from a host vector; the dtype is `T`'s. `data.len()` must
    /// equal the shape's element count
    /// ([`Error::ShapeMismatch`](crate::Error::ShapeMismatch)).
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
    /// row-major order so a given `rng` state is reproducible independent of
    /// the backend.
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

    /// A 1-D range `[start, end)` stepped by `step` (T20 is the sole owner of
    /// `arange`; the indexing set builds on it).
    ///
    /// `step` must be non-zero and every bound finite
    /// ([`Error::InvalidArg`](crate::Error::InvalidArg) otherwise). The value
    /// count is `ceil((end - start) / step)`, or zero when `step` points away
    /// from `end` (an empty range, matching PyTorch rather than erroring).
    /// Values are generated as `f64` (`start + i * step`) and narrowed to
    /// `dtype`; a [`Bool`](crate::DType::Bool) `dtype` is
    /// [`Error::InvalidArg`](crate::Error::InvalidArg) since a stepped range is
    /// not a boolean sequence, and so is a range whose element count would not
    /// fit in `usize`.
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

    // ---- host transfer / movement (T20 fills the bodies) -----------------

    /// Copy the tensor's elements to a host `Vec` in row-major order (a host
    /// boundary: synchronizes the backend). Dtype must be `T`, else
    /// [`Error::DTypeMismatch`](crate::Error::DTypeMismatch).
    pub fn to_vec<T: Element>(&self) -> Result<Vec<T>> {
        let host = dispatch::backend(self.device()).transfer_out(self.view())?;
        <T as HostConv>::try_from_cpu_storage(&host, "to_vec")
    }

    /// Read a single-element tensor as a scalar of type `T`. The tensor must
    /// have exactly one element ([`Error::InvalidArg`](crate::Error::InvalidArg))
    /// and dtype `T` ([`Error::DTypeMismatch`](crate::Error::DTypeMismatch)).
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
        let host = dispatch::backend(self.device()).transfer_out(self.view())?;
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

    /// Move to `device`. A differentiable op: the backward pass moves the
    /// cotangent back to the source device. `cpu→cpu` is the identity (and,
    /// currently, the only reachable case — accelerator backends land in T61).
    pub fn to_device(&self, device: &Device) -> Result<Tensor> {
        if self.device() == *device {
            return Ok(self.clone());
        }
        let host = dispatch::backend(self.device()).transfer_out(self.view())?;
        let storage = dispatch::backend(*device).transfer_in(host)?;
        let layout = Layout::contiguous(self.shape().clone())?;
        let out = Tensor::from_parts(storage, layout);
        let src = self.device();
        // The backward seam is infallible (`BackwardFn` yields
        // `Option<Tensor>`, not `Result`), so a failed move back becomes "no
        // gradient for this input". Moving a cotangent to the device its
        // forward input already lived on can only fail if that device itself
        // is gone, which the forward pass would have failed on first.
        Ok(autograd::record(
            "to_device",
            out,
            &[self],
            Box::new(move |g| vec![g.to_device(&src).ok()]),
        ))
    }

    /// Cast to `dtype` via the backend Cast kernel. A differentiable op: the
    /// backward pass casts the cotangent back to the input dtype (meaningful
    /// for float↔float; the engine drops gradients for integer/bool inputs).
    pub fn to_dtype(&self, dtype: DType) -> Result<Tensor> {
        if self.dtype() == dtype {
            return Ok(self.clone());
        }
        let storage = dispatch::backend(self.device()).cast(self.view(), dtype)?;
        let layout = Layout::contiguous(self.shape().clone())?;
        let out = Tensor::from_parts(storage, layout);
        let src = self.dtype();
        Ok(autograd::record(
            "to_dtype",
            out,
            &[self],
            Box::new(move |g| vec![g.to_dtype(src).ok()]),
        ))
    }

    /// A contiguous copy in row-major order, or `self` unchanged when already
    /// contiguous. Value-identity, so it is transparent to autograd (the
    /// cotangent flows straight through).
    pub fn contiguous(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        let storage = dispatch::backend(self.device()).copy_strided(self.view())?;
        let layout = Layout::contiguous(self.shape().clone())?;
        let out = Tensor::from_parts(storage, layout);
        Ok(autograd::record(
            "contiguous",
            out,
            &[self],
            Box::new(move |g| vec![Some(g.clone())]),
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
    /// T23's `sum`/`sum_keepdim`.)
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
        // Highest axis first: dropping axis `k` leaves every axis below `k`
        // at its original index.
        let mut cur = self.detach_shallow();
        if matches!(dtype, DType::F16 | DType::BF16) {
            cur = cur.to_dtype(DType::F32)?;
        }
        let backend = dispatch::backend(self.device());
        for &axis in reduce_axes.iter().rev() {
            let storage = backend.reduce(ReduceOp::Sum, cur.view(), axis)?;
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

// `Tensor` must be `Send + Sync` (exploration §4.1). This fails to compile if
// any field of `Inner` ever loses those bounds.
const _: () = {
    fn assert_send_sync<T: Send + Sync>() {}
    fn check() {
        assert_send_sync::<Tensor>();
    }
    let _ = check;
};

#[cfg(test)]
mod tests {
    use super::*;

    const CPU: Device = Device::Cpu;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /// A contiguous f32 tensor on CPU.
    fn t_f32(data: &[f32], shape: impl Into<Shape>) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape, &CPU).unwrap()
    }

    /// Re-view `t` through `layout` over the same storage. The only way to
    /// build a non-contiguous `Tensor` before T21 lands the public view ops.
    fn re_view(t: &Tensor, layout: Layout) -> Tensor {
        Tensor::from_parts(t.storage().clone(), layout)
    }

    /// The address of the f32 buffer behind `t`, for "did this share storage
    /// or allocate a copy?" assertions.
    fn f32_buf_ptr(t: &Tensor) -> *const f32 {
        match t.storage() {
            Storage::Cpu(CpuStorage::F32(v)) => v.as_ptr(),
            _ => panic!("expected an f32 CPU tensor"),
        }
    }

    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    #[test]
    fn zeros_ones_full_shape_dtype_device() {
        let z = Tensor::zeros([2, 3], DType::F32, &CPU).unwrap();
        assert_eq!(z.dims(), &[2, 3]);
        assert_eq!(z.shape(), &Shape::from([2, 3]));
        assert_eq!(z.rank(), 2);
        assert_eq!(z.num_elements(), 6);
        assert_eq!(z.dtype(), DType::F32);
        assert_eq!(z.device(), CPU);
        assert!(z.is_contiguous());
        assert_eq!(z.to_vec::<f32>().unwrap(), vec![0.0; 6]);

        let o = Tensor::ones([4], DType::F32, &CPU).unwrap();
        assert_eq!(o.to_vec::<f32>().unwrap(), vec![1.0; 4]);

        let f = Tensor::full([2, 2], -1.5, DType::F32, &CPU).unwrap();
        assert_eq!(f.to_vec::<f32>().unwrap(), vec![-1.5; 4]);
    }

    #[test]
    fn constructors_cover_every_dtype_flavour() {
        // Integer fills truncate toward zero; bool fills are `value != 0`.
        let i = Tensor::full([3], 2.9, DType::I64, &CPU).unwrap();
        assert_eq!(i.dtype(), DType::I64);
        assert_eq!(i.to_vec::<i64>().unwrap(), vec![2, 2, 2]);

        let b = Tensor::ones([2], DType::Bool, &CPU).unwrap();
        assert_eq!(b.to_vec::<bool>().unwrap(), vec![true, true]);
        let b = Tensor::zeros([2], DType::Bool, &CPU).unwrap();
        assert_eq!(b.to_vec::<bool>().unwrap(), vec![false, false]);

        // f16/bf16/f64 tensors can be *built* even though their kernels are
        // deferred (T60): the six Element impls exist from the start.
        let h = Tensor::full([2], 1.0, DType::F16, &CPU).unwrap();
        assert_eq!(h.dtype(), DType::F16);
        assert_eq!(
            h.to_vec::<half::f16>().unwrap(),
            vec![half::f16::from_f32(1.0); 2]
        );
        let d = Tensor::full([2], 0.25, DType::F64, &CPU).unwrap();
        assert_eq!(d.to_vec::<f64>().unwrap(), vec![0.25, 0.25]);
    }

    #[test]
    fn scalar_and_empty_shapes_are_constructible() {
        let s = Tensor::full((), 7.0, DType::F32, &CPU).unwrap();
        assert_eq!(s.rank(), 0);
        assert_eq!(s.num_elements(), 1);
        assert_eq!(s.to_scalar::<f32>().unwrap(), 7.0);

        let e = Tensor::zeros([0, 3], DType::F32, &CPU).unwrap();
        assert_eq!(e.num_elements(), 0);
        assert!(e.to_vec::<f32>().unwrap().is_empty());
    }

    #[test]
    fn from_vec_round_trips_and_checks_length() {
        let t = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        assert_eq!(t.dims(), &[2, 3]);
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(
            t.to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );

        // dtype comes from `T`, not from an argument.
        let i = Tensor::from_vec(vec![1i64, 2, 3], [3], &CPU).unwrap();
        assert_eq!(i.dtype(), DType::I64);
        let b = Tensor::from_vec(vec![true, false], [2], &CPU).unwrap();
        assert_eq!(b.dtype(), DType::Bool);

        // Length must equal the shape's element count.
        assert!(matches!(
            Tensor::from_vec(vec![1.0f32, 2.0], [3], &CPU),
            Err(Error::ShapeMismatch { op: "from_vec", .. })
        ));
    }

    #[test]
    fn full_overflowing_shape_is_invalid_arg() {
        // The single overflow-validation point is the contiguous layout.
        assert!(matches!(
            Tensor::zeros([usize::MAX, usize::MAX], DType::F32, &CPU),
            Err(Error::InvalidArg { op: "layout", .. })
        ));
    }

    #[test]
    fn rand_is_in_unit_interval_and_reproducible() {
        let mut rng = Rng::seed(1234);
        let a = Tensor::rand([64], DType::F32, &CPU, &mut rng).unwrap();
        let values = a.to_vec::<f32>().unwrap();
        assert_eq!(values.len(), 64);
        assert!(values.iter().all(|&v| (0.0..1.0).contains(&v)));
        // Not a constant tensor.
        assert!(values.iter().any(|&v| v != values[0]));

        // Same seed, same draw sequence.
        let mut rng2 = Rng::seed(1234);
        let b = Tensor::rand([64], DType::F32, &CPU, &mut rng2).unwrap();
        assert_eq!(b.to_vec::<f32>().unwrap(), values);
    }

    #[test]
    fn randn_is_roughly_standard_normal() {
        let mut rng = Rng::seed(7);
        let a = Tensor::randn([4096], DType::F32, &CPU, &mut rng).unwrap();
        let v = a.to_vec::<f32>().unwrap();
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
        assert!(mean.abs() < 0.1, "mean {mean}");
        assert!((var - 1.0).abs() < 0.15, "var {var}");
    }

    #[test]
    fn rand_requires_a_float_dtype() {
        let mut rng = Rng::seed(1);
        assert!(matches!(
            Tensor::rand([4], DType::I64, &CPU, &mut rng),
            Err(Error::InvalidArg { op: "rand", .. })
        ));
        assert!(matches!(
            Tensor::randn([4], DType::Bool, &CPU, &mut rng),
            Err(Error::InvalidArg { op: "randn", .. })
        ));
        // Every float dtype is accepted.
        for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
            let t = Tensor::rand([2], dtype, &CPU, &mut rng).unwrap();
            assert_eq!(t.dtype(), dtype);
        }
    }

    #[test]
    fn arange_counts_and_values() {
        let a = Tensor::arange(0.0, 5.0, 1.0, DType::F32, &CPU).unwrap();
        assert_eq!(a.dims(), &[5]);
        assert_eq!(a.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        // Non-integral count rounds up (PyTorch's ceil rule).
        let a = Tensor::arange(0.0, 1.0, 0.3, DType::F32, &CPU).unwrap();
        assert_eq!(a.dims(), &[4]); // ceil(1/0.3) == 4
        let v = a.to_vec::<f32>().unwrap();
        assert!((v[3] - 0.9).abs() < 1e-6);

        // Negative step counts down.
        let a = Tensor::arange(3.0, 0.0, -1.0, DType::F32, &CPU).unwrap();
        assert_eq!(a.to_vec::<f32>().unwrap(), vec![3.0, 2.0, 1.0]);

        // I64 is the indexing dtype the T25 utilities build on.
        let a = Tensor::arange(0.0, 4.0, 1.0, DType::I64, &CPU).unwrap();
        assert_eq!(a.dtype(), DType::I64);
        assert_eq!(a.to_vec::<i64>().unwrap(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn arange_empty_and_invalid_ranges() {
        // A step pointing away from `end` yields an empty tensor, not an error.
        let a = Tensor::arange(0.0, 5.0, -1.0, DType::F32, &CPU).unwrap();
        assert_eq!(a.dims(), &[0]);
        assert!(a.to_vec::<f32>().unwrap().is_empty());
        let a = Tensor::arange(2.0, 2.0, 1.0, DType::F32, &CPU).unwrap();
        assert_eq!(a.num_elements(), 0);

        assert!(matches!(
            Tensor::arange(0.0, 5.0, 0.0, DType::F32, &CPU),
            Err(Error::InvalidArg { op: "arange", .. })
        ));
        assert!(matches!(
            Tensor::arange(0.0, 5.0, 1.0, DType::Bool, &CPU),
            Err(Error::InvalidArg { op: "arange", .. })
        ));
        // Non-finite bounds are rejected rather than silently saturating the
        // element count.
        for (s, e, st) in [
            (f64::NAN, 5.0, 1.0),
            (0.0, f64::INFINITY, 1.0),
            (0.0, 5.0, f64::NAN),
        ] {
            assert!(matches!(
                Tensor::arange(s, e, st, DType::F32, &CPU),
                Err(Error::InvalidArg { op: "arange", .. })
            ));
        }
        // A finite range whose count cannot fit in `usize` is loud, not an
        // allocation abort.
        assert!(matches!(
            Tensor::arange(0.0, f64::MAX, 1.0, DType::F32, &CPU),
            Err(Error::InvalidArg { op: "arange", .. })
        ));
    }

    // ------------------------------------------------------------------
    // Host transfer
    // ------------------------------------------------------------------

    #[test]
    fn to_vec_walks_strided_views_in_row_major_order() {
        let t = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let transposed = re_view(&t, t.layout().transpose(0, 1).unwrap());
        assert_eq!(transposed.dims(), &[3, 2]);
        assert!(!transposed.is_contiguous());
        assert_eq!(
            transposed.to_vec::<f32>().unwrap(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );

        // Broadcast (stride-0) axes repeat elements on the way out.
        let row = t_f32(&[10.0, 20.0, 30.0], [1, 3]);
        let b = re_view(
            &row,
            row.layout().broadcast_to(&Shape::from([2, 3])).unwrap(),
        );
        assert_eq!(
            b.to_vec::<f32>().unwrap(),
            vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
        );
    }

    #[test]
    fn to_vec_is_dtype_strict() {
        let t = t_f32(&[1.0, 2.0], [2]);
        // No implicit promotion on the way to the host either: reading an f32
        // tensor as i64/f64 is a structured error, not a conversion.
        assert!(matches!(
            t.to_vec::<i64>(),
            Err(Error::DTypeMismatch {
                op: "to_vec",
                expected: DType::I64,
                got: DType::F32
            })
        ));
        assert!(matches!(
            t.to_vec::<f64>(),
            Err(Error::DTypeMismatch { op: "to_vec", .. })
        ));
        // `to_scalar` inherits the strictness (element count is checked
        // first, so use a single-element tensor here).
        assert!(matches!(
            t_f32(&[1.0], ()).to_scalar::<i64>(),
            Err(Error::DTypeMismatch { .. })
        ));
        // `item` is the deliberate exception: dtype-agnostic by design.
        assert_eq!(t_f32(&[1.0], ()).item().unwrap(), 1.0);
    }

    #[test]
    fn to_scalar_requires_exactly_one_element() {
        let one = t_f32(&[42.0], [1, 1, 1]);
        assert_eq!(one.to_scalar::<f32>().unwrap(), 42.0);
        let many = t_f32(&[1.0, 2.0], [2]);
        assert!(matches!(
            many.to_scalar::<f32>(),
            Err(Error::InvalidArg {
                op: "to_scalar",
                ..
            })
        ));
        let empty = Tensor::zeros([0], DType::F32, &CPU).unwrap();
        assert!(matches!(
            empty.to_scalar::<f32>(),
            Err(Error::InvalidArg { .. })
        ));
    }

    #[test]
    fn item_reads_any_dtype_as_f64() {
        assert_eq!(t_f32(&[2.5], ()).item().unwrap(), 2.5);
        assert_eq!(
            Tensor::from_vec(vec![7i64], (), &CPU)
                .unwrap()
                .item()
                .unwrap(),
            7.0
        );
        assert_eq!(
            Tensor::from_vec(vec![true], (), &CPU)
                .unwrap()
                .item()
                .unwrap(),
            1.0
        );
        assert_eq!(
            Tensor::from_vec(vec![false], (), &CPU)
                .unwrap()
                .item()
                .unwrap(),
            0.0
        );
        assert_eq!(
            Tensor::full((), 1.5, DType::F16, &CPU)
                .unwrap()
                .item()
                .unwrap(),
            1.5
        );
        assert_eq!(
            Tensor::full((), 1.5, DType::BF16, &CPU)
                .unwrap()
                .item()
                .unwrap(),
            1.5
        );
        assert_eq!(
            Tensor::full((), 0.5, DType::F64, &CPU)
                .unwrap()
                .item()
                .unwrap(),
            0.5
        );
        assert!(matches!(
            t_f32(&[1.0, 2.0], [2]).item(),
            Err(Error::InvalidArg { op: "item", .. })
        ));
    }

    #[test]
    fn item_reads_the_first_logical_element_of_a_view() {
        // A single-element *view* into a larger buffer must read the element
        // the view points at, not `storage[0]`.
        let t = t_f32(&[1.0, 2.0, 3.0, 4.0], [4]);
        let last = re_view(&t, t.layout().narrow(0, 3, 1).unwrap());
        assert_eq!(last.item().unwrap(), 4.0);
        assert_eq!(last.to_scalar::<f32>().unwrap(), 4.0);
    }

    // ------------------------------------------------------------------
    // Movement: to_device / to_dtype / contiguous
    // ------------------------------------------------------------------

    #[test]
    fn to_device_same_device_is_the_identity() {
        let t = t_f32(&[1.0, 2.0], [2]);
        let moved = t.to_device(&CPU).unwrap();
        assert_eq!(moved.device(), CPU);
        // Identity: the very same buffer, no copy.
        assert_eq!(f32_buf_ptr(&t), f32_buf_ptr(&moved));
    }

    #[test]
    fn to_dtype_casts_explicitly_and_never_promotes_implicitly() {
        let f = t_f32(&[1.9, -1.9, 0.0], [3]);
        let i = f.to_dtype(DType::I64).unwrap();
        assert_eq!(i.dtype(), DType::I64);
        assert_eq!(i.to_vec::<i64>().unwrap(), vec![1, -1, 0]);
        // ...and back.
        assert_eq!(
            i.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap(),
            vec![1.0, -1.0, 0.0]
        );
        // Numeric -> Bool is `x != 0`; Bool -> numeric is 0/1.
        let b = f.to_dtype(DType::Bool).unwrap();
        assert_eq!(b.to_vec::<bool>().unwrap(), vec![true, true, false]);
        assert_eq!(
            b.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap(),
            vec![1.0, 1.0, 0.0]
        );

        // The identity cast is free and keeps the same buffer.
        let same = f.to_dtype(DType::F32).unwrap();
        assert_eq!(f32_buf_ptr(&f), f32_buf_ptr(&same));

        let half = f.to_dtype(DType::F16).unwrap();
        assert_eq!(half.dtype(), DType::F16);
        assert_eq!(
            half.to_dtype(DType::BF16)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            vec![1.8984375, -1.8984375, 0.0]
        );
        // F64 cast scope remains deferred and loud.
        assert!(matches!(
            f.to_dtype(DType::F64),
            Err(Error::Unsupported { op: "to_dtype", .. })
        ));
    }

    #[test]
    fn to_dtype_materializes_strided_sources() {
        let t = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let transposed = re_view(&t, t.layout().transpose(0, 1).unwrap());
        let cast = transposed.to_dtype(DType::I64).unwrap();
        assert_eq!(cast.dims(), &[3, 2]);
        assert!(cast.is_contiguous());
        assert_eq!(cast.to_vec::<i64>().unwrap(), vec![1, 4, 2, 5, 3, 6]);
    }

    #[test]
    fn contiguous_is_free_when_already_contiguous() {
        let t = t_f32(&[1.0, 2.0, 3.0], [3]);
        assert!(t.is_contiguous());
        let c = t.contiguous().unwrap();
        assert_eq!(f32_buf_ptr(&t), f32_buf_ptr(&c));
    }

    #[test]
    fn contiguous_copies_permuted_and_narrowed_views() {
        let t = t_f32(&(0..24).map(|x| x as f32).collect::<Vec<_>>(), [2, 3, 4]);

        // Permuted: dims [4, 2, 3].
        let permuted = re_view(&t, t.layout().permute(&[2, 0, 1]).unwrap());
        assert!(!permuted.is_contiguous());
        let expected = permuted.to_vec::<f32>().unwrap();
        let c = permuted.contiguous().unwrap();
        assert!(c.is_contiguous());
        assert_eq!(c.dims(), &[4, 2, 3]);
        assert_ne!(f32_buf_ptr(&permuted), f32_buf_ptr(&c));
        assert_eq!(c.to_vec::<f32>().unwrap(), expected);

        // Narrowed (non-zero offset, gaps between rows).
        let narrowed = re_view(&t, t.layout().narrow(2, 1, 2).unwrap());
        assert!(!narrowed.is_contiguous());
        let expected = narrowed.to_vec::<f32>().unwrap();
        let c = narrowed.contiguous().unwrap();
        assert!(c.is_contiguous());
        assert_eq!(c.dims(), &[2, 3, 2]);
        assert_eq!(c.to_vec::<f32>().unwrap(), expected);
        // The copy is dense: 12 live elements, not the original 24-slot buffer.
        assert_eq!(c.storage().len(), 12);

        // Broadcast views materialize their repeats.
        let row = t_f32(&[1.0, 2.0], [1, 2]);
        let b = re_view(
            &row,
            row.layout().broadcast_to(&Shape::from([3, 2])).unwrap(),
        );
        let c = b.contiguous().unwrap();
        assert_eq!(c.dims(), &[3, 2]);
        assert_eq!(c.storage().len(), 6);
        assert_eq!(
            c.to_vec::<f32>().unwrap(),
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
        );
    }

    // ------------------------------------------------------------------
    // detach
    // ------------------------------------------------------------------

    #[test]
    fn detach_shares_storage_and_layout_and_carries_no_node() {
        let t = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let view = re_view(&t, t.layout().transpose(0, 1).unwrap());

        let d = view.detach();
        assert!(d.node().is_none());
        // Storage is shared (an Arc bump, not an element copy)...
        assert_eq!(f32_buf_ptr(&view), f32_buf_ptr(&d));
        // ...and so is the layout: detach is not a `contiguous()`.
        assert_eq!(d.layout(), view.layout());
        assert_eq!(d.dims(), &[3, 2]);
        assert!(!d.is_contiguous());
        assert_eq!(d.to_vec::<f32>().unwrap(), view.to_vec::<f32>().unwrap());

        // `detach` is exactly the public spelling of `detach_shallow`.
        let shallow = view.detach_shallow();
        assert_eq!(f32_buf_ptr(&shallow), f32_buf_ptr(&d));
        assert_eq!(shallow.layout(), d.layout());

        // An untraced tensor detaches to an equivalent untraced tensor; the
        // node-dropping half of the contract cannot be exercised until T30
        // makes `record`/`make_leaf` build real nodes.
        assert!(t.detach().node().is_none());
    }

    // ------------------------------------------------------------------
    // sum_to — the broadcast-gradient reducer
    // ------------------------------------------------------------------

    #[test]
    fn sum_to_identity_returns_the_same_buffer() {
        let t = t_f32(&[1.0, 2.0, 3.0, 4.0], [2, 2]);
        let s = t.sum_to(&[2, 2]).unwrap();
        assert_eq!(s.dims(), &[2, 2]);
        assert_eq!(f32_buf_ptr(&t), f32_buf_ptr(&s));

        // Size-1 axes that were never expanded are also a no-op.
        let t = t_f32(&[1.0, 2.0, 3.0], [1, 3]);
        let s = t.sum_to(&[1, 3]).unwrap();
        assert_eq!(f32_buf_ptr(&t), f32_buf_ptr(&s));

        // Rank-0 to rank-0.
        let t = t_f32(&[5.0], ());
        assert_eq!(t.sum_to(&[]).unwrap().item().unwrap(), 5.0);
    }

    #[test]
    fn sum_to_scalar_target_sums_everything() {
        let t = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let s = t.sum_to(&[]).unwrap();
        assert_eq!(s.rank(), 0);
        assert_eq!(s.num_elements(), 1);
        assert_eq!(s.to_scalar::<f32>().unwrap(), 21.0);

        // The rank-1 size-1 target is the *other* scalar spelling: shape [1].
        let s = t.sum_to(&[1]).unwrap();
        assert_eq!(s.dims(), &[1]);
        assert_eq!(s.to_vec::<f32>().unwrap(), vec![21.0]);
    }

    #[test]
    fn sum_to_partial_broadcast_keeps_size_one_axes() {
        // [[1,2,3],[4,5,6]]
        let t = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);

        // Column vector operand: sum over the last axis, keep it at 1.
        let s = t.sum_to(&[2, 1]).unwrap();
        assert_eq!(s.dims(), &[2, 1]);
        assert_eq!(s.to_vec::<f32>().unwrap(), vec![6.0, 15.0]);

        // Row vector operand: sum over the leading axis, keep it at 1.
        let s = t.sum_to(&[1, 3]).unwrap();
        assert_eq!(s.dims(), &[1, 3]);
        assert_eq!(s.to_vec::<f32>().unwrap(), vec![5.0, 7.0, 9.0]);

        // Both axes at 1: everything summed, rank preserved.
        let s = t.sum_to(&[1, 1]).unwrap();
        assert_eq!(s.dims(), &[1, 1]);
        assert_eq!(s.to_vec::<f32>().unwrap(), vec![21.0]);
    }

    #[test]
    fn sum_to_drops_leading_axes() {
        // A [2, 3] operand broadcast against a [4, 2, 3] one: the cotangent
        // is [4, 2, 3] and the leading axis must be summed away and dropped.
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let t = t_f32(&data, [4, 2, 3]);
        let s = t.sum_to(&[2, 3]).unwrap();
        assert_eq!(s.dims(), &[2, 3]);
        // out[i,j] = sum over b of data[b, i, j]; the four planes are offset
        // by 6 each, so out = base + (0+6+12+18) = base + 36.
        assert_eq!(
            s.to_vec::<f32>().unwrap(),
            vec![36.0, 40.0, 44.0, 48.0, 52.0, 56.0]
        );

        // A [3] operand: two leading axes dropped.
        let s = t.sum_to(&[3]).unwrap();
        assert_eq!(s.dims(), &[3]);
        // Column sums of all 8 rows: rows start at 0,3,6,...,21.
        let expected: Vec<f32> = (0..3)
            .map(|c| (0..8).map(|r| (r * 3 + c) as f32).sum())
            .collect();
        assert_eq!(s.to_vec::<f32>().unwrap(), expected);

        // Leading axes and an expanded aligned axis at once.
        let s = t.sum_to(&[1, 3]).unwrap();
        assert_eq!(s.dims(), &[1, 3]);
        assert_eq!(s.to_vec::<f32>().unwrap(), expected);
    }

    #[test]
    fn sum_to_is_the_transpose_of_broadcast_to() {
        // Property: for every operand shape that broadcasts to `[2, 3, 4]`,
        // summing a cotangent of ones back to that operand must yield the
        // number of output elements each operand element fed, i.e. the
        // broadcast multiplicity.
        let out_dims = [2usize, 3, 4];
        let total: usize = out_dims.iter().product();
        let g = Tensor::ones(out_dims, DType::F32, &CPU).unwrap();
        for target in [
            vec![2usize, 3, 4],
            vec![1, 3, 4],
            vec![2, 1, 4],
            vec![2, 3, 1],
            vec![1, 1, 4],
            vec![1, 1, 1],
            vec![3, 4],
            vec![1, 4],
            vec![4],
            vec![1],
            vec![],
        ] {
            let reduced = g.sum_to(&target).unwrap();
            assert_eq!(reduced.dims(), target.as_slice(), "target {target:?}");
            let n: usize = target.iter().product();
            let multiplicity = (total / n) as f32;
            assert_eq!(
                reduced.to_vec::<f32>().unwrap(),
                vec![multiplicity; n],
                "target {target:?}"
            );
        }
    }

    #[test]
    fn sum_to_handles_strided_sources_and_other_dtypes() {
        // A non-contiguous cotangent must reduce correctly (the backend reduce
        // walks strides; no `contiguous()` is required first).
        let t = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let transposed = re_view(&t, t.layout().transpose(0, 1).unwrap()); // [3, 2]
        let s = transposed.sum_to(&[1, 2]).unwrap();
        assert_eq!(s.dims(), &[1, 2]);
        // Transposed rows are [1,4],[2,5],[3,6]; column sums are [6, 15].
        assert_eq!(s.to_vec::<f32>().unwrap(), vec![6.0, 15.0]);

        // dtype and device survive the reduction.
        let i = Tensor::from_vec(vec![1i64, 2, 3, 4], [2, 2], &CPU).unwrap();
        let s = i.sum_to(&[2, 1]).unwrap();
        assert_eq!(s.dtype(), DType::I64);
        assert_eq!(s.device(), CPU);
        assert_eq!(s.to_vec::<i64>().unwrap(), vec![3, 7]);
    }

    #[test]
    fn sum_to_accumulates_in_the_wide_acc_type() {
        for dtype in [DType::F16, DType::BF16] {
            // Native f16 addition stalls at 2048 and bf16 at 256; Acc = f32.
            let t = Tensor::full([4096], 1.0, dtype, &CPU).unwrap();
            let s = t.sum_to(&[]).unwrap();
            assert_eq!(s.dtype(), dtype);
            assert_eq!(s.item().unwrap(), 4096.0);
        }
    }

    #[test]
    fn sum_to_keeps_reduced_multi_axis_accumulation_wide_until_the_target_shape() {
        let width = 65_520;
        let mut f16_values = vec![half::f16::ONE; width];
        f16_values.extend(vec![half::f16::NEG_ONE; width]);
        let f16 = Tensor::from_vec(f16_values, [2, width], &CPU).unwrap();
        assert_eq!(f16.sum_to(&[1, 1]).unwrap().item().unwrap(), 0.0);

        let mut bf16_values = Vec::with_capacity(514);
        for index in 0..257 {
            bf16_values.push(half::bf16::ONE);
            bf16_values.push(if index < 256 {
                half::bf16::NEG_ONE
            } else {
                half::bf16::ZERO
            });
        }
        let base = Tensor::from_vec(bf16_values, [257, 2], &CPU).unwrap();
        let strided = re_view(&base, base.layout().transpose(0, 1).unwrap());
        assert_eq!(strided.sum_to(&[1, 1]).unwrap().item().unwrap(), 1.0);
    }

    #[test]
    fn sum_to_over_an_empty_axis_is_zero() {
        // Broadcasting a [3] operand to [0, 3] produces no outputs, so the
        // gradient contribution is zero, not an error.
        let t = Tensor::zeros([0, 3], DType::F32, &CPU).unwrap();
        let s = t.sum_to(&[3]).unwrap();
        assert_eq!(s.dims(), &[3]);
        assert_eq!(s.to_vec::<f32>().unwrap(), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn sum_to_rejects_shapes_that_never_broadcast() {
        let t = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);

        // Target of higher rank than the source.
        assert!(matches!(
            t.sum_to(&[1, 2, 3]),
            Err(Error::ShapeMismatch { op: "sum_to", .. })
        ));
        // Aligned axis that is neither equal nor 1 on the target side.
        assert!(matches!(
            t.sum_to(&[2, 2]),
            Err(Error::ShapeMismatch { op: "sum_to", .. })
        ));
        // Target axis *larger* than the source's: that is a broadcast, not a
        // reduction.
        assert!(matches!(
            t.sum_to(&[4, 3]),
            Err(Error::ShapeMismatch { op: "sum_to", .. })
        ));
        // The reported shapes name both sides.
        match t.sum_to(&[5]) {
            Err(Error::ShapeMismatch { lhs, rhs, .. }) => {
                assert_eq!(lhs, Shape::from([2, 3]));
                assert_eq!(rhs, Shape::from([5]));
            }
            _ => panic!("expected a ShapeMismatch"),
        }

        // Bool has no sum: the backend says so rather than inventing one.
        let b = Tensor::from_vec(vec![true, false], [2], &CPU).unwrap();
        assert!(matches!(
            b.sum_to(&[1]),
            Err(Error::Unsupported { op: "reduce", .. })
        ));
    }
}
