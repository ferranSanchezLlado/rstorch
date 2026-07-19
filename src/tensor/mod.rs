//! The [`Tensor`] type and its op surface: one concrete tensor, one spelling
//! per operation (exploration §4.1–4.2).
//!
//! **Contract file** (T01²/T20). T01 defines [`Tensor`]/[`Inner`], the
//! accessors, the crate-internal plumbing (`from_parts`, `view`), and the
//! autograd delegation ([`traced`](Tensor::traced)/[`backward`](Tensor::backward)).
//! **T20** fills the constructor and host-transfer/movement `todo!()` bodies;
//! op families (T21–T27) add methods in `tensor/ops/*` behind the frozen
//! [`record`](crate::autograd) seam. No signature here changes after T01.

// The crate-internal plumbing (`from_parts*`, `view`, `storage`, `layout`,
// `node`, `detach_shallow`) is consumed by the wave-2/3 kernel, op, and
// autograd tasks; the integrator removes this allow at v3-m1. Public
// constructors/accessors are unaffected (they are never dead).
#![allow(dead_code)]

pub(crate) mod ops;

use crate::autograd::{self, Grads};
use crate::backend::View;
use crate::device::Device;
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::rng::Rng;
use crate::shape::Shape;
use crate::storage::Storage;
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
    /// no autograd node. Differentiable ops treat it as a constant. T20 fills
    /// the body.
    pub fn detach(&self) -> Tensor {
        todo!("T20: detach")
    }

    /// Internal shallow detach used by the detached-output capture rule
    /// (exploration §4.3): a fresh `Inner` sharing storage, `node: None`,
    /// built before the traced output is assembled. T20 fills the body.
    pub(crate) fn detach_shallow(&self) -> Tensor {
        todo!("T20: detach_shallow")
    }

    // ---- constructors (T20 fills the bodies) -----------------------------

    /// A tensor of zeros.
    pub fn zeros(shape: impl Into<Shape>, dtype: DType, device: &Device) -> Result<Tensor> {
        let _ = (shape.into(), dtype, device);
        todo!("T20: zeros")
    }

    /// A tensor of ones.
    pub fn ones(shape: impl Into<Shape>, dtype: DType, device: &Device) -> Result<Tensor> {
        let _ = (shape.into(), dtype, device);
        todo!("T20: ones")
    }

    /// A tensor filled with `value` (narrowed to `dtype`).
    pub fn full(
        shape: impl Into<Shape>,
        value: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let _ = (shape.into(), value, dtype, device);
        todo!("T20: full")
    }

    /// A tensor from a host vector; the dtype is `T`'s. `data.len()` must
    /// equal the shape's element count
    /// ([`Error::ShapeMismatch`](crate::Error::ShapeMismatch)).
    pub fn from_vec<T: Element>(
        data: Vec<T>,
        shape: impl Into<Shape>,
        device: &Device,
    ) -> Result<Tensor> {
        let _ = (data, shape.into(), device);
        todo!("T20: from_vec")
    }

    /// Uniform samples in `[0, 1)` (float `dtype` only). Consumes randomness
    /// from `rng`.
    pub fn rand(
        shape: impl Into<Shape>,
        dtype: DType,
        device: &Device,
        rng: &mut Rng,
    ) -> Result<Tensor> {
        let _ = (shape.into(), dtype, device, rng);
        todo!("T20: rand")
    }

    /// Standard-normal samples (float `dtype` only). Consumes randomness from
    /// `rng`.
    pub fn randn(
        shape: impl Into<Shape>,
        dtype: DType,
        device: &Device,
        rng: &mut Rng,
    ) -> Result<Tensor> {
        let _ = (shape.into(), dtype, device, rng);
        todo!("T20: randn")
    }

    /// A 1-D range `[start, end)` stepped by `step` (T20 is the sole owner of
    /// `arange`; the indexing set builds on it).
    pub fn arange(
        start: f64,
        end: f64,
        step: f64,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let _ = (start, end, step, dtype, device);
        todo!("T20: arange")
    }

    // ---- host transfer / movement (T20 fills the bodies) -----------------

    /// Copy the tensor's elements to a host `Vec` in row-major order (a host
    /// boundary: synchronizes the backend). Dtype must be `T`.
    pub fn to_vec<T: Element>(&self) -> Result<Vec<T>> {
        todo!("T20: to_vec")
    }

    /// Read a rank-0 (or single-element) tensor as a scalar of type `T`.
    pub fn to_scalar<T: Element>(&self) -> Result<T> {
        todo!("T20: to_scalar")
    }

    /// Read a single-element tensor as `f64` (dtype-agnostic convenience).
    pub fn item(&self) -> Result<f64> {
        todo!("T20: item")
    }

    /// Move to `device` (a differentiable op; cpu→cpu is trivial). T20 fills
    /// the body.
    pub fn to_device(&self, device: &Device) -> Result<Tensor> {
        let _ = device;
        todo!("T20: to_device")
    }

    /// Cast to `dtype` via the backend Cast kernel (a differentiable op). T20
    /// fills the body.
    pub fn to_dtype(&self, dtype: DType) -> Result<Tensor> {
        let _ = dtype;
        todo!("T20: to_dtype")
    }

    /// A contiguous copy (or `self` when already contiguous). T20 fills the
    /// body.
    pub fn contiguous(&self) -> Result<Tensor> {
        todo!("T20: contiguous")
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
