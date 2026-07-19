//! Strided layouts: how a tensor's logical shape maps onto its storage.
//!
//! **Contract file** (T01). The struct and every signature here are frozen;
//! T02 fills the `todo!()` bodies (ported v2 layout math plus the new
//! permute/narrow/broadcast-stride rules). Doc comments on each method are
//! the normative semantics T02 implements and property-tests against.
//!
//! Conventions:
//! - Row-major (C order): the last axis is the fastest-varying one in a
//!   contiguous layout.
//! - Strides are in **elements** (not bytes), `usize`, one per axis.
//!   Stride 0 encodes a broadcast axis (the axis repeats the same
//!   elements). There are no negative strides (no `flip` in the
//!   vocabulary).
//! - `offset` is the element index in storage of the logical element at
//!   all-zero coordinates.
//! - Axes arriving here are already resolved to `[0, rank)` — negative-axis
//!   handling happens in the op layer via [`Shape::resolve_axis`].

// Consumed by T02 and the W3 op tasks; the integrator removes this allow
// at v3-m1 once consumers exist.
#![allow(dead_code)]

use crate::error::Result;
use crate::shape::Shape;

/// A strided view description: shape + per-axis element strides + start
/// offset into the storage buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Layout {
    shape: Shape,
    strides: Box<[usize]>,
    offset: usize,
}

impl Layout {
    /// The canonical contiguous (row-major, offset 0) layout for `shape`.
    ///
    /// Errors with [`crate::Error::InvalidArg`] if the element count
    /// overflows `usize` (the crate's single overflow validation point:
    /// every tensor construction passes through here or
    /// [`Layout::from_parts`]).
    pub(crate) fn contiguous(shape: impl Into<Shape>) -> Result<Layout> {
        let _ = shape.into();
        todo!("T02: contiguous stride computation (port v2)")
    }

    /// Build a layout from raw parts, validating `strides.len() ==
    /// shape.rank()` and that the maximal reachable element index fits in
    /// the addressable range.
    pub(crate) fn from_parts(shape: Shape, strides: Box<[usize]>, offset: usize) -> Result<Layout> {
        let _ = (shape, strides, offset);
        todo!("T02: from_parts validation")
    }

    /// The logical shape of this view.
    pub(crate) fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Dimension sizes, outermost first.
    pub(crate) fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Per-axis element strides (see module docs; 0 = broadcast axis).
    pub(crate) fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Element offset of the logical origin in storage.
    pub(crate) fn offset(&self) -> usize {
        self.offset
    }

    /// Rank of the view.
    pub(crate) fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Logical element count of the view.
    pub(crate) fn num_elements(&self) -> usize {
        self.shape.num_elements()
    }

    /// Whether this layout is exactly the canonical contiguous layout:
    /// offset 0 and row-major strides (computed right-to-left with unit
    /// innermost stride). Size-1 axes must still carry the canonical
    /// stride for `true` (v2 rule, kept for byte-identical interchange).
    pub(crate) fn is_contiguous(&self) -> bool {
        todo!("T02: contiguity check (port v2)")
    }

    /// Swap two axes (both pre-resolved). Zero-copy: permutes the shape
    /// and stride entries; offset unchanged.
    pub(crate) fn transpose(&self, a: usize, b: usize) -> Result<Layout> {
        let _ = (a, b);
        todo!("T02: transpose")
    }

    /// Reorder all axes by `perm` (a permutation of `0..rank`, validated:
    /// each axis exactly once, length == rank, else
    /// [`crate::Error::InvalidArg`]). Zero-copy.
    pub(crate) fn permute(&self, perm: &[usize]) -> Result<Layout> {
        let _ = perm;
        todo!("T02: general permute (new work, exploration §3.2)")
    }

    /// Restrict `axis` to `[start, start + len)`. Zero-copy: the new
    /// offset is `offset + start * strides[axis]`; the axis size becomes
    /// `len`; strides unchanged. `start + len` must not exceed the axis
    /// size ([`crate::Error::IndexOutOfBounds`]); `len == 0` is valid and
    /// yields an empty view.
    pub(crate) fn narrow(&self, axis: usize, start: usize, len: usize) -> Result<Layout> {
        let _ = (axis, start, len);
        todo!("T02: narrow (new work)")
    }

    /// Broadcast this view to `target` shape (NumPy rules, right-aligned;
    /// the target must be a valid broadcast of the current shape, else
    /// [`crate::Error::ShapeMismatch`]). Zero-copy: broadcast axes
    /// (including new leading axes) get stride 0; existing size-1 axes
    /// expanding to `n > 1` get stride 0.
    pub(crate) fn broadcast_to(&self, target: &Shape) -> Result<Layout> {
        let _ = target;
        todo!("T02: broadcast-stride computation (new work)")
    }

    /// Remove a size-1 axis (pre-resolved; the axis must have size 1, else
    /// [`crate::Error::InvalidArg`]). Zero-copy: drops the shape/stride
    /// entries.
    pub(crate) fn squeeze(&self, axis: usize) -> Result<Layout> {
        let _ = axis;
        todo!("T02: squeeze")
    }

    /// Insert a size-1 axis at `axis` (pre-resolved insertion position in
    /// `[0, rank]`). Zero-copy; the inserted stride is chosen so the
    /// layout of a contiguous tensor stays contiguous.
    pub(crate) fn unsqueeze(&self, axis: usize) -> Result<Layout> {
        let _ = axis;
        todo!("T02: unsqueeze")
    }

    /// Attempt to view this layout as `new_shape` **without copying**
    /// (PyTorch `reshape` semantics, exploration §4.2): returns
    /// `Some(layout)` when the elements of the new shape can be addressed
    /// by some stride assignment over the existing storage walk order —
    /// always true for contiguous layouts; true for permuted/narrowed
    /// layouts only when merged/split axes remain stride-compatible.
    /// Returns `None` when a copy is required (the caller materializes via
    /// `copy_strided` and reshapes the copy). Element counts must already
    /// match (checked by the caller, which owns the
    /// [`crate::Error::ReshapeMismatch`] error).
    pub(crate) fn reshape_view(&self, new_shape: Shape) -> Option<Layout> {
        let _ = new_shape;
        todo!("T02: view-or-copy reshape rule")
    }
}
