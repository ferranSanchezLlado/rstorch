//! Strided layouts: how a tensor's logical shape maps onto its storage.
//!
//! The doc comment on each method is its normative semantics, and is what the
//! property tests check against.
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

use crate::error::{Error, Result};
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
    /// overflows `usize`. Together with [`Layout::from_parts`] — which every
    /// other constructor here, including `broadcast_to`, routes through — these
    /// are the crate's overflow validation points, so no `Layout` can exist
    /// whose dims disagree with its own element count.
    pub(crate) fn contiguous(shape: impl Into<Shape>) -> Result<Layout> {
        let shape = shape.into();
        let rank = shape.rank();
        let mut strides = vec![0usize; rank];
        // Row-major: innermost axis has unit stride; each axis stride is the
        // product of the sizes of all axes to its right.
        let mut stride = 1usize;
        for (idx, &dim) in shape.dims().iter().enumerate().rev() {
            strides[idx] = stride;
            stride = stride.checked_mul(dim).ok_or_else(|| Error::InvalidArg {
                op: "layout",
                msg: format!("element count of shape {shape} overflows usize"),
            })?;
        }
        Ok(Layout {
            shape,
            strides: strides.into_boxed_slice(),
            offset: 0,
        })
    }

    /// Build a layout from raw parts, validating `strides.len() ==
    /// shape.rank()` and that the maximal reachable element index fits in
    /// the addressable range.
    pub(crate) fn from_parts(shape: Shape, strides: Box<[usize]>, offset: usize) -> Result<Layout> {
        if strides.len() != shape.rank() {
            return Err(Error::InvalidArg {
                op: "layout",
                msg: format!(
                    "stride count {} does not match rank {} of shape {shape}",
                    strides.len(),
                    shape.rank()
                ),
            });
        }
        // The element count must not wrap, independently of the stride check
        // below: a broadcast layout has stride 0 on every expanded axis, so its
        // highest reachable index stays at `offset` no matter how large the
        // dims are. Without this, `[usize::MAX, usize::MAX]` would produce a
        // layout reporting those dims with a wrapped `num_elements`, which then
        // disagrees with itself everywhere downstream.
        if shape.checked_num_elements().is_none() {
            return Err(Error::InvalidArg {
                op: "layout",
                msg: format!("element count of shape {shape} overflows usize"),
            });
        }
        let layout = Layout {
            shape,
            strides,
            offset,
        };
        // Validate that the highest element index the view can address fits
        // in `usize` (no wrap when the kernels walk it). A zero-element view
        // reaches no elements, so it is always in range.
        layout.max_storage_index()?;
        Ok(layout)
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

    /// One past the highest storage index this view can address, i.e. the
    /// minimum backing-storage length that fully contains the view. Zero for
    /// an empty view.
    ///
    /// Errors with [`crate::Error::InvalidArg`] if that bound overflows
    /// `usize`.
    fn max_storage_index(&self) -> Result<usize> {
        if self.num_elements() == 0 {
            return Ok(0);
        }
        let overflow = || Error::InvalidArg {
            op: "layout",
            msg: format!(
                "layout of shape {} at offset {} overflows the addressable range",
                self.shape, self.offset
            ),
        };
        // Every axis is non-zero here, so the maximal index is reached at the
        // last coordinate (`dim - 1`) along every axis.
        let mut max = self.offset;
        for (&dim, &stride) in self.shape.dims().iter().zip(self.strides.iter()) {
            let contribution = (dim - 1).checked_mul(stride).ok_or_else(overflow)?;
            max = max.checked_add(contribution).ok_or_else(overflow)?;
        }
        max.checked_add(1).ok_or_else(overflow)
    }

    /// Whether this layout is exactly the canonical contiguous layout:
    /// offset 0 and row-major strides (computed right-to-left with unit
    /// innermost stride). Size-1 axes must still carry the canonical
    /// stride for `true`, so the predicate matches what a file format's
    /// contiguity check will say about the same buffer.
    pub(crate) fn is_contiguous(&self) -> bool {
        if self.offset != 0 {
            return false;
        }
        let mut expected = 1usize;
        for (&dim, &stride) in self.shape.dims().iter().zip(self.strides.iter()).rev() {
            if stride != expected {
                return false;
            }
            let Some(next) = expected.checked_mul(dim) else {
                return false;
            };
            expected = next;
        }
        true
    }

    /// Swap two axes (both pre-resolved). Zero-copy: permutes the shape
    /// and stride entries; offset unchanged.
    pub(crate) fn transpose(&self, a: usize, b: usize) -> Result<Layout> {
        let rank = self.rank();
        if a >= rank || b >= rank {
            return Err(Error::InvalidAxis {
                op: "transpose",
                axis: a.max(b) as isize,
                rank,
            });
        }
        let mut dims = self.shape.dims().to_vec();
        let mut strides = self.strides.to_vec();
        dims.swap(a, b);
        strides.swap(a, b);
        Ok(Layout {
            shape: Shape::from(dims),
            strides: strides.into_boxed_slice(),
            offset: self.offset,
        })
    }

    /// Reorder all axes by `perm` (a permutation of `0..rank`, validated:
    /// each axis exactly once, length == rank, else
    /// [`crate::Error::InvalidArg`]). Zero-copy.
    pub(crate) fn permute(&self, perm: &[usize]) -> Result<Layout> {
        let rank = self.rank();
        if perm.len() != rank {
            return Err(Error::InvalidArg {
                op: "permute",
                msg: format!("permutation of length {} for rank {rank}", perm.len()),
            });
        }
        // Each axis in `0..rank` must appear exactly once.
        let mut seen = vec![false; rank];
        for &axis in perm {
            if axis >= rank {
                return Err(Error::InvalidArg {
                    op: "permute",
                    msg: format!("axis {axis} out of range for rank {rank}"),
                });
            }
            if seen[axis] {
                return Err(Error::InvalidArg {
                    op: "permute",
                    msg: format!("axis {axis} appears more than once in permutation"),
                });
            }
            seen[axis] = true;
        }
        let dims: Vec<usize> = perm.iter().map(|&i| self.shape.dims()[i]).collect();
        let strides: Vec<usize> = perm.iter().map(|&i| self.strides[i]).collect();
        Ok(Layout {
            shape: Shape::from(dims),
            strides: strides.into_boxed_slice(),
            offset: self.offset,
        })
    }

    /// Restrict `axis` to `[start, start + len)`. Zero-copy: the new
    /// offset is `offset + start * strides[axis]`; the axis size becomes
    /// `len`; strides unchanged. `start + len` must not exceed the axis
    /// size ([`crate::Error::IndexOutOfBounds`]); `len == 0` is valid and
    /// yields an empty view.
    pub(crate) fn narrow(&self, axis: usize, start: usize, len: usize) -> Result<Layout> {
        let rank = self.rank();
        if axis >= rank {
            return Err(Error::InvalidAxis {
                op: "narrow",
                axis: axis as isize,
                rank,
            });
        }
        let size = self.shape.dims()[axis];
        // `start + len` is the exclusive end; it must land within the axis.
        let end = start.checked_add(len);
        if end.is_none_or(|end| end > size) {
            return Err(Error::IndexOutOfBounds {
                op: "narrow",
                index: end.map_or(i64::MAX, |end| end as i64),
                axis,
                size,
            });
        }
        let mut dims = self.shape.dims().to_vec();
        dims[axis] = len;
        // Advancing `start` steps along the axis shifts the origin by
        // `start * stride`. For a broadcast axis (stride 0) this is a no-op,
        // which is correct: the axis repeats a single element.
        let offset = self.offset + start * self.strides[axis];
        Ok(Layout {
            shape: Shape::from(dims),
            strides: self.strides.clone(),
            offset,
        })
    }

    /// Broadcast this view to `target` shape (NumPy rules, right-aligned;
    /// the target must be a valid broadcast of the current shape, else
    /// [`crate::Error::ShapeMismatch`]). Zero-copy: broadcast axes
    /// (including new leading axes) get stride 0; existing size-1 axes
    /// expanding to `n > 1` get stride 0.
    pub(crate) fn broadcast_to(&self, target: &Shape) -> Result<Layout> {
        let src = self.shape.dims();
        let dst = target.dims();
        if dst.len() < src.len() {
            return Err(Error::ShapeMismatch {
                op: "broadcast_to",
                lhs: self.shape.clone(),
                rhs: target.clone(),
            });
        }
        let pad = dst.len() - src.len();
        let mut strides = vec![0usize; dst.len()];
        // New leading axes (the right-alignment padding) keep stride 0.
        for i in 0..src.len() {
            let s = src[i];
            let d = dst[pad + i];
            if s == d {
                strides[pad + i] = self.strides[i];
            } else if s == 1 {
                // Size-1 axis expanding to `d`: repeat via stride 0.
                strides[pad + i] = 0;
            } else {
                return Err(Error::ShapeMismatch {
                    op: "broadcast_to",
                    lhs: self.shape.clone(),
                    rhs: target.clone(),
                });
            }
        }
        // Through `from_parts`, not a direct struct literal: the target shape is
        // caller-supplied, so its element count still has to be validated even
        // though every expanded axis has stride 0.
        Layout::from_parts(target.clone(), strides.into_boxed_slice(), self.offset)
    }

    /// Remove a size-1 axis (pre-resolved; the axis must have size 1, else
    /// [`crate::Error::InvalidArg`]). Zero-copy: drops the shape/stride
    /// entries.
    pub(crate) fn squeeze(&self, axis: usize) -> Result<Layout> {
        let rank = self.rank();
        if axis >= rank {
            return Err(Error::InvalidAxis {
                op: "squeeze",
                axis: axis as isize,
                rank,
            });
        }
        if self.shape.dims()[axis] != 1 {
            return Err(Error::InvalidArg {
                op: "squeeze",
                msg: format!(
                    "cannot squeeze axis {axis} of size {} (only size-1 axes)",
                    self.shape.dims()[axis]
                ),
            });
        }
        let mut dims = self.shape.dims().to_vec();
        let mut strides = self.strides.to_vec();
        dims.remove(axis);
        strides.remove(axis);
        Ok(Layout {
            shape: Shape::from(dims),
            strides: strides.into_boxed_slice(),
            offset: self.offset,
        })
    }

    /// Insert a size-1 axis at `axis` (pre-resolved insertion position in
    /// `[0, rank]`). Zero-copy; the inserted stride is chosen so the
    /// layout of a contiguous tensor stays contiguous.
    pub(crate) fn unsqueeze(&self, axis: usize) -> Result<Layout> {
        let rank = self.rank();
        if axis > rank {
            return Err(Error::InvalidAxis {
                op: "unsqueeze",
                axis: axis as isize,
                rank,
            });
        }
        // The new axis has size 1, so its stride never affects addressing.
        // To keep a contiguous input contiguous, the inserted stride must be
        // the canonical row-major stride at that position: the product of the
        // sizes to its right. For the axis currently at `axis` that is
        // `dims[axis] * strides[axis]`; appending at the end gives the unit
        // innermost stride.
        let inserted_stride = if axis < rank {
            self.shape.dims()[axis] * self.strides[axis]
        } else {
            1
        };
        let mut dims = self.shape.dims().to_vec();
        let mut strides = self.strides.to_vec();
        dims.insert(axis, 1);
        strides.insert(axis, inserted_stride);
        Ok(Layout {
            shape: Shape::from(dims),
            strides: strides.into_boxed_slice(),
            offset: self.offset,
        })
    }

    /// Attempt to view this layout as `new_shape` **without copying**
    /// (PyTorch `reshape` semantics): returns
    /// `Some(layout)` when the elements of the new shape can be addressed
    /// by some stride assignment over the existing storage walk order —
    /// always true for contiguous layouts; true for permuted/narrowed
    /// layouts only when merged/split axes remain stride-compatible.
    /// Returns `None` when a copy is required (the caller materializes via
    /// `copy_strided` and reshapes the copy). Element counts must already
    /// match (checked by the caller, which owns the
    /// [`crate::Error::ReshapeMismatch`] error).
    pub(crate) fn reshape_view(&self, new_shape: Shape) -> Option<Layout> {
        // A contiguous layout can always be reshaped by recomputing
        // row-major strides; take the fast path directly (this also covers
        // the common case cheaply).
        if self.is_contiguous() {
            // Contiguous by definition fits the addressable range for the
            // same element count, so `contiguous` cannot overflow here.
            return Layout::contiguous(new_shape).ok();
        }

        // Empty views carry no elements to address: any shape with the same
        // (zero) element count is a valid contiguous view at the same offset.
        if self.num_elements() == 0 {
            let mut layout = Layout::contiguous(new_shape).ok()?;
            layout.offset = self.offset;
            return Some(layout);
        }

        // The general non-contiguous case: PyTorch's `computeStride`. Walk
        // old axes right-to-left, coalescing each maximal run of axes that
        // are storage-contiguous with one another into a "chunk"; distribute
        // the new axes across chunks, assigning strides so a chunk is walked
        // in the same storage order. If the new axes cannot be partitioned to
        // line up with the chunk boundaries, a view is impossible.
        let old_dims = self.shape.dims();
        let old_strides: &[usize] = &self.strides;
        let new_dims = new_shape.dims();

        // Filter out size-1 old axes: their stride is irrelevant to the walk
        // order and PyTorch ignores them when detecting contiguous runs.
        let old: Vec<(usize, usize)> = old_dims
            .iter()
            .zip(old_strides.iter())
            .filter(|&(&d, _)| d != 1)
            .map(|(&d, &s)| (d, s))
            .collect();

        let mut new_strides = vec![0usize; new_dims.len()];

        let mut view_d = new_dims.len(); // one past the rightmost unassigned new axis
        let mut chunk_base_stride = old.last().map(|&(_, s)| s).unwrap_or(1);
        let mut tensor_numel = 1usize; // product of old dims in the current chunk
        let mut view_numel = 1usize; // product of new dims assigned to the chunk

        let mut oi = old.len();
        while oi > 0 {
            oi -= 1;
            let (od, _os) = old[oi];
            tensor_numel *= od;
            // Close the chunk at a break in contiguity, i.e. when the next
            // old axis to the left is not storage-adjacent to this chunk.
            let chunk_ends = oi == 0 || {
                let (_prev_d, prev_s) = old[oi - 1];
                prev_s != tensor_numel * chunk_base_stride
            };
            if chunk_ends {
                // Greedily assign new axes (right-to-left) into this chunk
                // until their product matches the chunk's element count.
                while view_d > 0 && (view_numel < tensor_numel || new_dims[view_d - 1] == 1) {
                    view_d -= 1;
                    new_strides[view_d] = view_numel * chunk_base_stride;
                    view_numel *= new_dims[view_d];
                }
                if view_numel != tensor_numel {
                    return None;
                }
                if oi > 0 {
                    // Start the next chunk to the left.
                    chunk_base_stride = old[oi - 1].1;
                    tensor_numel = 1;
                    view_numel = 1;
                }
            }
        }
        // Any leading new axes must be size-1 (they carry the innermost
        // chunk's stride). If unassigned axes remain, the view fails.
        while view_d > 0 && new_dims[view_d - 1] == 1 {
            view_d -= 1;
            new_strides[view_d] = view_numel * chunk_base_stride;
        }
        if view_d != 0 {
            return None;
        }

        Layout::from_parts(new_shape, new_strides.into_boxed_slice(), self.offset).ok()
    }
}

#[cfg(test)]
mod tests;
