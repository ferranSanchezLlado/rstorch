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
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Reference model: a layout is a function from logical coordinates to a
    // storage index. Every property test compares the layout's declared
    // (strides, offset) against a from-scratch enumeration of that mapping.
    // ------------------------------------------------------------------

    /// The storage index of the logical element at `coords` under `layout`.
    fn storage_index(layout: &Layout, coords: &[usize]) -> usize {
        let mut idx = layout.offset();
        for (c, s) in coords.iter().zip(layout.strides().iter()) {
            idx += c * s;
        }
        idx
    }

    /// Enumerate every logical coordinate of `dims` in row-major order.
    fn all_coords(dims: &[usize]) -> Vec<Vec<usize>> {
        let total: usize = dims.iter().product();
        if total == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(total);
        let mut coords = vec![0usize; dims.len()];
        loop {
            out.push(coords.clone());
            // Increment like an odometer, rightmost axis fastest.
            let mut axis = dims.len();
            loop {
                if axis == 0 {
                    return out;
                }
                axis -= 1;
                coords[axis] += 1;
                if coords[axis] < dims[axis] {
                    break;
                }
                coords[axis] = 0;
            }
        }
    }

    /// The full row-major storage walk of `layout` (index per logical
    /// element, in row-major logical order).
    fn walk(layout: &Layout) -> Vec<usize> {
        all_coords(layout.dims())
            .iter()
            .map(|c| storage_index(layout, c))
            .collect()
    }

    /// A tiny seeded xorshift PRNG so property tests are reproducible with no
    /// external crate.
    struct Prng(u64);
    impl Prng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    // ------------------------------------------------------------------
    // contiguous / from_parts / is_contiguous
    // ------------------------------------------------------------------

    #[test]
    fn contiguous_row_major_strides() {
        let l = Layout::contiguous([2, 3, 4]).unwrap();
        assert_eq!(l.strides(), &[12, 4, 1]);
        assert_eq!(l.offset(), 0);
        assert!(l.is_contiguous());
        // Scalar.
        let s = Layout::contiguous(()).unwrap();
        assert_eq!(s.rank(), 0);
        assert!(s.is_contiguous());
        assert_eq!(s.num_elements(), 1);
    }

    #[test]
    fn contiguous_overflow_is_invalid_arg() {
        let big = usize::MAX;
        let err = Layout::contiguous([big, big]).unwrap_err();
        assert!(matches!(err, Error::InvalidArg { op: "layout", .. }));
    }

    /// A broadcast layout has stride 0 on every expanded axis, so its highest
    /// reachable index cannot detect an overflowing element count. Without a
    /// separate check, `broadcast_to([usize::MAX, usize::MAX])` yielded a layout
    /// that reported those dims while `num_elements()` wrapped — self-
    /// inconsistent in release, and a panic from a plain getter in debug.
    #[test]
    fn broadcast_to_an_overflowing_target_is_invalid_arg() {
        let base = Layout::contiguous([1, 1]).unwrap();
        let err = base
            .broadcast_to(&Shape::from([usize::MAX, usize::MAX]))
            .unwrap_err();
        assert!(
            matches!(err, Error::InvalidArg { op: "layout", .. }),
            "{err:?}"
        );
        // An ordinary broadcast still works, and stays consistent.
        let ok = base.broadcast_to(&Shape::from([4, 3])).unwrap();
        assert_eq!(ok.dims(), &[4, 3]);
        assert_eq!(ok.num_elements(), 12);
        assert_eq!(ok.strides(), &[0, 0]);
    }

    #[test]
    fn from_parts_validates_rank_and_bounds() {
        assert!(matches!(
            Layout::from_parts(Shape::from([2, 3]), vec![3].into_boxed_slice(), 0),
            Err(Error::InvalidArg { .. })
        ));
        // Max index = offset + (2-1)*3 + (2-1)*1 = 4, needs len >= 5. This
        // just validates addressability, not against a concrete storage.
        assert!(Layout::from_parts(Shape::from([2, 2]), vec![3, 1].into_boxed_slice(), 0).is_ok());
        // Overflow.
        assert!(matches!(
            Layout::from_parts(
                Shape::from([2, 2]),
                vec![usize::MAX, 1].into_boxed_slice(),
                0
            ),
            Err(Error::InvalidArg { .. })
        ));
        // Zero-element views never overflow regardless of strides.
        assert!(
            Layout::from_parts(
                Shape::from([0, 2]),
                vec![usize::MAX, 1].into_boxed_slice(),
                0
            )
            .is_ok()
        );
    }

    #[test]
    fn is_contiguous_classification() {
        // Canonical.
        assert!(Layout::contiguous([2, 3]).unwrap().is_contiguous());
        // Non-zero offset.
        assert!(
            !Layout::from_parts(Shape::from([2, 3]), vec![3, 1].into_boxed_slice(), 1)
                .unwrap()
                .is_contiguous()
        );
        // Transposed strides are not contiguous.
        assert!(
            !Layout::from_parts(Shape::from([3, 2]), vec![1, 3].into_boxed_slice(), 0)
                .unwrap()
                .is_contiguous()
        );
        // Size-1 axes must carry the canonical stride.
        assert!(Layout::contiguous([2, 1, 3]).unwrap().is_contiguous());
        // A broadcast (stride-0) axis is not contiguous.
        assert!(
            !Layout::from_parts(Shape::from([2, 3]), vec![0, 1].into_boxed_slice(), 0)
                .unwrap()
                .is_contiguous()
        );
    }

    // ------------------------------------------------------------------
    // transpose / permute
    // ------------------------------------------------------------------

    #[test]
    fn transpose_swaps_axes_zero_copy() {
        let l = Layout::contiguous([2, 3, 4]).unwrap();
        let t = l.transpose(0, 2).unwrap();
        assert_eq!(t.dims(), &[4, 3, 2]);
        assert_eq!(t.strides(), &[1, 4, 12]);
        assert_eq!(t.offset(), l.offset());
        assert!(matches!(l.transpose(0, 3), Err(Error::InvalidAxis { .. })));
    }

    #[test]
    fn permute_is_a_reorder_of_transposes() {
        let dims = [2usize, 3, 4, 5];
        let l = Layout::contiguous(dims).unwrap();
        // permute must reproduce the coordinate->storage mapping under axis
        // relabeling for every permutation of a rank-4 shape.
        let perms = all_permutations(4);
        for perm in perms {
            let p = l.permute(&perm).unwrap();
            assert_eq!(p.offset(), l.offset());
            let expected_dims: Vec<usize> = perm.iter().map(|&i| dims[i]).collect();
            assert_eq!(p.dims(), expected_dims.as_slice());
            // For each permuted coordinate, the storage index equals the base
            // layout's index at the un-permuted coordinate.
            for coords in all_coords(p.dims()) {
                let mut base_coords = vec![0usize; 4];
                for (new_axis, &old_axis) in perm.iter().enumerate() {
                    base_coords[old_axis] = coords[new_axis];
                }
                assert_eq!(storage_index(&p, &coords), storage_index(&l, &base_coords));
            }
        }
    }

    #[test]
    fn permute_rejects_bad_permutations() {
        let l = Layout::contiguous([2, 3, 4]).unwrap();
        assert!(matches!(l.permute(&[0, 1]), Err(Error::InvalidArg { .. })));
        assert!(matches!(
            l.permute(&[0, 1, 3]),
            Err(Error::InvalidArg { .. })
        ));
        assert!(matches!(
            l.permute(&[0, 0, 1]),
            Err(Error::InvalidArg { .. })
        ));
    }

    fn all_permutations(n: usize) -> Vec<Vec<usize>> {
        fn recurse(remaining: &[usize], acc: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
            if remaining.is_empty() {
                out.push(acc.clone());
                return;
            }
            for i in 0..remaining.len() {
                let mut rest = remaining.to_vec();
                let v = rest.remove(i);
                acc.push(v);
                recurse(&rest, acc, out);
                acc.pop();
            }
        }
        let mut out = Vec::new();
        recurse(&(0..n).collect::<Vec<_>>(), &mut Vec::new(), &mut out);
        out
    }

    // ------------------------------------------------------------------
    // narrow
    // ------------------------------------------------------------------

    #[test]
    fn narrow_shifts_offset_and_shrinks_axis() {
        let l = Layout::contiguous([4, 5]).unwrap();
        let n = l.narrow(1, 1, 3).unwrap();
        assert_eq!(n.dims(), &[4, 3]);
        assert_eq!(n.strides(), l.strides());
        assert_eq!(n.offset(), 1); // 1 * stride[1] = 1
        // Every element of the narrowed view points at the base element at
        // the shifted coordinate.
        for coords in all_coords(n.dims()) {
            let base_coords = vec![coords[0], coords[1] + 1];
            assert_eq!(storage_index(&n, &coords), storage_index(&l, &base_coords));
        }
    }

    #[test]
    fn narrow_empty_and_bounds() {
        let l = Layout::contiguous([4, 5]).unwrap();
        // len == 0 yields an empty view.
        let e = l.narrow(0, 2, 0).unwrap();
        assert_eq!(e.dims(), &[0, 5]);
        assert_eq!(e.num_elements(), 0);
        // Out of bounds.
        assert!(matches!(
            l.narrow(0, 3, 2),
            Err(Error::IndexOutOfBounds { op: "narrow", .. })
        ));
        assert!(matches!(l.narrow(2, 0, 1), Err(Error::InvalidAxis { .. })));
        // Overflowing start+len.
        assert!(matches!(
            l.narrow(0, usize::MAX, 1),
            Err(Error::IndexOutOfBounds { .. })
        ));
    }

    // ------------------------------------------------------------------
    // broadcast_to
    // ------------------------------------------------------------------

    #[test]
    fn broadcast_stride_zero_for_expanded_axes() {
        let l = Layout::contiguous([1, 3]).unwrap(); // strides [3, 1]
        let b = l.broadcast_to(&Shape::from([2, 4, 3])).unwrap();
        assert_eq!(b.dims(), &[2, 4, 3]);
        // Leading new axis -> 0; expanded size-1 axis -> 0; matching axis
        // keeps its stride.
        assert_eq!(b.strides(), &[0, 0, 1]);
        // Broadcasting repeats: every leading/expanded coordinate maps to the
        // same base element.
        for coords in all_coords(b.dims()) {
            let base = vec![0usize, coords[2]];
            assert_eq!(storage_index(&b, &coords), storage_index(&l, &base));
        }
    }

    #[test]
    fn broadcast_rejects_incompatible() {
        let l = Layout::contiguous([3, 2]).unwrap();
        // 3 cannot broadcast to 4.
        assert!(matches!(
            l.broadcast_to(&Shape::from([4, 2])),
            Err(Error::ShapeMismatch {
                op: "broadcast_to",
                ..
            })
        ));
        // Target of lower rank is invalid.
        assert!(matches!(
            l.broadcast_to(&Shape::from([2])),
            Err(Error::ShapeMismatch { .. })
        ));
    }

    // ------------------------------------------------------------------
    // squeeze / unsqueeze
    // ------------------------------------------------------------------

    #[test]
    fn squeeze_unsqueeze_round_trip() {
        let l = Layout::contiguous([2, 3]).unwrap();
        let u = l.unsqueeze(1).unwrap();
        assert_eq!(u.dims(), &[2, 1, 3]);
        assert!(u.is_contiguous());
        let s = u.squeeze(1).unwrap();
        assert_eq!(s, l);
        // Append position.
        let end = l.unsqueeze(2).unwrap();
        assert_eq!(end.dims(), &[2, 3, 1]);
        assert!(end.is_contiguous());
    }

    #[test]
    fn squeeze_errors() {
        let l = Layout::contiguous([2, 3]).unwrap();
        assert!(matches!(l.squeeze(0), Err(Error::InvalidArg { .. })));
        assert!(matches!(l.squeeze(5), Err(Error::InvalidAxis { .. })));
        assert!(matches!(l.unsqueeze(3), Err(Error::InvalidAxis { .. })));
    }

    // ------------------------------------------------------------------
    // reshape_view: view-or-copy decision + correctness
    // ------------------------------------------------------------------

    #[test]
    fn reshape_contiguous_always_views() {
        let l = Layout::contiguous([2, 3, 4]).unwrap();
        let r = l.reshape_view(Shape::from([6, 4])).unwrap();
        assert!(r.is_contiguous());
        assert_eq!(r.strides(), &[4, 1]);
        // Splitting an axis.
        let r = l.reshape_view(Shape::from([2, 2, 2, 3])).unwrap();
        assert_eq!(r.strides(), &[12, 6, 3, 1]);
        // The flattened walk order matches.
        assert_eq!(walk(&l), walk(&r));
    }

    #[test]
    fn reshape_transposed_merge_needs_copy() {
        // Transpose makes the layout non-contiguous; merging the two axes is
        // not expressible as a stride view.
        let l = Layout::contiguous([2, 3]).unwrap().transpose(0, 1).unwrap();
        assert_eq!(l.dims(), &[3, 2]);
        assert!(l.reshape_view(Shape::from([6])).is_none());
    }

    #[test]
    fn reshape_transposed_split_can_view() {
        // A non-contiguous layout can still be viewed when the new axes line
        // up with contiguous runs: splitting an axis that is itself a
        // contiguous run keeps a valid view.
        let l = Layout::contiguous([4, 6]).unwrap().transpose(0, 1).unwrap();
        // dims [6, 4], strides [1, 6]. Splitting the size-4 axis (stride 6,
        // contiguous run of the original inner axis) into [2, 2] is a view.
        let r = l.reshape_view(Shape::from([6, 2, 2])).unwrap();
        // The walk order must be identical to the source's.
        assert_eq!(walk(&l), walk(&r));
    }

    #[test]
    fn reshape_size_one_axes_are_free() {
        let l = Layout::contiguous([2, 3]).unwrap().transpose(0, 1).unwrap();
        // Inserting/removing size-1 axes never forces a copy.
        let r = l.reshape_view(Shape::from([1, 3, 1, 2, 1])).unwrap();
        assert_eq!(walk(&l), walk(&r));
    }

    // ------------------------------------------------------------------
    // Exhaustive small-shape property tests: every view op preserves the
    // coordinate->storage mapping, and reshape_view (when it returns Some)
    // reproduces the source's storage walk order exactly.
    // ------------------------------------------------------------------

    /// All shapes with `rank` axes, each dim in `1..=max_dim`.
    fn all_shapes(rank: usize, max_dim: usize) -> Vec<Vec<usize>> {
        if rank == 0 {
            return vec![Vec::new()];
        }
        let mut out = Vec::new();
        for tail in all_shapes(rank - 1, max_dim) {
            for d in 1..=max_dim {
                let mut s = vec![d];
                s.extend_from_slice(&tail);
                out.push(s);
            }
        }
        out
    }

    #[test]
    fn exhaustive_reshape_view_walk_order_holds() {
        // For a range of small contiguous / permuted / narrowed source
        // layouts, whenever reshape_view returns a view, its storage walk
        // must equal the source's walk (same elements in the same order).
        for rank in 1..=3usize {
            for dims in all_shapes(rank, 3) {
                let base = Layout::contiguous(dims.clone()).unwrap();
                let numel: usize = dims.iter().product();
                if numel == 0 {
                    continue;
                }

                // Build several source views over the same storage.
                let mut sources = vec![base.clone()];
                if rank >= 2 {
                    sources.push(base.transpose(0, rank - 1).unwrap());
                }
                if rank >= 3 {
                    sources.push(base.permute(&[2, 0, 1]).unwrap());
                }
                // A narrowed view of the leading axis.
                if dims[0] >= 2 {
                    sources.push(base.narrow(0, 0, dims[0] - 1).unwrap());
                }

                for src in &sources {
                    let src_numel = src.num_elements();
                    let src_walk = walk(src);
                    // Try every same-element-count target of rank 1..=3.
                    for tgt_rank in 1..=3usize {
                        for tgt in all_shapes(tgt_rank, 6) {
                            let tnumel: usize = tgt.iter().product();
                            if tnumel != src_numel {
                                continue;
                            }
                            if let Some(view) = src.reshape_view(Shape::from(tgt.clone())) {
                                assert_eq!(view.dims(), tgt.as_slice());
                                assert_eq!(view.offset(), src.offset());
                                assert_eq!(
                                    walk(&view),
                                    src_walk,
                                    "reshape {:?} -> {:?} produced a view with a \
                                     different walk order (src strides {:?} off {})",
                                    src.dims(),
                                    tgt,
                                    src.strides(),
                                    src.offset(),
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn reshape_view_contiguous_target_is_never_none() {
        // A contiguous source can ALWAYS be viewed to any same-numel shape.
        for rank in 1..=3usize {
            for dims in all_shapes(rank, 4) {
                let base = Layout::contiguous(dims.clone()).unwrap();
                let numel: usize = dims.iter().product();
                if numel == 0 {
                    continue;
                }
                for tgt_rank in 1..=3usize {
                    for tgt in all_shapes(tgt_rank, 8) {
                        let tnumel: usize = tgt.iter().product();
                        if tnumel != numel {
                            continue;
                        }
                        let view = base
                            .reshape_view(Shape::from(tgt.clone()))
                            .expect("contiguous source must always view");
                        assert!(view.is_contiguous());
                    }
                }
            }
        }
    }

    #[test]
    fn seeded_random_views_preserve_mapping() {
        // Randomised stress: build random strided source layouts (via
        // sequences of transpose/narrow/unsqueeze on a contiguous base),
        // then check that every op's declared strides reproduce the naive
        // coordinate->storage mapping, and reshape views keep the walk order.
        let mut rng = Prng(0x9E3779B97F4A7C15);
        for _ in 0..2000 {
            let rank = 1 + rng.below(4);
            let dims: Vec<usize> = (0..rank).map(|_| 1 + rng.below(4)).collect();
            let mut layout = Layout::contiguous(dims.clone()).unwrap();

            // Apply a random sequence of zero-copy view ops.
            let steps = rng.below(4);
            for _ in 0..steps {
                let r = layout.rank();
                match rng.below(4) {
                    0 if r >= 2 => {
                        let a = rng.below(r);
                        let b = rng.below(r);
                        layout = layout.transpose(a, b).unwrap();
                    }
                    1 => {
                        let axis = rng.below(layout.rank());
                        let size = layout.dims()[axis];
                        if size >= 1 {
                            let start = rng.below(size + 1).min(size);
                            let maxlen = size - start;
                            let len = if maxlen == 0 {
                                0
                            } else {
                                rng.below(maxlen + 1)
                            };
                            layout = layout.narrow(axis, start, len).unwrap();
                        }
                    }
                    2 => {
                        let pos = rng.below(layout.rank() + 1);
                        layout = layout.unsqueeze(pos).unwrap();
                    }
                    _ => {
                        // Broadcast a random size-1 axis (or a fresh leading
                        // axis) to a small size.
                        let cur = layout.shape().clone();
                        let mut tgt = cur.dims().to_vec();
                        // Expand any size-1 axis.
                        let mut changed = false;
                        for d in tgt.iter_mut() {
                            if *d == 1 && rng.below(2) == 0 {
                                *d = 1 + rng.below(3);
                                changed = true;
                            }
                        }
                        if changed {
                            layout = layout.broadcast_to(&Shape::from(tgt)).unwrap();
                        }
                    }
                }
            }

            // Invariant 1: strides length == rank; num_elements consistent.
            assert_eq!(layout.strides().len(), layout.rank());
            let expected_numel: usize = layout.dims().iter().product();
            assert_eq!(layout.num_elements(), expected_numel);

            // Invariant 2: a reshape view, when granted, walks identically.
            let numel = layout.num_elements();
            if numel == 0 || numel > 64 {
                continue;
            }
            let src_walk = walk(&layout);
            // Try a couple of random same-numel targets.
            for _ in 0..4 {
                let divisors: Vec<usize> =
                    (1..=numel).filter(|&d| numel.is_multiple_of(d)).collect();
                let a = divisors[rng.below(divisors.len())];
                let b = numel / a;
                let tgt = Shape::from([a, b]);
                if let Some(view) = layout.reshape_view(tgt) {
                    assert_eq!(view.num_elements(), numel);
                    assert_eq!(walk(&view), src_walk);
                }
            }
        }
    }
}
