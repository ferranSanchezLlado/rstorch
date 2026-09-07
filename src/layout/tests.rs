//! Property tests for the layout math: every constructor and view operation
//! against the normative semantics stated on each method.

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
