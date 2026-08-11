//! Indexing CPU kernels.
//!
//! Semantics on
//! [`BackendOps`](crate::backend::BackendOps): index values are bounds
//! checked ([`Error::IndexOutOfBounds`](crate::Error)), never UB;
//! `scatter_add` accumulates in `Acc`.
//!
//! # Design
//!
//! - **Stride-aware, nothing pre-materialized.** Each kernel addresses its
//!   source through [`Layout::strides`](crate::layout::Layout::strides) from
//!   [`Layout::offset`](crate::layout::Layout::offset), so transposed,
//!   narrowed, and broadcast inputs are read in place. Outputs are freshly
//!   allocated, dense, row-major buffers.
//! - **One coordinate walk.** All four kernels are the same loop: advance a
//!   row-major position, replace the indexed axis's coordinate with a
//!   looked-up index, and address the other side. The walk is the shared
//!   `super::host::Walk` odometer, which carries one running storage index per
//!   side and advances it by strides — so no division runs per element.
//!   [`place_values`] gives each axis of the (dense) output its row-major
//!   place value, which is that side's stride vector.
//! - **Whole rows at a time where the layout allows it.** When the axes
//!   *inside* the indexed one are contiguous in storage, one index selects a
//!   contiguous run of `Π dims[axis+1..]` elements, and that run is moved with
//!   a single slice copy instead of one decode per element — an embedding
//!   lookup or a `[batch, features]` gather becomes one `memcpy` per row.
//!   `index_select`/`index_add` take that path; `gather`/`scatter_add` cannot
//!   (their index grid picks per element, so the run is one element by
//!   construction) and only get the division-free walk.
//! - **Bounds first, work second.** The whole index buffer is read and
//!   validated into `usize` positions *before* any element is touched: a bad
//!   index is [`Error::IndexOutOfBounds`](crate::Error::IndexOutOfBounds)
//!   naming the offending value, never a partial write or an out-of-range
//!   read. Negative indices are rejected, not wrapped Python-style.
//! - **Accumulation in `Acc`.** `index_add`/`scatter_add` seed a wide
//!   [`Acc`](crate::dtype::Element::Acc) buffer from the base tensor,
//!   accumulate every contribution there, and narrow exactly once at output —
//!   so an `f16` embedding gradient with thousands of repeated rows does not
//!   saturate.

use super::cpu_storage;
use super::host::{Walk, dense_offset};
use crate::backend::View;
use crate::backend::cpu::acc::NumAcc;
use crate::backend::cpu::dispatch::{CpuElement, dispatch_all, dispatch_numeric};
use crate::device::Device;
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::shape::Shape;
use crate::storage::{CpuStorage, Storage};

// ---------------------------------------------------------------------------
// Shared plumbing
// ---------------------------------------------------------------------------

/// Row-major place values for `dims`: `place[a]` is the product of every
/// dimension to the right of axis `a` (1 for the innermost axis).
///
/// These are exactly the strides of a dense row-major buffer of `dims`, so a
/// kernel writing its output uses them as that side's stride vector.
fn place_values(dims: &[usize]) -> Vec<usize> {
    let mut place = vec![1usize; dims.len()];
    for a in (0..dims.len().saturating_sub(1)).rev() {
        place[a] = place[a + 1] * dims[a + 1];
    }
    place
}

/// Number of elements one index selects in a single slice copy: the product of
/// `dims[axis + 1..]` when those axes are contiguous in storage under
/// `strides`, else `None`.
///
/// Size-1 axes are skipped (their only coordinate is 0, so their stride never
/// contributes to an address), which is the same rule
/// `super::host::dense_offset` applies to a whole view. A broadcast axis
/// (stride 0, size > 1) inside the indexed one disqualifies the run.
fn trailing_run(dims: &[usize], strides: &[usize], axis: usize) -> Option<usize> {
    let mut run = 1usize;
    for a in (axis + 1..dims.len()).rev() {
        if dims[a] == 1 {
            continue;
        }
        if strides[a] != run {
            return None;
        }
        run = run.checked_mul(dims[a])?;
    }
    Some(run)
}

/// Read an index view in row-major order and bounds-check every value against
/// an axis of size `size`, yielding `usize` positions.
///
/// Reading and checking are one pass over one allocation: the index tensor is
/// small but this runs once per call, and a `DataLoader` batch is nothing but
/// these calls. The view may be strided (a transposed or narrowed index tensor
/// is legal); values are validated in the same logical order the kernels walk,
/// so the reported [`Error::IndexOutOfBounds`](crate::Error::IndexOutOfBounds)
/// is the first offending value in row-major order. A non-`I64` index tensor is
/// [`Error::DTypeMismatch`](crate::Error::DTypeMismatch) — indices are `I64` by
/// contract, never silently cast. Negative values are rejected, not wrapped
/// Python-style.
fn resolved_indices(
    indices: View<'_>,
    axis: usize,
    size: usize,
    op: &'static str,
) -> Result<Vec<usize>> {
    let cpu = cpu_storage(indices);
    let data = match cpu {
        CpuStorage::I64(v) => v.as_slice(),
        other => {
            return Err(Error::DTypeMismatch {
                op,
                expected: DType::I64,
                got: other.dtype(),
            });
        }
    };
    let layout = indices.layout();
    let total = layout.num_elements();
    let resolve = |v: i64| {
        let oob = || Error::IndexOutOfBounds {
            op,
            index: v,
            axis,
            size,
        };
        let pos = usize::try_from(v).map_err(|_| oob())?;
        if pos >= size { Err(oob()) } else { Ok(pos) }
    };
    if let Some(start) = dense_offset(layout) {
        return data[start..start + total]
            .iter()
            .map(|&v| resolve(v))
            .collect();
    }
    let mut walk = Walk::new(layout.dims(), [layout.strides()], [layout.offset()]);
    (0..total)
        .map(|_| {
            let v = data[walk.index(0)];
            walk.step();
            resolve(v)
        })
        .collect()
}

/// Guard a pre-resolved axis against the source rank. The op layer resolves
/// axes before dispatch, so this only fires on an internal contract break —
/// but a kernel must never index a stride slice out of range.
fn check_axis(op: &'static str, axis: usize, rank: usize) -> Result<()> {
    if axis >= rank {
        return Err(Error::InvalidAxis {
            op,
            axis: axis as isize,
            rank,
        });
    }
    Ok(())
}

/// Guard an expected rank.
fn check_rank(op: &'static str, expected: usize, got: usize) -> Result<()> {
    if expected == got {
        Ok(())
    } else {
        Err(Error::RankMismatch { op, expected, got })
    }
}

/// Guard that two operands share a dtype (no implicit promotion).
fn check_dtype(op: &'static str, expected: DType, got: DType) -> Result<()> {
    if expected == got {
        Ok(())
    } else {
        Err(Error::DTypeMismatch { op, expected, got })
    }
}

// ---------------------------------------------------------------------------
// index_select — slice-wise gather, 1-D index
// ---------------------------------------------------------------------------

/// `out[.., k, ..] = x[.., picks[k], ..]`, where `out_dims` is `x`'s shape
/// with `axis` resized to `picks.len()` and the output is dense row-major.
///
/// The output is produced in row-major order either way, so both paths write
/// the same elements in the same order — the fast path just moves a whole row
/// per index instead of decoding a coordinate per element.
fn index_select_generic<E: Copy>(
    data: &[E],
    x_strides: &[usize],
    x_offset: usize,
    out_dims: &[usize],
    axis: usize,
    picks: &[usize],
) -> Vec<E> {
    let n: usize = out_dims.iter().product();
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }
    let axis_stride = x_strides[axis];

    // Fast path: the axes inside `axis` are contiguous in the source, so each
    // pick is one slice copy of `run` elements. `out_dims` agrees with the
    // source dims on every axis but `axis`, which this run never spans.
    if let Some(run) = trailing_run(out_dims, x_strides, axis) {
        let outer = &out_dims[..axis];
        let mut walk = Walk::new(outer, [x_strides], [x_offset]);
        for _ in 0..outer.iter().product::<usize>() {
            let base = walk.index(0);
            for &pick in picks {
                let at = base + pick * axis_stride;
                out.extend_from_slice(&data[at..at + run]);
            }
            walk.step();
        }
        return out;
    }

    // General path: one element at a time, but the walk carries the source
    // index for every axis *except* the indexed one (stride zeroed there), so
    // the per-element work is one multiply-add.
    let mut strides = x_strides[..out_dims.len()].to_vec();
    strides[axis] = 0;
    let mut walk = Walk::new(out_dims, [&strides], [x_offset]);
    for _ in 0..n {
        out.push(data[walk.index(0) + picks[walk.coord(axis)] * axis_stride]);
        walk.step();
    }
    out
}

/// See [`BackendOps::index_select`](crate::backend::BackendOps::index_select).
pub(crate) fn index_select(x: View<'_>, axis: usize, indices: View<'_>) -> Result<Storage> {
    const OP: &str = "index_select";
    let layout = x.layout();
    check_axis(OP, axis, layout.rank())?;
    check_rank(OP, 1, indices.layout().rank())?;
    let picks = resolved_indices(indices, axis, layout.dims()[axis], OP)?;

    let mut out_dims = layout.dims().to_vec();
    out_dims[axis] = picks.len();
    let strides = layout.strides();
    let offset = layout.offset();

    let values = cpu_storage(x);
    Ok(dispatch_all!(x.dtype(), E => {
        E::storage(index_select_generic(
            E::slice(values), strides, offset, &out_dims, axis, &picks,
        ))
    }))
}

// ---------------------------------------------------------------------------
// gather — per-element gather, same-rank index grid
// ---------------------------------------------------------------------------

/// `out[c0, .., c_axis, ..] = x[c0, .., picks[flat], ..]`, where `picks` is
/// the same-rank index grid read row-major and `out_dims` is its shape
/// (PyTorch `gather` semantics).
fn gather_generic<E: Copy>(
    data: &[E],
    x_strides: &[usize],
    x_offset: usize,
    out_dims: &[usize],
    axis: usize,
    picks: &[usize],
) -> Vec<E> {
    // One index per output element, by construction: `out_dims` *is* the
    // index grid's shape.
    debug_assert_eq!(picks.len(), out_dims.iter().product::<usize>());
    let mut out = Vec::with_capacity(picks.len());
    if picks.is_empty() {
        return out;
    }
    // There is no row to copy here — consecutive outputs along the innermost
    // axis carry different indices — so this is the division-free walk only,
    // with the indexed axis's stride zeroed out of the walk and applied from
    // the index grid instead.
    let axis_stride = x_strides[axis];
    let mut strides = x_strides[..out_dims.len()].to_vec();
    strides[axis] = 0;
    let mut walk = Walk::new(out_dims, [&strides], [x_offset]);
    for &pick in picks {
        out.push(data[walk.index(0) + pick * axis_stride]);
        walk.step();
    }
    out
}

/// See [`BackendOps::gather`](crate::backend::BackendOps::gather).
pub(crate) fn gather(x: View<'_>, axis: usize, indices: View<'_>) -> Result<Storage> {
    const OP: &str = "gather";
    let layout = x.layout();
    let idx_layout = indices.layout();
    check_axis(OP, axis, layout.rank())?;
    check_rank(OP, layout.rank(), idx_layout.rank())?;
    // PyTorch's rule: the index grid may be smaller than the source on every
    // axis other than the gathered one, never larger.
    for (a, (&i, &s)) in idx_layout.dims().iter().zip(layout.dims()).enumerate() {
        if a != axis && i > s {
            return Err(Error::ShapeMismatch {
                op: OP,
                lhs: layout.shape().clone(),
                rhs: idx_layout.shape().clone(),
            });
        }
    }
    let picks = resolved_indices(indices, axis, layout.dims()[axis], OP)?;

    let out_dims = idx_layout.dims();
    let strides = layout.strides();
    let offset = layout.offset();
    let values = cpu_storage(x);
    Ok(dispatch_all!(x.dtype(), E => {
        E::storage(gather_generic(
            E::slice(values), strides, offset, out_dims, axis, &picks,
        ))
    }))
}

// ---------------------------------------------------------------------------
// Accumulating kernels (index_add / scatter_add)
// ---------------------------------------------------------------------------

/// Seed the wide accumulator buffer from the base tensor, in row-major order
/// over its logical shape. Shared first step of both accumulating kernels.
///
/// A dense base (the common case: `index_add`'s base is usually a freshly
/// allocated zero tensor) widens straight off the slice; anything strided goes
/// through the division-free walk.
fn seed_acc<E>(x_data: &[E], x_layout: &Layout) -> Vec<E::Acc>
where
    E: Element,
{
    let total = x_layout.num_elements();
    if let Some(start) = dense_offset(x_layout) {
        return x_data[start..start + total]
            .iter()
            .map(|&e| e.to_acc())
            .collect();
    }
    let mut walk = Walk::new(x_layout.dims(), [x_layout.strides()], [x_layout.offset()]);
    (0..total)
        .map(|_| {
            let v = x_data[walk.index(0)].to_acc();
            walk.step();
            v
        })
        .collect()
}

/// `out = x; out[.., picks[k], ..] += src[.., k, ..]` — the slice-wise,
/// 1-D-index accumulate that backs `index_select`'s backward. Repeated
/// positions in `picks` accumulate (the embedding backward's whole point).
/// Both paths add the same contributions to each accumulator cell in the same
/// order (outer coordinates, then index, then position within the row), so the
/// result is bitwise identical whichever runs.
fn index_add_generic<E>(
    x_data: &[E],
    x_layout: &Layout,
    src_data: &[E],
    src_layout: &Layout,
    axis: usize,
    picks: &[usize],
) -> Vec<E>
where
    E: Element,
    E::Acc: NumAcc,
{
    let x_place = place_values(x_layout.dims());
    let mut acc = seed_acc(x_data, x_layout);

    let src_dims = src_layout.dims();
    let src_strides = src_layout.strides();
    let total = src_layout.num_elements();
    if total > 0 {
        // Fast path: the axes inside `axis` are contiguous in the source. They
        // are contiguous in the (dense) destination by construction and, since
        // `src` has the base's dims off `axis`, both runs are the same length —
        // so one index adds one row to one row, index-free on both sides.
        if let Some(run) = trailing_run(src_dims, src_strides, axis) {
            let outer = &src_dims[..axis];
            let mut walk = Walk::new(outer, [src_strides, &x_place], [src_layout.offset(), 0]);
            for _ in 0..outer.iter().product::<usize>() {
                let (from_base, dst_base) = (walk.index(0), walk.index(1));
                for (k, &pick) in picks.iter().enumerate() {
                    let from = from_base + k * src_strides[axis];
                    let dst = dst_base + pick * x_place[axis];
                    for (a, &s) in acc[dst..dst + run]
                        .iter_mut()
                        .zip(&src_data[from..from + run])
                    {
                        *a = a.add(s.to_acc());
                    }
                }
                walk.step();
            }
            return acc.into_iter().map(E::from_acc).collect();
        }

        // General path: one element at a time, with the destination stride of
        // the indexed axis zeroed out of the walk and applied from `picks`.
        let mut dst_strides = x_place.clone();
        dst_strides[axis] = 0;
        let mut walk = Walk::new(
            src_dims,
            [src_strides, &dst_strides],
            [src_layout.offset(), 0],
        );
        for _ in 0..total {
            let dst = walk.index(1) + picks[walk.coord(axis)] * x_place[axis];
            acc[dst] = acc[dst].add(src_data[walk.index(0)].to_acc());
            walk.step();
        }
    }

    acc.into_iter().map(E::from_acc).collect()
}

/// `out = x; out[c0, .., picks[flat], ..] += src[c0, .., c_axis, ..]` — the
/// per-element, same-rank-index accumulate that backs `gather`'s backward.
/// Duplicate destinations accumulate.
fn scatter_add_generic<E>(
    x_data: &[E],
    x_layout: &Layout,
    src_data: &[E],
    src_layout: &Layout,
    idx_dims: &[usize],
    axis: usize,
    picks: &[usize],
) -> Vec<E>
where
    E: Element,
    E::Acc: NumAcc,
{
    let x_place = place_values(x_layout.dims());
    let mut acc = seed_acc(x_data, x_layout);

    // One index per grid position, by construction.
    debug_assert_eq!(picks.len(), idx_dims.iter().product::<usize>());
    if !picks.is_empty() {
        // Per-element by nature (see the module header): the walk carries the
        // source and destination indices for every axis but the scattered one.
        let mut dst_strides = x_place.clone();
        dst_strides[axis] = 0;
        let src_strides = src_layout.strides();
        let mut walk = Walk::new(
            idx_dims,
            [src_strides, &dst_strides],
            [src_layout.offset(), 0],
        );
        for &pick in picks {
            let dst = walk.index(1) + pick * x_place[axis];
            acc[dst] = acc[dst].add(src_data[walk.index(0)].to_acc());
            walk.step();
        }
    }

    acc.into_iter().map(E::from_acc).collect()
}

/// The dtype-generic body of an accumulating kernel, so [`dispatch_acc`] can
/// monomorphize it once per element type. A closure cannot carry the `for<E>`
/// quantification this needs, hence a trait with a generic method.
trait AccKernel {
    /// Run the kernel over the typed base and source slices.
    fn run<E>(&self, x: &[E], src: &[E]) -> Vec<E>
    where
        E: Element,
        E::Acc: NumAcc;
}

/// [`index_add`]'s body as an [`AccKernel`].
struct IndexAddKernel<'a> {
    x_layout: &'a Layout,
    src_layout: &'a Layout,
    axis: usize,
    picks: &'a [usize],
}

impl AccKernel for IndexAddKernel<'_> {
    fn run<E>(&self, x: &[E], src: &[E]) -> Vec<E>
    where
        E: Element,
        E::Acc: NumAcc,
    {
        index_add_generic(
            x,
            self.x_layout,
            src,
            self.src_layout,
            self.axis,
            self.picks,
        )
    }
}

/// [`scatter_add`]'s body as an [`AccKernel`].
struct ScatterAddKernel<'a> {
    x_layout: &'a Layout,
    src_layout: &'a Layout,
    idx_dims: &'a [usize],
    axis: usize,
    picks: &'a [usize],
}

impl AccKernel for ScatterAddKernel<'_> {
    fn run<E>(&self, x: &[E], src: &[E]) -> Vec<E>
    where
        E: Element,
        E::Acc: NumAcc,
    {
        scatter_add_generic(
            x,
            self.x_layout,
            src,
            self.src_layout,
            self.idx_dims,
            self.axis,
            self.picks,
        )
    }
}

/// Route an accumulating kernel over the base/source storage pair (which
/// share a dtype, checked by the caller).
///
/// `Bool` has no wide accumulator, so accumulating into a boolean tensor is
/// [`Error::Unsupported`](crate::Error::Unsupported) rather than an invented
/// "or" semantics.
fn dispatch_acc<K: AccKernel>(
    op: &'static str,
    device: Device,
    x: &CpuStorage,
    src: &CpuStorage,
    kernel: K,
) -> Result<Storage> {
    dispatch_numeric!(x.dtype(), op, device, E => {
        Ok(E::storage(kernel.run(E::slice(x), E::slice(src))))
    })
}

/// See [`BackendOps::index_add`](crate::backend::BackendOps::index_add).
pub(crate) fn index_add(
    x: View<'_>,
    axis: usize,
    indices: View<'_>,
    src: View<'_>,
) -> Result<Storage> {
    const OP: &str = "index_add";
    let x_layout = x.layout();
    let src_layout = src.layout();
    check_axis(OP, axis, x_layout.rank())?;
    check_rank(OP, 1, indices.layout().rank())?;
    check_dtype(OP, x.dtype(), src.dtype())?;
    let picks = resolved_indices(indices, axis, x_layout.dims()[axis], OP)?;

    // `src` must be `x`'s shape with the indexed axis resized to the index
    // count: whole slices, one per index (PyTorch `index_add` semantics).
    let mut want = x_layout.dims().to_vec();
    want[axis] = picks.len();
    if src_layout.dims() != want.as_slice() {
        return Err(Error::ShapeMismatch {
            op: OP,
            lhs: Shape::from(want),
            rhs: src_layout.shape().clone(),
        });
    }

    let x_cpu = cpu_storage(x);
    let src_cpu = cpu_storage(src);
    dispatch_acc(
        OP,
        x.device(),
        x_cpu,
        src_cpu,
        IndexAddKernel {
            x_layout,
            src_layout,
            axis,
            picks: &picks,
        },
    )
}

/// See [`BackendOps::scatter_add`](crate::backend::BackendOps::scatter_add).
pub(crate) fn scatter_add(
    x: View<'_>,
    axis: usize,
    indices: View<'_>,
    src: View<'_>,
) -> Result<Storage> {
    const OP: &str = "scatter_add";
    let x_layout = x.layout();
    let idx_layout = indices.layout();
    let src_layout = src.layout();
    check_axis(OP, axis, x_layout.rank())?;
    check_rank(OP, x_layout.rank(), idx_layout.rank())?;
    check_rank(OP, x_layout.rank(), src_layout.rank())?;
    check_dtype(OP, x.dtype(), src.dtype())?;

    let idx_dims = idx_layout.dims();
    // The index grid selects from `src` position-for-position, so it may not
    // exceed `src` on any axis, nor `x` on any axis other than the scattered
    // one (PyTorch `scatter_add_` shape rule).
    for (&i, &s) in idx_dims.iter().zip(src_layout.dims()) {
        if i > s {
            return Err(Error::ShapeMismatch {
                op: OP,
                lhs: src_layout.shape().clone(),
                rhs: idx_layout.shape().clone(),
            });
        }
    }
    for (a, (&i, &s)) in idx_dims.iter().zip(x_layout.dims()).enumerate() {
        if a != axis && i > s {
            return Err(Error::ShapeMismatch {
                op: OP,
                lhs: x_layout.shape().clone(),
                rhs: idx_layout.shape().clone(),
            });
        }
    }

    let picks = resolved_indices(indices, axis, x_layout.dims()[axis], OP)?;

    let x_cpu = cpu_storage(x);
    let src_cpu = cpu_storage(src);
    dispatch_acc(
        OP,
        x.device(),
        x_cpu,
        src_cpu,
        ScatterAddKernel {
            x_layout,
            src_layout,
            idx_dims,
            axis,
            picks: &picks,
        },
    )
}

#[cfg(test)]
mod tests;
