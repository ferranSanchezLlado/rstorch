//! Indexing CPU kernels.
//!
//! Signatures frozen by T01; **T25** fills the bodies. Semantics on
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
//!   saturate (the v2 sum-saturation bug, fixed by contract).

use super::host::{Walk, dense_offset};
use crate::backend::View;
use crate::backend::cpu::acc::NumAcc;
use crate::device::Device;
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::shape::Shape;
use crate::storage::{CpuStorage, Storage};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Shared plumbing
// ---------------------------------------------------------------------------

/// Borrow the [`CpuStorage`] behind a CPU view, or report the op as
/// unsupported on a non-CPU device (a Metal view never reaches a CPU kernel
/// in practice; this keeps the match total without an `unimplemented!`).
// `op` is only read by the `metal`-gated arm; on a CPU-only build it is unused.
#[cfg_attr(
    not(all(feature = "metal", target_os = "macos")),
    allow(unused_variables)
)]
fn cpu_storage<'a>(x: View<'a>, op: &'static str) -> Result<&'a CpuStorage> {
    match x.storage() {
        Storage::Cpu(s) => Ok(s),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Storage::Metal(_) => Err(Error::Unsupported {
            op,
            device: x.device(),
            dtype: x.dtype(),
        }),
    }
}

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
    let cpu = cpu_storage(indices, op)?;
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

    let out = match cpu_storage(x, OP)? {
        CpuStorage::F16(v) => CpuStorage::F16(Arc::new(index_select_generic(
            v, strides, offset, &out_dims, axis, &picks,
        ))),
        CpuStorage::BF16(v) => CpuStorage::BF16(Arc::new(index_select_generic(
            v, strides, offset, &out_dims, axis, &picks,
        ))),
        CpuStorage::F32(v) => CpuStorage::F32(Arc::new(index_select_generic(
            v, strides, offset, &out_dims, axis, &picks,
        ))),
        CpuStorage::F64(v) => CpuStorage::F64(Arc::new(index_select_generic(
            v, strides, offset, &out_dims, axis, &picks,
        ))),
        CpuStorage::I64(v) => CpuStorage::I64(Arc::new(index_select_generic(
            v, strides, offset, &out_dims, axis, &picks,
        ))),
        CpuStorage::Bool(v) => CpuStorage::Bool(Arc::new(index_select_generic(
            v, strides, offset, &out_dims, axis, &picks,
        ))),
    };
    Ok(Storage::Cpu(out))
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
    let out = match cpu_storage(x, OP)? {
        CpuStorage::F16(v) => CpuStorage::F16(Arc::new(gather_generic(
            v, strides, offset, out_dims, axis, &picks,
        ))),
        CpuStorage::BF16(v) => CpuStorage::BF16(Arc::new(gather_generic(
            v, strides, offset, out_dims, axis, &picks,
        ))),
        CpuStorage::F32(v) => CpuStorage::F32(Arc::new(gather_generic(
            v, strides, offset, out_dims, axis, &picks,
        ))),
        CpuStorage::F64(v) => CpuStorage::F64(Arc::new(gather_generic(
            v, strides, offset, out_dims, axis, &picks,
        ))),
        CpuStorage::I64(v) => CpuStorage::I64(Arc::new(gather_generic(
            v, strides, offset, out_dims, axis, &picks,
        ))),
        CpuStorage::Bool(v) => CpuStorage::Bool(Arc::new(gather_generic(
            v, strides, offset, out_dims, axis, &picks,
        ))),
    };
    Ok(Storage::Cpu(out))
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
    let out = match (x, src) {
        (CpuStorage::F16(a), CpuStorage::F16(b)) => CpuStorage::F16(Arc::new(kernel.run(a, b))),
        (CpuStorage::BF16(a), CpuStorage::BF16(b)) => CpuStorage::BF16(Arc::new(kernel.run(a, b))),
        (CpuStorage::F32(a), CpuStorage::F32(b)) => CpuStorage::F32(Arc::new(kernel.run(a, b))),
        (CpuStorage::F64(a), CpuStorage::F64(b)) => CpuStorage::F64(Arc::new(kernel.run(a, b))),
        (CpuStorage::I64(a), CpuStorage::I64(b)) => CpuStorage::I64(Arc::new(kernel.run(a, b))),
        (CpuStorage::Bool(_), _) => {
            return Err(Error::Unsupported {
                op,
                device,
                dtype: DType::Bool,
            });
        }
        // The op layer and the dtype guard above make this unreachable; a
        // kernel still reports rather than panics.
        (a, b) => {
            return Err(Error::DTypeMismatch {
                op,
                expected: a.dtype(),
                got: b.dtype(),
            });
        }
    };
    Ok(Storage::Cpu(out))
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

    let x_cpu = cpu_storage(x, OP)?;
    let src_cpu = cpu_storage(src, OP)?;
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

    let x_cpu = cpu_storage(x, OP)?;
    let src_cpu = cpu_storage(src, OP)?;
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
mod tests {
    use super::*;

    // ----- helpers ------------------------------------------------------

    fn f32_storage(v: Vec<f32>) -> Storage {
        Storage::Cpu(CpuStorage::F32(Arc::new(v)))
    }
    fn i64_storage(v: Vec<i64>) -> Storage {
        Storage::Cpu(CpuStorage::I64(Arc::new(v)))
    }
    fn f16_storage(v: Vec<f32>) -> Storage {
        Storage::Cpu(CpuStorage::F16(Arc::new(
            v.into_iter().map(half::f16::from_f32).collect(),
        )))
    }
    fn bf16_storage(v: Vec<f32>) -> Storage {
        Storage::Cpu(CpuStorage::BF16(Arc::new(
            v.into_iter().map(half::bf16::from_f32).collect(),
        )))
    }
    fn bool_storage(v: Vec<bool>) -> Storage {
        Storage::Cpu(CpuStorage::Bool(Arc::new(v)))
    }

    fn as_f32(s: &Storage) -> Vec<f32> {
        match s {
            Storage::Cpu(CpuStorage::F32(v)) => v.as_ref().clone(),
            _ => panic!("expected f32 storage"),
        }
    }
    fn as_f16(s: &Storage) -> Vec<f32> {
        match s {
            Storage::Cpu(CpuStorage::F16(v)) => v.iter().map(|e| e.to_f32()).collect(),
            _ => panic!("expected f16 storage"),
        }
    }
    fn as_bf16(s: &Storage) -> Vec<f32> {
        match s {
            Storage::Cpu(CpuStorage::BF16(v)) => v.iter().map(|e| e.to_f32()).collect(),
            _ => panic!("expected bf16 storage"),
        }
    }
    fn as_i64(s: &Storage) -> Vec<i64> {
        match s {
            Storage::Cpu(CpuStorage::I64(v)) => v.as_ref().clone(),
            _ => panic!("expected i64 storage"),
        }
    }
    fn as_bool(s: &Storage) -> Vec<bool> {
        match s {
            Storage::Cpu(CpuStorage::Bool(v)) => v.as_ref().clone(),
            _ => panic!("expected bool storage"),
        }
    }

    fn lay(dims: impl Into<Shape>) -> Layout {
        Layout::contiguous(dims).unwrap()
    }

    // ----- index_select --------------------------------------------------

    #[test]
    fn index_select_picks_rows() {
        // [[1,2],[3,4],[5,6]] — the embedding lookup shape.
        let s = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let l = lay([3, 2]);
        let idx = i64_storage(vec![2, 0, 2]);
        let il = lay([3]);
        let out = index_select(View::new(&s, &l), 0, View::new(&idx, &il)).unwrap();
        assert_eq!(as_f32(&out), vec![5.0, 6.0, 1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn index_select_picks_columns_and_middle_axes() {
        let s = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let l = lay([2, 3]);
        let idx = i64_storage(vec![2, 1]);
        let il = lay([2]);
        let out = index_select(View::new(&s, &l), 1, View::new(&idx, &il)).unwrap();
        assert_eq!(as_f32(&out), vec![3.0, 2.0, 6.0, 5.0]);

        // Middle axis of a rank-3 source.
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let s = f32_storage(data);
        let l = lay([2, 3, 4]);
        let idx = i64_storage(vec![1]);
        let il = lay([1]);
        let out = index_select(View::new(&s, &l), 1, View::new(&idx, &il)).unwrap();
        // Rows 1 of each plane: 4..8 and 16..20.
        assert_eq!(
            as_f32(&out),
            vec![4.0, 5.0, 6.0, 7.0, 16.0, 17.0, 18.0, 19.0]
        );
    }

    #[test]
    fn index_select_reads_strided_sources_and_indices() {
        // [2,3] transposed to [3,2]; selecting rows of the transposed view
        // must walk the source strides, not the raw buffer.
        let s = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let l = lay([2, 3]).transpose(0, 1).unwrap();
        assert_eq!(l.dims(), &[3, 2]);
        let idx = i64_storage(vec![0, 2]);
        let out = index_select(View::new(&s, &l), 0, View::new(&idx, &lay([2]))).unwrap();
        // Transposed rows are [1,4], [2,5], [3,6].
        assert_eq!(as_f32(&out), vec![1.0, 4.0, 3.0, 6.0]);

        // A strided *index* view is legal too: stride 2 over [0, 9, 2] reads
        // indices 0 and 2 and never touches the (out-of-range) middle value.
        let idx = i64_storage(vec![0, 9, 2]);
        let strided =
            Layout::from_parts(Shape::from([2]), vec![2usize].into_boxed_slice(), 0).unwrap();
        let out = index_select(View::new(&s, &l), 0, View::new(&idx, &strided)).unwrap();
        assert_eq!(as_f32(&out), vec![1.0, 4.0, 3.0, 6.0]);
    }

    #[test]
    fn index_select_handles_every_dtype_and_empty_indices() {
        let s = i64_storage(vec![7, 8, 9]);
        let l = lay([3]);
        let idx = i64_storage(vec![1, 1]);
        let il = lay([2]);
        let out = index_select(View::new(&s, &l), 0, View::new(&idx, &il)).unwrap();
        assert_eq!(as_i64(&out), vec![8, 8]);

        let s = bool_storage(vec![true, false, true]);
        let out = index_select(View::new(&s, &l), 0, View::new(&idx, &il)).unwrap();
        assert_eq!(as_bool(&out), vec![false, false]);

        // An empty index list yields an empty result, not an error.
        let empty = i64_storage(vec![]);
        let el = lay([0]);
        let s = f32_storage(vec![1.0, 2.0, 3.0]);
        let out = index_select(View::new(&s, &l), 0, View::new(&empty, &el)).unwrap();
        assert!(as_f32(&out).is_empty());
    }

    #[test]
    fn index_select_bounds_and_shape_errors() {
        let s = f32_storage(vec![1.0, 2.0, 3.0]);
        let l = lay([3]);
        let il = lay([1]);

        let idx = i64_storage(vec![3]);
        assert!(matches!(
            index_select(View::new(&s, &l), 0, View::new(&idx, &il)),
            Err(Error::IndexOutOfBounds {
                op: "index_select",
                index: 3,
                axis: 0,
                size: 3
            })
        ));

        // Negative indices are rejected, never wrapped.
        let idx = i64_storage(vec![-1]);
        assert!(matches!(
            index_select(View::new(&s, &l), 0, View::new(&idx, &il)),
            Err(Error::IndexOutOfBounds {
                op: "index_select",
                index: -1,
                ..
            })
        ));

        // The index tensor must be I64 and rank 1.
        let bad = f32_storage(vec![0.0]);
        assert!(matches!(
            index_select(View::new(&s, &l), 0, View::new(&bad, &il)),
            Err(Error::DTypeMismatch {
                op: "index_select",
                expected: DType::I64,
                got: DType::F32
            })
        ));
        let idx = i64_storage(vec![0, 1]);
        let two_d = lay([1, 2]);
        assert!(matches!(
            index_select(View::new(&s, &l), 0, View::new(&idx, &two_d)),
            Err(Error::RankMismatch {
                op: "index_select",
                expected: 1,
                got: 2
            })
        ));

        // An axis beyond the source rank is loud, not an out-of-range read.
        assert!(matches!(
            index_select(View::new(&s, &l), 4, View::new(&idx, &lay([2]))),
            Err(Error::InvalidAxis {
                op: "index_select",
                rank: 1,
                ..
            })
        ));
    }

    // ----- index_add ------------------------------------------------------

    #[test]
    fn index_add_accumulates_repeated_slices() {
        // Base [3,2] zeros; add two rows into slot 1 and one into slot 0.
        let base = f32_storage(vec![0.0; 6]);
        let bl = lay([3, 2]);
        let idx = i64_storage(vec![1, 0, 1]);
        let il = lay([3]);
        let src = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0]);
        let sl = lay([3, 2]);
        let out = index_add(
            View::new(&base, &bl),
            0,
            View::new(&idx, &il),
            View::new(&src, &sl),
        )
        .unwrap();
        // Row 0 gets [3,4]; row 1 gets [1,2] + [10,20]; row 2 untouched.
        assert_eq!(as_f32(&out), vec![3.0, 4.0, 11.0, 22.0, 0.0, 0.0]);
    }

    #[test]
    fn index_add_keeps_the_base_values() {
        let base = f32_storage(vec![1.0, 1.0, 1.0]);
        let bl = lay([3]);
        let idx = i64_storage(vec![2]);
        let il = lay([1]);
        let src = f32_storage(vec![5.0]);
        let sl = lay([1]);
        let out = index_add(
            View::new(&base, &bl),
            0,
            View::new(&idx, &il),
            View::new(&src, &sl),
        )
        .unwrap();
        assert_eq!(as_f32(&out), vec![1.0, 1.0, 6.0]);
    }

    #[test]
    fn index_add_accumulates_in_the_wide_acc_type() {
        // 4096 f16 ones into a single row: native f16 addition saturates at
        // 2048, the `Acc = f32` contract does not.
        let base = f16_storage(vec![0.0]);
        let bl = lay([1]);
        let idx = i64_storage(vec![0; 4096]);
        let il = lay([4096]);
        let src = f16_storage(vec![1.0; 4096]);
        let sl = lay([4096]);
        let out = index_add(
            View::new(&base, &bl),
            0,
            View::new(&idx, &il),
            View::new(&src, &sl),
        )
        .unwrap();
        assert_eq!(as_f16(&out), vec![4096.0]);
    }

    #[test]
    fn bf16_index_add_accumulates_in_the_wide_acc_type() {
        let base = bf16_storage(vec![0.0]);
        let idx = i64_storage(vec![0; 4096]);
        let src = bf16_storage(vec![1.0; 4096]);
        let out = index_add(
            View::new(&base, &lay([1])),
            0,
            View::new(&idx, &lay([4096])),
            View::new(&src, &lay([4096])),
        )
        .unwrap();
        assert_eq!(as_bf16(&out), vec![4096.0]);
    }

    #[test]
    fn index_add_reads_strided_bases_and_sources() {
        // Base is a transposed [2,3] -> [3,2] view; the accumulator must be
        // seeded through the strides and the output emitted row-major.
        let base = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let bl = lay([2, 3]).transpose(0, 1).unwrap(); // [[1,4],[2,5],[3,6]]
        let idx = i64_storage(vec![0]);
        let il = lay([1]);
        let src = f32_storage(vec![10.0, 20.0]);
        let sl = lay([1, 2]);
        let out = index_add(
            View::new(&base, &bl),
            0,
            View::new(&idx, &il),
            View::new(&src, &sl),
        )
        .unwrap();
        assert_eq!(as_f32(&out), vec![11.0, 24.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn index_add_shape_dtype_and_bool_errors() {
        let base = f32_storage(vec![0.0; 4]);
        let bl = lay([2, 2]);
        let idx = i64_storage(vec![0]);
        let il = lay([1]);

        // src must be the base shape with the indexed axis at the index count.
        let src = f32_storage(vec![1.0, 2.0, 3.0, 4.0]);
        assert!(matches!(
            index_add(
                View::new(&base, &bl),
                0,
                View::new(&idx, &il),
                View::new(&src, &lay([2, 2]))
            ),
            Err(Error::ShapeMismatch {
                op: "index_add",
                ..
            })
        ));

        // No implicit promotion between base and source.
        let src = i64_storage(vec![1, 2]);
        assert!(matches!(
            index_add(
                View::new(&base, &bl),
                0,
                View::new(&idx, &il),
                View::new(&src, &lay([1, 2]))
            ),
            Err(Error::DTypeMismatch {
                op: "index_add",
                ..
            })
        ));

        // Bool has no wide accumulator.
        let base = bool_storage(vec![false; 4]);
        let src = bool_storage(vec![true, true]);
        assert!(matches!(
            index_add(
                View::new(&base, &bl),
                0,
                View::new(&idx, &il),
                View::new(&src, &lay([1, 2]))
            ),
            Err(Error::Unsupported {
                op: "index_add",
                dtype: DType::Bool,
                ..
            })
        ));
    }

    // ----- gather ---------------------------------------------------------

    #[test]
    fn gather_picks_per_element_along_an_axis() {
        // [[1,2,3],[4,5,6]]; gather along axis 1 with [[0,2],[1,1]].
        let s = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let l = lay([2, 3]);
        let idx = i64_storage(vec![0, 2, 1, 1]);
        let il = lay([2, 2]);
        let out = gather(View::new(&s, &l), 1, View::new(&idx, &il)).unwrap();
        assert_eq!(as_f32(&out), vec![1.0, 3.0, 5.0, 5.0]);

        // Along axis 0: index [[1,0,1]] picks per column.
        let idx = i64_storage(vec![1, 0, 1]);
        let il = lay([1, 3]);
        let out = gather(View::new(&s, &l), 0, View::new(&idx, &il)).unwrap();
        assert_eq!(as_f32(&out), vec![4.0, 2.0, 6.0]);
    }

    #[test]
    fn gather_reads_strided_sources() {
        let s = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let l = lay([2, 3]).transpose(0, 1).unwrap(); // [[1,4],[2,5],[3,6]]
        let idx = i64_storage(vec![1, 0, 1]);
        let il = lay([3, 1]);
        let out = gather(View::new(&s, &l), 1, View::new(&idx, &il)).unwrap();
        assert_eq!(as_f32(&out), vec![4.0, 2.0, 6.0]);
    }

    #[test]
    fn gather_bounds_and_shape_errors() {
        let s = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let l = lay([2, 3]);

        let idx = i64_storage(vec![0, 3]);
        assert!(matches!(
            gather(View::new(&s, &l), 1, View::new(&idx, &lay([1, 2]))),
            Err(Error::IndexOutOfBounds {
                op: "gather",
                index: 3,
                axis: 1,
                size: 3
            })
        ));

        // Same-rank index grid required.
        let idx = i64_storage(vec![0, 1]);
        assert!(matches!(
            gather(View::new(&s, &l), 1, View::new(&idx, &lay([2]))),
            Err(Error::RankMismatch {
                op: "gather",
                expected: 2,
                got: 1
            })
        ));

        // A non-gathered axis larger than the source is a shape error.
        let idx = i64_storage(vec![0; 9]);
        assert!(matches!(
            gather(View::new(&s, &l), 1, View::new(&idx, &lay([3, 3]))),
            Err(Error::ShapeMismatch { op: "gather", .. })
        ));
    }

    // ----- scatter_add ----------------------------------------------------

    #[test]
    fn scatter_add_accumulates_duplicate_destinations() {
        let base = f32_storage(vec![0.0; 6]);
        let bl = lay([2, 3]);
        // Both columns of row 0 land on index 1; row 1 spreads out.
        let idx = i64_storage(vec![1, 1, 0, 2]);
        let il = lay([2, 2]);
        let src = f32_storage(vec![1.0, 2.0, 3.0, 4.0]);
        let sl = lay([2, 2]);
        let out = scatter_add(
            View::new(&base, &bl),
            1,
            View::new(&idx, &il),
            View::new(&src, &sl),
        )
        .unwrap();
        assert_eq!(as_f32(&out), vec![0.0, 3.0, 0.0, 3.0, 0.0, 4.0]);
    }

    /// PyTorch's rule is `index.size(d) <= src.size(d)`, so `src` may be
    /// strictly larger than the index grid: the grid names which `src`
    /// positions participate, and the rest are simply never read.
    ///
    /// Every other `scatter_add` test passes an index grid of exactly `src`'s
    /// shape, which makes "walk the grid" and "walk `src`" indistinguishable.
    /// The Metal kernel walked `src`, so it visited positions the grid never
    /// named and decoded one flat counter against two different shapes. Hand
    /// computed rather than differential, so it pins the semantics here
    /// without needing a second backend to agree with.
    #[test]
    fn scatter_add_reads_only_the_positions_the_index_grid_names() {
        let base = f32_storage(vec![0.0; 6]);
        let bl = lay([2, 3]);
        // A [2, 2] grid selecting from a [2, 3] source: column 2 of `src`
        // (30.0 and 60.0) lies outside the grid and must not contribute.
        let idx = i64_storage(vec![0, 2, 1, 1]);
        let il = lay([2, 2]);
        let src = f32_storage(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        let sl = lay([2, 3]);
        let out = scatter_add(
            View::new(&base, &bl),
            1,
            View::new(&idx, &il),
            View::new(&src, &sl),
        )
        .unwrap();
        // row 0: src[0,0]=10 -> col 0, src[0,1]=20 -> col 2.
        // row 1: src[1,0]=40 and src[1,1]=50 both -> col 1.
        assert_eq!(out.len(), 6, "the output has `x`'s shape, not `src`'s");
        assert_eq!(as_f32(&out), vec![10.0, 0.0, 20.0, 0.0, 90.0, 0.0]);
    }

    #[test]
    fn scatter_add_is_the_transpose_of_gather() {
        // Property: scattering ones through the same index grid a gather used
        // counts how many times each source element was read.
        let s = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let l = lay([2, 3]);
        let idx = i64_storage(vec![0, 2, 1, 1]);
        let il = lay([2, 2]);
        let picked = gather(View::new(&s, &l), 1, View::new(&idx, &il)).unwrap();
        assert_eq!(as_f32(&picked), vec![1.0, 3.0, 5.0, 5.0]);

        let base = f32_storage(vec![0.0; 6]);
        let ones = f32_storage(vec![1.0; 4]);
        let counts = scatter_add(
            View::new(&base, &l),
            1,
            View::new(&idx, &il),
            View::new(&ones, &il),
        )
        .unwrap();
        assert_eq!(as_f32(&counts), vec![1.0, 0.0, 1.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn scatter_add_accumulates_in_the_wide_acc_type() {
        let base = f16_storage(vec![0.0, 0.0]);
        let bl = lay([1, 2]);
        let idx = i64_storage(vec![0; 4096]);
        let il = lay([1, 4096]);
        let src = f16_storage(vec![1.0; 4096]);
        let sl = lay([1, 4096]);
        let out = scatter_add(
            View::new(&base, &bl),
            1,
            View::new(&idx, &il),
            View::new(&src, &sl),
        )
        .unwrap();
        assert_eq!(as_f16(&out), vec![4096.0, 0.0]);
    }

    #[test]
    fn bf16_scatter_add_accumulates_in_the_wide_acc_type() {
        let base = bf16_storage(vec![0.0, 0.0]);
        let idx = i64_storage(vec![0; 4096]);
        let src = bf16_storage(vec![1.0; 4096]);
        let out = scatter_add(
            View::new(&base, &lay([1, 2])),
            1,
            View::new(&idx, &lay([1, 4096])),
            View::new(&src, &lay([1, 4096])),
        )
        .unwrap();
        assert_eq!(as_bf16(&out), vec![4096.0, 0.0]);
    }

    #[test]
    fn scatter_add_bounds_and_shape_errors() {
        let base = f32_storage(vec![0.0; 6]);
        let bl = lay([2, 3]);
        let src = f32_storage(vec![1.0; 4]);
        let sl = lay([2, 2]);

        let idx = i64_storage(vec![0, 5, 0, 0]);
        assert!(matches!(
            scatter_add(
                View::new(&base, &bl),
                1,
                View::new(&idx, &lay([2, 2])),
                View::new(&src, &sl)
            ),
            Err(Error::IndexOutOfBounds {
                op: "scatter_add",
                index: 5,
                axis: 1,
                size: 3
            })
        ));

        // The index grid may not exceed `src`.
        let idx = i64_storage(vec![0; 6]);
        assert!(matches!(
            scatter_add(
                View::new(&base, &bl),
                1,
                View::new(&idx, &lay([2, 3])),
                View::new(&src, &sl)
            ),
            Err(Error::ShapeMismatch {
                op: "scatter_add",
                ..
            })
        ));

        // Bool has no wide accumulator here either.
        let base = bool_storage(vec![false; 6]);
        let src = bool_storage(vec![true; 4]);
        let idx = i64_storage(vec![0; 4]);
        assert!(matches!(
            scatter_add(
                View::new(&base, &bl),
                1,
                View::new(&idx, &lay([2, 2])),
                View::new(&src, &sl)
            ),
            Err(Error::Unsupported {
                op: "scatter_add",
                dtype: DType::Bool,
                ..
            })
        ));
    }

    // ----- fast paths vs. the naive per-element decode ---------------------
    //
    // Every kernel above grew a whole-row `memcpy` tier and a division-free
    // odometer. Both must be *indistinguishable* from the per-element decode
    // they replaced, so the decode is reproduced here verbatim and the two are
    // compared over a matrix of layouts chosen to hit each tier:
    //
    // - dense source, indexed on the outer axis  -> row-copy tier
    // - dense source, indexed on the innermost axis -> run length 1
    // - transposed / inner-narrowed / broadcast source -> general walk
    //
    // For the accumulating kernels the comparison is `assert_eq!` on the
    // *result elements*, which for floats is bitwise equality — the fast path
    // is required to add the same contributions to each cell in the same
    // order, so no tolerance is involved or allowed.

    /// The storage index of logical (row-major) position `i` of `layout`,
    /// whose place values are `place`. The decode the kernels used before the
    /// odometer replaced it.
    fn naive_storage_index(layout: &Layout, place: &[usize], i: usize) -> usize {
        let dims = layout.dims();
        let strides = layout.strides();
        let mut idx = layout.offset();
        for a in 0..dims.len() {
            idx += ((i / place[a]) % dims[a]) * strides[a];
        }
        idx
    }

    /// Pre-optimization `index_select_generic`.
    fn naive_index_select(
        data: &[f32],
        x_strides: &[usize],
        x_offset: usize,
        out_dims: &[usize],
        axis: usize,
        picks: &[usize],
    ) -> Vec<f32> {
        let place = place_values(out_dims);
        let n: usize = out_dims.iter().product();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut src = x_offset;
            for a in 0..out_dims.len() {
                let c = (i / place[a]) % out_dims[a];
                src += if a == axis { picks[c] } else { c } * x_strides[a];
            }
            out.push(data[src]);
        }
        out
    }

    /// Pre-optimization `gather_generic`.
    fn naive_gather(
        data: &[f32],
        x_strides: &[usize],
        x_offset: usize,
        out_dims: &[usize],
        axis: usize,
        picks: &[usize],
    ) -> Vec<f32> {
        let place = place_values(out_dims);
        let mut out = Vec::with_capacity(picks.len());
        for (i, &pick) in picks.iter().enumerate() {
            let mut src = x_offset;
            for a in 0..out_dims.len() {
                let c = if a == axis {
                    pick
                } else {
                    (i / place[a]) % out_dims[a]
                };
                src += c * x_strides[a];
            }
            out.push(data[src]);
        }
        out
    }

    /// Pre-optimization `index_add_generic` (f32, so `Acc` is f32 too).
    fn naive_index_add(
        x_data: &[f32],
        x_layout: &Layout,
        src_data: &[f32],
        src_layout: &Layout,
        axis: usize,
        picks: &[usize],
    ) -> Vec<f32> {
        let x_place = place_values(x_layout.dims());
        let mut acc: Vec<f32> = (0..x_layout.num_elements())
            .map(|i| x_data[naive_storage_index(x_layout, &x_place, i)])
            .collect();
        let src_dims = src_layout.dims();
        let src_strides = src_layout.strides();
        let src_place = place_values(src_dims);
        for i in 0..src_layout.num_elements() {
            let mut from = src_layout.offset();
            let mut dst = 0usize;
            for a in 0..src_dims.len() {
                let c = (i / src_place[a]) % src_dims[a];
                from += c * src_strides[a];
                dst += if a == axis { picks[c] } else { c } * x_place[a];
            }
            acc[dst] += src_data[from];
        }
        acc
    }

    /// Pre-optimization `scatter_add_generic` (f32).
    fn naive_scatter_add(
        x_data: &[f32],
        x_layout: &Layout,
        src_data: &[f32],
        src_layout: &Layout,
        idx_dims: &[usize],
        axis: usize,
        picks: &[usize],
    ) -> Vec<f32> {
        let x_place = place_values(x_layout.dims());
        let mut acc: Vec<f32> = (0..x_layout.num_elements())
            .map(|i| x_data[naive_storage_index(x_layout, &x_place, i)])
            .collect();
        let idx_place = place_values(idx_dims);
        let src_strides = src_layout.strides();
        for (i, &pick) in picks.iter().enumerate() {
            let mut from = src_layout.offset();
            let mut dst = 0usize;
            for a in 0..idx_dims.len() {
                let c = (i / idx_place[a]) % idx_dims[a];
                from += c * src_strides[a];
                dst += if a == axis { pick } else { c } * x_place[a];
            }
            acc[dst] += src_data[from];
        }
        acc
    }

    /// `0.5, 1.0, 1.5, …` — values a float sum reorder would expose, in a
    /// buffer long enough for every layout below plus an offset.
    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| 0.5 * (i as f32 + 1.0)).collect()
    }

    /// The rank-3 source layouts the equivalence tests sweep, paired with the
    /// axis to index and a label. All are views over `ramp(240)`.
    fn source_layouts() -> Vec<(&'static str, Layout, usize)> {
        let base = lay([3, 4, 5]);
        vec![
            // Dense: outer axis is the row-copy tier (run = 20).
            ("dense, axis 0", base.clone(), 0),
            // Dense: middle axis, run = 5.
            ("dense, axis 1", base.clone(), 1),
            // Dense: innermost axis, run = 1.
            ("dense, axis 2", base.clone(), 2),
            // Offset run: still dense, so still the row-copy tier.
            ("outer narrow, axis 1", base.narrow(0, 1, 2).unwrap(), 1),
            // Inner narrow: the axes inside 0 are no longer contiguous, so
            // axis 0 falls to the general walk.
            ("inner narrow, axis 0", base.narrow(2, 1, 3).unwrap(), 0),
            // Transposed: reordered strides, general walk.
            ("transposed 0/2, axis 0", base.transpose(0, 2).unwrap(), 0),
            ("transposed 1/2, axis 1", base.transpose(1, 2).unwrap(), 1),
            // Permuted.
            ("permuted, axis 2", base.permute(&[2, 0, 1]).unwrap(), 2),
            // Broadcast innermost axis: stride 0 inside the indexed axis, so
            // the row-copy tier must refuse it.
            (
                "broadcast innermost, axis 0",
                lay([3, 4, 1])
                    .broadcast_to(&Shape::from([3, 4, 5]))
                    .unwrap(),
                0,
            ),
            // Broadcast leading axis, indexed on a later axis.
            (
                "broadcast leading, axis 1",
                lay([1, 4, 5])
                    .broadcast_to(&Shape::from([3, 4, 5]))
                    .unwrap(),
                1,
            ),
        ]
    }

    #[test]
    fn index_select_fast_and_general_paths_match_the_naive_decode() {
        let data = ramp(240);
        for (name, layout, axis) in source_layouts() {
            let size = layout.dims()[axis];
            // Repeats, reversal and a truncated list, so `picks.len()` differs
            // from the source size in both directions.
            for picks in [
                vec![0usize],
                (0..size).collect::<Vec<_>>(),
                (0..size).rev().collect::<Vec<_>>(),
                vec![size - 1, 0, size - 1],
            ] {
                let mut out_dims = layout.dims().to_vec();
                out_dims[axis] = picks.len();
                let got = index_select_generic(
                    &data,
                    layout.strides(),
                    layout.offset(),
                    &out_dims,
                    axis,
                    &picks,
                );
                let want = naive_index_select(
                    &data,
                    layout.strides(),
                    layout.offset(),
                    &out_dims,
                    axis,
                    &picks,
                );
                assert_eq!(got, want, "{name} picks={picks:?}");
            }
        }
    }

    #[test]
    fn gather_matches_the_naive_decode_on_strided_sources() {
        let data = ramp(240);
        for (name, layout, axis) in source_layouts() {
            let out_dims = layout.dims().to_vec();
            let size = layout.dims()[axis];
            // One index per output element, cycling so neighbours differ.
            let picks: Vec<usize> = (0..out_dims.iter().product::<usize>())
                .map(|i| (i * 3 + 1) % size)
                .collect();
            let got = gather_generic(
                &data,
                layout.strides(),
                layout.offset(),
                &out_dims,
                axis,
                &picks,
            );
            let want = naive_gather(
                &data,
                layout.strides(),
                layout.offset(),
                &out_dims,
                axis,
                &picks,
            );
            assert_eq!(got, want, "{name}");
        }
    }

    #[test]
    fn index_add_fast_and_general_paths_match_the_naive_decode_bitwise() {
        let x_data = ramp(240);
        // `src` gets its own buffer so a mixed-up read is visible.
        let src_data: Vec<f32> = ramp(240).iter().map(|v| -v - 0.25).collect();
        let base = lay([3, 4, 5]);
        for axis in 0..3 {
            let size = base.dims()[axis];
            for picks in [
                (0..size).collect::<Vec<_>>(),
                (0..size).rev().collect::<Vec<_>>(),
                // Duplicates: several `src` slices land on one base slice.
                // (What makes the *order* of those adds observable is
                // `index_add_accumulates_duplicates_in_source_order`, below —
                // this ramp's sums are exact in f32, so order alone would not
                // show up here.)
                vec![0usize; size],
                vec![size - 1, 0, size - 1],
            ] {
                let mut src_dims = base.dims().to_vec();
                src_dims[axis] = picks.len();
                // A dense `src` (row-copy tier) and a transposed one (general
                // walk) must both agree with the decode.
                let dense = lay(src_dims.clone());
                let swapped = {
                    let (a, b) = (axis, (axis + 1) % 3);
                    let mut d = src_dims.clone();
                    d.swap(a, b);
                    lay(d).transpose(a, b).unwrap()
                };
                for (label, src_layout) in [("dense src", dense), ("transposed src", swapped)] {
                    assert_eq!(src_layout.dims(), &src_dims[..], "{label}: dims");
                    let got: Vec<f32> =
                        index_add_generic(&x_data, &base, &src_data, &src_layout, axis, &picks);
                    let want =
                        naive_index_add(&x_data, &base, &src_data, &src_layout, axis, &picks);
                    assert_eq!(got, want, "axis={axis} {label} picks={picks:?}");
                }
            }
        }
    }

    #[test]
    fn scatter_add_matches_the_naive_decode_bitwise() {
        let x_data = ramp(240);
        let src_data: Vec<f32> = ramp(240).iter().map(|v| -v - 0.25).collect();
        let base = lay([3, 4, 5]);
        for axis in 0..3 {
            let size = base.dims()[axis];
            let idx_dims = base.dims().to_vec();
            let n: usize = idx_dims.iter().product();
            // Collide deliberately: `% (size.min(2))` sends many grid cells to
            // the same destination on every axis.
            let picks: Vec<usize> = (0..n).map(|i| i % size.min(2)).collect();
            for (label, src_layout) in [
                ("dense src", lay(idx_dims.clone())),
                (
                    "transposed src",
                    lay([idx_dims[1], idx_dims[0], idx_dims[2]])
                        .transpose(0, 1)
                        .unwrap(),
                ),
            ] {
                let got: Vec<f32> = scatter_add_generic(
                    &x_data,
                    &base,
                    &src_data,
                    &src_layout,
                    &idx_dims,
                    axis,
                    &picks,
                );
                let want = naive_scatter_add(
                    &x_data,
                    &base,
                    &src_data,
                    &src_layout,
                    &idx_dims,
                    axis,
                    &picks,
                );
                assert_eq!(got, want, "axis={axis} {label}");
            }
        }
    }

    #[test]
    fn trailing_run_admits_only_genuinely_contiguous_inner_axes() {
        let base = lay([3, 4, 5]);
        // Dense: the run is the product of the axes inside the indexed one.
        assert_eq!(trailing_run(base.dims(), base.strides(), 0), Some(20));
        assert_eq!(trailing_run(base.dims(), base.strides(), 1), Some(5));
        assert_eq!(trailing_run(base.dims(), base.strides(), 2), Some(1));

        // Narrowing the innermost axis gives dims [3,4,3] / strides [20,5,1]:
        // consecutive axis-1 rows are 5 apart but only 3 wide, so a gap opens
        // and no run may span axis 1.
        let narrowed = base.narrow(2, 1, 3).unwrap();
        assert_eq!(trailing_run(narrowed.dims(), narrowed.strides(), 0), None);
        // The surviving 3 innermost elements are still consecutive, though, so
        // a pick along axis 1 copies a run of 3 — and a pick along axis 2 has
        // no inner axes at all, hence the trivial run of 1.
        assert_eq!(
            trailing_run(narrowed.dims(), narrowed.strides(), 1),
            Some(3)
        );
        assert_eq!(
            trailing_run(narrowed.dims(), narrowed.strides(), 2),
            Some(1)
        );

        // A broadcast axis inside the indexed one repeats elements and must be
        // refused, even though its size is > 1.
        let bcast = lay([3, 4, 1])
            .broadcast_to(&Shape::from([3, 4, 5]))
            .unwrap();
        assert_eq!(trailing_run(bcast.dims(), bcast.strides(), 0), None);

        // A size-1 axis contributes nothing to addressing, so an odd stride on
        // one does not break the run.
        let odd = Layout::from_parts(
            Shape::from([2, 1, 6]),
            vec![6usize, 999, 1].into_boxed_slice(),
            0,
        )
        .unwrap();
        assert_eq!(trailing_run(odd.dims(), odd.strides(), 0), Some(6));
    }

    /// The one property the `index_add` row-copy tier could plausibly break
    /// and the sweep above could not see: **the order** in which duplicate
    /// picks accumulate into the same destination cell.
    ///
    /// `index_add` is `index_select`'s backward, so this is the embedding
    /// gradient: many source rows summing into one row of the base. The fast
    /// path adds a whole row at a time (outer coordinate, then `k`, then
    /// position within the row); the decode it replaced added one element at a
    /// time in row-major `src` order. Those are the same order, and this test
    /// is what holds them to it.
    ///
    /// The values are chosen so that floating-point addition is *not*
    /// associative over them, and the test asserts that itself before asserting
    /// the kernel: `1.0` followed by four ties-away-from-zero-sized crumbs sums
    /// to exactly `1.0` in ascending order (each crumb is below half an ulp of
    /// 1.0 on its own) but to `1.0 + 1 ulp` when the crumbs are added together
    /// first. So a reordered accumulation cannot pass.
    #[test]
    fn index_add_accumulates_duplicates_in_source_order() {
        const CRUMB: f32 = 3e-8;
        let (rows, cols) = (5usize, 3usize);

        // Self-check: these values really are order-sensitive in f32.
        let ascending = {
            let mut a = 1.0f32;
            for _ in 1..rows {
                a += CRUMB;
            }
            a
        };
        let descending = {
            let mut a = 0.0f32;
            for _ in 1..rows {
                a += CRUMB;
            }
            a + 1.0
        };
        assert_ne!(
            ascending, descending,
            "the test's own values must be order-sensitive, else it proves nothing"
        );

        // Every source row lands on base row 0.
        let picks = vec![0usize; rows];
        let x_layout = lay([2, cols]);
        let x_data = vec![0.0f32; x_layout.num_elements()];
        // Row 0 is the big value, rows 1.. are the crumbs.
        let src_data: Vec<f32> = (0..rows * cols)
            .map(|i| if i < cols { 1.0 } else { CRUMB })
            .collect();

        // Dense source: the row-copy tier (run = cols).
        let dense = lay([rows, cols]);
        assert_eq!(trailing_run(dense.dims(), dense.strides(), 0), Some(cols));
        let got: Vec<f32> = index_add_generic(&x_data, &x_layout, &src_data, &dense, 0, &picks);
        assert_eq!(
            got,
            naive_index_add(&x_data, &x_layout, &src_data, &dense, 0, &picks),
            "row-copy tier reordered the accumulation"
        );
        assert_eq!(got[0], ascending, "…and the order is source order");

        // Transposed source: the general element-at-a-time walk, same order.
        let transposed = lay([cols, rows]).transpose(0, 1).unwrap();
        assert_eq!(transposed.dims(), &[rows, cols]);
        assert_eq!(
            trailing_run(transposed.dims(), transposed.strides(), 0),
            None
        );
        let got: Vec<f32> =
            index_add_generic(&x_data, &x_layout, &src_data, &transposed, 0, &picks);
        assert_eq!(
            got,
            naive_index_add(&x_data, &x_layout, &src_data, &transposed, 0, &picks),
            "general walk reordered the accumulation"
        );
    }

    /// The same order guarantee for `scatter_add`, whose grid picks one
    /// destination per element: collide every grid cell of a column onto one
    /// base cell and check the sum against the decode.
    #[test]
    fn scatter_add_accumulates_collisions_in_grid_order() {
        const CRUMB: f32 = 3e-8;
        let (rows, cols) = (5usize, 3usize);
        let x_layout = lay([2, cols]);
        let x_data = vec![0.0f32; x_layout.num_elements()];
        let idx_dims = vec![rows, cols];
        // Whole grid scatters onto base row 0, along axis 0.
        let picks = vec![0usize; rows * cols];
        let src_data: Vec<f32> = (0..rows * cols)
            .map(|i| if i < cols { 1.0 } else { CRUMB })
            .collect();
        let src_layout = lay([rows, cols]);
        let got: Vec<f32> = scatter_add_generic(
            &x_data,
            &x_layout,
            &src_data,
            &src_layout,
            &idx_dims,
            0,
            &picks,
        );
        assert_eq!(
            got,
            naive_scatter_add(
                &x_data,
                &x_layout,
                &src_data,
                &src_layout,
                &idx_dims,
                0,
                &picks
            ),
            "scatter_add reordered the accumulation"
        );
        // Ascending grid order: 1.0 first, then the crumbs, each lost.
        assert_eq!(got[0], 1.0f32);
    }
}
