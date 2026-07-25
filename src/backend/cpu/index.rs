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
//! - **One coordinate walk.** All four kernels are the same loop: decode a
//!   row-major logical position into per-axis coordinates, replace the
//!   indexed axis's coordinate with a looked-up index, and address the other
//!   side. [`place_values`] gives each axis its row-major place value, so the
//!   decode is two integer ops per axis.
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

use crate::backend::View;
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
#[cfg_attr(not(feature = "metal"), allow(unused_variables))]
fn cpu_storage<'a>(x: View<'a>, op: &'static str) -> Result<&'a CpuStorage> {
    match x.storage() {
        Storage::Cpu(s) => Ok(s),
        #[cfg(feature = "metal")]
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
/// The coordinate of logical position `i` on axis `a` is then
/// `(i / place[a]) % dims[a]` — the odometer decode every kernel below uses.
fn place_values(dims: &[usize]) -> Vec<usize> {
    let mut place = vec![1usize; dims.len()];
    for a in (0..dims.len().saturating_sub(1)).rev() {
        place[a] = place[a + 1] * dims[a + 1];
    }
    place
}

/// The storage index of logical (row-major) position `i` of `layout`, whose
/// place values are `place`.
#[inline]
fn storage_index(layout: &Layout, place: &[usize], i: usize) -> usize {
    let dims = layout.dims();
    let strides = layout.strides();
    let mut idx = layout.offset();
    for a in 0..dims.len() {
        idx += ((i / place[a]) % dims[a]) * strides[a];
    }
    idx
}

/// Read an index view into a row-major `Vec<i64>`.
///
/// The view may be strided (a transposed or narrowed index tensor is legal);
/// values come out in the same logical order the kernels walk. A non-`I64`
/// index tensor is [`Error::DTypeMismatch`](crate::Error::DTypeMismatch) —
/// indices are `I64` by contract, never silently cast.
fn read_indices(indices: View<'_>, op: &'static str) -> Result<Vec<i64>> {
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
    let place = place_values(layout.dims());
    Ok((0..layout.num_elements())
        .map(|i| data[storage_index(layout, &place, i)])
        .collect())
}

/// Bounds-check raw index values against an axis of size `size`, returning
/// them as `usize` positions.
fn resolve_indices(
    values: &[i64],
    axis: usize,
    size: usize,
    op: &'static str,
) -> Result<Vec<usize>> {
    values
        .iter()
        .map(|&v| {
            let oob = || Error::IndexOutOfBounds {
                op,
                index: v,
                axis,
                size,
            };
            let pos = usize::try_from(v).map_err(|_| oob())?;
            if pos >= size { Err(oob()) } else { Ok(pos) }
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

/// Wide-accumulator addition for the `Acc` contract.
///
/// Implemented for exactly the accumulator types
/// [`Element::Acc`](crate::dtype::Element::Acc) yields for a numeric element
/// (`f32` for `f16`/`bf16`/`f32`, `f64`, `i64`). `Bool` has `Acc = bool`,
/// which deliberately does not implement this: accumulating booleans is not
/// part of the kernel contract, and the dispatch below rejects it loudly.
trait AccAdd: Copy {
    /// Widening addition step.
    fn add(self, other: Self) -> Self;
}

impl AccAdd for f32 {
    fn add(self, other: Self) -> Self {
        self + other
    }
}

impl AccAdd for f64 {
    fn add(self, other: Self) -> Self {
        self + other
    }
}

impl AccAdd for i64 {
    fn add(self, other: Self) -> Self {
        // Matches the reduction kernels: integer overflow wraps rather than
        // panicking in debug and silently differing in release.
        self.wrapping_add(other)
    }
}

// ---------------------------------------------------------------------------
// index_select — slice-wise gather, 1-D index
// ---------------------------------------------------------------------------

/// `out[.., k, ..] = x[.., picks[k], ..]`, where `out_dims` is `x`'s shape
/// with `axis` resized to `picks.len()` and the output is dense row-major.
fn index_select_generic<E: Copy>(
    data: &[E],
    x_strides: &[usize],
    x_offset: usize,
    out_dims: &[usize],
    axis: usize,
    picks: &[usize],
) -> Vec<E> {
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

/// See [`BackendOps::index_select`](crate::backend::BackendOps::index_select).
pub(crate) fn index_select(x: View<'_>, axis: usize, indices: View<'_>) -> Result<Storage> {
    const OP: &str = "index_select";
    let layout = x.layout();
    check_axis(OP, axis, layout.rank())?;
    check_rank(OP, 1, indices.layout().rank())?;
    let picks = resolve_indices(&read_indices(indices, OP)?, axis, layout.dims()[axis], OP)?;

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
    let picks = resolve_indices(&read_indices(indices, OP)?, axis, layout.dims()[axis], OP)?;

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
fn seed_acc<E>(x_data: &[E], x_layout: &Layout, x_place: &[usize]) -> Vec<E::Acc>
where
    E: Element,
{
    (0..x_layout.num_elements())
        .map(|i| x_data[storage_index(x_layout, x_place, i)].to_acc())
        .collect()
}

/// `out = x; out[.., picks[k], ..] += src[.., k, ..]` — the slice-wise,
/// 1-D-index accumulate that backs `index_select`'s backward. Repeated
/// positions in `picks` accumulate (the embedding backward's whole point).
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
    E::Acc: AccAdd,
{
    let x_place = place_values(x_layout.dims());
    let mut acc = seed_acc(x_data, x_layout, &x_place);

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
        acc[dst] = acc[dst].add(src_data[from].to_acc());
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
    E::Acc: AccAdd,
{
    let x_place = place_values(x_layout.dims());
    let mut acc = seed_acc(x_data, x_layout, &x_place);

    // One index per grid position, by construction.
    debug_assert_eq!(picks.len(), idx_dims.iter().product::<usize>());
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
        acc[dst] = acc[dst].add(src_data[from].to_acc());
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
        E::Acc: AccAdd;
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
        E::Acc: AccAdd,
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
        E::Acc: AccAdd,
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
    let picks = resolve_indices(&read_indices(indices, OP)?, axis, x_layout.dims()[axis], OP)?;

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

    let picks = resolve_indices(&read_indices(indices, OP)?, axis, x_layout.dims()[axis], OP)?;

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
}
