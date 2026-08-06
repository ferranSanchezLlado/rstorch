//! Reduction CPU kernels.
//!
//! Signatures frozen by T01; **T11** fills the bodies — ported v2 loops
//! **adapted to the `Element::Acc` contract** (v2's native paths
//! accumulate in dtype; do not port verbatim). Semantics on
//! [`BackendOps`](crate::backend::BackendOps).
//!
//! Every reduction walks the source view stride-aware (it never assumes a
//! contiguous buffer), accumulates in the wide
//! [`Acc`](crate::dtype::Element::Acc) type, and casts back to the element
//! type exactly once at output (the implemented fix for the v2 f16
//! sum-saturation bug). The reduced axis is dropped from the result shape;
//! the op layer re-inserts it for the `_keepdim` spellings.

use crate::backend::{ArgReduceOp, ReduceOp, View};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::shape::Shape;
use crate::storage::{CpuStorage, Storage};

/// Wide-accumulator arithmetic used by the reduction kernels.
///
/// Implemented for exactly the accumulator types the `Acc` contract yields
/// for a numeric element (`f32` for `f16`/`bf16`/`f32`, `f64`, `i64`). `Bool`
/// has `Acc = bool`, which deliberately does not implement this trait: bool
/// reductions are not part of the kernel contract and are rejected before a
/// generic reduce is ever instantiated.
trait Acc: Copy {
    /// The additive identity (`Sum`/`Mean` seed).
    const ZERO: Self;
    /// Widening sum step.
    fn add(self, other: Self) -> Self;
    /// Running maximum, propagating NaN (so a NaN anywhere in the axis wins,
    /// matching PyTorch).
    fn max(self, other: Self) -> Self;
    /// Running minimum, propagating NaN.
    fn min(self, other: Self) -> Self;
    /// Divide an accumulated sum by a (wide) element count for `Mean`. The
    /// count is passed as `usize` and widened inside the impl.
    fn div_count(self, count: usize) -> Self;
    /// Order two accumulated values for `argmax`/`argmin`, with the first
    /// occurrence winning a tie (as in PyTorch).
    ///
    /// NaN never reaches here: [`arg_reduce_generic`] settles a NaN candidate
    /// before comparing, because a NaN must win *both* directions and so
    /// cannot be expressed as a position in any single total order.
    fn order(self, other: Self) -> std::cmp::Ordering;
    /// Whether this value is NaN — always `false` for integer accumulators.
    ///
    /// Arg-reductions need this because a NaN anywhere in the line is the
    /// selected element for both `argmax` and `argmin`, which is what keeps
    /// them consistent with `max`/`min` (both of which propagate NaN).
    fn is_nan(self) -> bool;
}

impl Acc for f32 {
    const ZERO: Self = 0.0;
    fn add(self, other: Self) -> Self {
        self + other
    }
    fn max(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            f32::NAN
        } else if self >= other {
            self
        } else {
            other
        }
    }
    fn min(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            f32::NAN
        } else if self <= other {
            self
        } else {
            other
        }
    }
    fn div_count(self, count: usize) -> Self {
        self / (count as f32)
    }
    fn order(self, other: Self) -> std::cmp::Ordering {
        self.partial_cmp(&other)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
}

impl Acc for f64 {
    const ZERO: Self = 0.0;
    fn add(self, other: Self) -> Self {
        self + other
    }
    fn max(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            f64::NAN
        } else if self >= other {
            self
        } else {
            other
        }
    }
    fn min(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            f64::NAN
        } else if self <= other {
            self
        } else {
            other
        }
    }
    fn div_count(self, count: usize) -> Self {
        self / (count as f64)
    }
    fn order(self, other: Self) -> std::cmp::Ordering {
        self.partial_cmp(&other)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
}

impl Acc for i64 {
    const ZERO: Self = 0;
    fn add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
    fn max(self, other: Self) -> Self {
        std::cmp::Ord::max(self, other)
    }
    fn min(self, other: Self) -> Self {
        std::cmp::Ord::min(self, other)
    }
    fn div_count(self, count: usize) -> Self {
        // Integer mean truncates toward zero (PyTorch does not offer an
        // integer mean, but the kernel contract keeps counts wide and casts
        // once; truncation is the only sensible integer division).
        self / (count as i64)
    }
    fn order(self, other: Self) -> std::cmp::Ordering {
        std::cmp::Ord::cmp(&self, &other)
    }
    fn is_nan(self) -> bool {
        false
    }
}

/// Enumerate the storage indices of the `axis` line through `x` whose other
/// coordinates are fixed by `outer` (a row-major index into the shape with
/// `axis` removed). Yields exactly `dims[axis]` indices, in ascending
/// coordinate order along the axis, applying `f` to each.
///
/// This is the stride-aware inner walk shared by every reduction: it reads
/// through [`Layout::strides`](crate::layout::Layout::strides) from the view
/// offset and so is correct for permuted, narrowed, and broadcast inputs.
fn for_each_on_axis(layout: &Layout, axis: usize, outer: usize, mut f: impl FnMut(usize, usize)) {
    let dims = layout.dims();
    let strides = layout.strides();
    // Decode `outer` into per-axis coordinates for every axis except `axis`,
    // accumulating the base storage index (the axis coordinate is 0 here).
    let mut base = layout.offset();
    let mut rem = outer;
    // Walk axes right-to-left so the rightmost non-reduced axis is fastest,
    // matching the row-major ordering the op layer expects of the output.
    for ax in (0..dims.len()).rev() {
        if ax == axis {
            continue;
        }
        let size = dims[ax];
        let coord = rem % size;
        rem /= size;
        base += coord * strides[ax];
    }
    let axis_stride = strides[axis];
    for pos in 0..dims[axis] {
        f(pos, base + pos * axis_stride);
    }
}

/// The output shape of a reduction over `axis`: the input shape with that
/// axis removed.
fn reduced_shape(layout: &Layout, axis: usize) -> Shape {
    let dims: Vec<usize> = layout
        .dims()
        .iter()
        .enumerate()
        .filter(|&(ax, _)| ax != axis)
        .map(|(_, &d)| d)
        .collect();
    Shape::from(dims)
}

/// Generic reduction over one axis, accumulating in `E::Acc` and casting back
/// once at output. `slice` is the whole backing buffer; `layout` addresses
/// the logical view over it.
fn reduce_generic<E>(op: ReduceOp, slice: &[E], layout: &Layout, axis: usize) -> Vec<E>
where
    E: Element,
    E::Acc: Acc,
{
    let out_shape = reduced_shape(layout, axis);
    let out_len = out_shape.num_elements();
    let axis_len = layout.dims()[axis];
    let mut out = Vec::with_capacity(out_len);
    for outer in 0..out_len {
        // Seed per op: Sum/Mean from ZERO; Max/Min from the first element so
        // an empty axis (guarded by the op layer's empty-reduction policy)
        // never dereferences a missing element here.
        let mut acc = <E::Acc as Acc>::ZERO;
        let mut first = true;
        for_each_on_axis(layout, axis, outer, |_pos, idx| {
            let v = slice[idx].to_acc();
            acc = match op {
                ReduceOp::Sum | ReduceOp::Mean => acc.add(v),
                ReduceOp::Max => {
                    if first {
                        v
                    } else {
                        acc.max(v)
                    }
                }
                ReduceOp::Min => {
                    if first {
                        v
                    } else {
                        acc.min(v)
                    }
                }
            };
            first = false;
        });
        let acc = if matches!(op, ReduceOp::Mean) {
            acc.div_count(axis_len)
        } else {
            acc
        };
        out.push(E::from_acc(acc));
    }
    out
}

/// Generic index-producing reduction (`argmax`/`argmin`) over one axis,
/// returning I64 positions of the winning element (first on ties). Comparison
/// is done in the wide accumulator so `f16`/`bf16` inputs are ordered exactly.
fn arg_reduce_generic<E>(op: ArgReduceOp, slice: &[E], layout: &Layout, axis: usize) -> Vec<i64>
where
    E: Element,
    E::Acc: Acc,
{
    use std::cmp::Ordering;
    let out_shape = reduced_shape(layout, axis);
    let out_len = out_shape.num_elements();
    let mut out = Vec::with_capacity(out_len);
    for outer in 0..out_len {
        let mut best_val: Option<E::Acc> = None;
        let mut best_pos: i64 = 0;
        for_each_on_axis(layout, axis, outer, |pos, idx| {
            let v = slice[idx].to_acc();
            let take = match best_val {
                None => true,
                // A NaN already holding the line keeps it: NaN outranks every
                // number in both directions, and the first occurrence wins.
                Some(cur) if cur.is_nan() => false,
                // A NaN candidate takes the line for `argmax` *and* `argmin`,
                // so the selected index always agrees with what `max`/`min`
                // report for that line (both propagate NaN). PyTorch does the
                // same; treating NaN as merely "very small" would make
                // `x.max(axis)` and `x.argmax(axis)` name different elements.
                Some(_) if v.is_nan() => true,
                Some(cur) => match op {
                    // Strictly better only: first occurrence wins on ties.
                    ArgReduceOp::ArgMax => v.order(cur) == Ordering::Greater,
                    ArgReduceOp::ArgMin => v.order(cur) == Ordering::Less,
                },
            };
            if take {
                best_val = Some(v);
                best_pos = pos as i64;
            }
        });
        out.push(best_pos);
    }
    out
}

/// See [`BackendOps::reduce`](crate::backend::BackendOps::reduce).
pub(crate) fn reduce(op: ReduceOp, x: View<'_>, axis: usize) -> Result<Storage> {
    let layout = x.layout();
    debug_assert!(
        axis < layout.rank(),
        "reduce: axis pre-resolved by op layer"
    );
    let cpu = cpu_storage(x, "reduce")?;
    let storage = match cpu {
        CpuStorage::F16(v) => {
            CpuStorage::F16(std::sync::Arc::new(reduce_generic(op, v, layout, axis)))
        }
        CpuStorage::BF16(v) => {
            CpuStorage::BF16(std::sync::Arc::new(reduce_generic(op, v, layout, axis)))
        }
        CpuStorage::F32(v) => {
            CpuStorage::F32(std::sync::Arc::new(reduce_generic(op, v, layout, axis)))
        }
        CpuStorage::F64(v) => {
            CpuStorage::F64(std::sync::Arc::new(reduce_generic(op, v, layout, axis)))
        }
        CpuStorage::I64(v) => {
            CpuStorage::I64(std::sync::Arc::new(reduce_generic(op, v, layout, axis)))
        }
        CpuStorage::Bool(_) => {
            return Err(Error::Unsupported {
                op: "reduce",
                device: x.device(),
                dtype: DType::Bool,
            });
        }
    };
    Ok(Storage::Cpu(storage))
}

/// See [`BackendOps::arg_reduce`](crate::backend::BackendOps::arg_reduce).
pub(crate) fn arg_reduce(op: ArgReduceOp, x: View<'_>, axis: usize) -> Result<Storage> {
    let layout = x.layout();
    debug_assert!(
        axis < layout.rank(),
        "arg_reduce: axis pre-resolved by op layer"
    );
    let cpu = cpu_storage(x, "arg_reduce")?;
    let positions = match cpu {
        CpuStorage::F16(v) => arg_reduce_generic(op, v, layout, axis),
        CpuStorage::BF16(v) => arg_reduce_generic(op, v, layout, axis),
        CpuStorage::F32(v) => arg_reduce_generic(op, v, layout, axis),
        CpuStorage::F64(v) => arg_reduce_generic(op, v, layout, axis),
        CpuStorage::I64(v) => arg_reduce_generic(op, v, layout, axis),
        CpuStorage::Bool(_) => {
            return Err(Error::Unsupported {
                op: "arg_reduce",
                device: x.device(),
                dtype: DType::Bool,
            });
        }
    };
    Ok(Storage::Cpu(CpuStorage::I64(std::sync::Arc::new(
        positions,
    ))))
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::View;
    use crate::layout::Layout;
    use std::sync::Arc;

    // ----- helpers ------------------------------------------------------

    fn f32_view<'a>(storage: &'a Storage, layout: &'a Layout) -> View<'a> {
        View::new(storage, layout)
    }

    fn f32_storage(data: Vec<f32>) -> Storage {
        Storage::Cpu(CpuStorage::F32(Arc::new(data)))
    }

    fn as_f32(s: &Storage) -> Vec<f32> {
        match s {
            Storage::Cpu(CpuStorage::F32(v)) => v.as_ref().clone(),
            _ => panic!("expected f32 storage"),
        }
    }

    fn as_i64(s: &Storage) -> Vec<i64> {
        match s {
            Storage::Cpu(CpuStorage::I64(v)) => v.as_ref().clone(),
            _ => panic!("expected i64 storage"),
        }
    }

    // ----- golden values ------------------------------------------------

    #[test]
    fn sum_over_axes_of_2x3() {
        // [[1,2,3],[4,5,6]]
        let s = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let l = Layout::contiguous([2, 3]).unwrap();
        // sum axis 0 -> [5, 7, 9]
        let r = reduce(ReduceOp::Sum, f32_view(&s, &l), 0).unwrap();
        assert_eq!(as_f32(&r), vec![5.0, 7.0, 9.0]);
        // sum axis 1 -> [6, 15]
        let r = reduce(ReduceOp::Sum, f32_view(&s, &l), 1).unwrap();
        assert_eq!(as_f32(&r), vec![6.0, 15.0]);
    }

    #[test]
    fn mean_max_min_over_axis() {
        let s = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let l = Layout::contiguous([2, 3]).unwrap();
        let r = reduce(ReduceOp::Mean, f32_view(&s, &l), 1).unwrap();
        assert_eq!(as_f32(&r), vec![2.0, 5.0]);
        let r = reduce(ReduceOp::Max, f32_view(&s, &l), 0).unwrap();
        assert_eq!(as_f32(&r), vec![4.0, 5.0, 6.0]);
        let r = reduce(ReduceOp::Min, f32_view(&s, &l), 1).unwrap();
        assert_eq!(as_f32(&r), vec![1.0, 4.0]);
    }

    #[test]
    fn reduce_rank1_to_scalar() {
        let s = f32_storage(vec![2.0, 5.0, 1.0, 4.0]);
        let l = Layout::contiguous([4]).unwrap();
        let r = reduce(ReduceOp::Sum, f32_view(&s, &l), 0).unwrap();
        // rank-1 reduction over the only axis -> rank-0 scalar (one element).
        assert_eq!(as_f32(&r), vec![12.0]);
        let r = reduce(ReduceOp::Max, f32_view(&s, &l), 0).unwrap();
        assert_eq!(as_f32(&r), vec![5.0]);
    }

    #[test]
    fn reduce_middle_axis_of_rank3() {
        // shape [2,3,2], sum over axis 1 -> [2,2].
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let s = f32_storage(data);
        let l = Layout::contiguous([2, 3, 2]).unwrap();
        let r = reduce(ReduceOp::Sum, f32_view(&s, &l), 1).unwrap();
        // out[b,c] = data[b,0,c]+data[b,1,c]+data[b,2,c]
        // b=0: (0+2+4, 1+3+5) = (6,9); b=1: (6+8+10, 7+9+11) = (24,27)
        assert_eq!(as_f32(&r), vec![6.0, 9.0, 24.0, 27.0]);
    }

    // ----- strided / permuted correctness -------------------------------

    #[test]
    fn reduce_over_transposed_view() {
        // Backing [[1,2,3],[4,5,6]] viewed transposed as [3,2]:
        // [[1,4],[2,5],[3,6]]. Sum over axis 1 -> [5,7,9].
        let s = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let base = Layout::contiguous([2, 3]).unwrap();
        let t = base.transpose(0, 1).unwrap();
        let r = reduce(ReduceOp::Sum, f32_view(&s, &t), 1).unwrap();
        assert_eq!(as_f32(&r), vec![5.0, 7.0, 9.0]);
        // Sum over axis 0 of the transposed view -> [6,15].
        let r = reduce(ReduceOp::Sum, f32_view(&s, &t), 0).unwrap();
        assert_eq!(as_f32(&r), vec![6.0, 15.0]);
    }

    #[test]
    fn reduce_over_narrowed_view() {
        // 3x4, narrow axis 1 to [1,3) -> 3x2 view, then sum over axis 1.
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let s = f32_storage(data);
        let base = Layout::contiguous([3, 4]).unwrap();
        let n = base.narrow(1, 1, 2).unwrap();
        // rows: [1,2],[5,6],[9,10]; sum axis1 -> [3,11,19]
        let r = reduce(ReduceOp::Sum, f32_view(&s, &n), 1).unwrap();
        assert_eq!(as_f32(&r), vec![3.0, 11.0, 19.0]);
    }

    #[test]
    fn reduce_over_broadcast_axis() {
        // A size-1 axis broadcast to 4: summing over it multiplies by 4.
        let s = f32_storage(vec![2.0, 3.0]);
        let base = Layout::contiguous([2, 1]).unwrap();
        let b = base.broadcast_to(&Shape::from([2, 4])).unwrap();
        let r = reduce(ReduceOp::Sum, f32_view(&s, &b), 1).unwrap();
        assert_eq!(as_f32(&r), vec![8.0, 12.0]);
        // Max over a broadcast axis is the single repeated value.
        let r = reduce(ReduceOp::Max, f32_view(&s, &b), 1).unwrap();
        assert_eq!(as_f32(&r), vec![2.0, 3.0]);
    }

    #[test]
    fn reduce_matches_naive_reference_random() {
        // Cross-check against a from-scratch reference over random strided
        // views (permute + narrow), for every axis and op.
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
            fn f(&mut self) -> f32 {
                (self.next() % 2000) as f32 / 100.0 - 10.0
            }
        }
        let mut rng = Prng(0xDEADBEEF);
        for _ in 0..300 {
            let dims = [2 + rng.below(3), 2 + rng.below(3), 2 + rng.below(3)];
            let numel: usize = dims.iter().product();
            let data: Vec<f32> = (0..numel).map(|_| rng.f()).collect();
            let s = f32_storage(data.clone());
            let base = Layout::contiguous(dims).unwrap();
            // Random view: identity, transpose, or narrow.
            let layout = match rng.below(3) {
                0 => base.clone(),
                1 => base.transpose(0, 2).unwrap(),
                _ => {
                    let a = rng.below(3);
                    let size = base.dims()[a];
                    base.narrow(a, 0, 1 + rng.below(size)).unwrap()
                }
            };
            for axis in 0..3 {
                for op in [ReduceOp::Sum, ReduceOp::Max, ReduceOp::Min] {
                    let got = as_f32(&reduce(op, f32_view(&s, &layout), axis).unwrap());
                    let expected = naive_reduce(op, &data, &base, &layout, axis);
                    for (g, e) in got.iter().zip(expected.iter()) {
                        assert!(
                            (g - e).abs() < 1e-4,
                            "mismatch op {op:?} axis {axis}: {g} vs {e}"
                        );
                    }
                }
            }
        }

        // Naive reference: materialize the logical view to a dense Vec, then
        // reduce with straightforward nested loops.
        fn naive_reduce(
            op: ReduceOp,
            backing: &[f32],
            _base: &Layout,
            layout: &Layout,
            axis: usize,
        ) -> Vec<f32> {
            let dims = layout.dims();
            let strides = layout.strides();
            let off = layout.offset();
            let out_dims: Vec<usize> = dims
                .iter()
                .enumerate()
                .filter(|&(a, _)| a != axis)
                .map(|(_, &d)| d)
                .collect();
            let out_len: usize = out_dims.iter().product::<usize>().max(1);
            let mut out = vec![0.0f32; out_len];
            for (oi, slot) in out.iter_mut().enumerate() {
                // Decode oi over out_dims (row-major).
                let mut coords = vec![0usize; dims.len()];
                let mut rem = oi;
                for a in (0..dims.len()).rev() {
                    if a == axis {
                        continue;
                    }
                    coords[a] = rem % dims[a];
                    rem /= dims[a];
                }
                let mut acc = match op {
                    ReduceOp::Sum | ReduceOp::Mean => 0.0,
                    ReduceOp::Max => f32::NEG_INFINITY,
                    ReduceOp::Min => f32::INFINITY,
                };
                for p in 0..dims[axis] {
                    coords[axis] = p;
                    let idx = off
                        + coords
                            .iter()
                            .zip(strides.iter())
                            .map(|(c, st)| c * st)
                            .sum::<usize>();
                    let v = backing[idx];
                    acc = match op {
                        ReduceOp::Sum | ReduceOp::Mean => acc + v,
                        ReduceOp::Max => acc.max(v),
                        ReduceOp::Min => acc.min(v),
                    };
                }
                *slot = acc;
            }
            out
        }
    }

    // ----- Acc contract -------------------------------------------------

    #[test]
    fn f16_sum_does_not_saturate() {
        // The whole point of the Acc contract: 4096 copies of 1.0 summed in
        // f16 native arithmetic saturates (f16 cannot represent 2049), but
        // accumulating in f32 and casting once at output yields the exact
        // 4096. 4096 IS representable in f16, so from_acc round-trips.
        let n = 4096usize;
        let data = vec![half::f16::from_f32(1.0); n];
        let s = Storage::Cpu(CpuStorage::F16(Arc::new(data)));
        let l = Layout::contiguous([n]).unwrap();
        let r = reduce(ReduceOp::Sum, View::new(&s, &l), 0).unwrap();
        let out = match &r {
            Storage::Cpu(CpuStorage::F16(v)) => v.as_ref().clone(),
            _ => panic!("expected f16"),
        };
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].to_f32(), 4096.0);
    }

    #[test]
    fn bf16_sum_does_not_stall_at_256() {
        let n = 4096usize;
        let data = vec![half::bf16::from_f32(1.0); n];
        let s = Storage::Cpu(CpuStorage::BF16(Arc::new(data)));
        let l = Layout::contiguous([n]).unwrap();
        let r = reduce(ReduceOp::Sum, View::new(&s, &l), 0).unwrap();
        let Storage::Cpu(CpuStorage::BF16(out)) = r else {
            panic!("expected bf16")
        };
        assert_eq!(out[0].to_f32(), 4096.0);
    }

    #[test]
    fn f16_mean_uses_wide_count() {
        // Mean of 3000 copies of 3.0: the sum (9000) overflows f16's exact
        // integer range, but the wide accumulator + wide count divide gives
        // exactly 3.0 back.
        let n = 3000usize;
        let data = vec![half::f16::from_f32(3.0); n];
        let s = Storage::Cpu(CpuStorage::F16(Arc::new(data)));
        let l = Layout::contiguous([n]).unwrap();
        let r = reduce(ReduceOp::Mean, View::new(&s, &l), 0).unwrap();
        let out = match &r {
            Storage::Cpu(CpuStorage::F16(v)) => v.as_ref().clone(),
            _ => panic!("expected f16"),
        };
        assert_eq!(out[0].to_f32(), 3.0);
    }

    #[test]
    fn large_count_f32_sum_exactness() {
        // Sum of exactly-representable f32 integers whose total exceeds the
        // f32 contiguous-integer range (2^24). Because f32's Acc is itself,
        // this documents the native-precision behaviour; it is exact here
        // because every partial sum stays an even integer within range.
        let n = 1 << 20; // 1,048,576
        let data = vec![2.0f32; n];
        let s = f32_storage(data);
        let l = Layout::contiguous([n]).unwrap();
        let r = reduce(ReduceOp::Sum, View::new(&s, &l), 0).unwrap();
        assert_eq!(as_f32(&r)[0], (2 * n) as f32);
    }

    #[test]
    fn i64_sum_accumulates_in_i64() {
        // Values whose sum exceeds i32 range: accumulation stays in i64.
        let data = vec![1_000_000_000i64; 5];
        let s = Storage::Cpu(CpuStorage::I64(Arc::new(data)));
        let l = Layout::contiguous([5]).unwrap();
        let r = reduce(ReduceOp::Sum, View::new(&s, &l), 0).unwrap();
        assert_eq!(as_i64(&r), vec![5_000_000_000]);
    }

    // ----- arg reductions ----------------------------------------------

    #[test]
    fn argmax_argmin_return_i64_positions() {
        // [[1,5,2],[4,0,6]]
        let s = f32_storage(vec![1.0, 5.0, 2.0, 4.0, 0.0, 6.0]);
        let l = Layout::contiguous([2, 3]).unwrap();
        // argmax axis 1 -> row maxima at positions [1, 2]
        let r = arg_reduce(ArgReduceOp::ArgMax, f32_view(&s, &l), 1).unwrap();
        assert_eq!(as_i64(&r), vec![1, 2]);
        // argmin axis 1 -> [0, 1]
        let r = arg_reduce(ArgReduceOp::ArgMin, f32_view(&s, &l), 1).unwrap();
        assert_eq!(as_i64(&r), vec![0, 1]);
        // argmax axis 0 -> per column: col0 max at row1(4>1), col1 row0(5>0),
        // col2 row1(6>2) -> [1,0,1]
        let r = arg_reduce(ArgReduceOp::ArgMax, f32_view(&s, &l), 0).unwrap();
        assert_eq!(as_i64(&r), vec![1, 0, 1]);
    }

    #[test]
    fn argmax_first_on_ties() {
        // All-equal axis: first index wins for both argmax and argmin.
        let s = f32_storage(vec![7.0, 7.0, 7.0, 7.0]);
        let l = Layout::contiguous([4]).unwrap();
        let r = arg_reduce(ArgReduceOp::ArgMax, f32_view(&s, &l), 0).unwrap();
        assert_eq!(as_i64(&r), vec![0]);
        let r = arg_reduce(ArgReduceOp::ArgMin, f32_view(&s, &l), 0).unwrap();
        assert_eq!(as_i64(&r), vec![0]);
    }

    #[test]
    fn argmax_over_strided_view() {
        // Transposed view: argmax along the (transposed) axis.
        let s = f32_storage(vec![1.0, 5.0, 2.0, 4.0, 0.0, 6.0]); // 2x3
        let base = Layout::contiguous([2, 3]).unwrap();
        let t = base.transpose(0, 1).unwrap(); // 3x2: [[1,4],[5,0],[2,6]]
        // argmax axis 1 -> [1,0,1]
        let r = arg_reduce(ArgReduceOp::ArgMax, f32_view(&s, &t), 1).unwrap();
        assert_eq!(as_i64(&r), vec![1, 0, 1]);
    }

    #[test]
    fn argmax_on_i64_input() {
        let s = Storage::Cpu(CpuStorage::I64(Arc::new(vec![3, 9, 1, 9])));
        let l = Layout::contiguous([4]).unwrap();
        let r = arg_reduce(ArgReduceOp::ArgMax, View::new(&s, &l), 0).unwrap();
        assert_eq!(as_i64(&r), vec![1]); // first 9
    }

    /// `max`/`min` propagate NaN, so `argmax`/`argmin` must select the NaN's
    /// index — otherwise `x.max(axis)` and `x.gather(axis, x.argmax(axis))`
    /// name different elements. Both directions, and the first NaN on ties.
    #[test]
    fn nan_propagates_in_extrema_and_is_selected_by_both_arg_reductions() {
        let s = f32_storage(vec![1.0, f32::NAN, 3.0]);
        let l = Layout::contiguous([3]).unwrap();

        let r = reduce(ReduceOp::Max, f32_view(&s, &l), 0).unwrap();
        assert!(as_f32(&r)[0].is_nan(), "max must propagate NaN");
        let r = reduce(ReduceOp::Min, f32_view(&s, &l), 0).unwrap();
        assert!(as_f32(&r)[0].is_nan(), "min must propagate NaN");

        let r = arg_reduce(ArgReduceOp::ArgMax, f32_view(&s, &l), 0).unwrap();
        assert_eq!(as_i64(&r), vec![1], "argmax must select the NaN");
        let r = arg_reduce(ArgReduceOp::ArgMin, f32_view(&s, &l), 0).unwrap();
        assert_eq!(as_i64(&r), vec![1], "argmin must select the NaN");

        // NaN wins even when it is not first, and the *first* NaN wins a tie.
        let s = f32_storage(vec![5.0, 1.0, f32::NAN, f32::NAN]);
        let l = Layout::contiguous([4]).unwrap();
        for op in [ArgReduceOp::ArgMax, ArgReduceOp::ArgMin] {
            let r = arg_reduce(op, f32_view(&s, &l), 0).unwrap();
            assert_eq!(as_i64(&r), vec![2], "{op:?} must pick the first NaN");
        }

        // A NaN in one line must not affect a clean neighbouring line.
        let s = f32_storage(vec![1.0, f32::NAN, 3.0, 4.0, 5.0, 6.0]);
        let l = Layout::contiguous([2, 3]).unwrap();
        let r = arg_reduce(ArgReduceOp::ArgMax, f32_view(&s, &l), 1).unwrap();
        assert_eq!(as_i64(&r), vec![1, 2], "clean rows keep ordinary argmax");
    }

    #[test]
    fn bool_reduction_is_unsupported() {
        let s = Storage::Cpu(CpuStorage::Bool(Arc::new(vec![true, false])));
        let l = Layout::contiguous([2]).unwrap();
        assert!(matches!(
            reduce(ReduceOp::Sum, View::new(&s, &l), 0),
            Err(Error::Unsupported {
                op: "reduce",
                dtype: DType::Bool,
                ..
            })
        ));
        assert!(matches!(
            arg_reduce(ArgReduceOp::ArgMax, View::new(&s, &l), 0),
            Err(Error::Unsupported {
                op: "arg_reduce",
                dtype: DType::Bool,
                ..
            })
        ));
    }
}
