//! Host-boundary CPU kernels: transfer, contiguous materialization,
//! fills, and dtype casts.
//!
//! Signatures frozen by T01; **T10a** fills the bodies. Semantics are
//! specified on [`BackendOps`](crate::backend::BackendOps) — these are the
//! delegation targets of `CpuBackend`.

use crate::backend::View;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::storage::{CpuStorage, Storage};

/// Extract the [`CpuStorage`] backing a view.
///
/// The CPU backend only ever receives CPU-resident views (the dispatcher
/// routes each device to its own backend), so a non-CPU storage here is an
/// internal contract violation and panics rather than returning an error.
fn cpu_storage<'a>(x: &View<'a>) -> &'a CpuStorage {
    match x.storage() {
        Storage::Cpu(s) => s,
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Storage::Metal(_) => {
            unreachable!("CPU backend received non-CPU storage; dispatcher invariant violated")
        }
    }
}

/// Incremental row-major odometer over `dims`, carrying one running storage
/// index per *lane*.
///
/// Each lane pairs a stride slice with a running index seeded from that lane's
/// storage offset. [`Walk::step`] advances the logical position by one
/// (rightmost axis fastest) and fixes up every lane's index with additions and
/// subtractions of strides only — never the
/// `offset + Σ (i / place[a]) % dims[a] * stride[a]` recomputation (two
/// divisions per axis) that the first version of these kernels paid *per
/// element*. This is the shared walk behind [`copy_view`] and the
/// `super::index` kernels.
///
/// A lane's stride slice may be **longer** than `dims`; only its first
/// `dims.len()` entries are read, so a caller can walk the outer axes of a
/// full stride vector without copying it. `dims` may be empty (a rank-0 view,
/// or a walk whose axes are all covered by an inner block): the walk then has
/// exactly one position.
///
/// Stepping past the last position wraps back to the start instead of
/// panicking; callers drive exactly `dims.iter().product()` positions.
pub(super) struct Walk<'a, const L: usize> {
    dims: &'a [usize],
    strides: [&'a [usize]; L],
    coords: Vec<usize>,
    index: [usize; L],
}

impl<'a, const L: usize> Walk<'a, L> {
    /// Start a walk at the all-zero coordinate, with lane `l` addressing
    /// `strides[l]` from `start[l]`.
    pub(super) fn new(dims: &'a [usize], strides: [&'a [usize]; L], start: [usize; L]) -> Self {
        debug_assert!(strides.iter().all(|s| s.len() >= dims.len()));
        Walk {
            dims,
            strides,
            coords: vec![0usize; dims.len()],
            index: start,
        }
    }

    /// Lane `lane`'s storage index at the current logical position.
    #[inline]
    pub(super) fn index(&self, lane: usize) -> usize {
        self.index[lane]
    }

    /// The current coordinate on `axis`.
    #[inline]
    pub(super) fn coord(&self, axis: usize) -> usize {
        self.coords[axis]
    }

    /// Advance one logical position (rightmost axis fastest).
    #[inline]
    pub(super) fn step(&mut self) {
        let mut axis = self.coords.len();
        while axis > 0 {
            axis -= 1;
            if self.coords[axis] + 1 < self.dims[axis] {
                self.coords[axis] += 1;
                for l in 0..L {
                    self.index[l] += self.strides[l][axis];
                }
                return;
            }
            // Carry: rewind this axis. Subtracting the contribution already
            // accumulated (rather than adding a step and correcting) keeps
            // every lane's index inside its addressable range at all times.
            for l in 0..L {
                self.index[l] -= self.coords[axis] * self.strides[l][axis];
            }
            self.coords[axis] = 0;
        }
    }
}

/// The storage offset of a view whose row-major logical walk *is* the storage
/// run `offset .. offset + num_elements`, or `None` when it is not.
///
/// This is the `memcpy` predicate for [`copy_view`]: a layout qualifies when
/// every axis of size > 1 carries the canonical row-major stride (computed
/// right-to-left from a unit innermost stride). Size-1 axes are skipped
/// because their only coordinate is 0, so their stride never contributes to an
/// address — which makes this strictly more permissive than
/// `Layout::is_contiguous` (it also accepts any non-zero offset, e.g. a
/// `narrow` of the outermost axis, and an unsqueezed axis carrying a
/// non-canonical stride). Broadcast axes (stride 0 with size > 1) never
/// qualify: they repeat elements and so cannot be a straight run.
///
/// Shared with the `super::index` kernels, which use it to seed an
/// accumulator from a dense base tensor without a coordinate walk.
pub(super) fn dense_offset(layout: &Layout) -> Option<usize> {
    let mut expected = 1usize;
    for (&dim, &stride) in layout.dims().iter().zip(layout.strides()).rev() {
        if dim == 1 {
            continue;
        }
        if stride != expected {
            return None;
        }
        expected = expected.checked_mul(dim)?;
    }
    Some(layout.offset())
}

/// Copy the logical elements a view addresses out of its full backing buffer
/// `src` into a fresh row-major `Vec`.
///
/// Three tiers, in cost order, all producing exactly the same elements in
/// exactly the same order (the difference is only how many at a time):
///
/// 1. **Whole-view run** ([`dense_offset`]): one `[T]::to_vec`, i.e. one
///    `memcpy`. This is the overwhelmingly common case — every freshly
///    allocated tensor and every `narrow` of the outermost axis.
/// 2. **Row runs**: the longest *trailing* group of axes that is contiguous in
///    storage is copied with one `extend_from_slice` per run, and only the
///    remaining outer axes are walked. A transposed matrix of rows, a
///    broadcast over leading axes, and an inner-axis `narrow` all land here.
/// 3. **Element by element**: when even the innermost axis is strided the run
///    length is 1 and this degenerates to the odometer walk, still without the
///    O(numel) index buffer the first implementation allocated.
///
/// The outer walk is a [`Walk`], so no division is executed per output element
/// in any tier.
fn copy_view<T: Copy>(src: &[T], layout: &Layout) -> Vec<T> {
    let total = layout.num_elements();
    if total == 0 {
        return Vec::new();
    }
    if let Some(start) = dense_offset(layout) {
        return src[start..start + total].to_vec();
    }

    let dims = layout.dims();
    let strides = layout.strides();
    // Longest trailing group of axes that is contiguous in storage: `run`
    // elements per copy, `dims[..outer]` left for the odometer.
    let mut run = 1usize;
    let mut outer = dims.len();
    for a in (0..dims.len()).rev() {
        if dims[a] != 1 {
            if strides[a] != run {
                break;
            }
            run *= dims[a];
        }
        outer = a;
    }

    let mut out = Vec::with_capacity(total);
    let mut walk = Walk::new(&dims[..outer], [strides], [layout.offset()]);
    for _ in 0..total / run {
        let at = walk.index(0);
        out.extend_from_slice(&src[at..at + run]);
        walk.step();
    }
    out
}

/// Whether `layout` addresses **the whole of** a buffer of `len` elements, in
/// order — the one case a row-major materialization can skip entirely.
fn is_whole_buffer(layout: &Layout, len: usize) -> bool {
    layout.num_elements() == len && dense_offset(layout) == Some(0)
}

/// Row-major materialization of one typed buffer under `layout`, for the
/// *host-interchange* path.
///
/// When the view is the whole buffer in order the buffer is **shared** (an
/// `Arc` bump, no copy at all). That is sound because storage buffers are
/// immutable once constructed — nothing in the crate takes a mutable borrow of
/// a `CpuStorage` payload, and every host reader clones for itself
/// (`HostConv::try_from_cpu_storage`) — so sharing is indistinguishable from
/// copying apart from the cost. `transfer_out`'s contract is "download as a
/// contiguous `CpuStorage`", which this satisfies; it promises no fresh
/// allocation, unlike [`copy_strided`].
///
/// This is what makes `cat`/`stack` cheap: their host assembly calls
/// `transfer_out` once per part, and for the usual dense part that is now free
/// instead of a per-element gather.
fn share_or_copy<T: Copy>(buf: &std::sync::Arc<Vec<T>>, layout: &Layout) -> std::sync::Arc<Vec<T>> {
    if is_whole_buffer(layout, buf.len()) {
        return std::sync::Arc::clone(buf);
    }
    std::sync::Arc::new(copy_view(buf, layout))
}

/// Row-major materialization into a **freshly allocated** buffer, for
/// [`copy_strided`], whose contract explicitly promises a fresh buffer.
///
/// The whole-buffer case still copies here, so `contiguous()` and the copying
/// branch of `reshape` keep handing back storage that shares nothing with
/// their input. Everything strided takes the same [`copy_view`] tiers as
/// [`share_or_copy`], so the fast paths are not given up — only the
/// zero-copy case is, and that case is unreachable from `copy_strided`'s
/// callers anyway (both test contiguity first).
fn copy_owned<T: Copy>(buf: &std::sync::Arc<Vec<T>>, layout: &Layout) -> std::sync::Arc<Vec<T>> {
    if is_whole_buffer(layout, buf.len()) {
        return std::sync::Arc::new(buf.as_ref().clone());
    }
    std::sync::Arc::new(copy_view(buf, layout))
}

/// Materialize a view into a contiguous row-major [`CpuStorage`], preserving
/// dtype, dispatching each dtype arm to `per_buffer`.
///
/// The two callers differ only in whether the whole-buffer case may share:
/// [`transfer_out`] passes [`share_or_copy`], [`copy_strided`] passes
/// [`copy_owned`].
macro_rules! materialize_with {
    ($x:expr, $per_buffer:ident) => {{
        let x = $x;
        let storage = cpu_storage(&x);
        let layout = x.layout();
        match storage {
            CpuStorage::F16(v) => CpuStorage::F16($per_buffer(v, layout)),
            CpuStorage::BF16(v) => CpuStorage::BF16($per_buffer(v, layout)),
            CpuStorage::F32(v) => CpuStorage::F32($per_buffer(v, layout)),
            CpuStorage::F64(v) => CpuStorage::F64($per_buffer(v, layout)),
            CpuStorage::I64(v) => CpuStorage::I64($per_buffer(v, layout)),
            CpuStorage::Bool(v) => CpuStorage::Bool($per_buffer(v, layout)),
        }
    }};
}

/// See [`BackendOps::transfer_in`](crate::backend::BackendOps::transfer_in).
/// For CPU this wraps the buffer unchanged.
pub(crate) fn transfer_in(host: CpuStorage) -> Result<Storage> {
    Ok(Storage::Cpu(host))
}

/// See [`BackendOps::transfer_out`](crate::backend::BackendOps::transfer_out).
pub(crate) fn transfer_out(x: View<'_>) -> Result<CpuStorage> {
    Ok(materialize_with!(x, share_or_copy))
}

/// See [`BackendOps::copy_strided`](crate::backend::BackendOps::copy_strided).
pub(crate) fn copy_strided(x: View<'_>) -> Result<Storage> {
    Ok(Storage::Cpu(materialize_with!(x, copy_owned)))
}

fn copy_into_typed<T: Copy>(src: &[T], src_layout: &Layout, dst: &mut [T], dst_layout: &Layout) {
    let len = src_layout.num_elements();
    if len == 0 {
        return;
    }
    if let (Some(from), Some(to)) = (dense_offset(src_layout), dense_offset(dst_layout)) {
        dst[to..to + len].copy_from_slice(&src[from..from + len]);
        return;
    }
    let mut walk = Walk::new(
        src_layout.dims(),
        [src_layout.strides(), dst_layout.strides()],
        [src_layout.offset(), dst_layout.offset()],
    );
    for _ in 0..len {
        dst[walk.index(1)] = src[walk.index(0)];
        walk.step();
    }
}

/// Device-independent destination-copy contract used by `cat`/`stack`.
#[allow(clippy::infallible_destructuring_match)]
pub(crate) fn copy_into(src: View<'_>, dst: &mut Storage, dst_layout: &Layout) -> Result<()> {
    if src.layout().shape() != dst_layout.shape() {
        return Err(Error::ShapeMismatch {
            op: "copy_into",
            lhs: src.layout().shape().clone(),
            rhs: dst_layout.shape().clone(),
        });
    }
    if src.dtype() != dst.dtype() {
        return Err(Error::DTypeMismatch {
            op: "copy_into",
            expected: src.dtype(),
            got: dst.dtype(),
        });
    }
    let src_layout = src.layout();
    let src = cpu_storage(&src);
    let dst = match dst {
        Storage::Cpu(dst) => dst,
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Storage::Metal(storage) => {
            return Err(Error::DeviceMismatch {
                op: "copy_into",
                expected: crate::Device::Cpu,
                got: storage.device(),
            });
        }
    };
    macro_rules! copy {
        ($from:expr, $to:expr) => {{
            let to = std::sync::Arc::make_mut($to);
            copy_into_typed($from, src_layout, to.as_mut_slice(), dst_layout)
        }};
    }
    match (src, dst) {
        (CpuStorage::F16(from), CpuStorage::F16(to)) => copy!(from, to),
        (CpuStorage::BF16(from), CpuStorage::BF16(to)) => copy!(from, to),
        (CpuStorage::F32(from), CpuStorage::F32(to)) => copy!(from, to),
        (CpuStorage::F64(from), CpuStorage::F64(to)) => copy!(from, to),
        (CpuStorage::I64(from), CpuStorage::I64(to)) => copy!(from, to),
        (CpuStorage::Bool(from), CpuStorage::Bool(to)) => copy!(from, to),
        _ => unreachable!("copy_into dtype validated"),
    }
    Ok(())
}

/// See [`BackendOps::full`](crate::backend::BackendOps::full).
pub(crate) fn full(len: usize, dtype: DType, value: f64) -> Result<Storage> {
    let storage = match dtype {
        DType::F16 => CpuStorage::F16(std::sync::Arc::new(vec![half::f16::from_f64(value); len])),
        DType::BF16 => {
            CpuStorage::BF16(std::sync::Arc::new(vec![half::bf16::from_f64(value); len]))
        }
        DType::F32 => CpuStorage::F32(std::sync::Arc::new(vec![value as f32; len])),
        DType::F64 => CpuStorage::F64(std::sync::Arc::new(vec![value; len])),
        DType::I64 => CpuStorage::I64(std::sync::Arc::new(vec![value as i64; len])),
        DType::Bool => CpuStorage::Bool(std::sync::Arc::new(vec![value != 0.0; len])),
    };
    Ok(Storage::Cpu(storage))
}

/// See [`BackendOps::cast`](crate::backend::BackendOps::cast). Supports the
/// reduced-precision matrix used by public algorithms: F16/BF16 with F32,
/// each other, I64, and Bool, plus the existing F32/I64/Bool lanes.
///
/// The source view is first materialized in row-major order, then each
/// element is converted with familiar Rust/PyTorch semantics:
/// float→int is a saturating truncation toward zero (Rust `as`), numeric→bool
/// is `x != 0`, and bool→numeric is `0`/`1`. An identity cast (`to` equals
/// the source dtype) returns a contiguous copy. F64 lanes remain outside this
/// task's cast scope and return [`Error::Unsupported`](crate::Error::Unsupported),
/// never a silent reinterpretation.
pub(crate) fn cast(x: View<'_>, to: DType) -> Result<Storage> {
    let from = x.dtype();
    let unsupported = || Error::Unsupported {
        op: "to_dtype",
        device: x.device(),
        dtype: from,
    };

    // Materialize the (possibly strided) source contiguously first so the
    // per-lane conversion is a flat map. `copy_owned`, not `share_or_copy`:
    // the identity lane hands `src` straight back as the result storage, and
    // this function documents that as "a contiguous copy".
    let src = materialize_with!(x, copy_owned);

    let out = match (from, to) {
        // Identity: a contiguous copy, no conversion.
        (a, b) if a == b => src,

        // Reduced floats <-> F32 and each other.
        (DType::F16, DType::F32) => match &src {
            CpuStorage::F16(v) => {
                CpuStorage::F32(std::sync::Arc::new(v.iter().map(|e| e.to_f32()).collect()))
            }
            _ => unreachable!("materialized dtype disagrees with view dtype"),
        },
        (DType::F32, DType::F16) => match &src {
            CpuStorage::F32(v) => CpuStorage::F16(std::sync::Arc::new(
                v.iter().map(|&e| half::f16::from_f32(e)).collect(),
            )),
            _ => unreachable!("materialized dtype disagrees with view dtype"),
        },
        (DType::BF16, DType::F32) => match &src {
            CpuStorage::BF16(v) => {
                CpuStorage::F32(std::sync::Arc::new(v.iter().map(|e| e.to_f32()).collect()))
            }
            _ => unreachable!("materialized dtype disagrees with view dtype"),
        },
        (DType::F32, DType::BF16) => match &src {
            CpuStorage::F32(v) => CpuStorage::BF16(std::sync::Arc::new(
                v.iter().map(|&e| half::bf16::from_f32(e)).collect(),
            )),
            _ => unreachable!("materialized dtype disagrees with view dtype"),
        },
        (DType::F16, DType::BF16) => match &src {
            CpuStorage::F16(v) => CpuStorage::BF16(std::sync::Arc::new(
                v.iter().map(|e| half::bf16::from_f32(e.to_f32())).collect(),
            )),
            _ => unreachable!("materialized dtype disagrees with view dtype"),
        },
        (DType::BF16, DType::F16) => match &src {
            CpuStorage::BF16(v) => CpuStorage::F16(std::sync::Arc::new(
                v.iter().map(|e| half::f16::from_f32(e.to_f32())).collect(),
            )),
            _ => unreachable!("materialized dtype disagrees with view dtype"),
        },

        // Float <-> I64.
        (DType::F32, DType::I64) => match &src {
            CpuStorage::F32(v) => {
                CpuStorage::I64(std::sync::Arc::new(v.iter().map(|&e| e as i64).collect()))
            }
            _ => unreachable!("materialized dtype disagrees with view dtype"),
        },
        (DType::I64, DType::F32) => match &src {
            CpuStorage::I64(v) => {
                CpuStorage::F32(std::sync::Arc::new(v.iter().map(|&e| e as f32).collect()))
            }
            _ => unreachable!("materialized dtype disagrees with view dtype"),
        },
        (DType::F16, DType::I64) => match &src {
            CpuStorage::F16(v) => CpuStorage::I64(std::sync::Arc::new(
                v.iter().map(|e| e.to_f32() as i64).collect(),
            )),
            _ => unreachable!("materialized dtype disagrees with view dtype"),
        },
        (DType::I64, DType::F16) => match &src {
            CpuStorage::I64(v) => CpuStorage::F16(std::sync::Arc::new(
                v.iter().map(|&e| half::f16::from_f32(e as f32)).collect(),
            )),
            _ => unreachable!("materialized dtype disagrees with view dtype"),
        },
        (DType::BF16, DType::I64) => match &src {
            CpuStorage::BF16(v) => CpuStorage::I64(std::sync::Arc::new(
                v.iter().map(|e| e.to_f32() as i64).collect(),
            )),
            _ => unreachable!("materialized dtype disagrees with view dtype"),
        },
        (DType::I64, DType::BF16) => match &src {
            CpuStorage::I64(v) => CpuStorage::BF16(std::sync::Arc::new(
                v.iter().map(|&e| half::bf16::from_f32(e as f32)).collect(),
            )),
            _ => unreachable!("materialized dtype disagrees with view dtype"),
        },

        // F32 <-> Bool.
        (DType::F32, DType::Bool) => match &src {
            CpuStorage::F32(v) => {
                CpuStorage::Bool(std::sync::Arc::new(v.iter().map(|&e| e != 0.0).collect()))
            }
            _ => unreachable!("materialized dtype disagrees with view dtype"),
        },
        (DType::Bool, DType::F32) => match &src {
            CpuStorage::Bool(v) => CpuStorage::F32(std::sync::Arc::new(
                v.iter().map(|&e| if e { 1.0 } else { 0.0 }).collect(),
            )),
            _ => unreachable!("materialized dtype disagrees with view dtype"),
        },
        (DType::F16, DType::Bool) => match &src {
            CpuStorage::F16(v) => CpuStorage::Bool(std::sync::Arc::new(
                v.iter().map(|e| e.to_f32() != 0.0).collect(),
            )),
            _ => unreachable!("materialized dtype disagrees with view dtype"),
        },
        (DType::Bool, DType::F16) => match &src {
            CpuStorage::Bool(v) => CpuStorage::F16(std::sync::Arc::new(
                v.iter()
                    .map(|&e| half::f16::from_f32(if e { 1.0 } else { 0.0 }))
                    .collect(),
            )),
            _ => unreachable!("materialized dtype disagrees with view dtype"),
        },
        (DType::BF16, DType::Bool) => match &src {
            CpuStorage::BF16(v) => CpuStorage::Bool(std::sync::Arc::new(
                v.iter().map(|e| e.to_f32() != 0.0).collect(),
            )),
            _ => unreachable!("materialized dtype disagrees with view dtype"),
        },
        (DType::Bool, DType::BF16) => match &src {
            CpuStorage::Bool(v) => CpuStorage::BF16(std::sync::Arc::new(
                v.iter()
                    .map(|&e| half::bf16::from_f32(if e { 1.0 } else { 0.0 }))
                    .collect(),
            )),
            _ => unreachable!("materialized dtype disagrees with view dtype"),
        },

        // I64 <-> Bool.
        (DType::I64, DType::Bool) => match &src {
            CpuStorage::I64(v) => {
                CpuStorage::Bool(std::sync::Arc::new(v.iter().map(|&e| e != 0).collect()))
            }
            _ => unreachable!("materialized dtype disagrees with view dtype"),
        },
        (DType::Bool, DType::I64) => match &src {
            CpuStorage::Bool(v) => CpuStorage::I64(std::sync::Arc::new(
                v.iter().map(|&e| i64::from(e)).collect(),
            )),
            _ => unreachable!("materialized dtype disagrees with view dtype"),
        },

        // F64 cast scope remains deferred: loud, never a silent reinterpretation.
        _ => return Err(unsupported()),
    };

    Ok(Storage::Cpu(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::Shape;
    use std::sync::Arc;

    // ------------------------------------------------------------------
    // Helpers: build storage/layout pairs and read results back.
    // ------------------------------------------------------------------

    fn f32_storage(v: Vec<f32>) -> Storage {
        Storage::Cpu(CpuStorage::F32(Arc::new(v)))
    }
    fn i64_storage(v: Vec<i64>) -> Storage {
        Storage::Cpu(CpuStorage::I64(Arc::new(v)))
    }
    fn bool_storage(v: Vec<bool>) -> Storage {
        Storage::Cpu(CpuStorage::Bool(Arc::new(v)))
    }

    fn as_f32(s: &CpuStorage) -> Vec<f32> {
        match s {
            CpuStorage::F32(v) => v.as_ref().clone(),
            other => panic!("expected F32, got {}", other.dtype()),
        }
    }
    fn as_i64(s: &CpuStorage) -> Vec<i64> {
        match s {
            CpuStorage::I64(v) => v.as_ref().clone(),
            other => panic!("expected I64, got {}", other.dtype()),
        }
    }
    fn as_bool(s: &CpuStorage) -> Vec<bool> {
        match s {
            CpuStorage::Bool(v) => v.as_ref().clone(),
            other => panic!("expected Bool, got {}", other.dtype()),
        }
    }

    fn cpu(storage: &Storage) -> &CpuStorage {
        match storage {
            Storage::Cpu(s) => s,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            _ => panic!("expected CPU storage"),
        }
    }

    // ------------------------------------------------------------------
    // transfer_in / transfer_out round-trip (contiguous)
    // ------------------------------------------------------------------

    #[test]
    fn transfer_in_wraps_unchanged() {
        let host = CpuStorage::F32(Arc::new(vec![1.0, 2.0, 3.0]));
        let dev = transfer_in(host).unwrap();
        assert_eq!(as_f32(cpu(&dev)), vec![1.0, 2.0, 3.0]);
        assert_eq!(dev.device(), crate::device::Device::Cpu);
    }

    #[test]
    fn round_trip_contiguous() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let storage = f32_storage(data.clone());
        let layout = Layout::contiguous([2, 3]).unwrap();
        let view = View::new(&storage, &layout);
        let out = transfer_out(view).unwrap();
        assert_eq!(as_f32(&out), data);
    }

    #[test]
    fn round_trip_scalar() {
        let storage = f32_storage(vec![42.0]);
        let layout = Layout::contiguous(()).unwrap();
        let view = View::new(&storage, &layout);
        let out = transfer_out(view).unwrap();
        assert_eq!(as_f32(&out), vec![42.0]);
    }

    #[test]
    fn round_trip_empty() {
        let storage = f32_storage(vec![]);
        let layout = Layout::contiguous([0, 3]).unwrap();
        let view = View::new(&storage, &layout);
        let out = transfer_out(view).unwrap();
        assert!(as_f32(&out).is_empty());
    }

    // ------------------------------------------------------------------
    // transfer_out / copy_strided materialize strided/permuted/broadcast
    // views into contiguous row-major order.
    // ------------------------------------------------------------------

    #[test]
    fn materialize_transposed_view() {
        // Row-major [2,3] buffer, transposed to [3,2]: the contiguous
        // materialization must reorder into the transposed row-major walk.
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3],[4,5,6]]
        let storage = f32_storage(data);
        let layout = Layout::contiguous([2, 3]).unwrap().transpose(0, 1).unwrap();
        assert_eq!(layout.dims(), &[3, 2]);
        let view = View::new(&storage, &layout);
        // Row-major walk of the [3,2] transposed view: (0,0)=1 (0,1)=4
        // (1,0)=2 (1,1)=5 (2,0)=3 (2,1)=6.
        let out = transfer_out(view).unwrap();
        assert_eq!(as_f32(&out), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn materialize_permuted_view() {
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let storage = f32_storage(data.clone());
        let base = Layout::contiguous([2, 3, 4]).unwrap();
        let perm = base.permute(&[2, 0, 1]).unwrap(); // dims [4,2,3]
        let view = View::new(&storage, &perm);
        let out = copy_strided(view).unwrap();
        // Reference: enumerate the permuted logical coords and read the base
        // element at the un-permuted coordinate.
        let base_strides = base.strides().to_vec();
        let mut expected = Vec::new();
        for i in 0..4 {
            for j in 0..2 {
                for k in 0..3 {
                    // permuted coord (i,j,k) -> base coord (j,k,i)
                    let idx = j * base_strides[0] + k * base_strides[1] + i * base_strides[2];
                    expected.push(data[idx]);
                }
            }
        }
        assert_eq!(as_f32(cpu(&out)), expected);
    }

    #[test]
    fn materialize_narrowed_view() {
        // [4,5] contiguous, narrow axis 1 to [1,3): a strided, offset view.
        let data: Vec<f32> = (0..20).map(|x| x as f32).collect();
        let storage = f32_storage(data.clone());
        let layout = Layout::contiguous([4, 5]).unwrap().narrow(1, 1, 3).unwrap();
        assert_eq!(layout.dims(), &[4, 3]);
        let view = View::new(&storage, &layout);
        let out = transfer_out(view).unwrap();
        let mut expected = Vec::new();
        for row in 0..4 {
            for col in 1..4 {
                expected.push(data[row * 5 + col]);
            }
        }
        assert_eq!(as_f32(&out), expected);
    }

    #[test]
    fn materialize_broadcast_view() {
        // [1,3] broadcast to [2,4,3]: stride-0 axes must repeat elements.
        let data = vec![10.0f32, 20.0, 30.0];
        let storage = f32_storage(data.clone());
        let layout = Layout::contiguous([1, 3])
            .unwrap()
            .broadcast_to(&Shape::from([2, 4, 3]))
            .unwrap();
        let view = View::new(&storage, &layout);
        let out = copy_strided(view).unwrap();
        // Every (leading, expanded) coordinate maps to data[col].
        let mut expected = Vec::new();
        for _ in 0..2 {
            for _ in 0..4 {
                for &e in &data {
                    expected.push(e);
                }
            }
        }
        assert_eq!(as_f32(cpu(&out)), expected);
        assert_eq!(cpu(&out).len(), 24);
    }

    #[test]
    fn copy_strided_result_is_contiguous_reusable() {
        // The materialized copy walked with a fresh contiguous layout must
        // reproduce the same elements: i.e. it is genuinely row-major.
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let storage = f32_storage(data);
        let src_layout = Layout::contiguous([3, 4]).unwrap().transpose(0, 1).unwrap();
        let view = View::new(&storage, &src_layout);
        let copied = copy_strided(view).unwrap();

        // Re-view the copy contiguously in the transposed shape and compare
        // element-for-element against a second transfer_out of the source.
        let expected = as_f32(&transfer_out(View::new(&storage, &src_layout)).unwrap());
        let new_layout = Layout::contiguous([4, 3]).unwrap();
        let got = as_f32(&transfer_out(View::new(&copied, &new_layout)).unwrap());
        assert_eq!(got, expected);
    }

    // ------------------------------------------------------------------
    // Fast paths: the three copy tiers must all agree with the naive
    // per-element gather they replaced, and the whole-buffer case must not
    // copy at all.
    // ------------------------------------------------------------------

    /// The original implementation: enumerate every logical position and read
    /// `offset + Σ coord[a] * stride[a]`. The reference every tier of
    /// [`copy_view`] is checked against.
    fn naive_gather(src: &[f32], layout: &Layout) -> Vec<f32> {
        let dims = layout.dims();
        let strides = layout.strides();
        let total = layout.num_elements();
        let mut out = Vec::with_capacity(total);
        let mut coords = vec![0usize; dims.len()];
        for _ in 0..total {
            let mut idx = layout.offset();
            for (c, s) in coords.iter().zip(strides.iter()) {
                idx += c * s;
            }
            out.push(src[idx]);
            let mut axis = dims.len();
            while axis > 0 {
                axis -= 1;
                coords[axis] += 1;
                if coords[axis] < dims[axis] {
                    break;
                }
                coords[axis] = 0;
            }
        }
        out
    }

    #[test]
    fn copy_view_matches_the_naive_gather_on_every_layout_shape() {
        let data: Vec<f32> = (0..120).map(|x| x as f32).collect();
        let base3 = Layout::contiguous([2, 3, 4]).unwrap();
        let cases: Vec<(&str, Layout)> = vec![
            ("contiguous", base3.clone()),
            ("scalar", Layout::contiguous(()).unwrap()),
            ("empty", Layout::contiguous([0, 5]).unwrap()),
            // Dense with a non-zero offset: the `memcpy` tier.
            ("outer narrow", base3.narrow(0, 1, 1).unwrap()),
            // Row runs: the trailing axes stay contiguous.
            ("inner narrow", base3.narrow(1, 1, 2).unwrap()),
            ("transposed outer", base3.transpose(0, 1).unwrap()),
            (
                "broadcast leading",
                Layout::contiguous([1, 4])
                    .unwrap()
                    .broadcast_to(&Shape::from([3, 2, 4]))
                    .unwrap(),
            ),
            // Run length 1: the element-at-a-time tier.
            ("transposed inner", base3.transpose(1, 2).unwrap()),
            (
                "broadcast innermost",
                Layout::contiguous([3, 1])
                    .unwrap()
                    .broadcast_to(&Shape::from([3, 4]))
                    .unwrap(),
            ),
            ("permuted", base3.permute(&[2, 0, 1]).unwrap()),
            // A size-1 axis whose stride is not the canonical one still
            // qualifies as dense: its only coordinate is 0.
            (
                "size-1 axis, odd stride",
                Layout::from_parts(Shape::from([1, 6]), vec![99usize, 1].into_boxed_slice(), 3)
                    .unwrap(),
            ),
        ];
        for (name, layout) in cases {
            let got = copy_view(&data, &layout);
            assert_eq!(got, naive_gather(&data, &layout), "{name}");
            assert_eq!(got.len(), layout.num_elements(), "{name}: length");
        }
    }

    #[test]
    fn transfer_out_shares_a_whole_contiguous_buffer_instead_of_copying() {
        let buf = Arc::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let storage = Storage::Cpu(CpuStorage::F32(Arc::clone(&buf)));
        let layout = Layout::contiguous([2, 3]).unwrap();
        let out = transfer_out(View::new(&storage, &layout)).unwrap();
        match &out {
            CpuStorage::F32(v) => assert!(
                Arc::ptr_eq(v, &buf),
                "a whole-buffer contiguous view must be shared, not copied"
            ),
            other => panic!("expected F32, got {}", other.dtype()),
        }

        // A partial view of the same buffer must *not* share it: it has to be
        // the narrowed window's own row-major buffer.
        let part = layout.narrow(0, 1, 1).unwrap();
        let out = transfer_out(View::new(&storage, &part)).unwrap();
        match &out {
            CpuStorage::F32(v) => {
                assert!(!Arc::ptr_eq(v, &buf));
                assert_eq!(v.as_ref(), &vec![4.0, 5.0, 6.0]);
            }
            other => panic!("expected F32, got {}", other.dtype()),
        }
    }

    #[test]
    fn copy_strided_always_allocates_even_for_a_whole_contiguous_buffer() {
        // `BackendOps::copy_strided` documents a *fresh* buffer, so unlike
        // `transfer_out` it must not take the sharing shortcut: `contiguous()`
        // and the copying branch of `reshape` are specified to hand back
        // storage that shares nothing with their input.
        let buf = Arc::new(vec![1.0f32, 2.0, 3.0, 4.0]);
        let storage = Storage::Cpu(CpuStorage::F32(Arc::clone(&buf)));
        let layout = Layout::contiguous([2, 2]).unwrap();
        let out = copy_strided(View::new(&storage, &layout)).unwrap();
        match &out {
            Storage::Cpu(CpuStorage::F32(v)) => {
                assert!(!Arc::ptr_eq(v, &buf), "copy_strided must allocate");
                assert_eq!(v.as_ref(), buf.as_ref(), "…with the same elements");
            }
            other => panic!("expected a CPU F32 storage, got {:?}", other.dtype()),
        }
    }

    #[test]
    fn dense_offset_accepts_offsets_and_rejects_reordering() {
        let base = Layout::contiguous([4, 5]).unwrap();
        assert_eq!(dense_offset(&base), Some(0));
        // Narrowing the outermost axis keeps a single run, at an offset.
        assert_eq!(dense_offset(&base.narrow(0, 2, 2).unwrap()), Some(10));
        // Narrowing an inner axis leaves gaps.
        assert_eq!(dense_offset(&base.narrow(1, 1, 2).unwrap()), None);
        // Transposing reorders; broadcasting repeats.
        assert_eq!(dense_offset(&base.transpose(0, 1).unwrap()), None);
        assert_eq!(
            dense_offset(
                &Layout::contiguous([1, 5])
                    .unwrap()
                    .broadcast_to(&Shape::from([4, 5]))
                    .unwrap()
            ),
            None
        );
    }

    // ------------------------------------------------------------------
    // full
    // ------------------------------------------------------------------

    #[test]
    fn full_fills_each_dtype() {
        assert_eq!(
            as_f32(cpu(&full(3, DType::F32, 2.5).unwrap())),
            vec![2.5, 2.5, 2.5]
        );
        assert_eq!(as_i64(cpu(&full(2, DType::I64, 7.0).unwrap())), vec![7, 7]);
        // Bool: non-zero is true, zero is false.
        assert_eq!(
            as_bool(cpu(&full(2, DType::Bool, 1.0).unwrap())),
            vec![true, true]
        );
        assert_eq!(
            as_bool(cpu(&full(2, DType::Bool, 0.0).unwrap())),
            vec![false, false]
        );
        // Zero-length fill.
        assert!(as_f32(cpu(&full(0, DType::F32, 1.0).unwrap())).is_empty());
    }

    #[test]
    fn full_narrows_float_to_int_truncating() {
        // f64 -> i64 truncates toward zero.
        assert_eq!(as_i64(cpu(&full(1, DType::I64, 2.9).unwrap())), vec![2]);
        assert_eq!(as_i64(cpu(&full(1, DType::I64, -2.9).unwrap())), vec![-2]);
    }

    // ------------------------------------------------------------------
    // cast: supported lanes
    // ------------------------------------------------------------------

    #[test]
    fn cast_f32_to_i64_truncates_toward_zero() {
        let storage = f32_storage(vec![1.9, -1.9, 0.0, 3.2]);
        let layout = Layout::contiguous([4]).unwrap();
        let out = cast(View::new(&storage, &layout), DType::I64).unwrap();
        assert_eq!(as_i64(cpu(&out)), vec![1, -1, 0, 3]);
    }

    #[test]
    fn cast_i64_to_f32() {
        let storage = i64_storage(vec![0, -5, 42]);
        let layout = Layout::contiguous([3]).unwrap();
        let out = cast(View::new(&storage, &layout), DType::F32).unwrap();
        assert_eq!(as_f32(cpu(&out)), vec![0.0, -5.0, 42.0]);
    }

    #[test]
    fn cast_f32_bool_round_trip() {
        let storage = f32_storage(vec![0.0, 1.0, -3.0, 0.0]);
        let layout = Layout::contiguous([4]).unwrap();
        let to_bool = cast(View::new(&storage, &layout), DType::Bool).unwrap();
        assert_eq!(as_bool(cpu(&to_bool)), vec![false, true, true, false]);
        // Bool -> F32 yields 0.0 / 1.0.
        let back = cast(View::new(&to_bool, &layout), DType::F32).unwrap();
        assert_eq!(as_f32(cpu(&back)), vec![0.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn cast_i64_bool_round_trip() {
        let storage = i64_storage(vec![0, 5, -1, 0]);
        let layout = Layout::contiguous([4]).unwrap();
        let to_bool = cast(View::new(&storage, &layout), DType::Bool).unwrap();
        assert_eq!(as_bool(cpu(&to_bool)), vec![false, true, true, false]);
        let back = cast(View::new(&to_bool, &layout), DType::I64).unwrap();
        assert_eq!(as_i64(cpu(&back)), vec![0, 1, 1, 0]);
    }

    #[test]
    fn cast_identity_is_a_copy() {
        let storage = f32_storage(vec![1.0, 2.0, 3.0]);
        let layout = Layout::contiguous([3]).unwrap();
        let out = cast(View::new(&storage, &layout), DType::F32).unwrap();
        assert_eq!(as_f32(cpu(&out)), vec![1.0, 2.0, 3.0]);

        let buffer = Arc::new(vec![half::bf16::from_f32(1.0); 3]);
        let storage = Storage::Cpu(CpuStorage::BF16(Arc::clone(&buffer)));
        let out = cast(View::new(&storage, &layout), DType::BF16).unwrap();
        let Storage::Cpu(CpuStorage::BF16(out)) = out else {
            panic!("expected bf16")
        };
        assert!(!Arc::ptr_eq(&buffer, &out));
        assert_eq!(buffer.as_ref(), out.as_ref());
    }

    #[test]
    fn cast_materializes_strided_source() {
        // Casting a transposed view must materialize in row-major order.
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [[1,2,3],[4,5,6]]
        let storage = f32_storage(data);
        let layout = Layout::contiguous([2, 3]).unwrap().transpose(0, 1).unwrap();
        let out = cast(View::new(&storage, &layout), DType::I64).unwrap();
        // Transposed row-major walk: 1,4,2,5,3,6 truncated to i64.
        assert_eq!(as_i64(cpu(&out)), vec![1, 4, 2, 5, 3, 6]);

        let reduced = cast(View::new(&storage, &layout), DType::BF16).unwrap();
        let dense = Layout::contiguous([3, 2]).unwrap();
        let back = cast(View::new(&reduced, &dense), DType::F32).unwrap();
        assert_eq!(as_f32(cpu(&back)), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    // ------------------------------------------------------------------
    // cast: reduced-precision matrix and unchanged F64 scope
    // ------------------------------------------------------------------

    // `Storage` (the Ok payload) is intentionally not `Debug`, so these
    // error assertions pattern-match the `Result` directly rather than
    // calling `unwrap_err` (which would require `T: Debug`).

    #[test]
    fn reduced_cast_matrix_covers_float_integer_and_bool_lanes() {
        let layout = Layout::contiguous([4]).unwrap();
        for reduced in [DType::F16, DType::BF16] {
            let source = cast(
                View::new(&f32_storage(vec![0.0, 1.75, -2.25, 3.5]), &layout),
                reduced,
            )
            .unwrap();
            let f32s = cast(View::new(&source, &layout), DType::F32).unwrap();
            assert_eq!(as_f32(cpu(&f32s)), vec![0.0, 1.75, -2.25, 3.5]);

            let ints = cast(View::new(&source, &layout), DType::I64).unwrap();
            assert_eq!(as_i64(cpu(&ints)), vec![0, 1, -2, 3]);
            let from_ints = cast(View::new(&ints, &layout), reduced).unwrap();
            let ints_back = cast(View::new(&from_ints, &layout), DType::F32).unwrap();
            assert_eq!(as_f32(cpu(&ints_back)), vec![0.0, 1.0, -2.0, 3.0]);

            let bools = cast(View::new(&source, &layout), DType::Bool).unwrap();
            assert_eq!(as_bool(cpu(&bools)), vec![false, true, true, true]);
            let from_bools = cast(View::new(&bools, &layout), reduced).unwrap();
            let bools_back = cast(View::new(&from_bools, &layout), DType::F32).unwrap();
            assert_eq!(as_f32(cpu(&bools_back)), vec![0.0, 1.0, 1.0, 1.0]);
        }

        let f16 = cast(
            View::new(&f32_storage(vec![0.5, -1.5, 2.0, 4.0]), &layout),
            DType::F16,
        )
        .unwrap();
        let bf16 = cast(View::new(&f16, &layout), DType::BF16).unwrap();
        let round_trip = cast(View::new(&bf16, &layout), DType::F16).unwrap();
        let round_trip_f32 = cast(View::new(&round_trip, &layout), DType::F32).unwrap();
        assert_eq!(as_f32(cpu(&round_trip_f32)), vec![0.5, -1.5, 2.0, 4.0]);
    }

    #[test]
    fn cast_f64_lanes_are_unsupported() {
        // F64 is a real dtype but its cast lanes are not in the m1 set.
        let storage = f32_storage(vec![1.0]);
        let layout = Layout::contiguous([1]).unwrap();
        assert!(matches!(
            cast(View::new(&storage, &layout), DType::F64),
            Err(Error::Unsupported { op: "to_dtype", .. })
        ));
    }
}
