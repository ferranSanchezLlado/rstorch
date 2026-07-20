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
        #[cfg(feature = "metal")]
        Storage::Metal(_) => {
            unreachable!("CPU backend received non-CPU storage; dispatcher invariant violated")
        }
    }
}

/// Row-major storage indices of every logical element of `layout`, in
/// row-major (C-order) logical order.
///
/// This is the one place strided/permuted/broadcast views are flattened
/// into contiguous order: it walks the logical coordinate space like an
/// odometer (rightmost axis fastest) and, for each coordinate, computes the
/// storage index `offset + Σ coord[a] * stride[a]`. Broadcast axes (stride
/// 0) repeat the same element, exactly as the layout contract specifies.
///
/// The returned vector has [`Layout::num_elements`] entries (empty for an
/// empty view).
fn row_major_indices(layout: &Layout) -> Vec<usize> {
    let dims = layout.dims();
    let strides = layout.strides();
    let total = layout.num_elements();
    if total == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(total);
    let mut coords = vec![0usize; dims.len()];
    loop {
        let mut idx = layout.offset();
        for (c, s) in coords.iter().zip(strides.iter()) {
            idx += c * s;
        }
        out.push(idx);
        // Odometer increment, rightmost axis fastest. A rank-0 (scalar) view
        // has one element and no axes: the first `push` above emits it and we
        // fall straight through to the return.
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

/// Gather a source buffer into a fresh contiguous vector following the
/// view's row-major walk. `src` is the full backing buffer; `indices` are
/// the storage positions produced by [`row_major_indices`].
fn gather<T: Copy>(src: &[T], indices: &[usize]) -> Vec<T> {
    indices.iter().map(|&i| src[i]).collect()
}

/// Materialize a view into a fresh contiguous [`CpuStorage`] in row-major
/// order, preserving dtype. This is the shared core of
/// [`transfer_out`] and [`copy_strided`].
fn materialize(x: View<'_>) -> CpuStorage {
    let storage = cpu_storage(&x);
    let indices = row_major_indices(x.layout());
    match storage {
        CpuStorage::F16(v) => CpuStorage::F16(std::sync::Arc::new(gather(v, &indices))),
        CpuStorage::BF16(v) => CpuStorage::BF16(std::sync::Arc::new(gather(v, &indices))),
        CpuStorage::F32(v) => CpuStorage::F32(std::sync::Arc::new(gather(v, &indices))),
        CpuStorage::F64(v) => CpuStorage::F64(std::sync::Arc::new(gather(v, &indices))),
        CpuStorage::I64(v) => CpuStorage::I64(std::sync::Arc::new(gather(v, &indices))),
        CpuStorage::Bool(v) => CpuStorage::Bool(std::sync::Arc::new(gather(v, &indices))),
    }
}

/// See [`BackendOps::transfer_in`](crate::backend::BackendOps::transfer_in).
/// For CPU this wraps the buffer unchanged.
pub(crate) fn transfer_in(host: CpuStorage) -> Result<Storage> {
    Ok(Storage::Cpu(host))
}

/// See [`BackendOps::transfer_out`](crate::backend::BackendOps::transfer_out).
pub(crate) fn transfer_out(x: View<'_>) -> Result<CpuStorage> {
    Ok(materialize(x))
}

/// See [`BackendOps::copy_strided`](crate::backend::BackendOps::copy_strided).
pub(crate) fn copy_strided(x: View<'_>) -> Result<Storage> {
    Ok(Storage::Cpu(materialize(x)))
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

/// See [`BackendOps::cast`](crate::backend::BackendOps::cast). Lanes now:
/// F32↔I64, F32↔Bool, I64↔Bool (T60 adds F16/BF16).
///
/// The source view is first materialized in row-major order, then each
/// element is converted with familiar Rust/PyTorch semantics:
/// float→int is a saturating truncation toward zero (Rust `as`), numeric→bool
/// is `x != 0`, and bool→numeric is `0`/`1`. An identity cast (`to` equals
/// the source dtype) returns a contiguous copy. Any lane outside the set
/// above — notably every `F16`/`BF16` lane, deferred to T60 — is
/// [`Error::Unsupported`](crate::Error::Unsupported), never a silent
/// reinterpretation.
pub(crate) fn cast(x: View<'_>, to: DType) -> Result<Storage> {
    let from = x.dtype();
    let unsupported = || Error::Unsupported {
        op: "to_dtype",
        device: x.device(),
        dtype: from,
    };

    // Materialize the (possibly strided) source contiguously first so the
    // per-lane conversion is a flat map.
    let src = materialize(x);

    let out = match (from, to) {
        // Identity: a contiguous copy, no conversion.
        (a, b) if a == b => src,

        // F32 <-> I64.
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

        // Every remaining lane (all F16/BF16 lanes, plus F64 lanes) is
        // deferred: loud, never a silent reinterpretation.
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
            #[cfg(feature = "metal")]
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
    }

    // ------------------------------------------------------------------
    // cast: unsupported lanes are loud (never a silent reinterpretation)
    // ------------------------------------------------------------------

    // `Storage` (the Ok payload) is intentionally not `Debug`, so these
    // error assertions pattern-match the `Result` directly rather than
    // calling `unwrap_err` (which would require `T: Debug`).

    #[test]
    fn cast_f16_lanes_are_unsupported() {
        let storage = f32_storage(vec![1.0, 2.0]);
        let layout = Layout::contiguous([2]).unwrap();
        assert!(matches!(
            cast(View::new(&storage, &layout), DType::F16),
            Err(Error::Unsupported {
                op: "to_dtype",
                dtype: DType::F32,
                ..
            })
        ));
    }

    #[test]
    fn cast_from_f16_is_unsupported() {
        let storage = Storage::Cpu(CpuStorage::F16(Arc::new(vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
        ])));
        let layout = Layout::contiguous([2]).unwrap();
        assert!(matches!(
            cast(View::new(&storage, &layout), DType::F32),
            Err(Error::Unsupported {
                op: "to_dtype",
                dtype: DType::F16,
                ..
            })
        ));
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
