//! Matmul CPU kernel.
//!
//! Signature frozen by T01; **T11** fills the body — ported v2 loops
//! adapted to the `Element::Acc` contract, batched over leading dims.
//! Semantics on [`BackendOps`](crate::backend::BackendOps).
//!
//! Both operands are rank ≥ 2. The trailing two axes are the matrix axes
//! (`[..., m, k]` × `[..., k, n]` → `[..., m, n]`); the leading axes are
//! batch dims that broadcast against each other under NumPy/PyTorch rules
//! (right-aligned; a size-1 batch axis repeats). Inner products accumulate in
//! the wide [`Acc`](crate::dtype::Element::Acc) type and cast back to the
//! element type exactly once per output element — the implemented fix for the
//! v2 native paths, which accumulated in dtype. The walk is stride-aware, so
//! transposed / narrowed / broadcast operand views are handled directly with
//! no pre-materialization.

use crate::backend::View;
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::shape::Shape;
use crate::storage::{CpuStorage, Storage};

/// Wide-accumulator arithmetic for matmul inner products. Implemented for the
/// numeric accumulator types (`f32` for `f16`/`bf16`/`f32`, `f64`, `i64`);
/// `Bool` has `Acc = bool` and deliberately does not implement it, so a bool
/// matmul is rejected before this generic code is instantiated.
trait MatAcc: Copy {
    /// Additive identity (inner-product seed).
    const ZERO: Self;
    /// Fused multiply-add step `self + a*b` (kept as separate ops so integer
    /// accumulation wraps deterministically rather than aborting on debug
    /// overflow — matmul over huge i64 magnitudes is a defined wrap).
    fn mul_add(self, a: Self, b: Self) -> Self;
}

impl MatAcc for f32 {
    const ZERO: Self = 0.0;
    fn mul_add(self, a: Self, b: Self) -> Self {
        self + a * b
    }
}

impl MatAcc for f64 {
    const ZERO: Self = 0.0;
    fn mul_add(self, a: Self, b: Self) -> Self {
        self + a * b
    }
}

impl MatAcc for i64 {
    const ZERO: Self = 0;
    fn mul_add(self, a: Self, b: Self) -> Self {
        self.wrapping_add(a.wrapping_mul(b))
    }
}

/// The resolved matmul geometry: batch shape (already broadcast), matrix
/// dims, and per-operand leading-batch strides padded to the batch rank.
struct Plan {
    /// Broadcast batch dims (may be empty for the pure 2-D case).
    batch: Vec<usize>,
    m: usize,
    k: usize,
    n: usize,
    /// lhs strides for the batch axes (0 where broadcast), length == batch rank.
    lhs_batch_strides: Vec<usize>,
    /// rhs strides for the batch axes (0 where broadcast), length == batch rank.
    rhs_batch_strides: Vec<usize>,
    lhs_offset: usize,
    rhs_offset: usize,
    // Trailing-axis strides.
    lhs_m_stride: usize,
    lhs_k_stride: usize,
    rhs_k_stride: usize,
    rhs_n_stride: usize,
}

/// Broadcast two batch-shape prefixes right-aligned, returning the output
/// batch dims and, for each operand, the stride to advance along each output
/// batch axis (0 where that axis is broadcast or padded).
fn plan(lhs: &Layout, rhs: &Layout) -> Result<Plan> {
    let lr = lhs.rank();
    let rr = rhs.rank();
    if lr < 2 || rr < 2 {
        return Err(Error::InvalidArg {
            op: "matmul",
            msg: format!("operands must be rank >= 2, got ranks {lr} and {rr}"),
        });
    }
    let ld = lhs.dims();
    let rd = rhs.dims();
    let ls = lhs.strides();
    let rs = rhs.strides();

    let m = ld[lr - 2];
    let k = ld[lr - 1];
    let rk = rd[rr - 2];
    let n = rd[rr - 1];
    if k != rk {
        // The op layer normally raises this; the kernel double-checks so a
        // mis-wired call is a loud error, not silent garbage.
        return Err(Error::ShapeMismatch {
            op: "matmul",
            lhs: lhs.shape().clone(),
            rhs: rhs.shape().clone(),
        });
    }

    let lb = &ld[..lr - 2];
    let rb = &rd[..rr - 2];
    let batch_rank = lb.len().max(rb.len());
    let mut batch = vec![0usize; batch_rank];
    let mut lhs_batch_strides = vec![0usize; batch_rank];
    let mut rhs_batch_strides = vec![0usize; batch_rank];
    for i in 0..batch_rank {
        // Right-aligned index into each operand's batch dims.
        let li = (i + lb.len()).checked_sub(batch_rank);
        let ri = (i + rb.len()).checked_sub(batch_rank);
        let ldim = li.map_or(1, |j| lb[j]);
        let rdim = ri.map_or(1, |j| rb[j]);
        let out = if ldim == rdim {
            ldim
        } else if ldim == 1 {
            rdim
        } else if rdim == 1 {
            ldim
        } else {
            return Err(Error::ShapeMismatch {
                op: "matmul",
                lhs: lhs.shape().clone(),
                rhs: rhs.shape().clone(),
            });
        };
        batch[i] = out;
        // A broadcast (size-1 or padded) batch axis contributes stride 0.
        lhs_batch_strides[i] = match li {
            Some(j) if lb[j] != 1 => ls[j],
            _ => 0,
        };
        rhs_batch_strides[i] = match ri {
            Some(j) if rb[j] != 1 => rs[j],
            _ => 0,
        };
    }

    Ok(Plan {
        batch,
        m,
        k,
        n,
        lhs_batch_strides,
        rhs_batch_strides,
        lhs_offset: lhs.offset(),
        rhs_offset: rhs.offset(),
        lhs_m_stride: ls[lr - 2],
        lhs_k_stride: ls[lr - 1],
        rhs_k_stride: rs[rr - 2],
        rhs_n_stride: rs[rr - 1],
    })
}

/// The output shape for a plan: broadcast batch dims followed by `[m, n]`.
fn output_shape(plan: &Plan) -> Shape {
    let mut dims = plan.batch.clone();
    dims.push(plan.m);
    dims.push(plan.n);
    Shape::from(dims)
}

/// Generic batched matmul accumulating each inner product in `E::Acc` and
/// casting once at output. `lhs`/`rhs` are the whole backing buffers.
fn matmul_generic<E>(lhs: &[E], rhs: &[E], plan: &Plan) -> Vec<E>
where
    E: Element,
    E::Acc: MatAcc,
{
    let batch_count: usize = plan.batch.iter().product::<usize>().max(1);
    let (m, k, n) = (plan.m, plan.k, plan.n);
    let mut out = Vec::with_capacity(batch_count * m * n);
    // Decode a linear batch index into per-axis coords (row-major over the
    // batch dims) and the corresponding operand base offsets.
    for b in 0..batch_count {
        let mut lhs_base = plan.lhs_offset;
        let mut rhs_base = plan.rhs_offset;
        let mut rem = b;
        for ax in (0..plan.batch.len()).rev() {
            let size = plan.batch[ax];
            let coord = rem % size;
            rem /= size;
            lhs_base += coord * plan.lhs_batch_strides[ax];
            rhs_base += coord * plan.rhs_batch_strides[ax];
        }
        for i in 0..m {
            let lhs_row = lhs_base + i * plan.lhs_m_stride;
            for j in 0..n {
                let rhs_col = rhs_base + j * plan.rhs_n_stride;
                let mut acc = <E::Acc as MatAcc>::ZERO;
                for p in 0..k {
                    let a = lhs[lhs_row + p * plan.lhs_k_stride].to_acc();
                    let bx = rhs[rhs_col + p * plan.rhs_k_stride].to_acc();
                    acc = acc.mul_add(a, bx);
                }
                out.push(E::from_acc(acc));
            }
        }
    }
    out
}

/// See [`BackendOps::matmul`](crate::backend::BackendOps::matmul).
pub(crate) fn matmul(lhs: View<'_>, rhs: View<'_>) -> Result<Storage> {
    if lhs.dtype() != rhs.dtype() {
        return Err(Error::DTypeMismatch {
            op: "matmul",
            expected: lhs.dtype(),
            got: rhs.dtype(),
        });
    }
    let plan = plan(lhs.layout(), rhs.layout())?;
    let lhs_cpu = cpu_storage(lhs, "matmul")?;
    let rhs_cpu = cpu_storage(rhs, "matmul")?;
    let storage = match (lhs_cpu, rhs_cpu) {
        (CpuStorage::F16(a), CpuStorage::F16(b)) => {
            CpuStorage::F16(std::sync::Arc::new(matmul_generic(a, b, &plan)))
        }
        (CpuStorage::BF16(a), CpuStorage::BF16(b)) => {
            CpuStorage::BF16(std::sync::Arc::new(matmul_generic(a, b, &plan)))
        }
        (CpuStorage::F32(a), CpuStorage::F32(b)) => {
            CpuStorage::F32(std::sync::Arc::new(matmul_generic(a, b, &plan)))
        }
        (CpuStorage::F64(a), CpuStorage::F64(b)) => {
            CpuStorage::F64(std::sync::Arc::new(matmul_generic(a, b, &plan)))
        }
        (CpuStorage::I64(a), CpuStorage::I64(b)) => {
            CpuStorage::I64(std::sync::Arc::new(matmul_generic(a, b, &plan)))
        }
        (CpuStorage::Bool(_), _) => {
            return Err(Error::Unsupported {
                op: "matmul",
                device: lhs.device(),
                dtype: DType::Bool,
            });
        }
        // dtype equality was checked above, so the remaining cross-variant
        // pairs are unreachable; report loudly rather than silently.
        _ => {
            return Err(Error::DTypeMismatch {
                op: "matmul",
                expected: lhs.dtype(),
                got: rhs.dtype(),
            });
        }
    };
    Ok(Storage::Cpu(storage))
}

/// Borrow the [`CpuStorage`] behind a CPU view, or report the op as
/// unsupported on a non-CPU device.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::View;
    use crate::layout::Layout;
    use std::sync::Arc;

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

    /// Expose the resolved output shape for shape-contract tests.
    fn matmul_shape(lhs: &Layout, rhs: &Layout) -> Shape {
        output_shape(&plan(lhs, rhs).unwrap())
    }

    // ----- golden 2-D ---------------------------------------------------

    #[test]
    fn matmul_2x3_by_3x2() {
        // A = [[1,2,3],[4,5,6]]  B = [[7,8],[9,10],[11,12]]
        // AB = [[58,64],[139,154]]
        let a = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = f32_storage(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let la = Layout::contiguous([2, 3]).unwrap();
        let lb = Layout::contiguous([3, 2]).unwrap();
        let r = matmul(View::new(&a, &la), View::new(&b, &lb)).unwrap();
        assert_eq!(as_f32(&r), vec![58.0, 64.0, 139.0, 154.0]);
        assert_eq!(matmul_shape(&la, &lb).dims(), &[2, 2]);
    }

    #[test]
    fn matmul_identity() {
        let a = f32_storage(vec![1.0, 2.0, 3.0, 4.0]);
        let id = f32_storage(vec![1.0, 0.0, 0.0, 1.0]);
        let l = Layout::contiguous([2, 2]).unwrap();
        let r = matmul(View::new(&a, &l), View::new(&id, &l)).unwrap();
        assert_eq!(as_f32(&r), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn matmul_vector_shaped_1xk_kx1() {
        // (1x3) x (3x1) -> (1x1) dot product.
        let a = f32_storage(vec![1.0, 2.0, 3.0]);
        let b = f32_storage(vec![4.0, 5.0, 6.0]);
        let la = Layout::contiguous([1, 3]).unwrap();
        let lb = Layout::contiguous([3, 1]).unwrap();
        let r = matmul(View::new(&a, &la), View::new(&b, &lb)).unwrap();
        assert_eq!(as_f32(&r), vec![32.0]); // 4+10+18
    }

    // ----- transposed / strided operands --------------------------------

    #[test]
    fn matmul_with_transposed_rhs() {
        // A (2x3) times B^T where B is (2x3): result (2x2).
        // The classic linear-layer pattern x @ w.T.
        let a = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); // 2x3
        let w = f32_storage(vec![1.0, 0.0, -1.0, 2.0, 1.0, 0.0]); // 2x3
        let la = Layout::contiguous([2, 3]).unwrap();
        let lw = Layout::contiguous([2, 3]).unwrap();
        let lw_t = lw.transpose(0, 1).unwrap(); // 3x2 view, strided
        let r = matmul(View::new(&a, &la), View::new(&w, &lw_t)).unwrap();
        // row0 . w0 = 1*1+2*0+3*-1 = -2 ; row0 . w1 = 1*2+2*1+3*0 = 4
        // row1 . w0 = 4-6 = -2 ; row1 . w1 = 8+5 = 13
        assert_eq!(as_f32(&r), vec![-2.0, 4.0, -2.0, 13.0]);
    }

    #[test]
    fn matmul_with_transposed_lhs() {
        // Backing A stored as 3x2; use its transpose (2x3) as lhs.
        let a = f32_storage(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]); // 3x2
        let la = Layout::contiguous([3, 2]).unwrap();
        let la_t = la.transpose(0, 1).unwrap(); // logical 2x3: [[1,2,3],[4,5,6]]
        let b = f32_storage(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let lb = Layout::contiguous([3, 2]).unwrap();
        let r = matmul(View::new(&a, &la_t), View::new(&b, &lb)).unwrap();
        assert_eq!(as_f32(&r), vec![58.0, 64.0, 139.0, 154.0]);
    }

    // ----- batched ------------------------------------------------------

    #[test]
    fn matmul_batched_3d() {
        // batch 2 of (2x2) x (2x2).
        let a = f32_storage(vec![
            1.0, 2.0, 3.0, 4.0, // batch 0
            5.0, 6.0, 7.0, 8.0, // batch 1
        ]);
        let b = f32_storage(vec![
            1.0, 0.0, 0.0, 1.0, // identity
            2.0, 0.0, 0.0, 2.0, // 2*identity
        ]);
        let la = Layout::contiguous([2, 2, 2]).unwrap();
        let lb = Layout::contiguous([2, 2, 2]).unwrap();
        let r = matmul(View::new(&a, &la), View::new(&b, &lb)).unwrap();
        // batch0 * I = batch0; batch1 * 2I = 2*batch1
        assert_eq!(as_f32(&r), vec![1.0, 2.0, 3.0, 4.0, 10.0, 12.0, 14.0, 16.0]);
        assert_eq!(matmul_shape(&la, &lb).dims(), &[2, 2, 2]);
    }

    #[test]
    fn matmul_broadcast_batch_lhs_single() {
        // lhs (1,2,2) broadcast against rhs (3,2,2): output (3,2,2).
        let a = f32_storage(vec![1.0, 0.0, 0.0, 1.0]); // one identity
        let b = f32_storage(vec![
            1.0, 2.0, 3.0, 4.0, //
            5.0, 6.0, 7.0, 8.0, //
            9.0, 10.0, 11.0, 12.0,
        ]);
        let la = Layout::contiguous([1, 2, 2]).unwrap();
        let lb = Layout::contiguous([3, 2, 2]).unwrap();
        let r = matmul(View::new(&a, &la), View::new(&b, &lb)).unwrap();
        // identity * each batch = that batch unchanged.
        assert_eq!(as_f32(&r), as_f32(&b));
        assert_eq!(matmul_shape(&la, &lb).dims(), &[3, 2, 2]);
    }

    #[test]
    fn matmul_broadcast_batch_rank_mismatch() {
        // lhs rank-2 (2x3) broadcasts against rhs (4,3,2): output (4,2,2).
        let a = f32_storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); // 2x3
        let b_data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let b = f32_storage(b_data);
        let la = Layout::contiguous([2, 3]).unwrap();
        let lb = Layout::contiguous([4, 3, 2]).unwrap();
        let r = matmul(View::new(&a, &la), View::new(&b, &lb)).unwrap();
        assert_eq!(matmul_shape(&la, &lb).dims(), &[4, 2, 2]);
        // Cross-check batch 0 by hand: a @ b[0], b[0]=[[0,1],[2,3],[4,5]].
        // row0: 1*0+2*2+3*4=16 ; 1*1+2*3+3*5=22
        let out = as_f32(&r);
        assert_eq!(&out[0..2], &[16.0, 22.0]);
    }

    // ----- Acc contract -------------------------------------------------

    #[test]
    fn f16_matmul_accumulates_in_f32() {
        // Inner dim large enough that a dtype-native f16 accumulation would
        // lose precision, but f32 accumulation is exact. (1 x k) . (k x 1)
        // of all-ones = k. k=4096 is exactly representable in f16 output.
        let k = 4096usize;
        let a = vec![half::f16::from_f32(1.0); k];
        let b = vec![half::f16::from_f32(1.0); k];
        let sa = Storage::Cpu(CpuStorage::F16(Arc::new(a)));
        let sb = Storage::Cpu(CpuStorage::F16(Arc::new(b)));
        let la = Layout::contiguous([1, k]).unwrap();
        let lb = Layout::contiguous([k, 1]).unwrap();
        let r = matmul(View::new(&sa, &la), View::new(&sb, &lb)).unwrap();
        let out = match &r {
            Storage::Cpu(CpuStorage::F16(v)) => v.as_ref().clone(),
            _ => panic!("expected f16"),
        };
        assert_eq!(out[0].to_f32(), 4096.0);
    }

    #[test]
    fn i64_matmul_accumulates_in_i64() {
        // Values whose products/sum exceed i32 range.
        let a = Storage::Cpu(CpuStorage::I64(Arc::new(vec![100_000, 100_000])));
        let b = Storage::Cpu(CpuStorage::I64(Arc::new(vec![100_000, 100_000])));
        let la = Layout::contiguous([1, 2]).unwrap();
        let lb = Layout::contiguous([2, 1]).unwrap();
        let r = matmul(View::new(&a, &la), View::new(&b, &lb)).unwrap();
        // 100000*100000*2 = 20,000,000,000
        assert_eq!(as_i64(&r), vec![20_000_000_000]);
    }

    // ----- cross-check vs naive reference on random strided inputs ------

    #[test]
    fn matmul_matches_naive_reference_random() {
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
                (self.next() % 400) as f32 / 100.0 - 2.0
            }
        }
        let mut rng = Prng(0x1234_5678_9ABC_DEF0);
        for _ in 0..200 {
            let batch = 1 + rng.below(2); // 1..2 batch
            let m = 1 + rng.below(3);
            let k = 1 + rng.below(4);
            let n = 1 + rng.below(3);
            let a: Vec<f32> = (0..batch * m * k).map(|_| rng.f()).collect();
            let b: Vec<f32> = (0..batch * k * n).map(|_| rng.f()).collect();
            let sa = f32_storage(a.clone());
            let sb = f32_storage(b.clone());
            let la = Layout::contiguous([batch, m, k]).unwrap();
            let lb = Layout::contiguous([batch, k, n]).unwrap();
            let got = as_f32(&matmul(View::new(&sa, &la), View::new(&sb, &lb)).unwrap());
            // naive
            let mut expected = vec![0.0f32; batch * m * n];
            for bi in 0..batch {
                for i in 0..m {
                    for j in 0..n {
                        let mut acc = 0.0f32;
                        for p in 0..k {
                            acc += a[bi * m * k + i * k + p] * b[bi * k * n + p * n + j];
                        }
                        expected[bi * m * n + i * n + j] = acc;
                    }
                }
            }
            for (g, e) in got.iter().zip(expected.iter()) {
                assert!((g - e).abs() < 1e-3, "matmul mismatch {g} vs {e}");
            }
        }
    }

    // ----- error contracts ---------------------------------------------

    #[test]
    fn matmul_inner_dim_mismatch_is_shape_error() {
        let a = f32_storage(vec![1.0; 6]); // 2x3
        let b = f32_storage(vec![1.0; 8]); // 4x2
        let la = Layout::contiguous([2, 3]).unwrap();
        let lb = Layout::contiguous([4, 2]).unwrap();
        assert!(matches!(
            matmul(View::new(&a, &la), View::new(&b, &lb)),
            Err(Error::ShapeMismatch { op: "matmul", .. })
        ));
    }

    #[test]
    fn matmul_incompatible_batch_is_shape_error() {
        let a = f32_storage(vec![1.0; 2 * 2 * 2]); // (2,2,2)
        let b = f32_storage(vec![1.0; 3 * 2 * 2]); // (3,2,2)
        let la = Layout::contiguous([2, 2, 2]).unwrap();
        let lb = Layout::contiguous([3, 2, 2]).unwrap();
        assert!(matches!(
            matmul(View::new(&a, &la), View::new(&b, &lb)),
            Err(Error::ShapeMismatch { op: "matmul", .. })
        ));
    }

    #[test]
    fn matmul_rank_too_low_is_invalid_arg() {
        let a = f32_storage(vec![1.0, 2.0, 3.0]);
        let b = f32_storage(vec![1.0, 2.0, 3.0]);
        let l = Layout::contiguous([3]).unwrap();
        assert!(matches!(
            matmul(View::new(&a, &l), View::new(&b, &l)),
            Err(Error::InvalidArg { op: "matmul", .. })
        ));
    }

    #[test]
    fn matmul_dtype_mismatch_is_loud() {
        let a = f32_storage(vec![1.0; 4]);
        let b = Storage::Cpu(CpuStorage::I64(Arc::new(vec![1; 4])));
        let l = Layout::contiguous([2, 2]).unwrap();
        assert!(matches!(
            matmul(View::new(&a, &l), View::new(&b, &l)),
            Err(Error::DTypeMismatch { op: "matmul", .. })
        ));
    }

    #[test]
    fn matmul_bool_is_unsupported() {
        let a = Storage::Cpu(CpuStorage::Bool(Arc::new(vec![true; 4])));
        let l = Layout::contiguous([2, 2]).unwrap();
        assert!(matches!(
            matmul(View::new(&a, &l), View::new(&a, &l)),
            Err(Error::Unsupported {
                op: "matmul",
                dtype: DType::Bool,
                ..
            })
        ));
    }
}
