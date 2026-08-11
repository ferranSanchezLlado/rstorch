//! The matmul kernel across loop orders, batch broadcasting, and stride-aware
//! operand views.

use super::*;
use crate::backend::View;
use crate::layout::Layout;
use crate::shape::Shape;
use crate::storage::CpuStorage;
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

/// The resolved output shape of a matmul: broadcast batch dims followed
/// by `[m, n]`.
fn matmul_shape(lhs: &Layout, rhs: &Layout) -> Shape {
    let p = plan(lhs, rhs).unwrap();
    let mut dims = p.batch.clone();
    dims.push(p.m);
    dims.push(p.n);
    Shape::from(dims)
}

/// xorshift64: reproducible inputs without pulling in a dependency.
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

    /// A value with a deliberately wide exponent spread (roughly 2^-12 to
    /// 2^11). Summing these is order-sensitive in the low mantissa bits, so
    /// a reassociated accumulation shows up in a bitwise comparison — which
    /// `f`, whose magnitudes all sit within 2^2, would mostly hide.
    fn wide(&mut self) -> f32 {
        let bits = self.next();
        let mantissa = (bits % 2048) as f32 / 1024.0 + 1.0;
        let exp = ((bits >> 11) % 24) as i32 - 12;
        let sign = if bits >> 63 == 0 { 1.0 } else { -1.0 };
        sign * mantissa * 2f32.powi(exp)
    }
}

/// `b` (logically `[k, n]`, row-major) rebuilt as a `[k, n]` *view* whose
/// `n` stride is `k` rather than 1: the buffer holds `bᵀ`, and transposing
/// its layout gives back `b`. This is the operand shape `Linear` feeds
/// `matmul` (`x @ w.T`), and it is what selects the strided loop order.
fn transposed_view_of(b: &[f32], k: usize, n: usize) -> (Storage, Layout) {
    let bt: Vec<f32> = (0..n * k).map(|idx| b[(idx % k) * n + idx / k]).collect();
    let layout = Layout::contiguous([n, k]).unwrap().transpose(0, 1).unwrap();
    (f32_storage(bt), layout)
}

/// The naive `i` → `j` → `p` nest both loop orders must match bit for bit,
/// with the same unfused `acc + a*b` step `NumAcc::mul_add` performs.
fn naive(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(m * n);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            out.push(acc);
        }
    }
    out
}

/// Bit-exact comparison. `assert_eq!` on `f32` would also accept a
/// `-0.0`/`0.0` swap; compare bit patterns so nothing is waved through.
fn assert_bitwise_eq(got: &[f32], expected: &[f32], what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: length");
    for (idx, (g, e)) in got.iter().zip(expected).enumerate() {
        assert_eq!(
            g.to_bits(),
            e.to_bits(),
            "{what}: element {idx} differs: {g:e} vs {e:e}"
        );
    }
}

// ----- loop order ---------------------------------------------------

#[test]
fn both_loop_orders_are_bitwise_identical_to_the_naive_nest() {
    // k is large enough, and the exponent spread wide enough, that any
    // reassociation of the k-term sum would perturb the mantissa.
    let (m, k, n) = (7usize, 129usize, 12usize);
    let mut rng = Prng(0x0BAD_F00D_DEAD_BEEF);
    let a: Vec<f32> = (0..m * k).map(|_| rng.wide()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| rng.wide()).collect();
    let expected = naive(&a, &b, m, k, n);

    let sa = f32_storage(a.clone());
    let la = Layout::contiguous([m, k]).unwrap();

    // Row-major rhs -> the unit-stride `i`/`p`/`j` row-accumulator order.
    let sb = f32_storage(b.clone());
    let lb = Layout::contiguous([k, n]).unwrap();
    assert_eq!(
        plan(&la, &lb).unwrap().rhs_n_stride,
        1,
        "expected the row-accumulator order"
    );
    let row_major = as_f32(&matmul(View::new(&sa, &la), View::new(&sb, &lb)).unwrap());
    assert_bitwise_eq(&row_major, &expected, "row-major rhs");

    // Transposed rhs view -> the column-blocked `i`/`j`/`p` order.
    let (sbt, lbt) = transposed_view_of(&b, k, n);
    assert_ne!(
        plan(&la, &lbt).unwrap().rhs_n_stride,
        1,
        "expected the strided order"
    );
    let strided = as_f32(&matmul(View::new(&sa, &la), View::new(&sbt, &lbt)).unwrap());
    assert_bitwise_eq(&strided, &expected, "transposed rhs view");
}

#[test]
fn column_block_tail_is_exact_for_every_width() {
    // The strided order accumulates COL_BLOCK columns at a time; n from 1
    // to 2*COL_BLOCK+1 covers a pure tail, whole blocks, and both mixes.
    let (m, k) = (3usize, 17usize);
    let mut rng = Prng(0xFEED_FACE_CAFE_D00D);
    for n in 1..=(2 * COL_BLOCK + 1) {
        let a: Vec<f32> = (0..m * k).map(|_| rng.wide()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| rng.wide()).collect();
        let expected = naive(&a, &b, m, k, n);
        let sa = f32_storage(a);
        let la = Layout::contiguous([m, k]).unwrap();
        let (sbt, lbt) = transposed_view_of(&b, k, n);
        let got = as_f32(&matmul(View::new(&sa, &la), View::new(&sbt, &lbt)).unwrap());
        assert_bitwise_eq(&got, &expected, &format!("strided rhs, n={n}"));
    }
}

#[test]
fn both_loop_orders_agree_on_a_batched_broadcast_operand() {
    // Batch decoding is shared by the two orders; check a broadcast rhs
    // still lands on the same bits through either.
    let (batch, m, k, n) = (3usize, 2usize, 33usize, 5usize);
    let mut rng = Prng(0x5EED_1234_5678_9ABC);
    let a: Vec<f32> = (0..batch * m * k).map(|_| rng.wide()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| rng.wide()).collect();
    let sa = f32_storage(a.clone());
    let la = Layout::contiguous([batch, m, k]).unwrap();

    let sb = f32_storage(b.clone());
    let lb = Layout::contiguous([1, k, n]).unwrap();
    let row_major = as_f32(&matmul(View::new(&sa, &la), View::new(&sb, &lb)).unwrap());

    let (sbt, lbt) = transposed_view_of(&b, k, n);
    let strided = as_f32(&matmul(View::new(&sa, &la), View::new(&sbt, &lbt)).unwrap());
    assert_bitwise_eq(&strided, &row_major, "batched broadcast rhs");

    // And each batch equals the 2-D product of that lhs slice.
    for bi in 0..batch {
        let expected = naive(&a[bi * m * k..], &b, m, k, n);
        assert_bitwise_eq(
            &row_major[bi * m * n..(bi + 1) * m * n],
            &expected,
            &format!("batch {bi}"),
        );
    }
}

#[test]
fn transposed_lhs_f32_path_is_bitwise_exact_for_training_shape() {
    // The first MLP weight gradient is `[784, 64] @ [64, 128]`; these
    // reduced dimensions exercise the identical transposed-lhs stride
    // pattern, a non-multiple row-block tail, and a non-trivial inner sum.
    let (m, k, n) = (131usize, 64usize, 19usize);
    let mut rng = Prng(0xA11C_E5E5_1234_5678);
    let logical_a: Vec<f32> = (0..m * k).map(|_| rng.wide()).collect();
    let stored_a: Vec<f32> = (0..k * m)
        .map(|idx| logical_a[(idx % m) * k + idx / m])
        .collect();
    let b: Vec<f32> = (0..k * n).map(|_| rng.wide()).collect();
    let expected = naive(&logical_a, &b, m, k, n);

    let sa = f32_storage(stored_a);
    let la = Layout::contiguous([k, m]).unwrap().transpose(0, 1).unwrap();
    let sb = f32_storage(b);
    let lb = Layout::contiguous([k, n]).unwrap();
    let got = as_f32(&matmul(View::new(&sa, &la), View::new(&sb, &lb)).unwrap());

    assert_eq!(matmul_shape(&la, &lb).dims(), &[m, n]);
    assert_bitwise_eq(&got, &expected, "training transposed lhs");
}

#[test]
fn transposed_lhs_f32_path_preserves_batched_broadcasting() {
    let (batch, m, k, n) = (3usize, 9usize, 17usize, 7usize);
    let mut rng = Prng(0xBA7C_4ED0_1234_5678);
    let logical_a: Vec<f32> = (0..m * k).map(|_| rng.wide()).collect();
    let stored_a: Vec<f32> = (0..k * m)
        .map(|idx| logical_a[(idx % m) * k + idx / m])
        .collect();
    let b: Vec<f32> = (0..batch * k * n).map(|_| rng.wide()).collect();
    let sa = f32_storage(stored_a);
    let la = Layout::contiguous([1, k, m])
        .unwrap()
        .transpose(1, 2)
        .unwrap();
    let sb = f32_storage(b.clone());
    let lb = Layout::contiguous([batch, k, n]).unwrap();
    let got = as_f32(&matmul(View::new(&sa, &la), View::new(&sb, &lb)).unwrap());

    assert_eq!(matmul_shape(&la, &lb).dims(), &[batch, m, n]);
    for bi in 0..batch {
        let expected = naive(&logical_a, &b[bi * k * n..(bi + 1) * k * n], m, k, n);
        assert_bitwise_eq(
            &got[bi * m * n..(bi + 1) * m * n],
            &expected,
            &format!("transposed lhs broadcast batch {bi}"),
        );
    }
}

#[test]
fn f16_wide_accumulator_holds_on_both_loop_orders() {
    // The `Acc` contract has to survive both paths: f16 accumulation of
    // ones stalls at 2048, while the required f32 accumulator reaches 4096.
    let (m, k, n) = (2usize, 4096usize, 6usize);
    let one = half::f16::from_f32(1.0);
    let f16s = |v: Vec<half::f16>| Storage::Cpu(CpuStorage::F16(Arc::new(v)));
    let as_f16 = |s: &Storage| match s {
        Storage::Cpu(CpuStorage::F16(v)) => v.as_ref().clone(),
        _ => panic!("expected f16 storage"),
    };
    let sa = f16s(vec![one; m * k]);
    let la = Layout::contiguous([m, k]).unwrap();

    let sb = f16s(vec![one; k * n]);
    let lb = Layout::contiguous([k, n]).unwrap();
    let row_major = as_f16(&matmul(View::new(&sa, &la), View::new(&sb, &lb)).unwrap());

    let sbt = f16s(vec![one; n * k]);
    let lbt = Layout::contiguous([n, k]).unwrap().transpose(0, 1).unwrap();
    let strided = as_f16(&matmul(View::new(&sa, &la), View::new(&sbt, &lbt)).unwrap());

    assert!(row_major.iter().all(|x| x.to_f32() == 4096.0));
    assert_eq!(row_major, strided);
}

#[test]
fn zero_sized_output_is_empty_and_does_not_panic() {
    // A zero-extent *batch* axis (the third case) used to panic with a
    // divide-by-zero while decoding batch coords: the batch count was
    // clamped up to 1, so the decode ran anyway and took `rem % 0`.
    for (ad, bd) in [
        (vec![0usize, 3], vec![3usize, 2]),
        (vec![2, 3], vec![3, 0]),
        (vec![0, 2, 3], vec![0, 3, 2]),
    ] {
        let la = Layout::contiguous(ad.clone()).unwrap();
        let lb = Layout::contiguous(bd.clone()).unwrap();
        // Size each buffer to its own view, so this exercises a genuine
        // zero-sized walk rather than an undersized storage.
        let a = f32_storage(vec![1.0; la.shape().num_elements()]);
        let b = f32_storage(vec![1.0; lb.shape().num_elements()]);
        let r = matmul(View::new(&a, &la), View::new(&b, &lb)).unwrap();
        assert!(
            as_f32(&r).is_empty(),
            "expected an empty output for {ad:?} @ {bd:?}"
        );
        assert_eq!(matmul_shape(&la, &lb).num_elements(), 0);
    }
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
fn bf16_matmul_accumulates_in_f32() {
    let k = 4096usize;
    let one = half::bf16::from_f32(1.0);
    let a = Storage::Cpu(CpuStorage::BF16(Arc::new(vec![one; k])));
    let b = Storage::Cpu(CpuStorage::BF16(Arc::new(vec![one; k * 2])));
    let la = Layout::contiguous([1, k]).unwrap();
    let lb = Layout::contiguous([k, 2]).unwrap();
    let r = matmul(View::new(&a, &la), View::new(&b, &lb)).unwrap();
    let Storage::Cpu(CpuStorage::BF16(out)) = r else {
        panic!("expected bf16")
    };
    assert!(out.iter().all(|value| value.to_f32() == 4096.0));

    let transposed = Storage::Cpu(CpuStorage::BF16(Arc::new(vec![one; k * 2])));
    let transposed_layout = Layout::contiguous([2, k]).unwrap().transpose(0, 1).unwrap();
    let r = matmul(
        View::new(&a, &la),
        View::new(&transposed, &transposed_layout),
    )
    .unwrap();
    let Storage::Cpu(CpuStorage::BF16(out)) = r else {
        panic!("expected bf16")
    };
    assert!(out.iter().all(|value| value.to_f32() == 4096.0));
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
