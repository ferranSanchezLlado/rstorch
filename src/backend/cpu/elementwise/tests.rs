//! Property tests for the element-wise iteration engine.
//!
//! The engine is validated against a **naive gather reference**: an
//! independent re-derivation of the `(coords → storage index)` mapping,
//! enumerating logical coordinates in row-major order. Every kernel is
//! exercised over contiguous, permuted, narrowed, and broadcast views for
//! `F32`, `I64`, and `Bool` (the gate dtypes, plus float-only coverage where
//! the op requires it).

use super::*;
use crate::backend::{BinaryOp, CmpOp, UnaryOp};
use crate::layout::Layout;
use crate::shape::Shape;

// ---------------------------------------------------------------------------
// Test scaffolding
// ---------------------------------------------------------------------------

/// A tiny seeded xorshift PRNG (no external crate), matching the style used in
/// `layout.rs` tests.
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

/// Read the flat contiguous output buffer of a kernel result for element `E`.
fn out_slice<E: TypedSlice>(storage: &Storage) -> Vec<E> {
    match storage {
        Storage::Cpu(cpu) => E::slice(cpu, "test").unwrap().to_vec(),
        #[cfg(feature = "metal")]
        _ => panic!("non-cpu storage in cpu test"),
    }
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

/// The naive-gather reference: the storage index of `coords` under `layout`,
/// derived independently of `Cursor`.
fn ref_index(layout: &Layout, coords: &[usize]) -> usize {
    let mut idx = layout.offset();
    for (c, s) in coords.iter().zip(layout.strides()) {
        idx += c * s;
    }
    idx
}

/// Gather every logical element of a `(data, layout)` view into a dense
/// row-major `Vec`, using the reference index math.
fn gather<E: Copy>(data: &[E], layout: &Layout) -> Vec<E> {
    all_coords(layout.dims())
        .iter()
        .map(|c| data[ref_index(layout, c)])
        .collect()
}

// ---------------------------------------------------------------------------
// View builders
// ---------------------------------------------------------------------------

/// Owns a storage buffer + layout so a `View` can borrow both.
struct Owned<E: TypedSlice> {
    storage: Storage,
    layout: Layout,
    _marker: std::marker::PhantomData<E>,
}

impl<E: TypedSlice> Owned<E> {
    fn new(data: Vec<E>, layout: Layout) -> Owned<E> {
        Owned {
            storage: E::into_storage(data),
            layout,
            _marker: std::marker::PhantomData,
        }
    }
    fn view(&self) -> View<'_> {
        View::new(&self.storage, &self.layout)
    }
    fn data(&self) -> Vec<E> {
        out_slice::<E>(&self.storage)
    }
}

/// A contiguous rank-`dims` f32 tensor `[0, 1, 2, ...]`.
fn contig_f32(dims: &[usize]) -> Owned<f32> {
    let n: usize = dims.iter().product();
    let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 3.0).collect();
    Owned::new(data, Layout::contiguous(dims.to_vec()).unwrap())
}

fn contig_i64(dims: &[usize]) -> Owned<i64> {
    let n: usize = dims.iter().product();
    let data: Vec<i64> = (0..n).map(|i| i as i64 - 5).collect();
    Owned::new(data, Layout::contiguous(dims.to_vec()).unwrap())
}

fn contig_bool(dims: &[usize]) -> Owned<bool> {
    let n: usize = dims.iter().product();
    let data: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
    Owned::new(data, Layout::contiguous(dims.to_vec()).unwrap())
}

// ---------------------------------------------------------------------------
// Cursor unit tests: the engine matches the reference on every view kind.
// ---------------------------------------------------------------------------

#[test]
fn cursor_matches_reference_on_contiguous_permuted_narrowed_broadcast() {
    let base = contig_f32(&[2, 3, 4]);
    let data = base.data();

    // Contiguous.
    let l = base.layout.clone();
    check_cursor(&data, &l);

    // Transposed / permuted.
    check_cursor(&data, &l.transpose(0, 2).unwrap());
    check_cursor(&data, &l.permute(&[2, 0, 1]).unwrap());

    // Narrowed.
    check_cursor(&data, &l.narrow(1, 1, 2).unwrap());
    check_cursor(&data, &l.narrow(2, 0, 3).unwrap().narrow(0, 1, 1).unwrap());

    // Broadcast: a size-1 axis expanded, plus a fresh leading axis.
    let small = contig_f32(&[1, 4]);
    let sdata = small.data();
    let b = small.layout.broadcast_to(&Shape::from([3, 5, 4])).unwrap();
    check_cursor(&sdata, &b);
}

/// The `Cursor` walk must reproduce the reference gather for `layout`.
fn check_cursor(data: &[f32], layout: &Layout) {
    let cursor = Cursor::new(layout);
    let n = layout.num_elements();
    let engine: Vec<f32> = (0..n).map(|i| data[cursor.index(i)]).collect();
    let reference = gather(data, layout);
    assert_eq!(engine, reference, "cursor mismatch for {:?}", layout.dims());
}

// ---------------------------------------------------------------------------
// binary
// ---------------------------------------------------------------------------

#[test]
fn binary_f32_over_broadcast_and_permuted_views() {
    // lhs: [2,3] contiguous; rhs: [3] broadcast to [2,3]; then permute both.
    let lhs = contig_f32(&[2, 3]);
    let rhs_base = contig_f32(&[1, 3]);
    let rhs_layout = rhs_base.layout.broadcast_to(&Shape::from([2, 3])).unwrap();
    let rhs = Owned::<f32>::new(rhs_base.data(), rhs_layout);

    for op in [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Div,
        BinaryOp::Maximum,
        BinaryOp::Minimum,
    ] {
        let got = binary(op, lhs.view(), rhs.view()).unwrap();
        let l = gather(&lhs.data(), &lhs.layout);
        let r = gather(&rhs.data(), &rhs.layout);
        let expected: Vec<f32> = l
            .iter()
            .zip(&r)
            .map(|(&a, &b)| binary_f32(op, a, b))
            .collect();
        assert_eq!(out_slice::<f32>(&got), expected, "binary f32 {op:?}");
    }
}

#[test]
fn binary_i64_arithmetic_matches_reference() {
    let lhs = contig_i64(&[2, 2, 2]);
    let rhs = contig_i64(&[2, 2, 2]);
    // Permute the lhs so a strided read is exercised.
    let lhs_perm = Owned::<i64>::new(lhs.data(), lhs.layout.transpose(0, 2).unwrap());

    for op in [
        BinaryOp::Add,
        BinaryOp::Sub,
        BinaryOp::Mul,
        BinaryOp::Maximum,
        BinaryOp::Minimum,
    ] {
        let got = binary(op, lhs_perm.view(), rhs.view()).unwrap();
        let l = gather(&lhs_perm.data(), &lhs_perm.layout);
        let r = gather(&rhs.data(), &rhs.layout);
        let expected: Vec<i64> = l
            .iter()
            .zip(&r)
            .map(|(&a, &b)| binary_i64(op, a, b))
            .collect();
        assert_eq!(out_slice::<i64>(&got), expected, "binary i64 {op:?}");
    }
}

/// Assert a kernel result is an `Unsupported` error for `dtype` (works
/// without `Storage: Debug`, which the frozen contract does not provide).
fn assert_unsupported(r: Result<Storage>, dtype: DType) {
    match r {
        Ok(_) => panic!("expected Unsupported error, got Ok"),
        Err(Error::Unsupported { dtype: d, .. }) => assert_eq!(d, dtype),
        Err(other) => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn binary_bool_is_unsupported() {
    let a = contig_bool(&[4]);
    let b = contig_bool(&[4]);
    assert_unsupported(binary(BinaryOp::Add, a.view(), b.view()), DType::Bool);
}

#[test]
fn binary_scalar_matches_reference() {
    let x = contig_f32(&[3, 4]).layout.transpose(0, 1).unwrap();
    let base = contig_f32(&[3, 4]);
    let xv = Owned::<f32>::new(base.data(), x);
    let got = binary_scalar(BinaryOp::Add, xv.view(), 2.5).unwrap();
    let expected: Vec<f32> = gather(&xv.data(), &xv.layout)
        .iter()
        .map(|&a| a + 2.5)
        .collect();
    assert_eq!(out_slice::<f32>(&got), expected);

    // i64 scalar div guards zero.
    let xi = contig_i64(&[4]);
    let got = binary_scalar(BinaryOp::Div, xi.view(), 0.0).unwrap();
    assert_eq!(out_slice::<i64>(&got), vec![0, 0, 0, 0]);
}

// ---------------------------------------------------------------------------
// unary
// ---------------------------------------------------------------------------

#[test]
fn unary_float_ops_match_reference_on_strided_view() {
    // Positive-only base so ln/sqrt are defined.
    let n = 12usize;
    let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.25 + 0.1).collect();
    let owned = Owned::<f32>::new(data, Layout::contiguous([3, 4]).unwrap());
    let strided = Owned::<f32>::new(owned.data(), owned.layout.transpose(0, 1).unwrap());

    for op in [
        UnaryOp::Relu,
        UnaryOp::Gelu,
        UnaryOp::Exp,
        UnaryOp::Ln,
        UnaryOp::Sqrt,
        UnaryOp::Tanh,
        UnaryOp::Sigmoid,
        UnaryOp::Neg,
        UnaryOp::Abs,
    ] {
        let got = unary(op, strided.view()).unwrap();
        let expected: Vec<f32> = gather(&strided.data(), &strided.layout)
            .iter()
            .map(|&a| unary_f64(op, a as f64) as f32)
            .collect();
        assert_eq!(out_slice::<f32>(&got), expected, "unary {op:?}");
    }
}

#[test]
fn unary_i64_neg_abs_only() {
    let x = contig_i64(&[5]); // -5..0
    let neg = unary(UnaryOp::Neg, x.view()).unwrap();
    assert_eq!(out_slice::<i64>(&neg), vec![5, 4, 3, 2, 1]);
    let abs = unary(UnaryOp::Abs, x.view()).unwrap();
    assert_eq!(out_slice::<i64>(&abs), vec![5, 4, 3, 2, 1]);
    // Float-only ops are unsupported on i64.
    for op in [UnaryOp::Relu, UnaryOp::Gelu, UnaryOp::Exp, UnaryOp::Sqrt] {
        assert_unsupported(unary(op, x.view()), DType::I64);
    }
}

#[test]
fn unary_bool_is_unsupported() {
    let x = contig_bool(&[4]);
    assert_unsupported(unary(UnaryOp::Neg, x.view()), DType::Bool);
}

#[test]
fn gelu_is_exact_not_tanh_approx() {
    // Exact GELU at x=1: 0.5 * 1 * (1 + erf(1/sqrt2)) = 0.8413447460685429...
    let x = Owned::<f64>::new(vec![1.0], Layout::contiguous([1]).unwrap());
    let got = unary(UnaryOp::Gelu, x.view()).unwrap();
    let v = out_slice::<f64>(&got)[0];
    let exact = 0.5 * (1.0 + erf(std::f64::consts::FRAC_1_SQRT_2));
    assert!((v - exact).abs() < 1e-12);
    // The tanh approximation of GELU at x=1:
    // 0.5 * (1 + tanh(sqrt(2/pi) * (1 + 0.044715))). It differs from exact by
    // ~1.5e-4, so a GELU that used it would be caught here.
    let tanh_approx: f64 =
        0.5 * (1.0 + (f64::sqrt(2.0 / std::f64::consts::PI) * (1.0 + 0.044715)).tanh());
    assert!((v - tanh_approx).abs() > 1e-4, "must not equal tanh approx");
}

#[test]
fn erf_matches_known_values() {
    // Reference values from a high-precision erf.
    let cases = [
        (0.0, 0.0),
        (0.5, 0.520_499_877_813_046_5),
        (1.0, 0.842_700_792_949_714_9),
        (2.0, 0.995_322_265_018_952_7),
        (-1.0, -0.842_700_792_949_714_9),
        (3.0, 0.999_977_909_503_001_4),
    ];
    for (x, want) in cases {
        assert!(
            (erf(x) - want).abs() < 1e-12,
            "erf({x}) = {} != {want}",
            erf(x)
        );
    }
    // Saturation and NaN.
    assert_eq!(erf(10.0), 1.0);
    assert_eq!(erf(-10.0), -1.0);
    assert!(erf(f64::NAN).is_nan());
}

// ---------------------------------------------------------------------------
// compare
// ---------------------------------------------------------------------------

#[test]
fn compare_produces_bool_matching_reference() {
    let lhs = contig_i64(&[2, 3]);
    let rhs_base = contig_i64(&[3]);
    let rhs = Owned::<i64>::new(
        rhs_base.data(),
        rhs_base.layout.broadcast_to(&Shape::from([2, 3])).unwrap(),
    );
    for op in [
        CmpOp::Eq,
        CmpOp::Ne,
        CmpOp::Lt,
        CmpOp::Le,
        CmpOp::Gt,
        CmpOp::Ge,
    ] {
        let got = compare(op, lhs.view(), rhs.view()).unwrap();
        assert_eq!(got.dtype(), DType::Bool);
        let l = gather(&lhs.data(), &lhs.layout);
        let r = gather(&rhs.data(), &rhs.layout);
        let expected: Vec<bool> = l
            .iter()
            .zip(&r)
            .map(|(a, b)| match op {
                CmpOp::Eq => a == b,
                CmpOp::Ne => a != b,
                CmpOp::Lt => a < b,
                CmpOp::Le => a <= b,
                CmpOp::Gt => a > b,
                CmpOp::Ge => a >= b,
            })
            .collect();
        assert_eq!(out_slice::<bool>(&got), expected, "compare i64 {op:?}");
    }
}

#[test]
fn compare_works_on_bool_and_f32() {
    let a = contig_bool(&[4]);
    let b = contig_bool(&[4]);
    let got = compare(CmpOp::Eq, a.view(), b.view()).unwrap();
    assert_eq!(out_slice::<bool>(&got), vec![true, true, true, true]);

    let x = contig_f32(&[4]);
    let y = contig_f32(&[4]);
    let lt = compare(CmpOp::Lt, x.view(), y.view()).unwrap();
    assert_eq!(out_slice::<bool>(&lt), vec![false, false, false, false]);
}

// ---------------------------------------------------------------------------
// where_cond
// ---------------------------------------------------------------------------

#[test]
fn where_selects_by_condition_over_broadcast() {
    // cond: [2,1] broadcast to [2,3]; on_true/on_false contiguous [2,3].
    let cond_base = contig_bool(&[2, 1]);
    let cond = Owned::<bool>::new(
        cond_base.data(),
        cond_base.layout.broadcast_to(&Shape::from([2, 3])).unwrap(),
    );
    let t = contig_f32(&[2, 3]);
    let f = Owned::<f32>::new(vec![-1.0; 6], Layout::contiguous([2, 3]).unwrap());

    let got = where_cond(cond.view(), t.view(), f.view()).unwrap();
    let c = gather(&cond.data(), &cond.layout);
    let tt = gather(&t.data(), &t.layout);
    let ff = gather(&f.data(), &f.layout);
    let expected: Vec<f32> = (0..6).map(|i| if c[i] { tt[i] } else { ff[i] }).collect();
    assert_eq!(out_slice::<f32>(&got), expected);
}

#[test]
fn where_works_for_i64_and_bool() {
    let cond = contig_bool(&[4]);
    let t = contig_i64(&[4]);
    let f = Owned::<i64>::new(vec![99; 4], Layout::contiguous([4]).unwrap());
    let got = where_cond(cond.view(), t.view(), f.view()).unwrap();
    let c = cond.data();
    let tt = t.data();
    let expected: Vec<i64> = (0..4).map(|i| if c[i] { tt[i] } else { 99 }).collect();
    assert_eq!(out_slice::<i64>(&got), expected);
}

// ---------------------------------------------------------------------------
// masked_fill
// ---------------------------------------------------------------------------

#[test]
fn masked_fill_replaces_where_mask_true_over_broadcast_mask() {
    let x = contig_f32(&[2, 3]);
    // mask: [3] broadcast to [2,3].
    let mask_base = contig_bool(&[3]);
    let mask = Owned::<bool>::new(
        mask_base.data(),
        mask_base.layout.broadcast_to(&Shape::from([2, 3])).unwrap(),
    );
    let got = masked_fill(x.view(), mask.view(), 7.0).unwrap();
    let xg = gather(&x.data(), &x.layout);
    let mg = gather(&mask.data(), &mask.layout);
    let expected: Vec<f32> = xg
        .iter()
        .zip(&mg)
        .map(|(&v, &m)| if m { 7.0 } else { v })
        .collect();
    assert_eq!(out_slice::<f32>(&got), expected);
}

#[test]
fn masked_fill_over_permuted_x() {
    let base = contig_i64(&[3, 4]);
    let x = Owned::<i64>::new(base.data(), base.layout.transpose(0, 1).unwrap());
    let mask = Owned::<bool>::new(
        (0..12).map(|i| i % 2 == 0).collect(),
        Layout::contiguous([4, 3]).unwrap(),
    );
    let got = masked_fill(x.view(), mask.view(), -1.0).unwrap();
    let xg = gather(&x.data(), &x.layout);
    let mg = mask.data();
    let expected: Vec<i64> = xg
        .iter()
        .zip(&mg)
        .map(|(&v, &m)| if m { -1 } else { v })
        .collect();
    assert_eq!(out_slice::<i64>(&got), expected);
}

// ---------------------------------------------------------------------------
// Empty views: kernels must not panic and produce empty output.
// ---------------------------------------------------------------------------

#[test]
fn empty_views_yield_empty_output() {
    let a = contig_f32(&[0, 3]);
    let b = contig_f32(&[0, 3]);
    let add = binary(BinaryOp::Add, a.view(), b.view()).unwrap();
    assert_eq!(add.len(), 0);
    let neg = unary(UnaryOp::Neg, a.view()).unwrap();
    assert_eq!(neg.len(), 0);
    let cmp = compare(CmpOp::Lt, a.view(), b.view()).unwrap();
    assert_eq!(cmp.len(), 0);

    let cond = contig_bool(&[0, 3]);
    let w = where_cond(cond.view(), a.view(), b.view()).unwrap();
    assert_eq!(w.len(), 0);
    let mf = masked_fill(a.view(), cond.view(), 1.0).unwrap();
    assert_eq!(mf.len(), 0);
}

// ---------------------------------------------------------------------------
// Randomised stress: random strided source views vs the naive reference,
// across F32 / I64 / Bool (the gate dtypes).
// ---------------------------------------------------------------------------

/// Build a random zero-copy view (transpose/narrow/broadcast chain) over a
/// fresh contiguous base of `dims`, returning the resulting layout.
fn random_view(rng: &mut Prng, dims: &[usize]) -> Layout {
    let mut layout = Layout::contiguous(dims.to_vec()).unwrap();
    for _ in 0..rng.below(3) {
        let r = layout.rank();
        match rng.below(3) {
            0 if r >= 2 => {
                let a = rng.below(r);
                let b = rng.below(r);
                layout = layout.transpose(a, b).unwrap();
            }
            1 => {
                let axis = rng.below(layout.rank());
                let size = layout.dims()[axis];
                let start = rng.below(size);
                let len = 1 + rng.below(size - start);
                layout = layout.narrow(axis, start, len).unwrap();
            }
            _ => {}
        }
    }
    layout
}

#[test]
fn randomised_binary_matches_reference_f32() {
    let mut rng = Prng(0xDEAD_BEEF_1234_5678);
    for _ in 0..400 {
        let rank = 1 + rng.below(3);
        let dims: Vec<usize> = (0..rank).map(|_| 1 + rng.below(3)).collect();
        let n: usize = dims.iter().product();
        // lhs strided view over a contiguous base; rhs contiguous same shape.
        let base: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3 - 1.0).collect();
        let lhs_layout = random_view(&mut rng, &dims);
        // rhs shares lhs's *shape* (post-view) via a fresh contiguous buffer.
        let vshape = lhs_layout.dims().to_vec();
        let vn: usize = vshape.iter().product();
        let rhs_data: Vec<f32> = (0..vn).map(|i| (i as f32) * 0.7 + 0.2).collect();

        let lhs = Owned::<f32>::new(base.clone(), lhs_layout);
        let rhs = Owned::<f32>::new(rhs_data, Layout::contiguous(vshape).unwrap());

        let op = [BinaryOp::Add, BinaryOp::Mul, BinaryOp::Maximum][rng.below(3)];
        let got = binary(op, lhs.view(), rhs.view()).unwrap();
        let l = gather(&lhs.data(), &lhs.layout);
        let r = gather(&rhs.data(), &rhs.layout);
        let expected: Vec<f32> = l
            .iter()
            .zip(&r)
            .map(|(&a, &b)| binary_f32(op, a, b))
            .collect();
        assert_eq!(out_slice::<f32>(&got), expected);
    }
}

#[test]
fn randomised_compare_matches_reference_i64() {
    let mut rng = Prng(0x0BAD_F00D_CAFE_0001);
    for _ in 0..400 {
        let rank = 1 + rng.below(3);
        let dims: Vec<usize> = (0..rank).map(|_| 1 + rng.below(3)).collect();
        let n: usize = dims.iter().product();
        let base: Vec<i64> = (0..n).map(|i| (i as i64) % 4 - 1).collect();
        let lhs_layout = random_view(&mut rng, &dims);
        let vshape = lhs_layout.dims().to_vec();
        let vn: usize = vshape.iter().product();
        let rhs_data: Vec<i64> = (0..vn).map(|i| (i as i64) % 3 - 1).collect();

        let lhs = Owned::<i64>::new(base.clone(), lhs_layout);
        let rhs = Owned::<i64>::new(rhs_data, Layout::contiguous(vshape).unwrap());

        let op = [CmpOp::Lt, CmpOp::Eq, CmpOp::Ge][rng.below(3)];
        let got = compare(op, lhs.view(), rhs.view()).unwrap();
        let l = gather(&lhs.data(), &lhs.layout);
        let r = gather(&rhs.data(), &rhs.layout);
        let expected: Vec<bool> = l
            .iter()
            .zip(&r)
            .map(|(a, b)| match op {
                CmpOp::Lt => a < b,
                CmpOp::Eq => a == b,
                CmpOp::Ge => a >= b,
                _ => unreachable!(),
            })
            .collect();
        assert_eq!(out_slice::<bool>(&got), expected);
    }
}

#[test]
fn randomised_masked_fill_matches_reference_bool_payload() {
    // Payload dtype = Bool exercises the Bool lane of masked_fill; mask is a
    // random strided view.
    let mut rng = Prng(0xF00D_BABE_2222_3333);
    for _ in 0..300 {
        let rank = 1 + rng.below(3);
        let dims: Vec<usize> = (0..rank).map(|_| 1 + rng.below(3)).collect();
        let n: usize = dims.iter().product();
        let x_data: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        let x = Owned::<bool>::new(x_data, Layout::contiguous(dims.clone()).unwrap());

        let mask_layout = random_view(&mut rng, &dims);
        // A mask over the *same* logical shape but possibly strided: build it
        // over a contiguous base of the pre-view dims and re-view.
        let mask_base: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
        let mask = Owned::<bool>::new(mask_base, mask_layout.clone());
        // masked_fill requires mask shape == x shape; only use views that
        // preserve the shape (transpose keeps numel but not dims order), so
        // restrict to same-dims views here.
        if mask.layout.dims() != x.layout.dims() {
            continue;
        }

        let got = masked_fill(x.view(), mask.view(), 1.0).unwrap();
        let xg = gather(&x.data(), &x.layout);
        let mg = gather(&mask.data(), &mask.layout);
        let expected: Vec<bool> = xg
            .iter()
            .zip(&mg)
            .map(|(&v, &m)| if m { true } else { v })
            .collect();
        assert_eq!(out_slice::<bool>(&got), expected);
    }
}
