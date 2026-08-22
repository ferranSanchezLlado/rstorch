//! The element-wise op surface: broadcasting, dtype rules, scalar variants,
//! and gradients.

use super::*;
use crate::device::Device;
use crate::error::Error;
use crate::shape::Shape;
use crate::testing::check_grad;

const CPU: Device = Device::Cpu;

fn t(data: &[f32], shape: impl Into<Shape>) -> Tensor {
    Tensor::from_vec(data.to_vec(), shape, &CPU).unwrap()
}

fn v(t: &Tensor) -> Vec<f32> {
    t.to_vec::<f32>().unwrap()
}

/// Re-view `x` through `layout` over the same storage — the only way to
/// build a non-contiguous tensor without the public view ops.
fn re_view(x: &Tensor, layout: Layout) -> Tensor {
    Tensor::from_parts(x.storage().clone(), layout)
}

fn close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= tol, "{a:?} vs {b:?}");
    }
}

// ------------------------------------------------------------------
// Arithmetic
// ------------------------------------------------------------------

#[test]
fn same_shape_arithmetic() {
    let a = t(&[1.0, 2.0, 3.0, 4.0], [2, 2]);
    let b = t(&[10.0, 20.0, 30.0, 40.0], [2, 2]);
    assert_eq!(v(&a.add(&b).unwrap()), vec![11.0, 22.0, 33.0, 44.0]);
    assert_eq!(v(&a.sub(&b).unwrap()), vec![-9.0, -18.0, -27.0, -36.0]);
    assert_eq!(v(&a.mul(&b).unwrap()), vec![10.0, 40.0, 90.0, 160.0]);
    close(&v(&b.div(&a).unwrap()), &[10.0, 10.0, 10.0, 10.0], 1e-6);
    // Shape, dtype and device survive.
    let s = a.add(&b).unwrap();
    assert_eq!(s.dims(), &[2, 2]);
    assert_eq!(s.dtype(), DType::F32);
    assert_eq!(s.device(), CPU);
    assert!(s.is_contiguous());
}

#[test]
fn broadcasting_follows_numpy_rules() {
    let m = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);

    // Row vector against a matrix.
    let row = t(&[10.0, 20.0, 30.0], [3]);
    let out = m.add(&row).unwrap();
    assert_eq!(out.dims(), &[2, 3]);
    assert_eq!(v(&out), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);

    // Column vector against a matrix.
    let col = t(&[100.0, 200.0], [2, 1]);
    assert_eq!(
        v(&m.add(&col).unwrap()),
        vec![101.0, 102.0, 103.0, 204.0, 205.0, 206.0]
    );

    // Outer-product style: [2, 1] × [1, 3] -> [2, 3].
    let a = t(&[1.0, 2.0], [2, 1]);
    let b = t(&[10.0, 20.0, 30.0], [1, 3]);
    let out = a.mul(&b).unwrap();
    assert_eq!(out.dims(), &[2, 3]);
    assert_eq!(v(&out), vec![10.0, 20.0, 30.0, 20.0, 40.0, 60.0]);

    // A rank-0 operand broadcasts against anything.
    let s = t(&[2.0], ());
    assert_eq!(v(&m.mul(&s).unwrap()), vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn scalar_variants() {
    let a = t(&[1.0, 2.0, 4.0], [3]);
    assert_eq!(v(&a.add_scalar(1.0).unwrap()), vec![2.0, 3.0, 5.0]);
    assert_eq!(v(&a.sub_scalar(1.0).unwrap()), vec![0.0, 1.0, 3.0]);
    assert_eq!(v(&a.mul_scalar(2.5).unwrap()), vec![2.5, 5.0, 10.0]);
    assert_eq!(v(&a.div_scalar(2.0).unwrap()), vec![0.5, 1.0, 2.0]);
    // The scalar is narrowed to the tensor's dtype.
    let i = Tensor::from_vec(vec![1i64, 2, 3], [3], &CPU).unwrap();
    let out = i.add_scalar(2.9).unwrap();
    assert_eq!(out.dtype(), DType::I64);
    assert_eq!(out.to_vec::<i64>().unwrap(), vec![3, 4, 5]);
}

#[test]
fn scalar_variants_walk_strided_inputs_and_name_themselves_in_errors() {
    let base = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let transposed = re_view(&base, base.layout().transpose(0, 1).unwrap());
    let out = transposed.mul_scalar(10.0).unwrap();
    assert_eq!(out.dims(), &[3, 2]);
    assert!(out.is_contiguous());
    assert_eq!(v(&out), vec![10.0, 40.0, 20.0, 50.0, 30.0, 60.0]);

    // The kernel reports the op *family* (`"add"`); the op layer relabels
    // it with the public method the caller actually used.
    let b = Tensor::from_vec(vec![true, false], [2], &CPU).unwrap();
    assert!(matches!(
        b.add_scalar(1.0),
        Err(Error::Unsupported {
            op: "add_scalar",
            ..
        })
    ));
    assert!(matches!(
        b.div_scalar(1.0),
        Err(Error::Unsupported {
            op: "div_scalar",
            ..
        })
    ));
}

#[test]
fn maximum_and_minimum() {
    let a = t(&[1.0, 5.0, -2.0], [3]);
    let b = t(&[3.0, 3.0, -3.0], [3]);
    assert_eq!(v(&a.maximum(&b).unwrap()), vec![3.0, 5.0, -2.0]);
    assert_eq!(v(&a.minimum(&b).unwrap()), vec![1.0, 3.0, -3.0]);
    // Broadcasting works here too.
    let s = t(&[0.0], ());
    assert_eq!(v(&a.maximum(&s).unwrap()), vec![1.0, 5.0, 0.0]);
}

/// `extremum_backward` is a pure function over already-detached operands,
/// so unlike the closures it lives inside it can be driven without the
/// engine exists. Ties and broadcast reduction are the parts worth
/// pinning down early.
#[test]
fn extremum_backward_splits_ties_and_reduces_broadcasts() {
    let a = t(&[1.0, 3.0, 5.0], [3]);
    let b = t(&[2.0, 3.0, 4.0], [3]);
    let g = t(&[10.0, 10.0, 10.0], [3]);

    // b, tie, a wins: the tied element hands each side half the cotangent.
    let grads = extremum_backward(&g, &a, &b, a.dims(), b.dims(), true).unwrap();
    assert_eq!(v(grads[0].as_ref().unwrap()), vec![0.0, 5.0, 10.0]);
    assert_eq!(v(grads[1].as_ref().unwrap()), vec![10.0, 5.0, 0.0]);

    // `minimum` flips which side wins, tie handling unchanged.
    let grads = extremum_backward(&g, &a, &b, a.dims(), b.dims(), false).unwrap();
    assert_eq!(v(grads[0].as_ref().unwrap()), vec![10.0, 5.0, 0.0]);
    assert_eq!(v(grads[1].as_ref().unwrap()), vec![0.0, 5.0, 10.0]);

    // A broadcast operand's cotangent comes back at that operand's own
    // shape: max([[1,1],[4,4]], [[2,3],[2,3]]) takes the row in the top
    // half and the column in the bottom half.
    let col = t(&[1.0, 4.0], [2, 1]);
    let row = t(&[2.0, 3.0], [2]);
    let g = t(&[1.0, 1.0, 1.0, 1.0], [2, 2]);
    let grads = extremum_backward(&g, &col, &row, col.dims(), row.dims(), true).unwrap();
    assert_eq!(grads[0].as_ref().unwrap().dims(), &[2, 1]);
    assert_eq!(v(grads[0].as_ref().unwrap()), vec![0.0, 2.0]);
    assert_eq!(grads[1].as_ref().unwrap().dims(), &[2]);
    assert_eq!(v(grads[1].as_ref().unwrap()), vec![1.0, 1.0]);
}

#[test]
fn integer_arithmetic_and_bool_rejection() {
    let a = Tensor::from_vec(vec![7i64, -7, 6], [3], &CPU).unwrap();
    let b = Tensor::from_vec(vec![2i64, 2, 0], [3], &CPU).unwrap();
    assert_eq!(a.add(&b).unwrap().to_vec::<i64>().unwrap(), vec![9, -5, 6]);
    // Integer division truncates; a zero divisor yields 0, never a panic.
    assert_eq!(a.div(&b).unwrap().to_vec::<i64>().unwrap(), vec![3, -3, 0]);

    let x = Tensor::from_vec(vec![true, false], [2], &CPU).unwrap();
    assert!(matches!(
        x.add(&x),
        Err(Error::Unsupported { op: "add", .. })
    ));
}

#[test]
fn strided_operands_are_broadcast_through_the_layout() {
    let base = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let transposed = re_view(&base, base.layout().transpose(0, 1).unwrap()); // [3, 2]
    assert!(!transposed.is_contiguous());
    let other = t(&[10.0, 100.0], [2]);
    let out = transposed.add(&other).unwrap();
    assert_eq!(out.dims(), &[3, 2]);
    assert!(out.is_contiguous());
    // Transposed rows are [1,4], [2,5], [3,6].
    assert_eq!(v(&out), vec![11.0, 104.0, 12.0, 105.0, 13.0, 106.0]);
}

#[test]
fn mismatched_operands_are_structured_errors() {
    let f = t(&[1.0, 2.0], [2]);
    let i = Tensor::from_vec(vec![1i64, 2], [2], &CPU).unwrap();
    assert!(matches!(
        f.add(&i),
        Err(Error::DTypeMismatch {
            op: "add",
            expected: DType::F32,
            got: DType::I64
        })
    ));
    let wide = t(&[1.0, 2.0, 3.0], [3]);
    match f.mul(&wide) {
        Err(Error::ShapeMismatch { op, lhs, rhs }) => {
            assert_eq!(op, "mul");
            assert_eq!(lhs, Shape::from([2]));
            assert_eq!(rhs, Shape::from([3]));
        }
        Err(e) => panic!("expected a ShapeMismatch, got {e}"),
        Ok(_) => panic!("expected a ShapeMismatch, got a tensor"),
    }
}

// ------------------------------------------------------------------
// Unary math
// ------------------------------------------------------------------

#[test]
fn unary_values() {
    let x = t(&[-2.0, -0.5, 0.0, 0.5, 2.0], [5]);
    assert_eq!(v(&x.relu().unwrap()), vec![0.0, 0.0, 0.0, 0.5, 2.0]);
    assert_eq!(v(&x.neg().unwrap()), vec![2.0, 0.5, -0.0, -0.5, -2.0]);
    assert_eq!(v(&x.abs().unwrap()), vec![2.0, 0.5, 0.0, 0.5, 2.0]);
    close(
        &v(&x.exp().unwrap()),
        &[0.135_335_28, 0.606_530_66, 1.0, 1.648_721_3, 7.389_056],
        1e-6,
    );
    close(
        &v(&x.tanh().unwrap()),
        &[-0.964_027_6, -0.462_117_16, 0.0, 0.462_117_16, 0.964_027_6],
        1e-6,
    );
    close(
        &v(&x.sigmoid().unwrap()),
        &[0.119_202_92, 0.377_540_67, 0.5, 0.622_459_33, 0.880_797_1],
        1e-6,
    );

    let p = t(&[1.0, 4.0, 9.0], [3]);
    close(&v(&p.sqrt().unwrap()), &[1.0, 2.0, 3.0], 1e-6);
    close(&v(&p.ln().unwrap()), &[0.0, 1.386_294_4, 2.197_224_6], 1e-6);
}

#[test]
fn gelu_is_exact_not_the_tanh_approximation() {
    let x = t(&[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], [6]);
    // 0.5·x·(1 + erf(x/√2)) evaluated in f64.
    let expected = [
        -0.045_500_264,
        -0.158_655_25,
        0.0,
        0.841_344_8,
        1.954_499_7,
        2.995_950_2,
    ];
    close(&v(&x.gelu().unwrap()), &expected, 1e-6);

    // The tanh approximation `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`
    // deviates from the exact form by ~4e-4 around x = 3 — two orders of
    // magnitude outside the tolerance asserted above — so this test fails
    // loudly if a backend ever swaps the approximation in.
    let tanh_approx = |x: f32| 0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh());
    let worst = v(&x)
        .iter()
        .zip(&expected)
        .map(|(&xi, &e)| (tanh_approx(xi) - e).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst > 1e-4,
        "tanh-approximation deviation was only {worst}"
    );
}

#[test]
fn unary_dtype_rules() {
    let i = Tensor::from_vec(vec![-3i64, 0, 4], [3], &CPU).unwrap();
    // Sign-preserving integer unaries are defined...
    assert_eq!(i.neg().unwrap().to_vec::<i64>().unwrap(), vec![3, 0, -4]);
    assert_eq!(i.abs().unwrap().to_vec::<i64>().unwrap(), vec![3, 0, 4]);
    // ...the float-only ones are loud.
    assert!(matches!(i.exp(), Err(Error::Unsupported { op: "exp", .. })));
    assert!(matches!(
        i.relu(),
        Err(Error::Unsupported { op: "relu", .. })
    ));
    let b = Tensor::from_vec(vec![true], [1], &CPU).unwrap();
    assert!(matches!(b.neg(), Err(Error::Unsupported { op: "neg", .. })));
}

#[test]
fn unary_materializes_strided_inputs() {
    let base = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let transposed = re_view(&base, base.layout().transpose(0, 1).unwrap());
    let out = transposed.neg().unwrap();
    assert_eq!(out.dims(), &[3, 2]);
    assert!(out.is_contiguous());
    assert_eq!(v(&out), vec![-1.0, -4.0, -2.0, -5.0, -3.0, -6.0]);
}

#[test]
fn sign_recip_and_erf_values() {
    let x = t(&[-2.0, -0.5, 0.0, 0.5, 2.0], [5]);
    assert_eq!(v(&x.sign().unwrap()), vec![-1.0, -1.0, 0.0, 1.0, 1.0]);

    // Reciprocals chosen to be exact in binary floating point, so this is an
    // equality and not a tolerance.
    let r = t(&[-2.0, -0.5, 0.25, 4.0], [4]);
    assert_eq!(v(&r.recip().unwrap()), vec![-0.5, -2.0, 4.0, 0.25]);

    // erf at points a high-precision reference was evaluated at.
    close(
        &v(&x.erf().unwrap()),
        &[-0.995_322_3, -0.520_499_9, 0.0, 0.520_499_9, 0.995_322_3],
        1e-6,
    );
    // The identity `gelu(x) = ½·x·(1 + erf(x/√2))` ties the exposed
    // primitive to the activation it was factored out of.
    let scaled = x.mul_scalar(std::f64::consts::FRAC_1_SQRT_2).unwrap();
    let composed = x
        .mul(&scaled.erf().unwrap().add_scalar(1.0).unwrap())
        .unwrap()
        .mul_scalar(0.5)
        .unwrap();
    close(&v(&composed), &v(&x.gelu().unwrap()), 1e-6);
}

#[test]
fn round_breaks_ties_to_even_not_away_from_zero() {
    // Halves in both signs, plus one non-tie for the ordinary case.
    let x = t(&[-2.5, -0.5, 0.5, 1.5, 2.5, 2.7], [6]);
    assert_eq!(v(&x.floor().unwrap()), vec![-3.0, -1.0, 0.0, 1.0, 2.0, 2.0]);
    assert_eq!(v(&x.ceil().unwrap()), vec![-2.0, -0.0, 1.0, 2.0, 3.0, 3.0]);
    // Ties to even: 0.5 -> 0, 1.5 -> 2, 2.5 -> 2. Rust's away-from-zero
    // `f32::round` would answer 1, 2, 3 and fail here.
    assert_eq!(v(&x.round().unwrap()), vec![-2.0, -0.0, 0.0, 2.0, 2.0, 3.0]);
    let away_from_zero: Vec<f32> = v(&x).iter().map(|v| v.round()).collect();
    assert_ne!(v(&x.round().unwrap()), away_from_zero);
}

#[test]
fn sign_and_recip_follow_ieee_at_zero_and_nan() {
    let x = t(&[f32::NAN, 0.0, -0.0, -3.0], [4]);
    let s = v(&x.sign().unwrap());
    // NaN is propagated, not collapsed to one of the three outcomes.
    assert!(s[0].is_nan(), "sign(NaN) was {}", s[0]);
    // Both zeros answer `+0`: the sign of a zero is not its direction.
    for (i, z) in [(1usize, 0.0f32), (2, -0.0)] {
        assert_eq!(s[i], 0.0, "sign({z}) was {}", s[i]);
        assert!(s[i].is_sign_positive(), "sign({z}) was a negative zero");
    }
    assert_eq!(s[3], -1.0);

    // `1/±0` is an infinity, not an error — and it keeps the zero's sign.
    let z = t(&[0.0, -0.0], [2]);
    let r = v(&z.recip().unwrap());
    assert_eq!(r[0], f32::INFINITY);
    assert_eq!(r[1], f32::NEG_INFINITY);
    // …and NaN survives the round trip too.
    assert!(v(&t(&[f32::NAN], [1]).recip().unwrap())[0].is_nan());
}

#[test]
fn clamp_rejects_an_empty_or_nan_interval() {
    let x = t(&[-1.0, 0.5, 2.0], [3]);
    for (min, max) in [(1.0, 0.0), (f64::NAN, 1.0), (0.0, f64::NAN)] {
        let got = x.clamp(min, max);
        assert!(
            matches!(got, Err(Error::InvalidArg { op: "clamp", .. })),
            "clamp({min}, {max}) should be an InvalidArg"
        );
    }

    // An *infinite* bound is not degenerate: it is the one-sided spelling,
    // and it must keep working.
    assert_eq!(
        v(&x.clamp(0.0, f64::INFINITY).unwrap()),
        vec![0.0, 0.5, 2.0]
    );
    assert_eq!(
        v(&x.clamp(f64::NEG_INFINITY, 1.0).unwrap()),
        vec![-1.0, 0.5, 1.0]
    );
    // An empty interval is rejected even when the two bounds are equal only
    // by a hair, so the check is `min > max` and not a tolerance.
    assert!(x.clamp(1.0, 1.0).is_ok());

    // The ordinary two-sided case, for reference.
    assert_eq!(v(&x.clamp(0.0, 1.0).unwrap()), vec![0.0, 0.5, 1.0]);
}

#[test]
fn pow_values_and_dtype_rule() {
    let x = t(&[1.0, 4.0, 9.0], [3]);
    close(&v(&x.pow(0.5).unwrap()), &[1.0, 2.0, 3.0], 1e-6);
    close(&v(&x.pow(2.0).unwrap()), &[1.0, 16.0, 81.0], 1e-5);
    close(&v(&x.pow(-1.0).unwrap()), &[1.0, 0.25, 1.0 / 9.0], 1e-6);
    // `x^0` is 1 everywhere, the `powf` convention.
    assert_eq!(v(&x.pow(0.0).unwrap()), vec![1.0, 1.0, 1.0]);

    // A fractional exponent has no integer meaning, so integers decline
    // rather than invent one.
    let i = Tensor::from_vec(vec![2i64, 3], [2], &CPU).unwrap();
    assert!(matches!(
        i.pow(2.0),
        Err(Error::Unsupported { op: "pow", .. })
    ));
}

// ------------------------------------------------------------------
// Comparisons
// ------------------------------------------------------------------

#[test]
fn comparisons_produce_bool() {
    let a = t(&[1.0, 2.0, 3.0], [3]);
    let b = t(&[3.0, 2.0, 1.0], [3]);
    let out = a.lt(&b).unwrap();
    assert_eq!(out.dtype(), DType::Bool);
    assert_eq!(out.to_vec::<bool>().unwrap(), vec![true, false, false]);
    assert_eq!(
        a.le(&b).unwrap().to_vec::<bool>().unwrap(),
        vec![true, true, false]
    );
    assert_eq!(
        a.gt(&b).unwrap().to_vec::<bool>().unwrap(),
        vec![false, false, true]
    );
    assert_eq!(
        a.ge(&b).unwrap().to_vec::<bool>().unwrap(),
        vec![false, true, true]
    );
    assert_eq!(
        a.eq(&b).unwrap().to_vec::<bool>().unwrap(),
        vec![false, true, false]
    );
    assert_eq!(
        a.ne(&b).unwrap().to_vec::<bool>().unwrap(),
        vec![true, false, true]
    );
}

#[test]
fn comparisons_broadcast_and_stay_untraced() {
    let m = t(&[1.0, 2.0, 3.0, 4.0], [2, 2]);
    let s = t(&[2.5], ());
    let out = m.gt(&s).unwrap();
    assert_eq!(out.dims(), &[2, 2]);
    assert_eq!(
        out.to_vec::<bool>().unwrap(),
        vec![false, false, true, true]
    );
    assert!(out.node().is_none());
    // Bool operands compare fine against each other.
    let x = Tensor::from_vec(vec![true, false], [2], &CPU).unwrap();
    let y = Tensor::from_vec(vec![true, true], [2], &CPU).unwrap();
    assert_eq!(
        x.eq(&y).unwrap().to_vec::<bool>().unwrap(),
        vec![true, false]
    );
    // Mixed dtypes are still an error, not a promotion.
    let i = Tensor::from_vec(vec![1i64, 2], [2], &CPU).unwrap();
    assert!(matches!(
        i.eq(&t(&[1.0, 2.0], [2])),
        Err(Error::DTypeMismatch { op: "eq", .. })
    ));
}

// ------------------------------------------------------------------
// masked_fill / where_cond
// ------------------------------------------------------------------

#[test]
fn masked_fill_replaces_true_positions() {
    let x = t(&[1.0, 2.0, 3.0, 4.0], [2, 2]);
    let mask = Tensor::from_vec(vec![false, true, true, false], [2, 2], &CPU).unwrap();
    let out = x.masked_fill(&mask, -1.0).unwrap();
    assert_eq!(out.dims(), &[2, 2]);
    assert_eq!(v(&out), vec![1.0, -1.0, -1.0, 4.0]);

    // The mask broadcasts (the causal-attention shape).
    let row_mask = Tensor::from_vec(vec![true, false], [2], &CPU).unwrap();
    assert_eq!(
        v(&x.masked_fill(&row_mask, 0.0).unwrap()),
        vec![0.0, 2.0, 0.0, 4.0]
    );

    // A non-bool mask is a dtype error.
    assert!(matches!(
        x.masked_fill(&x, 0.0),
        Err(Error::DTypeMismatch {
            op: "masked_fill",
            expected: DType::Bool,
            ..
        })
    ));
}

#[test]
fn where_cond_selects_per_element() {
    let cond = Tensor::from_vec(vec![true, false, true], [3], &CPU).unwrap();
    let a = t(&[1.0, 2.0, 3.0], [3]);
    let b = t(&[10.0, 20.0, 30.0], [3]);
    assert_eq!(v(&cond.where_cond(&a, &b).unwrap()), vec![1.0, 20.0, 3.0]);

    // All three operands broadcast: [2, 1] cond, [3] values.
    let cond = Tensor::from_vec(vec![true, false], [2, 1], &CPU).unwrap();
    let a = t(&[1.0, 2.0, 3.0], [3]);
    let b = t(&[-1.0], ());
    let out = cond.where_cond(&a, &b).unwrap();
    assert_eq!(out.dims(), &[2, 3]);
    assert_eq!(v(&out), vec![1.0, 2.0, 3.0, -1.0, -1.0, -1.0]);

    // A non-bool condition, and value operands of differing dtypes, are
    // both structured errors.
    assert!(matches!(
        a.where_cond(&a, &b),
        Err(Error::DTypeMismatch { op: "where", .. })
    ));
    let i = Tensor::from_vec(vec![1i64, 2, 3], [3], &CPU).unwrap();
    assert!(matches!(
        cond.where_cond(&a, &i),
        Err(Error::DTypeMismatch { op: "where", .. })
    ));
}

// ------------------------------------------------------------------
// Backward — finite differences against the single `check_grad` harness.
//
// `check_grad` requires `f` to produce a **scalar**, so every case
// scalarizes with the weighted sum `wsum`.
//
// The weighting is load-bearing. A one-element input cannot distinguish a
// correct backward from one that mixes elements up, and an *unweighted*
// `sum_all` hands every element the same cotangent — which cannot tell a
// gradient placed in the right slot from one transposed, reversed, or
// broadcast-summed into the wrong slot. `wsum`'s weights are pairwise
// distinct, so any misplacement shows up as a wrong number.
// ------------------------------------------------------------------

const EPS: f64 = 1e-3;
const TOL: f64 = 1e-3;

/// `Σ w ⊙ x` with pairwise-distinct constant weights: a scalar objective
/// whose gradient w.r.t. `x` is `w` rather than a constant.
fn wsum(x: &Tensor) -> Result<Tensor> {
    let w: Vec<f32> = (0..x.num_elements())
        .map(|i| 0.25 + 0.5 * (i as f32))
        .collect();
    x.mul(&Tensor::from_vec(w, x.dims().to_vec(), &CPU)?)?
        .sum_all()
}

/// The left operand of the binary cases: no element equals its `rhs`
/// partner, so `maximum`/`minimum` are locally smooth (a tie is a kink
/// finite differences cannot see through).
fn lhs() -> Tensor {
    t(&[1.5, -0.75, 2.25, 0.5, -1.25, 3.0], [2, 3])
}

fn rhs() -> Tensor {
    t(&[-0.5, 2.0, 1.25, -2.5, 0.75, -1.5], [2, 3])
}

#[test]
fn grad_binary_arithmetic() {
    type BinaryCase = fn(&[Tensor]) -> Result<Tensor>;
    let cases: [BinaryCase; 6] = [
        |i| wsum(&i[0].add(&i[1])?),
        |i| wsum(&i[0].sub(&i[1])?),
        |i| wsum(&i[0].mul(&i[1])?),
        |i| wsum(&i[0].div(&i[1])?),
        |i| wsum(&i[0].maximum(&i[1])?),
        |i| wsum(&i[0].minimum(&i[1])?),
    ];
    for f in cases {
        check_grad(f, &[lhs(), rhs()], EPS, TOL).unwrap();
    }
}

#[test]
fn grad_binary_broadcast_reduces_through_sum_to() {
    // A rank-0 operand against a rank-1 one: the lhs cotangent has to be
    // summed back down over the padded leading axis.
    check_grad(
        |i: &[Tensor]| i[0].mul(&i[1]),
        &[t(&[2.0], ()), t(&[-3.0], [1])],
        EPS,
        TOL,
    )
    .unwrap();

    // A [3] operand against a [2, 3] one: the rhs cotangent is summed
    // over the *padded* axis only, keeping its per-column placement.
    check_grad(
        |i: &[Tensor]| wsum(&i[0].mul(&i[1])?),
        &[lhs(), t(&[-0.5, 2.0, 1.25], [3])],
        EPS,
        TOL,
    )
    .unwrap();

    // …and a [2, 1] operand, summed over the *existing* size-1 axis, so
    // the two reduction paths in `sum_to` are both exercised.
    check_grad(
        |i: &[Tensor]| wsum(&i[0].div(&i[1])?),
        &[lhs(), t(&[-2.5, 0.75], [2, 1])],
        EPS,
        TOL,
    )
    .unwrap();
}

#[test]
fn grad_scalar_variants() {
    type ScalarCase = fn(&[Tensor]) -> Result<Tensor>;
    let cases: [ScalarCase; 4] = [
        |i| wsum(&i[0].add_scalar(2.0)?),
        |i| wsum(&i[0].sub_scalar(2.0)?),
        |i| wsum(&i[0].mul_scalar(-3.0)?),
        |i| wsum(&i[0].div_scalar(4.0)?),
    ];
    for f in cases {
        check_grad(f, &[lhs()], EPS, TOL).unwrap();
    }

    // `pow` needs its own point: a non-integer exponent is undefined on the
    // negatives `lhs` holds, and `p·xᵖ⁻¹` is best conditioned away from 0.
    check_grad(
        |i: &[Tensor]| wsum(&i[0].pow(1.5)?),
        &[t(&[0.7, 1.3, 2.5], [3])],
        EPS,
        TOL,
    )
    .unwrap();

    // `clamp`'s mask has three regions, and `lhs` straddles [0, 2]: two
    // elements below, two inside, two above. None sits *on* a bound, so
    // ±EPS never crosses one and the closed-interval rule stays visible as
    // a gradient of exactly 1 inside and exactly 0 outside.
    check_grad(
        |i: &[Tensor]| wsum(&i[0].clamp(0.0, 2.0)?),
        &[lhs()],
        EPS,
        TOL,
    )
    .unwrap();
}

#[test]
fn grad_unary_family() {
    // Several points per op, both signs where the domain allows, all far
    // enough from a kink (`relu`/`abs` at 0, the step of a rounding op at an
    // integer or of `round` at a half-integer) that `±EPS` stays on one side.
    type UnaryCase = (fn(&[Tensor]) -> Result<Tensor>, &'static [f32]);
    let cases: [UnaryCase; 15] = [
        (|i| wsum(&i[0].relu()?), &[0.7, -1.3, 2.5]),
        (|i| wsum(&i[0].gelu()?), &[0.7, -1.3, 2.5, -0.2]),
        (|i| wsum(&i[0].exp()?), &[0.3, -1.1, 1.4]),
        (|i| wsum(&i[0].ln()?), &[1.7, 0.4, 3.2]),
        (|i| wsum(&i[0].sqrt()?), &[2.3, 0.6, 4.1]),
        (|i| wsum(&i[0].tanh()?), &[0.4, -1.5, 2.2]),
        (|i| wsum(&i[0].sigmoid()?), &[0.4, -1.5, 2.2]),
        (|i| wsum(&i[0].neg()?), &[0.9, -2.0, 0.1]),
        (|i| wsum(&i[0].abs()?), &[-1.2, 0.8, 2.6]),
        (|i| wsum(&i[0].erf()?), &[0.4, -1.1, 1.6]),
        (|i| wsum(&i[0].recip()?), &[0.7, -1.3, 2.5]),
        // The four piecewise-constant ops. Their gradient is an exact zero
        // away from a step, and the finite difference agrees only because
        // both perturbed evaluations land in the same step — which is what
        // pins the "subgradient is 0" contract rather than skipping it.
        (|i| wsum(&i[0].sign()?), &[0.7, -1.3, 2.5]),
        (|i| wsum(&i[0].floor()?), &[0.3, -1.2, 2.7]),
        (|i| wsum(&i[0].ceil()?), &[0.3, -1.2, 2.7]),
        (|i| wsum(&i[0].round()?), &[0.3, -1.2, 2.7]),
    ];
    for (f, at) in cases {
        check_grad(f, &[t(at, [at.len()])], EPS, TOL).unwrap();
    }
    // GELU's backward has a special case at exactly zero (Φ(0) = 1/2).
    check_grad(|i| i[0].gelu(), &[t(&[0.0], ())], EPS, TOL).unwrap();
}

#[test]
fn grad_masking_ops() {
    // A mixed mask, so one call covers both the kept and the dropped
    // branch and the gradient has to land on the right elements.
    let mask = Tensor::from_vec(vec![false, true, true, false, true, false], [2, 3], &CPU).unwrap();

    let m = mask.clone();
    check_grad(
        move |i: &[Tensor]| wsum(&i[0].masked_fill(&m, 0.0)?),
        &[lhs()],
        EPS,
        TOL,
    )
    .unwrap();

    let c = mask;
    check_grad(
        move |i: &[Tensor]| wsum(&c.where_cond(&i[0], &i[1])?),
        &[lhs(), rhs()],
        EPS,
        TOL,
    )
    .unwrap();
}
