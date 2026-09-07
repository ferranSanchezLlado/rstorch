//! The reduction family: values against independent references, the
//! `correction` conventions, keepdim spellings, and gradients.

use super::*;
use crate::device::Device;
use crate::dtype::DType;
use crate::shape::Shape;
use crate::testing::check_grad;

const CPU: Device = Device::Cpu;

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

fn t(data: &[f32], shape: impl Into<Shape>) -> Tensor {
    Tensor::from_vec(data.to_vec(), shape, &CPU).unwrap()
}

fn v(x: &Tensor) -> Vec<f32> {
    x.to_vec::<f32>().unwrap()
}

fn close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length: {a:?} vs {b:?}");
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= tol, "{a:?} vs {b:?}");
    }
}

/// Re-view `x` through `layout` — the way this file's tests build
/// non-contiguous inputs without the public view ops.
fn re_view(x: &Tensor, layout: Layout) -> Tensor {
    Tensor::from_parts(x.storage().clone(), layout)
}

// ------------------------------------------------------------------
// Forward: the three spellings
// ------------------------------------------------------------------

#[test]
fn sum_mean_max_min_over_each_axis() {
    // [[1, 2, 3], [4, 5, 6]]
    let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);

    assert_eq!(v(&x.sum(0).unwrap()), vec![5.0, 7.0, 9.0]);
    assert_eq!(v(&x.sum(1).unwrap()), vec![6.0, 15.0]);
    assert_eq!(v(&x.mean(0).unwrap()), vec![2.5, 3.5, 4.5]);
    assert_eq!(v(&x.mean(1).unwrap()), vec![2.0, 5.0]);
    assert_eq!(v(&x.max(0).unwrap()), vec![4.0, 5.0, 6.0]);
    assert_eq!(v(&x.max(1).unwrap()), vec![3.0, 6.0]);
    assert_eq!(v(&x.min(0).unwrap()), vec![1.0, 2.0, 3.0]);
    assert_eq!(v(&x.min(1).unwrap()), vec![1.0, 4.0]);

    // The reduced axis is gone.
    assert_eq!(x.sum(0).unwrap().dims(), &[3]);
    assert_eq!(x.sum(1).unwrap().dims(), &[2]);
}

#[test]
fn keepdim_holds_the_axis_at_one() {
    let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    for (kd, dropped) in [
        (x.sum_keepdim(1).unwrap(), x.sum(1).unwrap()),
        (x.mean_keepdim(1).unwrap(), x.mean(1).unwrap()),
        (x.max_keepdim(1).unwrap(), x.max(1).unwrap()),
        (x.min_keepdim(1).unwrap(), x.min(1).unwrap()),
    ] {
        assert_eq!(kd.dims(), &[2, 1]);
        assert_eq!(v(&kd), v(&dropped));
    }
    assert_eq!(x.sum_keepdim(0).unwrap().dims(), &[1, 3]);
    // The point of keepdim: the result broadcasts back against the input.
    assert_eq!(x.div(&x.sum_keepdim(-1).unwrap()).unwrap().dims(), &[2, 3]);
}

#[test]
fn all_variants_reduce_to_a_rank_zero_scalar() {
    let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    for (out, expected) in [
        (x.sum_all().unwrap(), 21.0),
        (x.mean_all().unwrap(), 3.5),
        (x.max_all().unwrap(), 6.0),
        (x.min_all().unwrap(), 1.0),
    ] {
        assert_eq!(out.rank(), 0);
        assert_eq!(out.num_elements(), 1);
        assert_eq!(out.item().unwrap(), expected);
    }

    // A rank-0 input is already its own reduction.
    let s = t(&[7.0], ());
    assert_eq!(s.sum_all().unwrap().item().unwrap(), 7.0);
    assert_eq!(s.max_all().unwrap().item().unwrap(), 7.0);
    assert_eq!(s.mean_all().unwrap().item().unwrap(), 7.0);

    // Rank 3, so the fold runs more than twice.
    let x = t(&(0..24).map(|i| i as f32).collect::<Vec<_>>(), [2, 3, 4]);
    assert_eq!(x.sum_all().unwrap().item().unwrap(), 276.0);
    assert_eq!(x.max_all().unwrap().item().unwrap(), 23.0);
    assert_eq!(x.min_all().unwrap().item().unwrap(), 0.0);
    assert_eq!(x.mean_all().unwrap().item().unwrap(), 11.5);
}

#[test]
fn reduced_all_axis_folds_narrow_only_after_the_complete_reduction() {
    let width = 65_520;
    let mut f16_values = vec![half::f16::ONE; width];
    f16_values.extend(vec![half::f16::NEG_ONE; width]);
    let f16 = Tensor::from_vec(f16_values, [2, width], &CPU).unwrap();
    assert_eq!(f16.sum_all().unwrap().item().unwrap(), 0.0);
    assert_eq!(f16.mean_all().unwrap().item().unwrap(), 0.0);

    // Build [257, 2], then view it as a non-contiguous [2, 257]. The two
    // logical rows sum to 257 and -256. Narrowing those partials to BF16
    // loses the unit before the final add; one F32 fold preserves it.
    let mut bf16_values = Vec::with_capacity(514);
    for index in 0..257 {
        bf16_values.push(half::bf16::ONE);
        bf16_values.push(if index < 256 {
            half::bf16::NEG_ONE
        } else {
            half::bf16::ZERO
        });
    }
    let base = Tensor::from_vec(bf16_values, [257, 2], &CPU).unwrap();
    let bf16 = re_view(&base, base.layout().transpose(0, 1).unwrap());
    assert_eq!(bf16.sum_all().unwrap().item().unwrap(), 1.0);
    assert!(bf16.mean_all().unwrap().item().unwrap() > 0.0);
}

#[test]
fn negative_axes_count_from_the_end() {
    let x = t(&(0..24).map(|i| i as f32).collect::<Vec<_>>(), [2, 3, 4]);
    assert_eq!(v(&x.sum(-1).unwrap()), v(&x.sum(2).unwrap()));
    assert_eq!(v(&x.mean(-3).unwrap()), v(&x.mean(0).unwrap()));
    assert_eq!(
        v(&x.max_keepdim(-2).unwrap()),
        v(&x.max_keepdim(1).unwrap())
    );
    assert_eq!(
        x.argmin(-1).unwrap().to_vec::<i64>().unwrap(),
        x.argmin(2).unwrap().to_vec::<i64>().unwrap()
    );
}

#[test]
fn rank_one_reductions_produce_scalars() {
    let x = t(&[2.0, 5.0, 1.0, 4.0], [4]);
    assert_eq!(x.sum(0).unwrap().rank(), 0);
    assert_eq!(x.sum(0).unwrap().item().unwrap(), 12.0);
    assert_eq!(x.max(0).unwrap().item().unwrap(), 5.0);
    assert_eq!(x.min(0).unwrap().item().unwrap(), 1.0);
    assert_eq!(x.mean(0).unwrap().item().unwrap(), 3.0);
    // keepdim keeps the rank instead.
    assert_eq!(x.sum_keepdim(0).unwrap().dims(), &[1]);
}

#[test]
fn reductions_walk_strided_and_broadcast_views() {
    let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);

    // Transposed view: [[1,4],[2,5],[3,6]].
    let tr = re_view(&x, x.layout().transpose(0, 1).unwrap());
    assert!(!tr.is_contiguous());
    assert_eq!(v(&tr.sum(1).unwrap()), vec![5.0, 7.0, 9.0]);
    assert_eq!(v(&tr.max(1).unwrap()), vec![4.0, 5.0, 6.0]);

    // Narrowed view (non-zero offset).
    let mid = re_view(&x, x.layout().narrow(1, 1, 2).unwrap());
    assert_eq!(v(&mid.sum(1).unwrap()), vec![5.0, 11.0]);

    // Broadcast (stride-0) view: the repeats count.
    let row = t(&[1.0, 2.0, 3.0], [1, 3]);
    let b = re_view(
        &row,
        row.layout().broadcast_to(&Shape::from([4, 3])).unwrap(),
    );
    assert_eq!(v(&b.sum(0).unwrap()), vec![4.0, 8.0, 12.0]);
    assert_eq!(v(&b.mean(0).unwrap()), vec![1.0, 2.0, 3.0]);
}

#[test]
fn integer_reductions_stay_integer() {
    let x = Tensor::from_vec(vec![1i64, 2, 3, 4, 5, 6], [2, 3], &CPU).unwrap();
    let s = x.sum(1).unwrap();
    assert_eq!(s.dtype(), DType::I64);
    assert_eq!(s.to_vec::<i64>().unwrap(), vec![6, 15]);
    assert_eq!(x.max_all().unwrap().to_scalar::<i64>().unwrap(), 6);
    assert_eq!(x.argmax(1).unwrap().to_vec::<i64>().unwrap(), vec![2, 2]);
}

// ------------------------------------------------------------------
// Forward: prod / norm
// ------------------------------------------------------------------

#[test]
fn prod_multiplies_each_line_and_the_whole_tensor() {
    let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);

    // Row products (1·2·3, 4·5·6), then column products.
    let rows = x.prod(1).unwrap();
    assert_eq!(rows.dims(), &[2]);
    assert_eq!(v(&rows), vec![6.0, 120.0]);
    let cols = x.prod(0).unwrap();
    assert_eq!(cols.dims(), &[3]);
    assert_eq!(v(&cols), vec![4.0, 10.0, 18.0]);

    // keepdim holds the reduced axis at 1 with the same numbers, and the
    // negative axis is the same axis.
    let k = x.prod_keepdim(1).unwrap();
    assert_eq!(k.dims(), &[2, 1]);
    assert_eq!(v(&k), vec![6.0, 120.0]);
    let k0 = x.prod_keepdim(-2).unwrap();
    assert_eq!(k0.dims(), &[1, 3]);
    assert_eq!(v(&k0), vec![4.0, 10.0, 18.0]);

    // `prod_all` is 6! as a rank-0 scalar, and it agrees with reducing one
    // axis after the other.
    let all = x.prod_all().unwrap();
    assert_eq!(all.rank(), 0);
    assert_eq!(all.item().unwrap(), 720.0);
    assert_eq!(x.prod(1).unwrap().prod(0).unwrap().item().unwrap(), 720.0);

    // A single zero anywhere annihilates the line it is on, and only that
    // line; negative signs multiply through.
    let z = t(&[1.0, 0.0, 3.0, -4.0, 5.0, -6.0], [2, 3]);
    assert_eq!(v(&z.prod(1).unwrap()), vec![0.0, 120.0]);
    assert_eq!(z.prod_all().unwrap().item().unwrap(), 0.0);
    assert_eq!(
        t(&[-2.0, 3.0, -4.0], [3]).prod(0).unwrap().item().unwrap(),
        24.0
    );

    // Integers stay integers.
    let i = Tensor::from_vec(vec![2i64, 3, 4], [3], &CPU).unwrap();
    let p = i.prod(0).unwrap();
    assert_eq!(p.dtype(), DType::I64);
    assert_eq!(p.to_scalar::<i64>().unwrap(), 24);
    assert_eq!(i.prod_all().unwrap().to_scalar::<i64>().unwrap(), 24);
}

#[test]
fn norm_is_the_p_norm_and_refuses_a_degenerate_p() {
    let x = t(&[3.0, 4.0], [2]);
    // The 3-4-5 triangle, and the L1 norm of the same vector.
    assert!((x.norm(0, 2.0).unwrap().item().unwrap() - 5.0).abs() < 1e-6);
    assert!((x.norm(0, 1.0).unwrap().item().unwrap() - 7.0).abs() < 1e-6);
    // Any other `p` routes through `pow`: (3³ + 4³)^(1/3) = 91^(1/3).
    let p3 = x.norm(0, 3.0).unwrap().item().unwrap();
    assert!((p3 - 91.0f64.cbrt()).abs() < 1e-5, "norm(0, 3.0) = {p3}");

    // `|x|` before the power, so the signs are absorbed on all three routes.
    let s = t(&[-3.0, 4.0], [2]);
    assert!((s.norm(0, 2.0).unwrap().item().unwrap() - 5.0).abs() < 1e-6);
    assert!((s.norm(0, 1.0).unwrap().item().unwrap() - 7.0).abs() < 1e-6);
    assert!((s.norm(0, 3.0).unwrap().item().unwrap() - 91.0f64.cbrt()).abs() < 1e-5);

    // Per line, and the keepdim spelling.
    let m = t(&[3.0, 4.0, 6.0, 8.0], [2, 2]);
    close(&v(&m.norm(1, 2.0).unwrap()), &[5.0, 10.0], 1e-5);
    let k = m.norm_keepdim(1, 2.0).unwrap();
    assert_eq!(k.dims(), &[2, 1]);
    close(&v(&k), &[5.0, 10.0], 1e-5);

    // `p = ∞` is *not* a spelling of the max-norm. Unguarded it sails
    // through — |x|^∞ ∈ {0, 1, ∞}, the outer exponent is 1/∞ = 0, and
    // `powf(_, 0) = 1` — and returns all ones with a zero gradient, which
    // is exactly the silent wrong answer this rejection exists to prevent.
    for p in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 0.0, -1.0] {
        assert!(
            matches!(x.norm(0, p), Err(Error::InvalidArg { op: "norm", .. })),
            "norm(0, {p}) should be an InvalidArg"
        );
        assert!(
            matches!(
                x.norm_keepdim(0, p),
                Err(Error::InvalidArg {
                    op: "norm_keepdim",
                    ..
                })
            ),
            "norm_keepdim(0, {p}) should be an InvalidArg"
        );
    }
}

// ------------------------------------------------------------------
// Forward: var / std
// ------------------------------------------------------------------

#[test]
fn var_and_std_use_correction_one() {
    // mean 2.5; Σ(x−x̄)² = 2.25 + 0.25 + 0.25 + 2.25 = 5; /(4−1).
    let x = t(&[1.0, 2.0, 3.0, 4.0], [4]);
    let expected = 5.0f32 / 3.0;
    close(
        &[x.var(0).unwrap().item().unwrap() as f32],
        &[expected],
        1e-6,
    );
    close(
        &[x.std(0).unwrap().item().unwrap() as f32],
        &[expected.sqrt()],
        1e-6,
    );
    close(
        &[x.var_all().unwrap().item().unwrap() as f32],
        &[expected],
        1e-6,
    );
    close(
        &[x.std_all().unwrap().item().unwrap() as f32],
        &[expected.sqrt()],
        1e-6,
    );

    // Per row, and the keepdim spelling.
    let x = t(&[1.0, 2.0, 3.0, 10.0, 12.0, 14.0], [2, 3]);
    close(&v(&x.var(1).unwrap()), &[1.0, 4.0], 1e-6);
    close(&v(&x.std(1).unwrap()), &[1.0, 2.0], 1e-6);
    assert_eq!(x.var_keepdim(1).unwrap().dims(), &[2, 1]);
    assert_eq!(x.std_keepdim(1).unwrap().dims(), &[2, 1]);
    close(&v(&x.var_keepdim(1).unwrap()), &[1.0, 4.0], 1e-6);

    // A constant line has zero variance (and the sqrt of it).
    let c = t(&[3.0, 3.0, 3.0], [3]);
    assert_eq!(c.var(0).unwrap().item().unwrap(), 0.0);
    assert_eq!(c.std(0).unwrap().item().unwrap(), 0.0);
}

#[test]
fn var_needs_two_samples_and_a_float_dtype() {
    let one = t(&[1.0, 2.0, 3.0], [3, 1]);
    // correction=1 over an axis of length 1: loud, not NaN.
    for r in [one.var(1), one.std(1), one.var_keepdim(1)] {
        assert!(matches!(r, Err(Error::InvalidArg { .. })), "expected err");
    }
    let single = t(&[1.0], [1]);
    assert!(matches!(single.var_all(), Err(Error::InvalidArg { .. })));

    let ints = Tensor::from_vec(vec![1i64, 2, 3], [3], &CPU).unwrap();
    assert!(matches!(
        ints.var(0),
        Err(Error::Unsupported { op: "var", .. })
    ));
    assert!(matches!(
        ints.std_all(),
        Err(Error::Unsupported { op: "std_all", .. })
    ));
}

// ------------------------------------------------------------------
// Forward: softmax / log_softmax
// ------------------------------------------------------------------

#[test]
fn softmax_normalizes_each_line() {
    let x = t(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], [2, 3]);
    let s = x.softmax(-1).unwrap();
    assert_eq!(s.dims(), &[2, 3]);
    // Hand-computed: exp shifted by the row max.
    let e = [(-2.0f32).exp(), (-1.0f32).exp(), 1.0];
    let z: f32 = e.iter().sum();
    close(&v(&s)[..3], &[e[0] / z, e[1] / z, e[2] / z], 1e-6);
    close(&v(&s)[3..], &[1.0 / 3.0; 3], 1e-6);
    // Every row sums to one.
    close(&v(&s.sum(-1).unwrap()), &[1.0, 1.0], 1e-6);

    // Softmax over the *leading* axis normalizes columns instead.
    let s0 = x.softmax(0).unwrap();
    close(&v(&s0.sum(0).unwrap()), &[1.0, 1.0, 1.0], 1e-6);
}

#[test]
fn fused_softmax_scope_and_numerics_match_the_composed_path() {
    let x = t(&[1000.0, 1001.0, 1002.0, -3.0, 0.5, 7.0], [2, 3]);
    let expected = composed_softmax("softmax", &x, 1).unwrap();
    let fused = try_fused_softmax(&x, 1).unwrap().unwrap();
    assert!(fused.is_contiguous());
    assert_eq!(fused.dims(), x.dims());
    close(&v(&fused), &v(&expected), 1e-6);
    close(&v(&x.softmax(-1).unwrap()), &v(&expected), 1e-6);

    let x64 = Tensor::from_vec(vec![1000.0f64, 1001.0, 1002.0], [1, 3], &CPU).unwrap();
    let expected64 = composed_softmax("softmax", &x64, 1)
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    let fused64 = try_fused_softmax(&x64, 1)
        .unwrap()
        .unwrap()
        .to_vec::<f64>()
        .unwrap();
    for (got, expected) in fused64.iter().zip(expected64) {
        assert!((got - expected).abs() <= 1e-15, "{fused64:?}");
    }
}

#[test]
fn non_last_axis_falls_back_and_reduced_last_axis_is_fused() {
    let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    assert!(try_fused_softmax(&x, 0).unwrap().is_none());
    close(
        &v(&x.softmax(0).unwrap()),
        &v(&composed_softmax("softmax", &x, 0).unwrap()),
        1e-6,
    );

    let f16 = Tensor::from_vec(
        [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .map(half::f16::from_f32)
            .to_vec(),
        [2, 3],
        &CPU,
    )
    .unwrap();
    let fused = try_fused_softmax(&f16, 1).unwrap().unwrap();
    assert_eq!(
        f16.softmax(-1).unwrap().to_vec::<half::f16>().unwrap(),
        fused.to_vec::<half::f16>().unwrap()
    );

    let bf16 = Tensor::from_vec(
        [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .map(half::bf16::from_f32)
            .to_vec(),
        [2, 3],
        &CPU,
    )
    .unwrap();
    let fused = try_fused_softmax(&bf16, 1).unwrap().unwrap();
    assert_eq!(
        bf16.softmax(-1).unwrap().to_vec::<half::bf16>().unwrap(),
        fused.to_vec::<half::bf16>().unwrap()
    );
}

#[test]
fn fused_softmax_reads_a_strided_input_and_returns_contiguous_output() {
    let base = t(&[1.0, 10.0, 2.0, 20.0, 3.0, 30.0], [3, 2]);
    let x = base.transpose(0, 1).unwrap();
    assert!(!x.is_contiguous());
    let expected = composed_softmax("softmax", &x, 1).unwrap();
    let got = x.softmax(-1).unwrap();
    assert!(got.is_contiguous());
    assert_eq!(got.dims(), &[2, 3]);
    close(&v(&got), &v(&expected), 1e-6);
}

#[test]
fn softmax_is_stable_and_shift_invariant() {
    // The naive exp of these overflows to +inf; the max shift does not.
    let big = t(&[1000.0, 1000.0, 1000.0], [3]);
    close(&v(&big.softmax(0).unwrap()), &[1.0 / 3.0; 3], 1e-6);

    let x = t(&[-3.0, 0.5, 2.0, 7.0], [4]);
    let shifted = x.add_scalar(500.0).unwrap();
    close(
        &v(&x.softmax(0).unwrap()),
        &v(&shifted.softmax(0).unwrap()),
        1e-6,
    );
    // Very negative logits underflow to exactly zero, not to NaN.
    let tiny = t(&[-1000.0, 0.0], [2]);
    close(&v(&tiny.softmax(0).unwrap()), &[0.0, 1.0], 1e-6);
}

#[test]
fn softmax_is_mask_aware() {
    // Row 0 is fully masked, row 1 keeps its first two logits.
    let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let mask = Tensor::from_vec(vec![true, true, true, false, false, true], [2, 3], &CPU).unwrap();
    let masked = x.masked_fill(&mask, f64::NEG_INFINITY).unwrap();

    let s = masked.softmax(-1).unwrap();
    // The fully-masked row is zeros, not NaN.
    assert_eq!(&v(&s)[..3], &[0.0, 0.0, 0.0]);
    // The partially-masked row renormalizes over its live entries
    // (logits 4 and 5, shifted by the row max of 5).
    let e = [(-1.0f32).exp(), 1.0f32];
    let z: f32 = e.iter().sum();
    close(&v(&s)[3..], &[e[0] / z, e[1] / z, 0.0], 1e-6);

    // log_softmax answers -inf where softmax answers 0.
    let l = masked.log_softmax(-1).unwrap();
    assert!(v(&l)[..3].iter().all(|p| *p == f32::NEG_INFINITY));
    close(&v(&l)[3..5], &[(e[0] / z).ln(), (e[1] / z).ln()], 1e-6);
    assert_eq!(v(&l)[5], f32::NEG_INFINITY);
}

#[test]
fn fused_softmax_backward_is_exact_and_masked_rows_stay_zero() {
    let x = t(
        &[
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.3,
            -1.2,
            2.0,
        ],
        [2, 3],
    )
    .traced()
    .unwrap();
    let w = t(&[0.25, 0.75, -0.5, 1.0, -2.0, 0.5], [2, 3]);
    let y = x.softmax(-1).unwrap();
    assert!(y.node().is_some());
    assert_eq!(&v(&y)[..3], &[0.0, 0.0, 0.0]);

    let expected = softmax_backward(&w, &y.detach(), 1).unwrap();
    let loss = y.mul(&w).unwrap().sum_all().unwrap();
    let grads = loss.backward().unwrap();
    let got = grads.wrt_input(&x).unwrap();
    close(&v(&got), &v(&expected), 1e-6);
    assert!(v(&got)[..3].iter().all(|value| *value == 0.0));
    assert!(v(&got).iter().all(|value| !value.is_nan()));

    let plain = t(&[1.0, 2.0, 3.0], [3]).softmax(-1).unwrap();
    assert!(plain.node().is_none());
}

#[test]
fn log_softmax_is_the_log_of_softmax() {
    let x = t(&[1.0, 2.0, 3.0, -1.0, 0.0, 4.0], [2, 3]);
    let expected: Vec<f32> = v(&x.softmax(-1).unwrap()).iter().map(|p| p.ln()).collect();
    close(&v(&x.log_softmax(-1).unwrap()), &expected, 1e-6);

    // Stable in the tail, where ln(softmax) would have flushed to -inf.
    let far = t(&[0.0, -200.0], [2]);
    let l = v(&far.log_softmax(0).unwrap());
    close(&l, &[0.0, -200.0], 1e-4);
    assert!(l[1].is_finite());
}

#[test]
fn softmax_requires_a_float_dtype() {
    let ints = Tensor::from_vec(vec![1i64, 2, 3], [3], &CPU).unwrap();
    assert!(matches!(
        ints.softmax(0),
        Err(Error::Unsupported { op: "softmax", .. })
    ));
    assert!(matches!(
        ints.log_softmax(0),
        Err(Error::Unsupported {
            op: "log_softmax",
            ..
        })
    ));
}

// ------------------------------------------------------------------
// Forward: argmax / argmin
// ------------------------------------------------------------------

#[test]
fn argmax_and_argmin_report_positions() {
    let x = t(&[1.0, 5.0, 3.0, 4.0, 2.0, 0.0], [2, 3]);
    let am = x.argmax(1).unwrap();
    assert_eq!(am.dtype(), DType::I64);
    assert_eq!(am.dims(), &[2]);
    assert_eq!(am.to_vec::<i64>().unwrap(), vec![1, 0]);
    assert_eq!(x.argmin(1).unwrap().to_vec::<i64>().unwrap(), vec![0, 2]);
    assert_eq!(x.argmax(0).unwrap().to_vec::<i64>().unwrap(), vec![1, 0, 0]);

    // keepdim keeps the rank (the shape `gather` wants).
    let kd = x.argmax_keepdim(1).unwrap();
    assert_eq!(kd.dims(), &[2, 1]);
    assert_eq!(kd.to_vec::<i64>().unwrap(), vec![1, 0]);
    assert_eq!(x.argmin_keepdim(-1).unwrap().dims(), &[2, 1]);

    // First occurrence wins a tie.
    let tied = t(&[2.0, 2.0, 1.0, 1.0], [4]);
    assert_eq!(tied.argmax(0).unwrap().to_vec::<i64>().unwrap(), vec![0]);
    assert_eq!(tied.argmin(0).unwrap().to_vec::<i64>().unwrap(), vec![2]);
}

#[test]
fn argmax_points_at_the_value_max_reports() {
    let x = t(&[0.5, -2.0, 7.0, 3.0, 3.5, -1.0], [2, 3]);
    let idx = x.argmax(1).unwrap().to_vec::<i64>().unwrap();
    let peaks = v(&x.max(1).unwrap());
    let rows = v(&x);
    for (row, (&i, &peak)) in idx.iter().zip(peaks.iter()).enumerate() {
        assert_eq!(rows[row * 3 + i as usize], peak);
    }
}

// ------------------------------------------------------------------
// The empty-reduction policy
// ------------------------------------------------------------------

#[test]
fn sum_over_an_empty_axis_is_the_identity() {
    let empty = Tensor::zeros([2, 0], DType::F32, &CPU).unwrap();
    let s = empty.sum(1).unwrap();
    assert_eq!(s.dims(), &[2]);
    assert_eq!(v(&s), vec![0.0, 0.0]);
    assert_eq!(empty.sum_keepdim(1).unwrap().dims(), &[2, 1]);
    assert_eq!(empty.sum_all().unwrap().item().unwrap(), 0.0);

    // ...including when the *other* axis is the empty one.
    let empty = Tensor::zeros([0, 3], DType::F32, &CPU).unwrap();
    assert_eq!(v(&empty.sum(0).unwrap()), vec![0.0, 0.0, 0.0]);
    assert_eq!(empty.sum_all().unwrap().item().unwrap(), 0.0);
    // Reducing the *non-empty* axis of an empty tensor stays empty.
    assert_eq!(empty.sum(1).unwrap().dims(), &[0]);
}

#[test]
fn every_other_reduction_refuses_an_empty_axis() {
    let empty = Tensor::zeros([2, 0], DType::F32, &CPU).unwrap();
    let cases: Vec<(&str, Result<Tensor>)> = vec![
        ("mean", empty.mean(1)),
        ("mean_keepdim", empty.mean_keepdim(1)),
        ("max", empty.max(1)),
        ("min", empty.min(1)),
        ("max_keepdim", empty.max_keepdim(1)),
        ("var", empty.var(1)),
        ("std", empty.std(1)),
        ("softmax", empty.softmax(1)),
        ("log_softmax", empty.log_softmax(1)),
        ("argmax", empty.argmax(1)),
        ("argmin_keepdim", empty.argmin_keepdim(1)),
        ("mean_all", empty.mean_all()),
        ("max_all", empty.max_all()),
        ("min_all", empty.min_all()),
        ("var_all", empty.var_all()),
    ];
    for (name, result) in cases {
        match result {
            Err(Error::InvalidArg { op, .. }) => assert_eq!(op, name),
            other => panic!("{name}: expected InvalidArg, got {:?}", other.is_ok()),
        }
    }
}

// ------------------------------------------------------------------
// Loud failures
// ------------------------------------------------------------------

#[test]
fn out_of_range_axes_name_the_op() {
    let x = t(&[1.0, 2.0], [2]);
    assert!(matches!(
        x.sum(1),
        Err(Error::InvalidAxis {
            op: "sum",
            axis: 1,
            rank: 1
        })
    ));
    assert!(matches!(
        x.mean_keepdim(-2),
        Err(Error::InvalidAxis {
            op: "mean_keepdim",
            ..
        })
    ));
    assert!(matches!(
        x.softmax(3),
        Err(Error::InvalidAxis { op: "softmax", .. })
    ));
    assert!(matches!(
        x.argmax(-9),
        Err(Error::InvalidAxis { op: "argmax", .. })
    ));
    // A rank-0 tensor has no axis to reduce at all.
    assert!(matches!(
        t(&[1.0], ()).sum(0),
        Err(Error::InvalidAxis {
            op: "sum",
            rank: 0,
            ..
        })
    ));
}

#[test]
fn bool_reductions_are_unsupported_and_named_after_the_caller() {
    let b = Tensor::from_vec(vec![true, false, true, true], [2, 2], &CPU).unwrap();
    assert!(matches!(
        b.sum(0),
        Err(Error::Unsupported { op: "sum", .. })
    ));
    assert!(matches!(
        b.max_keepdim(1),
        Err(Error::Unsupported {
            op: "max_keepdim",
            ..
        })
    ));
    assert!(matches!(
        b.sum_all(),
        Err(Error::Unsupported { op: "sum_all", .. })
    ));
    assert!(matches!(
        b.argmax(1),
        Err(Error::Unsupported { op: "argmax", .. })
    ));
    // The route for counting a mask is an explicit cast.
    assert_eq!(
        b.to_dtype(DType::F32)
            .unwrap()
            .sum_all()
            .unwrap()
            .item()
            .unwrap(),
        3.0
    );
}

// ------------------------------------------------------------------
// Backward: the value-level pieces that do not need the engine
// ------------------------------------------------------------------
//
// These tests exercise the *helpers* the backward closures are built
// from, which are ordinary value-level functions; the finite-difference
// cases that check the closures end to end are further down.

#[test]
fn spread_broadcasts_the_cotangent_back_over_the_axis() {
    // Cotangent of `sum(axis=1)` on a [2, 3] input.
    let g = t(&[1.0, 2.0], [2]);
    let out = spread(&g, 1, false, &[2, 3], None).unwrap();
    assert_eq!(out.dims(), &[2, 3]);
    assert_eq!(v(&out), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

    // The keepdim spelling receives an already-shaped cotangent.
    let g = t(&[1.0, 2.0], [2, 1]);
    let out = spread(&g, 1, true, &[2, 3], None).unwrap();
    assert_eq!(v(&out), vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

    // `mean` scales by 1/n on the way.
    let g = t(&[1.0, 2.0], [2]);
    let out = spread(&g, 1, false, &[2, 3], Some(1.0 / 3.0)).unwrap();
    close(
        &v(&out),
        &[
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            2.0 / 3.0,
            2.0 / 3.0,
            2.0 / 3.0,
        ],
        1e-6,
    );

    // An empty axis takes the cotangent to an empty tensor, not an error.
    let g = t(&[1.0, 2.0], [2]);
    let out = spread(&g, 1, false, &[2, 0], None).unwrap();
    assert_eq!(out.dims(), &[2, 0]);
    assert!(v(&out).is_empty());
}

#[test]
fn extremum_routing_splits_ties_evenly() {
    // Row 0 has a unique max, row 1 has two tied maxima.
    let x = t(&[1.0, 5.0, 3.0, 4.0, 4.0, 2.0], [2, 3]);
    let g = t(&[10.0, 6.0], [2]);
    let out = x.max(1).unwrap();
    let routed = route_to_extrema(&g, &x, &out, 1, false).unwrap();
    assert_eq!(routed.dims(), &[2, 3]);
    assert_eq!(v(&routed), vec![0.0, 10.0, 0.0, 3.0, 3.0, 0.0]);

    // Minima route the same way.
    let out = x.min(1).unwrap();
    let routed = route_to_extrema(&g, &x, &out, 1, false).unwrap();
    assert_eq!(v(&routed), vec![10.0, 0.0, 0.0, 0.0, 0.0, 6.0]);

    // ...and the keepdim spelling takes a keepdim cotangent.
    let g = t(&[10.0, 6.0], [2, 1]);
    let out = x.max_keepdim(1).unwrap();
    let routed = route_to_extrema(&g, &x, &out, 1, true).unwrap();
    assert_eq!(v(&routed), vec![0.0, 10.0, 0.0, 3.0, 3.0, 0.0]);
}

/// Pins the NaN decision documented on `route_to_extrema`: the
/// forward propagates NaN, so the backward does too — the *whole* line
/// goes NaN, and the lines beside it are untouched.
#[test]
fn a_nan_line_propagates_nan_through_the_max_backward() {
    // Row 0 is clean, row 1 holds a NaN, row 2 is all NaN (the 0/0 case
    // Nothing compares equal to the NaN extremum).
    let x = t(
        &[
            1.0,
            5.0,
            3.0,
            4.0,
            f32::NAN,
            2.0,
            f32::NAN,
            f32::NAN,
            f32::NAN,
        ],
        [3, 3],
    );
    for reduced in [x.max(1).unwrap(), x.min(1).unwrap()] {
        // The forward is NaN on both poisoned rows to begin with.
        let forward = v(&reduced);
        assert!(forward[0].is_finite(), "{forward:?}");
        assert!(forward[1].is_nan() && forward[2].is_nan(), "{forward:?}");

        let g = t(&[10.0, 6.0, 7.0], [3]);
        let routed = v(&route_to_extrema(&g, &x, &reduced, 1, false).unwrap());
        // Row 0 is unaffected by its neighbours' NaNs.
        assert!(routed[..3].iter().all(|e| e.is_finite()), "{routed:?}");
        assert!((routed[..3].iter().sum::<f32>() - 10.0).abs() < 1e-6);
        // Rows 1 and 2 are NaN everywhere, not just at the NaN element,
        // and not zero.
        assert!(routed[3..].iter().all(|e| e.is_nan()), "{routed:?}");
    }
}

/// The same decision seen from the public surface: a NaN input poisons
/// the gradient the engine hands back, rather than vanishing into a zero.
#[test]
fn a_nan_max_poisons_the_gradient_end_to_end() {
    let x = t(&[1.0, f32::NAN, 3.0], [3]).traced().unwrap();
    let grads = x.max_all().unwrap().backward().unwrap();
    let g = grads.wrt_input(&x).unwrap().to_vec::<f32>().unwrap();
    assert!(g.iter().all(|e| e.is_nan()), "{g:?}");
}

// ------------------------------------------------------------------
// Backward: finite differences against the single `check_grad` harness.
// ------------------------------------------------------------------

const EPS: f64 = 1e-3;
const TOL: f64 = 1e-4;

/// A fixed non-constant weighting, so a reduction whose output is
/// constant (softmax rows sum to 1) still has a non-trivial scalar
/// objective.
fn weights(dims: &[usize]) -> Tensor {
    let n: usize = dims.iter().product();
    let data: Vec<f32> = (0..n).map(|i| 0.25 + 0.5 * (i as f32)).collect();
    Tensor::from_vec(data, dims.to_vec(), &CPU).unwrap()
}

/// Finite-difference one single-input objective at the file's fixed
/// `EPS`/`TOL`.
fn fd(f: impl Fn(&[Tensor]) -> Result<Tensor>, x: &Tensor) {
    check_grad(f, std::slice::from_ref(x), EPS, TOL).unwrap();
}

#[test]
fn grad_sum_and_mean() {
    let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let w = weights(&[2]);
    fd(|xs| xs[0].sum(1)?.mul(&w)?.sum_all(), &x);
    fd(|xs| xs[0].mean(1)?.mul(&w)?.sum_all(), &x);
    fd(|xs| xs[0].sum_all(), &x);
    fd(|xs| xs[0].mean_all(), &x);

    let wk = weights(&[2, 1]);
    fd(|xs| xs[0].sum_keepdim(1)?.mul(&wk)?.sum_all(), &x);
    fd(|xs| xs[0].mean_keepdim(-1)?.mul(&wk)?.sum_all(), &x);
}

#[test]
fn grad_max_and_min_route_to_the_winners() {
    // Distinct values: the cotangent goes to exactly one element per line
    // (finite differences agree only away from ties).
    let x = t(&[1.0, 5.0, 3.0, 4.0, 0.5, 2.0], [2, 3]);
    let w = weights(&[2]);
    fd(|xs| xs[0].max(1)?.mul(&w)?.sum_all(), &x);
    fd(|xs| xs[0].min(1)?.mul(&w)?.sum_all(), &x);
    fd(|xs| xs[0].max_all(), &x);
    fd(|xs| xs[0].min_all(), &x);

    let wk = weights(&[2, 1]);
    fd(|xs| xs[0].max_keepdim(1)?.mul(&wk)?.sum_all(), &x);
}

#[test]
fn grad_var_and_std() {
    let x = t(&[1.0, 2.0, 4.0, 8.0, 3.0, 5.0], [2, 3]);
    let w = weights(&[2]);
    fd(|xs| xs[0].var(1)?.mul(&w)?.sum_all(), &x);
    fd(|xs| xs[0].std(1)?.mul(&w)?.sum_all(), &x);
    fd(|xs| xs[0].var_all(), &x);
    fd(|xs| xs[0].std_all(), &x);
}

#[test]
fn grad_softmax_and_log_softmax() {
    let x = t(&[0.3, -1.2, 2.0, 0.7, 1.1, -0.4], [2, 3]);
    let w = weights(&[2, 3]);
    fd(|xs| xs[0].softmax(-1)?.mul(&w)?.sum_all(), &x);
    fd(|xs| xs[0].log_softmax(-1)?.mul(&w)?.sum_all(), &x);
    // Along the leading axis too.
    fd(|xs| xs[0].softmax(0)?.mul(&w)?.sum_all(), &x);
}

#[test]
fn grad_prod_matches_finite_differences_away_from_zero() {
    // No element is zero, so a central difference is meaningful and the
    // leave-one-out product is finite and non-degenerate in every slot.
    let x = t(&[1.5, -2.0, 0.5, 3.0, -1.25, 2.5], [2, 3]);
    let w = weights(&[2]);
    fd(|xs| xs[0].prod(1)?.mul(&w)?.sum_all(), &x);
    let w0 = weights(&[3]);
    fd(|xs| xs[0].prod(0)?.mul(&w0)?.sum_all(), &x);
    fd(|xs| xs[0].prod_all(), &x);

    let wk = weights(&[2, 1]);
    fd(|xs| xs[0].prod_keepdim(1)?.mul(&wk)?.sum_all(), &x);
}

/// `∂∏/∂xₖ = ∏_{j≠k} xⱼ`, which is finite everywhere — a product is
/// multilinear — including at a zero, where it is generally *not* zero.
/// Computing it as `∏/xₖ` is `0/0` exactly there, so the leave-one-out
/// product has to be built directly. Finite differences cannot police this:
/// the wrong answer was `NaN`, and `NaN` is not a number a central
/// difference disagrees with. Hence values.
#[test]
fn grad_prod_builds_the_leave_one_out_product_at_a_zero() {
    /// The gradient of the (scalarized) output of `f` w.r.t. `x`, flattened.
    fn grad(x: &Tensor, f: impl Fn(&Tensor) -> Result<Tensor>) -> Vec<f32> {
        let traced = x.traced().unwrap();
        let out = f(&traced).unwrap();
        let grads = out.sum_all().unwrap().backward().unwrap();
        v(&grads.wrt_input(&traced).unwrap())
    }

    // No zero: every slot receives the product of the others — 3·4, 2·4, 2·3.
    assert_eq!(
        grad(&t(&[2.0, 3.0, 4.0], [3]), |x| x.prod(0)),
        vec![12.0, 8.0, 6.0]
    );
    // Exactly one zero: the zero slot alone is sensitive, and it receives
    // the product of the others (2·4 = 8). Every other slot is flat, because
    // its own leave-one-out product still contains the zero.
    assert_eq!(
        grad(&t(&[2.0, 0.0, 4.0], [3]), |x| x.prod(0)),
        vec![0.0, 8.0, 0.0]
    );
    // Two zeros: no leave-one-out product escapes them, so the whole
    // gradient is zero.
    assert_eq!(
        grad(&t(&[0.0, 0.0, 4.0], [3]), |x| x.prod(0)),
        vec![0.0, 0.0, 0.0]
    );
    // Per line, and neither line's zero count leaks into the other: row 0
    // has none, row 1 has one.
    assert_eq!(
        grad(&t(&[2.0, 3.0, 0.0, 5.0], [2, 2]), |x| x.prod(1)),
        vec![3.0, 2.0, 5.0, 0.0]
    );
    // The keepdim spelling goes through the same backward.
    assert_eq!(
        grad(&t(&[2.0, 0.0], [1, 2]), |x| x.prod_keepdim(1)),
        vec![0.0, 2.0]
    );
}

#[test]
fn grad_norm_flows_through_its_composition() {
    // Every element away from zero: `|x|` has a kink there and the p-norm
    // inherits it, so finite differences need to stay off it.
    let x = t(&[1.5, -2.0, 0.5, 3.0, -1.25, 2.5], [2, 3]);
    let w = weights(&[2]);
    // p = 2 (mul + sum + sqrt) and p = 1 (abs + sum): the two special-cased
    // routes that never touch `pow`.
    fd(|xs| xs[0].norm(1, 2.0)?.mul(&w)?.sum_all(), &x);
    fd(|xs| xs[0].norm(1, 1.0)?.mul(&w)?.sum_all(), &x);
    // …and a `p` that does, twice: `|x|^p` and then the outer `^(1/p)`.
    fd(|xs| xs[0].norm(1, 3.0)?.mul(&w)?.sum_all(), &x);

    let w0 = weights(&[3]);
    fd(|xs| xs[0].norm(0, 2.0)?.mul(&w0)?.sum_all(), &x);
    let wk = weights(&[2, 1]);
    fd(|xs| xs[0].norm_keepdim(-1, 2.0)?.mul(&wk)?.sum_all(), &x);
}

#[test]
fn grad_flows_through_a_reduction_chain() {
    // The shape a normalization layer has: subtract the mean, divide by
    // the standard deviation, reduce to a scalar.
    let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 7.0], [2, 3]);
    let w = weights(&[2, 3]);
    fd(
        |xs| {
            let centered = xs[0].sub(&xs[0].mean_keepdim(-1)?)?;
            let scaled = centered.div(&xs[0].std_keepdim(-1)?)?;
            scaled.mul(&w)?.sum_all()
        },
        &x,
    );
}
