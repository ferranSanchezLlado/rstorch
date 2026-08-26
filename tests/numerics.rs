//! Numerical behavior tests: finite-difference gradients and small independent
//! oracles.
//!
//! The suite covers tied-parameter accumulation, exact identities for
//! activations and reductions, masking and empty-axis policies, and gradients
//! for the less common tensor operations.
//!
//! These are integration tests because they exercise the public API and the
//! finite-difference helper used by downstream code.

use rstorch::nn::{Mode, Param};
use rstorch::testing::check_grad;
use rstorch::{Device, Error, Result, Tensor};

const CPU: Device = Device::Cpu;

fn t(data: &[f32], shape: impl Into<rstorch::Shape>) -> Tensor {
    Tensor::from_vec(data.to_vec(), shape, &CPU).unwrap()
}

fn v(x: &Tensor) -> Vec<f32> {
    x.to_vec::<f32>().unwrap()
}

/// `Σ w ⊙ x` with pairwise-distinct constant weights: a scalar readout whose
/// cotangent is not a constant, so a gradient that lands in the wrong slot is
/// a wrong *number* and not merely a wrong arrangement.
fn wsum(x: &Tensor) -> Result<Tensor> {
    let w: Vec<f32> = (0..x.num_elements())
        .map(|i| 0.25 + 0.5 * (i as f32))
        .collect();
    x.mul(&Tensor::from_vec(w, x.dims().to_vec(), &CPU)?)?
        .sum_all()
}

// ===========================================================================
// 1. Tied weights
// ===========================================================================

/// The tied-weight objective: `w` is used **twice** — once to project `x` up
/// (`x @ wᵀ`) and once to project the result back down (`h @ w`), the shape
/// an input embedding tied to an output head has.
///
/// Written against plain tensors so `check_grad` can drive it; the `Param`
/// spelling below runs the identical arithmetic through `Param::get`.
fn tied(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let hidden = x.matmul(&w.transpose(-2, -1)?)?;
    wsum(&hidden.matmul(w)?)
}

/// Finite differences over an objective that uses one tensor on two paths.
///
/// This is the numeric half of the accumulation claim: the true derivative of
/// `f(w)` w.r.t. `w` is the sum of both paths' contributions, so a backward
/// that overwrote one with the other would disagree with the central
/// difference. Nothing here trusts the engine's bookkeeping.
#[test]
fn tied_weight_gradient_matches_finite_differences() {
    let x = t(&[0.5, -1.25, 2.0, 0.75, -0.5, 1.5], [2, 3]);
    let w = t(&[0.3, -0.7, 1.1, 0.9, 0.2, -1.3], [2, 3]);
    check_grad(|i| tied(&i[0], &i[1]), &[x, w], 1e-3, 1e-3).unwrap();
}

/// The same objective through a [`Param`], plus the decomposition that makes
/// "accumulated, not overwritten" a statement about numbers.
///
/// `detach` cuts one path at a time. The two single-path gradients are both
/// non-zero and **not equal to each other**, so:
///
/// * if the engine overwrote, the tied gradient would equal one of them;
/// * if it accumulated, it equals their sum — which is what is asserted.
#[test]
fn a_param_used_twice_accumulates_both_cotangents() {
    let x = t(&[0.5, -1.25, 2.0, 0.75, -0.5, 1.5], [2, 3]);
    let w = Param::new(t(&[0.3, -0.7, 1.1, 0.9, 0.2, -1.3], [2, 3]));

    // Both paths live: one `Param`, one cached leaf, two uses.
    let leaf = w.get(Mode::TRAIN);
    let tied_grad = v(&tied(&x, &leaf)
        .unwrap()
        .backward()
        .unwrap()
        .wrt(&w)
        .unwrap());

    // Path A only: the second use is detached, so only `x @ wᵀ` carries a
    // gradient.
    let leaf = w.get(Mode::TRAIN);
    let frozen = leaf.detach();
    let only_a = x
        .matmul(&leaf.transpose(-2, -1).unwrap())
        .unwrap()
        .matmul(&frozen)
        .unwrap();
    let grad_a = v(&wsum(&only_a).unwrap().backward().unwrap().wrt(&w).unwrap());

    // Path B only: the first use is detached.
    let leaf = w.get(Mode::TRAIN);
    let frozen = leaf.detach();
    let only_b = x
        .matmul(&frozen.transpose(-2, -1).unwrap())
        .unwrap()
        .matmul(&leaf)
        .unwrap();
    let grad_b = v(&wsum(&only_b).unwrap().backward().unwrap().wrt(&w).unwrap());

    // The premise of the test: the two paths really do contribute different,
    // non-trivial gradients. Without this the sum below proves nothing.
    assert!(grad_a.iter().any(|g| g.abs() > 1e-3), "{grad_a:?}");
    assert!(grad_b.iter().any(|g| g.abs() > 1e-3), "{grad_b:?}");
    assert!(
        grad_a
            .iter()
            .zip(&grad_b)
            .any(|(a, b)| (a - b).abs() > 1e-3),
        "the two paths must differ for the accumulation to be observable"
    );

    for (i, (&got, (&a, &b))) in tied_grad.iter().zip(grad_a.iter().zip(&grad_b)).enumerate() {
        let sum = a + b;
        assert!(
            (got - sum).abs() <= 1e-4 * sum.abs().max(1.0),
            "element {i}: tied gradient {got} is not path A {a} + path B {b} = {sum}"
        );
        // Spelt out: it is not either path alone.
        assert!(
            (got - a).abs() > 1e-6 || (a - b).abs() < 1e-6,
            "element {i}: tied gradient equals path A alone — overwritten, not accumulated"
        );
    }
}

/// Tying survives a `Param::set` (the optimizer step): the rebuilt leaf keeps
/// the parameter's identity, so the *next* graph still accumulates under the
/// same key.
#[test]
fn tying_still_accumulates_after_a_param_set() {
    let x = t(&[0.5, -1.25, 2.0, 0.75, -0.5, 1.5], [2, 3]);
    let mut w = Param::new(t(&[0.3, -0.7, 1.1, 0.9, 0.2, -1.3], [2, 3]));

    let before = v(&tied(&x, &w.get(Mode::TRAIN))
        .unwrap()
        .backward()
        .unwrap()
        .wrt(&w)
        .unwrap());

    w.set(t(&[0.3, -0.7, 1.1, 0.9, 0.2, -1.3], [2, 3])).unwrap();
    let after = v(&tied(&x, &w.get(Mode::TRAIN))
        .unwrap()
        .backward()
        .unwrap()
        .wrt(&w)
        .unwrap());

    assert_eq!(before.len(), after.len());
    for (i, (&a, &b)) in before.iter().zip(&after).enumerate() {
        assert!((a - b).abs() < 1e-5, "element {i}: {a} vs {b}");
    }
}

// ===========================================================================
// 2. Analytic identities — exact, and independent of any other library
// ===========================================================================

/// The standard normal CDF Φ at a few points, to 16 digits.
///
/// These are **tabulated mathematical constants**, not reference vectors:
/// Φ(z) = ½(1 + erf(z/√2)) is in every statistics table and is not a property
/// of any implementation. They are what lets this suite tell rstorch's
/// *exact* (erf) GELU from the tanh approximation without `PyTorch`: the two
/// disagree by ~1.5e-4 at z = 1 and ~5e-4 at z = 2, while an honest erf
/// agrees to f32 precision.
const PHI: [(f64, f64); 5] = [
    (-1.0, 0.158_655_253_931_457_05),
    (-0.5, 0.308_537_538_725_986_9),
    (0.5, 0.691_462_461_274_013_1),
    (1.0, 0.841_344_746_068_542_9),
    (2.0, 0.977_249_868_051_820_8),
];

#[test]
fn analytic_exact_gelu_and_derivative() {
    for (z, phi) in PHI {
        let got = t(&[z as f32], ()).gelu().unwrap().item().unwrap();
        let want = z * phi;
        assert!(
            (got - want).abs() < 1e-6,
            "gelu({z}) = {got}, expected {want} (= {z} · Φ({z}))"
        );
    }

    assert_eq!(t(&[0.0], ()).gelu().unwrap().item().unwrap(), 0.0);

    let x = t(&[0.0], ()).traced().unwrap();
    let grads = x.gelu().unwrap().backward().unwrap();
    let derivative = grads.wrt_input(&x).unwrap().item().unwrap();
    assert!(
        (derivative - 0.5).abs() < 1e-6,
        "gelu'(0) = {derivative}, expected 0.5"
    );
}

#[test]
fn analytic_variance_uses_correction_one() {
    // Sample variance of 1, 2, 3: mean 2, squared deviations 1 + 0 + 1 = 2,
    // divided by n − 1 = 2, so exactly 1. With correction = 0 it would be ⅔ —
    // the two answers are far apart, which is the point of the case.
    let x = t(&[1.0, 2.0, 3.0], [3]);
    assert!((x.var_all().unwrap().item().unwrap() - 1.0).abs() < 1e-6);
    assert!((x.std_all().unwrap().item().unwrap() - 1.0).abs() < 1e-6);

    // Sample variance of 1, 2, 3, 4: deviations ±1.5, ±0.5 → 5 / 3.
    let x = t(&[1.0, 2.0, 3.0, 4.0], [4]);
    let want = 5.0 / 3.0;
    assert!((x.var_all().unwrap().item().unwrap() - want).abs() < 1e-6);

    // Axis-wise, on two rows with different spreads.
    let x = t(&[1.0, 2.0, 3.0, 2.0, 4.0, 6.0], [2, 3]);
    let got = v(&x.var(1).unwrap());
    assert!(
        (got[0] - 1.0).abs() < 1e-6 && (got[1] - 4.0).abs() < 1e-6,
        "{got:?}"
    );
}

#[test]
fn analytic_correction_one_rejects_a_single_sample() {
    // n − 1 = 0: there is no unbiased variance, and the policy is to say so
    // rather than return NaN.
    let x = t(&[3.5], [1]);
    assert!(matches!(x.var_all(), Err(Error::InvalidArg { .. })));
    assert!(matches!(x.std_all(), Err(Error::InvalidArg { .. })));
}

#[test]
fn analytic_softmax_of_a_uniform_row_is_one_over_n() {
    for n in [1usize, 3, 7] {
        let x = Tensor::zeros([1, n], rstorch::DType::F32, &CPU).unwrap();
        let p = v(&x.softmax(-1).unwrap());
        let l = v(&x.log_softmax(-1).unwrap());
        let want_p = 1.0 / n as f32;
        let want_l = -(n as f32).ln();
        for (i, (&pi, &li)) in p.iter().zip(&l).enumerate() {
            assert!((pi - want_p).abs() < 1e-6, "n={n} i={i}: {pi} vs {want_p}");
            assert!((li - want_l).abs() < 1e-5, "n={n} i={i}: {li} vs {want_l}");
        }
    }
}

#[test]
fn analytic_log_softmax_is_shift_invariant_and_sums_through_exp_to_one() {
    let x = t(&[0.3, -1.2, 2.0, 0.7], [1, 4]);
    let shifted = x.add_scalar(100.0).unwrap();
    let a = v(&x.log_softmax(-1).unwrap());
    let b = v(&shifted.log_softmax(-1).unwrap());
    for (i, (&ai, &bi)) in a.iter().zip(&b).enumerate() {
        assert!((ai - bi).abs() < 1e-5, "element {i}: {ai} vs {bi}");
    }
    let total: f32 = v(&x.softmax(-1).unwrap()).iter().sum();
    assert!((total - 1.0).abs() < 1e-6, "{total}");
}

#[test]
fn analytic_masked_softmax_ignores_the_masked_columns() {
    // Masking a column must give exactly the softmax of the surviving
    // columns — an identity, checkable without a reference.
    let logits = t(&[0.3, -1.2, 2.0, 0.7], [1, 4]);
    let mask = Tensor::from_vec(vec![false, true, false, true], [1, 4], &CPU).unwrap();
    let masked = logits
        .masked_fill(&mask, f64::NEG_INFINITY)
        .unwrap()
        .softmax(-1)
        .unwrap();
    let got = v(&masked);

    let survivors = t(&[0.3, 2.0], [1, 2]);
    let want = v(&survivors.softmax(-1).unwrap());

    assert_eq!(got[1], 0.0, "a masked column must be exactly zero: {got:?}");
    assert_eq!(got[3], 0.0, "a masked column must be exactly zero: {got:?}");
    assert!((got[0] - want[0]).abs() < 1e-6, "{got:?} vs {want:?}");
    assert!((got[2] - want[1]).abs() < 1e-6, "{got:?} vs {want:?}");
}

#[test]
fn analytic_a_fully_masked_row_is_zeros_not_nan() {
    // rstorch's documented divergence from PyTorch: exp(-inf − -inf) is NaN,
    // and NaN is not an answer. `p = 0` / `log p = -inf` is.
    let logits = t(&[0.3, -1.2, 2.0, 0.7, 1.1, -0.4], [2, 3]);
    let mask = Tensor::from_vec(vec![false, false, false, true, true, true], [2, 3], &CPU).unwrap();
    let masked = logits.masked_fill(&mask, f64::NEG_INFINITY).unwrap();

    let p = v(&masked.softmax(-1).unwrap());
    assert!(p.iter().all(|e| e.is_finite()), "{p:?}");
    assert_eq!(&p[3..], &[0.0, 0.0, 0.0], "fully-masked row must be zeros");
    assert!(((p[0] + p[1] + p[2]) - 1.0).abs() < 1e-6, "{p:?}");

    let l = v(&masked.log_softmax(-1).unwrap());
    assert!(
        l[3..].iter().all(|e| *e == f32::NEG_INFINITY),
        "fully-masked row of log_softmax must be -inf: {l:?}"
    );
    assert!(l[..3].iter().all(|e| e.is_finite()), "{l:?}");
}

#[test]
fn analytic_empty_reduction_policy() {
    // `sum` has an identity, so it returns it; nothing else does, so nothing
    // else guesses. (`src/tensor/ops/reduce.rs`.)
    let empty = Tensor::zeros([2, 0], rstorch::DType::F32, &CPU).unwrap();

    let summed = empty.sum(1).unwrap();
    assert_eq!(summed.dims(), &[2]);
    assert_eq!(v(&summed), vec![0.0, 0.0]);
    assert_eq!(empty.sum_all().unwrap().item().unwrap(), 0.0);

    macro_rules! refuses {
        ($($call:expr),+ $(,)?) => {
            $(assert!(
                matches!($call, Err(Error::InvalidArg { .. })),
                concat!(stringify!($call), " must refuse an empty axis"),
            );)+
        };
    }
    refuses!(
        empty.mean(1),
        empty.max(1),
        empty.min(1),
        empty.var(1),
        empty.std(1),
        empty.argmax(1),
        empty.argmin(1),
        empty.softmax(1),
        empty.log_softmax(1),
        empty.mean_all(),
        empty.max_all(),
        empty.min_all(),
    );
}

// ===========================================================================
// 3. Finite-difference coverage
// ===========================================================================

/// A [`wsum`](wsum)-style scalar readout for a `Vec<Tensor>`-producing op
/// (`split`/`chunk`/`sort`/`topk`), so every output slice contributes
/// distinctly and a gradient landing in the wrong slice is a wrong number.
fn wsum_all(xs: &[Tensor]) -> Result<Tensor> {
    let mut total = Tensor::zeros((), rstorch::DType::F32, &CPU)?;
    for x in xs {
        total = total.add(&wsum(x)?)?;
    }
    Ok(total)
}

#[test]
fn grad_clamp_zero_outside_the_range_one_inside() {
    let x = t(&[-2.0, 0.3, 0.9, 5.0], [4]);
    check_grad(|xs| xs[0].clamp(0.0, 1.0)?.sum_all(), &[x], 1e-3, 1e-2).unwrap();
    // The boundary check itself, exactly: outside receives zero, inside one.
    let xt = t(&[-2.0, 0.3, 0.9, 5.0], [4]).traced().unwrap();
    let grads = xt
        .clamp(0.0, 1.0)
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    assert_eq!(v(&grads.wrt_input(&xt).unwrap()), vec![0.0, 1.0, 1.0, 0.0]);
}

#[test]
fn grad_pow_matches_finite_differences() {
    let x = t(&[0.5, 1.5, 2.5], [3]);
    check_grad(
        |xs| xs[0].pow(3.0)?.mul(&xs[0])?.sum_all(),
        &[x],
        1e-3,
        1e-2,
    )
    .unwrap();
}

#[test]
fn grad_recip_matches_finite_differences() {
    let x = t(&[0.5, 1.5, -2.5], [3]);
    check_grad(|xs| wsum(&xs[0].recip()?), &[x], 1e-4, 1e-2).unwrap();
}

#[test]
fn grad_erf_matches_finite_differences() {
    let x = t(&[-1.0, 0.0, 0.5, 1.5], [4]);
    check_grad(|xs| wsum(&xs[0].erf()?), &[x], 1e-3, 1e-2).unwrap();
}

#[test]
fn non_differentiable_unaries_report_zero_gradient() {
    // sign/floor/ceil/round are piecewise constant: the decision (§ R22) is
    // an explicit zero cotangent, not a refusal to trace.
    for name in ["sign", "floor", "ceil", "round"] {
        let xt = t(&[-1.7, 0.0, 2.3], [3]).traced().unwrap();
        let out = match name {
            "sign" => xt.sign(),
            "floor" => xt.floor(),
            "ceil" => xt.ceil(),
            "round" => xt.round(),
            _ => unreachable!(),
        }
        .unwrap();
        let grads = wsum(&out).unwrap().backward().unwrap();
        assert_eq!(
            v(&grads.wrt_input(&xt).unwrap()),
            vec![0.0, 0.0, 0.0],
            "{name} must report an exact zero gradient"
        );
    }
}

#[test]
fn grad_prod_matches_finite_differences_away_from_zero() {
    // Away from zero, `prod`'s documented `out / x` formula is exact.
    let x = t(&[1.5, 2.0, -0.5, 3.0], [4]);
    check_grad(|xs| xs[0].prod(0), &[x], 1e-3, 1e-2).unwrap();
}

#[test]
fn grad_norm_matches_finite_differences() {
    let x = t(&[3.0, -4.0, 1.0, 2.0], [4]);
    check_grad(|xs| xs[0].norm(0, 2.0), &[x], 1e-3, 1e-2).unwrap();
}

#[test]
fn grad_cumsum_matches_finite_differences() {
    let x = t(&[1.0, 2.0, 3.0, 4.0], [4]);
    check_grad(|xs| wsum(&xs[0].cumsum(0)?), &[x], 1e-3, 1e-2).unwrap();
}

#[test]
fn grad_tril_triu_matches_finite_differences() {
    let x = t(
        &(0..9).map(|i| 1.0 + i as f32 * 0.3).collect::<Vec<_>>(),
        [3, 3],
    );
    check_grad(
        |xs| wsum(&xs[0].tril(0)?),
        std::slice::from_ref(&x),
        1e-3,
        1e-2,
    )
    .unwrap();
    check_grad(|xs| wsum(&xs[0].triu(1)?), &[x], 1e-3, 1e-2).unwrap();
}

#[test]
fn grad_repeat_sums_the_tiled_copies() {
    let x = t(&[1.0, 2.0], [2]);
    check_grad(
        |xs| wsum(&xs[0].repeat(&[3])?),
        std::slice::from_ref(&x),
        1e-3,
        1e-2,
    )
    .unwrap();
    // Exact check: d/dx sum(repeat(x, 3)) = 3 for every element.
    let xt = x.traced().unwrap();
    let grads = xt
        .repeat(&[3])
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    assert_eq!(v(&grads.wrt_input(&xt).unwrap()), vec![3.0, 3.0]);
}

#[test]
fn grad_flip_matches_finite_differences() {
    let x = t(&[1.0, 2.0, 3.0, 4.0], [4]);
    check_grad(|xs| wsum(&xs[0].flip(0)?), &[x], 1e-3, 1e-2).unwrap();
}

#[test]
fn grad_take_along_dim_matches_finite_differences() {
    let x = t(&[1.0, 2.0, 3.0, 4.0], [2, 2]);
    let idx = Tensor::from_vec(vec![1i64, 0], [2, 1], &CPU).unwrap();
    check_grad(|xs| wsum(&xs[0].take_along_dim(1, &idx)?), &[x], 1e-3, 1e-2).unwrap();
}

#[test]
fn grad_sort_and_topk_values_match_finite_differences() {
    let x = t(&[3.0, 1.2, 4.0, 1.7, 5.0], [5]);
    check_grad(
        |xs| wsum_all(&[xs[0].sort(0, false)?.0]),
        std::slice::from_ref(&x),
        1e-3,
        1e-2,
    )
    .unwrap();
    check_grad(
        |xs| wsum_all(&[xs[0].topk(3, 0, true)?.0]),
        &[x],
        1e-3,
        1e-2,
    )
    .unwrap();
}

#[test]
fn grad_split_and_chunk_route_the_cotangent_to_the_right_slice() {
    let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0], [5]);
    check_grad(
        |xs| wsum_all(&xs[0].split(&[2, 3], 0)?),
        std::slice::from_ref(&x),
        1e-3,
        1e-2,
    )
    .unwrap();
    check_grad(|xs| wsum_all(&xs[0].chunk(2, 0)?), &[x], 1e-3, 1e-2).unwrap();
}

#[test]
fn grad_silu_matches_finite_differences() {
    let x = t(&[-1.0, 0.0, 0.5, 2.0], [4]);
    check_grad(|xs| wsum(&xs[0].silu()?), &[x], 1e-3, 1e-2).unwrap();
}

#[test]
fn grad_leaky_relu_matches_finite_differences() {
    let x = t(&[-1.0, -0.1, 0.1, 2.0], [4]);
    check_grad(|xs| wsum(&xs[0].leaky_relu(0.1)?), &[x], 1e-3, 1e-2).unwrap();
}

#[test]
fn grad_softplus_matches_finite_differences() {
    let x = t(&[-3.0, -0.5, 0.5, 3.0], [4]);
    check_grad(|xs| wsum(&xs[0].softplus()?), &[x], 1e-3, 1e-2).unwrap();
}

#[test]
fn grad_elu_matches_finite_differences() {
    let x = t(&[-2.0, -0.3, 0.3, 2.0], [4]);
    check_grad(|xs| wsum(&xs[0].elu(1.0)?), &[x], 1e-3, 1e-2).unwrap();
}

#[test]
fn one_hot_is_not_differentiable() {
    // `one_hot` takes an integral input, so it never enters the graph — the
    // same contract as `argmax`/`argmin`.
    let ids = Tensor::from_vec(vec![2i64, 0], [2], &CPU).unwrap();
    let oh = ids.one_hot(3).unwrap();
    assert!(oh.backward().is_err(), "one_hot output must not be traced");
}

#[test]
fn analytic_silu_and_softplus_and_elu_match_hand_computed_values() {
    // silu(0) = 0; softplus(0) = ln 2; elu(-1, alpha=1) = e^-1 - 1.
    let x = t(&[0.0], [1]);
    assert!((v(&x.silu().unwrap())[0]).abs() < 1e-6);
    assert!((v(&x.softplus().unwrap())[0] - 2f32.ln()).abs() < 1e-6);
    let neg_one = t(&[-1.0], [1]);
    let want = (-1.0f32).exp() - 1.0;
    assert!((v(&neg_one.elu(1.0).unwrap())[0] - want).abs() < 1e-6);
}
