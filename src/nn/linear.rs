//! [`Linear`] — the fully connected affine layer.
//!
//! The weight is stored `[out_features, in_features]`, `PyTorch`'s orientation,
//! and the forward pass transposes it *as a view*
//! (`x.matmul(&w.transpose(-2, -1)?)`): `matmul` consumes strided views, so the
//! transposed weight is never materialized. Storing `[in, out]` instead would
//! avoid the transpose but make every checkpoint incompatible with the
//! conventional layout, which is the worse trade.

use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::nn::{Forward, Mode, Param};
use crate::rng::Rng;
use crate::tensor::Tensor;

/// The affine map `y = x·Wᵀ + b`.
///
/// Parameter paths are part of the persistence contract: `weight` with shape
/// `[out_features, in_features]` and, when present, `bias` with shape
/// `[out_features]`.
///
/// # Shapes
///
/// The input's trailing axis must be `in_features`; every leading axis is a
/// batch axis carried through unchanged (`[B, I]` → `[B, O]`,
/// `[B, T, I]` → `[B, T, O]`). At least one batch axis is required: a rank-1
/// `[I]` input is rejected rather than promoted, following
/// [`matmul`](crate::Tensor::matmul)'s "no implicit vector promotion" rule —
/// spell the single sample `[1, I]`.
///
/// # Initialization
///
/// The weight is drawn from the Kaiming (He) uniform distribution for a `ReLU`
/// nonlinearity — `U(-√(6/fan_in), √(6/fan_in))`, `fan_in = in_features` — and
/// the bias starts at zero. Constructors always produce
/// [`F32`](crate::DType::F32) parameters; convert afterwards with
/// [`ModuleExt::to_dtype`](crate::nn::ModuleExt::to_dtype).
///
/// This is *not* bug-compatible with `PyTorch`'s `nn.Linear`, whose default is
/// `kaiming_uniform_(a=√5)` — a bound of `1/√fan_in`, some 2.4× smaller — with
/// a uniformly drawn bias. The textbook He bound is the better default for
/// `ReLU` stacks; a script ported from `PyTorch` that depends on the exact initial
/// distribution should load a checkpoint rather than rely on either default.
///
/// ```
/// use rstorch::nn::{Forward, Linear, Mode};
/// use rstorch::{DType, Device, Rng, Tensor};
///
/// let dev = Device::Cpu;
/// let mut rng = Rng::seed(0);
/// let mut fc = Linear::new(3, 2, &dev, &mut rng)?;
/// let x = Tensor::zeros([4, 3], DType::F32, &dev)?;
/// assert_eq!(fc.forward(&x, Mode::EVAL)?.dims(), &[4, 2]);
/// # Ok::<(), rstorch::Error>(())
/// ```
#[derive(rstorch::Module)]
pub struct Linear {
    /// `[out_features, in_features]`, path `weight`.
    weight: Param,
    /// `[out_features]`, path `bias`; absent for a bias-free layer.
    bias: Option<Param>,
}

impl Linear {
    /// A layer mapping `in_features` → `out_features`, with a bias, drawing
    /// `in_features * out_features` samples from `rng` (see the
    /// [type docs](Linear#initialization) for the distribution).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArg`] (`op: "Linear::new"`) if either dimension is zero
    /// — a zero `in_features` has no Kaiming bound, and a layer with no
    /// outputs is a construction mistake rather than a degenerate case worth
    /// supporting — or if the weight's element count overflows `usize`.
    pub fn new(
        in_features: usize,
        out_features: usize,
        device: &Device,
        rng: &mut Rng,
    ) -> Result<Linear> {
        const OP: &str = "Linear::new";
        if in_features == 0 || out_features == 0 {
            return Err(Error::InvalidArg {
                op: OP,
                msg: format!(
                    "in_features and out_features must be non-zero \
                     (got {in_features} and {out_features})"
                ),
            });
        }
        let count = in_features
            .checked_mul(out_features)
            .ok_or_else(|| Error::InvalidArg {
                op: OP,
                msg: format!("{in_features} * {out_features} weights overflow usize"),
            })?;

        // Kaiming-uniform for ReLU: √(6 / fan_in). Drawn on the host in
        // row-major order (as `Tensor::rand` does) so a given seed reproduces
        // the same weights on every backend.
        let bound = (6.0 / in_features as f64).sqrt();
        let values: Vec<f32> = (0..count)
            .map(|_| rng.uniform(-bound, bound) as f32)
            .collect();

        Ok(Linear {
            weight: Param::new(Tensor::from_vec(
                values,
                [out_features, in_features],
                device,
            )?),
            bias: Some(Param::new(Tensor::zeros(
                [out_features],
                DType::F32,
                device,
            )?)),
        })
    }

    /// [`new`](Linear::new) with the weight drawn by `init` instead of the
    /// hard-coded Kaiming-uniform scheme. `init` receives the weight's shape
    /// (`[out_features, in_features]`) and `device`, and must return a
    /// tensor of exactly that shape — the natural way to plug in
    /// [`nn::init`](crate::nn::init)'s named initializers:
    ///
    /// ```
    /// # use rstorch::nn::{self, Linear};
    /// # use rstorch::{DType, Device, Rng};
    /// # fn main() -> rstorch::Result<()> {
    /// let dev = Device::Cpu;
    /// let mut rng = Rng::seed(0);
    /// let fc = Linear::with_init(4, 8, &dev, |shape, device| {
    ///     nn::init::xavier_uniform(shape.to_vec(), 1.0, DType::F32, device, &mut rng)
    /// })?;
    /// assert_eq!((fc.in_features(), fc.out_features()), (4, 8));
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// The bias is zero-initialized, as in [`new`](Linear::new), but in the
    /// weight's own dtype and on the weight's own device — an `init` closure
    /// is free to return an `F64` or off-device weight, and a bias that
    /// disagreed with it would make every `forward` fail on the add. Drop it
    /// afterward with [`without_bias`](Linear::without_bias) if unwanted.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArg`] (`op: "Linear::with_init"`) under the same
    /// zero-dimension/overflow conditions as [`new`](Linear::new), whatever
    /// `init` itself returns, or [`Error::ShapeMismatch`] if `init`'s output
    /// is not exactly `[out_features, in_features]`.
    pub fn with_init(
        in_features: usize,
        out_features: usize,
        device: &Device,
        init: impl FnOnce(&[usize], &Device) -> Result<Tensor>,
    ) -> Result<Linear> {
        const OP: &str = "Linear::with_init";
        if in_features == 0 || out_features == 0 {
            return Err(Error::invalid_arg(
                OP,
                format!(
                    "in_features and out_features must be non-zero \
                     (got {in_features} and {out_features})"
                ),
            ));
        }
        in_features.checked_mul(out_features).ok_or_else(|| {
            Error::invalid_arg(
                OP,
                format!("{in_features} * {out_features} weights overflow usize"),
            )
        })?;

        let expected = [out_features, in_features];
        let weight = init(&expected, device)?;
        if weight.dims() != expected {
            return Err(Error::shape_mismatch(OP, expected, weight.shape()));
        }

        let bias = Tensor::zeros([out_features], weight.dtype(), &weight.device())?;
        Ok(Linear {
            weight: Param::new(weight),
            bias: Some(Param::new(bias)),
        })
    }

    /// Drop this layer's bias (`y = x·Wᵀ`), as pre-norm transformer blocks and
    /// tied output heads want.
    ///
    /// Consuming, so it reads as part of construction:
    /// `Linear::new(512, 512, &dev, &mut rng)?.without_bias()`. The bias is
    /// initialized to zeros and therefore consumes no randomness, so removing
    /// it afterwards leaves the weight — and the caller's `Rng` stream —
    /// exactly as it was.
    #[must_use]
    pub fn without_bias(mut self) -> Linear {
        self.bias = None;
        self
    }

    /// The input width (the weight's trailing axis).
    pub fn in_features(&self) -> usize {
        self.weight.value().dims()[1]
    }

    /// The output width (the weight's leading axis).
    pub fn out_features(&self) -> usize {
        self.weight.value().dims()[0]
    }

    /// The weight parameter, shape `[out_features, in_features]`.
    pub fn weight(&self) -> &Param {
        &self.weight
    }

    /// The bias parameter, shape `[out_features]`, or `None` for a
    /// [bias-free](Linear::without_bias) layer.
    pub fn bias(&self) -> Option<&Param> {
        self.bias.as_ref()
    }
}

impl Forward for Linear {
    type Output = Tensor;

    /// `x·Wᵀ + b`, with the weight transposed as a view (no copy).
    ///
    /// # Errors
    ///
    /// - [`Error::RankMismatch`] (`op: "Linear::forward"`, `expected: 2`) for a
    ///   rank-0 or rank-1 input: the trailing axis is the feature axis and at
    ///   least one batch axis must precede it (see the
    ///   [type docs](Linear#shapes)).
    /// - [`Error::ShapeMismatch`] (`op: "Linear::forward"`) if `x`'s trailing
    ///   axis is not [`in_features`](Linear::in_features) (`lhs` is the
    ///   weight's shape — the requirement — and `rhs` the input's).
    /// - whatever [`matmul`](crate::Tensor::matmul) reports for a device or
    ///   dtype mismatch against the parameters.
    fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        if x.rank() < 2 {
            return Err(Error::RankMismatch {
                op: "Linear::forward",
                expected: 2,
                got: x.rank(),
            });
        }
        if x.dims().last() != Some(&self.in_features()) {
            return Err(Error::shape_mismatch(
                "Linear::forward",
                self.weight.value().shape(),
                x.shape(),
            ));
        }
        let y = x.matmul(&self.weight.get(mode).transpose(-2, -1)?)?;
        match &self.bias {
            // `[O]` broadcasts against the output's trailing axis.
            Some(bias) => y.add(&bias.get(mode)),
            None => Ok(y),
        }
    }
}

/// The `Debug` line of a linear layer, runtime or typed: the geometry rather
/// than the values, `Linear(3 -> 2, bias)`.
pub(crate) fn debug_linear(
    f: &mut std::fmt::Formatter<'_>,
    in_features: usize,
    out_features: usize,
    has_bias: bool,
) -> std::fmt::Result {
    let bias = if has_bias { "bias" } else { "no bias" };
    write!(f, "Linear({in_features} -> {out_features}, {bias})")
}

impl std::fmt::Debug for Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        debug_linear(
            f,
            self.in_features(),
            self.out_features(),
            self.bias.is_some(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{self, Module, ModuleExt};
    use crate::testing::check_grad;

    const CPU: Device = Device::Cpu;

    fn t(data: &[f32], shape: impl Into<crate::shape::Shape>) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape, &CPU).unwrap()
    }

    fn v(x: &Tensor) -> Vec<f32> {
        x.to_vec::<f32>().unwrap()
    }

    /// A layer with hand-set parameters: `weight` `[O, I]`, `bias` `[O]`.
    fn fixed(weight: &[f32], out_features: usize, in_features: usize, bias: &[f32]) -> Linear {
        let mut fc = Linear::new(in_features, out_features, &CPU, &mut Rng::seed(0)).unwrap();
        let mut state = fc.state_dict().unwrap();
        state
            .insert("weight".to_string(), t(weight, [out_features, in_features]))
            .unwrap();
        state
            .insert("bias".to_string(), t(bias, [out_features]))
            .unwrap();
        fc.load_state_dict(&state).unwrap();
        fc
    }

    // ---- shapes and forward values ---------------------------------------

    #[test]
    fn parameter_paths_and_shapes_are_the_persistence_contract() {
        let fc = Linear::new(3, 2, &CPU, &mut Rng::seed(1)).unwrap();
        let state = fc.state_dict().unwrap();
        assert_eq!(state.keys().collect::<Vec<_>>(), ["bias", "weight"]);
        assert_eq!(state["weight"].dims(), &[2, 3]);
        assert_eq!(state["bias"].dims(), &[2]);
        assert_eq!(fc.num_params(), 2 * 3 + 2);
        assert_eq!((fc.in_features(), fc.out_features()), (3, 2));
        assert_eq!(fc.weight().value().dtype(), DType::F32);
        assert!(fc.bias().is_some());
    }

    #[test]
    fn forward_computes_x_times_weight_transposed_plus_bias() {
        // W = [[1, 2, 3], [4, 5, 6]], b = [10, 20], x = [[1, 0, -1], [2, 2, 2]]
        let mut fc = fixed(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3, &[10.0, 20.0]);
        let y = fc
            .forward(&t(&[1.0, 0.0, -1.0, 2.0, 2.0, 2.0], [2, 3]), Mode::EVAL)
            .unwrap();
        assert_eq!(y.dims(), &[2, 2]);
        // row 0: [1-3, 4-6] + [10, 20] = [8, 18]
        // row 1: [2+4+6, 8+10+12] + [10, 20] = [22, 50]
        assert_eq!(v(&y), vec![8.0, 18.0, 22.0, 50.0]);
    }

    #[test]
    fn leading_axes_are_batch_axes() {
        let mut fc = Linear::new(4, 3, &CPU, &mut Rng::seed(2)).unwrap();
        let x = Tensor::zeros([5, 7, 4], DType::F32, &CPU).unwrap();
        assert_eq!(fc.forward(&x, Mode::EVAL).unwrap().dims(), &[5, 7, 3]);
    }

    #[test]
    fn the_weight_is_never_materialized_transposed() {
        // The transposed weight is a view: it is not contiguous, and the
        // parameter it came from is untouched.
        let fc = Linear::new(3, 2, &CPU, &mut Rng::seed(3)).unwrap();
        let view = fc.weight().value().transpose(-2, -1).unwrap();
        assert_eq!(view.dims(), &[3, 2]);
        assert!(!view.is_contiguous());
        assert_eq!(fc.weight().value().dims(), &[2, 3]);
    }

    #[test]
    fn without_bias_drops_the_parameter_and_the_addend() {
        let mut fc = fixed(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3, &[10.0, 20.0]).without_bias();
        assert!(fc.bias().is_none());
        assert_eq!(
            fc.state_dict().unwrap().keys().collect::<Vec<_>>(),
            ["weight"],
            "a bias-free layer must not emit a bias path"
        );
        assert_eq!(fc.num_params(), 6);
        let y = fc
            .forward(&t(&[1.0, 0.0, -1.0], [1, 3]), Mode::EVAL)
            .unwrap();
        assert_eq!(v(&y), vec![-2.0, -2.0]);
    }

    #[test]
    fn without_bias_leaves_the_weight_and_the_rng_untouched() {
        // Zeros cost no randomness, so the bias-free layer's weight equals the
        // biased layer's and the caller's stream is at the same point.
        let mut a = Rng::seed(99);
        let mut b = Rng::seed(99);
        let with = Linear::new(4, 3, &CPU, &mut a).unwrap();
        let without = Linear::new(4, 3, &CPU, &mut b).unwrap().without_bias();
        assert_eq!(v(with.weight().value()), v(without.weight().value()));
        assert_eq!(a.state(), b.state());
    }

    #[test]
    fn same_seed_gives_the_same_weights_and_different_seeds_do_not() {
        let a = Linear::new(6, 5, &CPU, &mut Rng::seed(7)).unwrap();
        let b = Linear::new(6, 5, &CPU, &mut Rng::seed(7)).unwrap();
        let c = Linear::new(6, 5, &CPU, &mut Rng::seed(8)).unwrap();
        assert_eq!(v(a.weight().value()), v(b.weight().value()));
        assert_ne!(v(a.weight().value()), v(c.weight().value()));
    }

    #[test]
    fn initial_weights_sit_inside_the_kaiming_bound() {
        let fan_in = 64;
        let fc = Linear::new(fan_in, 32, &CPU, &mut Rng::seed(11)).unwrap();
        let bound = (6.0 / fan_in as f64).sqrt() as f32;
        let values = v(fc.weight().value());
        assert!(values.iter().all(|w| w.abs() <= bound), "outside ±{bound}");
        // Not a degenerate fill: the distribution actually spreads.
        let max = values.iter().fold(0.0f32, |m, w| m.max(w.abs()));
        assert!(max > 0.5 * bound, "max |w| = {max}, bound {bound}");
        // The bias starts at zero.
        assert!(v(fc.bias().unwrap().value()).iter().all(|b| *b == 0.0));
    }

    #[test]
    fn a_wrong_trailing_axis_is_a_named_shape_mismatch() {
        let mut fc = Linear::new(3, 2, &CPU, &mut Rng::seed(4)).unwrap();
        for bad in [vec![2usize, 4], vec![5, 7, 2]] {
            let x = Tensor::zeros(bad, DType::F32, &CPU).unwrap();
            let err = fc.forward(&x, Mode::EVAL).unwrap_err();
            assert!(
                matches!(
                    err,
                    Error::ShapeMismatch {
                        op: "Linear::forward",
                        ..
                    }
                ),
                "{err}"
            );
        }
    }

    #[test]
    fn a_rank_one_input_is_rejected_rather_than_promoted() {
        // `matmul` has no implicit vector promotion, so neither does `Linear`;
        // the error is named here rather than leaking `matmul`'s.
        let mut fc = Linear::new(3, 2, &CPU, &mut Rng::seed(14)).unwrap();
        let err = fc
            .forward(&t(&[1.0, 2.0, 3.0], [3]), Mode::EVAL)
            .unwrap_err();
        assert!(
            matches!(
                err,
                Error::RankMismatch {
                    op: "Linear::forward",
                    expected: 2,
                    got: 1,
                }
            ),
            "{err}"
        );
        // The same sample spelled `[1, I]` goes through.
        assert_eq!(
            fc.forward(&t(&[1.0, 2.0, 3.0], [1, 3]), Mode::EVAL)
                .unwrap()
                .dims(),
            &[1, 2]
        );
    }

    #[test]
    fn zero_dimensions_are_rejected() {
        for (i, o) in [(0, 2), (2, 0)] {
            let err = Linear::new(i, o, &CPU, &mut Rng::seed(5)).unwrap_err();
            assert!(
                matches!(
                    err,
                    Error::InvalidArg {
                        op: "Linear::new",
                        ..
                    }
                ),
                "{err}"
            );
        }
    }

    #[test]
    fn debug_reports_the_geometry() {
        let fc = Linear::new(3, 2, &CPU, &mut Rng::seed(6)).unwrap();
        assert_eq!(format!("{fc:?}"), "Linear(3 -> 2, bias)");
        assert_eq!(
            format!("{:?}", fc.without_bias()),
            "Linear(3 -> 2, no bias)"
        );
    }

    // ---- gradients --------------------------------------------------------

    #[test]
    fn input_gradients_match_finite_differences() {
        let fc = std::cell::RefCell::new(fixed(
            &[0.5, -1.5, 2.0, 0.25, 1.0, -0.75],
            2,
            3,
            &[0.5, -0.25],
        ));
        // A non-constant readout, so every output element contributes a
        // distinct weight to the scalar objective.
        let readout = t(&[1.0, -2.0, 3.0, 0.5], [2, 2]);
        check_grad(
            |i| {
                let y = fc.borrow_mut().forward(&i[0], Mode::TRAIN)?;
                y.mul(&readout)?.sum_all()
            },
            &[t(&[1.0, 0.0, -1.0, 2.0, 2.0, 2.0], [2, 3])],
            1e-3,
            1e-3,
        )
        .unwrap();
    }

    #[test]
    fn parameter_gradients_are_exact_for_a_summed_output() {
        // For L = Σ(x·Wᵀ + b) with x of shape [B, I]: ∂L/∂W[o, i] = Σ_b x[b, i]
        // (the same column sum for every output row) and ∂L/∂b[o] = B.
        let mut fc = fixed(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3, &[0.0, 0.0]);
        let x = t(&[1.0, 0.0, -1.0, 2.0, 2.0, 2.0], [2, 3]);
        let grads = fc
            .forward(&x, Mode::TRAIN)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();

        let dw = grads.wrt(fc.weight()).unwrap();
        assert_eq!(dw.dims(), &[2, 3]);
        // column sums of x: [3, 2, 1], once per output row
        assert_eq!(v(&dw), vec![3.0, 2.0, 1.0, 3.0, 2.0, 1.0]);

        let db = grads.wrt(fc.bias().unwrap()).unwrap();
        assert_eq!(db.dims(), &[2]);
        assert_eq!(v(&db), vec![2.0, 2.0]);
    }

    #[test]
    fn parameter_gradients_match_finite_differences() {
        // The same check `testing::check_grad` performs, but over a *parameter*
        // rather than an input: perturb one weight element through
        // `load_state_dict` and difference the loss.
        let base = [0.5, -1.5, 2.0, 0.25, 1.0, -0.75];
        let x = t(&[1.0, 0.5, -1.0, 2.0, -2.0, 0.25], [2, 3]);
        let readout = t(&[1.0, -2.0, 3.0, 0.5], [2, 2]);
        let loss = |weight: &[f32]| -> f32 {
            let mut fc = fixed(weight, 2, 3, &[0.5, -0.25]);
            // A quadratic readout, so the loss is not linear in the weight and
            // a wrong scale factor cannot hide.
            let y = fc.forward(&x, Mode::TRAIN).unwrap();
            y.mul(&y)
                .unwrap()
                .mul(&readout)
                .unwrap()
                .sum_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap()
        };

        let mut fc = fixed(&base, 2, 3, &[0.5, -0.25]);
        let y = fc.forward(&x, Mode::TRAIN).unwrap();
        let grads = y
            .mul(&y)
            .unwrap()
            .mul(&readout)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let analytic = v(&grads.wrt(fc.weight()).unwrap());

        let eps = 1e-2f32;
        for j in 0..base.len() {
            let mut plus = base;
            let mut minus = base;
            plus[j] += eps;
            minus[j] -= eps;
            let numeric = (loss(&plus) - loss(&minus)) / (2.0 * eps);
            assert!(
                (analytic[j] - numeric).abs() <= 1e-2 * numeric.abs().max(1.0),
                "weight {j}: analytic {} vs numeric {numeric}",
                analytic[j]
            );
        }
    }

    #[test]
    fn eval_records_nothing_but_eval_recorded_does() {
        let mut fc = Linear::new(3, 2, &CPU, &mut Rng::seed(12)).unwrap();
        let x = t(&[1.0, 2.0, 3.0], [1, 3]);

        let out = fc.forward(&x, Mode::EVAL).unwrap().sum_all().unwrap();
        assert!(matches!(out.backward(), Err(Error::NotTraced { .. })));

        // Fine-tuning: eval behavior, gradients still flow to the weight.
        let grads = fc
            .forward(&x, Mode::EVAL.recorded())
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        assert_eq!(
            v(&grads.wrt(fc.weight()).unwrap()),
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn the_walk_emits_weight_then_bias() {
        // `StateDict` is sorted by path, so the *walk* order is asserted here.
        let fc = Linear::new(3, 2, &CPU, &mut Rng::seed(13)).unwrap();
        let mut visited = Vec::new();
        fc.visit(&mut nn::Visitor::new(&mut |path, _| {
            visited.push(path.to_string());
        }));
        assert_eq!(visited, ["weight", "bias"]);
    }
}
