//! [`Dropout`] — inverted dropout over a layer-owned [`Rng`] stream.
//!
//! Two design points, both consequences of the crate having **no ambient
//! state**:
//!
//! * the layer owns its generator, split off the caller's `Rng` at
//!   construction — there is no thread-local, no global seed, and no
//!   "dropout context" threaded through `forward`; and
//! * whether the mask is applied is read off [`Mode::is_training`], the
//!   behavior axis, so `Mode::TRAIN.frozen()` (MC-dropout sampling) still
//!   drops while `Mode::EVAL.recorded()` (fine-tuning) does not.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::nn::{Forward, Mode};
use crate::rng::Rng;
use crate::tensor::Tensor;

/// The dtype the keep/drop mask is drawn in, whatever the input's dtype: the
/// mask a given seed produces then depends only on the seed and the shape, so
/// a model converted to `f16` drops exactly the same elements it did in `f32`.
///
/// Fixed at `f32` on purpose, with the cost understood: the draw is
/// activation-sized and uploaded, so an `f16` activation moves twice its own
/// bytes per dropout per step. Following the activation dtype would buy that
/// bandwidth back by spending accuracy. `MASK_DTYPE` is the precision of the
/// uniform draws and of the `p` they are compared against, and `f16` neither
/// represents a typical `p` exactly (`0.1` becomes `0.0999755859375`) nor
/// resolves `[0.5, 1)` more finely than `2^-11`, so the realized drop rate
/// shifts and a different set of elements drops. It would also cost the
/// property above — the same seed would stop dropping the same positions once a
/// model is converted. The scale factor is not the issue: `1/(1 - p)` is
/// applied by `mul_scalar` in the activation's own dtype and never touches this
/// constant. Either way the numbers move, so this is a pre-1.0 decision or
/// never, and it is deliberately not changed.
const MASK_DTYPE: DType = DType::F32;

/// Zero each element independently with probability `p` during training, and
/// scale what survives by `1/(1 - p)`.
///
/// That scaling is the **inverted-dropout** convention: it keeps the expected
/// value of every activation equal to its undropped value, so evaluation is
/// the plain identity — no rescaling at inference time, nothing to remember to
/// switch off.
///
/// The layer holds no parameters and no buffers; its only state is the `Rng`
/// stream, which is `#[module(skip)]`ed out of the parameter walk (it is
/// neither trainable nor part of a checkpoint's tensor set).
/// Consequently, a tensor state dictionary does not preserve the exact next
/// dropout mask; applications that require bit-for-bit resume must persist and
/// restore the layer's random stream as separate application state.
///
/// ```
/// use rstorch::nn::{Dropout, Forward, Mode};
/// use rstorch::{DType, Device, Rng, Tensor};
///
/// let dev = Device::Cpu;
/// let mut rng = Rng::seed(0);
/// let mut drop = Dropout::new(0.5, &mut rng)?;
/// let x = Tensor::ones([8, 16], DType::F32, &dev)?;
///
/// // Eval is the identity, element for element.
/// assert_eq!(drop.forward(&x, Mode::EVAL)?.to_vec::<f32>()?, x.to_vec::<f32>()?);
///
/// // Training keeps a scaled element or drops it outright.
/// let y = drop.forward(&x, Mode::TRAIN)?;
/// assert!(y.to_vec::<f32>()?.iter().all(|&v| v == 0.0 || v == 2.0));
/// # Ok::<(), rstorch::Error>(())
/// ```
#[derive(rstorch::Module)]
pub struct Dropout {
    /// The drop probability, in `[0, 1)`.
    p: f64,
    /// This layer's own stream. Not a parameter, not a buffer: skipped.
    #[module(skip)]
    rng: Rng,
}

impl Dropout {
    /// A dropout layer with drop probability `p`, seeding its own stream by
    /// [splitting](Rng::split) `rng` (which advances `rng`, so two layers
    /// constructed from one generator get decorrelated masks).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArg`] (`op: "Dropout::new"`) unless `p` is in
    /// `[0, 1)`. `p == 1` is excluded rather than clamped: it would zero every
    /// activation and scale by infinity, which is a configuration mistake, not
    /// a limit worth taking.
    pub fn new(p: f64, rng: &mut Rng) -> Result<Dropout> {
        if !(p.is_finite() && (0.0..1.0).contains(&p)) {
            return Err(Error::InvalidArg {
                op: "Dropout::new",
                msg: format!("p must be in [0, 1), got {p}"),
            });
        }
        Ok(Dropout {
            p,
            rng: rng.split(),
        })
    }

    /// The drop probability.
    pub fn p(&self) -> f64 {
        self.p
    }
}

impl Forward for Dropout {
    type Output = Tensor;

    /// Apply the mask under a training [`Mode`]; return `x` unchanged
    /// otherwise (and for `p == 0`, which has nothing to drop).
    ///
    /// The mask is a constant as far as autograd is concerned, so the
    /// gradient of a kept element is `1/(1 - p)` and of a dropped one `0` —
    /// which is what makes the two passes consistent.
    ///
    /// Whether the output is *traced* follows the input, not `mode`: the
    /// recording axis gates [`Param::get`](crate::nn::Param::get), and this
    /// layer has no parameters, so `Mode::TRAIN.frozen()` over an already-traced
    /// activation still records. It is a parameter freeze, not a `no_grad`
    /// block; MC-dropout gets its cheap pass by feeding an untraced input.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArg`] (`op: "Dropout::forward"`) for a non-float `x`:
    /// scaling an integer tensor by `1/(1 - p)` would truncate, and an index
    /// buffer is structure rather than an activation.
    fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        if !mode.is_training() || self.p == 0.0 {
            return Ok(x.clone());
        }
        if !x.dtype().is_float() {
            return Err(Error::InvalidArg {
                op: "Dropout::forward",
                msg: format!("dropout applies to float activations, got {}", x.dtype()),
            });
        }
        let device = x.device();
        let draws = Tensor::rand(x.dims().to_vec(), MASK_DTYPE, &device, &mut self.rng)?;
        // `rand` samples `[0, 1)`, so `draw < p` happens with probability `p`.
        let threshold = Tensor::full([1], self.p, MASK_DTYPE, &device)?;
        let dropped = draws.lt(&threshold)?;
        x.mul_scalar(1.0 / (1.0 - self.p))?
            .masked_fill(&dropped, 0.0)
    }
}

impl std::fmt::Debug for Dropout {
    /// Reports `p` only: the stream's raw state is not useful in a log line.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Dropout(p={})", self.p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::nn::{self, Module, ModuleExt};
    use crate::testing::check_grad;

    const CPU: Device = Device::Cpu;

    /// A moderately large all-ones input, so mask statistics are meaningful.
    fn ones(len: usize) -> Tensor {
        Tensor::ones([len], DType::F32, &CPU).unwrap()
    }

    fn v(x: &Tensor) -> Vec<f32> {
        x.to_vec::<f32>().unwrap()
    }

    fn dropout(p: f64, seed: u64) -> Dropout {
        Dropout::new(p, &mut Rng::seed(seed)).unwrap()
    }

    // ---- Mode gating ------------------------------------------------------

    #[test]
    fn eval_is_the_identity() {
        let mut d = dropout(0.5, 1);
        let x = ones(256);
        let y = d.forward(&x, Mode::EVAL).unwrap();
        assert_eq!(v(&y), v(&x));
        // …and it consumed no randomness, so the stream is untouched.
        assert_eq!(d.rng.state(), dropout(0.5, 1).rng.state());
    }

    #[test]
    fn eval_recorded_is_also_the_identity() {
        // Fine-tuning reads the *behavior* axis: recording does not turn
        // dropout back on.
        let mut d = dropout(0.5, 2);
        let x = ones(64).traced().unwrap();
        let y = d.forward(&x, Mode::EVAL.recorded()).unwrap();
        assert_eq!(v(&y), v(&x));
        // Still differentiable — the identity passes the graph through.
        assert_eq!(
            v(&y.sum_all()
                .unwrap()
                .backward()
                .unwrap()
                .wrt_input(&x)
                .unwrap()),
            vec![1.0; 64]
        );
    }

    #[test]
    fn train_zeros_some_elements_and_scales_the_rest() {
        let p = 0.25;
        let mut d = dropout(p, 3);
        let n = 4096;
        let y = v(&d.forward(&ones(n), Mode::TRAIN).unwrap());
        let scale = (1.0 / (1.0 - p)) as f32;
        assert!(
            y.iter().all(|&value| value == 0.0 || value == scale),
            "every element must be dropped or scaled by {scale}"
        );
        let dropped = y.iter().filter(|&&value| value == 0.0).count();
        // Binomial(4096, 0.25) has sd ≈ 27.7; ±5 sd is ~139.
        assert!(
            (dropped as f64 - p * n as f64).abs() < 140.0,
            "dropped {dropped} of {n} at p={p}"
        );
        // The expectation is preserved by the inverted-dropout scaling.
        let mean = f64::from(y.iter().sum::<f32>()) / n as f64;
        assert!((mean - 1.0).abs() < 0.05, "mean {mean}");
    }

    #[test]
    fn train_frozen_still_drops() {
        // MC-dropout sampling: train *behavior* with recording off. The mask is
        // applied because `is_training()` says so, independently of the
        // recording axis.
        let mut d = dropout(0.5, 4);
        let x = ones(512);
        let y = d.forward(&x, Mode::TRAIN.frozen()).unwrap();
        assert!(v(&y).contains(&0.0), "mask not applied");
        // Nothing upstream is traced and this layer has no parameters, so no
        // graph exists to differentiate.
        assert!(matches!(
            y.sum_all().unwrap().backward(),
            Err(Error::NotTraced { .. })
        ));
    }

    #[test]
    fn a_traced_input_keeps_its_graph_under_a_frozen_mode() {
        // The recording axis gates `Param::get`, not tracing in general: an
        // already-traced activation still records through a parameter-free
        // layer, which is what makes `Mode::TRAIN.frozen()` a *parameter*
        // freeze rather than a `no_grad` block.
        let mut d = dropout(0.5, 41);
        let x = ones(64).traced().unwrap();
        let y = d.forward(&x, Mode::TRAIN.frozen()).unwrap();
        let grad = v(&y
            .sum_all()
            .unwrap()
            .backward()
            .unwrap()
            .wrt_input(&x)
            .unwrap());
        for (value, g) in v(&y).iter().zip(&grad) {
            assert_eq!(*g, if *value == 0.0 { 0.0 } else { 2.0 });
        }
    }

    #[test]
    fn p_zero_is_the_identity_even_in_training() {
        let mut d = dropout(0.0, 5);
        let x = ones(32);
        assert_eq!(v(&d.forward(&x, Mode::TRAIN).unwrap()), v(&x));
    }

    // ---- determinism ------------------------------------------------------

    #[test]
    fn the_same_seed_gives_the_same_mask() {
        let (mut a, mut b) = (dropout(0.5, 1234), dropout(0.5, 1234));
        let x = ones(1024);
        // Every forward, not just the first: the two streams stay in lockstep.
        for _ in 0..3 {
            assert_eq!(
                v(&a.forward(&x, Mode::TRAIN).unwrap()),
                v(&b.forward(&x, Mode::TRAIN).unwrap())
            );
        }
    }

    #[test]
    fn successive_forwards_draw_fresh_masks() {
        let mut d = dropout(0.5, 6);
        let x = ones(1024);
        let first = v(&d.forward(&x, Mode::TRAIN).unwrap());
        let second = v(&d.forward(&x, Mode::TRAIN).unwrap());
        assert_ne!(first, second);
    }

    #[test]
    fn a_different_seed_gives_a_different_mask() {
        let x = ones(1024);
        let first = v(&dropout(0.5, 7).forward(&x, Mode::TRAIN).unwrap());
        let second = v(&dropout(0.5, 8).forward(&x, Mode::TRAIN).unwrap());
        assert_ne!(first, second);
    }

    #[test]
    fn two_layers_from_one_generator_get_decorrelated_streams() {
        let mut parent = Rng::seed(2024);
        let (mut a, mut b) = (
            Dropout::new(0.5, &mut parent).unwrap(),
            Dropout::new(0.5, &mut parent).unwrap(),
        );
        let x = ones(1024);
        assert_ne!(
            v(&a.forward(&x, Mode::TRAIN).unwrap()),
            v(&b.forward(&x, Mode::TRAIN).unwrap())
        );
    }

    #[test]
    fn the_mask_dtype_is_pinned_so_the_mask_is_independent_of_the_input_dtype() {
        // Not an accident of the implementation: `MASK_DTYPE` is deliberately
        // f32 whatever the activation dtype, so one seed drops one set of
        // positions and a model converted to f64 or f16 keeps dropping exactly
        // those. Drawing the mask in the activation's dtype would buy back the
        // upload bandwidth (an f16 activation moves twice its own bytes here)
        // and break this: f16 cannot hold a typical `p` exactly, so the
        // comparison `draw < p` would select differently.
        let x32 = ones(256);
        let x64 = Tensor::ones([256], DType::F64, &CPU).unwrap();
        let x16 = Tensor::ones([256], DType::F16, &CPU).unwrap();
        let dropped32: Vec<bool> = v(&dropout(0.5, 9).forward(&x32, Mode::TRAIN).unwrap())
            .iter()
            .map(|&value| value == 0.0)
            .collect();
        let dropped64: Vec<bool> = dropout(0.5, 9)
            .forward(&x64, Mode::TRAIN)
            .unwrap()
            .to_vec::<f64>()
            .unwrap()
            .iter()
            .map(|&value| value == 0.0)
            .collect();
        let dropped16: Vec<bool> = dropout(0.5, 9)
            .forward(&x16, Mode::TRAIN)
            .unwrap()
            .to_vec::<half::f16>()
            .unwrap()
            .iter()
            .map(|&value| value == half::f16::ZERO)
            .collect();
        assert_eq!(dropped32, dropped64);
        assert_eq!(dropped32, dropped16);
    }

    // ---- gradients and structure -----------------------------------------

    #[test]
    fn gradients_match_finite_differences() {
        // A fresh layer per evaluation reproduces the mask exactly, which is
        // what makes the objective deterministic enough to difference.
        check_grad(
            |i| {
                dropout(0.5, 31)
                    .forward(&i[0], Mode::TRAIN)?
                    .mul(&i[0])?
                    .sum_all()
            },
            &[Tensor::from_vec(
                vec![1.0f32, -2.0, 0.5, 3.0, -1.5, 0.25, 2.5, -0.75],
                [8],
                &CPU,
            )
            .unwrap()],
            1e-3,
            1e-3,
        )
        .unwrap();
    }

    #[test]
    fn dropped_positions_get_no_gradient_and_kept_ones_get_the_scale() {
        let mut d = dropout(0.5, 32);
        let x = ones(64).traced().unwrap();
        let y = d.forward(&x, Mode::TRAIN).unwrap();
        let grad = v(&y
            .sum_all()
            .unwrap()
            .backward()
            .unwrap()
            .wrt_input(&x)
            .unwrap());
        for (value, g) in v(&y).iter().zip(&grad) {
            assert_eq!(*g, if *value == 0.0 { 0.0 } else { 2.0 });
        }
    }

    #[test]
    fn a_non_float_input_is_rejected() {
        let idx = Tensor::from_vec(vec![1i64, 2, 3], [3], &CPU).unwrap();
        let err = dropout(0.5, 10).forward(&idx, Mode::TRAIN).unwrap_err();
        assert!(
            matches!(
                err,
                Error::InvalidArg {
                    op: "Dropout::forward",
                    ..
                }
            ),
            "{err}"
        );
        // Eval short-circuits before the check: the identity is dtype-blind.
        assert!(dropout(0.5, 10).forward(&idx, Mode::EVAL).is_ok());
    }

    #[test]
    fn p_outside_the_unit_interval_is_rejected() {
        for bad in [-0.1, 1.0, 1.5, f64::NAN] {
            let err = Dropout::new(bad, &mut Rng::seed(0)).unwrap_err();
            assert!(
                matches!(
                    err,
                    Error::InvalidArg {
                        op: "Dropout::new",
                        ..
                    }
                ),
                "{bad}: {err}"
            );
        }
        assert_eq!(dropout(0.3, 0).p(), 0.3);
    }

    #[test]
    fn the_layer_holds_no_parameters_and_no_buffers() {
        let d = dropout(0.5, 11);
        assert!(d.state_dict().is_empty(), "rng must not be a leaf");
        assert_eq!(d.num_params(), 0);
        let mut visited = 0;
        d.visit(&mut nn::Visitor::new(&mut |_, _| visited += 1));
        assert_eq!(visited, 0);
    }

    #[test]
    fn debug_reports_p() {
        assert_eq!(format!("{:?}", dropout(0.25, 12)), "Dropout(p=0.25)");
    }
}
