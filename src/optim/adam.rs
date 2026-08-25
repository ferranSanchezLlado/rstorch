//! [`Adam`] and [`AdamW`] — one implementation, two names.

use crate::autograd::Grads;
use crate::backend::{FusedOp, dispatch};
use crate::error::{Error, Result};
use crate::nn::{Module, Param};
use crate::persist::Envelope;
use crate::tensor::Tensor;

use super::engine::{self, Decode, Engine, Rule, Update};
use super::state::{self, Incoming};

/// The hyperparameters of one parameter group (or of the optimizer itself).
///
/// A group is declared with [`Adam::group`] and states only its *differences*
/// from the optimizer's own settings — the standard transformer recipe:
///
/// ```
/// # use rstorch::optim::AdamW;
/// let opt = AdamW::new(3e-4, 0.1)
///     .group(|path| path.ends_with("bias") || path.contains("norm"),
///            |g| g.weight_decay(0.0));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct AdamGroup {
    lr_scale: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
}

impl AdamGroup {
    /// Scale this group's learning rate relative to the optimizer's
    /// (discriminative learning rates).
    ///
    /// A *scale* rather than an absolute rate, so `set_lr` and the
    /// [`schedule`](super::schedule) functions keep working: the effective rate
    /// is always `optimizer lr × scale`.
    #[must_use]
    pub fn lr_scale(mut self, scale: f64) -> AdamGroup {
        self.lr_scale = scale;
        self
    }

    /// The exponential decay rates of the first and second moment estimates.
    #[must_use]
    pub fn betas(mut self, beta1: f64, beta2: f64) -> AdamGroup {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// The denominator's numerical-stability term.
    #[must_use]
    pub fn eps(mut self, eps: f64) -> AdamGroup {
        self.eps = eps;
        self
    }

    /// Weight decay — coupled into the gradient for [`Adam`], applied directly
    /// to the parameter for [`AdamW`] (`0.0` disables it).
    #[must_use]
    pub fn weight_decay(mut self, weight_decay: f64) -> AdamGroup {
        self.weight_decay = weight_decay;
        self
    }
}

/// Adam's per-parameter buffers: the two moment estimates. The step clock that
/// rides alongside is `engine::ParamState`'s.
pub(crate) struct Moments {
    m: Tensor,
    v: Tensor,
}

/// Adam as the shared [`Engine`] sees it. `decoupled` chooses where weight
/// decay enters, which is the whole of the Adam/AdamW difference.
#[derive(Clone, Copy)]
pub(crate) struct AdamRule {
    pub(crate) decoupled: bool,
}

impl Rule for AdamRule {
    type Hyper = AdamGroup;
    type Buffers = Moments;

    const NAME: &'static str = "Adam";
    /// The same tag for both names, because they are the same optimizer.
    const KIND: &'static str = "adam";
    const HYPERS: &'static [&'static str] =
        &["lr", "beta1", "beta2", "eps", "weight_decay", "decoupled"];

    fn lr_scale(hyper: &AdamGroup) -> f64 {
        hyper.lr_scale
    }

    fn check(&self, param: &Param, hyper: AdamGroup, lr: f64, clock: u64) -> Result<()> {
        // Exactly the scalars the step will hand the kernel, including its
        // own bias-correction clock.
        let scalars = kernel_scalars(lr, hyper, self.decoupled, clock);
        let acc = param.value().dtype().accumulation_dtype();
        crate::optim::validate::adam_scalars("step", &scalars, acc)
    }

    /// The formula in [`Adam`]'s docs.
    ///
    /// A backend with a fused `AdamStep` kernel runs it in one pass; where there
    /// is none the same arithmetic is spelled out in the public op vocabulary,
    /// which is the definition the two agree on. `weights` and `grad` arrive
    /// already widened to the accumulation dtype, and the value returned is
    /// narrowed back by the caller.
    fn update(&self, update: Update<'_, Self>) -> Result<(Tensor, Moments)> {
        let Update {
            hyper,
            lr,
            weights,
            grad,
            previous,
            clock,
        } = update;
        let decoupled = self.decoupled;
        let (previous_m, previous_v) = match previous {
            Some(moments) => (moments.m.clone(), moments.v.clone()),
            None => {
                let zeros = Tensor::zeros(weights.dims(), weights.dtype(), &weights.device())?;
                (zeros.clone(), zeros)
            }
        };
        let scalars = kernel_scalars(lr, hyper, decoupled, clock);
        // Materialize the four backend inputs in a short scope. The fallback
        // consumes `grad` and the weight/moment handles after this call.
        let fused = {
            let inputs = [
                weights.ready_view()?,
                grad.ready_view()?,
                previous_m.ready_view()?,
                previous_v.ready_view()?,
            ];
            dispatch::backend(inputs[0].device()).fused(FusedOp::AdamStep, &inputs, &scalars)
        };

        let [next, m, v] = match fused {
            Ok(outputs) => {
                let outputs = engine::fused_outputs(Self::NAME, outputs, 3, &weights)?;
                // `fused_outputs` validated the length as 3 and pushes
                // exactly one tensor per output, so this cannot fail.
                outputs
                    .try_into()
                    .unwrap_or_else(|_| unreachable!("fused Adam output count validated as 3"))
            }
            Err(Error::Unsupported { .. }) => {
                let [
                    _lr,
                    _beta1,
                    _beta2,
                    _eps,
                    _decay,
                    correction1,
                    correction2,
                    _flag,
                ] = scalars;
                let mut g = grad;
                if hyper.weight_decay != 0.0 && !decoupled {
                    g = g.add(&weights.mul_scalar(hyper.weight_decay)?)?;
                }
                let next_m = previous_m
                    .mul_scalar(hyper.beta1)?
                    .add(&g.mul_scalar(1.0 - hyper.beta1)?)?;
                let next_v = previous_v
                    .mul_scalar(hyper.beta2)?
                    .add(&g.mul(&g)?.mul_scalar(1.0 - hyper.beta2)?)?;

                let m_hat = next_m.div_scalar(correction1)?;
                let v_hat = next_v.div_scalar(correction2)?;
                let direction = m_hat.div(&v_hat.sqrt()?.add_scalar(hyper.eps)?)?;

                let mut next = weights;
                if hyper.weight_decay != 0.0 && decoupled {
                    next = next.mul_scalar(1.0 - lr * hyper.weight_decay)?;
                }
                next = next.sub(&direction.mul_scalar(lr)?)?;
                [next, next_m, next_v]
            }
            Err(error) => return Err(error),
        };
        Ok((next, Moments { m, v }))
    }

    fn hypers(&self, base: AdamGroup) -> Vec<(&'static str, f64)> {
        vec![
            ("beta1", base.beta1),
            ("beta2", base.beta2),
            ("eps", base.eps),
            ("weight_decay", base.weight_decay),
            ("decoupled", f64::from(u8::from(self.decoupled))),
        ]
    }

    fn buffers(moments: &Moments) -> Vec<(&'static str, &Tensor)> {
        vec![("m", &moments.m), ("v", &moments.v)]
    }

    fn adopt(&mut self, base: &mut AdamGroup, incoming: &Incoming) -> Result<()> {
        // The writer only ever emits 0 or 1 (`hypers`, above), and the fused
        // kernel's own validator is exact, so anything else is a malformed
        // file rather than a value to coerce — `0.5` or `nan` must not load
        // silently as AdamW.
        let saved_decoupled = match incoming.hyper("decoupled")? {
            0.0 => false,
            1.0 => true,
            other => {
                return Err(Error::persistence(format!(
                    "optimizer state has decoupled={other}, which is neither 0 nor 1"
                )));
            }
        };
        // A coupled-decay checkpoint and an AdamW are different algorithms, so
        // that is a rejection rather than a silent reconfiguration.
        if saved_decoupled != self.decoupled {
            return Err(Error::persistence(format!(
                "optimizer state has decoupled={saved_decoupled}, loaded into an \
                     optimizer with decoupled={} (Adam and AdamW apply weight decay \
                     differently; build the one the checkpoint was saved from)",
                self.decoupled
            )));
        }
        base.beta1 = incoming.hyper("beta1")?;
        base.beta2 = incoming.hyper("beta2")?;
        base.eps = incoming.hyper("eps")?;
        base.weight_decay = incoming.hyper("weight_decay")?;
        Ok(())
    }

    fn decode(&self, decode: Decode<'_, Self>) -> Result<Moments> {
        let Decode {
            path, saved, value, ..
        } = decode;
        let mut m = None;
        let mut v = None;
        for (name, host) in &saved.buffers {
            match name.as_str() {
                "m" => m = Some(state::restore_buffer(host, value, path, name)?),
                "v" => v = Some(state::restore_buffer(host, value, path, name)?),
                other => return Err(state::unknown_buffer(Self::KIND, path, other)),
            }
        }
        let missing = match (m, v) {
            (Some(m), Some(v)) => return Ok(Moments { m, v }),
            (None, _) => "m",
            (_, None) => "v",
        };
        Err(Error::persistence(format!(
            "optimizer state for `{path}` is missing the `{missing}` moment \
                 (Adam keeps both or neither)"
        )))
    }
}

/// The eight scalars the Adam kernel takes for one parameter, at that
/// parameter's own step count `clock` and its group's effective `lr`.
///
/// The pre-pass range-checks *exactly* these, so a step's validation and its
/// arithmetic cannot disagree about what the kernel will be handed.
fn kernel_scalars(lr: f64, hyper: AdamGroup, decoupled: bool, clock: u64) -> [f64; 8] {
    // Each parameter's own clock, so a late joiner is bias-corrected as a first
    // update rather than as step `steps`.
    //
    // Saturating at `i32::MAX` is exact, not a fallback: `beta^t` for
    // `beta < 1` has already underflowed to 0 long before `t = 2³¹`, so every
    // larger `t` gives the same correction factor of 1.
    let t = clock.min(i32::MAX as u64) as i32;
    [
        lr,
        hyper.beta1,
        hyper.beta2,
        hyper.eps,
        hyper.weight_decay,
        1.0 - hyper.beta1.powi(t),
        1.0 - hyper.beta2.powi(t),
        f64::from(u8::from(decoupled)),
    ]
}

/// Adam, with decoupled weight decay available under the name [`AdamW`].
///
/// The update, for each non-frozen parameter with a gradient, at that
/// parameter's own step count `t`:
///
/// ```text
/// g ← grad (+ weight_decay · w, when decay is coupled)
/// m ← β₁ · m + (1 − β₁) · g          v ← β₂ · v + (1 − β₂) · g²
/// m̂ ← m / (1 − β₁ᵗ)                  v̂ ← v / (1 − β₂ᵗ)
/// w ← w · (1 − lr · weight_decay)    (only when decay is decoupled)
/// w ← w − lr · m̂ / (√v̂ + eps)
/// ```
///
/// # Per-parameter step clocks
///
/// `t` is **per parameter**, not a global counter. A parameter that joins the
/// update late — it was frozen for a warmup phase, or a new head was attached
/// to a pretrained trunk — starts at `t = 1`, so its
/// first update is bias-corrected as a first update. A global clock would
/// divide a fresh, near-zero moment estimate by an almost-saturated correction
/// factor and take a step several times too small.
///
/// # Adam vs `AdamW`
///
/// One implementation and one type: the only difference is *where* weight decay
/// enters, so it is a flag ([`Adam::decoupled`]) rather than a second code path.
/// [`AdamW::new`] is the named constructor that sets it.
///
/// ```
/// # use rstorch::nn::{Mode, Param};
/// # use rstorch::optim::Adam;
/// # use rstorch::{DType, Device, Result, Tensor};
/// # fn main() -> Result<()> {
/// #[derive(rstorch::Module)]
/// struct Model {
///     w: Param,
/// }
///
/// let dev = Device::Cpu;
/// let mut model = Model {
///     w: Param::new(Tensor::full([2], 1.0, DType::F32, &dev)?),
/// };
/// let mut opt = Adam::new(0.1);
///
/// let w = model.w.get(Mode::TRAIN);
/// let loss = w.mul(&w)?.sum_all()?;
/// opt.step(&mut model, loss.backward()?)?;
///
/// // The first Adam step is ±lr whatever the gradient's magnitude.
/// let updated = model.w.value().to_vec::<f32>()?;
/// assert!((updated[0] - 0.9).abs() < 1e-6, "{updated:?}");
/// # Ok(())
/// # }
/// ```
pub struct Adam {
    pub(crate) engine: Engine<AdamRule>,
}

impl Adam {
    /// Adam at learning rate `lr` with the usual defaults
    /// (β = (0.9, 0.999), eps = 1e-8, no weight decay).
    pub fn new(lr: f64) -> Adam {
        Adam {
            engine: Engine::new(
                lr,
                AdamRule { decoupled: false },
                AdamGroup {
                    lr_scale: 1.0,
                    beta1: 0.9,
                    beta2: 0.999,
                    eps: 1e-8,
                    weight_decay: 0.0,
                },
            ),
        }
    }

    /// Set the moment decay rates for every parameter (see [`AdamGroup`]).
    #[must_use]
    pub fn betas(mut self, beta1: f64, beta2: f64) -> Adam {
        let base = self.engine.base_mut();
        base.beta1 = beta1;
        base.beta2 = beta2;
        self
    }

    /// Set the numerical-stability term for every parameter.
    #[must_use]
    pub fn eps(mut self, eps: f64) -> Adam {
        self.engine.base_mut().eps = eps;
        self
    }

    /// Set the weight decay for every parameter. Whether it is coupled or
    /// decoupled is [`decoupled`](Adam::decoupled)'s business.
    #[must_use]
    pub fn weight_decay(mut self, weight_decay: f64) -> Adam {
        self.engine.base_mut().weight_decay = weight_decay;
        self
    }

    /// Choose where weight decay enters: `true` applies it straight to the
    /// parameter (`AdamW`), `false` folds it into the gradient (classic Adam).
    #[must_use]
    pub fn decoupled(mut self, decoupled: bool) -> Adam {
        self.engine.rule.decoupled = decoupled;
        self
    }

    /// Whether weight decay is decoupled (i.e. whether this is an `AdamW`).
    pub fn is_decoupled(&self) -> bool {
        self.engine.rule.decoupled
    }

    /// Declare a parameter group: parameters whose dotted path satisfies
    /// `predicate` are updated with `configure(defaults)` instead of the
    /// optimizer's own hyperparameters.
    ///
    /// The **first** matching group wins, so declare the most specific
    /// predicate first. `configure` is re-evaluated on every step against the
    /// current defaults, so a later [`set_lr`](Adam::set_lr) or builder call
    /// still reaches groups that do not override that particular field.
    #[must_use]
    pub fn group(
        mut self,
        predicate: impl Fn(&str) -> bool + Send + Sync + 'static,
        configure: impl Fn(AdamGroup) -> AdamGroup + Send + Sync + 'static,
    ) -> Adam {
        self.engine.group(predicate, configure);
        self
    }

    /// The current base learning rate.
    pub fn lr(&self) -> f64 {
        self.engine.lr()
    }

    /// Set the base learning rate — the hook every
    /// [`schedule`](super::schedule) uses. Group
    /// [`lr_scale`](AdamGroup::lr_scale) factors apply on top, so a schedule
    /// moves every group together.
    pub fn set_lr(&mut self, lr: f64) {
        self.engine.set_lr(lr);
    }

    /// How many times [`step`](Adam::step) has **succeeded** (the argument the
    /// [`schedule`](super::schedule) functions take); a rejected step leaves it
    /// untouched. Individual parameters may have taken fewer — see the
    /// per-parameter clocks in the type docs.
    pub fn steps(&self) -> u64 {
        self.engine.steps()
    }

    /// Apply one update to every non-frozen parameter of `model`, consuming
    /// `grads`.
    ///
    /// # Errors
    ///
    /// - [`Error::MissingGrad`](crate::Error::MissingGrad), naming the dotted
    ///   path, if a non-frozen parameter has no gradient — the loud answer to an
    ///   accidentally untrained weight. Nothing is updated in that case: the
    ///   whole walk is validated before the first swap.
    /// - [`Error::ShapeMismatch`](crate::Error::ShapeMismatch) /
    ///   [`Error::DTypeMismatch`](crate::Error::DTypeMismatch) /
    ///   [`Error::DeviceMismatch`](crate::Error::DeviceMismatch) if a gradient
    ///   does not match its parameter, and
    ///   [`Error::InvalidArg`](crate::Error::InvalidArg) if a parameter is not a
    ///   float tensor or the module visits one parameter twice.
    pub fn step(&mut self, model: &mut dyn Module, grads: Grads) -> Result<()> {
        self.engine.step(model, grads)
    }

    /// Write this optimizer's state (hyperparameters, per-parameter step clocks
    /// and both moment buffers) into `envelope`, keyed by `model`'s dotted
    /// paths.
    ///
    /// Parameter *groups* are not saved: they are closures, reconstructed by
    /// building the optimizer the same way.
    ///
    /// # Errors
    ///
    /// [`Error::Persistence`](crate::Error::Persistence) if `envelope` already
    /// carries optimizer state, or if a host transfer fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use rstorch::nn::{Mode, Param};
    /// use rstorch::optim::Adam;
    /// use rstorch::persist::Envelope;
    /// use rstorch::{DType, Device, Result, Tensor};
    ///
    /// #[derive(rstorch::Module)]
    /// struct Model {
    ///     w: Param,
    /// }
    ///
    /// # fn main() -> Result<()> {
    /// let dev = Device::Cpu;
    /// let mut model = Model {
    ///     w: Param::new(Tensor::full([2], 1.0, DType::F32, &dev)?),
    /// };
    /// let mut opt = Adam::new(0.1);
    /// let w = model.w.get(Mode::TRAIN);
    /// let loss = w.mul(&w)?.sum_all()?;
    /// opt.step(&mut model, loss.backward()?)?;
    ///
    /// let mut envelope = Envelope::new();
    /// opt.save_state(&model, &mut envelope)?;
    ///
    /// let mut restored = Adam::new(0.1);
    /// restored.load_state(&model, &envelope)?;
    /// assert_eq!(restored.param_steps(&model.w), opt.param_steps(&model.w));
    /// # Ok(())
    /// # }
    /// ```
    pub fn save_state(&self, model: &dyn Module, envelope: &mut Envelope) -> Result<()> {
        self.engine.save(model, envelope)
    }

    /// Restore the state [`save_state`](Adam::save_state) wrote, resolving paths
    /// against `model` (which must be the model the checkpoint was saved for).
    ///
    /// All-or-nothing: every buffer is decoded and checked against its
    /// parameter before any of the optimizer's own state is replaced.
    ///
    /// # Errors
    ///
    /// [`Error::Persistence`](crate::Error::Persistence) if the envelope carries
    /// no optimizer state, was written by a different optimizer, names a
    /// parameter `model` does not have, holds a buffer whose shape/dtype does
    /// not match its parameter, is missing one of the two moments, or disagrees
    /// about [`decoupled`](Adam::decoupled) — a coupled-decay checkpoint and an
    /// `AdamW` are different algorithms, so that is a rejection rather than a
    /// silent reconfiguration.
    pub fn load_state(&mut self, model: &dyn Module, envelope: &Envelope) -> Result<()> {
        self.engine.load(model, envelope)
    }

    /// The number of updates `param` has received from this optimizer — its
    /// own bias-correction clock `t`, which is **not** [`steps`](Adam::steps)
    /// when the parameter was frozen for a while or joined the model late.
    ///
    /// `0` for a parameter this optimizer has never updated (including one it
    /// has never seen).
    pub fn param_steps(&self, param: &Param) -> u64 {
        self.engine.param_steps(param)
    }
}

/// **`AdamW`**: Adam with *decoupled* weight decay (Loshchilov & Hutter, 2019).
///
/// One implementation, two names. The decay term is the only
/// difference — coupled decay adds `weight_decay · w` to the gradient, where
/// decoupled decay multiplies the parameter by `1 − lr · weight_decay` — so
/// `AdamW` is the named constructor for [`Adam`] with that flag set rather than
/// a second copy of the algorithm:
///
/// ```
/// # use rstorch::optim::{Adam, AdamW};
/// let opt = AdamW::new(3e-4, 0.1);
/// assert!(opt.is_decoupled());
/// // The same optimizer, spelled the long way:
/// let same = Adam::new(3e-4).decoupled(true).weight_decay(0.1);
/// assert_eq!(opt.lr(), same.lr());
/// ```
///
/// Everything else — parameter groups, `set_lr`, schedules, state persistence —
/// is [`Adam`]'s surface, because the value *is* an [`Adam`].
pub struct AdamW;

impl AdamW {
    /// An [`Adam`] at learning rate `lr` with decoupled `weight_decay`.
    // Returning `Adam` rather than `Self` is the whole point: `AdamW` is a
    // *name* for a configuration of the one implementation, not a second one.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(lr: f64, weight_decay: f64) -> Adam {
        Adam::new(lr).decoupled(true).weight_decay(weight_decay)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Mode;
    use crate::optim::testkit::{
        Net, REGRESSION_XS, Solo, assert_an_exhausted_clock_outranks_an_invalid_hyperparameter,
        assert_converges_on_a_tiny_regression, assert_every_rejection_is_atomic,
        assert_next_step_is_refused, assert_state_round_trips, close, with_clocks,
    };

    /// One update of `model` under `opt`, panicking on anything but success.
    fn step(opt: &mut Adam, model: &mut Net) {
        let loss = model.unit_grad_loss(Mode::TRAIN).unwrap();
        opt.step(model, loss.backward().unwrap()).unwrap();
    }

    #[test]
    fn the_first_step_is_the_learning_rate_whatever_the_gradient_size() {
        // Adam normalizes by the second moment, so the very first step is ±lr.
        // Two models with gradients differing by 1000x must move identically.
        for scale in [1.0f64, 1000.0] {
            let mut model = Net::ones();
            let mut opt = Adam::new(0.1);
            let loss = model
                .unit_grad_loss(Mode::TRAIN)
                .unwrap()
                .mul_scalar(scale)
                .unwrap();
            opt.step(&mut model, loss.backward().unwrap()).unwrap();
            close(f64::from(model.at("trunk.weight")), 0.9, 1e-6);
        }
    }

    #[test]
    fn a_constant_gradient_keeps_the_step_at_the_learning_rate() {
        // With g constant, m̂ and v̂ are both g at every t, so the update is
        // exactly lr per step — the readable invariant this suite leans on.
        let mut model = Net::ones();
        let mut opt = Adam::new(0.1);
        for k in 1..=4 {
            step(&mut opt, &mut model);
            close(
                f64::from(model.at("trunk.weight")),
                1.0 - 0.1 * f64::from(k),
                1e-5,
            );
        }
        assert_eq!(opt.steps(), 4);
    }

    // ---- the per-parameter step clock -------------------------------------

    /// What a *global* step clock would produce for a parameter taking its
    /// first update at global step `t`: the fresh, near-zero moments divided by
    /// an almost-saturated bias correction.
    fn global_clock_step(lr: f64, beta1: f64, beta2: f64, t: i32) -> f64 {
        let m_hat = (1.0 - beta1) / (1.0 - beta1.powi(t));
        let v_hat = (1.0 - beta2) / (1.0 - beta2.powi(t));
        lr * m_hat / v_hat.sqrt()
    }

    #[test]
    fn a_late_joining_parameter_is_bias_corrected_as_a_first_update() {
        let mut model = Net::ones();
        let mut opt = Adam::new(0.1);
        // `head.bias` sits out a five-step warmup, then joins.
        model.head.bias.freeze();
        for _ in 0..5 {
            step(&mut opt, &mut model);
        }
        assert_eq!(opt.param_steps(&model.head.bias), 0);
        model.head.bias.unfreeze();
        step(&mut opt, &mut model);

        assert_eq!(opt.steps(), 6);
        // Its own clock says "first update", so it takes a full-size step…
        assert_eq!(opt.param_steps(&model.head.bias), 1);
        close(f64::from(model.at("head.bias")), 0.9, 1e-6);

        // …and that is emphatically not what a global clock would have done:
        // at t = 6 the correction factors would have shrunk it by about half.
        let global = global_clock_step(0.1, 0.9, 0.999, 6);
        close(global, 0.0522, 1e-3);
        assert!(
            (f64::from(model.at("head.bias")) - (1.0 - global)).abs() > 0.04,
            "the late joiner was corrected as if it had been there from step 0"
        );
        // The five veterans are on their own, older clock.
        assert_eq!(opt.param_steps(&model.trunk.weight), 6);
    }

    // ---- Adam vs AdamW ----------------------------------------------------

    #[test]
    fn decoupled_decay_is_the_only_difference_between_adam_and_adamw() {
        // Coupled: the decay term joins the gradient and is then normalized
        // away by the second moment, so the first step is still lr.
        let mut coupled_model = Net::ones();
        let mut coupled = Adam::new(0.1).weight_decay(0.5);
        assert!(!coupled.is_decoupled());
        step(&mut coupled, &mut coupled_model);
        close(f64::from(coupled_model.at("trunk.weight")), 0.9, 1e-6);

        // Decoupled: w ← w·(1 − lr·wd) = 0.95 before the −lr step.
        let mut decoupled_model = Net::ones();
        let mut decoupled = AdamW::new(0.1, 0.5);
        assert!(decoupled.is_decoupled());
        step(&mut decoupled, &mut decoupled_model);
        close(f64::from(decoupled_model.at("trunk.weight")), 0.85, 1e-6);
    }

    #[test]
    fn adamw_is_the_same_value_as_a_configured_adam() {
        let named = AdamW::new(3e-4, 0.1);
        let spelled_out = Adam::new(3e-4).decoupled(true).weight_decay(0.1);
        assert_eq!(named.lr(), spelled_out.lr());
        assert_eq!(named.is_decoupled(), spelled_out.is_decoupled());
        // Both are an `Adam`; there is no second implementation to diverge.
        let mut a = Net::ones();
        let mut b = Net::ones();
        let mut named = named;
        let mut spelled_out = spelled_out;
        step(&mut named, &mut a);
        step(&mut spelled_out, &mut b);
        assert_eq!(a.snapshot(), b.snapshot());
    }

    #[test]
    fn groups_keep_decay_off_biases_and_norm_gains() {
        let mut model = Net::ones();
        let mut opt = AdamW::new(0.1, 0.5).group(
            |path| path.ends_with("bias") || path.contains("norm"),
            |g| g.weight_decay(0.0),
        );
        step(&mut opt, &mut model);
        for path in ["trunk.bias", "head.bias", "norm.weight"] {
            close(f64::from(model.at(path)), 0.9, 1e-6);
        }
        for path in ["trunk.weight", "head.weight"] {
            close(f64::from(model.at(path)), 0.85, 1e-6);
        }
    }

    #[test]
    fn a_group_can_retune_the_betas_and_eps() {
        // β₁ = 0 makes the first moment the raw gradient; with β₂ = 0 too, the
        // direction is g/(|g| + eps) ≈ 1 — still one step of lr, but through a
        // different arithmetic path, which proves the group reached the update.
        let mut model = Net::ones();
        let mut opt = Adam::new(0.1).group(
            |path| path.starts_with("trunk."),
            |g| g.betas(0.0, 0.0).eps(1.0),
        );
        step(&mut opt, &mut model);
        // trunk: dir = 1/(1 + 1) = 0.5 -> 1 - 0.05.
        close(f64::from(model.at("trunk.weight")), 0.95, 1e-6);
        // head: the defaults -> 1 - 0.1.
        close(f64::from(model.at("head.weight")), 0.9, 1e-6);
    }

    // ---- the formula, against an independent implementation ---------------

    /// Adam on one scalar, written out in plain `f64` — no tensors, no shared
    /// code with the implementation under test. The reference the next two tests
    /// compare against.
    #[allow(clippy::too_many_arguments)]
    fn scalar_adam(
        mut w: f64,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        decoupled: bool,
        steps: i32,
    ) -> f64 {
        let (mut m, mut v) = (0.0f64, 0.0f64);
        for t in 1..=steps {
            let mut g = 2.0 * w; // dL/dw of w²
            if !decoupled {
                g += weight_decay * w;
            }
            m = beta1 * m + (1.0 - beta1) * g;
            v = beta2 * v + (1.0 - beta2) * g * g;
            let m_hat = m / (1.0 - beta1.powi(t));
            let v_hat = v / (1.0 - beta2.powi(t));
            if decoupled {
                w *= 1.0 - lr * weight_decay;
            }
            w -= lr * m_hat / (v_hat.sqrt() + eps);
        }
        w
    }

    #[test]
    fn the_update_matches_an_independent_scalar_adam() {
        for (label, decoupled) in [("coupled", false), ("decoupled", true)] {
            let mut model = Solo::new(1.5);
            let mut opt = Adam::new(0.05)
                .betas(0.85, 0.99)
                .eps(1e-7)
                .weight_decay(0.03)
                .decoupled(decoupled);
            for _ in 0..8 {
                let loss = model.square_loss(Mode::TRAIN).unwrap();
                opt.step(&mut model, loss.backward().unwrap()).unwrap();
            }
            let want = scalar_adam(1.5, 0.05, 0.85, 0.99, 1e-7, 0.03, decoupled, 8);
            // The implementation accumulates in f32, the reference in f64.
            assert!(
                (model.value() - want).abs() < 1e-5,
                "{label}: {} vs {want}",
                model.value()
            );
        }
    }

    #[test]
    fn a_late_joiner_follows_a_freshly_started_optimizers_trajectory() {
        // The per-parameter clock, stated as an equivalence rather than as a
        // single number: a parameter that starts updating at global step 5 must
        // follow exactly the trajectory a brand-new optimizer would have given
        // it. Under a global clock its first four steps would all be wrong.
        let mut late = Solo::new(1.5);
        late.w.freeze();
        let mut opt = Adam::new(0.05);
        let mut unrelated = Net::ones();
        for _ in 0..5 {
            // Runs the optimizer's own clock up without touching `late`.
            step(&mut opt, &mut unrelated);
        }
        late.w.unfreeze();
        for _ in 0..4 {
            let loss = late.square_loss(Mode::TRAIN).unwrap();
            opt.step(&mut late, loss.backward().unwrap()).unwrap();
        }

        assert_eq!(opt.steps(), 9);
        assert_eq!(opt.param_steps(&late.w), 4);
        let want = scalar_adam(1.5, 0.05, 0.9, 0.999, 1e-8, 0.0, false, 4);
        assert!(
            (late.value() - want).abs() < 1e-5,
            "{} vs {want}",
            late.value()
        );
    }

    // ---- the gate: convergence -------------------------------------------

    /// An out-of-range hyperparameter in a group that matches only *some*
    /// parameters must reject the whole step, leaving every parameter, every
    /// per-parameter clock, and `steps` untouched.
    ///
    /// The group ordering matters: `trunk.*` sorts before `head.*` in the walk,
    /// so validating inside the mutating pass would already have moved the
    /// trunk by the time the bad `head.*` group was reached — and no retry can
    /// undo that, because the trunk's bias-correction clock has advanced too.
    #[test]
    fn an_invalid_group_hyperparameter_rejects_the_whole_step() {
        for bad in [
            Adam::new(0.1).group(|path| path.starts_with("head."), |g| g.eps(0.0)),
            Adam::new(0.1).group(|path| path.starts_with("head."), |g| g.betas(1.0, 0.999)),
            Adam::new(0.1).group(
                |path| path.starts_with("head."),
                |g| g.weight_decay(f64::NAN),
            ),
            Adam::new(0.1).group(|path| path.starts_with("head."), |g| g.lr_scale(-1.0)),
        ] {
            let mut opt = bad;
            let mut model = Net::ones();
            let before = model.snapshot();

            let loss = model.unit_grad_loss(Mode::TRAIN).unwrap();
            let err = opt
                .step(&mut model, loss.backward().unwrap())
                .expect_err("an invalid group hyperparameter must reject the step");
            assert!(matches!(err, Error::InvalidArg { .. }), "{err}");

            assert_eq!(
                model.snapshot(),
                before,
                "a rejected step must not move any parameter"
            );
            assert_eq!(opt.steps(), 0, "a rejected step must not advance `steps`");
            assert_eq!(
                opt.param_steps(&model.trunk.weight),
                0,
                "a rejected step must not advance a parameter's own clock"
            );
        }
    }

    #[test]
    fn converges_on_a_tiny_regression() {
        let mut opt = Adam::new(0.05);
        assert_converges_on_a_tiny_regression(800, |model| {
            let loss = model.loss(&REGRESSION_XS, Mode::TRAIN).unwrap();
            opt.step(model, loss.backward().unwrap()).unwrap();
        });
    }

    // ---- the loudness gate ----------------------------------------------

    #[test]
    fn an_untraced_weight_is_missing_grad_and_no_moment_is_touched() {
        let mut model = Net::ones();
        let mut opt = Adam::new(0.1);
        step(&mut opt, &mut model);
        let before = model.snapshot();

        let loss = model.untraced_head_bias_loss(Mode::TRAIN).unwrap();
        let err = opt.step(&mut model, loss.backward().unwrap()).unwrap_err();
        assert!(
            matches!(&err, Error::MissingGrad { op: "step", path } if path == "head.bias"),
            "{err}"
        );
        // Nothing moved, and no clock advanced — so a caught mistake can be
        // fixed and the step retried without corrupting the trajectory.
        assert_eq!(model.snapshot(), before);
        assert_eq!(opt.steps(), 1);
        assert_eq!(opt.param_steps(&model.trunk.weight), 1);
    }

    #[test]
    fn a_rejected_step_leaves_parameters_moments_and_clocks_untouched() {
        assert_every_rejection_is_atomic(|| AdamW::new(0.1, 0.01).engine);
    }

    #[test]
    fn an_exhausted_clock_outranks_an_invalid_hyperparameter() {
        assert_an_exhausted_clock_outranks_an_invalid_hyperparameter(Adam::new(0.1).engine);
    }

    // ---- state persistence ----------------------------------------------

    #[test]
    fn state_round_trips_through_an_envelope_and_a_resumed_run_matches() {
        assert_state_round_trips(
            "adam-state",
            || AdamW::new(0.1, 0.01).betas(0.8, 0.99).eps(1e-7).engine,
            // Deliberately wrong hyperparameters. The wrong decay coupling is
            // the one thing loading cannot fix (see
            // `a_coupled_checkpoint_will_not_load_into_an_adamw`), so the
            // resumed optimizer declares `decoupled` and lets the file do the
            // rest.
            Adam::new(999.0).decoupled(true).engine,
        );
    }

    #[test]
    fn a_checkpoint_carries_both_moments_of_every_parameter() {
        let mut model = Net::ones();
        let mut opt = Adam::new(0.1);
        step(&mut opt, &mut model);
        let mut envelope = Envelope::new();
        opt.save_state(&model, &mut envelope).unwrap();
        // Five parameters, two moments each, under the reserved namespace.
        assert_eq!(envelope.tensors().len(), 10);
        assert!(envelope.tensors().contains_key("optim.trunk.weight.m"));
        assert!(envelope.tensors().contains_key("optim.trunk.weight.v"));
    }

    #[test]
    fn near_max_adam_clocks_preserve_moment_bytes_and_reject_atomically() {
        let mut model = Net::ones();
        let mut seeded = Adam::new(0.1);
        step(&mut seeded, &mut model);
        let mut saved = Envelope::new();
        seeded.save_state(&model, &mut saved).unwrap();
        let near_max = with_clocks(&saved, 0, u64::MAX - 1);

        let mut opt = Adam::new(9.0);
        opt.load_state(&model, &near_max).unwrap();
        let mut restored = Envelope::new();
        opt.save_state(&model, &mut restored).unwrap();
        assert_eq!(restored, near_max);
        step(&mut opt, &mut model);
        assert_eq!(opt.param_steps(&model.trunk.weight), u64::MAX);

        assert_next_step_is_refused(&mut opt.engine, &mut model, "parameter `");
    }

    #[test]
    fn exhausted_adam_parameter_clock_names_its_path_before_mutation() {
        let mut model = Net::ones();
        let mut opt = Adam::new(0.1);
        step(&mut opt, &mut model);
        opt.engine.set_clock(&model.trunk.weight, u64::MAX);
        model.trunk.weight.freeze();
        step(&mut opt, &mut model);
        assert_eq!(opt.param_steps(&model.trunk.weight), u64::MAX);
        model.trunk.weight.unfreeze();

        assert_next_step_is_refused(&mut opt.engine, &mut model, "`trunk.weight`");
    }

    #[test]
    fn saving_twice_into_one_envelope_is_rejected() {
        let mut model = Net::ones();
        let mut opt = Adam::new(0.1);
        step(&mut opt, &mut model);
        let mut envelope = Envelope::new();
        opt.save_state(&model, &mut envelope).unwrap();
        let msg = opt
            .save_state(&model, &mut envelope)
            .unwrap_err()
            .to_string();
        assert!(msg.contains("already carries optimizer state"), "{msg}");
    }

    #[test]
    fn a_coupled_checkpoint_will_not_load_into_an_adamw() {
        let mut model = Net::ones();
        let mut coupled = Adam::new(0.1).weight_decay(0.5);
        step(&mut coupled, &mut model);
        let mut envelope = Envelope::new();
        coupled.save_state(&model, &mut envelope).unwrap();

        // Adam and AdamW apply decay differently, so this is a rejection
        // rather than a silent reconfiguration of the resumed run.
        let msg = AdamW::new(0.1, 0.5)
            .load_state(&model, &envelope)
            .unwrap_err()
            .to_string();
        assert!(msg.contains("decoupled"), "{msg}");
        // The same state loads into the optimizer it was saved from.
        assert!(
            Adam::new(0.0)
                .weight_decay(0.0)
                .load_state(&model, &envelope)
                .is_ok()
        );
    }

    /// Rewrite one `hyper.*` line of a saved optimizer section.
    fn with_hyper(envelope: &Envelope, key: &str, value: &str) -> Envelope {
        let section = envelope.section("optimizer").unwrap();
        let patched: String = section
            .lines()
            .map(|line| {
                if line.starts_with(&format!("hyper.{key}=")) {
                    format!("hyper.{key}={value}")
                } else {
                    line.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        let mut out = Envelope::new();
        out.set_section("optimizer", &patched).unwrap();
        for (k, t) in envelope.tensors() {
            out.insert_tensor(k.clone(), t.clone());
        }
        out
    }

    #[test]
    fn a_hyperparameter_out_of_range_is_rejected_at_load_not_at_the_next_step() {
        let mut model = Net::ones();
        let mut opt = Adam::new(0.1);
        step(&mut opt, &mut model);
        let mut envelope = Envelope::new();
        opt.save_state(&model, &mut envelope).unwrap();

        // Adopting these and failing later would leave an optimizer that can
        // never step again and cannot be repaired, which is the opposite of
        // the all-or-nothing load `optim`'s module docs promise.
        for (key, value) in [("beta1", "nan"), ("beta2", "2"), ("eps", "-1")] {
            let broken = with_hyper(&envelope, key, value);
            let mut target = Adam::new(0.1);
            assert!(
                target.load_state(&model, &broken).is_err(),
                "hyper.{key}={value} must be refused by the load itself"
            );
            // Refused before anything was adopted: the optimizer still steps.
            step(&mut target, &mut model);
        }
    }

    #[test]
    fn a_decoupled_flag_that_is_neither_zero_nor_one_is_rejected() {
        let mut model = Net::ones();
        let mut opt = Adam::new(0.1);
        step(&mut opt, &mut model);
        let mut envelope = Envelope::new();
        opt.save_state(&model, &mut envelope).unwrap();

        // The writer only ever emits 0 or 1, so anything else is a malformed
        // file — and `!= 0.0` would have quietly loaded both of these as
        // AdamW.
        for value in ["0.5", "nan"] {
            let broken = with_hyper(&envelope, "decoupled", value);
            let msg = Adam::new(0.1)
                .load_state(&model, &broken)
                .unwrap_err()
                .to_string();
            assert!(msg.contains("neither 0 nor 1"), "{value}: {msg}");
        }
    }

    #[test]
    fn a_half_saved_moment_pair_is_rejected() {
        let mut model = Net::ones();
        let mut opt = Adam::new(0.1);
        step(&mut opt, &mut model);
        let mut full = Envelope::new();
        opt.save_state(&model, &mut full).unwrap();

        // Rebuild the envelope without the second moments.
        let mut half = Envelope::new();
        half.set_section("optimizer", full.section("optimizer").unwrap())
            .unwrap();
        for (key, tensor) in full.tensors() {
            // Not a file path: `.v` here is an optimizer-state tensor-key
            // suffix, so `Path::extension()` doesn't apply.
            #[allow(clippy::case_sensitive_file_extension_comparisons)]
            if !key.ends_with(".v") {
                half.insert_tensor(key.clone(), tensor.clone());
            }
        }
        let msg = Adam::new(0.1)
            .load_state(&model, &half)
            .unwrap_err()
            .to_string();
        assert!(msg.contains("missing the `v` moment"), "{msg}");
    }

    #[test]
    fn a_moment_of_the_wrong_shape_is_rejected() {
        let mut model = Net::ones();
        let mut opt = Adam::new(0.1);
        step(&mut opt, &mut model);
        let mut envelope = Envelope::new();
        opt.save_state(&model, &mut envelope).unwrap();

        let mut wrong = Envelope::new();
        wrong
            .set_section("optimizer", envelope.section("optimizer").unwrap())
            .unwrap();
        for (key, tensor) in envelope.tensors() {
            if key == "optim.trunk.weight.m" {
                wrong.insert_tensor(
                    key.clone(),
                    crate::persist::HostTensor::from_bytes(
                        crate::dtype::DType::F32,
                        vec![3],
                        vec![0u8; 12],
                    )
                    .unwrap(),
                );
            } else {
                wrong.insert_tensor(key.clone(), tensor.clone());
            }
        }
        let msg = Adam::new(0.1)
            .load_state(&model, &wrong)
            .unwrap_err()
            .to_string();
        assert!(msg.contains("`trunk.weight.m` has shape"), "{msg}");
    }
}
