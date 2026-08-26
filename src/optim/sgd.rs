//! [`Sgd`] — stochastic gradient descent with momentum and weight decay.

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
/// A group is declared with [`Sgd::group`] and states only its *differences*
/// from the optimizer's own settings:
///
/// ```
/// # use rstorch::optim::Sgd;
/// let opt = Sgd::new(0.1)
///     .momentum(0.9)
///     .weight_decay(1e-4)
///     // The standard recipe: no decay on biases and norm gains.
///     .group(|path| path.ends_with("bias") || path.contains("norm"),
///            |g| g.weight_decay(0.0));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SgdGroup {
    lr_scale: f64,
    momentum: f64,
    weight_decay: f64,
}

impl SgdGroup {
    /// Scale this group's learning rate relative to the optimizer's
    /// (discriminative learning rates).
    ///
    /// A *scale* rather than an absolute rate, so `set_lr` and the
    /// [`schedule`](super::schedule) functions keep working: the effective rate
    /// is always `optimizer lr × scale`.
    #[must_use]
    pub fn lr_scale(mut self, scale: f64) -> SgdGroup {
        self.lr_scale = scale;
        self
    }

    /// Momentum coefficient (`0.0` disables momentum, and with it the velocity
    /// buffer).
    #[must_use]
    pub fn momentum(mut self, momentum: f64) -> SgdGroup {
        self.momentum = momentum;
        self
    }

    /// L2 weight decay, added to the gradient (`0.0` disables it).
    #[must_use]
    pub fn weight_decay(mut self, weight_decay: f64) -> SgdGroup {
        self.weight_decay = weight_decay;
        self
    }
}

/// SGD's per-parameter buffers: its velocity, which only a momentum run keeps.
/// The step clock that rides alongside is `engine::ParamState`'s.
pub(crate) type Velocity = Option<Tensor>;

/// SGD as the shared [`Engine`] sees it. Plain SGD has nothing to configure
/// beyond its groups, so the rule itself carries no state.
#[derive(Clone, Copy)]
pub(crate) struct SgdRule;

impl Rule for SgdRule {
    type Hyper = SgdGroup;
    type Buffers = Velocity;

    const NAME: &'static str = "SGD";
    const KIND: &'static str = "sgd";
    const HYPERS: &'static [&'static str] = &["lr", "momentum", "weight_decay"];

    fn lr_scale(hyper: &SgdGroup) -> f64 {
        hyper.lr_scale
    }

    fn check(&self, _param: &Param, hyper: SgdGroup, lr: f64, _clock: u64) -> Result<()> {
        // Exactly the scalars the step will hand the kernel for this
        // parameter. SGD does not read its step clock: it has no bias
        // correction.
        crate::optim::validate::sgd_scalars(
            "step",
            lr,
            hyper.momentum,
            hyper.weight_decay,
            _param.value().dtype().accumulation_dtype(),
        )
    }

    /// The formula in [`Sgd`]'s docs.
    ///
    /// A backend with a fused `SgdStep` kernel runs it in one pass; where there
    /// is none the same arithmetic is spelled out in the public op vocabulary,
    /// which is the definition the two agree on. `weights` and `grad` arrive
    /// already widened to the accumulation dtype, and the value returned is
    /// narrowed back by the caller.
    fn update(&self, update: Update<'_, Self>) -> Result<(Tensor, Velocity)> {
        let Update {
            hyper,
            lr,
            weights,
            grad,
            previous,
            clock: _,
        } = update;
        let previous_velocity = previous.cloned().flatten();
        let scalars = [lr, hyper.momentum, hyper.weight_decay];
        // Keep backend views in a short scope. The fallback below moves
        // `grad`; no view may keep that borrow alive across the match.
        let fused = {
            let mut inputs = vec![weights.ready_view()?, grad.ready_view()?];
            if hyper.momentum != 0.0
                && let Some(velocity) = &previous_velocity
            {
                inputs.push(velocity.ready_view()?);
            }
            dispatch::backend(inputs[0].device()).fused(FusedOp::SgdStep, &inputs, &scalars)
        };

        match fused {
            Ok(outputs) => {
                let mut outputs = engine::fused_outputs(
                    Self::NAME,
                    outputs,
                    if hyper.momentum == 0.0 { 1 } else { 2 },
                    &weights,
                )?;
                let next = outputs.remove(0);
                let velocity = if hyper.momentum == 0.0 {
                    previous_velocity
                } else {
                    Some(outputs.remove(0))
                };
                Ok((next, velocity))
            }
            Err(Error::Unsupported { .. }) => {
                let mut g = grad;
                if hyper.weight_decay != 0.0 {
                    g = g.add(&weights.mul_scalar(hyper.weight_decay)?)?;
                }
                let direction = if hyper.momentum == 0.0 {
                    g
                } else {
                    // PyTorch's initialization: the first velocity *is* the
                    // gradient, so a momentum run and a plain run take the same
                    // first step.
                    match &previous_velocity {
                        Some(previous) => previous.mul_scalar(hyper.momentum)?.add(&g)?,
                        None => g,
                    }
                };
                let velocity = if hyper.momentum == 0.0 {
                    previous_velocity
                } else {
                    Some(direction.clone())
                };
                let next = weights.sub(&direction.mul_scalar(lr)?)?;
                Ok((next, velocity))
            }
            Err(error) => Err(error),
        }
    }

    fn hypers(&self, base: SgdGroup) -> Vec<(&'static str, f64)> {
        vec![
            ("momentum", base.momentum),
            ("weight_decay", base.weight_decay),
        ]
    }

    fn buffers(velocity: &Velocity) -> Vec<(&'static str, &Tensor)> {
        velocity.iter().map(|v| ("velocity", v)).collect()
    }

    fn adopt(&mut self, base: &mut SgdGroup, incoming: &Incoming) -> Result<()> {
        base.momentum = incoming.hyper("momentum")?;
        base.weight_decay = incoming.hyper("weight_decay")?;
        Ok(())
    }

    fn decode(&self, decode: Decode<'_, Self>) -> Result<Velocity> {
        let Decode {
            path,
            saved,
            value,
            incoming,
            base,
            groups,
        } = decode;
        let mut velocity = None;
        for (name, host) in &saved.buffers {
            match name.as_str() {
                "velocity" => {
                    velocity = Some(state::restore_buffer(host, value, path, name)?);
                }
                other => return Err(state::unknown_buffer(Self::KIND, path, other)),
            }
        }
        // Whether a velocity is *required* depends on the momentum in force for
        // this parameter's group, not on the base the file recorded. A group
        // that sets `momentum(0.0)` keeps no velocity however large the base is,
        // and a group that sets a non-zero momentum over a zero base keeps one —
        // checking the base alone rejects the first (a valid configuration that
        // then cannot be resumed at all) and waves the second through (the
        // truncated file this check exists to catch). Overrides are code and so
        // are the optimizer's own; only the base comes from the file.
        let saved_base = SgdGroup {
            momentum: incoming.hyper("momentum")?,
            ..base
        };
        let effective = groups.resolve_with_base(saved_base, path);
        if velocity.is_none() && effective.momentum != 0.0 && saved.clock > 0 {
            return Err(Error::persistence(format!(
                "optimizer state for `{path}` has no velocity buffer, but it was \
                     saved by a momentum run (momentum={}) after {} update(s), \
                     which always writes one",
                effective.momentum, saved.clock
            )));
        }
        Ok(velocity)
    }
}

/// Stochastic gradient descent.
///
/// The update, for each non-frozen parameter with a gradient:
///
/// ```text
/// g ← grad + weight_decay · w
/// v ← momentum · v + g            (v starts at g; skipped when momentum = 0)
/// w ← w − lr · v
/// ```
///
/// [`step`](Sgd::step) **consumes** the [`Grads`] by move — there is no
/// `zero_grad`, because there is no gradient state to clear — and a non-frozen
/// parameter with no gradient is [`Error::MissingGrad`](crate::Error::MissingGrad)
/// naming its path, never a silent skip.
///
/// ```
/// # use rstorch::nn::{Mode, Param};
/// # use rstorch::optim::Sgd;
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
/// let mut opt = Sgd::new(0.1);
///
/// let w = model.w.get(Mode::TRAIN);
/// let loss = w.mul(&w)?.sum_all()?;
/// opt.step(&mut model, loss.backward()?)?;
///
/// // d(w²)/dw = 2w = 2, so w ← 1 − 0.1 · 2.
/// assert_eq!(model.w.value().to_vec::<f32>()?, vec![0.8, 0.8]);
/// # Ok(())
/// # }
/// ```
pub struct Sgd {
    pub(crate) engine: Engine<SgdRule>,
}

impl Sgd {
    /// Plain SGD at learning rate `lr`: no momentum, no weight decay.
    pub fn new(lr: f64) -> Sgd {
        Sgd {
            engine: Engine::new(
                lr,
                SgdRule,
                SgdGroup {
                    lr_scale: 1.0,
                    momentum: 0.0,
                    weight_decay: 0.0,
                },
            ),
        }
    }

    /// Set the momentum coefficient for every parameter (see [`SgdGroup`]).
    #[must_use]
    pub fn momentum(mut self, momentum: f64) -> Sgd {
        self.engine.base_mut().momentum = momentum;
        self
    }

    /// Set the L2 weight decay for every parameter (see [`SgdGroup`]).
    #[must_use]
    pub fn weight_decay(mut self, weight_decay: f64) -> Sgd {
        self.engine.base_mut().weight_decay = weight_decay;
        self
    }

    /// Declare a parameter group: parameters whose dotted path satisfies
    /// `predicate` are updated with `configure(defaults)` instead of the
    /// optimizer's own hyperparameters.
    ///
    /// The **first** matching group wins, so declare the most specific
    /// predicate first. `configure` is re-evaluated on every step against the
    /// current defaults, so a later [`set_lr`](Sgd::set_lr) or builder call
    /// still reaches groups that do not override that particular field.
    #[must_use]
    pub fn group(
        mut self,
        predicate: impl Fn(&str) -> bool + Send + Sync + 'static,
        configure: impl Fn(SgdGroup) -> SgdGroup + Send + Sync + 'static,
    ) -> Sgd {
        self.engine.group(predicate, configure);
        self
    }

    /// The current base learning rate.
    pub fn lr(&self) -> f64 {
        self.engine.lr()
    }

    /// Set the base learning rate — the hook every
    /// [`schedule`](super::schedule) uses. Group [`lr_scale`](SgdGroup::lr_scale)
    /// factors apply on top, so a schedule moves every group together.
    pub fn set_lr(&mut self, lr: f64) {
        self.engine.set_lr(lr);
    }

    /// How many times [`step`](Sgd::step) has **succeeded** (the argument the
    /// [`schedule`](super::schedule) functions take). A rejected step leaves
    /// this untouched, so the schedule and the model never disagree about how
    /// far the run has got.
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
    /// and velocity buffers) into `envelope`, keyed by `model`'s dotted paths.
    ///
    /// Parameter *groups* are not saved: they are closures, reconstructed by
    /// building the optimizer the same way.
    ///
    /// # Errors
    ///
    /// [`Error::Persistence`](crate::Error::Persistence) if `envelope` already
    /// carries optimizer state, or if a host transfer fails.
    pub fn save_state(&self, model: &dyn Module, envelope: &mut Envelope) -> Result<()> {
        self.engine.save(model, envelope)
    }

    /// Restore the state [`save_state`](Sgd::save_state) wrote, resolving paths
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
    /// not match its parameter, or is a momentum run in which a parameter that
    /// has already been updated carries no velocity — a momentum step always
    /// writes one, so its absence is a truncated file, and resuming without it
    /// would silently restart that parameter's velocity from its next gradient.
    pub fn load_state(&mut self, model: &dyn Module, envelope: &Envelope) -> Result<()> {
        self.engine.load(model, envelope)
    }

    /// The number of updates `param` has received from this optimizer — its
    /// **own** step clock, which is not [`steps`](Sgd::steps) when the
    /// parameter was frozen for a while or joined the model late. SGD has no
    /// bias correction, so this is bookkeeping the caller can inspect (and
    /// state that survives a checkpoint) rather than something the update
    /// formula reads.
    ///
    /// `0` for a parameter this optimizer has never updated.
    pub fn param_steps(&self, param: &Param) -> u64 {
        self.engine.param_steps(param)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Mode, ModuleExt};
    use crate::optim::schedule;
    use crate::optim::testkit::{
        Net, REGRESSION_XS, Solo, assert_an_exhausted_clock_outranks_an_invalid_hyperparameter,
        assert_converges_on_a_tiny_regression, assert_every_rejection_is_atomic,
        assert_next_step_is_refused, assert_state_round_trips, close, with_clocks,
    };

    /// One update of `model` under `opt`, panicking on anything but success.
    fn step(opt: &mut Sgd, model: &mut Net) {
        let loss = model.unit_grad_loss(Mode::TRAIN).unwrap();
        opt.step(model, loss.backward().unwrap()).unwrap();
    }

    #[test]
    fn plain_sgd_subtracts_the_scaled_gradient() {
        let mut model = Net::ones();
        let mut opt = Sgd::new(0.1);
        step(&mut opt, &mut model);
        // Every gradient is 1, so every parameter moves by exactly -lr.
        for (path, value) in model.snapshot() {
            close(f64::from(value), 0.9, 1e-6);
            assert!(!path.is_empty());
        }
        assert_eq!(opt.steps(), 1);
        step(&mut opt, &mut model);
        close(f64::from(model.at("trunk.weight")), 0.8, 1e-6);
        assert_eq!(opt.param_steps(&model.trunk.weight), 2);
    }

    #[test]
    fn momentum_accumulates_the_velocity() {
        let mut model = Net::ones();
        let mut opt = Sgd::new(0.1).momentum(0.9);
        // v₁ = g = 1            -> w = 1 - 0.1·1    = 0.9
        step(&mut opt, &mut model);
        close(f64::from(model.at("trunk.weight")), 0.9, 1e-6);
        // v₂ = 0.9·1 + 1 = 1.9  -> w = 0.9 - 0.1·1.9 = 0.71
        step(&mut opt, &mut model);
        close(f64::from(model.at("trunk.weight")), 0.71, 1e-6);
    }

    #[test]
    fn weight_decay_is_added_to_the_gradient() {
        let mut model = Net::ones();
        let mut opt = Sgd::new(0.1).weight_decay(0.5);
        // g = 1 + 0.5·1 = 1.5 -> w = 1 - 0.15 = 0.85
        step(&mut opt, &mut model);
        close(f64::from(model.at("trunk.weight")), 0.85, 1e-6);
        // g = 1 + 0.5·0.85 = 1.425 -> w = 0.85 - 0.1425
        step(&mut opt, &mut model);
        close(f64::from(model.at("trunk.weight")), 0.7075, 1e-6);
    }

    #[test]
    fn groups_select_by_path_and_the_first_match_wins() {
        let mut model = Net::ones();
        let mut opt = Sgd::new(0.1)
            .weight_decay(0.5)
            // The standard recipe: no decay on biases and norm gains.
            .group(
                |path| path.ends_with("bias") || path.contains("norm"),
                |g| g.weight_decay(0.0),
            )
            // A wider predicate declared second: it matches every path, and
            // must still lose to the one above wherever that one matched.
            .group(|_| true, |g| g.weight_decay(2.0));
        step(&mut opt, &mut model);

        // No decay: 1 - 0.1·1 = 0.9.
        for path in ["trunk.bias", "head.bias", "norm.weight"] {
            close(f64::from(model.at(path)), 0.9, 1e-6);
        }
        // The second group's decay of 2.0: 1 - 0.1·(1 + 2·1) = 0.7.
        for path in ["trunk.weight", "head.weight"] {
            close(f64::from(model.at(path)), 0.7, 1e-6);
        }
    }

    #[test]
    fn lr_scale_is_relative_so_set_lr_moves_every_group() {
        let mut model = Net::ones();
        let mut opt = Sgd::new(0.2).group(|path| path.starts_with("trunk."), |g| g.lr_scale(0.1));
        step(&mut opt, &mut model);
        close(f64::from(model.at("trunk.weight")), 0.98, 1e-6); // 0.2 · 0.1
        close(f64::from(model.at("head.weight")), 0.8, 1e-6);

        // A schedule only ever touches the base rate; the scale rides along.
        opt.set_lr(0.5);
        assert_eq!(opt.lr(), 0.5);
        step(&mut opt, &mut model);
        close(f64::from(model.at("trunk.weight")), 0.93, 1e-6); // 0.98 - 0.05
        close(f64::from(model.at("head.weight")), 0.3, 1e-6);
    }

    #[test]
    fn a_schedule_drives_the_base_rate_through_set_lr() {
        let mut model = Net::ones();
        let mut opt = Sgd::new(0.1);
        // Two steps at 1.0, then the staircase drops by 10x.
        for _ in 0..4 {
            opt.set_lr(schedule::step_decay(0.1, 0.1, 2, opt.steps()));
            step(&mut opt, &mut model);
        }
        // 1 - 0.1 - 0.1 - 0.01 - 0.01
        close(f64::from(model.at("trunk.weight")), 0.78, 1e-6);
        close(opt.lr(), 0.01, 1e-12);
    }

    // ---- the formula, against an independent implementation ---------------

    /// SGD on one scalar in plain `f64`: no tensors, no shared code with the
    /// implementation under test.
    fn scalar_sgd(mut w: f64, lr: f64, momentum: f64, weight_decay: f64, steps: u32) -> f64 {
        let mut velocity: Option<f64> = None;
        for _ in 0..steps {
            let g = 2.0 * w + weight_decay * w; // dL/dw of w², plus L2 decay
            let direction = if momentum == 0.0 {
                g
            } else {
                // PyTorch's initialization: the first velocity *is* the gradient.
                let v = velocity.map_or(g, |previous| momentum * previous + g);
                velocity = Some(v);
                v
            };
            w -= lr * direction;
        }
        w
    }

    #[test]
    fn the_update_matches_an_independent_scalar_sgd() {
        // A gradient that moves with the parameter, so a mis-ordered decay or a
        // mis-initialized velocity cannot hide behind a constant.
        for momentum in [0.0, 0.9] {
            let mut model = Solo::new(1.5);
            let mut opt = Sgd::new(0.05).momentum(momentum).weight_decay(0.03);
            for _ in 0..8 {
                let loss = model.square_loss(Mode::TRAIN).unwrap();
                opt.step(&mut model, loss.backward().unwrap()).unwrap();
            }
            let want = scalar_sgd(1.5, 0.05, momentum, 0.03, 8);
            assert!(
                (model.value() - want).abs() < 1e-5,
                "momentum {momentum}: {} vs {want}",
                model.value()
            );
        }
    }

    // ---- the gate: a tiny regression loss must actually go down -----------

    #[test]
    fn converges_on_a_tiny_regression() {
        let mut opt = Sgd::new(0.1).momentum(0.5);
        assert_converges_on_a_tiny_regression(400, |model| {
            let loss = model.loss(&REGRESSION_XS, Mode::TRAIN).unwrap();
            opt.step(model, loss.backward().unwrap()).unwrap();
        });
    }

    // ---- missing-gradient check --------------------------------------------

    #[test]
    fn an_untraced_weight_is_missing_grad_and_nothing_moves() {
        let mut model = Net::ones();
        let mut opt = Sgd::new(0.1);
        let before = model.snapshot();

        // The forward read `head.bias.value()` instead of `.get(mode)`, so its
        // gradient never exists — the accidental-freezing bug, made loud.
        let loss = model.untraced_head_bias_loss(Mode::TRAIN).unwrap();
        let err = opt.step(&mut model, loss.backward().unwrap()).unwrap_err();
        assert!(
            matches!(&err, Error::MissingGrad { op: "step", path } if path == "head.bias"),
            "{err}"
        );
        assert!(err.to_string().contains("`head.bias`"), "{err}");

        // The pre-pass rejected it, so the model is exactly as it was — and the
        // step count did not advance either.
        assert_eq!(model.snapshot(), before);
        assert_eq!(opt.steps(), 0);
    }

    #[test]
    fn a_rejected_step_leaves_parameters_velocities_and_clocks_untouched() {
        assert_every_rejection_is_atomic(|| Sgd::new(0.1).momentum(0.9).weight_decay(0.01).engine);
    }

    #[test]
    fn an_exhausted_clock_outranks_an_invalid_hyperparameter() {
        assert_an_exhausted_clock_outranks_an_invalid_hyperparameter(
            Sgd::new(0.1).momentum(0.9).engine,
        );
    }

    #[test]
    fn an_explicitly_frozen_parameter_is_skipped_and_keeps_its_own_clock() {
        let mut model = Net::ones();
        let mut opt = Sgd::new(0.1);
        model.head.bias.freeze();
        for _ in 0..3 {
            step(&mut opt, &mut model);
        }
        close(f64::from(model.at("head.bias")), 1.0, 1e-6);
        assert_eq!(opt.param_steps(&model.head.bias), 0);

        // Unfrozen after a warmup: its own clock starts now, the optimizer's
        // does not restart.
        model.head.bias.unfreeze();
        step(&mut opt, &mut model);
        assert_eq!(opt.steps(), 4);
        assert_eq!(opt.param_steps(&model.head.bias), 1);
        assert_eq!(opt.param_steps(&model.trunk.weight), 4);
        close(f64::from(model.at("head.bias")), 0.9, 1e-6);
    }

    // ---- state persistence ------------------------------------------------

    #[test]
    fn state_round_trips_through_an_envelope_and_a_resumed_run_matches() {
        assert_state_round_trips(
            "sgd-state",
            || Sgd::new(0.1).momentum(0.9).weight_decay(0.01).engine,
            Sgd::new(999.0).engine,
        );
    }

    /// The "a momentum run always writes a velocity" integrity check must use
    /// the momentum in force for each parameter's *group*, not the saved base.
    ///
    /// Reading the base alone is wrong in both directions: a group that turns
    /// momentum off legitimately stores no velocity, and rejecting it makes a
    /// valid configuration impossible to checkpoint and resume; conversely a
    /// group that turns momentum *on* over a zero base does store one, and its
    /// absence is the truncated file the check exists to catch.
    #[test]
    fn the_velocity_integrity_check_follows_group_momentum() {
        // Direction 1: momentum off for biases must round-trip cleanly.
        let mut model = Net::ones();
        let mut opt = Sgd::new(0.1)
            .momentum(0.9)
            .group(|path| path.ends_with("bias"), |g| g.momentum(0.0));
        step(&mut opt, &mut model);
        let mut envelope = Envelope::new();
        opt.save_state(&model, &mut envelope).unwrap();

        let mut resumed = Sgd::new(0.1)
            .momentum(0.9)
            .group(|path| path.ends_with("bias"), |g| g.momentum(0.0));
        resumed
            .load_state(&model, &envelope)
            .expect("a group with momentum(0.0) must be resumable");

        // Direction 2: momentum on for weights over a zero base — a velocity is
        // written, so a file missing one must still be rejected.
        let mut model = Net::ones();
        let mut opt = Sgd::new(0.1).group(|path| path.ends_with("weight"), |g| g.momentum(0.9));
        step(&mut opt, &mut model);
        let mut envelope = Envelope::new();
        opt.save_state(&model, &mut envelope).unwrap();

        // Drop every velocity tensor, as a truncated file would.
        let mut stripped = Envelope::new();
        stripped
            .set_section(
                "optimizer",
                envelope.section("optimizer").unwrap().to_string(),
            )
            .unwrap();
        let mut dropped = 0;
        for (key, tensor) in envelope.tensors() {
            if key.ends_with("velocity") {
                dropped += 1;
            } else {
                stripped.insert_tensor(key.clone(), tensor.clone());
            }
        }
        assert!(dropped > 0, "the fixture must contain a velocity to drop");

        let mut resumed = Sgd::new(0.1).group(|path| path.ends_with("weight"), |g| g.momentum(0.9));
        let err = resumed
            .load_state(&model, &stripped)
            .expect_err("a missing velocity under group momentum must be rejected");
        assert!(
            err.to_string().contains("no velocity buffer"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn restore_uses_the_adopted_base_for_group_decoding() {
        let mut model = Net::ones();
        let mut source = Sgd::new(0.1).weight_decay(1.0).group(
            |path| path == "trunk.weight",
            |group| {
                if group.weight_decay > 0.5 {
                    group.momentum(0.9)
                } else {
                    group.momentum(0.0)
                }
            },
        );
        step(&mut source, &mut model);

        let mut envelope = Envelope::new();
        source.save_state(&model, &mut envelope).unwrap();
        let mut stripped = Envelope::new();
        stripped
            .set_section(
                "optimizer",
                envelope.section("optimizer").unwrap().to_string(),
            )
            .unwrap();
        for (key, tensor) in envelope.tensors() {
            if !key.ends_with("velocity") {
                stripped.insert_tensor(key.clone(), tensor.clone());
            }
        }

        let mut target = Sgd::new(0.1).weight_decay(0.0).group(
            |path| path == "trunk.weight",
            |group| {
                if group.weight_decay > 0.5 {
                    group.momentum(0.9)
                } else {
                    group.momentum(0.0)
                }
            },
        );
        let error = target
            .load_state(&model, &stripped)
            .expect_err("adopted weight decay must select the saved momentum group");
        assert!(
            error.to_string().contains("no velocity buffer"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn max_loaded_clock_rejection_does_not_replace_sgd_state() {
        let mut model = Net::ones();
        let mut opt = Sgd::new(0.1).momentum(0.9);
        step(&mut opt, &mut model);
        let mut before = Envelope::new();
        opt.save_state(&model, &mut before).unwrap();
        let hostile = with_clocks(&before, u64::MAX, 1);
        let model_before = model.snapshot();

        let err = opt.load_state(&model, &hostile).unwrap_err();
        assert!(matches!(err, Error::Persistence { .. }), "{err}");
        assert!(err.to_string().contains("`steps`"), "{err}");

        let mut after = Envelope::new();
        opt.save_state(&model, &mut after).unwrap();
        assert_eq!(after, before);
        assert_eq!(model.snapshot(), model_before);
    }

    #[test]
    fn near_max_sgd_clocks_advance_once_then_reject_without_mutation() {
        let mut model = Net::ones();
        let mut seeded = Sgd::new(0.1).momentum(0.9);
        step(&mut seeded, &mut model);
        let mut saved = Envelope::new();
        seeded.save_state(&model, &mut saved).unwrap();
        let near_max = with_clocks(&saved, 0, u64::MAX - 1);

        let mut opt = Sgd::new(9.0);
        opt.load_state(&model, &near_max).unwrap();
        assert_eq!(opt.param_steps(&model.trunk.weight), u64::MAX - 1);
        step(&mut opt, &mut model);
        assert_eq!(opt.param_steps(&model.trunk.weight), u64::MAX);

        assert_next_step_is_refused(&mut opt.engine, &mut model, "parameter `");
    }

    #[test]
    fn exhausted_sgd_global_clock_is_a_structured_error_before_mutation() {
        let mut model = Net::ones();
        let mut opt = Sgd::new(0.1).momentum(0.9);
        step(&mut opt, &mut model);
        opt.engine.set_steps(u64::MAX);

        assert_next_step_is_refused(&mut opt.engine, &mut model, "SGD global step clock");
    }

    #[test]
    fn a_velocity_less_run_still_round_trips() {
        // Momentum 0 keeps no buffer at all, so the section carries clocks and
        // nothing else — the encoding must survive that.
        let mut model = Net::ones();
        let mut opt = Sgd::new(0.25);
        step(&mut opt, &mut model);
        let mut envelope = Envelope::new();
        opt.save_state(&model, &mut envelope).unwrap();
        assert!(envelope.tensors().is_empty());

        let mut resumed = Sgd::new(0.0);
        resumed.load_state(&model, &envelope).unwrap();
        close(resumed.lr(), 0.25, 1e-12);
        assert_eq!(resumed.param_steps(&model.trunk.weight), 1);
    }

    #[test]
    fn a_momentum_run_missing_its_velocity_is_rejected() {
        // A momentum step always writes a velocity, so a clock without one is a
        // truncated file. Accepting it would restart that parameter's velocity
        // from its next gradient and quietly change the trajectory — the same
        // failure class as a missing gradient, so it gets the same treatment.
        let mut model = Net::ones();
        let mut opt = Sgd::new(0.1).momentum(0.9);
        step(&mut opt, &mut model);
        let mut full = Envelope::new();
        opt.save_state(&model, &mut full).unwrap();

        let mut truncated = Envelope::new();
        truncated
            .set_section("optimizer", full.section("optimizer").unwrap())
            .unwrap();
        for (key, tensor) in full.tensors() {
            if key != "optim.trunk.weight.velocity" {
                truncated.insert_tensor(key.clone(), tensor.clone());
            }
        }
        let msg = Sgd::new(0.1)
            .load_state(&model, &truncated)
            .unwrap_err()
            .to_string();
        assert!(msg.contains("`trunk.weight` has no velocity"), "{msg}");

        // The same file is perfectly valid for a momentum-free run, which never
        // had a velocity to lose.
        let mut plain_model = Net::ones();
        let mut plain = Sgd::new(0.1);
        step(&mut plain, &mut plain_model);
        let mut plain_envelope = Envelope::new();
        plain.save_state(&plain_model, &mut plain_envelope).unwrap();
        assert!(
            Sgd::new(0.0)
                .load_state(&plain_model, &plain_envelope)
                .is_ok()
        );
    }

    #[test]
    fn a_step_leaves_no_autograd_history_in_a_parameter() {
        // The invariant the whole design rests on: a `Param`'s value is a plain
        // value, so `Param::get` under a non-recording mode hands out a
        // graph-less tensor and no step can leak a graph into the next one.
        let mut model = Net::ones();
        let mut opt = Sgd::new(0.1).momentum(0.9).weight_decay(0.01);
        step(&mut opt, &mut model);
        step(&mut opt, &mut model);
        for (path, value) in model.state_dict().unwrap() {
            assert!(value.backward().is_err(), "graph survived into `{path}`");
        }
        assert!(model.trunk.weight.get(Mode::EVAL).backward().is_err());
    }

    #[test]
    fn an_untouched_parameter_saves_no_state() {
        // A parameter the optimizer has never updated has no clock and no
        // buffer, so a resumed run bias-corrects it as a first update.
        let mut model = Net::ones();
        let mut opt = Sgd::new(0.1).momentum(0.9);
        model.head.bias.freeze();
        step(&mut opt, &mut model);

        let mut envelope = Envelope::new();
        opt.save_state(&model, &mut envelope).unwrap();
        let section = envelope.section("optimizer").unwrap();
        assert!(section.contains("clock.trunk.weight=1"), "{section}");
        assert!(!section.contains("clock.head.bias"), "{section}");
        assert!(!envelope.tensors().contains_key("optim.head.bias.velocity"));
    }

    #[test]
    fn loading_an_adam_checkpoint_into_an_sgd_is_rejected() {
        let mut model = Net::ones();
        let mut adam = super::super::Adam::new(0.1);
        let loss = model.unit_grad_loss(Mode::TRAIN).unwrap();
        adam.step(&mut model, loss.backward().unwrap()).unwrap();
        let mut envelope = Envelope::new();
        adam.save_state(&model, &mut envelope).unwrap();

        let msg = Sgd::new(0.1)
            .load_state(&model, &envelope)
            .unwrap_err()
            .to_string();
        assert!(msg.contains("saved by `adam`"), "{msg}");
    }

    #[test]
    fn state_for_a_different_model_is_rejected() {
        let mut model = Net::ones();
        let mut opt = Sgd::new(0.1).momentum(0.9);
        step(&mut opt, &mut model);
        let mut envelope = Envelope::new();
        opt.save_state(&model, &mut envelope).unwrap();

        // A model with different paths: the checkpoint belongs elsewhere.
        #[derive(rstorch::Module)]
        struct Other {
            elsewhere: Param,
        }
        let other = Other {
            elsewhere: Param::new(crate::optim::testkit::t(&[1.0])),
        };
        let msg = Sgd::new(0.1)
            .load_state(&other, &envelope)
            .unwrap_err()
            .to_string();
        assert!(msg.contains("which this model does not have"), "{msg}");
    }
}
