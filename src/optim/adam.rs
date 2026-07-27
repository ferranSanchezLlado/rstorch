//! [`Adam`] and [`AdamW`] — one implementation, two names (exploration §4.4).

use std::collections::HashMap;

use crate::autograd::{GradKey, Grads};
use crate::backend::{FusedOp, dispatch};
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::nn::{Module, Param};
use crate::persist::Envelope;
use crate::storage::Storage;
use crate::tensor::Tensor;

use super::engine::{self, Groups};
use super::state::{self, OutgoingParam};

/// The `kind` tag in a saved state section — the same for both names, because
/// they are the same optimizer.
const KIND: &str = "adam";
const HYPERS: [&str; 6] = ["lr", "beta1", "beta2", "eps", "weight_decay", "decoupled"];

fn fused_outputs(outputs: Vec<Storage>, like: &Tensor) -> Result<[Tensor; 3]> {
    if outputs.len() != 3 {
        return Err(Error::Backend {
            op: "step",
            msg: format!("fused Adam returned {} outputs, expected 3", outputs.len()),
        });
    }
    let mut tensors = Vec::with_capacity(3);
    for (index, storage) in outputs.into_iter().enumerate() {
        if storage.dtype() != like.dtype()
            || storage.device() != like.device()
            || storage.len() != like.num_elements()
        {
            return Err(Error::Backend {
                op: "step",
                msg: format!(
                    "fused Adam output {index} has dtype {}, device {}, and {} elements; \
                     expected dtype {}, device {}, and shape {}",
                    storage.dtype(),
                    storage.device(),
                    storage.len(),
                    like.dtype(),
                    like.device(),
                    like.shape()
                ),
            });
        }
        tensors.push(Tensor::from_parts(
            storage,
            Layout::contiguous(like.shape().clone())?,
        ));
    }
    tensors.try_into().map_err(|_| unreachable!())
}

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

/// Per-parameter state: the two moment estimates and **this parameter's own
/// step clock**.
struct AdamState {
    clock: u64,
    m: Tensor,
    v: Tensor,
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
/// `t` is **per parameter**, not a global counter (exploration §4.4). A
/// parameter that joins the update late — it was frozen for a warmup phase, or
/// a new head was attached to a pretrained trunk — starts at `t = 1`, so its
/// first update is bias-corrected as a first update. A global clock would
/// divide a fresh, near-zero moment estimate by an almost-saturated correction
/// factor and take a step several times too small.
///
/// # Adam vs AdamW
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
    lr: f64,
    decoupled: bool,
    groups: Groups<AdamGroup>,
    state: HashMap<GradKey, AdamState>,
    steps: u64,
}

impl Adam {
    /// Adam at learning rate `lr` with the usual defaults
    /// (β = (0.9, 0.999), eps = 1e-8, no weight decay).
    pub fn new(lr: f64) -> Adam {
        Adam {
            lr,
            decoupled: false,
            groups: Groups::new(AdamGroup {
                lr_scale: 1.0,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            }),
            state: HashMap::new(),
            steps: 0,
        }
    }

    /// Set the moment decay rates for every parameter (see [`AdamGroup`]).
    #[must_use]
    pub fn betas(mut self, beta1: f64, beta2: f64) -> Adam {
        let base = self.groups.base_mut();
        base.beta1 = beta1;
        base.beta2 = beta2;
        self
    }

    /// Set the numerical-stability term for every parameter.
    #[must_use]
    pub fn eps(mut self, eps: f64) -> Adam {
        self.groups.base_mut().eps = eps;
        self
    }

    /// Set the weight decay for every parameter. Whether it is coupled or
    /// decoupled is [`decoupled`](Adam::decoupled)'s business.
    #[must_use]
    pub fn weight_decay(mut self, weight_decay: f64) -> Adam {
        self.groups.base_mut().weight_decay = weight_decay;
        self
    }

    /// Choose where weight decay enters: `true` applies it straight to the
    /// parameter (AdamW), `false` folds it into the gradient (classic Adam).
    #[must_use]
    pub fn decoupled(mut self, decoupled: bool) -> Adam {
        self.decoupled = decoupled;
        self
    }

    /// Whether weight decay is decoupled (i.e. whether this is an AdamW).
    pub fn is_decoupled(&self) -> bool {
        self.decoupled
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
        self.groups.push(predicate, configure);
        self
    }

    /// The current base learning rate.
    pub fn lr(&self) -> f64 {
        self.lr
    }

    /// Set the base learning rate — the hook every
    /// [`schedule`](super::schedule) uses. Group
    /// [`lr_scale`](AdamGroup::lr_scale) factors apply on top, so a schedule
    /// moves every group together.
    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// How many times [`step`](Adam::step) has **succeeded** (the argument the
    /// [`schedule`](super::schedule) functions take); a rejected step leaves it
    /// untouched. Individual parameters may have taken fewer — see the
    /// per-parameter clocks in the type docs.
    pub fn steps(&self) -> u64 {
        self.steps
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
        let Adam {
            lr,
            decoupled,
            groups,
            state,
            steps,
        } = self;
        let base_lr = *lr;
        let decoupled = *decoupled;
        engine::apply("step", model, grads, |path, param, grad| {
            let hyper = groups.resolve(path);
            let lr = base_lr * hyper.lr_scale;
            let dtype = param.value().dtype();
            // Wide moments: an f16/bf16 parameter accumulates in f32 and is
            // narrowed back exactly once, at the end.
            let acc = engine::accum_dtype(dtype);
            let weights = param.value().to_dtype(acc)?;
            let grad = grad.to_dtype(acc)?;

            let previous = state.get(&param.grad_key());
            let next_clock = previous.map_or(1, |entry| entry.clock + 1);
            let (previous_m, previous_v) = match previous {
                Some(entry) => (entry.m.clone(), entry.v.clone()),
                None => {
                    let zeros = Tensor::zeros(weights.dims(), acc, &weights.device())?;
                    (zeros.clone(), zeros)
                }
            };
            // This parameter's own clock, so a late joiner is bias-corrected
            // as a first update rather than as step `steps`.
            let t = i32::try_from(next_clock).unwrap_or(i32::MAX);
            let correction1 = 1.0 - hyper.beta1.powi(t);
            let correction2 = 1.0 - hyper.beta2.powi(t);
            let scalars = [
                lr,
                hyper.beta1,
                hyper.beta2,
                hyper.eps,
                hyper.weight_decay,
                correction1,
                correction2,
                f64::from(u8::from(decoupled)),
            ];
            let inputs = [
                weights.view(),
                grad.view(),
                previous_m.view(),
                previous_v.view(),
            ];

            let [next, next_m, next_v] = match dispatch::backend(weights.device()).fused(
                FusedOp::AdamStep,
                &inputs,
                &scalars,
            ) {
                Ok(outputs) => fused_outputs(outputs, &weights)?,
                Err(Error::Unsupported { .. }) => {
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
            param.set(next.to_dtype(dtype)?)?;
            state.insert(
                param.grad_key(),
                AdamState {
                    clock: next_clock,
                    m: next_m,
                    v: next_v,
                },
            );
            Ok(())
        })?;
        *steps += 1;
        Ok(())
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
    pub fn save_state(&self, model: &dyn Module, envelope: &mut Envelope) -> Result<()> {
        let base = *self.groups.base();
        let paths = engine::param_paths(model);
        let outgoing: Vec<OutgoingParam<'_>> = paths
            .iter()
            .filter_map(|(path, key)| {
                let entry = self.state.get(key)?;
                Some(OutgoingParam {
                    path,
                    clock: entry.clock,
                    buffers: vec![("m", &entry.m), ("v", &entry.v)],
                })
            })
            .collect();
        state::save(
            envelope,
            KIND,
            &[
                ("lr", self.lr),
                ("beta1", base.beta1),
                ("beta2", base.beta2),
                ("eps", base.eps),
                ("weight_decay", base.weight_decay),
                ("decoupled", f64::from(u8::from(self.decoupled))),
            ],
            self.steps,
            &outgoing,
        )
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
    /// AdamW are different algorithms, so that is a rejection rather than a
    /// silent reconfiguration.
    pub fn load_state(&mut self, model: &dyn Module, envelope: &Envelope) -> Result<()> {
        let incoming = state::load(envelope, KIND)?;
        incoming.expect_hypers(&HYPERS)?;
        let saved_decoupled = incoming.hyper("decoupled")? != 0.0;
        if saved_decoupled != self.decoupled {
            return Err(Error::Persistence {
                msg: format!(
                    "optimizer state has decoupled={saved_decoupled}, loaded into an \
                     optimizer with decoupled={} (Adam and AdamW apply weight decay \
                     differently; build the one the checkpoint was saved from)",
                    self.decoupled
                ),
            });
        }
        let values = engine::param_values(model);
        let keys: HashMap<String, GradKey> = engine::param_paths(model).into_iter().collect();

        let mut restored = HashMap::new();
        for (path, saved) in incoming.params() {
            let (key, value) = state::locate(&keys, &values, path)?;
            let mut m = None;
            let mut v = None;
            for (name, host) in &saved.buffers {
                match name.as_str() {
                    "m" => m = Some(state::restore_buffer(host, value, path, name)?),
                    "v" => v = Some(state::restore_buffer(host, value, path, name)?),
                    other => return Err(state::unknown_buffer(KIND, path, other)),
                }
            }
            let missing = match (m, v) {
                (Some(m), Some(v)) => {
                    restored.insert(
                        key,
                        AdamState {
                            clock: saved.clock,
                            m,
                            v,
                        },
                    );
                    continue;
                }
                (None, _) => "m",
                (_, None) => "v",
            };
            return Err(Error::Persistence {
                msg: format!(
                    "optimizer state for `{path}` is missing the `{missing}` moment \
                     (Adam keeps both or neither)"
                ),
            });
        }

        self.lr = incoming.hyper("lr")?;
        let (beta1, beta2) = (incoming.hyper("beta1")?, incoming.hyper("beta2")?);
        let (eps, weight_decay) = (incoming.hyper("eps")?, incoming.hyper("weight_decay")?);
        let base = self.groups.base_mut();
        base.beta1 = beta1;
        base.beta2 = beta2;
        base.eps = eps;
        base.weight_decay = weight_decay;
        self.steps = incoming.steps();
        self.state = restored;
        Ok(())
    }

    /// The number of updates `param` has received from this optimizer — its
    /// own bias-correction clock `t`, which is **not** [`steps`](Adam::steps)
    /// when the parameter was frozen for a while or joined the model late.
    ///
    /// `0` for a parameter this optimizer has never updated (including one it
    /// has never seen).
    pub fn param_steps(&self, param: &Param) -> u64 {
        self.state
            .get(&param.grad_key())
            .map_or(0, |entry| entry.clock)
    }
}

/// **AdamW**: Adam with *decoupled* weight decay (Loshchilov & Hutter, 2019).
///
/// One implementation, two names (exploration §4.4). The decay term is the only
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
    use crate::optim::testkit::{Affine, Net, Solo, close, scalar, tmpdir, values};
    use crate::persist::Limits;

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

    #[test]
    fn converges_on_a_tiny_regression() {
        let xs = [-1.0f32, 0.0, 1.0, 2.0];
        let mut model = Affine::zeros();
        let mut opt = Adam::new(0.05);

        let first = scalar(&model.loss(&xs, Mode::TRAIN).unwrap());
        let mut latest = first;
        for _ in 0..800 {
            let loss = model.loss(&xs, Mode::TRAIN).unwrap();
            opt.step(&mut model, loss.backward().unwrap()).unwrap();
            latest = scalar(&model.loss(&xs, Mode::TRAIN).unwrap());
        }
        assert!(latest < first, "loss rose: {first} -> {latest}");
        assert!(latest < 1e-4, "did not converge: {latest}");
        close(f64::from(values(model.weight.value())[0]), 3.0, 1e-2);
        close(f64::from(values(model.bias.value())[0]), 2.0, 1e-2);
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
            matches!(&err, Error::MissingGrad { path } if path == "head.bias"),
            "{err}"
        );
        // Nothing moved, and no clock advanced — so a caught mistake can be
        // fixed and the step retried without corrupting the trajectory.
        assert_eq!(model.snapshot(), before);
        assert_eq!(opt.steps(), 1);
        assert_eq!(opt.param_steps(&model.trunk.weight), 1);
    }

    // ---- state persistence ----------------------------------------------

    #[test]
    fn state_round_trips_through_an_envelope_and_a_resumed_run_matches() {
        let dir = tmpdir("adam-state");
        let path = dir.join("run.rstorch");

        let mut reference_model = Net::ones();
        let mut reference = AdamW::new(0.1, 0.01).betas(0.8, 0.99).eps(1e-7);
        for _ in 0..6 {
            step(&mut reference, &mut reference_model);
        }

        let mut model = Net::ones();
        let mut opt = AdamW::new(0.1, 0.01).betas(0.8, 0.99).eps(1e-7);
        for _ in 0..3 {
            step(&mut opt, &mut model);
        }
        let mut envelope = Envelope::new();
        opt.save_state(&model, &mut envelope).unwrap();
        // Both moments per parameter, under the reserved namespace.
        assert_eq!(envelope.tensors().len(), 10);
        assert!(envelope.tensors().contains_key("optim.trunk.weight.m"));
        assert!(envelope.tensors().contains_key("optim.trunk.weight.v"));
        envelope.save(&path, &Limits::defaults()).unwrap();

        let loaded = Envelope::load(&path, &Limits::defaults()).unwrap();
        // Deliberately wrong hyperparameters, and the wrong decay coupling is
        // the one thing that cannot be fixed by loading (see below), so the
        // resumed optimizer declares `decoupled` and lets the file do the rest.
        let mut resumed = Adam::new(999.0).decoupled(true);
        resumed.load_state(&model, &loaded).unwrap();
        close(resumed.lr(), 0.1, 1e-12);
        assert_eq!(resumed.steps(), 3);
        assert_eq!(resumed.param_steps(&model.trunk.weight), 3);
        for _ in 0..3 {
            step(&mut resumed, &mut model);
        }

        assert_eq!(model.snapshot(), reference_model.snapshot());
        std::fs::remove_dir_all(&dir).unwrap();
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
