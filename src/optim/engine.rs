//! The machinery both optimizers share: path-predicate parameter groups, the
//! parameter walk that makes the `MissingGrad` check loud, and the step
//! skeleton around each optimizer's update formula.
//!
//! `Sgd` and `Adam` differ only in their per-parameter update formula and their
//! moment state; everything around it — resolving hyperparameters for a dotted
//! path, validating the gradient set, advancing step clocks, widening and
//! narrowing values, and swapping them in — lives here so both behave
//! identically. What each optimizer supplies is two closures: a scalar range
//! check and the formula itself (see [`step`]).

use std::collections::{HashMap, HashSet};

use crate::autograd::{GradKey, Grads};
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::nn::visit::{Leaf, LeafMut, visit_all, visit_all_mut};
use crate::nn::{Module, Param};
use crate::storage::Storage;
use crate::tensor::Tensor;

/// A parameter-path predicate: given a dotted visitor path
/// (`blocks.3.attn.qkv.weight`), does this group apply?
pub(crate) type PathPredicate = Box<dyn Fn(&str) -> bool + Send + Sync>;

/// A group's override: the group's hyperparameters as a function of the
/// optimizer's base hyperparameters. Stored as a function (not a snapshot) so
/// a group states only its *differences* and later changes to the base — a
/// `set_lr`, a builder call made after the group — still reach it.
type Override<H> = Box<dyn Fn(H) -> H + Send + Sync>;

/// Base hyperparameters plus path-predicate overrides.
///
/// The first predicate that matches a path wins; a path no predicate matches
/// gets the base hyperparameters unchanged.
pub(crate) struct Groups<H> {
    base: H,
    overrides: Vec<(PathPredicate, Override<H>)>,
}

impl<H: Clone> Groups<H> {
    /// Start from `base` with no groups.
    pub(crate) fn new(base: H) -> Groups<H> {
        Groups {
            base,
            overrides: Vec::new(),
        }
    }

    /// The base hyperparameters (what an unmatched path gets).
    pub(crate) fn base(&self) -> &H {
        &self.base
    }

    /// Mutate the base hyperparameters (the optimizer's own builder methods).
    pub(crate) fn base_mut(&mut self) -> &mut H {
        &mut self.base
    }

    /// Append a group: paths matching `predicate` get `configure(base)`.
    pub(crate) fn push(
        &mut self,
        predicate: impl Fn(&str) -> bool + Send + Sync + 'static,
        configure: impl Fn(H) -> H + Send + Sync + 'static,
    ) {
        self.overrides
            .push((Box::new(predicate), Box::new(configure)));
    }

    /// The hyperparameters in force for `path` — the first matching group's
    /// override applied to the base, or the base itself.
    pub(crate) fn resolve(&self, path: &str) -> H {
        self.resolve_with_base(self.base.clone(), path)
    }

    /// [`resolve`](Self::resolve) against a caller-supplied base instead of the
    /// stored one.
    ///
    /// Group predicates and overrides are *code*, so they are never persisted;
    /// a checkpoint carries only the base hyperparameters. Restoring one
    /// therefore needs the effective value for a path under the **saved** base
    /// but the **current** overrides, without mutating the optimizer before its
    /// validation has finished.
    pub(crate) fn resolve_with_base(&self, base: H, path: &str) -> H {
        for (matches, configure) in &self.overrides {
            if matches(path) {
                return configure(base);
            }
        }
        base
    }
}

/// One parameter's optimizer state: its own step clock plus whatever moment
/// buffers the optimizer keeps (SGD's optional velocity, Adam's `m`/`v` pair).
///
/// The clock lives out here rather than inside each optimizer's buffers so that
/// advancing it, persisting it and reporting it are written once.
pub(crate) struct ParamState<B> {
    pub(crate) clock: u64,
    pub(crate) buffers: B,
}

/// An optimizer's per-parameter state, keyed by gradient identity rather than
/// by path: a parameter that moves in the module tree keeps its own moments.
pub(crate) type States<B> = HashMap<GradKey, ParamState<B>>;

/// The number of updates `param` has had from an optimizer holding `state`, and
/// `0` for one it has never updated.
pub(crate) fn param_steps<B>(state: &States<B>, param: &Param) -> u64 {
    state.get(&param.grad_key()).map_or(0, |entry| entry.clock)
}

/// The validation walk both optimizers run *before* the first `Param::set`,
/// where `kind` names the optimizer ("SGD"/"Adam") in the clock rejection.
///
/// For every non-frozen parameter, in walk order: its step clock — absent from
/// `state` for a parameter this optimizer has never updated — must have room
/// for one more update, and `check`, the optimizer's own scalar range check,
/// must accept the hyperparameters resolved for that path together with the
/// clock value the step would give it.
///
/// Hyperparameters are range-checked here, before a single `Param::set`. The
/// kernel checks them too, but it runs once per parameter *during* the mutating
/// walk, so an out-of-range value in a group that matches only some parameters
/// would stop the step half-applied — with the already-updated parameters'
/// clocks advanced and `steps` not, which no retry can repair. `optim`'s module
/// docs promise the opposite.
///
/// An exhausted clock outranks an invalid hyperparameter, and the whole walk
/// runs before either is reported: a clock rejection wins even when the
/// parameter that carries it is visited *after* the one whose scalars were
/// refused. Which of the two a caller sees is observable, so the order is part
/// of the contract rather than an accident.
fn prepass<H: Clone, B>(
    kind: &'static str,
    model: &dyn Module,
    groups: &Groups<H>,
    state: &States<B>,
    mut check: impl FnMut(&Param, H, u64) -> Result<()>,
) -> Result<()> {
    let mut exhausted = None;
    let mut invalid = None;
    visit_all(model, &mut |path, leaf| {
        let Leaf::Param(param) = leaf else {
            return;
        };
        if param.is_frozen() {
            return;
        }
        let clock = state.get(&param.grad_key()).map(|entry| entry.clock);
        if clock.is_some_and(|clock| clock.checked_add(1).is_none()) {
            exhausted.get_or_insert_with(|| path.to_string());
            return;
        }
        if invalid.is_some() {
            return;
        }
        if let Err(error) = check(param, groups.resolve(path), clock.unwrap_or(0) + 1) {
            invalid = Some(error);
        }
    });
    if let Some(path) = exhausted {
        return Err(Error::InvalidArg {
            op: "step",
            msg: format!(
                "{kind} step clock for parameter `{path}` cannot be advanced past u64::MAX"
            ),
        });
    }
    if let Some(error) = invalid {
        return Err(error);
    }
    Ok(())
}

/// Drive one optimizer step over `model`, consuming `grads`.
///
/// Three passes, because a half-applied step is exactly the silent-corruption
/// class this library exists to remove:
///
/// 1. **Validate the read-only walk.** Every non-frozen parameter must be a
///    float tensor with a gradient of matching shape, dtype and device, and no
///    parameter may be visited twice. The first violation is returned and
///    *nothing* has been modified — in particular a non-frozen parameter with
///    no gradient is [`Error::MissingGrad`] naming its path, never a silent
///    skip.
/// 2. **Check the mutable walk reaches exactly the same parameters** — a dry
///    pass that touches no value. A hand-written `Module` whose `visit_mut`
///    forgets a leaf would otherwise leave that parameter silently untrained,
///    which is the same bug class as a missing gradient and gets the same loud
///    treatment (the sibling of the check in
///    [`nn::load_state_dict`](crate::nn::load_state_dict)).
/// 3. **Apply.** `update` is called with each parameter's path, the parameter,
///    and its gradient (moved out of `grads`); passes 1–2 have already proved
///    the lookups and shapes, so only a backend failure can stop it here.
///
/// Frozen parameters are skipped legitimately, because freezing is explicit.
/// Gradient entries for anything the walk does not reach (a traced input, a
/// parameter of another model) are dropped with `grads`.
fn apply(
    op: &'static str,
    model: &mut dyn Module,
    grads: Grads,
    mut update: impl FnMut(&str, &mut Param, Tensor) -> Result<()>,
) -> Result<()> {
    let mut grads = grads;

    let mut failure: Option<Error> = None;
    let mut seen: HashMap<GradKey, String> = HashMap::new();
    // The non-frozen parameters pass 1 validated, so pass 2 can prove the
    // mutable walk reaches every one of them.
    let mut expected: HashMap<GradKey, String> = HashMap::new();
    visit_all(&*model, &mut |path, leaf| {
        if failure.is_some() {
            return;
        }
        let Leaf::Param(param) = leaf else {
            return;
        };
        if let Some(first) = seen.insert(param.grad_key(), path.to_string()) {
            failure = Some(Error::InvalidArg {
                op,
                msg: format!(
                    "one parameter is visited twice (as `{first}` and as `{path}`): the \
                     optimizer would step it twice in one update. Own a tied parameter \
                     once and write both uses inline"
                ),
            });
            return;
        }
        if param.is_frozen() {
            return;
        }
        expected.insert(param.grad_key(), path.to_string());
        let value = param.value();
        if !value.dtype().is_float() {
            failure = Some(Error::InvalidArg {
                op,
                msg: format!(
                    "parameter `{path}` has dtype {}: only floating-point parameters \
                     can be optimized (an integer buffer is structure, not precision)",
                    value.dtype()
                ),
            });
            return;
        }
        // The loudness gate: no gradient for a non-frozen parameter is an
        // error naming the path, so an untraced weight access (wrong `Mode`,
        // a forward that read `Param::value` instead of `Param::get`) is
        // caught at the very next step instead of silently freezing it.
        // Only a genuinely *absent* gradient is `MissingGrad`. `wrt` also
        // narrows to the parameter's dtype, so a backend failure in that
        // conversion arrives here too — reporting it as "no gradient" would send
        // the reader hunting an untraced-weight bug that does not exist.
        let grad = match grads.wrt(param) {
            Ok(grad) => grad,
            Err(Error::NotTraced { .. }) => {
                failure = Some(Error::MissingGrad {
                    path: path.to_string(),
                });
                return;
            }
            Err(error) => {
                failure = Some(error);
                return;
            }
        };
        if grad.dims() != value.dims() {
            failure = Some(Error::ShapeMismatch {
                op,
                lhs: value.shape().clone(),
                rhs: grad.shape().clone(),
            });
        } else if grad.dtype() != value.dtype() {
            failure = Some(Error::DTypeMismatch {
                op,
                expected: value.dtype(),
                got: grad.dtype(),
            });
        } else if grad.device() != value.device() {
            failure = Some(Error::DeviceMismatch {
                op,
                expected: value.device(),
                got: grad.device(),
            });
        }
    });
    if let Some(e) = failure {
        return Err(e);
    }

    // ---- pass 2: the mutable walk must reach exactly the same parameters ----
    let mut unreached: HashSet<GradKey> = expected.keys().copied().collect();
    let mut failure: Option<Error> = None;
    visit_all_mut(model, &mut |path, leaf| {
        if failure.is_some() {
            return;
        }
        let LeafMut::Param(param) = leaf else {
            return;
        };
        if param.is_frozen() {
            return;
        }
        if !unreached.remove(&param.grad_key()) {
            failure = Some(Error::InvalidArg {
                op,
                msg: format!(
                    "`{path}` is emitted by visit_mut but was not validated by visit: \
                     the module's two walks disagree, so a parameter would be stepped \
                     unchecked or twice"
                ),
            });
        }
    });
    if let Some(e) = failure {
        return Err(e);
    }
    if let Some(key) = unreached.iter().next() {
        let path = &expected[key];
        return Err(Error::InvalidArg {
            op,
            msg: format!(
                "`{path}` is emitted by visit but not by visit_mut: the module's two \
                 walks disagree, so that parameter would silently never be updated"
            ),
        });
    }

    // ---- pass 3: apply ----
    let mut failure: Option<Error> = None;
    visit_all_mut(model, &mut |path, leaf| {
        if failure.is_some() {
            return;
        }
        let LeafMut::Param(param) = leaf else {
            return;
        };
        if param.is_frozen() {
            return;
        }
        // Passes 1–2 proved this lookup succeeds. Drained *wide*: pass 1 already
        // checked the gradient against the parameter's own dtype, and both
        // optimizers widen to the accumulation dtype anyway, so narrowing here
        // would only throw precision away (see `Grads::take_wide`).
        let Some(grad) = (match grads.take_wide(param.grad_key()) {
            Ok(grad) => grad,
            Err(error) => {
                failure = Some(error);
                return;
            }
        }) else {
            failure = Some(Error::MissingGrad {
                path: path.to_string(),
            });
            return;
        };
        if let Err(e) = update(path, param, grad) {
            failure = Some(e);
        }
    });
    match failure {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Everything one parameter's update formula is handed, with the parts that are
/// the same for both optimizers already done.
pub(crate) struct Update<'a, H, B> {
    /// The hyperparameters in force for this parameter's path — the same values
    /// the pre-pass range-checked.
    pub(crate) hyper: H,
    /// The parameter's current value, widened to its accumulation dtype.
    pub(crate) weights: Tensor,
    /// The gradient, in the same (wide) dtype as `weights`.
    pub(crate) grad: Tensor,
    /// This parameter's buffers as the last update left them, or `None` for a
    /// parameter this optimizer has never updated.
    pub(crate) previous: Option<&'a B>,
    /// This parameter's **own** step count once this update lands: `1` for a
    /// first update, whatever the optimizer's global `steps` says.
    pub(crate) clock: u64,
}

/// One optimizer step: everything `Sgd::step` and `Adam::step` do around their
/// update formula, with `kind` naming the optimizer ("SGD"/"Adam") in errors.
///
/// The order is the contract [`optim`](super)'s module docs state normatively,
/// so it is written once:
///
/// 1. the global step clock must have room for one more update;
/// 2. [`prepass`] validates every parameter's clock and hyperparameters before
///    the first `Param::set`;
/// 3. [`apply`] runs the three-pass walk, and for each parameter this function
///    advances the clock, widens the value and the gradient to the accumulation
///    dtype, calls `update` for the new value and buffers, narrows the value
///    back to the parameter's own dtype and records the new state;
/// 4. `steps` advances only once the whole walk has succeeded, so a rejected
///    step leaves the schedule and the model agreeing about how far the run got.
///
/// `check` is the optimizer's scalar range check, run per parameter in step 2
/// against the same hyperparameters and clock `update` will see in step 3.
#[allow(clippy::too_many_arguments)] // The optimizer's three state fields have
// to arrive separately: they are borrowed disjointly out of `&mut self`.
pub(crate) fn step<H: Clone, B>(
    kind: &'static str,
    model: &mut dyn Module,
    grads: Grads,
    groups: &Groups<H>,
    steps: &mut u64,
    state: &mut States<B>,
    check: impl FnMut(&Param, H, u64) -> Result<()>,
    mut update: impl FnMut(Update<'_, H, B>) -> Result<(Tensor, B)>,
) -> Result<()> {
    let next_steps = steps.checked_add(1).ok_or_else(|| Error::InvalidArg {
        op: "step",
        msg: format!("{kind} global step clock cannot be advanced past u64::MAX"),
    })?;
    // Clocks and hyperparameters are checked before any `Param::set`; see
    // `prepass` for why the kernel's own range check is too late.
    prepass(kind, model, groups, state, check)?;
    apply("step", model, grads, |path, param, grad| {
        let dtype = param.value().dtype();
        // Wide arithmetic: an f16/bf16 parameter's moments are kept in f32 and
        // the replacement value is narrowed back exactly once, at the end.
        let acc = dtype.accumulation_dtype();
        let previous = state.get(&param.grad_key());
        let clock = match previous {
            Some(entry) => entry
                .clock
                .checked_add(1)
                .ok_or_else(|| Error::InvalidArg {
                    op: "step",
                    msg: format!(
                        "{kind} step clock for parameter `{path}` cannot be advanced past u64::MAX"
                    ),
                })?,
            None => 1,
        };
        let (next, buffers) = update(Update {
            hyper: groups.resolve(path),
            weights: param.value().to_dtype(acc)?,
            grad: grad.to_dtype(acc)?,
            previous: previous.map(|entry| &entry.buffers),
            clock,
        })?;
        param.set(next.to_dtype(dtype)?)?;
        state.insert(param.grad_key(), ParamState { clock, buffers });
        Ok(())
    })?;
    *steps = next_steps;
    Ok(())
}

/// Re-attach the `count` storages a fused optimizer kernel returned to tensors
/// shaped like `like`, where `kind` names the optimizer ("SGD"/"Adam").
///
/// A backend that answers with the wrong number of buffers, or with one whose
/// dtype/device/length does not match the parameter it belongs to, is a
/// [`Error::Backend`] rather than a silently mis-shaped parameter.
pub(crate) fn fused_outputs(
    kind: &'static str,
    outputs: Vec<Storage>,
    count: usize,
    like: &Tensor,
) -> Result<Vec<Tensor>> {
    if outputs.len() != count {
        return Err(Error::Backend {
            op: "step",
            msg: format!(
                "fused {kind} returned {} outputs, expected {count}",
                outputs.len()
            ),
        });
    }
    let mut tensors = Vec::with_capacity(count);
    for (index, storage) in outputs.into_iter().enumerate() {
        if storage.dtype() != like.dtype()
            || storage.device() != like.device()
            || storage.len() != like.num_elements()
        {
            return Err(Error::Backend {
                op: "step",
                msg: format!(
                    "fused {kind} output {index} has dtype {}, device {}, and {} elements; \
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
    Ok(tensors)
}

/// The dotted paths of every parameter of `model`, paired with its gradient
/// identity — the path ↔ state key mapping optimizer-state persistence needs
/// (state lives under `GradKey`, files are keyed by path).
///
/// Deliberately **not** filtered by [`Param::is_frozen`]: a parameter frozen for
/// a warmup phase still owns optimizer state and a step clock, and dropping its
/// path here would silently lose both across a checkpoint — so unfreezing it
/// later would restart its moments from zero.
pub(crate) fn param_paths(model: &dyn Module) -> Vec<(String, GradKey)> {
    let mut out = Vec::new();
    visit_all(model, &mut |path, leaf| {
        if let Leaf::Param(param) = leaf {
            out.push((path.to_string(), param.grad_key()));
        }
    });
    out
}

/// The value tensor of the parameter at `path`, for validating a loaded
/// moment buffer against the parameter it belongs to.
pub(crate) fn param_values(model: &dyn Module) -> HashMap<String, Tensor> {
    let mut out = HashMap::new();
    visit_all(model, &mut |path, leaf| {
        if let Leaf::Param(param) = leaf {
            out.insert(path.to_string(), param.value().clone());
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Mode;
    use crate::nn::visit::{Visitor, VisitorMut};
    use crate::optim::testkit::{CPU, t};

    /// Drive `apply` with an update that only records what it was handed, so a
    /// test can assert on the *walk* without an optimizer's arithmetic in the
    /// way.
    fn record_walk(model: &mut dyn Module, grads: Grads) -> Result<Vec<String>> {
        let mut visited = Vec::new();
        apply("step", model, grads, |path, _param, _grad| {
            visited.push(path.to_string());
            Ok(())
        })?;
        Ok(visited)
    }

    /// A `Grads` built by hand, so a test can hand the engine a deliberately
    /// mismatched gradient the autograd engine would never produce.
    fn forged(pairs: Vec<(&Param, Tensor)>) -> Grads {
        Grads::from_pairs(
            pairs
                .into_iter()
                .map(|(p, g)| (p.grad_key(), g))
                .collect::<HashMap<_, _>>(),
        )
    }

    #[test]
    fn groups_resolve_by_first_match() {
        #[derive(Clone, Copy, PartialEq, Debug)]
        struct Hyper {
            decay: f64,
        }
        let mut groups = Groups::new(Hyper { decay: 0.1 });
        groups.push(|p| p.ends_with("bias"), |_| Hyper { decay: 0.0 });
        // A second, wider predicate that also matches every bias: it must never
        // win, because the first match does.
        groups.push(|_| true, |_| Hyper { decay: 9.0 });

        assert_eq!(groups.resolve("fc.bias").decay, 0.0);
        assert_eq!(groups.resolve("fc.weight").decay, 9.0);
        // The base is what an unmatched path would get…
        assert_eq!(groups.base().decay, 0.1);
        // …and mutating it reaches the groups that do not override that field,
        // because an override is a function of the base, not a snapshot.
        groups.base_mut().decay = 0.5;
        assert_eq!(groups.resolve("fc.bias").decay, 0.0);
    }

    /// A model whose *read* walk emits one parameter under two names — the
    /// double-step hazard the design forbids (own a tied parameter once).
    struct Doubled {
        p: Param,
    }
    impl Module for Doubled {
        fn visit(&self, v: &mut Visitor) {
            v.param("a", &self.p);
            v.param("b", &self.p);
        }
        fn visit_mut(&mut self, v: &mut VisitorMut) {
            v.param("a", &mut self.p);
        }
    }

    #[test]
    fn a_parameter_reached_twice_is_rejected_before_any_update() {
        let mut m = Doubled {
            p: Param::new(t(&[1.0])),
        };
        let grads = m.p.get(Mode::TRAIN).sum_all().unwrap().backward().unwrap();
        let err = record_walk(&mut m, grads).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("visited twice"), "{msg}");
        assert!(msg.contains('`') && msg.contains("`b`"), "{msg}");
    }

    /// A hand-written `Module` whose mutable walk forgets `b`: under a naive
    /// implementation `b` would silently never be updated.
    struct Lopsided {
        a: Param,
        b: Param,
    }
    impl Module for Lopsided {
        fn visit(&self, v: &mut Visitor) {
            v.param("a", &self.a);
            v.param("b", &self.b);
        }
        fn visit_mut(&mut self, v: &mut VisitorMut) {
            v.param("a", &mut self.a);
        }
    }

    #[test]
    fn a_parameter_the_mutable_walk_forgets_is_loud() {
        let mut m = Lopsided {
            a: Param::new(t(&[1.0])),
            b: Param::new(t(&[1.0])),
        };
        let grads =
            m.a.get(Mode::TRAIN)
                .add(&m.b.get(Mode::TRAIN))
                .unwrap()
                .sum_all()
                .unwrap()
                .backward()
                .unwrap();
        let msg = record_walk(&mut m, grads).unwrap_err().to_string();
        assert!(
            msg.contains("`b` is emitted by visit but not by visit_mut"),
            "{msg}"
        );
    }

    /// The mirror image: a mutable walk that reaches a parameter the read walk
    /// never validated, so its gradient would be applied unchecked.
    struct Extra {
        a: Param,
        b: Param,
    }
    impl Module for Extra {
        fn visit(&self, v: &mut Visitor) {
            v.param("a", &self.a);
        }
        fn visit_mut(&mut self, v: &mut VisitorMut) {
            v.param("a", &mut self.a);
            v.param("b", &mut self.b);
        }
    }

    #[test]
    fn a_parameter_only_the_mutable_walk_reaches_is_loud() {
        let mut m = Extra {
            a: Param::new(t(&[1.0])),
            b: Param::new(t(&[1.0])),
        };
        let grads = m.a.get(Mode::TRAIN).sum_all().unwrap().backward().unwrap();
        let msg = record_walk(&mut m, grads).unwrap_err().to_string();
        assert!(
            msg.contains("`b` is emitted by visit_mut but was not validated by visit"),
            "{msg}"
        );
    }

    #[derive(rstorch::Module)]
    struct One {
        w: Param,
    }

    #[test]
    fn an_integer_parameter_cannot_be_optimized() {
        let mut m = One {
            w: Param::new(Tensor::from_vec(vec![1i64, 2], [2], &CPU).unwrap()),
        };
        // An i64 parameter has no gradient at all, but the dtype rejection
        // comes first so the message names the real problem.
        let msg = record_walk(&mut m, Grads::from_pairs(HashMap::new()))
            .unwrap_err()
            .to_string();
        assert!(msg.contains("only floating-point parameters"), "{msg}");
        assert!(msg.contains("`w`"), "{msg}");
    }

    #[test]
    fn a_mismatched_gradient_is_rejected_by_kind() {
        let mut m = One {
            w: Param::new(t(&[1.0, 2.0])),
        };

        // Wrong shape.
        let grads = forged(vec![(&m.w, t(&[1.0, 2.0, 3.0]))]);
        assert!(matches!(
            record_walk(&mut m, grads),
            Err(Error::ShapeMismatch { op: "step", .. })
        ));

        // Wrong dtype.
        let ints = Tensor::from_vec(vec![1i64, 1], [2], &CPU).unwrap();
        let grads = forged(vec![(&m.w, ints)]);
        assert!(matches!(
            record_walk(&mut m, grads),
            Err(Error::DTypeMismatch { op: "step", .. })
        ));

        // …and the rejections changed nothing.
        assert_eq!(m.w.value().to_vec::<f32>().unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn frozen_parameters_are_skipped_and_need_no_gradient() {
        #[derive(rstorch::Module)]
        struct Two {
            a: Param,
            b: Param,
        }
        let mut m = Two {
            a: Param::new(t(&[1.0])),
            b: Param::new(t(&[1.0])),
        };
        m.b.freeze();
        // Only `a` is traced, because a frozen `Param::get` hands out the value.
        let grads =
            m.a.get(Mode::TRAIN)
                .add(&m.b.get(Mode::TRAIN))
                .unwrap()
                .sum_all()
                .unwrap()
                .backward()
                .unwrap();
        assert_eq!(record_walk(&mut m, grads).unwrap(), vec!["a".to_string()]);
    }

    #[test]
    fn param_paths_and_values_agree_with_the_walk() {
        let m = One {
            w: Param::new(t(&[3.0])),
        };
        let paths = param_paths(&m);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].0, "w");
        assert_eq!(paths[0].1, m.w.grad_key());
        assert_eq!(param_values(&m)["w"].to_vec::<f32>().unwrap(), vec![3.0f32]);
    }
}
