//! Optimizers: [`Sgd`], [`Adam`]/[`AdamW`], path-predicate parameter groups,
//! per-parameter step clocks.
//!
//! # There is no `zero_grad`
//!
//! `step` **consumes** the [`Grads`](crate::Grads) by move. Gradients are a
//! return value, not state hanging off the parameters, so there is nothing to
//! zero and reusing the same gradients after the optimizer consumes them is a
//! compile error rather than silent double-counting. Ignoring a `Grads` value
//! emits its `#[must_use]` warning. Micro-batch accumulation and
//! clipping are explicit linear pipelines on the `Grads` itself
//! (`acc = acc.merge(step)?`, `grads.clip_norm(1.0)?`).
//!
//! # Accidental freezing is loud
//!
//! A non-frozen parameter the optimizer visits which has **no gradient** in the
//! `Grads` is [`Error::MissingGrad`](crate::Error::MissingGrad) naming its
//! dotted path — never a silent skip. That one rule is what makes the whole
//! family of "my weights never trained" bugs (a forward that read
//! [`Param::value`](crate::nn::Param::value) instead of
//! [`Param::get`](crate::nn::Param::get), a [`Mode`](crate::nn::Mode) without
//! recording, a layer left out of the graph) detectable at the very next step.
//! Freezing is available, but only explicitly, via
//! [`Param::freeze`](crate::nn::Param::freeze).
//!
//! The check is a *pre-pass*: the whole parameter walk is validated before the
//! first value is swapped, so a rejected step leaves the model exactly as it
//! was.
//!
//! Stated precisely, because a half-guarantee is worse than none: *every*
//! rejection this layer can diagnose — a missing gradient, a gradient whose
//! shape/dtype/device does not match its parameter, a non-float parameter, a
//! module that visits one parameter twice or whose two walks disagree — is
//! raised before the first [`Param::set`](crate::nn::Param::set), and neither
//! the model nor any moment buffer or step clock moves. Once that pre-pass has
//! passed, the only remaining failure is the backend's (an allocation failure
//! mid-update), and *that* one can leave the update partially applied; it is
//! reported, not swallowed, but a run must treat it as fatal rather than retry
//! the step.
//!
//! # Writing your own optimizer
//!
//! Closed extension point, on paper: `Rule`/`Engine` are `pub(crate)`
//! (`src/optim/engine.rs`), and the generic parameter walk they run on cannot
//! be *started* from outside:
//! [`Module::visit_mut`](crate::nn::Module::visit_mut) and
//! [`VisitorMut`](crate::nn::VisitorMut) are public, but the walk's sink type
//! `LeafMut` and `VisitorMut::new` are `pub(crate)` (`src/nn/visit.rs`), so a
//! third-party type can *be* visited without being able to visit. Neither
//! omission actually closes it — an optimizer for *your own* model never
//! needed a generic walk, because you already wrote the struct and can name its
//! [`Param`](crate::nn::Param) fields directly. `examples/custom_optimizer.rs`
//! builds RMSprop this way from [`Param::get`](crate::nn::Param::get)/
//! [`set`](crate::nn::Param::set)/[`is_frozen`](crate::nn::Param::is_frozen),
//! [`Grads::wrt`](crate::Grads::wrt), and
//! [`Envelope::set_section`](crate::persist::Envelope::set_section)/
//! [`insert_tensor`](crate::persist::Envelope::insert_tensor) for its own
//! checkpoint state.
//!
//! What that recipe does **not** inherit, so nobody assumes it comes free:
//!
//! - **The all-or-nothing pre-pass.** `Sgd`/`Adam::step` validate every
//!   parameter — a missing gradient, a shape/dtype/device mismatch — before
//!   touching any of them; a rejected step leaves the model untouched. A
//!   hand-written loop that updates parameters one at a time as it visits
//!   them can leave a partially-stepped model on a failure partway through.
//! - **Path-predicate parameter groups.** `AdamW::new(..).group(path_predicate,
//!   overrides)` gives one optimizer instance different hyperparameters for
//!   different parameters by dotted `state_dict` path. A hand-written
//!   optimizer has no such mechanism unless it builds one.
//!
//! # There is no `Optimizer` trait
//!
//! [`Sgd`] and [`Adam`] are concrete types — every layer and every optimizer
//! is. Generic-over-optimizer code is not a
//! first-hour need, and another dynamic trait would widen the core vocabulary;
//! a caller who wants to switch optimizers at runtime
//! writes an enum over the two.
//!
//! Internally the two *are* one implementation: each is a crate-private
//! `Engine` over a crate-private `Rule` naming the only things they differ in —
//! an update formula, the buffers it carries, and the scalars a checkpoint
//! holds. Everything else, this module's guarantees included, is written once.
//! That trait is not exported and cannot be named or implemented from outside,
//! so it is machinery rather than a sixth trait in the public vocabulary.
//!
//! # Parameter groups
//!
//! Groups are path predicates on the builder, which is all the standard
//! transformer recipe needs:
//!
//! ```
//! use rstorch::optim::AdamW;
//!
//! let opt = AdamW::new(3e-4, 0.1)
//!     // No weight decay on biases and norm gains.
//!     .group(|path| path.ends_with("bias") || path.contains("norm"),
//!            |g| g.weight_decay(0.0))
//!     // Discriminative learning rates use the same mechanism.
//!     .group(|path| path.starts_with("trunk."), |g| g.lr_scale(0.1));
//! ```
//!
//! The **first** matching predicate wins, and a group states only its
//! differences from the optimizer's own hyperparameters — so
//! [`set_lr`](Adam::set_lr) and the [`schedule`] functions keep moving every
//! group together.
//!
//! # Per-parameter step clocks
//!
//! Bias correction is driven by each parameter's **own** update count, not a
//! global counter. A parameter that joins the update late — unfrozen after a
//! warmup phase, or a fresh head attached to a pretrained trunk — is
//! bias-corrected as a *first* update. Under a global clock its near-zero
//! moment estimate would be divided by an almost-saturated correction factor,
//! and it would take a step several times too small.
//!
//! # State persistence
//!
//! `save_state`/`load_state` ride the versioned
//! [`Envelope`](crate::persist::Envelope): moment buffers as ordinary
//! safetensors tensors under an `optim.` key namespace, hyperparameters and
//! step clocks as an opaque `optimizer` section. Groups are *not* saved — they
//! are closures, reconstructed by building the optimizer the same way.
//!
//! Both halves are all-or-nothing in the same sense as the step: a load decodes
//! and checks every buffer against the parameter it belongs to before it
//! replaces any of the optimizer's own state.
//!
//! **Scope.** This layer owns *optimizer* state. It deliberately does not
//! decide the model-level checkpoint surface — which sections a full checkpoint
//! carries, how `config` + weights reconstruct a model, or where the
//! [`Rng`](crate::Rng) section is written. Those are one decision, and
//! inventing half of it here would be the wrong half.
//! What this module guarantees is that an optimizer's state can be written into
//! an envelope that already holds a model's tensors, and read back out of one,
//! without the two colliding: model keys and `optim.*` keys share a file.
//!
//! # Cost
//!
//! An update is defined in the public op vocabulary — that spelling is the
//! reference definition, and each parameter costs a handful of small tensor
//! allocations per step, a measured hotspot at MLP scale. So where the backend
//! offers a fused step kernel it is the path taken, and the composed form is
//! the fallback for a backend that declines the fused op. Either way the
//! public surface is the same one; nothing here changes shape with it.
//!
//! # Reduced precision
//!
//! F16/BF16 parameters retain their storage dtype, while SGD momentum and
//! Adam/AdamW moments are F32 and persist as F32. Update arithmetic widens the
//! current parameter for the step and narrows the replacement parameter once.
//! The gradient needs no widening: it is *already* wide, because the autograd
//! engine accumulates a reduced-precision node's cotangent in F32 and the
//! optimizer drains that value directly. Narrowing it in between would be the
//! one avoidable precision loss in the step — a gradient past F16's range would
//! reach the update as `inf`, and one merely past its spacing would be rounded.
//! ([`Grads::wrt`](crate::Grads::wrt), which a *caller* uses to inspect a
//! gradient, still reports it in the parameter's own dtype, as `PyTorch` does.)
//! There are deliberately no persistent master weights: the checkpoint contains
//! the reduced model values and wide optimizer state only.

mod adam;
mod engine;
pub mod schedule;
mod sgd;
mod state;
pub(crate) mod validate;

pub use adam::{Adam, AdamGroup, AdamW};
pub use sgd::{Sgd, SgdGroup};

#[cfg(test)]
pub(crate) mod testkit;
