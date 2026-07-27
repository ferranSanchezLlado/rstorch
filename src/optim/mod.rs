//! Optimizers: [`Sgd`], [`Adam`]/[`AdamW`], path-predicate parameter groups,
//! per-parameter step clocks (exploration §4.4).
//!
//! # There is no `zero_grad`
//!
//! `step` **consumes** the [`Grads`](crate::Grads) by move. Gradients are a
//! return value, not state hanging off the parameters, so there is nothing to
//! zero and applying the same gradients twice is a compile error rather than
//! silent double-counting (exploration §4.3, §5). Micro-batch accumulation and
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
//! # There is no `Optimizer` trait
//!
//! [`Sgd`] and [`Adam`] are concrete types (exploration §4.1: "every layer and
//! every optimizer are concrete types"). Generic-over-optimizer code is not a
//! first-hour need, and a sixth public trait would cost the design's
//! five-public-traits claim; a caller who wants to switch optimizers at runtime
//! writes an enum over the two. The shared machinery is crate-private instead,
//! so the two cannot drift.
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
//! `save_state`/`load_state` ride T17's versioned
//! [`Envelope`](crate::persist::Envelope): moment buffers as ordinary
//! safetensors tensors under an `optim.` key namespace, hyperparameters and
//! step clocks as an opaque `optimizer` section. Groups are *not* saved — they
//! are closures, reconstructed by building the optimizer the same way.
//!
//! Both halves are all-or-nothing in the same sense as the step: a load decodes
//! and checks every buffer against the parameter it belongs to before it
//! replaces any of the optimizer's own state.
//!
//! **Scope.** This layer owns *optimizer* state and the crate-internal
//! `Tensor` ↔ [`HostTensor`](crate::persist::HostTensor) bridge that carries it.
//! It deliberately does not decide the model-level checkpoint surface — which
//! sections a full checkpoint carries, how `config` + weights reconstruct a
//! model, or where the [`Rng`](crate::Rng) section is written. Those are one
//! decision (T52's), and inventing half of it here would be the wrong half.
//! What this module guarantees is that an optimizer's state can be written into
//! an envelope that already holds a model's tensors, and read back out of one,
//! without the two colliding: model keys and `optim.*` keys share a file.
//!
//! # Cost
//!
//! An update is written in the public op vocabulary, so each parameter costs a
//! handful of small tensor allocations per step. That was a measured hotspot in
//! v2 at MLP scale; the fix is a fused backend kernel behind the same public
//! surface (T48), not a different API here.
//!
//! # Reduced precision
//!
//! F16/BF16 parameters retain their storage dtype, while SGD momentum and
//! Adam/AdamW moments are F32 and persist as F32. Update arithmetic widens the
//! current parameter and gradient for the step and narrows the replacement
//! parameter once. There are deliberately no persistent master weights: the
//! checkpoint contains the reduced model values and wide optimizer state only.

mod adam;
mod engine;
pub mod schedule;
mod sgd;
mod state;

// The `Tensor` ↔ `HostTensor` bridge. Crate-visible rather than private to this
// module because `persist` names "the nn/optim runtime (T40/T44)" as its owner
// and the model-level checkpoint surface (T52) needs the same two functions;
// nothing in it is public.
pub(crate) mod host;

pub use adam::{Adam, AdamGroup, AdamW};
pub use sgd::{Sgd, SgdGroup};

#[cfg(test)]
pub(crate) mod testkit;
