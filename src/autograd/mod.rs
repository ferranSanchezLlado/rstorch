//! Autograd: tracing is data flow, gradients are a linear value
//! (exploration §4.3).
//!
//! **Contract module** (T01). This file defines the autograd *types* and
//! the *seams* the rest of the crate codes against; the engine itself is
//! **T30**, which replaces the `todo!()`/no-op bodies without changing any
//! signature here:
//!
//! - [`record`] — the seam every differentiable op calls to (maybe) wrap its
//!   forward output in a graph node. T01's body is a **no-op** that returns
//!   the output untraced, so W3 op tasks can write their forward paths and
//!   backward closures against a stable API; they stay inert until T30 fills
//!   the engine (T31 then activates the `#[ignore]`d finite-difference
//!   tests). op tasks never touch [`Node`] internals — only [`record`].
//! - [`make_leaf`] — the seam [`Param`](crate::nn::Param) and
//!   [`traced`] use to build a leaf tensor carrying a fresh [`GradKey`].
//! - [`Grads`] — the linear result of [`backward`]: `!Clone`,
//!   `#[must_use]`, consumed by the optimizer via move.
//!
//! # The detached-output capture rule (exploration §4.3)
//!
//! A backward closure may capture the op's **output only in detached form**
//! (a fresh `Inner` that shares storage but has `node: None`), built *before*
//! the traced output is assembled. Output-dependent formulas (sigmoid, tanh,
//! softmax) need the output value; capturing the *traced* output instead
//! would create an `Arc` cycle (output node → closure → output node) that
//! leaks the whole graph. The op author is responsible for honoring this;
//! [`record`] takes the already-built forward output and the closure
//! separately so the rule is expressible. A strong-count leak test gates T30.

// Node's fields, the Grads structural helpers, and the leaf/record seams are
// consumed by T30 (engine), T40 (Param), and T44 (optimizer); the integrator
// removes this allow at v3-m2 once the engine and its consumers exist.
#![allow(dead_code)]

use crate::error::Result;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// A stable identity for one gradient slot: either a
/// [`Param`](crate::nn::Param) leaf or a [`traced`]-created input leaf.
///
/// Minted from a single process-wide atomic counter. This is *identity*,
/// not ambient grad state — nothing about tracing is decided by a global;
/// recording happens iff an input actually carries a [`Node`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct GradKey(u64);

static NEXT_GRAD_KEY: AtomicU64 = AtomicU64::new(1);

impl GradKey {
    /// Mint a fresh, unique key.
    pub(crate) fn fresh() -> GradKey {
        GradKey(NEXT_GRAD_KEY.fetch_add(1, Ordering::Relaxed))
    }
}

/// A backward closure: given the cotangent (upstream gradient) of the op's
/// output, produce the cotangent contribution for each input in order
/// (`None` where an input needs no gradient). `Send + Sync` so the immutable
/// graph is shareable across threads (`backward` is a pure function over it).
pub(crate) type BackwardFn = Box<dyn Fn(&Tensor) -> Vec<Option<Tensor>> + Send + Sync>;

/// A node in the autograd graph. Immutable once built; shared via
/// `Arc<Node>` from every [`Tensor`] that participates in the computation.
///
/// **T30 owns these internals.** T01 defines the shell so
/// [`Inner`](crate::tensor) can hold `Option<Arc<Node>>` and op tasks can
/// pass traced tensors around opaquely. T30 may restructure the fields; no
/// code outside this module reads them.
#[allow(dead_code)]
pub(crate) struct Node {
    /// The public op name that created this node (for diagnostics).
    op: &'static str,
    /// Set on **leaf** nodes (from [`make_leaf`]); `None` on interior nodes.
    key: Option<GradKey>,
    /// Parent nodes, one slot per input (`None` = a non-traced input).
    inputs: Vec<Option<Arc<Node>>>,
    /// The backward closure; `None` on leaves.
    backward: Option<BackwardFn>,
}

/// The seam every differentiable op calls after computing its forward
/// `output`. If any of `inputs` is traced, the returned tensor carries a new
/// interior [`Node`] wiring `backward` to the inputs' nodes; otherwise the
/// output is returned untraced.
///
/// `output` **must be detached** (`node: None`); `backward` may close over it
/// (see the detached-output capture rule in the module docs). `backward`
/// receives the output cotangent and returns one optional cotangent per
/// entry of `inputs`, in the same order.
///
/// T01 body: **no-op** — returns `output` unchanged. T30 replaces it with the
/// graph-building implementation (signature frozen).
pub(crate) fn record(
    op: &'static str,
    output: Tensor,
    inputs: &[&Tensor],
    backward: BackwardFn,
) -> Tensor {
    let _ = (op, inputs, backward);
    output
}

/// Build a leaf tensor: `value` with a fresh leaf [`Node`] carrying `key`, so
/// that using it in a computation records a graph rooted at `key`. Shares
/// `value`'s storage (no copy). Used by [`Param::get`](crate::nn::Param::get)
/// (cached leaf) and [`traced`].
///
/// T01 body: returns `value` untraced (a no-op leaf). T30 replaces it.
pub(crate) fn make_leaf(value: Tensor, key: GradKey) -> Tensor {
    let _ = key;
    value
}

/// Turn `t` into a traced leaf for grad-wrt-input (exploration §4.3):
/// `let xt = x.traced(); let y = f(&xt)?; let g = y.backward()?;
/// g.wrt_input(&xt)`. The **returned** binding must be the one used in the
/// computation *and* in the lookup.
///
/// Errors with [`Error::InvalidArg`](crate::Error::InvalidArg)
/// (`op: "traced"`) if `t` already carries a graph — double-tracing is a bug.
/// (`NotTraced` is reserved for the opposite no-graph condition.) T30 fills
/// the body.
pub(crate) fn traced(t: &Tensor) -> Result<Tensor> {
    let _ = t;
    todo!("T30: traced() leaf creation")
}

/// Run reverse-mode autodiff from `t` (a scalar or the seed-1 convention T30
/// documents), returning a fresh [`Grads`]. Pure: an immutable walk over the
/// `Arc` graph, iterative (no recursion — 100k-node drop/backward stress
/// gates T30). Repeated use of a leaf (weight tying) accumulates.
///
/// [`Error::NotTraced`](crate::Error::NotTraced) if `t` carries no graph.
/// T30 fills the body.
pub(crate) fn backward(t: &Tensor) -> Result<Grads> {
    let _ = t;
    todo!("T30: backward() reverse pass")
}

/// The result of [`Tensor::backward`](crate::Tensor::backward): the
/// gradients, as a **linear** value (exploration §4.3, §5).
///
/// `Grads` is deliberately **not `Clone`** and is `#[must_use]`: the
/// optimizer consumes it by move (`opt.step(&mut model, grads)`), so applying
/// the same gradients twice is a compile error and forgetting to step is a
/// warning. Micro-batch accumulation and clipping are explicit linear
/// pipelines (`acc = acc.merge(step)?`, `grads.clip_norm(1.0)?`).
#[must_use]
pub struct Grads {
    grads: HashMap<GradKey, Tensor>,
}

impl Grads {
    /// Build from a key→gradient map (T30's engine output).
    pub(crate) fn from_pairs(grads: HashMap<GradKey, Tensor>) -> Grads {
        Grads { grads }
    }

    /// Remove and return the gradient for `key` (optimizer drain path).
    pub(crate) fn take(&mut self, key: GradKey) -> Option<Tensor> {
        self.grads.remove(&key)
    }

    /// Whether a gradient is present for `key`.
    pub(crate) fn contains(&self, key: GradKey) -> bool {
        self.grads.contains_key(&key)
    }

    /// Number of gradient entries.
    pub fn len(&self) -> usize {
        self.grads.len()
    }

    /// Whether there are no gradients.
    pub fn is_empty(&self) -> bool {
        self.grads.is_empty()
    }

    /// Accumulate `other` into `self` (element-wise add on matching keys,
    /// union of keys), consuming both — the explicit micro-batch
    /// accumulation `acc = acc.merge(step)?`. T30 fills the body.
    pub fn merge(self, other: Grads) -> Result<Grads> {
        let _ = other;
        todo!("T30: Grads::merge")
    }

    /// Scale every gradient by `factor`, consuming `self`. T30 fills the body.
    pub fn scale(self, factor: f64) -> Result<Grads> {
        let _ = factor;
        todo!("T30: Grads::scale")
    }

    /// Clip by global L2 norm to `max_norm`, consuming `self`. T30 fills the
    /// body.
    pub fn clip_norm(self, max_norm: f64) -> Result<Grads> {
        let _ = max_norm;
        todo!("T30: Grads::clip_norm")
    }

    /// The gradient with respect to `param` (a clone of the stored handle;
    /// gradients are cheap `Arc`-backed tensors). T30 fills the body.
    pub fn wrt(&self, param: &crate::nn::Param) -> Result<Tensor> {
        let _ = param;
        todo!("T30: Grads::wrt")
    }

    /// The gradient with respect to a traced input (exploration §4.3).
    /// `input` must be the tensor returned by
    /// [`Tensor::traced`](crate::Tensor::traced) and used in the
    /// computation. T30 fills the body.
    pub fn wrt_input(&self, input: &Tensor) -> Result<Tensor> {
        let _ = input;
        todo!("T30: Grads::wrt_input")
    }
}
