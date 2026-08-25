//! Autograd: tracing is data flow, gradients are a linear value.
//!
//! The types and the *seams* the rest of the crate codes against:
//!
//! - [`record`] — the seam every differentiable op calls to (maybe) wrap its
//!   forward output in a graph node. Op tasks never touch [`Node`] internals —
//!   only [`record`].
//! - [`make_leaf`] — the seam [`Param`](crate::nn::Param) and
//!   [`traced`] use to build a leaf tensor carrying a fresh [`GradKey`].
//! - [`Grads`] — the linear result of [`backward`]: `!Clone`,
//!   `#[must_use]`, consumed by the optimizer via move.
//!
//! # How the engine works
//!
//! A [`Node`] is an immutable record of "this value came from `op` applied to
//! these parent nodes, and here is how to push a cotangent back through it".
//! Nodes are shared by `Arc`, so the graph is a DAG: a value used twice has
//! one node with two consumers, and a [`Param`](crate::nn::Param) used twice
//! (weight tying) is *one* leaf node reached along two paths.
//!
//! - **Tracing is data flow.** There is no ambient grad mode: [`record`]
//!   builds a node **iff** at least one input already carries one, and the
//!   only sources of leaf nodes are [`make_leaf`] (behind
//!   [`Param::get`](crate::nn::Param::get)) and [`traced`]. An eval-mode
//!   forward therefore retains nothing.
//! - **Only float tensors trace.** [`record`] returns a non-float output
//!   untraced: an `i64`/`bool` value has no cotangent, so a chain through
//!   `to_dtype(DType::I64)` simply stops being traced there — and a later
//!   `backward()` on it is the loud [`Error::NotTraced`] rather than a silent
//!   zero.
//! - **[`backward`] is a pure function over the graph.** It walks an
//!   *iterative* reverse topological order (an explicit worklist; a 100k-node
//!   chain must not touch the stack), keeps one cotangent per node in a map,
//!   sums the contributions that arrive from several consumers in F32 for
//!   F16/BF16 nodes (narrowing once when the fan-in is complete), and drops each
//!   cotangent as soon as its node has been processed. Graph nodes are never
//!   mutated; deferred storage may fill its write-once computation cache, so
//!   the same graph can still be differentiated twice and from several
//!   threads.
//! - **[`Drop`] is iterative too.** Dropping the head of a long chain would
//!   otherwise recurse once per node; [`Node`]'s `Drop` moves the parents onto
//!   a worklist instead.
//!
//! # The detached-output capture rule
//!
//! A backward closure may capture the op's **output only in detached form**
//! (a fresh `Inner` that shares storage but has `node: None`), built *before*
//! the traced output is assembled. Output-dependent formulas (sigmoid, tanh,
//! softmax) need the output value; capturing the *traced* output instead
//! would create an `Arc` cycle (output node → closure → output node) that
//! leaks the whole graph. The op author is responsible for honoring this;
//! [`record`] takes the already-built forward output and the closure
//! separately so the rule is expressible. A strong-count leak test gates it.

use crate::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};
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
/// output, produce the cotangent contribution for each input in order.
///
/// The two ways of "no tensor here" are deliberately distinct:
///
/// - `None` in the vector means *this input legitimately receives no
///   gradient* — an integer index operand, a boolean mask, the non-selected
///   side of `max`. It is a fact about the derivative, not a failure.
/// - `Err` means the backward computation itself failed (an unsupported
///   kernel, an allocation failure). It aborts [`backward`] and surfaces to
///   the caller unchanged, so a backend error never masquerades as a missing
///   gradient.
///
/// `Send + Sync` so the immutable graph is shareable across threads
/// (`backward` is a pure function over it).
pub(crate) type BackwardFn = Box<dyn Fn(&Tensor) -> Result<Vec<Option<Tensor>>> + Send + Sync>;

/// A node in the autograd graph. Immutable once built; shared via
/// `Arc<Node>` from every [`Tensor`] that participates in the computation.
///
/// Node *identity* is the address of its `Arc` allocation: the engine holds an
/// `Arc` to every node it is walking, so two live nodes can never share one.
/// That keeps the struct free of an id field and makes "the same value used
/// twice" literally the same node.
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

/// Identity of a node: the address of its `Arc` allocation.
///
/// Sound as a map key only while an `Arc` to the node is held — which the
/// engine does for the whole walk (the topological order owns one per node),
/// so no address can be recycled underneath it.
fn node_id(node: &Arc<Node>) -> usize {
    Arc::as_ptr(node) as usize
}

/// Free a graph **iteratively**.
///
/// The natural recursive drop of `inputs: Vec<Option<Arc<Node>>>` costs one
/// stack frame per node, so a 100k-op chain (a long unrolled RNN, a deep
/// residual stack) would overflow the stack when its head tensor goes out of
/// scope. Instead the node's parents move onto an explicit worklist: each
/// `Arc` whose last reference this drop removes is unwrapped, its own parents
/// are pushed, and the emptied `Node` is dropped with nothing left to recurse
/// into.
impl Drop for Node {
    fn drop(&mut self) {
        let mut work: Vec<Arc<Node>> = self.inputs.drain(..).flatten().collect();
        while let Some(parent) = work.pop() {
            // `Err` = other references remain; that `Arc` is simply released.
            if let Ok(mut node) = Arc::try_unwrap(parent) {
                work.extend(node.inputs.drain(..).flatten());
                // `node` drops here with `inputs` already empty, so its own
                // `Drop` finds no parents and cannot recurse.
            }
        }
    }
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
/// A non-float `output` is never traced: integers and booleans carry no
/// cotangent, so a cast to [`I64`](crate::DType::I64) ends the graph instead
/// of extending it with an unusable node.
pub(crate) fn record(
    op: &'static str,
    output: Tensor,
    inputs: &[&Tensor],
    backward: BackwardFn,
) -> Tensor {
    if !output.dtype().is_float() {
        return output;
    }
    let parents: Vec<Option<Arc<Node>>> = inputs.iter().map(|t| t.node().cloned()).collect();
    if parents.iter().all(Option::is_none) {
        // Nothing upstream is traced: recording here would build a node no
        // `backward` could ever reach a leaf through.
        return output;
    }
    let node = Arc::new(Node {
        op,
        key: None,
        inputs: parents,
        backward: Some(backward),
    });
    Tensor::from_parts_traced(output.storage().clone(), output.layout().clone(), node)
}

/// Build a leaf tensor: `value` with a fresh leaf [`Node`] carrying `key`, so
/// that using it in a computation records a graph rooted at `key`. Shares
/// `value`'s storage (no copy). Used by [`Param::get`](crate::nn::Param::get)
/// (cached leaf) and [`traced`].
///
/// One `Param` has **one** leaf node, so using it twice in a forward pass
/// (weight tying) reaches the same node along two paths and [`backward`]
/// accumulates both contributions under the one `key`.
pub(crate) fn make_leaf(value: Tensor, key: GradKey) -> Tensor {
    let node = Arc::new(Node {
        op: "leaf",
        key: Some(key),
        inputs: Vec::new(),
        backward: None,
    });
    Tensor::from_parts_traced(value.storage().clone(), value.layout().clone(), node)
}

/// Turn `t` into a traced leaf for grad-wrt-input:
/// `let xt = x.traced(); let y = f(&xt)?; let g = y.backward()?;
/// g.wrt_input(&xt)`. The **returned** binding must be the one used in the
/// computation *and* in the lookup.
///
/// # Errors
///
/// [`Error::InvalidArg`] (`op: "traced"`) if `t` already carries a graph —
/// double-tracing is a bug — or if `t` is not a float tensor, since only
/// floating-point values have gradients. ([`Error::NotTraced`] is reserved for
/// the opposite no-graph condition.)
pub(crate) fn traced(t: &Tensor) -> Result<Tensor> {
    if t.node().is_some() {
        return Err(Error::InvalidArg {
            op: "traced",
            msg: "tensor is already traced; traced() takes an untraced value \
                  (use the binding it returned, not a fresh traced() call)"
                .to_string(),
        });
    }
    if !t.dtype().is_float() {
        return Err(Error::InvalidArg {
            op: "traced",
            msg: format!(
                "traced() requires a float dtype, got {} (only floating-point \
                 tensors have gradients)",
                t.dtype()
            ),
        });
    }
    Ok(make_leaf(t.clone(), GradKey::fresh()))
}

/// Run reverse-mode autodiff from `t`, returning a fresh [`Grads`].
///
/// # The seed
///
/// The walk starts from a cotangent of **ones** shaped like `t`, so
/// `t.backward()` differentiates `sum(t)` with respect to every leaf. For the
/// scalar `t` of a training loop — the only shape the flagship loop uses —
/// that is the ordinary `d loss / d θ`.
///
/// # Purity and repeated use
///
/// The graph is never mutated: `backward` is an immutable walk over the `Arc`
/// DAG, iterative (an explicit worklist, so a 100k-node chain costs heap, not
/// stack), and may be run twice or from several threads on the same graph.
/// A leaf reached along several paths — a tied weight, a value used twice —
/// accumulates every contribution under its one [`GradKey`].
///
/// # Errors
///
/// [`Error::NotTraced`] (`op: "backward"`) if `t` carries no graph. A
/// graph-less backward is loud, never an empty [`Grads`].
/// Whatever a backward closure's tensor ops report propagates out of here
/// unchanged, naming the op that failed: a backend failure during the backward
/// pass is that error, never a silently missing gradient that resurfaces as a
/// misleading [`Error::MissingGrad`] at the next `step`.
pub(crate) fn backward(t: &Tensor) -> Result<Grads> {
    let root = t
        .node()
        .cloned()
        .ok_or(Error::NotTraced { op: "backward" })?;
    // A deferred root can outlive the guard that created it. Realize and
    // synchronize only when this root is still pending; ordinary eager or
    // already-materialized paths remain asynchronous.
    crate::lazy::flush_tensors(&[t])?;
    let seed = Tensor::ones(t.dims(), t.dtype(), &t.device())?;

    let mut cotangents: HashMap<usize, Accumulated> = HashMap::new();
    accumulate_wide(&mut cotangents, node_id(&root), seed)?;
    let order = topological_order(root);
    let mut grads: HashMap<GradKey, Accumulated> = HashMap::new();

    // `order` is a reverse topological order rooted at `t`: every consumer of
    // a node precedes it, so a node's cotangent is complete the moment it is
    // popped and can be handed to the closure once.
    for node in order {
        let Some(cotangent) = cotangents.remove(&node_id(&node)) else {
            // This node feeds nothing that reached the root along a
            // gradient-carrying path.
            continue;
        };
        if let Some(key) = node.key {
            merge_accumulated(&mut grads, key, cotangent)?;
            // Leaves have no parents and no backward closure.
            continue;
        }
        let Some(backward) = node.backward.as_ref() else {
            continue;
        };
        let cotangent = cotangent.finish()?;
        for (parent, contribution) in node.inputs.iter().zip(backward(&cotangent)?) {
            let (Some(parent), Some(contribution)) = (parent, contribution) else {
                continue;
            };
            // Detach defensively: a cotangent must never carry a graph of its
            // own, whatever a backward closure built it from.
            accumulate_wide(&mut cotangents, node_id(parent), contribution.detach())?;
        }
    }
    let result = Grads { grads };
    let has_pending = result
        .grads
        .values()
        .any(|accumulated| accumulated.value.storage().is_pending());
    if has_pending {
        let values: Vec<&Tensor> = result
            .grads
            .values()
            .map(|accumulated| &accumulated.value)
            .collect();
        crate::lazy::flush_tensors(&values)?;
    }
    Ok(result)
}

/// The nodes reachable from `root`, in reverse topological order (`root`
/// first, every node before its parents).
///
/// An explicit two-phase stack — push-unvisited, then emit-on-second-visit —
/// so depth costs heap rather than stack frames.
fn topological_order(root: Arc<Node>) -> Vec<Arc<Node>> {
    let mut post_order: Vec<Arc<Node>> = Vec::new();
    let mut visited: HashSet<usize> = HashSet::new();
    let mut stack: Vec<(Arc<Node>, bool)> = vec![(root, false)];
    while let Some((node, expanded)) = stack.pop() {
        if expanded {
            post_order.push(node);
            continue;
        }
        if !visited.insert(node_id(&node)) {
            continue;
        }
        stack.push((Arc::clone(&node), true));
        for parent in node.inputs.iter().flatten() {
            stack.push((Arc::clone(parent), false));
        }
    }
    // Post-order lists a node after all of its parents; reversing it yields a
    // topological order of the consumer→producer edges.
    post_order.reverse();
    post_order
}

/// One node's fan-in cotangent. Reduced values are held in F32 until every
/// incoming edge has contributed. Interior-node values narrow once before the
/// backward closure runs; leaf values remain wide inside [`Grads`] until they
/// are observed or consumed. Keeping the original dtype beside the tensor
/// avoids adding dtype metadata to the crate-private graph-node contract.
struct Accumulated {
    value: Tensor,
    dtype: DType,
}

impl Accumulated {
    fn from_tensor(value: Tensor) -> Accumulated {
        Accumulated {
            dtype: value.dtype(),
            value,
        }
    }

    fn new(value: Tensor) -> Result<Accumulated> {
        let mut accumulated = Accumulated::from_tensor(value);
        accumulated.ensure_wide()?;
        Ok(accumulated)
    }

    fn ensure_wide(&mut self) -> Result<()> {
        let accumulation_dtype = self.dtype.accumulation_dtype();
        if self.value.dtype() != accumulation_dtype {
            self.value = self.value.to_dtype(accumulation_dtype)?;
        }
        Ok(())
    }

    fn add(&mut self, value: Tensor) -> Result<()> {
        if value.dtype() != self.dtype {
            return Err(Error::DTypeMismatch {
                op: "backward",
                expected: self.dtype,
                got: value.dtype(),
            });
        }
        let value = if self.value.dtype() != value.dtype() {
            value.to_dtype(self.value.dtype())?
        } else {
            value
        };
        self.value = self.value.add(&value)?;
        Ok(())
    }

    fn merge(&mut self, mut other: Accumulated) -> Result<()> {
        if other.dtype != self.dtype {
            return Err(Error::DTypeMismatch {
                op: "backward",
                expected: self.dtype,
                got: other.dtype,
            });
        }
        self.ensure_wide()?;
        other.ensure_wide()?;
        self.value = self.value.add(&other.value)?;
        Ok(())
    }

    fn scale(&mut self, factor: f64) -> Result<()> {
        self.ensure_wide()?;
        self.value = self.value.mul_scalar(factor)?;
        Ok(())
    }

    fn observed(&self) -> Result<Tensor> {
        Accumulated {
            value: self.value.clone(),
            dtype: self.dtype,
        }
        .finish()
    }

    fn wide(&self) -> Result<Tensor> {
        let mut accumulated = Accumulated {
            value: self.value.clone(),
            dtype: self.dtype,
        };
        accumulated.ensure_wide()?;
        Ok(accumulated.value)
    }

    fn finish(self) -> Result<Tensor> {
        if self.value.dtype() == self.dtype {
            Ok(self.value)
        } else {
            self.value.to_dtype(self.dtype)
        }
    }
}

fn accumulate_wide<K: std::hash::Hash + Eq>(
    map: &mut HashMap<K, Accumulated>,
    key: K,
    value: Tensor,
) -> Result<()> {
    match map.remove(&key) {
        Some(mut existing) => {
            existing.add(value)?;
            map.insert(key, existing);
        }
        None => {
            map.insert(key, Accumulated::new(value)?);
        }
    }
    Ok(())
}

fn merge_accumulated<K: std::hash::Hash + Eq>(
    map: &mut HashMap<K, Accumulated>,
    key: K,
    value: Accumulated,
) -> Result<()> {
    match map.remove(&key) {
        Some(mut existing) => {
            existing.merge(value)?;
            map.insert(key, existing);
        }
        None => {
            map.insert(key, value);
        }
    }
    Ok(())
}

/// The result of [`Tensor::backward`](crate::Tensor::backward): the
/// gradients, as a **linear** value (§5).
///
/// `Grads` is deliberately **not `Clone`** and is `#[must_use]`: the
/// optimizer consumes it by move (`opt.step(&mut model, grads)`), so reusing a
/// moved value is a compile error and forgetting to use it is a warning.
/// Micro-batch accumulation and clipping are explicit linear
/// pipelines (`acc = acc.merge(step)?`, `grads.clip_norm(1.0)?`).
///
/// # Examples
///
/// ```
/// use rstorch::prelude::*;
///
/// let x = Tensor::from_vec(vec![2.0f32], [1], &Device::Cpu)?;
/// let param = Param::new(x);
/// let y = param.get(Mode::TRAIN).mul_scalar(3.0)?;
/// let grads = y.backward()?;
/// assert_eq!(grads.wrt(&param)?.to_vec::<f32>()?, vec![3.0]);
/// # Ok::<(), rstorch::Error>(())
/// ```
#[must_use]
pub struct Grads {
    grads: HashMap<GradKey, Accumulated>,
}

impl Grads {
    /// Remove and return the gradient for `key` in its **accumulation** dtype
    /// (the optimizer drain path).
    ///
    /// The engine accumulates an F16/BF16 parameter's gradient in F32 and only
    /// narrows it on the way out. The optimizer immediately widens it again, so
    /// draining through [`take`](Self::take) would round-trip
    /// F32 → F16 → F32 for no reason: a gradient that legitimately exceeds
    /// F16's range becomes `inf` (and then `NaN` once Adam divides), and one
    /// that merely exceeds F16's *spacing* is silently rounded. Handing over
    /// the value that was already computed avoids both.
    ///
    /// [`wrt`](Self::wrt) deliberately keeps narrowing: a gradient a *caller*
    /// inspects matches its parameter's dtype, as in `PyTorch`.
    pub(crate) fn take_wide(&mut self, key: GradKey) -> Result<Option<Tensor>> {
        self.grads
            .remove(&key)
            .map(|accumulated| accumulated.wide())
            .transpose()
    }

    /// Validate a gradient's metadata without materializing the public,
    /// parameter-dtype observation returned by [`wrt`](Self::wrt).
    ///
    /// Optimizers use this before draining the already-wide accumulation so
    /// reduced-precision gradients are not narrowed merely for validation.
    pub(crate) fn validate_wrt(
        &self,
        key: GradKey,
        expected: &Tensor,
        op: &'static str,
    ) -> Result<()> {
        let Some(accumulated) = self.grads.get(&key) else {
            return Err(Error::NotTraced { op });
        };
        if accumulated.value.dims() != expected.dims() {
            return Err(Error::ShapeMismatch {
                op,
                lhs: expected.shape().clone(),
                rhs: accumulated.value.shape().clone(),
            });
        }
        if accumulated.dtype != expected.dtype() {
            return Err(Error::DTypeMismatch {
                op,
                expected: expected.dtype(),
                got: accumulated.dtype,
            });
        }
        if accumulated.value.device() != expected.device() {
            return Err(Error::DeviceMismatch {
                op,
                expected: expected.device(),
                got: accumulated.value.device(),
            });
        }
        Ok(())
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
    /// accumulation `acc = acc.merge(step)?`.
    ///
    /// # Errors
    ///
    /// Whatever the element-wise add reports for a key both sides hold with
    /// incompatible shapes, dtypes or devices — that is, gradients from two
    /// genuinely different models.
    pub fn merge(self, other: Grads) -> Result<Grads> {
        let mut grads = self.grads;
        for (key, value) in other.grads {
            merge_accumulated(&mut grads, key, value)?;
        }
        Ok(Grads { grads })
    }

    /// Scale every gradient by `factor`, consuming `self` — the second half of
    /// micro-batch accumulation (`acc.scale(1.0 / steps as f64)?`).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArg`] (`op: "scale"`) if `factor` is not finite, and
    /// whatever the scalar multiply reports.
    pub fn scale(self, factor: f64) -> Result<Grads> {
        if !factor.is_finite() {
            return Err(Error::InvalidArg {
                op: "scale",
                msg: format!("scale factor must be finite, got {factor}"),
            });
        }
        let mut grads = self.grads;
        for value in grads.values_mut() {
            value.scale(factor)?;
        }
        Ok(Grads { grads })
    }

    /// The **global** L2 norm over every gradient at once (`√Σ‖gₖ‖²`): the
    /// number to log for a step, and exactly the value
    /// [`clip_norm`](Self::clip_norm) compares its budget against. Read it
    /// *before* clipping — that is the pre-clipping norm a training log wants,
    /// and `clip_norm` consumes the `Grads`:
    ///
    /// # Examples
    ///
    /// ```
    /// use rstorch::prelude::*;
    ///
    /// let param = Param::new(Tensor::from_vec(vec![1.0f32], [1], &Device::Cpu)?);
    /// let grads = param.get(Mode::TRAIN).mul_scalar(3.0)?.backward()?;
    /// let grad_norm = grads.norm()?; // log this, before clipping
    /// assert!((grad_norm - 3.0).abs() < 1e-6);
    /// let grads = grads.clip_norm(1.0)?;
    /// // Scaling by `1/3` in f32 lands near 1.0, not exactly on it — which
    /// // is why the sibling unit test compares with a tolerance too.
    /// assert!((grads.norm()? - 1.0).abs() < 1e-6);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    ///
    /// An empty `Grads` has norm `0.0`. Unlike `clip_norm`, a non-finite norm
    /// is returned rather than rejected: a diverged step is precisely what a
    /// log is for.
    ///
    /// # Errors
    ///
    /// Whatever the widen/square/sum ops report.
    pub fn norm(&self) -> Result<f64> {
        self.global_norm()
    }

    /// Clip by **global** L2 norm to `max_norm`, consuming `self`: the norm is
    /// taken over every gradient at once (`√Σ‖gₖ‖²`), and if it exceeds
    /// `max_norm` all of them are scaled by `max_norm / norm`. A norm already
    /// within budget leaves the gradients untouched. The norm itself is
    /// [`norm`](Self::norm).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArg`] (`op: "clip_norm"`) if `max_norm` is not finite
    /// and positive, or if the global norm is not finite (a `NaN`/infinite
    /// gradient cannot be clipped into a meaningful one — the loud answer to a
    /// diverged step). Also whatever the square/sum/scale ops report.
    pub fn clip_norm(self, max_norm: f64) -> Result<Grads> {
        if !max_norm.is_finite() || max_norm <= 0.0 {
            return Err(Error::InvalidArg {
                op: "clip_norm",
                msg: format!("max_norm must be finite and positive, got {max_norm}"),
            });
        }
        let norm = self.global_norm()?;
        if !norm.is_finite() {
            return Err(Error::InvalidArg {
                op: "clip_norm",
                msg: format!("global gradient norm is not finite ({norm})"),
            });
        }
        if norm <= max_norm {
            return Ok(self);
        }
        self.scale(max_norm / norm)
    }

    /// The global L2 norm, shared by [`norm`](Self::norm) and
    /// [`clip_norm`](Self::clip_norm) so the value that is logged and the value
    /// that is clipped against cannot drift apart.
    fn global_norm(&self) -> Result<f64> {
        // Accumulate the per-gradient sums of squares **on the device**, then
        // read once. Calling `item()` per gradient made the global norm cost
        // one full accelerator round trip per parameter — the single largest
        // source of host synchronization left in a training step, and pure
        // overhead, since none of the intermediate values are wanted on the
        // host.
        //
        // Partials are grouped by accumulation dtype because a mixed-precision
        // model yields both `f32` (from `f16`/`bf16`/`f32` gradients) and `f64`
        // partials, which cannot be added to each other on the device. In
        // practice that is one group, hence one read; a mixed model reads once
        // per dtype, still a constant rather than a per-parameter cost.
        let mut partials: Vec<(crate::DType, Tensor)> = Vec::new();
        for value in self.grads.values() {
            let wide = value.wide()?;
            let squared = wide.mul(&wide)?.sum_all()?;
            match partials
                .iter_mut()
                .find(|(dtype, _)| *dtype == squared.dtype())
            {
                Some((_, acc)) => *acc = acc.add(&squared)?,
                None => partials.push((squared.dtype(), squared)),
            }
        }
        let mut total = 0.0f64;
        for (_, partial) in &partials {
            total += partial.item()?;
        }
        Ok(total.sqrt())
    }

    /// The gradient with respect to `param` (a clone of the stored handle;
    /// gradients are cheap `Arc`-backed tensors).
    ///
    /// # Errors
    ///
    /// [`Error::NotTraced`] (`op: "wrt"`) when this `Grads` holds nothing for
    /// the parameter: it was frozen, or it was read under a non-recording
    /// [`Mode`](crate::nn::Mode), or its value never reached the tensor
    /// `backward()` was called on.
    pub fn wrt(&self, param: &crate::nn::Param) -> Result<Tensor> {
        self.grads
            .get(&param.grad_key())
            .map(Accumulated::observed)
            .transpose()?
            .ok_or(Error::NotTraced { op: "wrt" })
    }

    /// The gradient with respect to a traced input.
    /// `input` must be the tensor returned by
    /// [`Tensor::traced`](crate::Tensor::traced) **and** the one used in the
    /// computation.
    ///
    /// # Errors
    ///
    /// - [`Error::NotTraced`] (`op: "wrt_input"`) if `input` carries no graph
    ///   at all — the classic slip of looking the *original* tensor up instead
    ///   of the binding [`traced`](crate::Tensor::traced) returned.
    /// - [`Error::InvalidArg`] (`op: "wrt_input"`) if `input` is an interior
    ///   value (the output of an op) rather than a traced leaf, or if it is a
    ///   traced leaf that never reached the differentiated tensor.
    pub fn wrt_input(&self, input: &Tensor) -> Result<Tensor> {
        let node = input.node().ok_or(Error::NotTraced { op: "wrt_input" })?;
        let key = node.key.ok_or_else(|| Error::InvalidArg {
            op: "wrt_input",
            msg: format!(
                "expected the leaf returned by traced(), got the output of `{}`",
                node.op
            ),
        })?;
        self.grads
            .get(&key)
            .map(Accumulated::observed)
            .transpose()?
            .ok_or_else(|| Error::InvalidArg {
                op: "wrt_input",
                msg: "no gradient for this traced input: the tensor traced() \
                  returned must be the one used in the computation"
                    .to_string(),
            })
    }
}

/// Reaching inside `Grads` from the test suites.
///
/// These are `#[cfg(test)]` and live in their own block so the production
/// `impl Grads` above reads as the API a caller actually has: the engine builds
/// `Grads` from its own accumulated map and the optimizer drains it through
/// [`Grads::take_wide`], so none of these three exist outside tests.
#[cfg(test)]
impl Grads {
    /// Build from a key→gradient map.
    pub(crate) fn from_pairs(grads: HashMap<GradKey, Tensor>) -> Grads {
        Grads {
            grads: grads
                .into_iter()
                .map(|(key, value)| (key, Accumulated::from_tensor(value)))
                .collect(),
        }
    }

    /// Remove and return the gradient for `key` in the parameter's own dtype.
    pub(crate) fn take(&mut self, key: GradKey) -> Result<Option<Tensor>> {
        self.grads.remove(&key).map(Accumulated::finish).transpose()
    }

    /// Whether a gradient is present for `key`.
    pub(crate) fn contains(&self, key: GradKey) -> bool {
        self.grads.contains_key(&key)
    }
}

#[cfg(test)]
mod tests;
