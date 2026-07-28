//! Autograd: tracing is data flow, gradients are a linear value
//! (exploration §4.3).
//!
//! **Contract module** (T01) + **engine** (T30). T01 defined the autograd
//! *types* and the *seams* the rest of the crate codes against; T30 filled the
//! bodies without changing a signature:
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
//!   cotangent as soon as its node has been processed. Nothing in the graph is
//!   mutated, so the same graph can be differentiated twice and from several
//!   threads.
//! - **[`Drop`] is iterative too.** Dropping the head of a long chain would
//!   otherwise recurse once per node; [`Node`]'s `Drop` moves the parents onto
//!   a worklist instead.
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
//! separately so the rule is expressible. A strong-count leak test gates it.

// `Node::key`/`Grads::{take, contains}` are consumed by T44 (the optimizer);
// the integrator removes this allow once that lands.
#![allow(dead_code)]

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
/// output, produce the cotangent contribution for each input in order
/// (`None` where an input needs no gradient). `Send + Sync` so the immutable
/// graph is shareable across threads (`backward` is a pure function over it).
pub(crate) type BackwardFn = Box<dyn Fn(&Tensor) -> Vec<Option<Tensor>> + Send + Sync>;

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

/// Turn `t` into a traced leaf for grad-wrt-input (exploration §4.3):
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
/// graph-less backward is loud, never an empty [`Grads`] (exploration §4.3).
/// Whatever a backward closure's tensor ops report surfaces as a *missing*
/// contribution rather than an error, because [`BackwardFn`] yields
/// `Option<Tensor>`; only the engine's own cotangent accumulation and seed
/// construction can fail here.
pub(crate) fn backward(t: &Tensor) -> Result<Grads> {
    let root = t
        .node()
        .cloned()
        .ok_or(Error::NotTraced { op: "backward" })?;
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
        for (parent, contribution) in node.inputs.iter().zip(backward(&cotangent)) {
            let (Some(parent), Some(contribution)) = (parent, contribution) else {
                continue;
            };
            // Detach defensively: a cotangent must never carry a graph of its
            // own, whatever a backward closure built it from.
            accumulate_wide(&mut cotangents, node_id(parent), contribution.detach())?;
        }
    }
    Ok(Grads { grads })
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
/// gradients, as a **linear** value (exploration §4.3, §5).
///
/// `Grads` is deliberately **not `Clone`** and is `#[must_use]`: the
/// optimizer consumes it by move (`opt.step(&mut model, grads)`), so applying
/// the same gradients twice is a compile error and forgetting to step is a
/// warning. Micro-batch accumulation and clipping are explicit linear
/// pipelines (`acc = acc.merge(step)?`, `grads.clip_norm(1.0)?`).
#[must_use]
pub struct Grads {
    grads: HashMap<GradKey, Accumulated>,
}

impl Grads {
    /// Build from a key→gradient map (the engine's output).
    pub(crate) fn from_pairs(grads: HashMap<GradKey, Tensor>) -> Grads {
        Grads {
            grads: grads
                .into_iter()
                .map(|(key, value)| (key, Accumulated::from_tensor(value)))
                .collect(),
        }
    }

    /// Remove and return the gradient for `key` (optimizer drain path).
    pub(crate) fn take(&mut self, key: GradKey) -> Result<Option<Tensor>> {
        self.grads.remove(&key).map(Accumulated::finish).transpose()
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

    /// Clip by **global** L2 norm to `max_norm`, consuming `self`: the norm is
    /// taken over every gradient at once (`√Σ‖gₖ‖²`), and if it exceeds
    /// `max_norm` all of them are scaled by `max_norm / norm`. A norm already
    /// within budget leaves the gradients untouched.
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
        let mut total = 0.0f64;
        for value in self.grads.values() {
            let wide = value.wide()?;
            total += wide.mul(&wide)?.sum_all()?.item()?;
        }
        let norm = total.sqrt();
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

    /// The gradient with respect to a traced input (exploration §4.3).
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::dtype::DType;
    use crate::nn::{Mode, Param};
    use std::sync::Weak;

    const CPU: Device = Device::Cpu;

    fn t(data: &[f32], shape: impl Into<crate::shape::Shape>) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape, &CPU).unwrap()
    }

    fn v(t: &Tensor) -> Vec<f32> {
        t.to_vec::<f32>().unwrap()
    }

    /// Element-wise comparison with an f32-sized slack, for the cases whose
    /// expected values are not exactly representable.
    fn assert_close(got: &Tensor, expected: &[f32]) {
        let got = v(got);
        assert_eq!(got.len(), expected.len(), "{got:?} vs {expected:?}");
        for (g, e) in got.iter().zip(expected) {
            assert!((g - e).abs() < 1e-6, "{got:?} vs {expected:?}");
        }
    }

    // ------------------------------------------------------------------
    // record / make_leaf / traced
    // ------------------------------------------------------------------

    #[test]
    fn record_is_inert_when_no_input_is_traced() {
        let a = t(&[1.0, 2.0], [2]);
        let b = t(&[3.0, 4.0], [2]);
        let out = a.add(&b).unwrap();
        assert!(out.node().is_none());
        assert!(out.backward().is_err());
    }

    #[test]
    fn record_traces_as_soon_as_one_input_carries_a_node() {
        let x = t(&[1.0, 2.0], [2]).traced().unwrap();
        let c = t(&[3.0, 4.0], [2]);
        assert!(x.add(&c).unwrap().node().is_some());
        assert!(c.add(&x).unwrap().node().is_some());
        // …and a value derived from it stays traced.
        assert!(
            x.sigmoid()
                .unwrap()
                .mul_scalar(2.0)
                .unwrap()
                .node()
                .is_some()
        );
    }

    #[test]
    fn record_refuses_to_trace_a_non_float_output() {
        // Casting out of float ends the graph: an i64 value has no cotangent.
        let x = t(&[1.5, -2.5], [2]).traced().unwrap();
        let ints = x.to_dtype(DType::I64).unwrap();
        assert!(ints.node().is_none());
        assert!(matches!(
            ints.backward(),
            Err(Error::NotTraced { op: "backward" })
        ));
    }

    #[test]
    fn make_leaf_shares_storage_and_carries_the_key() {
        let value = t(&[1.0, 2.0, 3.0], [3]);
        let key = GradKey::fresh();
        let leaf = make_leaf(value.clone(), key);
        assert_eq!(leaf.node().unwrap().key, Some(key));
        assert_eq!(v(&leaf), v(&value));
        assert_eq!(leaf.dims(), value.dims());
    }

    #[test]
    fn traced_rejects_double_tracing_and_non_float_dtypes() {
        let x = t(&[1.0], [1]);
        let xt = x.traced().unwrap();
        assert!(matches!(
            xt.traced(),
            Err(Error::InvalidArg { op: "traced", .. })
        ));
        let ints = Tensor::from_vec(vec![1i64, 2], [2], &CPU).unwrap();
        assert!(matches!(
            ints.traced(),
            Err(Error::InvalidArg { op: "traced", .. })
        ));
        // `detach` undoes tracing, so the detached value may be traced again.
        assert!(xt.detach().traced().is_ok());
    }

    // ------------------------------------------------------------------
    // backward
    // ------------------------------------------------------------------

    #[test]
    fn backward_on_a_graph_less_tensor_is_not_traced() {
        let x = t(&[1.0, 2.0], [2]);
        assert!(matches!(
            x.backward(),
            Err(Error::NotTraced { op: "backward" })
        ));
    }

    #[test]
    fn backward_of_a_small_chain_matches_the_closed_form() {
        // y = sum(3·x²) → dy/dx = 6·x
        let x = t(&[1.0, -2.0, 0.5], [3]);
        let xt = x.traced().unwrap();
        let y = xt.mul(&xt).unwrap().mul_scalar(3.0).unwrap();
        let grads = y.backward().unwrap();
        assert_eq!(grads.len(), 1);
        assert_eq!(v(&grads.wrt_input(&xt).unwrap()), vec![6.0, -12.0, 3.0]);
    }

    #[test]
    fn backward_seeds_with_ones_so_it_differentiates_the_sum() {
        // f = sum(x) over a [2, 2] tensor: every entry gets exactly 1.
        let xt = t(&[1.0, 2.0, 3.0, 4.0], [2, 2]).traced().unwrap();
        let g = xt.mul_scalar(2.0).unwrap().backward().unwrap();
        let g = g.wrt_input(&xt).unwrap();
        assert_eq!(g.dims(), &[2, 2]);
        assert_eq!(v(&g), vec![2.0; 4]);
    }

    #[test]
    fn repeated_use_of_one_leaf_accumulates() {
        // Three uses of the same leaf: x·x + x → 2x + 1.
        let x = t(&[2.0, -1.0], [2]);
        let xt = x.traced().unwrap();
        let y = xt.mul(&xt).unwrap().add(&xt).unwrap();
        let g = y.backward().unwrap();
        assert_eq!(v(&g.wrt_input(&xt).unwrap()), vec![5.0, -1.0]);
    }

    #[test]
    fn a_tied_param_accumulates_both_contributions() {
        // One `Param` read twice in a forward pass is one leaf node reached
        // along two paths — the weight-tying case.
        let p = Param::new(t(&[3.0, 4.0], [2]));
        let a = p.get(Mode::TRAIN);
        let b = p.get(Mode::TRAIN);
        // The two reads are the same node…
        assert!(std::ptr::eq(
            Arc::as_ptr(a.node().unwrap()),
            Arc::as_ptr(b.node().unwrap())
        ));
        // …so d(sum(a·b))/dp = 2p.
        let g = a.mul(&b).unwrap().backward().unwrap();
        assert_eq!(g.len(), 1);
        assert_eq!(v(&g.wrt(&p).unwrap()), vec![6.0, 8.0]);
    }

    #[test]
    fn a_diamond_sums_both_paths_before_the_closure_runs() {
        // y = (2x) + (3x) → dy/dx = 5, and the shared node must be visited
        // once, after both branches have contributed.
        let xt = t(&[1.0, 1.0], [2]).traced().unwrap();
        let shared = xt.mul_scalar(1.0).unwrap();
        let y = shared
            .mul_scalar(2.0)
            .unwrap()
            .add(&shared.mul_scalar(3.0).unwrap())
            .unwrap();
        let g = y.backward().unwrap();
        assert_eq!(v(&g.wrt_input(&xt).unwrap()), vec![5.0, 5.0]);
    }

    #[test]
    fn reduced_graph_fan_in_accumulates_repeated_adds_in_f32() {
        for dtype in [DType::F16, DType::BF16] {
            let x = Tensor::ones((), dtype, &CPU).unwrap().traced().unwrap();
            let mut y = x.mul_scalar(1.0).unwrap();
            for _ in 1..4096 {
                y = y.add(&x.mul_scalar(1.0).unwrap()).unwrap();
            }
            let grad = y.backward().unwrap().wrt_input(&x).unwrap();
            assert_eq!(grad.dtype(), dtype);
            assert_eq!(grad.item().unwrap(), 4096.0);
        }
    }

    #[test]
    fn several_leaves_land_under_their_own_keys() {
        let a = t(&[1.0, 2.0], [2]).traced().unwrap();
        let b = t(&[3.0, 4.0], [2]).traced().unwrap();
        let g = a.mul(&b).unwrap().backward().unwrap();
        assert_eq!(g.len(), 2);
        assert_eq!(v(&g.wrt_input(&a).unwrap()), vec![3.0, 4.0]);
        assert_eq!(v(&g.wrt_input(&b).unwrap()), vec![1.0, 2.0]);
    }

    #[test]
    fn backward_is_pure_and_repeatable() {
        let xt = t(&[1.0, 2.0], [2]).traced().unwrap();
        let y = xt.mul(&xt).unwrap();
        let first = y.backward().unwrap();
        let second = y.backward().unwrap();
        assert_eq!(
            v(&first.wrt_input(&xt).unwrap()),
            v(&second.wrt_input(&xt).unwrap())
        );
    }

    #[test]
    fn broadcast_gradients_reduce_back_to_the_operand_shape() {
        let row = t(&[1.0, 2.0, 3.0], [1, 3]).traced().unwrap();
        let m = t(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [2, 3]);
        let g = row.mul(&m).unwrap().backward().unwrap();
        let g = g.wrt_input(&row).unwrap();
        assert_eq!(g.dims(), &[1, 3]);
        assert_eq!(v(&g), vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn reduced_broadcast_gradients_keep_multi_axis_sum_to_wide() {
        for dtype in [DType::F16, DType::BF16] {
            let (width, values) = if dtype == DType::F16 {
                let width = 65_520;
                let mut values = vec![1.0f32; width];
                values.extend(vec![-1.0; width]);
                (width, values)
            } else {
                let width = 257;
                let mut values = vec![1.0f32; width];
                values.extend((0..width).map(|index| if index < 256 { -1.0 } else { 0.0 }));
                (width, values)
            };
            let leaf = Tensor::ones([1, 1], dtype, &CPU).unwrap().traced().unwrap();
            let weights = match dtype {
                DType::F16 => Tensor::from_vec(
                    values.into_iter().map(half::f16::from_f32).collect(),
                    [2, width],
                    &CPU,
                ),
                DType::BF16 => Tensor::from_vec(
                    values.into_iter().map(half::bf16::from_f32).collect(),
                    [2, width],
                    &CPU,
                ),
                _ => unreachable!(),
            }
            .unwrap();
            let grad = leaf
                .mul(&weights)
                .unwrap()
                .sum_all()
                .unwrap()
                .backward()
                .unwrap()
                .wrt_input(&leaf)
                .unwrap();
            let expected = if dtype == DType::F16 { 0.0 } else { 1.0 };
            assert_eq!(grad.item().unwrap(), expected);
        }
    }

    // ------------------------------------------------------------------
    // Mode / Param interaction
    // ------------------------------------------------------------------

    #[test]
    fn eval_mode_and_frozen_params_record_nothing() {
        let mut p = Param::new(t(&[1.0, 2.0], [2]));
        assert!(p.get(Mode::EVAL).node().is_none());
        assert!(p.get(Mode::TRAIN.frozen()).node().is_none());
        assert!(p.get(Mode::EVAL.recorded()).node().is_some());

        p.freeze();
        assert!(p.get(Mode::TRAIN).node().is_none());
        let out = p.get(Mode::TRAIN).mul_scalar(2.0).unwrap();
        assert!(out.node().is_none());
        assert!(out.backward().is_err());

        p.unfreeze();
        assert!(p.get(Mode::TRAIN).node().is_some());
    }

    #[test]
    fn param_set_rebuilds_the_leaf_and_leaves_a_live_graph_intact() {
        let mut p = Param::new(t(&[2.0], [1]));
        let y = p.get(Mode::TRAIN).mul(&p.get(Mode::TRAIN)).unwrap();
        p.set(t(&[10.0], [1])).unwrap();
        // The live graph kept the value it was built from — differentiating
        // it still yields 2·2, not 2·10 — and the identity is unchanged, so
        // the gradient is still found under the parameter's key.
        let g = y.backward().unwrap();
        assert_eq!(v(&g.wrt(&p).unwrap()), vec![4.0]);

        // The fresh leaf is a *different* node carrying the same key.
        let g2 = p
            .get(Mode::TRAIN)
            .mul(&p.get(Mode::TRAIN))
            .unwrap()
            .backward()
            .unwrap();
        assert_eq!(v(&g2.wrt(&p).unwrap()), vec![20.0]);
    }

    // ------------------------------------------------------------------
    // Grads — the linear surface
    // ------------------------------------------------------------------

    fn grads_of(x: &Tensor, scale: f64) -> (Tensor, Grads) {
        let xt = x.traced().unwrap();
        let g = xt.mul_scalar(scale).unwrap().backward().unwrap();
        (xt, g)
    }

    #[test]
    fn merge_unions_keys_and_sums_the_overlap() {
        let x = t(&[1.0, 1.0], [2]);
        let xt = x.traced().unwrap();
        let a = xt.mul_scalar(2.0).unwrap().backward().unwrap();
        let b = xt.mul_scalar(5.0).unwrap().backward().unwrap();
        let (yt, c) = grads_of(&t(&[1.0], [1]), 3.0);

        let merged = a.merge(b).unwrap().merge(c).unwrap();
        assert_eq!(merged.len(), 2);
        assert_eq!(v(&merged.wrt_input(&xt).unwrap()), vec![7.0, 7.0]);
        assert_eq!(v(&merged.wrt_input(&yt).unwrap()), vec![3.0]);
    }

    #[test]
    fn repeated_reduced_grads_merges_retain_the_wide_accumulator() {
        for dtype in [DType::F16, DType::BF16] {
            let x = Tensor::ones((), dtype, &CPU).unwrap().traced().unwrap();
            let one_grad = || x.mul_scalar(1.0).unwrap().backward().unwrap();
            let mut merged = one_grad();
            for _ in 1..4096 {
                merged = merged.merge(one_grad()).unwrap();
            }
            let grad = merged.wrt_input(&x).unwrap();
            assert_eq!(grad.dtype(), dtype);
            assert_eq!(grad.item().unwrap(), 4096.0);
        }
    }

    #[test]
    fn scale_multiplies_every_entry() {
        let (xt, g) = grads_of(&t(&[1.0, 1.0], [2]), 4.0);
        let g = g.scale(0.25).unwrap();
        assert_eq!(v(&g.wrt_input(&xt).unwrap()), vec![1.0, 1.0]);

        let (_, g) = grads_of(&t(&[1.0], [1]), 1.0);
        assert!(matches!(
            g.scale(f64::NAN),
            Err(Error::InvalidArg { op: "scale", .. })
        ));
    }

    #[test]
    fn clip_norm_only_bites_above_the_budget() {
        // Two leaves with gradients [3, 0] and [4]: global norm 5.
        let a = t(&[1.0, 1.0], [2]).traced().unwrap();
        let b = t(&[1.0], [1]).traced().unwrap();
        let build = || {
            let lhs = a.mul(&t(&[3.0, 0.0], [2])).unwrap();
            let rhs = b.mul_scalar(4.0).unwrap();
            lhs.sum_all().unwrap().add(&rhs.sum_all().unwrap()).unwrap()
        };

        // Under budget: untouched.
        let g = build().backward().unwrap().clip_norm(10.0).unwrap();
        assert_eq!(v(&g.wrt_input(&a).unwrap()), vec![3.0, 0.0]);
        assert_eq!(v(&g.wrt_input(&b).unwrap()), vec![4.0]);

        // Over budget: scaled by 1/5 to land on it.
        let g = build().backward().unwrap().clip_norm(1.0).unwrap();
        assert_close(&g.wrt_input(&a).unwrap(), &[0.6, 0.0]);
        assert_close(&g.wrt_input(&b).unwrap(), &[0.8]);

        let g = build().backward().unwrap();
        assert!(matches!(
            g.clip_norm(0.0),
            Err(Error::InvalidArg {
                op: "clip_norm",
                ..
            })
        ));
    }

    #[test]
    fn clip_norm_of_an_empty_grads_is_a_no_op() {
        let g = Grads::from_pairs(HashMap::new());
        assert!(g.is_empty());
        let g = g.clip_norm(1.0).unwrap();
        assert_eq!(g.len(), 0);
    }

    #[test]
    fn reduced_clip_norm_widens_before_square_and_sum() {
        for dtype in [DType::F16, DType::BF16] {
            let x = Tensor::ones([4096], dtype, &CPU).unwrap().traced().unwrap();
            let grads = x.sum_all().unwrap().backward().unwrap();
            let clipped = grads.clip_norm(32.0).unwrap().wrt_input(&x).unwrap();
            assert_eq!(clipped.dtype(), dtype);
            assert!(
                clipped
                    .to_dtype(DType::F32)
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .all(|&value| value == 0.5)
            );
        }
    }

    #[test]
    fn lookups_are_loud_about_the_wrong_binding() {
        let x = t(&[1.0, 2.0], [2]);
        let (xt, g) = grads_of(&x, 2.0);

        // The original, untraced tensor.
        assert!(matches!(
            g.wrt_input(&x),
            Err(Error::NotTraced { op: "wrt_input" })
        ));
        // An interior value rather than the leaf.
        let interior = xt.mul_scalar(1.0).unwrap();
        assert!(matches!(
            g.wrt_input(&interior),
            Err(Error::InvalidArg {
                op: "wrt_input",
                ..
            })
        ));
        // A leaf that took no part in the computation.
        let other = t(&[1.0], [1]).traced().unwrap();
        assert!(matches!(
            g.wrt_input(&other),
            Err(Error::InvalidArg {
                op: "wrt_input",
                ..
            })
        ));
        // An untrained parameter.
        let p = Param::new(t(&[1.0], [1]));
        assert!(matches!(g.wrt(&p), Err(Error::NotTraced { op: "wrt" })));
    }

    #[test]
    fn take_and_contains_drain_by_key() {
        let p = Param::new(t(&[2.0], [1]));
        let mut g = p
            .get(Mode::TRAIN)
            .mul_scalar(3.0)
            .unwrap()
            .backward()
            .unwrap();
        assert!(g.contains(p.grad_key()));
        assert_eq!(v(&g.take(p.grad_key()).unwrap().unwrap()), vec![3.0]);
        assert!(!g.contains(p.grad_key()));
        assert!(g.take(p.grad_key()).unwrap().is_none());
        assert!(g.is_empty());
    }

    // ------------------------------------------------------------------
    // Leak / capture discipline
    // ------------------------------------------------------------------

    #[test]
    fn a_dropped_sigmoid_chain_frees_every_node() {
        // The detached-output capture rule: `sigmoid`'s backward needs its own
        // output, and capturing the *traced* one would make node → closure →
        // node an `Arc` cycle that never frees. Strong counts are the witness.
        let leaf = t(&[0.5, -0.25, 0.75], [3]).traced().unwrap();
        let leaf_node = Arc::clone(leaf.node().unwrap());
        assert_eq!(Arc::strong_count(&leaf_node), 2);

        let mut y = leaf.clone();
        for _ in 0..16 {
            y = y.sigmoid().unwrap().tanh().unwrap().mul(&y).unwrap();
        }
        assert!(Arc::strong_count(&leaf_node) > 2);

        // Backward must not extend the graph either.
        let g = y.backward().unwrap();
        assert_eq!(g.len(), 1);
        let count_after_backward = Arc::strong_count(&leaf_node);
        drop(g);

        drop(y);
        drop(leaf);
        assert_eq!(
            Arc::strong_count(&leaf_node),
            1,
            "graph leaked (count after backward was {count_after_backward})"
        );
    }

    #[test]
    fn gradients_carry_no_graph_of_their_own() {
        let xt = t(&[0.5, 1.5], [2]).traced().unwrap();
        let g = xt.sigmoid().unwrap().backward().unwrap();
        assert!(g.wrt_input(&xt).unwrap().node().is_none());
    }

    // ------------------------------------------------------------------
    // 100k-node stress: neither the walk nor the drop may use the stack
    // ------------------------------------------------------------------

    /// A synthetic identity chain `depth` nodes deep over a rank-0 tensor,
    /// built without touching the op layer so the stress test measures the
    /// engine rather than 100k kernel launches. Returns the head tensor and a
    /// `Weak` to the leaf node (alive iff the graph has not been freed).
    fn identity_chain(depth: usize) -> (Tensor, Weak<Node>) {
        let value = Tensor::full((), 1.0, DType::F32, &CPU).unwrap();
        let leaf = Arc::new(Node {
            op: "leaf",
            key: Some(GradKey::fresh()),
            inputs: Vec::new(),
            backward: None,
        });
        let weak = Arc::downgrade(&leaf);
        let mut head = leaf;
        for _ in 0..depth {
            head = Arc::new(Node {
                op: "identity",
                key: None,
                inputs: vec![Some(head)],
                backward: Some(Box::new(|g: &Tensor| vec![Some(g.clone())])),
            });
        }
        let tensor =
            Tensor::from_parts_traced(value.storage().clone(), value.layout().clone(), head);
        (tensor, weak)
    }

    /// Run `body` on a thread with a deliberately small stack: a recursive
    /// drop or walk over 100k nodes overflows it, an iterative one does not.
    fn on_a_small_stack(body: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(512 * 1024)
            .spawn(body)
            .expect("spawn")
            .join()
            .expect("the engine must not recurse per node");
    }

    #[test]
    fn dropping_a_100k_node_graph_is_iterative() {
        on_a_small_stack(|| {
            let (head, leaf) = identity_chain(100_000);
            assert!(leaf.upgrade().is_some());
            drop(head);
            assert!(leaf.upgrade().is_none(), "the chain must be freed");
        });
    }

    #[test]
    fn backward_over_a_100k_node_graph_is_iterative() {
        on_a_small_stack(|| {
            let (head, _) = identity_chain(100_000);
            let grads = backward(&head).unwrap();
            assert_eq!(grads.len(), 1);
        });
    }

    #[test]
    fn a_long_chain_of_real_ops_walks_and_drops_iteratively() {
        // The same stress through the op layer, at a size where 20k kernel
        // launches stay cheap: `x` scaled by 1.0 twenty thousand times still
        // has gradient 1.
        on_a_small_stack(|| {
            let xt = Tensor::full((), 2.0, DType::F32, &CPU)
                .unwrap()
                .traced()
                .unwrap();
            let mut y = xt.clone();
            for _ in 0..20_000 {
                y = y.mul_scalar(1.0).unwrap();
            }
            let g = y.backward().unwrap();
            assert_eq!(g.wrt_input(&xt).unwrap().item().unwrap(), 1.0);
            drop(g);
            drop(y);
        });
    }

    #[test]
    fn the_graph_is_shareable_across_threads() {
        // `backward` is a pure function over an immutable `Arc` graph, so two
        // threads may differentiate the same tensor concurrently.
        let xt = t(&[1.0, 2.0, 3.0], [3]).traced().unwrap();
        let y = xt.mul(&xt).unwrap();
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    let y = y.clone();
                    s.spawn(move || y.backward().unwrap().len())
                })
                .collect();
            for h in handles {
                assert_eq!(h.join().unwrap(), 1);
            }
        });
    }
}
