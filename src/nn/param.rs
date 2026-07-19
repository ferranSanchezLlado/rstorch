//! [`Param`] — a trainable tensor with stable identity (exploration §4.3).
//!
//! **Contract file** (T01²/T40). T01 authors the linear-identity API and its
//! core behavior (get/set/freeze — the crux of the design's safety claims);
//! T40 owns the surrounding module runtime (`state_dict`, replication, etc.,
//! in [`util`](crate::nn) and the layer crates). No signature here changes
//! after T01.

use crate::autograd::{self, GradKey};
use crate::nn::Mode;
use crate::tensor::Tensor;

/// A trainable parameter: a value tensor plus a **stable identity** and one
/// cached traced leaf.
///
/// `Param` is deliberately **not `Clone`** — a model owning `Param`s is not
/// `Clone`, which is what makes the sanctioned replication path (construct +
/// `load_state_dict`) explicit rather than an accidental `.clone()`. Identity
/// is a crate-private `GradKey` minted at [`new`](Param::new): the optimizer keys
/// gradients by it, and a parameter used twice in one graph (weight tying)
/// accumulates both contributions under the one key.
pub struct Param {
    /// The current value. Replaced wholesale by [`set`](Param::set); old
    /// values stay alive inside any graph that captured them (`Arc`).
    value: Tensor,
    /// The cached traced leaf, rebuilt on every [`set`](Param::set) and
    /// returned by [`get`](Param::get) under a recording, non-frozen mode.
    leaf: Tensor,
    /// Stable gradient identity (see the type docs).
    key: GradKey,
    /// Per-parameter freeze flag (exploration §4.3): explicit, not a `Mode`
    /// side effect. A frozen param never traces and is skipped by the
    /// optimizer's completeness check.
    frozen: bool,
}

impl Param {
    /// Wrap `value` as a fresh trainable parameter with a new identity and
    /// its cached traced leaf.
    pub fn new(value: Tensor) -> Param {
        let key = GradKey::fresh();
        let leaf = autograd::make_leaf(value.clone(), key);
        Param {
            value,
            leaf,
            key,
            frozen: false,
        }
    }

    /// The **only** way a parameter enters a computation. Returns the cached
    /// traced leaf when `mode.records() && !self.is_frozen()`, otherwise the
    /// plain value (an eval/frozen access records nothing).
    pub fn get(&self, mode: Mode) -> Tensor {
        if mode.records() && !self.frozen {
            self.leaf.clone()
        } else {
            self.value.clone()
        }
    }

    /// Swap the value (optimizer step / checkpoint load) and rebuild the
    /// cached leaf. Any graph that already captured the previous value keeps
    /// it alive and consistent (`Param::set` swaps the `Arc`; exploration
    /// §5 "Param updated under a live graph").
    ///
    /// Returns [`Error`](crate::Error) if `value` is incompatible with the
    /// parameter (T40 defines the checks — e.g. shape/dtype invariants for
    /// optimizer state); T01's body performs the swap and leaf rebuild.
    pub fn set(&mut self, value: Tensor) -> crate::error::Result<()> {
        self.leaf = autograd::make_leaf(value.clone(), self.key);
        self.value = value;
        Ok(())
    }

    /// Mark this parameter frozen: [`get`](Param::get) never traces it and
    /// the optimizer skips it (legitimately, because freezing is explicit).
    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    /// Undo [`freeze`](Param::freeze).
    pub fn unfreeze(&mut self) {
        self.frozen = false;
    }

    /// Whether this parameter is frozen.
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// The current value tensor.
    pub fn value(&self) -> &Tensor {
        &self.value
    }

    /// The stable gradient identity (optimizer / `Grads` lookup key).
    // Consumed by T30 (`Grads::wrt`) and T44 (optimizer keying); the
    // integrator removes this allow once those land.
    #[allow(dead_code)]
    pub(crate) fn grad_key(&self) -> GradKey {
        self.key
    }
}

// Behavioral tests for get()/set()/freeze() require constructible tensors
// (T20) and a live engine (T30), so they live with those tasks. The
// `Param: !Clone` and step-by-move linearity guarantees are covered by the
// pinned-toolchain compile-fail suite T30 owns (exploration §6).
