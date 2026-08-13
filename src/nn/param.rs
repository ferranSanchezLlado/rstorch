//! [`Param`] — a trainable tensor with stable identity.
//!
//! The linear-identity API here — get/set/freeze — is the crux of the
//! design's safety claims. The surrounding module runtime (`state_dict`,
//! replication) lives in [`util`](crate::nn) and the layer modules.

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
///
/// # Examples
///
/// ```
/// use rstorch::prelude::*;
///
/// let w = Param::new(Tensor::from_vec(vec![1.0f32, 2.0], [2], &Device::Cpu)?);
/// let y = w.get(Mode::TRAIN).sum_all()?;
/// let grads = y.backward()?;
/// assert_eq!(grads.wrt(&w)?.to_vec::<f32>()?, vec![1.0, 1.0]);
/// # Ok::<(), rstorch::Error>(())
/// ```
pub struct Param {
    /// The current value. Replaced wholesale by [`set`](Param::set); old
    /// values stay alive inside any graph that captured them (`Arc`).
    value: Tensor,
    /// The cached traced leaf, rebuilt on every [`set`](Param::set) and
    /// returned by [`get`](Param::get) under a recording, non-frozen mode.
    leaf: Tensor,
    /// Stable gradient identity (see the type docs).
    key: GradKey,
    /// Per-parameter freeze flag: explicit, not a `Mode`
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
    /// # Errors
    ///
    /// [`Error::ShapeMismatch`](crate::Error::ShapeMismatch)
    /// (`op: "Param::set"`) if `value` has a different shape. A parameter's
    /// shape is fixed for its lifetime: the optimizer's moment buffers, the
    /// layer's arithmetic, and every `state_dict` consumer assume it, so a
    /// wrongly-shaped update is a bug caught here rather than a model that
    /// quietly changes geometry. Dtype and device *may* change — that is
    /// precisely what [`nn::to_dtype`](crate::nn::to_dtype) and
    /// [`nn::to_device`](crate::nn::to_device) do.
    pub fn set(&mut self, value: Tensor) -> crate::error::Result<()> {
        if value.shape() != self.value.shape() {
            return Err(crate::error::Error::ShapeMismatch {
                op: "Param::set",
                lhs: self.value.shape().clone(),
                rhs: value.shape().clone(),
            });
        }
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
    pub(crate) fn grad_key(&self) -> GradKey {
        self.key
    }
}

// The `Param: !Clone` and step-by-move linearity guarantees are covered by the
// pinned-toolchain compile-fail suite.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::dtype::DType;
    use crate::error::Error;

    fn t(values: &[f32]) -> Tensor {
        Tensor::from_vec(values.to_vec(), [values.len()], &Device::Cpu).unwrap()
    }

    fn values(t: &Tensor) -> Vec<f32> {
        t.to_vec::<f32>().unwrap()
    }

    /// `sum(w ⊙ w)`, whose gradient is `2w` — and which reaches the cached
    /// leaf twice, so it doubles as the tied-use accumulation probe.
    fn square_sum(p: &Param, mode: Mode) -> Tensor {
        let w = p.get(mode);
        w.mul(&w).unwrap().sum_all().unwrap()
    }

    #[test]
    fn get_traces_under_train_and_not_under_eval() {
        let p = Param::new(t(&[1.0, 2.0]));
        assert!(square_sum(&p, Mode::TRAIN).backward().is_ok());
        assert!(matches!(
            square_sum(&p, Mode::EVAL).backward(),
            Err(Error::NotTraced { .. })
        ));
    }

    // ---- the two off-diagonal modes --------------------

    #[test]
    fn eval_recorded_still_produces_gradients() {
        // Fine-tuning: eval *behavior* (dropout off, frozen BatchNorm stats)
        // with recording on. The recording axis is what `Param::get` reads.
        let mode = Mode::EVAL.recorded();
        assert!(!mode.is_training() && mode.records());

        let p = Param::new(t(&[1.0, 2.0, 3.0]));
        let grads = square_sum(&p, mode).backward().unwrap();
        assert_eq!(values(&grads.wrt(&p).unwrap()), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn train_frozen_records_nothing() {
        // MC-dropout sampling: train behavior, no graph.
        let mode = Mode::TRAIN.frozen();
        assert!(mode.is_training() && !mode.records());

        let p = Param::new(t(&[1.0, 2.0, 3.0]));
        let out = square_sum(&p, mode);
        assert_eq!(values(&out), vec![14.0]);
        assert!(matches!(
            out.backward(),
            Err(Error::NotTraced { op: "backward" })
        ));
    }

    #[test]
    fn tied_use_accumulates_under_one_key() {
        // `w` reaches the one cached leaf along two paths; the engine sums
        // both contributions (d/dw of w⊙w is 2w, not w).
        let p = Param::new(t(&[1.0, 2.0, 3.0]));
        let grads = square_sum(&p, Mode::TRAIN).backward().unwrap();
        assert_eq!(values(&grads.wrt(&p).unwrap()), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn frozen_param_never_traces_and_has_no_gradient() {
        let mut p = Param::new(t(&[1.0, 2.0]));
        p.freeze();
        assert!(p.is_frozen());
        assert!(matches!(
            square_sum(&p, Mode::TRAIN).backward(),
            Err(Error::NotTraced { .. })
        ));

        p.unfreeze();
        assert!(!p.is_frozen());
        let grads = square_sum(&p, Mode::TRAIN).backward().unwrap();
        assert_eq!(values(&grads.wrt(&p).unwrap()), vec![2.0, 4.0]);
    }

    #[test]
    fn set_swaps_the_value_and_rebuilds_the_leaf_under_one_identity() {
        let mut p = Param::new(t(&[1.0, 2.0]));
        // A live graph over the *old* value stays valid after the swap.
        let old_loss = square_sum(&p, Mode::TRAIN);
        p.set(t(&[10.0, 20.0])).unwrap();

        assert_eq!(values(p.value()), vec![10.0, 20.0]);
        assert_eq!(values(&p.get(Mode::EVAL)), vec![10.0, 20.0]);
        // The old graph still differentiates the value it captured…
        assert_eq!(
            values(&old_loss.backward().unwrap().wrt(&p).unwrap()),
            vec![2.0, 4.0]
        );
        // …and the identity is unchanged, so a fresh graph looks up the same
        // parameter with the new value.
        let grads = square_sum(&p, Mode::TRAIN).backward().unwrap();
        assert_eq!(values(&grads.wrt(&p).unwrap()), vec![20.0, 40.0]);
    }

    #[test]
    fn set_rejects_a_reshape() {
        let mut p = Param::new(t(&[1.0, 2.0]));
        let err = p.set(t(&[1.0, 2.0, 3.0])).unwrap_err();
        assert!(matches!(
            err,
            Error::ShapeMismatch {
                op: "Param::set",
                ..
            }
        ));
        // The rejected swap changed nothing.
        assert_eq!(values(p.value()), vec![1.0, 2.0]);
    }

    #[test]
    fn distinct_params_have_distinct_identities() {
        let a = Param::new(t(&[1.0]));
        let b = Param::new(t(&[2.0]));
        let grads = square_sum(&a, Mode::TRAIN).backward().unwrap();
        assert!(grads.wrt(&a).is_ok());
        assert!(matches!(grads.wrt(&b), Err(Error::NotTraced { .. })));
    }

    #[test]
    fn value_carries_no_graph() {
        // What `value()` exposes is a plain value, whatever the mode; only
        // `get()` hands out the traced leaf.
        let p = Param::new(Tensor::zeros([2], DType::F32, &Device::Cpu).unwrap());
        assert!(p.value().backward().is_err());
    }
}
