//! Losses: [`cross_entropy`](Tensor::cross_entropy) over `I64` class
//! labels — with the padding-aware
//! [`cross_entropy_ignore_index`](Tensor::cross_entropy_ignore_index)
//! spelling — and [`mse_loss`](Tensor::mse_loss).
//!
//! Both return a **rank-0 scalar** and both reduce by the **mean**: there is
//! no `Reduction` enum and no options struct — one spelling per operation. A caller who wants the sum multiplies the mean back up;
//! a caller who wants the per-element losses composes them from the op
//! surface directly.
//!
//! # Labels are tensors, and they stay on-device
//!
//! `cross_entropy(&logits, &labels_i64)` is one path: the
//! labels are an ordinary [`I64`](crate::DType::I64) tensor on the logits'
//! device, the ignore mask is built on-device by comparison, and nothing here
//! reads an element back to the host — not even the count of unmasked rows,
//! which is a device-side `sum` of the mask.
//!
//! # Mask-aware
//!
//! Two independent kinds of masking meet in this file, and both are safe:
//!
//! - **Masked classes.** A `-inf` logit (a forbidden vocabulary entry) has
//!   probability zero and receives no gradient, because the loss is built on
//!   the mask-aware [`log_softmax`](Tensor::log_softmax) rather than on
//!   `ln(softmax(x))`. A row that is entirely `-inf` yields `+inf`, not `NaN`.
//! - **Masked rows.** `cross_entropy_ignore_index` drops the rows whose label
//!   equals the sentinel (padding positions in a packed batch) from both the
//!   sum and the divisor, and their gradient is exactly zero. When *every*
//!   row is ignored the loss is `0` with a zero gradient, rather than `0/0`.
//!
//! # Backward
//!
//! Both losses are **fused**: the forward runs on `detach`ed inputs — so the
//! `log_softmax`/`gather`/`sum` composition records nothing — and the op
//! contributes a single node whose closure spells the classical gradient
//! directly:
//!
//! | forward | backward w.r.t. the float input |
//! |---|---|
//! | `cross_entropy` | `(softmax(x) − onehot(target)) · keep / count` |
//! | `mse_loss` | `2·(pred − target) / n` (and its negation for `target`) |
//!
//! That is one pass over the logits instead of a scatter-add graph, and it is
//! exact at the `-inf` boundary where `exp(log_softmax(x))` is a clean `0`.
//! Every tensor the closures capture is detached (the detached-output capture
//! rule); the labels are integral and take no gradient, so
//! only the float input is listed as a graph input.

use super::{require_dtype, same_device, same_dtype};
use crate::autograd::{self, BackwardFn};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// Validate the logits of a classification loss: a float `[rows, classes]`
/// matrix with at least one class. Returns `(rows, classes)`.
fn check_logits(op: &'static str, logits: &Tensor) -> Result<(usize, usize)> {
    if !logits.dtype().is_float() {
        return Err(Error::Unsupported {
            op,
            device: logits.device(),
            dtype: logits.dtype(),
        });
    }
    let (rows, classes) = match logits.dims() {
        &[rows, classes] => (rows, classes),
        d => {
            return Err(Error::RankMismatch {
                op,
                expected: 2,
                got: d.len(),
            });
        }
    };
    if classes == 0 {
        return Err(Error::InvalidArg {
            op,
            msg: format!(
                "{op} needs at least one class, got shape {}",
                logits.shape()
            ),
        });
    }
    Ok((rows, classes))
}

/// Validate the labels: a 1-D [`I64`](DType::I64) tensor of one label per
/// logits row, on the logits' device.
fn check_targets(op: &'static str, logits: &Tensor, targets: &Tensor, rows: usize) -> Result<()> {
    require_dtype(op, targets, DType::I64)?;
    same_device(op, logits, targets)?;
    match targets.dims() {
        &[n] if n == rows => Ok(()),
        &[_] => Err(Error::ShapeMismatch {
            op,
            lhs: logits.shape().clone(),
            rhs: targets.shape().clone(),
        }),
        d => Err(Error::RankMismatch {
            op,
            expected: 1,
            got: d.len(),
        }),
    }
}

/// Everything the `cross_entropy` backward needs, all of it detached: the
/// log-probabilities the forward already computed, the bounds-safe labels,
/// the float keep-mask (`None` when no row is ignored), the divisor, and the
/// class count.
struct CrossEntropyBackward {
    logp: Tensor,
    safe: Tensor,
    keep: Option<Tensor>,
    divisor: Tensor,
    classes: usize,
}

impl CrossEntropyBackward {
    /// `(softmax(x) − onehot(target)) · keep / count · g`, the cotangent of
    /// the logits. `g` is the rank-0 cotangent of the scalar loss and
    /// broadcasts over the whole matrix.
    fn grad(&self, g: &Tensor) -> Result<Tensor> {
        let device = self.logp.device();
        // onehot[r][c] == (safe[r] == c), in the logits' float dtype.
        let all = Tensor::index_range(self.classes, &device)?.reshape([1, self.classes])?;
        let onehot = self.safe.eq(&all)?.to_dtype(self.logp.dtype())?;
        let d = self.logp.exp()?.sub(&onehot)?;
        let output_dtype = d.dtype();
        let d = if d.dtype() != self.divisor.dtype() {
            d.to_dtype(self.divisor.dtype())?
        } else {
            d
        };
        let d = match &self.keep {
            Some(k) => d.mul(k)?,
            None => d,
        };
        let g = if g.dtype() != d.dtype() {
            g.to_dtype(d.dtype())?
        } else {
            g.clone()
        };
        let grad = d.div(&self.divisor)?.mul(&g)?;
        if grad.dtype() == output_dtype {
            Ok(grad)
        } else {
            grad.to_dtype(output_dtype)
        }
    }
}

/// `2·(pred − target)/n · g`, the cotangent of `mse_loss`'s prediction (the
/// target's is its negation). `scale` is `2/n`.
fn mse_grad(diff: &Tensor, scale: f64, g: &Tensor) -> Result<Tensor> {
    let dtype = diff.dtype();
    let accumulation_dtype = dtype.accumulation_dtype();
    if accumulation_dtype != dtype {
        let diff = diff.to_dtype(accumulation_dtype)?;
        let g = g.to_dtype(accumulation_dtype)?;
        diff.mul_scalar(scale)?.mul(&g)?.to_dtype(dtype)
    } else {
        diff.mul_scalar(scale)?.mul(g)
    }
}

/// The shared body of the two `cross_entropy` spellings; `ignore_index` is
/// `None` for the plain one.
fn cross_entropy_impl(
    op: &'static str,
    logits: &Tensor,
    targets: &Tensor,
    ignore_index: Option<i64>,
) -> Result<Tensor> {
    let (rows, classes) = check_logits(op, logits)?;
    check_targets(op, logits, targets, rows)?;
    let device = logits.device();
    let dtype = logits.dtype();

    // The forward runs on a *detached* input so this composition records no
    // graph of its own — the fused node recorded at the end is the only one.
    let logp = logits.detach().log_softmax(1)?;
    let labels = targets.detach().reshape([rows, 1])?;

    // `keep[r]` is false exactly on an ignored row; `safe` replaces its
    // (possibly negative sentinel) label with class 0 so `gather` stays in
    // bounds. Both are built on-device.
    let keep = match ignore_index {
        Some(sentinel) => {
            let s = Tensor::from_vec(vec![sentinel], [1, 1], &device)?;
            Some(labels.ne(&s)?)
        }
        None => None,
    };
    let safe = match &keep {
        Some(k) => k.where_cond(&labels, &Tensor::zeros([1, 1], DType::I64, &device)?)?,
        None => labels,
    };

    // An out-of-range class label surfaces from `gather`; the offending value
    // came from `targets`, not from an index the user handed to `gather`, so
    // the error is re-labelled with the loss method the caller used.
    let nll = logp.gather(1, &safe).map_err(|e| e.with_op(op))?.neg()?;
    let accumulation_dtype = dtype.accumulation_dtype();
    let reduced = accumulation_dtype != dtype;
    let nll = if reduced {
        nll.to_dtype(accumulation_dtype)?
    } else {
        nll
    };
    let (nll, divisor, backward_keep) = match &keep {
        Some(k) => {
            let zero = Tensor::zeros((), accumulation_dtype, &device)?;
            // At least one, so an all-ignored batch is 0/1 = 0, not 0/0.
            let one = Tensor::ones((), accumulation_dtype, &device)?;
            let float_keep = k.to_dtype(accumulation_dtype)?;
            let count = float_keep.sum_all()?.maximum(&one)?;
            (k.where_cond(&nll, &zero)?, count, Some(float_keep))
        }
        None => (
            nll,
            Tensor::full((), rows.max(1) as f64, accumulation_dtype, &device)?,
            None,
        ),
    };
    let out = nll.sum_all()?.div(&divisor)?;
    let out = if reduced { out.to_dtype(dtype)? } else { out };

    let bwd = CrossEntropyBackward {
        logp,
        safe,
        keep: backward_keep,
        divisor,
        classes,
    };
    let backward: BackwardFn = Box::new(move |g| Ok(vec![Some(bwd.grad(g)?)]));
    Ok(autograd::record(op, out, &[logits], backward))
}

impl Tensor {
    /// Mean softmax cross-entropy of `[rows, classes]` logits against one
    /// [`I64`](crate::DType::I64) class label per row.
    ///
    /// `self` is the **logits** — raw scores, never softmax outputs; the
    /// normalization happens inside, in its numerically stable spelling.
    /// `targets` is a 1-D `[rows]` label tensor on the same device. The
    /// result is a rank-0 scalar, the mean over rows of `−log p(target)`.
    ///
    /// Higher-rank inputs are the caller's reshape: a `[batch, time, vocab]`
    /// language-model output is `logits.reshape([batch * time, vocab])` with
    /// its labels reshaped to match. For a batch with padding, see
    /// [`cross_entropy_ignore_index`](Tensor::cross_entropy_ignore_index).
    /// A zero-row batch has loss `0` (there is nothing to average, and
    /// nothing wrong).
    ///
    /// # Errors
    ///
    /// - [`Unsupported`](crate::Error::Unsupported) if the logits are not a
    ///   float dtype; [`DTypeMismatch`](crate::Error::DTypeMismatch) if
    ///   `targets` is not [`I64`](crate::DType::I64).
    /// - [`RankMismatch`](crate::Error::RankMismatch) if the logits are not
    ///   rank 2 or `targets` is not rank 1;
    ///   [`ShapeMismatch`](crate::Error::ShapeMismatch) if there is not
    ///   exactly one label per row.
    /// - [`DeviceMismatch`](crate::Error::DeviceMismatch) if the labels live
    ///   on another device.
    /// - [`InvalidArg`](crate::Error::InvalidArg) if there are no classes,
    ///   and [`IndexOutOfBounds`](crate::Error::IndexOutOfBounds) for a
    ///   negative label or one at/past `classes`.
    ///
    /// ```
    /// # use rstorch::{Device, Tensor};
    /// let dev = Device::Cpu;
    /// // Two rows of three equal logits: the loss is ln(3), whatever the labels.
    /// let logits = Tensor::from_vec(vec![0.0f32; 6], [2, 3], &dev)?;
    /// let labels = Tensor::from_vec(vec![2i64, 0], [2], &dev)?;
    /// let loss = logits.cross_entropy(&labels)?;
    /// assert!(loss.dims().is_empty());
    /// assert!((loss.item()? - 3f64.ln()).abs() < 1e-6);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn cross_entropy(&self, targets: &Tensor) -> Result<Tensor> {
        cross_entropy_impl("cross_entropy", self, targets, None)
    }

    /// [`cross_entropy`](Tensor::cross_entropy) with the rows whose label
    /// equals `ignore_index` dropped — the padded-batch spelling.
    ///
    /// Ignored rows contribute nothing to the sum *and* nothing to the
    /// divisor, so the result is the mean over the kept rows only, and their
    /// logits receive exactly zero gradient. The sentinel is an ordinary
    /// `i64` and need not be a valid class (`PyTorch`'s `-100` is the
    /// conventional choice); labels that are neither the sentinel nor a valid
    /// class are still a loud
    /// [`IndexOutOfBounds`](crate::Error::IndexOutOfBounds). A batch in which
    /// every row is ignored has loss `0` and zero gradient.
    ///
    /// # Errors
    /// As [`cross_entropy`](Tensor::cross_entropy).
    ///
    /// ```
    /// # use rstorch::{Device, Tensor};
    /// let dev = Device::Cpu;
    /// let logits = Tensor::from_vec(vec![0.0f32; 6], [2, 3], &dev)?;
    /// // The second row is padding: the loss is the first row's alone.
    /// let labels = Tensor::from_vec(vec![2i64, -100], [2], &dev)?;
    /// let loss = logits.cross_entropy_ignore_index(&labels, -100)?;
    /// assert!((loss.item()? - 3f64.ln()).abs() < 1e-6);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn cross_entropy_ignore_index(
        &self,
        targets: &Tensor,
        ignore_index: i64,
    ) -> Result<Tensor> {
        cross_entropy_impl(
            "cross_entropy_ignore_index",
            self,
            targets,
            Some(ignore_index),
        )
    }

    /// Mean squared error between `self` (the prediction) and `target`, as a
    /// rank-0 scalar: `mean((self − target)²)`.
    ///
    /// The two shapes must be **identical** — there is no broadcasting here,
    /// because a silently broadcast `[n]` against `[n, 1]` is the classic way
    /// to compute a loss over `n²` phantom pairs. Both operands are
    /// differentiable (the gradient w.r.t. `target` is the negation of the
    /// gradient w.r.t. `self`).
    ///
    /// # Errors
    ///
    /// [`Unsupported`](crate::Error::Unsupported) on a non-float dtype,
    /// [`DTypeMismatch`](crate::Error::DTypeMismatch) /
    /// [`DeviceMismatch`](crate::Error::DeviceMismatch) if the operands
    /// disagree, [`ShapeMismatch`](crate::Error::ShapeMismatch) if their
    /// shapes are not equal, and [`InvalidArg`](crate::Error::InvalidArg) on
    /// empty inputs (a mean of no elements).
    ///
    /// ```
    /// # use rstorch::{Device, Tensor};
    /// let dev = Device::Cpu;
    /// let pred = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3], &dev)?;
    /// let target = Tensor::from_vec(vec![1.0f32, 4.0, 6.0], [3], &dev)?;
    /// // (0² + 2² + 3²) / 3
    /// assert!((pred.mse_loss(&target)?.item()? - 13.0 / 3.0).abs() < 1e-6);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn mse_loss(&self, target: &Tensor) -> Result<Tensor> {
        const OP: &str = "mse_loss";
        if !self.dtype().is_float() {
            return Err(Error::Unsupported {
                op: OP,
                device: self.device(),
                dtype: self.dtype(),
            });
        }
        same_dtype(OP, self, target)?;
        same_device(OP, self, target)?;
        if self.dims() != target.dims() {
            return Err(Error::ShapeMismatch {
                op: OP,
                lhs: self.shape().clone(),
                rhs: target.shape().clone(),
            });
        }
        let n = self.num_elements();
        if n == 0 {
            return Err(Error::InvalidArg {
                op: OP,
                msg: "mse_loss of an empty tensor is undefined (a mean of no elements)".to_string(),
            });
        }

        // Detached forward (see the module docs): one fused node, not a
        // sub/mul/sum graph.
        let diff = self.detach().sub(&target.detach())?;
        let out = diff.mul(&diff)?.mean_all()?;

        let scale = 2.0 / n as f64;
        // `d(pred)` and `d(target)` are exact negatives of one another.
        let backward: BackwardFn = Box::new(move |g| {
            let d = mse_grad(&diff, scale, g)?;
            let neg = d.neg()?;
            Ok(vec![Some(d), Some(neg)])
        });
        Ok(autograd::record(OP, out, &[self, target], backward))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::shape::Shape;
    use crate::testing::check_grad;

    const CPU: Device = Device::Cpu;

    fn t(data: &[f32], shape: impl Into<Shape>) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape, &CPU).unwrap()
    }

    fn labels(data: &[i64]) -> Tensor {
        Tensor::from_vec(data.to_vec(), [data.len()], &CPU).unwrap()
    }

    /// `ln Σ exp(x)`, computed on the host as the reference.
    fn logsumexp(row: &[f32]) -> f64 {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        max + row
            .iter()
            .map(|&x| (x as f64 - max).exp())
            .sum::<f64>()
            .ln()
    }

    // ------------------------------------------------------------------
    // cross_entropy forward
    // ------------------------------------------------------------------

    #[test]
    fn cross_entropy_matches_the_hand_computed_mean_nll() {
        let row = [1.0f32, 2.0, 3.0];
        let logits = t(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], [2, 3]);
        let loss = logits.cross_entropy(&labels(&[2, 0])).unwrap();

        let lse = logsumexp(&row);
        let expected = f64::midpoint(lse - 3.0, lse - 1.0);
        assert!(loss.dims().is_empty(), "the loss is a rank-0 scalar");
        assert!((loss.item().unwrap() - expected).abs() < 1e-6);
    }

    #[test]
    fn cross_entropy_of_uniform_logits_is_ln_classes() {
        let logits = t(&[0.0; 8], [2, 4]);
        let loss = logits.cross_entropy(&labels(&[3, 1])).unwrap();
        assert!((loss.item().unwrap() - 4f64.ln()).abs() < 1e-6);
    }

    #[test]
    fn masked_out_classes_do_not_poison_the_loss() {
        // A `-inf` logit is a forbidden class: it drops out of the
        // normalizer instead of producing a NaN.
        let logits = t(&[0.0, f32::NEG_INFINITY, 1.0], [1, 3]);
        let loss = logits.cross_entropy(&labels(&[2])).unwrap().item().unwrap();
        let expected = logsumexp(&[0.0, 1.0]) - 1.0;
        assert!(loss.is_finite(), "got {loss}");
        assert!((loss - expected).abs() < 1e-6);
    }

    // ------------------------------------------------------------------
    // cross_entropy_ignore_index
    // ------------------------------------------------------------------

    #[test]
    fn ignored_rows_leave_both_the_sum_and_the_divisor() {
        let logits = t(&[1.0, 2.0, 3.0, 9.0, 9.0, 9.0], [2, 3]);
        let kept = logits.narrow(0, 0, 1).unwrap();
        let ignored = logits
            .cross_entropy_ignore_index(&labels(&[2, -100]), -100)
            .unwrap();
        // Exactly the first row's own loss — the divisor is 1, not 2.
        let alone = kept.cross_entropy(&labels(&[2])).unwrap();
        assert!((ignored.item().unwrap() - alone.item().unwrap()).abs() < 1e-6);
    }

    #[test]
    fn an_all_ignored_batch_is_zero_not_nan() {
        let logits = t(&[1.0, 2.0, 3.0, 4.0], [2, 2]);
        let loss = logits
            .cross_entropy_ignore_index(&labels(&[-100, -100]), -100)
            .unwrap();
        assert_eq!(loss.item().unwrap(), 0.0);
    }

    #[test]
    fn the_sentinel_only_excuses_the_rows_that_carry_it() {
        let logits = t(&[1.0, 2.0, 3.0, 4.0], [2, 2]);
        assert!(matches!(
            logits.cross_entropy_ignore_index(&labels(&[5, -100]), -100),
            Err(Error::IndexOutOfBounds {
                op: "cross_entropy_ignore_index",
                index: 5,
                ..
            })
        ));
    }

    // ------------------------------------------------------------------
    // Loud arguments
    // ------------------------------------------------------------------

    #[test]
    fn labels_must_be_i64_one_per_row_and_in_range() {
        let logits = t(&[1.0, 2.0, 3.0, 4.0], [2, 2]);
        assert!(matches!(
            logits.cross_entropy(&t(&[0.0, 1.0], [2])),
            Err(Error::DTypeMismatch {
                op: "cross_entropy",
                expected: DType::I64,
                got: DType::F32,
            })
        ));
        assert!(matches!(
            logits.cross_entropy(&labels(&[0, 1, 0])),
            Err(Error::ShapeMismatch {
                op: "cross_entropy",
                ..
            })
        ));
        assert!(matches!(
            logits.cross_entropy(&labels(&[0, 2])),
            Err(Error::IndexOutOfBounds {
                op: "cross_entropy",
                index: 2,
                ..
            })
        ));
        assert!(matches!(
            logits.cross_entropy(&labels(&[0, -1])),
            Err(Error::IndexOutOfBounds {
                op: "cross_entropy",
                index: -1,
                ..
            })
        ));
    }

    #[test]
    fn an_empty_batch_is_zero_not_nan() {
        // A zero-row batch (the tail of a drop-nothing loader) has no rows to
        // average, and no error to report: the divisor is held at 1.
        let logits = t(&[], [0, 3]);
        assert_eq!(
            logits.cross_entropy(&labels(&[])).unwrap().item().unwrap(),
            0.0
        );
    }

    #[test]
    fn logits_must_be_a_float_matrix() {
        let flat = t(&[1.0, 2.0], [2]);
        assert!(matches!(
            flat.cross_entropy(&labels(&[0, 1])),
            Err(Error::RankMismatch {
                op: "cross_entropy",
                expected: 2,
                got: 1,
            })
        ));
        let ints = Tensor::from_vec(vec![1i64, 2, 3, 4], [2, 2], &CPU).unwrap();
        assert!(matches!(
            ints.cross_entropy(&labels(&[0, 1])),
            Err(Error::Unsupported {
                op: "cross_entropy",
                ..
            })
        ));
    }

    // ------------------------------------------------------------------
    // mse_loss
    // ------------------------------------------------------------------

    #[test]
    fn mse_loss_matches_the_hand_computed_mean_square() {
        let pred = t(&[1.0, 2.0, 3.0, 4.0], [2, 2]);
        let target = t(&[1.0, 4.0, 6.0, 0.0], [2, 2]);
        // (0 + 4 + 9 + 16) / 4
        let loss = pred.mse_loss(&target).unwrap();
        assert!(loss.dims().is_empty());
        assert!((loss.item().unwrap() - 29.0 / 4.0).abs() < 1e-6);
        // A perfect prediction costs nothing.
        assert_eq!(pred.mse_loss(&pred).unwrap().item().unwrap(), 0.0);
    }

    #[test]
    fn mse_loss_refuses_to_broadcast() {
        let pred = t(&[1.0, 2.0], [2]);
        let target = t(&[1.0, 2.0], [2, 1]);
        assert!(matches!(
            pred.mse_loss(&target),
            Err(Error::ShapeMismatch { op: "mse_loss", .. })
        ));
        let empty = t(&[], [0]);
        assert!(matches!(
            empty.mse_loss(&empty),
            Err(Error::InvalidArg { op: "mse_loss", .. })
        ));
    }

    // ------------------------------------------------------------------
    // Backward helpers: these tests exercise the value-level functions the
    // backward closures are made of. The finite-difference cases that check
    // the closures end to end are further down.
    // ------------------------------------------------------------------

    /// Rebuild what `cross_entropy_impl` captures, for a batch with no
    /// ignored rows.
    fn ce_backward(logits: &Tensor, target: &[i64], classes: usize) -> CrossEntropyBackward {
        CrossEntropyBackward {
            logp: logits.log_softmax(1).unwrap(),
            safe: Tensor::from_vec(target.to_vec(), [target.len(), 1], &CPU).unwrap(),
            keep: None,
            divisor: Tensor::full((), target.len() as f64, DType::F32, &CPU).unwrap(),
            classes,
        }
    }

    #[test]
    fn cross_entropy_backward_is_softmax_minus_onehot_over_rows() {
        let logits = t(&[1.0, 2.0, 3.0, 0.0, 0.0, 0.0], [2, 3]);
        let g = t(&[2.0], ()); // a rank-0 cotangent, so the scale is visible
        let grad = ce_backward(&logits, &[2, 0], 3).grad(&g).unwrap();

        let mut expected = logits.softmax(1).unwrap().to_vec::<f32>().unwrap();
        expected[2] -= 1.0; // row 0's target
        expected[3] -= 1.0; // row 1's target
        let got = grad.to_vec::<f32>().unwrap();
        assert_eq!(grad.dims(), &[2, 3]);
        for (a, b) in got.iter().zip(&expected) {
            assert!((a - b * 2.0 / 2.0).abs() < 1e-6, "{got:?} vs {expected:?}");
        }
        // A cross-entropy gradient sums to zero along each row.
        assert!(
            got.chunks(3)
                .all(|row| row.iter().sum::<f32>().abs() < 1e-6)
        );
    }

    #[test]
    fn cross_entropy_backward_zeroes_the_ignored_rows() {
        let logits = t(&[1.0, 2.0, 3.0, 0.0, 1.0, 0.0], [2, 3]);
        let bwd = CrossEntropyBackward {
            logp: logits.log_softmax(1).unwrap(),
            // The ignored row's label was clamped to class 0 by the forward.
            safe: Tensor::from_vec(vec![2i64, 0], [2, 1], &CPU).unwrap(),
            keep: Some(t(&[1.0, 0.0], [2, 1])),
            divisor: Tensor::full((), 1.0, DType::F32, &CPU).unwrap(),
            classes: 3,
        };
        let grad = bwd.grad(&t(&[1.0], ())).unwrap().to_vec::<f32>().unwrap();
        assert_eq!(&grad[3..], &[0.0, 0.0, 0.0]);
        // The kept row is undivided: one row, divisor 1.
        let alone = ce_backward(&logits.narrow(0, 0, 1).unwrap(), &[2], 3)
            .grad(&t(&[1.0], ()))
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        for (a, b) in grad[..3].iter().zip(&alone) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn mse_backward_is_twice_the_difference_over_n() {
        let diff = t(&[1.0, -2.0, 0.5, 0.0], [2, 2]);
        let grad = mse_grad(&diff, 2.0 / 4.0, &t(&[3.0], ()))
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(grad, vec![1.5, -3.0, 0.75, 0.0]);
    }

    // ------------------------------------------------------------------
    // Backward: finite differences against the single `check_grad` harness.
    // ------------------------------------------------------------------

    const EPS: f64 = 1e-3;
    const TOL: f64 = 1e-4;

    #[test]
    fn grad_cross_entropy() {
        let logits = t(&[0.5, -1.0, 2.0, 0.25, 1.5, -0.75], [2, 3]);
        let y = labels(&[2, 0]);
        check_grad(
            |xs| xs[0].cross_entropy(&y),
            std::slice::from_ref(&logits),
            EPS,
            TOL,
        )
        .unwrap();
    }

    #[test]
    fn grad_cross_entropy_ignore_index_is_zero_on_ignored_rows() {
        let logits = t(&[0.5, -1.0, 2.0, 0.25, 1.5, -0.75], [2, 3]);
        let y = labels(&[1, -100]);
        check_grad(
            |xs| xs[0].cross_entropy_ignore_index(&y, -100),
            std::slice::from_ref(&logits),
            EPS,
            TOL,
        )
        .unwrap();
    }

    #[test]
    fn grad_mse_loss_flows_to_both_operands() {
        let pred = t(&[1.0, -2.0, 0.5, 3.0], [2, 2]);
        let target = t(&[0.25, 1.0, -1.5, 2.0], [2, 2]);
        check_grad(|xs| xs[0].mse_loss(&xs[1]), &[pred, target], EPS, TOL).unwrap();
    }
}
