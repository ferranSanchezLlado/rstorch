//! Matrix multiplication (T24): the 2-D case and the batched case, with
//! NumPy/PyTorch broadcasting over the leading batch axes.
//!
//! One spelling, one entry point: [`Tensor::matmul`]. The **trailing two
//! axes** are the matrix axes (`[…, m, k] × […, k, n] → […, m, n]`);
//! everything in front of them is a batch axis, and the two operands' batch
//! prefixes broadcast right-aligned under the usual rules (a size-1 or
//! missing batch axis repeats). Both operands must be rank ≥ 2: there is no
//! implicit vector promotion (PyTorch's rank-1 special cases), because "the
//! last two axes are the matrix" is the rule that composes with batching
//! without surprises.
//!
//! The op layer validates (device, dtype, rank, inner dimension, batch
//! broadcast), computes the output shape, and hands the two **strided views**
//! straight to `BackendOps::matmul`. A transposed, narrowed or broadcast
//! operand is never materialized first — the kernel walks strides — which is
//! what makes `x.matmul(&w.get(mode).transpose(-2, -1)?)` (the
//! `Linear`/weight-tying spelling of exploration §4.4) allocation-free on the
//! operand side. Inner products accumulate in the wide
//! [`Acc`](crate::dtype::Element::Acc) type per the backend contract, so an
//! `f16` matmul sums in `f32` and narrows exactly once.
//!
//! # Backward
//!
//! With `c = a @ b` and `g` the cotangent of `c`:
//!
//! | input | gradient |
//! |---|---|
//! | `a` | `g @ bᵀ`, then summed back to `a`'s shape |
//! | `b` | `aᵀ @ g`, then summed back to `b`'s shape |
//!
//! Both transposes are over the **last two axes only** (`transpose(-2, -1)`),
//! leaving the batch axes alone. Each product is computed at the *output's*
//! broadcast batch shape and reduced to the operand's own shape with the
//! shared `Tensor::sum_to` helper — exactly the transpose of the forward
//! batch broadcast, and the same reduction the element-wise binaries use.
//!
//! The closure captures the two **inputs** in detached form (they are what
//! the formulas need); it never captures the output, so the detached-output
//! capture rule of exploration §4.3 is satisfied trivially and no `Arc` cycle
//! through the output node can form.

use crate::autograd::record;
use crate::backend::dispatch;
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::shape::Shape;
use crate::tensor::Tensor;

/// The "these two shapes cannot be multiplied" error, naming the public op
/// and carrying both **full** operand shapes (not just the offending batch
/// prefixes, which on their own would not tell the caller much).
fn mismatch(lhs: &Tensor, rhs: &Tensor) -> Error {
    Error::ShapeMismatch {
        op: "matmul",
        lhs: lhs.shape().clone(),
        rhs: rhs.shape().clone(),
    }
}

/// Validate the whole shape contract of `lhs @ rhs` and return the output
/// shape: the broadcast batch dims followed by `[m, n]`.
///
/// This mirrors the kernel's own geometry planning; the kernel re-checks and
/// reports loudly, but the op layer is where the user-facing error is named
/// and carries the full shapes.
fn output_shape(lhs: &Tensor, rhs: &Tensor) -> Result<Shape> {
    let (lr, rr) = (lhs.rank(), rhs.rank());
    if lr < 2 || rr < 2 {
        return Err(Error::InvalidArg {
            op: "matmul",
            msg: format!("operands must be rank >= 2, got ranks {lr} and {rr}"),
        });
    }
    let (ld, rd) = (lhs.dims(), rhs.dims());
    if ld[lr - 1] != rd[rr - 2] {
        return Err(mismatch(lhs, rhs));
    }
    let batch = Shape::from(ld[..lr - 2].to_vec())
        .broadcast_with(&Shape::from(rd[..rr - 2].to_vec()), "matmul")
        .map_err(|_| mismatch(lhs, rhs))?;

    let mut dims = batch.dims().to_vec();
    dims.push(ld[lr - 2]);
    dims.push(rd[rr - 1]);
    Ok(Shape::from(dims))
}

impl Tensor {
    /// Matrix product, batched and broadcasting over the leading axes.
    ///
    /// The last two axes of each operand are the matrix
    /// (`[…, m, k] @ […, k, n] → […, m, n]`); the leading axes are batch dims
    /// that broadcast right-aligned against each other. Both operands must be
    /// rank ≥ 2. The result is freshly allocated and contiguous, and keeps the
    /// operands' dtype and device.
    ///
    /// Operands are consumed as strided views: transposing one costs nothing
    /// beyond the layout (`a.matmul(&b.transpose(-2, -1)?)` does not copy `b`).
    ///
    /// # Errors
    ///
    /// - [`Error::DeviceMismatch`] / [`Error::DTypeMismatch`] when the
    ///   operands disagree — no implicit transfer, no implicit promotion.
    /// - [`Error::InvalidArg`] when either operand has rank < 2.
    /// - [`Error::ShapeMismatch`] when the inner dimensions differ or the
    ///   batch prefixes do not broadcast.
    /// - [`Error::Unsupported`] for a dtype the backend cannot multiply
    ///   (notably [`Bool`](crate::DType::Bool)).
    ///
    /// An output with **no elements** (a zero-sized batch, `m` or `n`) is
    /// short-circuited to an empty allocation before dispatch: there is
    /// nothing to multiply, so no kernel runs and no dtype-support error is
    /// raised for that degenerate case.
    ///
    /// ```
    /// use rstorch::{Device, Tensor};
    /// let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], &Device::Cpu)?;
    /// let b = Tensor::from_vec(
    ///     vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0],
    ///     [3, 2],
    ///     &Device::Cpu,
    /// )?;
    /// let c = a.matmul(&b)?;
    /// assert_eq!(c.dims(), &[2, 2]);
    /// assert_eq!(c.to_vec::<f32>()?, vec![58.0, 64.0, 139.0, 154.0]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        if self.device() != rhs.device() {
            return Err(Error::DeviceMismatch {
                op: "matmul",
                expected: self.device(),
                got: rhs.device(),
            });
        }
        if self.dtype() != rhs.dtype() {
            return Err(Error::DTypeMismatch {
                op: "matmul",
                expected: self.dtype(),
                got: rhs.dtype(),
            });
        }
        let layout = Layout::contiguous(output_shape(self, rhs)?)?;
        let backend = dispatch::backend(self.device());
        let storage = if layout.num_elements() == 0 {
            backend.full(0, self.dtype(), 0.0)?
        } else {
            backend.matmul(self.view(), rhs.view())?
        };
        let out = Tensor::from_parts(storage, layout);

        let (a_dims, b_dims) = (self.dims().to_vec(), rhs.dims().to_vec());
        let (a, b) = (self.detach(), rhs.detach());
        Ok(record(
            "matmul",
            out,
            &[self, rhs],
            Box::new(move |g| {
                // Each product lands at the output's broadcast batch shape;
                // `sum_to` folds the batch axes this operand did not have (or
                // held at 1) back down, which is the transpose of the forward
                // broadcast.
                let da = || -> Result<Tensor> { g.matmul(&b.transpose(-2, -1)?)?.sum_to(&a_dims) };
                let db = || -> Result<Tensor> { a.transpose(-2, -1)?.matmul(g)?.sum_to(&b_dims) };
                vec![da().ok(), db().ok()]
            }),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::dtype::DType;
    use crate::testing::check_grad;

    const CPU: Device = Device::Cpu;

    /// A contiguous f32 tensor on CPU.
    fn t(data: &[f32], shape: impl Into<Shape>) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape, &CPU).unwrap()
    }

    /// `1.0, 2.0, …` of the right length, viewed as `shape`.
    fn iota(shape: impl Into<Shape>) -> Tensor {
        let shape = shape.into();
        let data: Vec<f32> = (0..shape.num_elements()).map(|v| v as f32 + 1.0).collect();
        Tensor::from_vec(data, shape, &CPU).unwrap()
    }

    fn v(t: &Tensor) -> Vec<f32> {
        t.to_vec::<f32>().unwrap()
    }

    // ------------------------------------------------------------------
    // Forward: golden values
    // ------------------------------------------------------------------

    #[test]
    fn matmul_2d_golden_values() {
        let a = iota([2, 3]); // [[1, 2, 3], [4, 5, 6]]
        let b = t(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2]);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.dims(), &[2, 2]);
        assert_eq!(v(&c), vec![58.0, 64.0, 139.0, 154.0]);
        // Shape, dtype, device and contiguity of the result.
        assert_eq!(c.dtype(), DType::F32);
        assert_eq!(c.device(), CPU);
        assert!(c.is_contiguous());
    }

    #[test]
    fn matmul_2d_non_square_and_rank_one_inner() {
        // [1, 3] @ [3, 1] -> [1, 1] (a dot product spelled with real matrices).
        let a = iota([1, 3]);
        let b = iota([3, 1]);
        assert_eq!(v(&a.matmul(&b).unwrap()), vec![1.0 + 4.0 + 9.0]);
        // The outer product: [3, 1] @ [1, 3] -> [3, 3].
        let outer = b.matmul(&a).unwrap();
        assert_eq!(outer.dims(), &[3, 3]);
        assert_eq!(v(&outer), vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0]);
    }

    #[test]
    fn matmul_batched() {
        // Two independent [2, 3] @ [3, 2] products.
        let a = iota([2, 2, 3]);
        let b = t(
            &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [2, 3, 2],
        );
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.dims(), &[2, 2, 2]);
        assert_eq!(v(&c), vec![4.0, 5.0, 10.0, 11.0, 24.0, 24.0, 33.0, 33.0]);
    }

    #[test]
    fn matmul_broadcasts_the_batch_prefix() {
        let mat = iota([2, 3]); // [[1, 2, 3], [4, 5, 6]]
        let batched = t(
            &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [2, 3, 2],
        );

        // A bare 2-D lhs against a batched rhs: the matrix repeats.
        let c = mat.matmul(&batched).unwrap();
        assert_eq!(c.dims(), &[2, 2, 2]);
        assert_eq!(v(&c), vec![4.0, 5.0, 10.0, 11.0, 6.0, 6.0, 15.0, 15.0]);

        // A leading size-1 axis broadcasts the same way.
        let c1 = mat.reshape([1, 2, 3]).unwrap().matmul(&batched).unwrap();
        assert_eq!(c1.dims(), &[2, 2, 2]);
        assert_eq!(v(&c1), v(&c));

        // …and symmetrically on the right-hand side.
        let d = iota([2, 2, 3]).matmul(&iota([3, 2])).unwrap();
        assert_eq!(d.dims(), &[2, 2, 2]);
        let per_batch = iota([2, 3]).matmul(&iota([3, 2])).unwrap();
        assert_eq!(v(&d)[..4], v(&per_batch)[..]);
    }

    #[test]
    fn implicit_broadcast_matches_an_explicit_one() {
        // Two batch axes, each operand broadcasting a different one:
        // [2, 1, 2, 3] @ [1, 3, 3, 2] -> [2, 3, 2, 2].
        let a = iota([2, 1, 2, 3]);
        let b = iota([1, 3, 3, 2]);
        let implicit = a.matmul(&b).unwrap();
        assert_eq!(implicit.dims(), &[2, 3, 2, 2]);

        let explicit = a
            .broadcast_to([2, 3, 2, 3])
            .unwrap()
            .matmul(&b.broadcast_to([2, 3, 3, 2]).unwrap())
            .unwrap();
        assert_eq!(v(&implicit), v(&explicit));

        // Cross-check one batch cell against the plain 2-D product.
        let cell = a
            .narrow(0, 1, 1)
            .unwrap()
            .reshape([2, 3])
            .unwrap()
            .matmul(&b.narrow(1, 2, 1).unwrap().reshape([3, 2]).unwrap())
            .unwrap();
        // Batch cell (1, 2) is the last of the six [2, 2] blocks.
        assert_eq!(v(&implicit)[20..], v(&cell)[..]);
    }

    #[test]
    fn strided_operands_are_multiplied_without_materializing() {
        let a = iota([2, 3]); // [[1, 2, 3], [4, 5, 6]]
        let b = iota([2, 3]);
        // a @ bᵀ : [2, 3] @ [3, 2] -> [2, 2], the `Linear` spelling.
        let c = a.matmul(&b.transpose(-2, -1).unwrap()).unwrap();
        assert_eq!(c.dims(), &[2, 2]);
        // rows of a · rows of b
        assert_eq!(v(&c), vec![14.0, 32.0, 32.0, 77.0]);

        // aᵀ @ b : [3, 2] @ [2, 3] -> [3, 3].
        let d = a.transpose(0, 1).unwrap().matmul(&b).unwrap();
        assert_eq!(d.dims(), &[3, 3]);
        assert_eq!(
            v(&d),
            vec![17.0, 22.0, 27.0, 22.0, 29.0, 36.0, 27.0, 36.0, 45.0]
        );

        // A narrowed (offset, non-zero-based) operand.
        let wide = iota([2, 4]);
        let narrowed = wide.narrow(1, 1, 3).unwrap(); // [[2, 3, 4], [6, 7, 8]]
        let e = narrowed.matmul(&iota([3, 1])).unwrap();
        assert_eq!(v(&e), vec![2.0 + 6.0 + 12.0, 6.0 + 14.0 + 24.0]);
    }

    #[test]
    fn matmul_preserves_integer_dtype() {
        let a = Tensor::from_vec(vec![1i64, 2, 3, 4], [2, 2], &CPU).unwrap();
        let b = Tensor::from_vec(vec![5i64, 6, 7, 8], [2, 2], &CPU).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.dtype(), DType::I64);
        assert_eq!(c.to_vec::<i64>().unwrap(), vec![19, 22, 43, 50]);
    }

    // ------------------------------------------------------------------
    // Forward: degenerate shapes
    // ------------------------------------------------------------------

    #[test]
    fn zero_sized_dimensions_produce_empty_or_zero_results() {
        // A zero-length inner dimension sums nothing: an all-zeros result.
        let a = Tensor::zeros([2, 0], DType::F32, &CPU).unwrap();
        let b = Tensor::zeros([0, 3], DType::F32, &CPU).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.dims(), &[2, 3]);
        assert_eq!(v(&c), vec![0.0; 6]);

        // A zero-sized batch axis: no products at all.
        let a = Tensor::zeros([0, 2, 3], DType::F32, &CPU).unwrap();
        let b = Tensor::zeros([0, 3, 4], DType::F32, &CPU).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.dims(), &[0, 2, 4]);
        assert!(v(&c).is_empty());

        // A zero-sized matrix axis.
        let a = Tensor::zeros([0, 3], DType::F32, &CPU).unwrap();
        let b = Tensor::zeros([3, 4], DType::F32, &CPU).unwrap();
        assert_eq!(a.matmul(&b).unwrap().dims(), &[0, 4]);
    }

    // ------------------------------------------------------------------
    // Forward: errors
    // ------------------------------------------------------------------

    #[test]
    fn inner_dimension_mismatch_is_a_shape_error() {
        let err = iota([2, 3]).matmul(&iota([4, 2])).unwrap_err();
        match err {
            Error::ShapeMismatch { op, lhs, rhs } => {
                assert_eq!(op, "matmul");
                assert_eq!(lhs.dims(), &[2, 3]);
                assert_eq!(rhs.dims(), &[4, 2]);
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn non_broadcastable_batch_prefix_reports_the_full_shapes() {
        let err = iota([2, 2, 3]).matmul(&iota([3, 3, 2])).unwrap_err();
        match err {
            Error::ShapeMismatch { op, lhs, rhs } => {
                assert_eq!(op, "matmul");
                // The *full* operand shapes, not the batch prefixes.
                assert_eq!(lhs.dims(), &[2, 2, 3]);
                assert_eq!(rhs.dims(), &[3, 3, 2]);
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn rank_below_two_is_rejected_on_either_side() {
        for err in [
            iota([3]).matmul(&iota([3, 2])).unwrap_err(),
            iota([2, 3]).matmul(&iota([3])).unwrap_err(),
            iota(()).matmul(&iota([1, 1])).unwrap_err(),
        ] {
            match err {
                Error::InvalidArg { op, msg } => {
                    assert_eq!(op, "matmul");
                    assert!(msg.contains("rank >= 2"), "{msg}");
                }
                other => panic!("expected InvalidArg, got {other:?}"),
            }
        }
    }

    #[test]
    fn mixed_dtypes_do_not_promote() {
        let a = iota([2, 2]);
        let b = Tensor::from_vec(vec![1i64, 2, 3, 4], [2, 2], &CPU).unwrap();
        match a.matmul(&b).unwrap_err() {
            Error::DTypeMismatch { op, expected, got } => {
                assert_eq!(op, "matmul");
                assert_eq!(expected, DType::F32);
                assert_eq!(got, DType::I64);
            }
            other => panic!("expected DTypeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn bool_has_no_matmul() {
        let a = Tensor::from_vec(vec![true, false, true, true], [2, 2], &CPU).unwrap();
        match a.matmul(&a).unwrap_err() {
            Error::Unsupported { op, dtype, .. } => {
                assert_eq!(op, "matmul");
                assert_eq!(dtype, DType::Bool);
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    // ------------------------------------------------------------------
    // Backward: finite-difference cases against the single `check_grad`
    // harness. `record()` is a no-op and `check_grad` is a stub until T30,
    // so these are `#[ignore]`d; **T31** removes the attribute.
    //
    // `check_grad` wants a scalar-valued `f`, and the reductions that would
    // supply one live in T23 (a sibling task, not this layer). Every case
    // therefore scalarizes with `pick`, selecting one output element with
    // T21's view ops; since `check_grad` perturbs *every* input element the
    // full gradient tensor is still checked, against a one-hot cotangent.
    // ------------------------------------------------------------------

    /// The element of `t` at row-major position `flat`, as a rank-0 tensor.
    fn pick(t: &Tensor, flat: usize) -> Result<Tensor> {
        let mut coords = vec![0usize; t.rank()];
        let mut rest = flat;
        for axis in (0..t.rank()).rev() {
            coords[axis] = rest % t.dims()[axis];
            rest /= t.dims()[axis];
        }
        let mut cur = t.clone();
        for (axis, &c) in coords.iter().enumerate() {
            cur = cur.narrow(axis as isize, c, 1)?;
        }
        cur.reshape(())
    }

    const EPS: f64 = 1e-3;
    const TOL: f64 = 1e-4;

    #[test]
    #[ignore = "T31 activates the FD suite once T30 lands the engine"]
    fn grad_matmul_2d() {
        let a = iota([2, 3]);
        let b = iota([3, 2]);
        for flat in 0..4 {
            check_grad(
                move |xs| pick(&xs[0].matmul(&xs[1])?, flat),
                &[a.clone(), b.clone()],
                EPS,
                TOL,
            )
            .unwrap();
        }
    }

    #[test]
    #[ignore = "T31 activates the FD suite once T30 lands the engine"]
    fn grad_matmul_batched() {
        let a = iota([2, 2, 3]);
        let b = iota([2, 3, 2]);
        check_grad(|xs| pick(&xs[0].matmul(&xs[1])?, 5), &[a, b], EPS, TOL).unwrap();
    }

    #[test]
    #[ignore = "T31 activates the FD suite once T30 lands the engine"]
    fn grad_matmul_broadcast_batch_is_summed_back() {
        // The lhs is a bare matrix reused across the batch: its gradient is
        // the sum over the batch axis (the `sum_to` path).
        let a = iota([2, 3]);
        let b = iota([2, 3, 2]);
        check_grad(
            |xs| pick(&xs[0].matmul(&xs[1])?, 3),
            &[a.clone(), b.clone()],
            EPS,
            TOL,
        )
        .unwrap();

        // …and symmetrically, with a size-1 batch axis on the right.
        let a = iota([2, 2, 3]);
        let b = iota([1, 3, 2]);
        check_grad(|xs| pick(&xs[0].matmul(&xs[1])?, 6), &[a, b], EPS, TOL).unwrap();
    }

    #[test]
    #[ignore = "T31 activates the FD suite once T30 lands the engine"]
    fn grad_matmul_through_a_transposed_operand() {
        // The `Linear` spelling `x @ wᵀ`: the transpose's backward and the
        // matmul's must compose.
        let x = iota([2, 3]);
        let w = iota([4, 3]);
        check_grad(
            |xs| pick(&xs[0].matmul(&xs[1].transpose(-2, -1)?)?, 5),
            &[x, w],
            EPS,
            TOL,
        )
        .unwrap();
    }

    #[test]
    #[ignore = "T31 activates the FD suite once T30 lands the engine"]
    fn grad_matmul_with_a_repeated_operand() {
        // `a @ a` accumulates two contributions into the same leaf.
        let a = iota([2, 2]);
        check_grad(
            |xs| pick(&xs[0].matmul(&xs[0])?, 2),
            std::slice::from_ref(&a),
            EPS,
            TOL,
        )
        .unwrap();
    }
}
