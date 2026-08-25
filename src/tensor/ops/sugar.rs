//! Operator sugar: the panicking tier of the two-tier fallibility policy.
//!
//! Every named method returns [`Result`](crate::Result). The `std::ops`
//! spellings — `a + b`, `a - b`, `a * b`, `a / b`, `-a`, plus the `tensor
//! <op> f64` and `f64 <op> tensor` forms — call exactly those methods and
//! **panic with the identical structured message** when they fail. The impls
//! are `#[track_caller]`, so the panic is reported at the user's expression,
//! not inside this file:
//!
//! ```text
//! thread 'main' panicked at src/main.rs:12:17:
//! add: shape mismatch: lhs [2, 3] vs rhs [4, 5]
//! ```
//!
//! That location is exact for the errors the sugar can actually raise:
//! shape, rank, dtype, device and rejected-argument failures are decided from
//! metadata before any kernel runs, so the reported line is the
//! offending expression. Backend failures are the other tier — execution is
//! batched, so an `Unsupported` or `Backend` error can be reported by a later
//! named call that forces completion rather than by the operator that queued
//! the work. See [`crate::Error`] for the full timing split.
//!
//! Nothing else lives here: there is no separate operator semantics, no
//! implicit promotion, and no in-place variant. `&a + &b` and
//! `a.add(&b)?` compute the same thing through the same code path, so the
//! sugar can never drift from the named surface.

use crate::error::Result;
use crate::tensor::Tensor;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Unwrap an op result, panicking with the error's `Display` message — the
/// same text the `Result` tier carries. `#[track_caller]` here and on the
/// operator impls chains the reported location all the way out to the user's
/// expression.
#[track_caller]
fn unwrap_op(result: Result<Tensor>) -> Tensor {
    match result {
        Ok(t) => t,
        Err(e) => panic!("{e}"),
    }
}

/// Generate the six ownership combinations of one binary operator: the four
/// `Tensor`/`&Tensor` pairings and the two `f64` right-hand-side forms.
macro_rules! binary_sugar {
    ($trait:ident, $method:ident, $named:ident, $scalar:ident, $sym:literal) => {
        #[doc = concat!("`", $sym, "` — the panicking spelling of [`Tensor::", stringify!($named), "`].")]
        impl $trait<&Tensor> for &Tensor {
            type Output = Tensor;
            #[track_caller]
            fn $method(self, rhs: &Tensor) -> Tensor {
                unwrap_op(Tensor::$named(self, rhs))
            }
        }

        #[doc = concat!("`", $sym, "` — the panicking spelling of [`Tensor::", stringify!($named), "`].")]
        impl $trait<Tensor> for &Tensor {
            type Output = Tensor;
            #[track_caller]
            fn $method(self, rhs: Tensor) -> Tensor {
                unwrap_op(Tensor::$named(self, &rhs))
            }
        }

        #[doc = concat!("`", $sym, "` — the panicking spelling of [`Tensor::", stringify!($named), "`].")]
        impl $trait<&Tensor> for Tensor {
            type Output = Tensor;
            #[track_caller]
            fn $method(self, rhs: &Tensor) -> Tensor {
                unwrap_op(Tensor::$named(&self, rhs))
            }
        }

        #[doc = concat!("`", $sym, "` — the panicking spelling of [`Tensor::", stringify!($named), "`].")]
        impl $trait<Tensor> for Tensor {
            type Output = Tensor;
            #[track_caller]
            fn $method(self, rhs: Tensor) -> Tensor {
                unwrap_op(Tensor::$named(&self, &rhs))
            }
        }

        #[doc = concat!("`tensor ", $sym, " f64` — the panicking spelling of [`Tensor::", stringify!($scalar), "`].")]
        impl $trait<f64> for &Tensor {
            type Output = Tensor;
            #[track_caller]
            fn $method(self, rhs: f64) -> Tensor {
                unwrap_op(Tensor::$scalar(self, rhs))
            }
        }

        #[doc = concat!("`tensor ", $sym, " f64` — the panicking spelling of [`Tensor::", stringify!($scalar), "`].")]
        impl $trait<f64> for Tensor {
            type Output = Tensor;
            #[track_caller]
            fn $method(self, rhs: f64) -> Tensor {
                unwrap_op(Tensor::$scalar(&self, rhs))
            }
        }
    };
}

binary_sugar!(Add, add, add, add_scalar, "+");
binary_sugar!(Sub, sub, sub, sub_scalar, "-");
binary_sugar!(Mul, mul, mul, mul_scalar, "*");
binary_sugar!(Div, div, div, div_scalar, "/");

/// Generate the two scalar-left-hand-side forms of one binary operator:
/// `f64 op Tensor` and `f64 op &Tensor`. `$reversed` computes `scalar op
/// tensor` — the operand order matters for `-` and `/` — and every error it
/// produces is relabelled to `$scalar`, so `2.0 - t` and `t - 2.0` fail with
/// the same op name even though the reversed form is a composition.
macro_rules! scalar_lhs_sugar {
    ($trait:ident, $method:ident, $reversed:ident, $scalar:ident, $sym:literal) => {
        #[doc = concat!("`f64 ", $sym, " tensor` — the panicking spelling of the reversed [`Tensor::", stringify!($scalar), "`].")]
        impl $trait<&Tensor> for f64 {
            type Output = Tensor;
            #[track_caller]
            fn $method(self, rhs: &Tensor) -> Tensor {
                unwrap_op($reversed(self, rhs).map_err(|e| e.with_op(stringify!($scalar))))
            }
        }

        #[doc = concat!("`f64 ", $sym, " tensor` — the panicking spelling of the reversed [`Tensor::", stringify!($scalar), "`].")]
        impl $trait<Tensor> for f64 {
            type Output = Tensor;
            #[track_caller]
            fn $method(self, rhs: Tensor) -> Tensor {
                unwrap_op($reversed(self, &rhs).map_err(|e| e.with_op(stringify!($scalar))))
            }
        }
    };
}

/// `scalar + tensor`. Addition commutes, so this *is* `add_scalar`.
fn scalar_add(scalar: f64, tensor: &Tensor) -> Result<Tensor> {
    tensor.add_scalar(scalar)
}

/// `scalar - tensor`. There is no reversed primitive and adding one would mean
/// a new kernel on every backend for no new semantics, so this is the
/// composition `(-tensor) + scalar`: two passes over the elements, both
/// through the cheap unary/scalar path, and two temporaries — the same count
/// as `full_like(scalar).sub(tensor)` but without materialising a broadcast
/// operand. The existing `neg` and `add_scalar` backwards already compose to
/// the correct `-1` derivative, so no gradient rule is added either.
///
/// The named methods are spelled as paths because the operator traits are in
/// scope in this module and would otherwise win method resolution.
/// Relabel a composed scalar-left result when realization is deferred.
fn relabel_scalar_result(op: &'static str, value: Tensor) -> Result<Tensor> {
    if !crate::lazy::enabled() {
        return Ok(value);
    }
    let storage = crate::lazy::relabel(op, value.storage(), value.layout())?;
    Ok(match value.node() {
        Some(node) => Tensor::from_parts_traced(storage, value.layout().clone(), node.clone()),
        None => Tensor::from_parts(storage, value.layout().clone()),
    })
}

fn scalar_sub(scalar: f64, tensor: &Tensor) -> Result<Tensor> {
    relabel_scalar_result("sub_scalar", Tensor::neg(tensor)?.add_scalar(scalar)?)
}

/// `scalar * tensor`. Multiplication commutes, so this *is* `mul_scalar`.
fn scalar_mul(scalar: f64, tensor: &Tensor) -> Result<Tensor> {
    tensor.mul_scalar(scalar)
}

/// `scalar / tensor`. There is no reciprocal primitive to scale, so this is
/// `full_like(scalar) / tensor`: one fill plus one divide, both already on
/// every backend. The constant left operand is untraced and already the
/// output shape, so `div` broadcasts nothing and its existing backward yields
/// `-scalar / tensor²` with no new rule.
fn scalar_div(scalar: f64, tensor: &Tensor) -> Result<Tensor> {
    let numerator = tensor.full_like(scalar)?;
    relabel_scalar_result("div_scalar", Tensor::div(&numerator, tensor)?)
}

scalar_lhs_sugar!(Add, add, scalar_add, add_scalar, "+");
scalar_lhs_sugar!(Sub, sub, scalar_sub, sub_scalar, "-");
scalar_lhs_sugar!(Mul, mul, scalar_mul, mul_scalar, "*");
scalar_lhs_sugar!(Div, div, scalar_div, div_scalar, "/");

/// `-tensor` — the panicking spelling of [`Tensor::neg`].
impl Neg for &Tensor {
    type Output = Tensor;
    #[track_caller]
    fn neg(self) -> Tensor {
        unwrap_op(Tensor::neg(self))
    }
}

/// `-tensor` — the panicking spelling of [`Tensor::neg`].
impl Neg for Tensor {
    type Output = Tensor;
    #[track_caller]
    fn neg(self) -> Tensor {
        unwrap_op(Tensor::neg(&self))
    }
}

#[cfg(test)]
mod tests {
    use crate::device::Device;
    use crate::dtype::DType;
    use crate::tensor::Tensor;

    const CPU: Device = Device::Cpu;

    fn t(data: &[f32], shape: impl Into<crate::shape::Shape>) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape, &CPU).unwrap()
    }

    fn v(t: &Tensor) -> Vec<f32> {
        t.to_vec::<f32>().unwrap()
    }

    /// The `Display` text of a failing op (`Tensor` has no `Debug` until
    /// so `unwrap_err` is not available here).
    fn err_text(r: crate::Result<Tensor>) -> String {
        match r {
            Ok(_) => panic!("expected the op to fail"),
            Err(e) => e.to_string(),
        }
    }

    /// Run `f`, expecting it to panic, and return the panic message.
    ///
    /// `Tensor` is deliberately not `UnwindSafe` — its autograd node holds a
    /// `Box<dyn Fn>` — so the closure is wrapped in `AssertUnwindSafe`. That
    /// is sound here: the tensors are read-only inputs and nothing observes
    /// them after the unwind.
    fn panic_message<F: FnOnce() -> Tensor>(f: F) -> String {
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
        std::panic::set_hook(hook);
        let payload = caught.expect_err("the operator should have panicked");
        payload
            .downcast_ref::<String>()
            .expect("panic payload should be a String")
            .clone()
    }

    #[test]
    fn operators_agree_with_the_named_methods() {
        let a = t(&[1.0, 2.0, 3.0, 4.0], [2, 2]);
        let b = t(&[10.0, 20.0, 30.0, 40.0], [2, 2]);
        assert_eq!(v(&(&a + &b)), v(&a.add(&b).unwrap()));
        assert_eq!(v(&(&a - &b)), v(&a.sub(&b).unwrap()));
        assert_eq!(v(&(&a * &b)), v(&a.mul(&b).unwrap()));
        assert_eq!(v(&(&b / &a)), v(&b.div(&a).unwrap()));
    }

    #[test]
    fn every_ownership_combination_compiles() {
        let a = t(&[1.0, 2.0], [2]);
        let b = t(&[3.0, 4.0], [2]);
        let expected = vec![4.0, 6.0];
        assert_eq!(v(&(&a + &b)), expected);
        assert_eq!(v(&(&a + b.clone())), expected);
        assert_eq!(v(&(a.clone() + &b)), expected);
        assert_eq!(v(&(a + b)), expected);
    }

    #[test]
    fn scalar_right_hand_sides() {
        let a = t(&[1.0, 2.0], [2]);
        assert_eq!(v(&(&a + 1.0)), vec![2.0, 3.0]);
        assert_eq!(v(&(&a - 1.0)), vec![0.0, 1.0]);
        assert_eq!(v(&(&a * 3.0)), vec![3.0, 6.0]);
        assert_eq!(v(&(a / 2.0)), vec![0.5, 1.0]);
    }

    #[test]
    fn operators_broadcast_like_the_named_methods() {
        let m = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
        let row = t(&[10.0, 20.0, 30.0], [3]);
        let out = &m + &row;
        assert_eq!(out.dims(), &[2, 3]);
        assert_eq!(v(&out), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn a_failing_operator_panics_with_the_named_error_message() {
        let a = t(&[1.0, 2.0], [2]);
        let b = t(&[1.0, 2.0, 3.0], [3]);
        let expected = err_text(a.mul(&b));
        assert_eq!(expected, "mul: shape mismatch: lhs [2] vs rhs [3]");
        assert_eq!(panic_message(|| &a * &b), expected);
    }

    #[test]
    fn dtype_mismatch_panics_through_the_operator_too() {
        let f = t(&[1.0], [1]);
        let i = Tensor::from_vec(vec![1i64], [1], &CPU).unwrap();
        let expected = err_text(f.add(&i));
        assert!(expected.starts_with("add: dtype mismatch"));
        assert!(expected.contains("to_dtype"));
        assert_eq!(panic_message(|| &f + &i), expected);
    }

    #[test]
    fn sugar_preserves_dtype_and_device() {
        let i = Tensor::from_vec(vec![1i64, 2], [2], &CPU).unwrap();
        let out = &i * 3.0;
        assert_eq!(out.dtype(), DType::I64);
        assert_eq!(out.device(), CPU);
        assert_eq!(out.to_vec::<i64>().unwrap(), vec![3, 6]);
    }

    #[test]
    fn negation_agrees_with_the_named_method_in_both_ownership_forms() {
        let a = t(&[1.0, -2.0, 0.0], [3]);
        let expected = v(&a.neg().unwrap());
        assert_eq!(expected, vec![-1.0, 2.0, 0.0]);
        assert_eq!(v(&-&a), expected);
        assert_eq!(v(&-a), expected);
    }

    #[test]
    fn scalar_left_hand_sides_agree_with_the_named_spelling() {
        let a = t(&[1.0, 2.0], [2]);
        assert_eq!(v(&(1.0 + &a)), v(&a.add_scalar(1.0).unwrap()));
        assert_eq!(v(&(3.0 * &a)), v(&a.mul_scalar(3.0).unwrap()));
        // The reversed forms have no named spelling; they equal the
        // composition the impl documents.
        assert_eq!(
            v(&(3.0 - &a)),
            v(&a.neg().unwrap().add_scalar(3.0).unwrap())
        );
        assert_eq!(
            v(&(8.0 / &a)),
            v(&Tensor::full([2], 8.0, DType::F32, &CPU)
                .unwrap()
                .div(&a)
                .unwrap())
        );
        // Both ownership forms of the right-hand side.
        assert_eq!(v(&(1.0 + a.clone())), vec![2.0, 3.0]);
        assert_eq!(v(&(3.0 * a.clone())), vec![3.0, 6.0]);
        assert_eq!(v(&(3.0 - a.clone())), vec![2.0, 1.0]);
        assert_eq!(v(&(8.0 / a)), vec![8.0, 4.0]);
    }

    #[test]
    fn reversed_scalar_operators_do_not_commute_their_operands() {
        // `[1, 4]` is chosen so every reversed result differs element-wise
        // from the forward one: an operand-order bug cannot pass this.
        let a = t(&[1.0, 4.0], [2]);
        assert_eq!(v(&(3.0 - &a)), vec![2.0, -1.0]);
        assert_eq!(v(&(&a - 3.0)), vec![-2.0, 1.0]);
        assert_eq!(v(&(8.0 / &a)), vec![8.0, 2.0]);
        assert_eq!(v(&(&a / 8.0)), vec![0.125, 0.5]);
    }

    #[test]
    fn reversed_scalar_operators_differentiate_through_their_composition() {
        let x = t(&[1.0, 2.0], [2]).traced().unwrap();

        // d/dx (3 - x) = -1.
        let g = (3.0 - &x).sum_all().unwrap().backward().unwrap();
        assert_eq!(v(&g.wrt_input(&x).unwrap()), vec![-1.0, -1.0]);

        // d/dx (8 / x) = -8/x²: -8 at x = 1, -2 at x = 2.
        let g = (8.0 / &x).sum_all().unwrap().backward().unwrap();
        assert_eq!(v(&g.wrt_input(&x).unwrap()), vec![-8.0, -2.0]);
    }

    #[test]
    fn a_failing_reversed_operator_panics_with_the_forward_scalar_op_name() {
        let b = Tensor::from_vec(vec![true, false], [2], &CPU).unwrap();
        let expected = err_text(b.sub_scalar(2.0));
        assert_eq!(expected, "sub_scalar: unsupported on cpu for dtype bool");
        assert_eq!(panic_message(|| 2.0 - &b), expected);

        let expected = err_text(b.div_scalar(2.0));
        assert_eq!(expected, "div_scalar: unsupported on cpu for dtype bool");
        assert_eq!(panic_message(|| 2.0 / &b), expected);
    }

    #[test]
    #[should_panic(expected = "neg: unsupported on cpu for dtype bool")]
    fn negating_a_bool_tensor_panics_with_the_named_error_message() {
        let b = Tensor::from_vec(vec![true], [1], &CPU).unwrap();
        let _ = -&b;
    }
}
