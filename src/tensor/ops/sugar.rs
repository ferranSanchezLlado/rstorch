//! Operator sugar: the panicking tier of the two-tier fallibility policy.
//!
//! Every named method returns [`Result`](crate::Result). The `std::ops`
//! spellings — `a + b`, `a - b`, `a * b`, `a / b`, plus the `tensor <op>
//! f64` forms — call exactly those methods and **panic with the identical
//! structured message** when they fail. The impls are `#[track_caller]`, so
//! the panic is reported at the user's expression, not inside this file:
//!
//! ```text
//! thread 'main' panicked at src/main.rs:12:17:
//! add: shape mismatch: lhs [2, 3] vs rhs [4, 5]
//! ```
//!
//! Nothing else lives here: there is no separate operator semantics, no
//! implicit promotion, and no in-place variant. `&a + &b` and
//! `a.add(&b)?` compute the same thing through the same code path, so the
//! sugar can never drift from the named surface.

use crate::error::Result;
use crate::tensor::Tensor;
use std::ops::{Add, Div, Mul, Sub};

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
        assert_eq!(v(&(a.clone() + b.clone())), expected);
    }

    #[test]
    fn scalar_right_hand_sides() {
        let a = t(&[1.0, 2.0], [2]);
        assert_eq!(v(&(&a + 1.0)), vec![2.0, 3.0]);
        assert_eq!(v(&(&a - 1.0)), vec![0.0, 1.0]);
        assert_eq!(v(&(&a * 3.0)), vec![3.0, 6.0]);
        assert_eq!(v(&(a.clone() / 2.0)), vec![0.5, 1.0]);
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
}
