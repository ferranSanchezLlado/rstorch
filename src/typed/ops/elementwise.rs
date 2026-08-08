//! Strict typed elementwise operations.

use super::{BooleanOutput, wrap};
use crate::typed::device::validate_binding;
use crate::typed::sealed::TypedTensor as SealedTypedTensor;
use crate::typed::{
    FloatElement, NumericElement, Placement, Tensor0, Tensor1, Tensor2, Tensor3, Tensor4, Tensor5,
    Tensor6, Tensor7, Tensor8, TypedTensor, typed_rank_table,
};
use crate::{Element, Error, Result};
use std::ops::{Add as TypedAdd, Div as TypedDiv, Mul as TypedMul, Sub as TypedSub};
use std::sync::Arc;

fn validate_operand<T: TypedTensor>(value: &T, op: &'static str) -> Result<()> {
    validate_binding::<T::Placement>(<T as SealedTypedTensor>::binding(value), op)
}

/// Requires two operands to share a binding and identical actual dimensions.
///
/// Two typed wrappers of the same type are the `L == R` case; nothing extra is
/// checked for them, so they use this function too.
fn validate_related<L, R>(lhs: &L, rhs: &R, op: &'static str) -> Result<()>
where
    L: TypedTensor,
    R: TypedTensor<Placement = L::Placement>,
{
    validate_operand(lhs, op)?;
    validate_operand(rhs, op)?;
    if !Arc::ptr_eq(
        <L as SealedTypedTensor>::binding(lhs),
        <R as SealedTypedTensor>::binding(rhs),
    ) {
        return Err(Error::InvalidArg {
            op,
            msg: "typed operands do not share the canonical placement binding".to_string(),
        });
    }
    if <L as SealedTypedTensor>::dynamic(lhs).dims()
        != <R as SealedTypedTensor>::dynamic(rhs).dims()
    {
        return Err(Error::ShapeMismatch {
            op,
            lhs: <L as SealedTypedTensor>::dynamic(lhs).shape().clone(),
            rhs: <R as SealedTypedTensor>::dynamic(rhs).shape().clone(),
        });
    }
    Ok(())
}

#[track_caller]
fn unwrap_op<T>(result: Result<T>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("{error}"),
    }
}

macro_rules! binary_numeric {
    ($method:ident) => {
        pub fn $method(&self, rhs: &Self) -> Result<Self> {
            let op = stringify!($method);
            validate_related(self, rhs, op)?;
            wrap(self, self.as_dynamic().$method(rhs.as_dynamic())?, op)
        }
    };
}

macro_rules! scalar_numeric {
    ($method:ident) => {
        pub fn $method(&self, value: f64) -> Result<Self> {
            let op = stringify!($method);
            validate_operand(self, op)?;
            wrap(self, self.as_dynamic().$method(value)?, op)
        }
    };
}

/// The element bound comes from the enclosing impl block, so one expansion
/// serves both the `NumericElement` and the `FloatElement` unary methods.
macro_rules! unary {
    ($method:ident) => {
        pub fn $method(&self) -> Result<Self> {
            let op = stringify!($method);
            validate_operand(self, op)?;
            wrap(self, self.as_dynamic().$method()?, op)
        }
    };
}

macro_rules! comparison {
    ($method:ident) => {
        pub fn $method(&self, rhs: &Self) -> Result<<Self as BooleanOutput>::Output> {
            let op = stringify!($method);
            validate_related(self, rhs, op)?;
            wrap(self, self.as_dynamic().$method(rhs.as_dynamic())?, op)
        }
    };
}

macro_rules! impl_elementwise {
    ($(($name:ident, $rank:literal, [$($dim:ident),*])),+ $(,)?) => {
        $(
            impl<$(const $dim: usize,)* E: NumericElement, P: Placement>
                $name<$($dim,)* E, P>
            {
                binary_numeric!(add);
                binary_numeric!(sub);
                binary_numeric!(mul);
                binary_numeric!(div);
                binary_numeric!(maximum);
                binary_numeric!(minimum);

                scalar_numeric!(add_scalar);
                scalar_numeric!(sub_scalar);
                scalar_numeric!(mul_scalar);
                scalar_numeric!(div_scalar);

                unary!(neg);
                unary!(abs);

                pub fn square(&self) -> Result<Self> {
                    validate_operand(self, "square")?;
                    wrap(self, self.as_dynamic().mul(self.as_dynamic())?, "square")
                }
            }

            impl<$(const $dim: usize,)* E: FloatElement, P: Placement>
                $name<$($dim,)* E, P>
            {
                unary!(relu);
                unary!(gelu);
                unary!(exp);
                unary!(ln);
                unary!(sqrt);
                unary!(tanh);
                unary!(sigmoid);
            }

            impl<$(const $dim: usize,)* E: Element, P: Placement>
                $name<$($dim,)* E, P>
            {
                comparison!(eq);
                comparison!(ne);
                comparison!(lt);
                comparison!(le);
                comparison!(gt);
                comparison!(ge);

                pub fn masked_fill(
                    &self,
                    mask: &$name<$($dim,)* bool, P>,
                    value: f64,
                ) -> Result<Self> {
                    validate_related(self, mask, "masked_fill")?;
                    wrap(
                        self,
                        self.as_dynamic().masked_fill(mask.as_dynamic(), value)?,
                        "masked_fill",
                    )
                }
            }

            impl<$(const $dim: usize,)* P: Placement> $name<$($dim,)* bool, P> {
                pub fn where_cond<E: Element>(
                    &self,
                    on_true: &$name<$($dim,)* E, P>,
                    on_false: &$name<$($dim,)* E, P>,
                ) -> Result<$name<$($dim,)* E, P>> {
                    validate_related(self, on_true, "where")?;
                    validate_related(self, on_false, "where")?;
                    validate_related(on_true, on_false, "where")?;
                    wrap(
                        self,
                        self.as_dynamic()
                            .where_cond(on_true.as_dynamic(), on_false.as_dynamic())?,
                        "where",
                    )
                }
            }

            macro_rules! binary_sugar_one {
                ($trait:ident, $method:ident, $named:ident, $scalar:ident) => {
                    impl<$(const $dim: usize,)* E: NumericElement, P: Placement>
                        $trait<&$name<$($dim,)* E, P>> for &$name<$($dim,)* E, P>
                    {
                        type Output = $name<$($dim,)* E, P>;
                        #[track_caller]
                        fn $method(self, rhs: &$name<$($dim,)* E, P>) -> Self::Output {
                            unwrap_op(<$name<$($dim,)* E, P>>::$named(self, rhs))
                        }
                    }

                    impl<$(const $dim: usize,)* E: NumericElement, P: Placement>
                        $trait<$name<$($dim,)* E, P>> for &$name<$($dim,)* E, P>
                    {
                        type Output = $name<$($dim,)* E, P>;
                        #[track_caller]
                        fn $method(self, rhs: $name<$($dim,)* E, P>) -> Self::Output {
                            unwrap_op(<$name<$($dim,)* E, P>>::$named(self, &rhs))
                        }
                    }

                    impl<$(const $dim: usize,)* E: NumericElement, P: Placement>
                        $trait<&$name<$($dim,)* E, P>> for $name<$($dim,)* E, P>
                    {
                        type Output = $name<$($dim,)* E, P>;
                        #[track_caller]
                        fn $method(self, rhs: &$name<$($dim,)* E, P>) -> Self::Output {
                            unwrap_op(<$name<$($dim,)* E, P>>::$named(&self, rhs))
                        }
                    }

                    impl<$(const $dim: usize,)* E: NumericElement, P: Placement>
                        $trait<$name<$($dim,)* E, P>> for $name<$($dim,)* E, P>
                    {
                        type Output = $name<$($dim,)* E, P>;
                        #[track_caller]
                        fn $method(self, rhs: $name<$($dim,)* E, P>) -> Self::Output {
                            unwrap_op(<$name<$($dim,)* E, P>>::$named(&self, &rhs))
                        }
                    }

                    impl<$(const $dim: usize,)* E: NumericElement, P: Placement> $trait<f64>
                        for &$name<$($dim,)* E, P>
                    {
                        type Output = $name<$($dim,)* E, P>;
                        #[track_caller]
                        fn $method(self, rhs: f64) -> Self::Output {
                            unwrap_op(self.$scalar(rhs))
                        }
                    }

                    impl<$(const $dim: usize,)* E: NumericElement, P: Placement> $trait<f64>
                        for $name<$($dim,)* E, P>
                    {
                        type Output = $name<$($dim,)* E, P>;
                        #[track_caller]
                        fn $method(self, rhs: f64) -> Self::Output {
                            unwrap_op(self.$scalar(rhs))
                        }
                    }
                };
            }

            binary_sugar_one!(TypedAdd, add, add, add_scalar);
            binary_sugar_one!(TypedSub, sub, sub, sub_scalar);
            binary_sugar_one!(TypedMul, mul, mul, mul_scalar);
            binary_sugar_one!(TypedDiv, div, div, div_scalar);
        )+
    };
}

typed_rank_table!(impl_elementwise);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed::{Cpu, DYN, DeviceCtx};
    use crate::{DType, Device, Shape, Tensor};

    fn values<E: Element>(tensor: &Tensor) -> Vec<E> {
        tensor.to_vec::<E>().unwrap()
    }

    #[test]
    fn typed_values_match_runtime_elementwise_operations() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let a = Tensor2::<2, 2>::from_vec(vec![-1.0, 2.0, 3.0, 4.0], [2, 2], &ctx).unwrap();
        let b = Tensor2::<2, 2>::from_vec(vec![2.0, 2.0, 1.0, 8.0], [2, 2], &ctx).unwrap();

        assert_eq!(
            values::<f32>(Tensor2::add(&a, &b).unwrap().as_dynamic()),
            vec![1.0, 4.0, 4.0, 12.0]
        );
        assert_eq!(
            values::<f32>(a.maximum(&b).unwrap().as_dynamic()),
            vec![2.0, 2.0, 3.0, 8.0]
        );
        assert_eq!(
            values::<f32>(a.abs().unwrap().as_dynamic()),
            vec![1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            values::<bool>(a.lt(&b).unwrap().as_dynamic()),
            vec![true, false, false, true]
        );

        let mask = a.lt(&b).unwrap();
        assert_eq!(
            values::<f32>(a.masked_fill(&mask, 9.0).unwrap().as_dynamic()),
            vec![9.0, 2.0, 3.0, 9.0]
        );
        assert_eq!(
            values::<f32>(mask.where_cond(&a, &b).unwrap().as_dynamic()),
            vec![-1.0, 2.0, 1.0, 4.0]
        );
    }

    #[test]
    fn dynamic_markers_do_not_enable_implicit_broadcasting() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let matrix = Tensor2::<DYN, DYN>::from_vec(vec![1.0f32; 6], [2, 3], &ctx).unwrap();
        let row = Tensor2::<DYN, DYN>::from_vec(vec![2.0f32; 3], [1, 3], &ctx).unwrap();
        assert!(matrix.as_dynamic().add(row.as_dynamic()).is_ok());
        assert!(matches!(
            Tensor2::add(&matrix, &row),
            Err(Error::ShapeMismatch { op: "add", lhs, rhs })
                if lhs == Shape::from([2, 3]) && rhs == Shape::from([1, 3])
        ));

        let expanded = Tensor2::<DYN, DYN>::try_from_dynamic(
            row.as_dynamic().broadcast_to([2, 3]).unwrap(),
            &ctx,
        )
        .unwrap();
        assert_eq!(
            values::<f32>(Tensor2::add(&matrix, &expanded).unwrap().as_dynamic()),
            vec![3.0; 6]
        );
    }

    /// Each binary method must report its OWN name. `binary_numeric!` derives the
    /// op from `stringify!($method)`, and only `add` was asserted anywhere — so
    /// replacing that with the literal `"add"`, making every method report
    /// `op: "add"`, passed the whole suite. One rejection per method fixes that.
    #[test]
    fn every_binary_method_reports_its_own_op_name() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let matrix = Tensor2::<DYN, DYN>::from_vec(vec![1.0f32; 6], [2, 3], &ctx).unwrap();
        let row = Tensor2::<DYN, DYN>::from_vec(vec![2.0f32; 3], [1, 3], &ctx).unwrap();

        // A DYN-marked shape disagreement defers to the runtime, so each call
        // rejects and the error carries the method's own op string.
        let cases: [(&str, Result<Tensor2<DYN, DYN>>); 6] = [
            ("add", Tensor2::add(&matrix, &row)),
            ("sub", Tensor2::sub(&matrix, &row)),
            ("mul", Tensor2::mul(&matrix, &row)),
            ("div", Tensor2::div(&matrix, &row)),
            ("maximum", Tensor2::maximum(&matrix, &row)),
            ("minimum", Tensor2::minimum(&matrix, &row)),
        ];
        for (expected, result) in cases {
            let error = result.expect_err("a strict shape disagreement must reject");
            assert!(
                matches!(&error, Error::ShapeMismatch { op, .. } if *op == expected),
                "expected op {expected:?}, got {error:?}"
            );
        }
    }

    #[test]
    fn operator_sugar_is_strict_and_matches_named_methods() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let a = Tensor1::<2>::from_vec(vec![1.0f32, 2.0], [2], &ctx).unwrap();
        let b = Tensor1::<2>::from_vec(vec![3.0f32, 4.0], [2], &ctx).unwrap();
        assert_eq!(values::<f32>((&a + &b).as_dynamic()), vec![4.0, 6.0]);
        assert_eq!(
            values::<f32>((a.clone() * 2.0).as_dynamic()),
            vec![2.0, 4.0]
        );

        let short = Tensor1::<DYN>::from_vec(vec![1.0f32], [1], &ctx).unwrap();
        let long = Tensor1::<DYN>::from_vec(vec![1.0f32, 2.0], [2], &ctx).unwrap();
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| &short + &long));
        assert!(panic.is_err());
    }

    #[test]
    fn runtime_errors_and_gradients_are_preserved() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let ints = Tensor1::<2, i64>::from_vec(vec![1, 2], [2], &ctx).unwrap();
        assert_eq!(
            values::<i64>(ints.div_scalar(0.0).unwrap().as_dynamic()),
            vec![0, 0]
        );

        let dynamic = Tensor::from_vec(vec![2.0f32, 3.0], [2], &Device::Cpu)
            .unwrap()
            .traced()
            .unwrap();
        let input = Tensor1::<2>::try_from_dynamic(dynamic, &ctx).unwrap();
        let output = input.square().unwrap();
        let grads = output.as_dynamic().backward().unwrap();
        assert_eq!(
            values::<f32>(&grads.wrt_input(input.as_dynamic()).unwrap()),
            vec![4.0, 6.0]
        );
    }

    #[test]
    fn forged_binding_is_rejected_before_runtime_delegation() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let dynamic = Tensor::zeros([2], DType::F32, &Device::Cpu).unwrap();
        let forged = Arc::new(crate::typed::DeviceBinding {
            device: Device::Cpu,
        });
        let value = <Tensor1<2> as SealedTypedTensor>::trusted_from_validated(dynamic, forged);
        assert!(matches!(
            value.neg(),
            Err(Error::InvalidArg { op: "neg", .. })
        ));
        drop(ctx);
    }

    fn assert_numeric_surface<T>(value: &T)
    where
        T: Clone
            + std::ops::Add<T, Output = T>
            + for<'a> std::ops::Add<&'a T, Output = T>
            + std::ops::Mul<f64, Output = T>,
        for<'a> &'a T: std::ops::Add<T, Output = T>
            + std::ops::Add<&'a T, Output = T>
            + std::ops::Mul<f64, Output = T>,
    {
        let _ = value;
    }

    #[test]
    fn representative_capability_and_operator_signatures_compile() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let value =
            Tensor8::<1, 1, 1, 1, 1, 1, 1, 1, i64>::from_vec(vec![1], [1; 8], &ctx).unwrap();
        assert_numeric_surface(&value);

        let bools = Tensor0::<bool>::from_vec(vec![true], [], &ctx).unwrap();
        let _: Tensor0<bool> = bools.eq(&bools).unwrap();
    }
}
