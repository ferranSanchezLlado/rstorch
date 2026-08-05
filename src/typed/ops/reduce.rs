use super::{
    ArgKeepDimOutput, ArgOutput, KeepDimOutput, RemoveAxisOutput, ScalarOutput, relabel_error_op,
};
use crate::typed::device::validate_binding;
use crate::typed::tensor::checked_wrap;
use crate::typed::{
    DYN, FloatElement, NumericElement, Placement, Tensor0, Tensor1, Tensor2, Tensor3, Tensor4,
    Tensor5, Tensor6, Tensor7, Tensor8, TypedTensor,
};
use crate::{Result, Tensor};
use std::sync::Arc;

fn wrap<T: TypedTensor, O: TypedTensor>(input: &T, output: Tensor, op: &'static str) -> Result<O> {
    checked_wrap(output, Arc::clone(input.binding()), op)
}

fn dynamic<'a, T: TypedTensor>(input: &'a T, op: &'static str) -> Result<&'a Tensor> {
    validate_binding::<T::Placement>(input.binding(), op)?;
    Ok(input.dynamic())
}

fn dynamic_reduction<T: TypedTensor>(
    input: &T,
    op: &'static str,
    reduction: impl FnOnce(&Tensor) -> Result<Tensor>,
) -> Result<Tensor> {
    reduction(dynamic(input, op)?).map_err(|error| relabel_error_op(op, error))
}

macro_rules! numeric_reductions {
    ($name:ident, [$($dim:ident),+], $removed:ty, $dynamic:ty, $arg_removed:ty, $arg_dynamic:ty) => {
        impl<$(const $dim: usize,)+ E: NumericElement, P: Placement>
            $name<$($dim,)+ E, P>
        {
            pub fn sum<const AXIS: usize>(
                &self,
            ) -> Result<<Self as RemoveAxisOutput<AXIS>>::Output>
            where
                Self: RemoveAxisOutput<AXIS>,
            {
                wrap(self, dynamic(self, "sum")?.sum(AXIS as isize)?, "sum")
            }

            pub fn sum_keepdim<const AXIS: usize>(
                &self,
            ) -> Result<<Self as KeepDimOutput<AXIS>>::Output>
            where
                Self: KeepDimOutput<AXIS>,
            {
                wrap(
                    self,
                    dynamic(self, "sum_keepdim")?.sum_keepdim(AXIS as isize)?,
                    "sum_keepdim",
                )
            }

            pub fn sum_all(&self) -> Result<<Self as ScalarOutput>::Output> {
                wrap(self, dynamic(self, "sum_all")?.sum_all()?, "sum_all")
            }

            pub fn mean<const AXIS: usize>(
                &self,
            ) -> Result<<Self as RemoveAxisOutput<AXIS>>::Output>
            where
                Self: RemoveAxisOutput<AXIS>,
            {
                wrap(self, dynamic(self, "mean")?.mean(AXIS as isize)?, "mean")
            }

            pub fn mean_keepdim<const AXIS: usize>(
                &self,
            ) -> Result<<Self as KeepDimOutput<AXIS>>::Output>
            where
                Self: KeepDimOutput<AXIS>,
            {
                wrap(
                    self,
                    dynamic(self, "mean_keepdim")?.mean_keepdim(AXIS as isize)?,
                    "mean_keepdim",
                )
            }

            pub fn mean_all(&self) -> Result<<Self as ScalarOutput>::Output> {
                wrap(self, dynamic(self, "mean_all")?.mean_all()?, "mean_all")
            }

            pub fn max<const AXIS: usize>(
                &self,
            ) -> Result<<Self as RemoveAxisOutput<AXIS>>::Output>
            where
                Self: RemoveAxisOutput<AXIS>,
            {
                wrap(self, dynamic(self, "max")?.max(AXIS as isize)?, "max")
            }

            pub fn max_keepdim<const AXIS: usize>(
                &self,
            ) -> Result<<Self as KeepDimOutput<AXIS>>::Output>
            where
                Self: KeepDimOutput<AXIS>,
            {
                wrap(
                    self,
                    dynamic(self, "max_keepdim")?.max_keepdim(AXIS as isize)?,
                    "max_keepdim",
                )
            }

            pub fn max_all(&self) -> Result<<Self as ScalarOutput>::Output> {
                wrap(self, dynamic(self, "max_all")?.max_all()?, "max_all")
            }

            pub fn min<const AXIS: usize>(
                &self,
            ) -> Result<<Self as RemoveAxisOutput<AXIS>>::Output>
            where
                Self: RemoveAxisOutput<AXIS>,
            {
                wrap(self, dynamic(self, "min")?.min(AXIS as isize)?, "min")
            }

            pub fn min_keepdim<const AXIS: usize>(
                &self,
            ) -> Result<<Self as KeepDimOutput<AXIS>>::Output>
            where
                Self: KeepDimOutput<AXIS>,
            {
                wrap(
                    self,
                    dynamic(self, "min_keepdim")?.min_keepdim(AXIS as isize)?,
                    "min_keepdim",
                )
            }

            pub fn min_all(&self) -> Result<<Self as ScalarOutput>::Output> {
                wrap(self, dynamic(self, "min_all")?.min_all()?, "min_all")
            }

            pub fn argmax<const AXIS: usize>(
                &self,
            ) -> Result<<Self as ArgOutput<AXIS>>::Output>
            where
                Self: ArgOutput<AXIS>,
            {
                wrap(self, dynamic(self, "argmax")?.argmax(AXIS as isize)?, "argmax")
            }

            pub fn argmax_keepdim<const AXIS: usize>(
                &self,
            ) -> Result<<Self as ArgKeepDimOutput<AXIS>>::Output>
            where
                Self: ArgKeepDimOutput<AXIS>,
            {
                wrap(
                    self,
                    dynamic(self, "argmax_keepdim")?.argmax_keepdim(AXIS as isize)?,
                    "argmax_keepdim",
                )
            }

            pub fn argmin<const AXIS: usize>(
                &self,
            ) -> Result<<Self as ArgOutput<AXIS>>::Output>
            where
                Self: ArgOutput<AXIS>,
            {
                wrap(self, dynamic(self, "argmin")?.argmin(AXIS as isize)?, "argmin")
            }

            pub fn argmin_keepdim<const AXIS: usize>(
                &self,
            ) -> Result<<Self as ArgKeepDimOutput<AXIS>>::Output>
            where
                Self: ArgKeepDimOutput<AXIS>,
            {
                wrap(
                    self,
                    dynamic(self, "argmin_keepdim")?.argmin_keepdim(AXIS as isize)?,
                    "argmin_keepdim",
                )
            }

            pub fn sum_dyn(&self, axis: isize) -> Result<$removed> {
                wrap(self, dynamic_reduction(self, "sum_dyn", |x| x.sum(axis))?, "sum_dyn")
            }

            pub fn sum_keepdim_dyn(&self, axis: isize) -> Result<$dynamic> {
                wrap(self, dynamic_reduction(self, "sum_keepdim_dyn", |x| x.sum_keepdim(axis))?, "sum_keepdim_dyn")
            }

            pub fn mean_dyn(&self, axis: isize) -> Result<$removed> {
                wrap(self, dynamic_reduction(self, "mean_dyn", |x| x.mean(axis))?, "mean_dyn")
            }

            pub fn mean_keepdim_dyn(&self, axis: isize) -> Result<$dynamic> {
                wrap(self, dynamic_reduction(self, "mean_keepdim_dyn", |x| x.mean_keepdim(axis))?, "mean_keepdim_dyn")
            }

            pub fn max_dyn(&self, axis: isize) -> Result<$removed> {
                wrap(self, dynamic_reduction(self, "max_dyn", |x| x.max(axis))?, "max_dyn")
            }

            pub fn max_keepdim_dyn(&self, axis: isize) -> Result<$dynamic> {
                wrap(self, dynamic_reduction(self, "max_keepdim_dyn", |x| x.max_keepdim(axis))?, "max_keepdim_dyn")
            }

            pub fn min_dyn(&self, axis: isize) -> Result<$removed> {
                wrap(self, dynamic_reduction(self, "min_dyn", |x| x.min(axis))?, "min_dyn")
            }

            pub fn min_keepdim_dyn(&self, axis: isize) -> Result<$dynamic> {
                wrap(self, dynamic_reduction(self, "min_keepdim_dyn", |x| x.min_keepdim(axis))?, "min_keepdim_dyn")
            }

            pub fn argmax_dyn(&self, axis: isize) -> Result<$arg_removed> {
                wrap(self, dynamic_reduction(self, "argmax_dyn", |x| x.argmax(axis))?, "argmax_dyn")
            }

            pub fn argmax_keepdim_dyn(&self, axis: isize) -> Result<$arg_dynamic> {
                wrap(self, dynamic_reduction(self, "argmax_keepdim_dyn", |x| x.argmax_keepdim(axis))?, "argmax_keepdim_dyn")
            }

            pub fn argmin_dyn(&self, axis: isize) -> Result<$arg_removed> {
                wrap(self, dynamic_reduction(self, "argmin_dyn", |x| x.argmin(axis))?, "argmin_dyn")
            }

            pub fn argmin_keepdim_dyn(&self, axis: isize) -> Result<$arg_dynamic> {
                wrap(self, dynamic_reduction(self, "argmin_keepdim_dyn", |x| x.argmin_keepdim(axis))?, "argmin_keepdim_dyn")
            }
        }
    };
}

macro_rules! float_reductions {
    ($name:ident, [$($dim:ident),+], $removed:ty, $dynamic:ty) => {
        impl<$(const $dim: usize,)+ E: FloatElement, P: Placement>
            $name<$($dim,)+ E, P>
        {
            pub fn var<const AXIS: usize>(
                &self,
            ) -> Result<<Self as RemoveAxisOutput<AXIS>>::Output>
            where
                Self: RemoveAxisOutput<AXIS>,
            {
                wrap(self, dynamic(self, "var")?.var(AXIS as isize)?, "var")
            }

            pub fn var_keepdim<const AXIS: usize>(
                &self,
            ) -> Result<<Self as KeepDimOutput<AXIS>>::Output>
            where
                Self: KeepDimOutput<AXIS>,
            {
                wrap(
                    self,
                    dynamic(self, "var_keepdim")?.var_keepdim(AXIS as isize)?,
                    "var_keepdim",
                )
            }

            pub fn var_all(&self) -> Result<<Self as ScalarOutput>::Output> {
                wrap(self, dynamic(self, "var_all")?.var_all()?, "var_all")
            }

            pub fn std<const AXIS: usize>(
                &self,
            ) -> Result<<Self as RemoveAxisOutput<AXIS>>::Output>
            where
                Self: RemoveAxisOutput<AXIS>,
            {
                wrap(self, dynamic(self, "std")?.std(AXIS as isize)?, "std")
            }

            pub fn std_keepdim<const AXIS: usize>(
                &self,
            ) -> Result<<Self as KeepDimOutput<AXIS>>::Output>
            where
                Self: KeepDimOutput<AXIS>,
            {
                wrap(
                    self,
                    dynamic(self, "std_keepdim")?.std_keepdim(AXIS as isize)?,
                    "std_keepdim",
                )
            }

            pub fn std_all(&self) -> Result<<Self as ScalarOutput>::Output> {
                wrap(self, dynamic(self, "std_all")?.std_all()?, "std_all")
            }

            pub fn softmax<const AXIS: usize>(&self) -> Result<Self>
            where
                Self: KeepDimOutput<AXIS>,
            {
                wrap(self, dynamic(self, "softmax")?.softmax(AXIS as isize)?, "softmax")
            }

            pub fn log_softmax<const AXIS: usize>(&self) -> Result<Self>
            where
                Self: KeepDimOutput<AXIS>,
            {
                wrap(
                    self,
                    dynamic(self, "log_softmax")?.log_softmax(AXIS as isize)?,
                    "log_softmax",
                )
            }

            pub fn var_dyn(&self, axis: isize) -> Result<$removed> {
                wrap(self, dynamic_reduction(self, "var_dyn", |x| x.var(axis))?, "var_dyn")
            }

            pub fn var_keepdim_dyn(&self, axis: isize) -> Result<$dynamic> {
                wrap(self, dynamic_reduction(self, "var_keepdim_dyn", |x| x.var_keepdim(axis))?, "var_keepdim_dyn")
            }

            pub fn std_dyn(&self, axis: isize) -> Result<$removed> {
                wrap(self, dynamic_reduction(self, "std_dyn", |x| x.std(axis))?, "std_dyn")
            }

            pub fn std_keepdim_dyn(&self, axis: isize) -> Result<$dynamic> {
                wrap(self, dynamic_reduction(self, "std_keepdim_dyn", |x| x.std_keepdim(axis))?, "std_keepdim_dyn")
            }

            pub fn softmax_dyn(&self, axis: isize) -> Result<Self> {
                wrap(self, dynamic_reduction(self, "softmax_dyn", |x| x.softmax(axis))?, "softmax_dyn")
            }

            pub fn log_softmax_dyn(&self, axis: isize) -> Result<Self> {
                wrap(self, dynamic_reduction(self, "log_softmax_dyn", |x| x.log_softmax(axis))?, "log_softmax_dyn")
            }
        }
    };
}

macro_rules! reductions_for_rank {
    ($name:ident, [$($dim:ident),+], $removed:ty, $dynamic:ty, $arg_removed:ty, $arg_dynamic:ty) => {
        numeric_reductions!($name, [$($dim),+], $removed, $dynamic, $arg_removed, $arg_dynamic);
        float_reductions!($name, [$($dim),+], $removed, $dynamic);
    };
}

reductions_for_rank!(Tensor1, [D0], Tensor0<E, P>, Tensor1<DYN, E, P>, Tensor0<i64, P>, Tensor1<DYN, i64, P>);
reductions_for_rank!(Tensor2, [D0, D1], Tensor1<DYN, E, P>, Tensor2<DYN, DYN, E, P>, Tensor1<DYN, i64, P>, Tensor2<DYN, DYN, i64, P>);
reductions_for_rank!(Tensor3, [D0, D1, D2], Tensor2<DYN, DYN, E, P>, Tensor3<DYN, DYN, DYN, E, P>, Tensor2<DYN, DYN, i64, P>, Tensor3<DYN, DYN, DYN, i64, P>);
reductions_for_rank!(Tensor4, [D0, D1, D2, D3], Tensor3<DYN, DYN, DYN, E, P>, Tensor4<DYN, DYN, DYN, DYN, E, P>, Tensor3<DYN, DYN, DYN, i64, P>, Tensor4<DYN, DYN, DYN, DYN, i64, P>);
reductions_for_rank!(Tensor5, [D0, D1, D2, D3, D4], Tensor4<DYN, DYN, DYN, DYN, E, P>, Tensor5<DYN, DYN, DYN, DYN, DYN, E, P>, Tensor4<DYN, DYN, DYN, DYN, i64, P>, Tensor5<DYN, DYN, DYN, DYN, DYN, i64, P>);
reductions_for_rank!(Tensor6, [D0, D1, D2, D3, D4, D5], Tensor5<DYN, DYN, DYN, DYN, DYN, E, P>, Tensor6<DYN, DYN, DYN, DYN, DYN, DYN, E, P>, Tensor5<DYN, DYN, DYN, DYN, DYN, i64, P>, Tensor6<DYN, DYN, DYN, DYN, DYN, DYN, i64, P>);
reductions_for_rank!(Tensor7, [D0, D1, D2, D3, D4, D5, D6], Tensor6<DYN, DYN, DYN, DYN, DYN, DYN, E, P>, Tensor7<DYN, DYN, DYN, DYN, DYN, DYN, DYN, E, P>, Tensor6<DYN, DYN, DYN, DYN, DYN, DYN, i64, P>, Tensor7<DYN, DYN, DYN, DYN, DYN, DYN, DYN, i64, P>);
reductions_for_rank!(Tensor8, [D0, D1, D2, D3, D4, D5, D6, D7], Tensor7<DYN, DYN, DYN, DYN, DYN, DYN, DYN, E, P>, Tensor8<DYN, DYN, DYN, DYN, DYN, DYN, DYN, DYN, E, P>, Tensor7<DYN, DYN, DYN, DYN, DYN, DYN, DYN, i64, P>, Tensor8<DYN, DYN, DYN, DYN, DYN, DYN, DYN, DYN, i64, P>);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed::sealed::TypedTensor as SealedTypedTensor;
    use crate::typed::{Cpu, DeviceBinding, DeviceCtx};
    use crate::{Device, Error, Grads};

    trait Same<T> {}
    impl<T> Same<T> for T {}

    fn exact<T: Same<Expected>, Expected>(_: &T) {}

    #[test]
    fn frozen_outputs_are_exact_for_every_rank() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let t1 = Tensor1::<1>::from_vec(vec![1.0], [1], &ctx).unwrap();
        let t2 = Tensor2::<1, 1>::from_vec(vec![1.0], [1, 1], &ctx).unwrap();
        let t3 = Tensor3::<1, 1, 1>::from_vec(vec![1.0], [1, 1, 1], &ctx).unwrap();
        let t4 = Tensor4::<1, 1, 1, 1>::from_vec(vec![1.0], [1, 1, 1, 1], &ctx).unwrap();
        let t5 = Tensor5::<1, 1, 1, 1, 1>::from_vec(vec![1.0], [1, 1, 1, 1, 1], &ctx).unwrap();
        let t6 =
            Tensor6::<1, 1, 1, 1, 1, 1>::from_vec(vec![1.0], [1, 1, 1, 1, 1, 1], &ctx).unwrap();
        let t7 = Tensor7::<1, 1, 1, 1, 1, 1, 1>::from_vec(vec![1.0], [1, 1, 1, 1, 1, 1, 1], &ctx)
            .unwrap();
        let t8 =
            Tensor8::<1, 1, 1, 1, 1, 1, 1, 1>::from_vec(vec![1.0], [1, 1, 1, 1, 1, 1, 1, 1], &ctx)
                .unwrap();

        exact::<_, Tensor0>(&t1.sum::<0>().unwrap());
        exact::<_, Tensor1<1>>(&t2.sum::<0>().unwrap());
        exact::<_, Tensor2<1, 1>>(&t3.sum::<0>().unwrap());
        exact::<_, Tensor3<1, 1, 1>>(&t4.sum::<0>().unwrap());
        exact::<_, Tensor4<1, 1, 1, 1>>(&t5.sum::<0>().unwrap());
        exact::<_, Tensor5<1, 1, 1, 1, 1>>(&t6.sum::<0>().unwrap());
        exact::<_, Tensor6<1, 1, 1, 1, 1, 1>>(&t7.sum::<0>().unwrap());
        exact::<_, Tensor7<1, 1, 1, 1, 1, 1, 1>>(&t8.sum::<0>().unwrap());
        exact::<_, Tensor8<1, 1, 1, 1, 1, 1, 1, 1>>(&t8.max_keepdim::<7>().unwrap());
        exact::<_, Tensor7<1, 1, 1, 1, 1, 1, 1, i64>>(&t8.argmax::<7>().unwrap());
    }

    #[test]
    fn typed_values_match_runtime_and_runtime_axes_erase_markers() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let x =
            Tensor2::<2, 3>::from_vec(vec![1.0f32, 5.0, 3.0, 4.0, 2.0, 6.0], [2, 3], &ctx).unwrap();

        assert_eq!(
            x.sum::<1>().unwrap().as_dynamic().to_vec::<f32>().unwrap(),
            x.as_dynamic().sum(1).unwrap().to_vec::<f32>().unwrap()
        );
        assert_eq!(
            x.mean::<0>().unwrap().as_dynamic().to_vec::<f32>().unwrap(),
            x.as_dynamic().mean(0).unwrap().to_vec::<f32>().unwrap()
        );
        assert_eq!(
            x.var::<1>().unwrap().as_dynamic().to_vec::<f32>().unwrap(),
            x.as_dynamic().var(1).unwrap().to_vec::<f32>().unwrap()
        );
        assert_eq!(
            x.softmax::<1>()
                .unwrap()
                .as_dynamic()
                .to_vec::<f32>()
                .unwrap(),
            x.as_dynamic().softmax(1).unwrap().to_vec::<f32>().unwrap()
        );
        assert_eq!(
            x.argmax::<1>()
                .unwrap()
                .as_dynamic()
                .to_vec::<i64>()
                .unwrap(),
            vec![1i64, 2]
        );

        let erased: Tensor1<DYN> = x.sum_dyn(-1).unwrap();
        let kept: Tensor2<DYN, DYN> = x.mean_keepdim_dyn(0).unwrap();
        let args: Tensor1<DYN, i64> = x.argmin_dyn(1).unwrap();
        assert_eq!(erased.dims(), [2]);
        assert_eq!(kept.dims(), [1, 3]);
        assert_eq!(args.as_dynamic().to_vec::<i64>().unwrap(), vec![0, 1]);
    }

    /// Every reduction's const `AXIS` must actually reach the runtime call.
    ///
    /// Axis 1 is used deliberately. The failure being guarded against is a
    /// method that ignores `AXIS` and passes a hardcoded `0`, so a test written
    /// with `AXIS = 0` would be satisfied by the very bug it exists to catch.
    /// The shape is non-square so a wrong axis changes the output length as well
    /// as its values — on a square tensor the two can coincide — and the data is
    /// asymmetric so no two axes agree by accident. `log_softmax` keeps its
    /// input shape, so only the values distinguish its axis.
    ///
    /// The twelve methods asserted here were each verified to survive replacing
    /// `AXIS as isize` with `0` before this test existed.
    #[test]
    fn every_reduction_passes_its_const_axis_to_the_runtime() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let x =
            Tensor2::<2, 3>::from_vec(vec![1.0f32, 5.0, 3.0, 4.0, 2.0, 6.0], [2, 3], &ctx).unwrap();

        macro_rules! agrees_with_runtime {
            ($name:literal, $element:ty, $typed:expr, $runtime:expr) => {{
                let typed = ($typed).unwrap();
                let runtime = ($runtime).unwrap();
                assert_eq!(
                    typed.dims().as_slice(),
                    runtime.dims(),
                    concat!(
                        $name,
                        ": wrong output shape, so the axis did not reach the runtime"
                    )
                );
                assert_eq!(
                    typed.as_dynamic().to_vec::<$element>().unwrap(),
                    runtime.to_vec::<$element>().unwrap(),
                    concat!($name, ": values disagree with the runtime on the same axis")
                );
            }};
        }

        agrees_with_runtime!(
            "sum_keepdim",
            f32,
            x.sum_keepdim::<1>(),
            x.as_dynamic().sum_keepdim(1)
        );
        agrees_with_runtime!(
            "mean_keepdim",
            f32,
            x.mean_keepdim::<1>(),
            x.as_dynamic().mean_keepdim(1)
        );
        agrees_with_runtime!(
            "max_keepdim",
            f32,
            x.max_keepdim::<1>(),
            x.as_dynamic().max_keepdim(1)
        );
        agrees_with_runtime!("min", f32, x.min::<1>(), x.as_dynamic().min(1));
        agrees_with_runtime!(
            "min_keepdim",
            f32,
            x.min_keepdim::<1>(),
            x.as_dynamic().min_keepdim(1)
        );
        agrees_with_runtime!(
            "argmax_keepdim",
            i64,
            x.argmax_keepdim::<1>(),
            x.as_dynamic().argmax_keepdim(1)
        );
        agrees_with_runtime!("argmin", i64, x.argmin::<1>(), x.as_dynamic().argmin(1));
        agrees_with_runtime!(
            "argmin_keepdim",
            i64,
            x.argmin_keepdim::<1>(),
            x.as_dynamic().argmin_keepdim(1)
        );
        agrees_with_runtime!(
            "var_keepdim",
            f32,
            x.var_keepdim::<1>(),
            x.as_dynamic().var_keepdim(1)
        );
        agrees_with_runtime!("std", f32, x.std::<1>(), x.as_dynamic().std(1));
        agrees_with_runtime!(
            "std_keepdim",
            f32,
            x.std_keepdim::<1>(),
            x.as_dynamic().std_keepdim(1)
        );
        agrees_with_runtime!(
            "log_softmax",
            f32,
            x.log_softmax::<1>(),
            x.as_dynamic().log_softmax(1)
        );

        // Guard the guard: axis 0 and axis 1 must genuinely differ for this data,
        // otherwise every assertion above would hold under a hardcoded 0.
        assert_ne!(
            x.as_dynamic().min(1).unwrap().to_vec::<f32>().unwrap(),
            x.as_dynamic().min(0).unwrap().to_vec::<f32>().unwrap()
        );
        assert_ne!(
            x.as_dynamic()
                .log_softmax(1)
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            x.as_dynamic()
                .log_softmax(0)
                .unwrap()
                .to_vec::<f32>()
                .unwrap()
        );
    }

    #[test]
    fn runtime_errors_and_empty_policies_are_preserved() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let x = Tensor2::<2, 3>::from_vec(vec![0.0f32; 6], [2, 3], &ctx).unwrap();
        macro_rules! assert_invalid_axis_op {
            ($expression:expr, $op:literal) => {
                assert!(matches!(
                    $expression,
                    Err(Error::InvalidAxis { op: $op, .. })
                ));
            };
        }
        assert_invalid_axis_op!(x.sum_dyn(2), "sum_dyn");
        assert_invalid_axis_op!(x.sum_keepdim_dyn(2), "sum_keepdim_dyn");
        assert_invalid_axis_op!(x.mean_dyn(2), "mean_dyn");
        assert_invalid_axis_op!(x.mean_keepdim_dyn(2), "mean_keepdim_dyn");
        assert_invalid_axis_op!(x.max_dyn(2), "max_dyn");
        assert_invalid_axis_op!(x.max_keepdim_dyn(2), "max_keepdim_dyn");
        assert_invalid_axis_op!(x.min_dyn(2), "min_dyn");
        assert_invalid_axis_op!(x.min_keepdim_dyn(2), "min_keepdim_dyn");
        assert_invalid_axis_op!(x.argmax_dyn(2), "argmax_dyn");
        assert_invalid_axis_op!(x.argmax_keepdim_dyn(2), "argmax_keepdim_dyn");
        assert_invalid_axis_op!(x.argmin_dyn(2), "argmin_dyn");
        assert_invalid_axis_op!(x.argmin_keepdim_dyn(2), "argmin_keepdim_dyn");
        assert_invalid_axis_op!(x.var_dyn(2), "var_dyn");
        assert_invalid_axis_op!(x.var_keepdim_dyn(2), "var_keepdim_dyn");
        assert_invalid_axis_op!(x.std_dyn(2), "std_dyn");
        assert_invalid_axis_op!(x.std_keepdim_dyn(2), "std_keepdim_dyn");
        assert_invalid_axis_op!(x.softmax_dyn(2), "softmax_dyn");
        assert_invalid_axis_op!(x.log_softmax_dyn(2), "log_softmax_dyn");

        let empty = Tensor2::<2, 0>::from_vec(Vec::<f32>::new(), [2, 0], &ctx).unwrap();
        assert_eq!(
            empty
                .sum::<1>()
                .unwrap()
                .as_dynamic()
                .to_vec::<f32>()
                .unwrap(),
            vec![0.0, 0.0]
        );
        assert!(matches!(empty.mean::<1>(), Err(Error::InvalidArg { .. })));
        assert!(matches!(empty.max::<1>(), Err(Error::InvalidArg { .. })));
        assert!(matches!(
            empty.softmax::<1>(),
            Err(Error::InvalidArg { .. })
        ));
    }

    #[test]
    fn reductions_reject_noncanonical_bindings_before_delegation() {
        let dynamic = Tensor::from_vec(vec![1.0f32, 2.0], [2], &Device::Cpu).unwrap();
        let forged = Arc::new(DeviceBinding {
            device: Device::Cpu,
        });
        let forged = <Tensor1<2> as SealedTypedTensor>::trusted_from_validated(dynamic, forged);

        assert!(matches!(
            forged.sum::<0>(),
            Err(Error::InvalidArg { op: "sum", .. })
        ));
        assert!(matches!(
            forged.sum_dyn(0),
            Err(Error::InvalidArg { op: "sum_dyn", .. })
        ));
    }

    #[test]
    fn typed_reductions_preserve_runtime_gradients() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let runtime = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2], &ctx.device())
            .unwrap()
            .traced()
            .unwrap();
        let input = Tensor2::<2, 2>::try_from_dynamic(runtime.clone(), &ctx).unwrap();
        let loss = input.mean::<1>().unwrap().sum_all().unwrap();
        let grads: Grads = loss.as_dynamic().backward().unwrap();
        assert_eq!(
            grads.wrt_input(&runtime).unwrap().to_vec::<f32>().unwrap(),
            vec![0.5; 4]
        );
    }
}
