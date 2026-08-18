use super::{
    ArgKeepDimOutput, ArgOutput, KeepDimOutput, RemoveAxisOutput, ScalarOutput, dynamic, wrap,
};
use crate::typed::{
    DYN, FloatElement, NumericElement, Placement, Tensor0, Tensor1, Tensor2, Tensor3, Tensor4,
    Tensor5, Tensor6, Tensor7, Tensor8, TypedTensor,
};
use crate::{Result, Tensor};

fn dynamic_reduction<T: TypedTensor>(
    input: &T,
    op: &'static str,
    reduction: impl FnOnce(&Tensor) -> Result<Tensor>,
) -> Result<Tensor> {
    reduction(dynamic(input, op)?).map_err(|error| error.with_op(op))
}

// One macro per reduction signature shape. Each expands to a single method whose
// name is also its runtime method name and its error op label; the element bound
// comes from the enclosing impl block, not from the macro.

macro_rules! remove_axis_reduction {
    ($method:ident) => {
        #[doc = concat!(
                            "Reduces axis `AXIS` with `",
                            stringify!($method),
                            "`, removing that axis from the output type. `AXIS` is checked at \
             compile time when it is statically known and the operation still \
             validates the runtime binding."
                        )]
        pub fn $method<const AXIS: usize>(&self) -> Result<<Self as RemoveAxisOutput<AXIS>>::Output>
        where
            Self: RemoveAxisOutput<AXIS>,
        {
            let op = stringify!($method);
            wrap(self, dynamic(self, op)?.$method(AXIS as isize)?, op)
        }
    };
}

macro_rules! keepdim_reduction {
    ($method:ident) => {
        #[doc = concat!(
                                    "Reduces axis `AXIS` with `",
                                    stringify!($method),
                                    "`, retaining that axis with length one in the output type."
                                )]
        pub fn $method<const AXIS: usize>(&self) -> Result<<Self as KeepDimOutput<AXIS>>::Output>
        where
            Self: KeepDimOutput<AXIS>,
        {
            let op = stringify!($method);
            wrap(self, dynamic(self, op)?.$method(AXIS as isize)?, op)
        }
    };
}

macro_rules! whole_reduction {
    ($method:ident) => {
        #[doc = concat!(
                    "Reduces all elements with `",
                    stringify!($method),
                    "` and returns a rank-zero tensor with the same element type and placement."
                )]
        pub fn $method(&self) -> Result<<Self as ScalarOutput>::Output> {
            let op = stringify!($method);
            wrap(self, dynamic(self, op)?.$method()?, op)
        }
    };
}

macro_rules! arg_reduction {
    ($method:ident) => {
        #[doc = concat!(
                            "Returns the `i64` index of the `",
                            stringify!($method),
                            "` value along axis `AXIS`, removing that axis from the output type."
                        )]
        pub fn $method<const AXIS: usize>(&self) -> Result<<Self as ArgOutput<AXIS>>::Output>
        where
            Self: ArgOutput<AXIS>,
        {
            let op = stringify!($method);
            wrap(self, dynamic(self, op)?.$method(AXIS as isize)?, op)
        }
    };
}

macro_rules! arg_keepdim_reduction {
    ($method:ident) => {
        #[doc = concat!(
                            "Returns the `i64` index of the `",
                            stringify!($method),
                            "` value along axis `AXIS`, retaining that axis with length one."
                        )]
        pub fn $method<const AXIS: usize>(&self) -> Result<<Self as ArgKeepDimOutput<AXIS>>::Output>
        where
            Self: ArgKeepDimOutput<AXIS>,
        {
            let op = stringify!($method);
            wrap(self, dynamic(self, op)?.$method(AXIS as isize)?, op)
        }
    };
}

macro_rules! shape_preserving_reduction {
    ($method:ident) => {
        #[doc = concat!(
                    "Applies `",
                    stringify!($method),
                    "` along axis `AXIS` while preserving the input shape and typed markers."
                )]
        pub fn $method<const AXIS: usize>(&self) -> Result<Self>
        where
            Self: KeepDimOutput<AXIS>,
        {
            let op = stringify!($method);
            wrap(self, dynamic(self, op)?.$method(AXIS as isize)?, op)
        }
    };
}

// Runtime-axis escape. The output type cannot be projected from an axis known
// only at runtime, so the enclosing rank macro passes it in.
macro_rules! dyn_reduction {
    ($method:ident, $runtime:ident, $output:ty) => {
        #[doc = concat!(
                            "Runtime-axis form of `",
                            stringify!($method),
                            "`. Because the axis is not a const generic, dynamic dimensions are \
             used for the affected output axes."
                        )]
        pub fn $method(&self, axis: isize) -> Result<$output> {
            let op = stringify!($method);
            wrap(self, dynamic_reduction(self, op, |x| x.$runtime(axis))?, op)
        }
    };
}

macro_rules! numeric_reductions {
    ($name:ident, [$($dim:ident),+], $removed:ty, $dynamic:ty, $arg_removed:ty, $arg_dynamic:ty) => {
        impl<$(const $dim: usize,)+ E: NumericElement, P: Placement>
            $name<$($dim,)+ E, P>
        {
            remove_axis_reduction!(sum);
            remove_axis_reduction!(mean);
            remove_axis_reduction!(max);
            remove_axis_reduction!(min);

            keepdim_reduction!(sum_keepdim);
            keepdim_reduction!(mean_keepdim);
            keepdim_reduction!(max_keepdim);
            keepdim_reduction!(min_keepdim);

            whole_reduction!(sum_all);
            whole_reduction!(mean_all);
            whole_reduction!(max_all);
            whole_reduction!(min_all);

            arg_reduction!(argmax);
            arg_reduction!(argmin);

            arg_keepdim_reduction!(argmax_keepdim);
            arg_keepdim_reduction!(argmin_keepdim);

            dyn_reduction!(sum_dyn, sum, $removed);
            dyn_reduction!(mean_dyn, mean, $removed);
            dyn_reduction!(max_dyn, max, $removed);
            dyn_reduction!(min_dyn, min, $removed);

            dyn_reduction!(sum_keepdim_dyn, sum_keepdim, $dynamic);
            dyn_reduction!(mean_keepdim_dyn, mean_keepdim, $dynamic);
            dyn_reduction!(max_keepdim_dyn, max_keepdim, $dynamic);
            dyn_reduction!(min_keepdim_dyn, min_keepdim, $dynamic);

            dyn_reduction!(argmax_dyn, argmax, $arg_removed);
            dyn_reduction!(argmin_dyn, argmin, $arg_removed);

            dyn_reduction!(argmax_keepdim_dyn, argmax_keepdim, $arg_dynamic);
            dyn_reduction!(argmin_keepdim_dyn, argmin_keepdim, $arg_dynamic);
        }
    };
}

macro_rules! float_reductions {
    ($name:ident, [$($dim:ident),+], $removed:ty, $dynamic:ty) => {
        impl<$(const $dim: usize,)+ E: FloatElement, P: Placement>
            $name<$($dim,)+ E, P>
        {
            remove_axis_reduction!(var);
            remove_axis_reduction!(std);

            keepdim_reduction!(var_keepdim);
            keepdim_reduction!(std_keepdim);

            whole_reduction!(var_all);
            whole_reduction!(std_all);

            shape_preserving_reduction!(softmax);
            shape_preserving_reduction!(log_softmax);

            dyn_reduction!(var_dyn, var, $removed);
            dyn_reduction!(std_dyn, std, $removed);

            dyn_reduction!(var_keepdim_dyn, var_keepdim, $dynamic);
            dyn_reduction!(std_keepdim_dyn, std_keepdim, $dynamic);

            dyn_reduction!(softmax_dyn, softmax, Self);
            dyn_reduction!(log_softmax_dyn, log_softmax, Self);
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
    use crate::typed::{DeviceBinding, DeviceCtx};
    use crate::{Device, Error, Grads};
    use std::sync::Arc;

    trait Same<T> {}
    impl<T> Same<T> for T {}

    fn exact<T: Same<Expected>, Expected>(_: &T) {}

    #[test]
    fn frozen_outputs_are_exact_for_every_rank() {
        let ctx = DeviceCtx::cpu().unwrap();
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
        let ctx = DeviceCtx::cpu().unwrap();
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
        let ctx = DeviceCtx::cpu().unwrap();
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
        let ctx = DeviceCtx::cpu().unwrap();
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
        let ctx = DeviceCtx::cpu().unwrap();
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
