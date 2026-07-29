use super::{
    BroadcastOutput, ConcatOutput, DynamicOutput, InsertAxisOutput, RemoveAxisOutput,
    ReplaceAxisOutput, ReshapeOutput, StackOutput, TransposeOutput, relabel_error_op,
};
use crate::typed::const_check::{assert_broadcast, assert_reshape_numel, assert_squeezable};
use crate::typed::device::validate_binding;
use crate::typed::sealed::TypedTensor as SealedTypedTensor;
use crate::typed::tensor::checked_wrap;
use crate::typed::{
    DYN, Placement, Tensor0, Tensor1, Tensor2, Tensor3, Tensor4, Tensor5, Tensor6, Tensor7,
    Tensor8, TypedTensor,
};
use crate::{Element, Error, Result, Shape, Tensor};
use std::sync::Arc;

fn dynamic<'a, T: TypedTensor>(input: &'a T, op: &'static str) -> Result<&'a Tensor> {
    validate_binding::<T::Placement>(input.binding(), op)?;
    Ok(input.dynamic())
}

fn validate_operands<T: TypedTensor>(tensors: &[&T], op: &'static str) -> Result<()> {
    let Some(first) = tensors.first() else {
        return Ok(());
    };
    validate_binding::<T::Placement>(first.binding(), op)?;
    for tensor in &tensors[1..] {
        validate_binding::<T::Placement>(tensor.binding(), op)?;
        if !Arc::ptr_eq(first.binding(), tensor.binding()) {
            return Err(Error::InvalidArg {
                op,
                msg: "operands do not share the canonical placement binding".into(),
            });
        }
    }
    Ok(())
}

macro_rules! impl_shape_ops {
    ($(($name:ident, [$($dim:ident),*])),+ $(,)?) => {
        $(
            impl<$(const $dim: usize,)* E: Element, P: Placement> $name<$($dim,)* E, P> {
                /// Reshapes into a caller-named typed target.
                pub fn reshape<Target>(&self, dims: Target::Dims) -> Result<Target>
                where
                    Target: TypedTensor<Elem = E, Placement = P>,
                    Self: ReshapeOutput<Target, Output = Target>,
                {
                    const {
                        assert_reshape_numel(
                            <Self as SealedTypedTensor>::MARKERS,
                            <Target as SealedTypedTensor>::MARKERS,
                        )
                    };
                    let tensor = dynamic(self, "reshape")?.reshape(Shape::from(dims.as_ref().to_vec()))?;
                    checked_wrap::<Target>(tensor, self.binding().clone(), "reshape")
                }

                /// Broadcasts explicitly into a caller-named typed target.
                pub fn broadcast_to<Target>(&self, dims: Target::Dims) -> Result<Target>
                where
                    Target: TypedTensor<Elem = E, Placement = P>,
                    Self: BroadcastOutput<Target, Output = Target>,
                {
                    const {
                        assert_broadcast(
                            <Self as SealedTypedTensor>::MARKERS,
                            <Target as SealedTypedTensor>::MARKERS,
                        )
                    };
                    let tensor = dynamic(self, "broadcast_to")?
                        .broadcast_to(Shape::from(dims.as_ref().to_vec()))?;
                    checked_wrap::<Target>(tensor, self.binding().clone(), "broadcast_to")
                }

                /// Permutes runtime axes and erases every dimension marker.
                pub fn permute(&self, axes: &[isize]) -> Result<<Self as DynamicOutput>::Output> {
                    let tensor = dynamic(self, "permute")?.permute(axes)?;
                    checked_wrap::<<Self as DynamicOutput>::Output>(
                        tensor,
                        self.binding().clone(),
                        "permute",
                    )
                }
            }
        )+
    };
}

impl_shape_ops! {
    (Tensor0, []),
    (Tensor1, [D0]),
    (Tensor2, [D0, D1]),
    (Tensor3, [D0, D1, D2]),
    (Tensor4, [D0, D1, D2, D3]),
    (Tensor5, [D0, D1, D2, D3, D4]),
    (Tensor6, [D0, D1, D2, D3, D4, D5]),
    (Tensor7, [D0, D1, D2, D3, D4, D5, D6]),
    (Tensor8, [D0, D1, D2, D3, D4, D5, D6, D7]),
}

macro_rules! impl_existing_axis_ops {
    ($(($name:ident, [$($dim:ident),+])),+ $(,)?) => {
        $(
            impl<$(const $dim: usize,)+ E: Element, P: Placement> $name<$($dim,)+ E, P> {
                /// Transposes two compile-time axes and swaps their markers.
                pub fn transpose<const A: usize, const B: usize>(
                    &self,
                ) -> Result<<Self as TransposeOutput<A, B>>::Output>
                where
                    Self: TransposeOutput<A, B>,
                {
                    let tensor = dynamic(self, "transpose")?.transpose(A as isize, B as isize)?;
                    checked_wrap::<<Self as TransposeOutput<A, B>>::Output>(
                        tensor,
                        self.binding().clone(),
                        "transpose",
                    )
                }

                /// Transposes two runtime axes and erases every dimension marker.
                pub fn transpose_dyn(
                    &self,
                    a: isize,
                    b: isize,
                ) -> Result<<Self as DynamicOutput>::Output> {
                    let tensor = dynamic(self, "transpose_dyn")?
                        .transpose(a, b)
                        .map_err(|error| relabel_error_op("transpose_dyn", error))?;
                    checked_wrap::<<Self as DynamicOutput>::Output>(
                        tensor,
                        self.binding().clone(),
                        "transpose_dyn",
                    )
                }

                /// Removes a compile-time axis whose marker is one or [`DYN`].
                pub fn squeeze<const AXIS: usize>(
                    &self,
                ) -> Result<<Self as RemoveAxisOutput<AXIS>>::Output>
                where
                    Self: RemoveAxisOutput<AXIS>,
                {
                    const {
                        assert_squeezable(<Self as SealedTypedTensor>::MARKERS[AXIS])
                    };
                    let tensor = dynamic(self, "squeeze")?.squeeze(AXIS as isize)?;
                    checked_wrap::<<Self as RemoveAxisOutput<AXIS>>::Output>(
                        tensor,
                        self.binding().clone(),
                        "squeeze",
                    )
                }

                /// Removes a runtime axis and returns the dynamic tensor escape.
                pub fn squeeze_dyn(&self, axis: isize) -> Result<Tensor> {
                    dynamic(self, "squeeze_dyn")?
                        .squeeze(axis)
                        .map_err(|error| relabel_error_op("squeeze_dyn", error))
                }

                /// Narrows a compile-time axis and erases that axis marker.
                pub fn narrow<const AXIS: usize>(
                    &self,
                    start: usize,
                    len: usize,
                ) -> Result<<Self as ReplaceAxisOutput<AXIS, DYN>>::Output>
                where
                    Self: ReplaceAxisOutput<AXIS, DYN>,
                {
                    let tensor = dynamic(self, "narrow")?.narrow(AXIS as isize, start, len)?;
                    checked_wrap::<<Self as ReplaceAxisOutput<AXIS, DYN>>::Output>(
                        tensor,
                        self.binding().clone(),
                        "narrow",
                    )
                }

                /// Narrows a runtime axis and erases every dimension marker.
                pub fn narrow_dyn(
                    &self,
                    axis: isize,
                    start: usize,
                    len: usize,
                ) -> Result<<Self as DynamicOutput>::Output> {
                    let tensor = dynamic(self, "narrow_dyn")?
                        .narrow(axis, start, len)
                        .map_err(|error| relabel_error_op("narrow_dyn", error))?;
                    checked_wrap::<<Self as DynamicOutput>::Output>(
                        tensor,
                        self.binding().clone(),
                        "narrow_dyn",
                    )
                }

                /// Concatenates homogeneous typed tensors along a compile-time axis.
                pub fn cat<const AXIS: usize>(
                    tensors: &[&Self],
                ) -> Result<<Self as ConcatOutput<AXIS>>::Output>
                where
                    Self: ConcatOutput<AXIS>,
                {
                    validate_operands(tensors, "cat")?;
                    let dynamic = tensors.iter().map(|tensor| tensor.dynamic()).collect::<Vec<_>>();
                    let tensor = Tensor::cat(&dynamic, AXIS as isize)?;
                    let binding = tensors
                        .first()
                        .expect("runtime cat accepted an empty operand list")
                        .binding()
                        .clone();
                    checked_wrap::<<Self as ConcatOutput<AXIS>>::Output>(tensor, binding, "cat")
                }
            }
        )+
    };
}

impl_existing_axis_ops! {
    (Tensor1, [D0]),
    (Tensor2, [D0, D1]),
    (Tensor3, [D0, D1, D2]),
    (Tensor4, [D0, D1, D2, D3]),
    (Tensor5, [D0, D1, D2, D3, D4]),
    (Tensor6, [D0, D1, D2, D3, D4, D5]),
    (Tensor7, [D0, D1, D2, D3, D4, D5, D6]),
    (Tensor8, [D0, D1, D2, D3, D4, D5, D6, D7]),
}

macro_rules! impl_rank_increasing_ops {
    ($(($name:ident, [$($dim:ident),*])),+ $(,)?) => {
        $(
            impl<$(const $dim: usize,)* E: Element, P: Placement> $name<$($dim,)* E, P> {
                /// Inserts a size-one axis at a compile-time position.
                pub fn unsqueeze<const AXIS: usize>(
                    &self,
                ) -> Result<<Self as InsertAxisOutput<AXIS, 1>>::Output>
                where
                    Self: InsertAxisOutput<AXIS, 1>,
                {
                    let tensor = dynamic(self, "unsqueeze")?.unsqueeze(AXIS as isize)?;
                    checked_wrap::<<Self as InsertAxisOutput<AXIS, 1>>::Output>(
                        tensor,
                        self.binding().clone(),
                        "unsqueeze",
                    )
                }

                /// Inserts a runtime axis and returns the dynamic tensor escape.
                pub fn unsqueeze_dyn(&self, axis: isize) -> Result<Tensor> {
                    dynamic(self, "unsqueeze_dyn")?
                        .unsqueeze(axis)
                        .map_err(|error| relabel_error_op("unsqueeze_dyn", error))
                }

                /// Stacks homogeneous typed tensors at a compile-time axis.
                pub fn stack<const AXIS: usize>(
                    tensors: &[&Self],
                ) -> Result<<Self as StackOutput<AXIS>>::Output>
                where
                    Self: StackOutput<AXIS>,
                {
                    validate_operands(tensors, "stack")?;
                    let dynamic = tensors.iter().map(|tensor| tensor.dynamic()).collect::<Vec<_>>();
                    let tensor = Tensor::stack(&dynamic, AXIS as isize)?;
                    let binding = tensors
                        .first()
                        .expect("runtime stack accepted an empty operand list")
                        .binding()
                        .clone();
                    checked_wrap::<<Self as StackOutput<AXIS>>::Output>(tensor, binding, "stack")
                }
            }
        )+
    };
}

impl_rank_increasing_ops! {
    (Tensor0, []),
    (Tensor1, [D0]),
    (Tensor2, [D0, D1]),
    (Tensor3, [D0, D1, D2]),
    (Tensor4, [D0, D1, D2, D3]),
    (Tensor5, [D0, D1, D2, D3, D4]),
    (Tensor6, [D0, D1, D2, D3, D4, D5]),
    (Tensor7, [D0, D1, D2, D3, D4, D5, D6]),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed::{Cpu, DeviceBinding, DeviceCtx};
    use crate::{Device, Error, testing::check_grad};

    fn ctx() -> DeviceCtx<Cpu> {
        DeviceCtx::cpu().unwrap()
    }

    fn values(tensor: &Tensor) -> Vec<f32> {
        tensor.to_vec::<f32>().unwrap()
    }

    #[test]
    fn compile_probes_pin_stable_output_mappings_and_rank_ceiling() {
        fn tensor2(_: Tensor2<3, 2>) {}
        fn tensor3(_: Tensor3<2, 1, 3>) {}
        fn tensor8(_: Tensor8<1, 2, 3, 4, 5, 6, 7, 1>) {}

        let ctx = ctx();
        let matrix =
            Tensor2::<2, 3>::from_vec((0..6).map(|x| x as f32).collect(), [2, 3], &ctx).unwrap();
        tensor2(matrix.transpose::<0, 1>().unwrap());
        tensor3(matrix.unsqueeze::<1>().unwrap());

        let rank7 =
            Tensor7::<1, 2, 3, 4, 5, 6, 7>::from_vec(vec![0.0; 5040], [1, 2, 3, 4, 5, 6, 7], &ctx)
                .unwrap();
        tensor8(rank7.unsqueeze::<7>().unwrap());
    }

    #[test]
    fn typed_shape_ops_match_runtime_values_and_layouts() {
        let ctx = ctx();
        let source =
            Tensor3::<2, 1, 3>::from_vec((0..6).map(|x| x as f32).collect(), [2, 1, 3], &ctx)
                .unwrap();

        let reshaped = source.reshape::<Tensor2<3, 2>>([3, 2]).unwrap();
        assert_eq!(reshaped.dims(), [3, 2]);
        assert_eq!(values(reshaped.as_dynamic()), vec![0., 1., 2., 3., 4., 5.]);

        let squeezed = source.squeeze::<1>().unwrap();
        assert_eq!(squeezed.dims(), [2, 3]);
        let unsqueezed = squeezed.unsqueeze::<1>().unwrap();
        assert_eq!(unsqueezed.dims(), source.dims());

        let transposed = source.transpose::<0, 2>().unwrap();
        let runtime = source.as_dynamic().transpose(0, 2).unwrap();
        assert_eq!(values(transposed.as_dynamic()), values(&runtime));
        assert_eq!(
            transposed.as_dynamic().is_contiguous(),
            runtime.is_contiguous()
        );

        let narrowed = source.narrow::<2>(1, 2).unwrap();
        assert_eq!(narrowed.dims(), [2, 1, 2]);
        assert_eq!(values(narrowed.as_dynamic()), vec![1., 2., 4., 5.]);

        let permuted = source.permute(&[2, 0, 1]).unwrap();
        assert_eq!(permuted.dims(), [3, 2, 1]);
        assert_eq!(
            values(permuted.as_dynamic()),
            values(&source.as_dynamic().permute(&[2, 0, 1]).unwrap())
        );
    }

    #[test]
    fn broadcast_cat_and_stack_match_runtime_operations() {
        let ctx = ctx();
        let row = Tensor2::<1, 3>::from_vec(vec![1., 2., 3.], [1, 3], &ctx).unwrap();
        let broadcast = row.broadcast_to::<Tensor2<2, 3>>([2, 3]).unwrap();
        assert_eq!(values(broadcast.as_dynamic()), vec![1., 2., 3., 1., 2., 3.]);
        assert!(!broadcast.as_dynamic().is_contiguous());

        let a = Tensor2::<DYN, 2>::from_vec(vec![1., 2.], [1, 2], &ctx).unwrap();
        let b = Tensor2::<DYN, 2>::from_vec(vec![3., 4., 5., 6.], [2, 2], &ctx).unwrap();
        let cat = Tensor2::cat::<0>(&[&a, &b]).unwrap();
        assert_eq!(cat.dims(), [3, 2]);
        assert_eq!(values(cat.as_dynamic()), vec![1., 2., 3., 4., 5., 6.]);

        let c = Tensor2::<2, 2>::from_vec(vec![1., 2., 3., 4.], [2, 2], &ctx).unwrap();
        let d = Tensor2::<2, 2>::from_vec(vec![5., 6., 7., 8.], [2, 2], &ctx).unwrap();
        let stack = Tensor2::stack::<1>(&[&c, &d]).unwrap();
        let runtime = Tensor::stack(&[c.as_dynamic(), d.as_dynamic()], 1).unwrap();
        assert_eq!(stack.dims(), [2, 2, 2]);
        assert_eq!(values(stack.as_dynamic()), values(&runtime));
    }

    #[test]
    fn dyn_markers_defer_relational_failures_to_runtime() {
        let ctx = ctx();
        let reshape = Tensor2::<DYN, 2>::from_vec(vec![0.; 6], [3, 2], &ctx).unwrap();
        assert!(matches!(
            reshape.reshape::<Tensor2<DYN, 4>>([2, 4]),
            Err(Error::ReshapeMismatch { op: "reshape", .. })
        ));

        let squeeze = Tensor2::<DYN, 3>::from_vec(vec![0.; 6], [2, 3], &ctx).unwrap();
        assert!(matches!(
            squeeze.squeeze::<0>(),
            Err(Error::InvalidArg { op: "squeeze", .. })
        ));

        let broadcast = Tensor2::<DYN, 3>::from_vec(vec![0.; 6], [2, 3], &ctx).unwrap();
        assert!(matches!(
            broadcast.broadcast_to::<Tensor2<DYN, 3>>([4, 3]),
            Err(Error::ShapeMismatch {
                op: "broadcast_to",
                ..
            })
        ));
    }

    #[test]
    fn shape_ops_reject_noncanonical_bindings_before_delegation() {
        let ctx = ctx();
        let dynamic = Tensor::from_vec(vec![1.0f32, 2.0], [2], &Device::Cpu).unwrap();
        let forged = Arc::new(DeviceBinding {
            device: Device::Cpu,
        });
        let forged = <Tensor1<2> as SealedTypedTensor>::trusted_from_validated(dynamic, forged);
        let canonical = Tensor1::<2>::from_vec(vec![3.0, 4.0], [2], &ctx).unwrap();

        assert!(matches!(
            forged.unsqueeze::<0>(),
            Err(Error::InvalidArg {
                op: "unsqueeze",
                ..
            })
        ));
        assert!(matches!(
            Tensor1::cat::<0>(&[&canonical, &forged]),
            Err(Error::InvalidArg { op: "cat", .. })
        ));
    }

    #[test]
    fn runtime_axis_escapes_relabel_errors_and_erase_as_documented() {
        let ctx = ctx();
        let source =
            Tensor2::<2, 3>::from_vec((0..6).map(|x| x as f32).collect(), [2, 3], &ctx).unwrap();
        let transposed = source.transpose_dyn(-1, 0).unwrap();
        assert_eq!(transposed.dims(), [3, 2]);
        let narrowed = source.narrow_dyn(-1, 1, 1).unwrap();
        assert_eq!(narrowed.dims(), [2, 1]);
        assert_eq!(source.unsqueeze_dyn(-1).unwrap().dims(), &[2, 3, 1]);
        assert!(matches!(
            source.squeeze_dyn(0),
            Err(Error::InvalidArg {
                op: "squeeze_dyn",
                ..
            })
        ));
        assert!(matches!(
            source.transpose_dyn(0, 2),
            Err(Error::InvalidAxis {
                op: "transpose_dyn",
                ..
            })
        ));
        assert!(matches!(
            source.narrow_dyn(2, 0, 1),
            Err(Error::InvalidAxis {
                op: "narrow_dyn",
                ..
            })
        ));
        assert!(matches!(
            source.unsqueeze_dyn(3),
            Err(Error::InvalidAxis {
                op: "unsqueeze_dyn",
                ..
            })
        ));
    }

    #[test]
    fn typed_view_chain_and_multi_input_ops_preserve_runtime_gradients() {
        let ctx = ctx();
        check_grad(
            |inputs| {
                let typed = Tensor3::<2, 3, 2, f64>::try_from_dynamic(inputs[0].clone(), &ctx)?;
                typed
                    .reshape::<Tensor3<2, 2, 3, f64>>([2, 2, 3])?
                    .transpose::<0, 2>()?
                    .narrow::<1>(0, 1)?
                    .into_dynamic()
                    .sum_all()
            },
            &[Tensor::from_vec(
                (0..12).map(|x| x as f64).collect(),
                [2, 3, 2],
                &ctx.device(),
            )
            .unwrap()],
            1e-3,
            1e-4,
        )
        .unwrap();

        check_grad(
            |inputs| {
                let a = Tensor1::<DYN, f64>::try_from_dynamic(inputs[0].clone(), &ctx)?;
                let b = Tensor1::<DYN, f64>::try_from_dynamic(inputs[1].clone(), &ctx)?;
                Tensor1::cat::<0>(&[&a, &b])?.into_dynamic().sum_all()
            },
            &[
                Tensor::from_vec(vec![1., 2.], [2], &ctx.device()).unwrap(),
                Tensor::from_vec(vec![3., 4., 5.], [3], &ctx.device()).unwrap(),
            ],
            1e-3,
            1e-4,
        )
        .unwrap();
    }
}
