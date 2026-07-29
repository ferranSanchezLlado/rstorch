use super::const_check::assert_refinement;
use super::device::{checked_relabel_binding, validate_binding};
use super::ops::{DynamicOutput, RefinementOf};
use super::sealed::TypedTensor as SealedTypedTensor;
use super::{DYN, DeviceBinding, DeviceCtx, Placement, TypedTensor, typed_rank_table};
use crate::{Element, Error, Result, Shape, Tensor};
use std::sync::Arc;

fn validate<T: TypedTensor>(
    tensor: &Tensor,
    binding: &Arc<DeviceBinding>,
    op: &'static str,
) -> Result<()> {
    if tensor.rank() != T::RANK {
        return Err(Error::RankMismatch {
            op,
            expected: T::RANK,
            got: tensor.rank(),
        });
    }

    for (axis, (&marker, &actual)) in <T as SealedTypedTensor>::MARKERS
        .iter()
        .zip(tensor.dims())
        .enumerate()
    {
        if marker != DYN && marker != actual {
            let mut expected = tensor.dims().to_vec();
            expected[axis] = marker;
            return Err(Error::ShapeMismatch {
                op,
                lhs: tensor.shape().clone(),
                rhs: Shape::from(expected),
            });
        }
    }

    if tensor.dtype() != T::Elem::DTYPE {
        return Err(Error::DTypeMismatch {
            op,
            expected: T::Elem::DTYPE,
            got: tensor.dtype(),
        });
    }
    if tensor.device() != binding.device {
        return Err(Error::DeviceMismatch {
            op,
            expected: binding.device,
            got: tensor.device(),
        });
    }
    validate_binding::<T::Placement>(binding, op)
}

pub(in crate::typed) fn checked_wrap<T: TypedTensor>(
    tensor: Tensor,
    binding: Arc<DeviceBinding>,
    op: &'static str,
) -> Result<T> {
    validate::<T>(&tensor, &binding, op)?;
    Ok(<T as SealedTypedTensor>::trusted_from_validated(
        tensor, binding,
    ))
}

macro_rules! impl_tensor_boundary {
    ($(($name:ident, $rank:literal, [$($dim:ident),*])),+ $(,)?) => {
        $(
            impl<$(const $dim: usize,)* E: Element, P: Placement>
                super::$name<$($dim,)* E, P>
            {
                /// Constructs a typed tensor from row-major host data.
                pub fn from_vec(
                    data: Vec<E>,
                    dims: [usize; $rank],
                    ctx: &DeviceCtx<P>,
                ) -> Result<Self> {
                    validate_binding::<P>(ctx.binding(), "from_vec")?;
                    for (&marker, &actual) in
                        <Self as SealedTypedTensor>::MARKERS.iter().zip(&dims)
                    {
                        if marker != DYN && marker != actual {
                            return Err(Error::ShapeMismatch {
                                op: "from_vec",
                                lhs: Shape::from(dims.to_vec()),
                                rhs: Shape::from(
                                    <Self as SealedTypedTensor>::MARKERS
                                        .iter()
                                        .zip(&dims)
                                        .map(|(&m, &d)| if m == DYN { d } else { m })
                                        .collect::<Vec<_>>(),
                                ),
                            });
                        }
                    }
                    let tensor = Tensor::from_vec(data, dims, &ctx.device())?;
                    checked_wrap::<Self>(
                        tensor,
                        Arc::clone(ctx.binding()),
                        "from_vec",
                    )
                }

                /// Checks and wraps an existing runtime tensor without copying it.
                pub fn try_from_dynamic(tensor: Tensor, ctx: &DeviceCtx<P>) -> Result<Self> {
                    checked_wrap::<Self>(
                        tensor,
                        Arc::clone(ctx.binding()),
                        "try_from_dynamic",
                    )
                }

                /// Borrows the unchanged runtime tensor.
                pub fn as_dynamic(&self) -> &Tensor {
                    <Self as SealedTypedTensor>::dynamic(self)
                }

                /// Removes compile-time metadata without copying the runtime tensor.
                pub fn into_dynamic(self) -> Tensor {
                    self.inner
                }

                /// Returns the tensor's actual runtime dimensions.
                pub fn dims(&self) -> [usize; $rank] {
                    let mut dims = [0; $rank];
                    dims.copy_from_slice(self.inner.dims());
                    dims
                }

                /// Checks this tensor against a more precise shape of the same rank.
                pub fn refine<Target>(self) -> Result<Target>
                where
                    Target: RefinementOf<Self>,
                {
                    const {
                        assert_refinement(
                            <Self as SealedTypedTensor>::MARKERS,
                            <Target as SealedTypedTensor>::MARKERS,
                        )
                    };
                    checked_wrap::<Target>(self.inner, self.binding, "refine")
                }

                /// Changes only the logical placement marker without moving or copying data.
                ///
                /// Both placement bindings must be canonical and identify the same physical
                /// device. The runtime tensor, storage, layout, and autograd identity are
                /// unchanged.
                pub fn relabel<Q: Placement>(
                    self,
                    target: &DeviceCtx<Q>,
                ) -> Result<super::$name<$($dim,)* E, Q>> {
                    let binding = checked_relabel_binding::<P, Q>(self.binding, target)?;
                    checked_wrap::<super::$name<$($dim,)* E, Q>>(
                        self.inner,
                        binding,
                        "relabel",
                    )
                }

                /// Replaces every static dimension marker with [`DYN`].
                pub fn erase_shape(self) -> Result<<Self as DynamicOutput>::Output> {
                    checked_wrap::<super::$name<$({ impl_tensor_boundary!(@dyn $dim) },)* E, P>>(
                        self.inner,
                        self.binding,
                        "erase_shape",
                    )
                }
            }
        )+
    };
    (@dyn $dim:ident) => { DYN };
}

typed_rank_table!(impl_tensor_boundary);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed::{Cpu, Tensor2};

    #[test]
    fn boundary_transitions_keep_the_runtime_tensor_arc() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let dynamic = Tensor::from_vec(vec![1.0f32, 2.0], [1, 2], &ctx.device())
            .unwrap()
            .traced()
            .unwrap();
        let original = dynamic.clone();
        let typed = Tensor2::<DYN, 2>::try_from_dynamic(dynamic, &ctx).unwrap();
        assert!(original.ptr_eq(typed.as_dynamic()));
        assert!(Arc::ptr_eq(
            <Tensor2<DYN, 2> as SealedTypedTensor>::binding(&typed),
            ctx.binding()
        ));

        let refined = typed.refine::<Tensor2<1, 2>>().unwrap();
        assert!(original.ptr_eq(refined.as_dynamic()));
        let erased = refined.erase_shape().unwrap();
        assert!(original.ptr_eq(erased.as_dynamic()));
        assert!(original.ptr_eq(&erased.into_dynamic()));
    }

    #[test]
    fn trusted_construction_rejects_noncanonical_binding_identity() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let dynamic = Tensor::zeros([1, 2], f32::DTYPE, &ctx.device()).unwrap();
        let forged = Arc::new(DeviceBinding {
            device: ctx.device(),
        });

        assert!(matches!(
            checked_wrap::<Tensor2<1, 2>>(dynamic, forged, "trusted_test"),
            Err(Error::InvalidArg {
                op: "trusted_test",
                ..
            })
        ));
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn wrong_runtime_device_is_rejected_without_metal_storage() {
        use crate::typed::{Metal, Tensor1};

        let dynamic = Tensor::zeros([1], f32::DTYPE, &crate::Device::Cpu).unwrap();
        let forged_metal_binding = Arc::new(DeviceBinding {
            device: crate::Device::Metal(0),
        });

        assert!(matches!(
            checked_wrap::<Tensor1<1, f32, Metal<0>>>(
                dynamic,
                forged_metal_binding,
                "wrong_device_test"
            ),
            Err(Error::DeviceMismatch {
                op: "wrong_device_test",
                expected: crate::Device::Metal(0),
                got: crate::Device::Cpu,
            })
        ));
    }
}
