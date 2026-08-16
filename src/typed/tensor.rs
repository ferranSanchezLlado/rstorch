use super::const_check::assert_refinement;
use super::device::{checked_relabel_binding, validate_binding};
use super::ops::{DynamicOutput, RefinementOf};
use super::sealed::TypedTensor as SealedTypedTensor;
use super::{DYN, DeviceBinding, DeviceCtx, Placement, TypedTensor, typed_rank_table};
use crate::{Element, Error, Result, Shape, Tensor};
use std::sync::Arc;

/// Builds the shape a typed target actually requires, for a `ShapeMismatch`
/// payload.
///
/// Every static marker is substituted and every `DYN` marker keeps the observed
/// dimension, so the reported shape is one the target would accept. Reporting
/// only the *first* contradicting axis — leaving the others at their observed
/// values — names a shape the target also rejects, which sends a reader looking
/// in the wrong place when two or more axes disagree.
fn required_shape(markers: &[usize], actual: &[usize]) -> Shape {
    Shape::from(
        markers
            .iter()
            .zip(actual)
            .map(|(&marker, &observed)| if marker == DYN { observed } else { marker })
            .collect::<Vec<_>>(),
    )
}

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

    for (&marker, &actual) in <T as SealedTypedTensor>::MARKERS.iter().zip(tensor.dims()) {
        if marker != DYN && marker != actual {
            return Err(Error::ShapeMismatch {
                op,
                lhs: tensor.shape().clone(),
                rhs: required_shape(<T as SealedTypedTensor>::MARKERS, tensor.dims()),
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
                ///
                /// # Examples
                ///
                /// ```
                /// use rstorch::typed::{DeviceCtx, Tensor2};
                ///
                /// # fn main() -> rstorch::Result<()> {
                /// let ctx = DeviceCtx::cpu()?;
                /// let x = Tensor2::<2, 2>::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2], &ctx)?;
                /// assert_eq!(x.dims(), [2, 2]);
                /// assert_eq!(x.to_vec()?, vec![1.0, 2.0, 3.0, 4.0]);
                /// # Ok(())
                /// # }
                /// ```
                ///
                /// # Errors
                ///
                /// [`Error::InvalidArg`] if `ctx`'s binding is not the
                /// canonical one for `P`; [`Error::ShapeMismatch`] if `dims`
                /// contradicts a static marker; otherwise propagates
                /// [`Tensor::from_vec`]'s own length/dtype error.
                pub fn from_vec(
                    data: Vec<E>,
                    dims: [usize; $rank],
                    ctx: &DeviceCtx<P>,
                ) -> Result<Self> {
                    validate_binding::<P>(ctx.binding(), "from_vec")?;
                    // Deliberately checked before `Tensor::from_vec` allocates,
                    // and reported against the requested `dims` — the runtime's
                    // own error would describe `lhs` as the flat data length.
                    for (&marker, &actual) in
                        <Self as SealedTypedTensor>::MARKERS.iter().zip(&dims)
                    {
                        if marker != DYN && marker != actual {
                            return Err(Error::ShapeMismatch {
                                op: "from_vec",
                                lhs: Shape::from(dims.to_vec()),
                                rhs: required_shape(
                                    <Self as SealedTypedTensor>::MARKERS,
                                    &dims,
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
                ///
                /// # Errors
                ///
                /// [`Error::RankMismatch`]/[`Error::ShapeMismatch`] if
                /// `tensor`'s rank or dimensions contradict `Self`;
                /// [`Error::DTypeMismatch`]/[`Error::DeviceMismatch`] if its
                /// dtype or device does not match; [`Error::InvalidArg`] if
                /// `ctx`'s binding is not the canonical one for `P`.
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
                ///
                /// # Errors
                ///
                /// As [`try_from_dynamic`](Self::try_from_dynamic); in
                /// practice only [`Error::ShapeMismatch`] is reachable, since
                /// `Target`'s markers are statically checked to refine
                /// `Self`'s.
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
                ///
                /// # Errors
                ///
                /// [`Error::InvalidArg`] if `self`'s or `target`'s binding is
                /// not canonical for `P`/`Q`; [`Error::DeviceMismatch`] if
                /// they name different physical devices.
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
                ///
                /// # Errors
                ///
                /// As [`try_from_dynamic`](Self::try_from_dynamic);
                /// unreachable in practice, since erasing markers to `DYN`
                /// only widens what the target accepts.
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
    use crate::typed::Tensor2;

    #[test]
    fn boundary_transitions_keep_the_runtime_tensor_arc() {
        let ctx = DeviceCtx::cpu().unwrap();
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
        let ctx = DeviceCtx::cpu().unwrap();
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

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn wrong_runtime_device_is_rejected_without_cuda_storage() {
        use crate::typed::{Cuda, Tensor1};

        let dynamic = Tensor::zeros([1], f32::DTYPE, &crate::Device::Cpu).unwrap();
        let forged_cuda_binding = Arc::new(DeviceBinding {
            device: crate::Device::Cuda(0),
        });

        assert!(matches!(
            checked_wrap::<Tensor1<1, f32, Cuda<0>>>(
                dynamic,
                forged_cuda_binding,
                "wrong_device_test"
            ),
            Err(Error::DeviceMismatch {
                op: "wrong_device_test",
                expected: crate::Device::Cuda(0),
                got: crate::Device::Cpu,
            })
        ));
    }
}
