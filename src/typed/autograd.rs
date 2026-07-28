use super::ops::{WithElement, WithPlacement};
use super::sealed::TypedTensor as SealedTypedTensor;
use super::tensor::checked_wrap;
use super::{DeviceCtx, FloatElement, Placement, TypedTensor, typed_rank_table};
use crate::{Element, Grads, Result};
use std::sync::Arc;

macro_rules! impl_typed_core {
    ($(($name:ident, $rank:literal, [$($dim:ident),*])),+ $(,)?) => {
        $(
            impl<$(const $dim: usize,)* E: Element, P: Placement>
                super::$name<$($dim,)* E, P>
            {
                /// Returns an untraced tensor sharing this tensor's storage and layout.
                pub fn detach(&self) -> Result<Self> {
                    checked_wrap(
                        SealedTypedTensor::dynamic(self).detach(),
                        Arc::clone(SealedTypedTensor::binding(self)),
                        "detach",
                    )
                }

                /// Returns a row-major contiguous tensor with unchanged typed metadata.
                pub fn contiguous(&self) -> Result<Self> {
                    checked_wrap(
                        SealedTypedTensor::dynamic(self).contiguous()?,
                        Arc::clone(SealedTypedTensor::binding(self)),
                        "contiguous",
                    )
                }

                /// Casts the element type while preserving shape and placement.
                pub fn to_dtype<F: Element>(&self) -> Result<<Self as WithElement<F>>::Output>
                where
                    Self: WithElement<F>,
                {
                    checked_wrap(
                        SealedTypedTensor::dynamic(self).to_dtype(F::DTYPE)?,
                        Arc::clone(SealedTypedTensor::binding(self)),
                        "to_dtype",
                    )
                }

                /// Moves the tensor while preserving shape and element type.
                pub fn to_device<Q: Placement>(
                    &self,
                    target: &DeviceCtx<Q>,
                ) -> Result<<Self as WithPlacement<Q>>::Output>
                where
                    Self: WithPlacement<Q>,
                {
                    checked_wrap(
                        SealedTypedTensor::dynamic(self).to_device(&target.device())?,
                        Arc::clone(target.binding()),
                        "to_device",
                    )
                }

                /// Copies the elements to host memory in row-major order.
                pub fn to_vec(&self) -> Result<Vec<E>> {
                    SealedTypedTensor::dynamic(self).to_vec::<E>()
                }

                /// Reads the sole element using this tensor's element type.
                pub fn to_scalar(&self) -> Result<E> {
                    SealedTypedTensor::dynamic(self).to_scalar::<E>()
                }

                /// Reads the sole element as `f64` regardless of element type.
                pub fn item(&self) -> Result<f64> {
                    SealedTypedTensor::dynamic(self).item()
                }
            }

            impl<$(const $dim: usize,)* E: FloatElement, P: Placement>
                super::$name<$($dim,)* E, P>
            {
                /// Returns a traced leaf for gradient lookup by typed input.
                pub fn traced(&self) -> Result<Self> {
                    checked_wrap(
                        SealedTypedTensor::dynamic(self).traced()?,
                        Arc::clone(SealedTypedTensor::binding(self)),
                        "traced",
                    )
                }

                /// Runs reverse-mode autodiff from this tensor.
                pub fn backward(&self) -> Result<Grads> {
                    SealedTypedTensor::dynamic(self).backward()
                }
            }
        )+
    };
}

typed_rank_table!(impl_typed_core);

/// Typed gradient lookup for a leaf returned by a typed `traced` call.
pub trait TypedGradsExt {
    /// Returns the gradient with the input's exact shape, element, and placement.
    fn wrt_typed_input<T: TypedTensor>(&self, input: &T) -> Result<T>;
}

impl TypedGradsExt for Grads {
    fn wrt_typed_input<T: TypedTensor>(&self, input: &T) -> Result<T> {
        checked_wrap(
            self.wrt_input(SealedTypedTensor::dynamic(input))?,
            Arc::clone(SealedTypedTensor::binding(input)),
            "wrt_typed_input",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed::{Cpu, Tensor0, Tensor1, Tensor2};
    use crate::{DType, Device, Error, Tensor};

    struct OtherCpu;
    impl Placement for OtherCpu {}

    fn cpu() -> DeviceCtx<Cpu> {
        DeviceCtx::cpu().unwrap()
    }

    #[test]
    fn host_reads_and_casts_match_the_runtime_api() {
        let ctx = cpu();
        let typed = Tensor2::<2, 2>::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2], &ctx).unwrap();
        let dynamic = typed.as_dynamic();

        assert_eq!(typed.to_vec().unwrap(), dynamic.to_vec::<f32>().unwrap());
        let typed_i64: Tensor2<2, 2, i64> = typed.to_dtype().unwrap();
        let dynamic_i64 = dynamic.to_dtype(DType::I64).unwrap();
        assert_eq!(
            typed_i64.to_vec().unwrap(),
            dynamic_i64.to_vec::<i64>().unwrap()
        );

        let scalar = Tensor0::<i64>::from_vec(vec![7], [], &ctx).unwrap();
        assert_eq!(scalar.to_scalar().unwrap(), 7);
        assert_eq!(scalar.item().unwrap(), 7.0);
    }

    #[test]
    fn identity_paths_preserve_runtime_graph_and_binding_identity() {
        let ctx = cpu();
        let other = DeviceCtx::<OtherCpu>::bind(Device::Cpu).unwrap();
        let traced = Tensor1::<2>::from_vec(vec![1.0, 2.0], [2], &ctx)
            .unwrap()
            .traced()
            .unwrap();

        let contiguous = traced.contiguous().unwrap();
        let same_dtype: Tensor1<2> = traced.to_dtype().unwrap();
        let moved: Tensor1<2, f32, OtherCpu> = traced.to_device(&other).unwrap();
        assert!(traced.as_dynamic().ptr_eq(contiguous.as_dynamic()));
        assert!(traced.as_dynamic().ptr_eq(same_dtype.as_dynamic()));
        assert!(traced.as_dynamic().ptr_eq(moved.as_dynamic()));
        assert!(Arc::ptr_eq(
            SealedTypedTensor::binding(&moved),
            other.binding()
        ));
    }

    #[test]
    fn typed_errors_are_the_runtime_errors() {
        let ctx = cpu();
        let typed = Tensor1::<2>::from_vec(vec![1.0, 2.0], [2], &ctx).unwrap();
        assert_eq!(
            format!("{}", typed.to_scalar().unwrap_err()),
            format!("{}", typed.as_dynamic().to_scalar::<f32>().unwrap_err())
        );
        assert!(matches!(
            typed.backward(),
            Err(Error::NotTraced { op: "backward" })
        ));

        let traced = typed.traced().unwrap();
        assert!(matches!(
            traced.traced(),
            Err(Error::InvalidArg { op: "traced", .. })
        ));
    }

    #[test]
    fn detach_removes_graph_while_cast_and_typed_lookup_preserve_it() {
        let ctx = cpu();
        let input = Tensor1::<3>::from_vec(vec![1.0, 2.0, 3.0], [3], &ctx)
            .unwrap()
            .traced()
            .unwrap();
        let detached = input.detach().unwrap();
        assert!(matches!(
            detached.backward(),
            Err(Error::NotTraced { op: "backward" })
        ));

        let narrow: Tensor1<3, half::f16> = input.to_dtype().unwrap();
        let output = narrow
            .as_dynamic()
            .mul(narrow.as_dynamic())
            .unwrap()
            .sum_all()
            .unwrap();
        let output = Tensor0::<half::f16>::try_from_dynamic(output, &ctx).unwrap();
        let grads = output.backward().unwrap();
        let grad: Tensor1<3> = grads.wrt_typed_input(&input).unwrap();
        assert_eq!(grad.to_vec().unwrap(), vec![2.0, 4.0, 6.0]);
        assert_eq!(grads.len(), 1);
    }

    #[test]
    fn typed_gradient_lookup_matches_dynamic_lookup_errors() {
        let ctx = cpu();
        let used = Tensor1::<1>::from_vec(vec![2.0], [1], &ctx)
            .unwrap()
            .traced()
            .unwrap();
        let unused = Tensor1::<1>::from_vec(vec![3.0], [1], &ctx)
            .unwrap()
            .traced()
            .unwrap();
        let output = used.as_dynamic().sum_all().unwrap();
        let grads = Tensor0::<f32>::try_from_dynamic(output, &ctx)
            .unwrap()
            .backward()
            .unwrap();

        let runtime = grads.wrt_input(unused.as_dynamic()).unwrap_err();
        let typed = grads.wrt_typed_input(&unused).unwrap_err();
        assert_eq!(format!("{typed}"), format!("{runtime}"));

        let plain = Tensor::from_vec(vec![1.0f32], [1], &Device::Cpu).unwrap();
        let plain = Tensor1::<1>::try_from_dynamic(plain, &ctx).unwrap();
        assert!(matches!(
            grads.wrt_typed_input(&plain),
            Err(Error::NotTraced { op: "wrt_input" })
        ));
    }
}
