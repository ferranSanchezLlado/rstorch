use super::{Forward, Mode, Module, ToDType, ToDevice, TypedVisitor, TypedVisitorMut};
use crate::Result;
use crate::nn::Forward as RuntimeForward;
use crate::typed::tensor::checked_wrap;
use crate::typed::{FloatElement, Placement, TypedTensor};
use std::sync::Arc;

/// A shape-preserving typed rectified linear unit.
///
/// Integer and boolean tensors do not satisfy this layer's floating-point
/// capability bound:
///
/// ```compile_fail
/// use rstorch::typed::{DeviceCtx, Tensor1, nn::{Forward, Mode, Relu}};
/// let ctx = DeviceCtx::cpu().unwrap();
/// let x = Tensor1::<2, i64>::from_vec(vec![1, -1], [2], &ctx).unwrap();
/// let _ = Relu.forward(&x, Mode::EVAL);
/// ```
pub struct Relu;

/// A shape-preserving typed exact Gaussian error linear unit.
pub struct Gelu;

impl<T> Forward<T> for Relu
where
    T: TypedTensor,
    T::Elem: FloatElement,
{
    type Output = T;

    fn forward(&mut self, input: &T, mode: Mode) -> Result<T> {
        let output = RuntimeForward::forward(&mut crate::nn::Relu, input.dynamic(), mode)?;
        checked_wrap(
            output,
            Arc::clone(input.binding()),
            "typed::nn::Relu::forward",
        )
    }
}

impl<T> Forward<T> for Gelu
where
    T: TypedTensor,
    T::Elem: FloatElement,
{
    type Output = T;

    fn forward(&mut self, input: &T, mode: Mode) -> Result<T> {
        let output = RuntimeForward::forward(&mut crate::nn::Gelu, input.dynamic(), mode)?;
        checked_wrap(
            output,
            Arc::clone(input.binding()),
            "typed::nn::Gelu::forward",
        )
    }
}

impl Module for Relu {
    fn visit(&self, _visitor: &mut TypedVisitor<'_>) {}
    fn visit_mut(&mut self, _visitor: &mut TypedVisitorMut<'_>) {}
}

impl Module for Gelu {
    fn visit(&self, _visitor: &mut TypedVisitor<'_>) {}
    fn visit_mut(&mut self, _visitor: &mut TypedVisitorMut<'_>) {}
}

macro_rules! impl_stateless_layer {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl Clone for $ty {
                fn clone(&self) -> Self { *self }
            }

            impl Copy for $ty {}

            impl Default for $ty {
                fn default() -> Self { Self }
            }

            impl std::fmt::Debug for $ty {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    f.write_str(stringify!($ty))
                }
            }

            impl PartialEq for $ty {
                fn eq(&self, _other: &Self) -> bool { true }
            }

            impl Eq for $ty {}

            impl<Q: Placement> ToDevice<Q> for $ty {
                type Output = Self;

                fn to_device(self, _target: &crate::typed::DeviceCtx<Q>) -> Result<Self> {
                    Ok(self)
                }
            }

            impl<F: FloatElement> ToDType<F> for $ty {
                type Output = Self;

                fn to_dtype(self) -> Result<Self> {
                    Ok(self)
                }
            }
        )+
    };
}

impl_stateless_layer!(Relu, Gelu);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed::{DeviceCtx, Tensor2};

    #[test]
    fn values_shapes_and_state_match_runtime_layers() {
        let ctx = DeviceCtx::cpu().unwrap();
        let input =
            Tensor2::<2, 3>::from_vec(vec![-2.0f32, -0.5, 0.0, 0.5, 1.0, 2.0], [2, 3], &ctx)
                .unwrap();

        for mode in [
            Mode::TRAIN,
            Mode::EVAL,
            Mode::TRAIN.frozen(),
            Mode::EVAL.recorded(),
        ] {
            let relu = Relu.forward(&input, mode).unwrap();
            let gelu = Gelu.forward(&input, mode).unwrap();
            assert_eq!(relu.dims(), [2, 3]);
            assert_eq!(
                relu.to_vec().unwrap(),
                input.as_dynamic().relu().unwrap().to_vec::<f32>().unwrap()
            );
            assert_eq!(
                gelu.to_vec().unwrap(),
                input.as_dynamic().gelu().unwrap().to_vec::<f32>().unwrap()
            );
        }
        assert!(super::super::state_dict(&Relu).unwrap().is_empty());
        assert!(super::super::state_dict(&Gelu).unwrap().is_empty());
    }

    #[test]
    fn gradients_are_runtime_gradients() {
        let ctx = DeviceCtx::cpu().unwrap();
        let input = Tensor2::<1, 3>::from_vec(vec![-1.0f32, 0.5, 2.0], [1, 3], &ctx)
            .unwrap()
            .traced()
            .unwrap();
        let output = Gelu.forward(&input, Mode::TRAIN).unwrap();
        let grads = output.as_dynamic().sum_all().unwrap().backward().unwrap();
        assert_eq!(grads.wrt_input(input.as_dynamic()).unwrap().dims(), &[1, 3]);
    }
}
