use super::MatmulOutput;
use crate::typed::const_check::assert_matmul_contract;
use crate::typed::device::validate_binding;
use crate::typed::sealed::TypedTensor as SealedTypedTensor;
use crate::typed::tensor::checked_wrap;
use crate::typed::{
    NumericElement, Placement, Tensor2, Tensor3, Tensor4, Tensor5, Tensor6, Tensor7, Tensor8,
    TypedTensor,
};
use crate::{Error, Result};
use std::sync::Arc;

macro_rules! impl_matmul {
    ($name:ident, [$($dim:ident),+]) => {
        impl<$(const $dim: usize,)+ E: NumericElement, P: Placement>
            $name<$($dim,)+ E, P>
        {
            /// Multiplies matrices without vector promotion or batch broadcasting.
            ///
            /// An unbatched rank-two right operand is accepted at every supported
            /// left rank. Batched operands must have the same rank, typed batch
            /// prefix, and actual batch dimensions.
            ///
            /// # Errors
            ///
            /// Returns a runtime tensor error for a deferred contraction mismatch,
            /// differing actual batch prefixes, or backend failure.
            ///
            /// Known contraction mismatches fail while monomorphizing the method:
            ///
            /// ```compile_fail
            /// use rstorch::typed::{DeviceCtx, Tensor2};
            /// let ctx = DeviceCtx::cpu().unwrap();
            /// let lhs = Tensor2::<2, 3>::from_vec(vec![0.0; 6], [2, 3], &ctx).unwrap();
            /// let rhs = Tensor2::<4, 2>::from_vec(vec![0.0; 8], [4, 2], &ctx).unwrap();
            /// let _ = lhs.matmul(&rhs);
            /// ```
            pub fn matmul<Rhs>(&self, rhs: &Rhs) -> Result<<Self as MatmulOutput<Rhs>>::Output>
            where
                Rhs: TypedTensor<Elem = E, Placement = P>,
                Self: MatmulOutput<Rhs>,
            {
                const {
                    assert_matmul_contract(
                        <Self as SealedTypedTensor>::MARKERS[Self::RANK - 1],
                        <Rhs as SealedTypedTensor>::MARKERS[Rhs::RANK - 2],
                    )
                };

                let lhs_binding = <Self as SealedTypedTensor>::binding(self);
                let rhs_binding = <Rhs as SealedTypedTensor>::binding(rhs);
                validate_binding::<P>(lhs_binding, "matmul")?;
                validate_binding::<P>(rhs_binding, "matmul")?;
                if !Arc::ptr_eq(lhs_binding, rhs_binding) {
                    return Err(Error::InvalidArg {
                        op: "matmul",
                        msg: "operands do not carry the same canonical placement binding".into(),
                    });
                }

                let lhs = <Self as SealedTypedTensor>::dynamic(self);
                let rhs = <Rhs as SealedTypedTensor>::dynamic(rhs);
                if Rhs::RANK > 2
                    && lhs.dims()[..Self::RANK - 2] != rhs.dims()[..Rhs::RANK - 2]
                {
                    return Err(Error::ShapeMismatch {
                        op: "matmul",
                        lhs: lhs.shape().clone(),
                        rhs: rhs.shape().clone(),
                    });
                }

                checked_wrap::<<Self as MatmulOutput<Rhs>>::Output>(
                    lhs.matmul(rhs)?,
                    Arc::clone(lhs_binding),
                    "matmul",
                )
            }
        }
    };
}

impl_matmul!(Tensor2, [D0, D1]);
impl_matmul!(Tensor3, [D0, D1, D2]);
impl_matmul!(Tensor4, [D0, D1, D2, D3]);
impl_matmul!(Tensor5, [D0, D1, D2, D3, D4]);
impl_matmul!(Tensor6, [D0, D1, D2, D3, D4, D5]);
impl_matmul!(Tensor7, [D0, D1, D2, D3, D4, D5, D6]);
impl_matmul!(Tensor8, [D0, D1, D2, D3, D4, D5, D6, D7]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed::{Cpu, DYN, DeviceCtx};
    use crate::{Device, Tensor};

    fn ctx() -> DeviceCtx<Cpu> {
        DeviceCtx::cpu().unwrap()
    }

    #[test]
    fn rank_two_values_match_runtime_and_keep_output_markers() {
        let ctx = ctx();
        let lhs =
            Tensor2::<2, 3>::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], &ctx).unwrap();
        let rhs = Tensor2::<3, 2>::from_vec(vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2], &ctx)
            .unwrap();
        let expected = lhs.as_dynamic().matmul(rhs.as_dynamic()).unwrap();

        let actual: Tensor2<2, 2> = lhs.matmul(&rhs).unwrap();
        assert_eq!(actual.dims(), [2, 2]);
        assert_eq!(
            actual.as_dynamic().to_vec::<f32>().unwrap(),
            expected.to_vec::<f32>().unwrap()
        );
    }

    #[test]
    fn every_batched_rank_accepts_an_unbatched_weight() {
        let ctx = ctx();
        let weight = Tensor2::<1, 1, i64>::from_vec(vec![3], [1, 1], &ctx).unwrap();

        macro_rules! check {
            ($ty:ty, $dims:expr) => {{
                let lhs = <$ty>::from_vec(vec![2], $dims, &ctx).unwrap();
                let out = lhs.matmul(&weight).unwrap();
                assert_eq!(out.as_dynamic().to_vec::<i64>().unwrap(), vec![6]);
                assert_eq!(out.dims(), $dims);
            }};
        }

        check!(Tensor3<1, 1, 1, i64>, [1, 1, 1]);
        check!(Tensor4<1, 1, 1, 1, i64>, [1, 1, 1, 1]);
        check!(Tensor5<1, 1, 1, 1, 1, i64>, [1, 1, 1, 1, 1]);
        check!(Tensor6<1, 1, 1, 1, 1, 1, i64>, [1, 1, 1, 1, 1, 1]);
        check!(Tensor7<1, 1, 1, 1, 1, 1, 1, i64>, [1, 1, 1, 1, 1, 1, 1]);
        check!(Tensor8<1, 1, 1, 1, 1, 1, 1, 1, i64>, [1, 1, 1, 1, 1, 1, 1, 1]);
    }

    #[test]
    fn same_rank_batched_values_match_runtime() {
        let ctx = ctx();
        let lhs =
            Tensor3::<2, 2, 3>::from_vec((1..=12).map(|x| x as f32).collect(), [2, 2, 3], &ctx)
                .unwrap();
        let rhs =
            Tensor3::<2, 3, 2>::from_vec((1..=12).map(|x| x as f32).collect(), [2, 3, 2], &ctx)
                .unwrap();
        let expected = lhs.as_dynamic().matmul(rhs.as_dynamic()).unwrap();

        let actual: Tensor3<2, 2, 2> = lhs.matmul(&rhs).unwrap();
        assert_eq!(
            actual.as_dynamic().to_vec::<f32>().unwrap(),
            expected.to_vec::<f32>().unwrap()
        );
    }

    #[test]
    fn dyn_contraction_mismatch_is_the_runtime_shape_error() {
        let ctx = ctx();
        let lhs = Tensor2::<2, DYN>::from_vec(vec![0.0f32; 6], [2, 3], &ctx).unwrap();
        let rhs = Tensor2::<DYN, 2>::from_vec(vec![0.0f32; 8], [4, 2], &ctx).unwrap();

        assert!(matches!(
            lhs.matmul(&rhs),
            Err(Error::ShapeMismatch { op: "matmul", .. })
        ));
    }

    #[test]
    fn actual_batch_prefix_mismatch_is_rejected_before_runtime_broadcast() {
        let ctx = ctx();
        let lhs = Tensor3::<DYN, 2, 3>::from_vec(vec![1.0f32; 12], [2, 2, 3], &ctx).unwrap();
        let rhs = Tensor3::<DYN, 3, 2>::from_vec(vec![1.0f32; 6], [1, 3, 2], &ctx).unwrap();
        assert!(lhs.as_dynamic().matmul(rhs.as_dynamic()).is_ok());

        assert!(matches!(
            lhs.matmul(&rhs),
            Err(Error::ShapeMismatch { op: "matmul", .. })
        ));
    }

    #[test]
    fn typed_delegation_preserves_runtime_gradients() {
        let ctx = ctx();
        let lhs_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let rhs_data = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];

        let typed_lhs_leaf = Tensor::from_vec(lhs_data.clone(), [2, 3], &Device::Cpu)
            .unwrap()
            .traced()
            .unwrap();
        let typed_rhs_leaf = Tensor::from_vec(rhs_data.clone(), [3, 2], &Device::Cpu)
            .unwrap()
            .traced()
            .unwrap();
        let typed_lhs = Tensor2::<2, 3>::try_from_dynamic(typed_lhs_leaf.clone(), &ctx).unwrap();
        let typed_rhs = Tensor2::<3, 2>::try_from_dynamic(typed_rhs_leaf.clone(), &ctx).unwrap();
        let typed_out = typed_lhs.matmul(&typed_rhs).unwrap();
        let typed_grads = typed_out.as_dynamic().backward().unwrap();

        let dynamic_lhs = Tensor::from_vec(lhs_data, [2, 3], &Device::Cpu)
            .unwrap()
            .traced()
            .unwrap();
        let dynamic_rhs = Tensor::from_vec(rhs_data, [3, 2], &Device::Cpu)
            .unwrap()
            .traced()
            .unwrap();
        let dynamic_out = dynamic_lhs.matmul(&dynamic_rhs).unwrap();
        let dynamic_grads = dynamic_out.backward().unwrap();

        assert_eq!(
            typed_grads
                .wrt_input(&typed_lhs_leaf)
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            dynamic_grads
                .wrt_input(&dynamic_lhs)
                .unwrap()
                .to_vec::<f32>()
                .unwrap()
        );
        assert_eq!(
            typed_grads
                .wrt_input(&typed_rhs_leaf)
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            dynamic_grads
                .wrt_input(&dynamic_rhs)
                .unwrap()
                .to_vec::<f32>()
                .unwrap()
        );
    }
}
