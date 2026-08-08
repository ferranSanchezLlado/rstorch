//! Typed loss operations.

use super::super::const_check::assert_loss_rows;
use super::super::{FloatElement, Placement, Tensor0, Tensor1, Tensor2, typed_rank_table};
use super::{dynamic, wrap};
use crate::Result;

macro_rules! impl_mse_loss {
    ($(($name:ident, $rank:literal, [$($dim:ident),*])),+ $(,)?) => {
        $(
            impl<$(const $dim: usize,)* E: FloatElement, P: Placement>
                super::super::$name<$($dim,)* E, P>
            {
                /// Returns the mean squared error against an exactly typed target.
                ///
                /// Shapes are strict, including the actual values represented by
                /// [`DYN`](super::super::DYN); this operation never broadcasts.
                pub fn mse_loss(&self, target: &Self) -> Result<Tensor0<E, P>> {
                    const OP: &str = "mse_loss";
                    let output = dynamic(self, OP)?.mse_loss(dynamic(target, OP)?)?;
                    wrap(self, output, OP)
                }
            }
        )+
    };
}

typed_rank_table!(impl_mse_loss);

impl<const ROWS: usize, const CLASSES: usize, E: FloatElement, P: Placement>
    Tensor2<ROWS, CLASSES, E, P>
{
    /// Returns mean cross-entropy for one `i64` class label per logits row.
    pub fn cross_entropy<const TARGET_ROWS: usize>(
        &self,
        targets: &Tensor1<TARGET_ROWS, i64, P>,
    ) -> Result<Tensor0<E, P>> {
        const { assert_loss_rows(ROWS, TARGET_ROWS) };
        self.cross_entropy_impl(targets, None, "cross_entropy")
    }

    /// Returns mean cross-entropy after excluding labels equal to `ignore_index`.
    ///
    /// An all-ignored batch retains the runtime operation's zero loss and zero
    /// gradient behavior.
    pub fn cross_entropy_ignore_index<const TARGET_ROWS: usize>(
        &self,
        targets: &Tensor1<TARGET_ROWS, i64, P>,
        ignore_index: i64,
    ) -> Result<Tensor0<E, P>> {
        const { assert_loss_rows(ROWS, TARGET_ROWS) };
        self.cross_entropy_impl(targets, Some(ignore_index), "cross_entropy_ignore_index")
    }

    fn cross_entropy_impl<const TARGET_ROWS: usize>(
        &self,
        targets: &Tensor1<TARGET_ROWS, i64, P>,
        ignore_index: Option<i64>,
        op: &'static str,
    ) -> Result<Tensor0<E, P>> {
        let logits = dynamic(self, op)?;
        let targets = dynamic(targets, op)?;
        let output = match ignore_index {
            Some(ignore_index) => logits.cross_entropy_ignore_index(targets, ignore_index)?,
            None => logits.cross_entropy(targets)?,
        };
        wrap(self, output, op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed::sealed::{DeviceBinding, TypedTensor as SealedTypedTensor};
    use crate::typed::{Cpu, DYN, DeviceCtx, Tensor3};
    use crate::{Device, Error, Tensor};
    use std::sync::Arc;

    fn cpu() -> DeviceCtx<Cpu> {
        DeviceCtx::cpu().unwrap()
    }

    fn assert_scalar<T>(_: &Tensor0<T, Cpu>)
    where
        T: FloatElement,
    {
    }

    #[test]
    fn mse_is_strict_and_returns_a_typed_scalar() {
        let ctx = cpu();
        let prediction =
            Tensor3::<DYN, 1, 2>::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 1, 2], &ctx).unwrap();
        let target =
            Tensor3::<DYN, 1, 2>::from_vec(vec![0.0f32, 2.0, 1.0, 4.0], [2, 1, 2], &ctx).unwrap();
        let loss = prediction.mse_loss(&target).unwrap();
        assert_scalar(&loss);
        assert_eq!(loss.as_dynamic().item().unwrap(), 1.25);

        let mismatched = Tensor3::<DYN, 1, 2>::from_vec(vec![0.0f32; 6], [3, 1, 2], &ctx).unwrap();
        assert!(matches!(
            prediction.mse_loss(&mismatched),
            Err(Error::ShapeMismatch { op: "mse_loss", .. })
        ));
    }

    #[test]
    fn mse_preserves_both_runtime_gradient_inputs_and_empty_error() {
        let ctx = cpu();
        let prediction = Tensor1::<2>::try_from_dynamic(
            Tensor::from_vec(vec![1.0f32, 3.0], [2], &Device::Cpu)
                .unwrap()
                .traced()
                .unwrap(),
            &ctx,
        )
        .unwrap();
        let target = Tensor1::<2>::try_from_dynamic(
            Tensor::from_vec(vec![0.0f32, 1.0], [2], &Device::Cpu)
                .unwrap()
                .traced()
                .unwrap(),
            &ctx,
        )
        .unwrap();
        let loss = prediction.mse_loss(&target).unwrap();
        let grads = loss.as_dynamic().backward().unwrap();
        assert_eq!(
            grads
                .wrt_input(prediction.as_dynamic())
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            vec![1.0, 2.0]
        );
        assert_eq!(
            grads
                .wrt_input(target.as_dynamic())
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            vec![-1.0, -2.0]
        );

        let empty = Tensor1::<0>::from_vec(Vec::<f32>::new(), [0], &ctx).unwrap();
        assert!(matches!(
            empty.mse_loss(&empty),
            Err(Error::InvalidArg { op: "mse_loss", .. })
        ));
    }

    #[test]
    fn cross_entropy_defers_dynamic_rows_and_delegates_errors() {
        let ctx = cpu();
        let logits = Tensor2::<DYN, 3>::from_vec(vec![0.0f32; 6], [2, 3], &ctx).unwrap();
        let labels = Tensor1::<2, i64>::from_vec(vec![2, 0], [2], &ctx).unwrap();
        let loss = logits.cross_entropy(&labels).unwrap();
        assert_scalar(&loss);
        assert!((loss.as_dynamic().item().unwrap() - 3f64.ln()).abs() < 1e-6);

        let wrong_rows = Tensor1::<DYN, i64>::from_vec(vec![0], [1], &ctx).unwrap();
        assert!(matches!(
            logits.cross_entropy(&wrong_rows),
            Err(Error::ShapeMismatch {
                op: "cross_entropy",
                ..
            })
        ));

        let invalid = Tensor1::<2, i64>::from_vec(vec![3, 0], [2], &ctx).unwrap();
        assert!(matches!(
            logits.cross_entropy(&invalid),
            Err(Error::IndexOutOfBounds {
                op: "cross_entropy",
                ..
            })
        ));
    }

    #[test]
    fn cross_entropy_keeps_empty_and_all_ignored_semantics_and_gradients() {
        let ctx = cpu();
        let empty_logits = Tensor2::<0, 3>::from_vec(Vec::<f32>::new(), [0, 3], &ctx).unwrap();
        let empty_labels = Tensor1::<0, i64>::from_vec(Vec::new(), [0], &ctx).unwrap();
        assert_eq!(
            empty_logits
                .cross_entropy(&empty_labels)
                .unwrap()
                .as_dynamic()
                .item()
                .unwrap(),
            0.0
        );

        let logits = Tensor2::<2, 3>::try_from_dynamic(
            Tensor::from_vec(vec![0.0f32; 6], [2, 3], &Device::Cpu)
                .unwrap()
                .traced()
                .unwrap(),
            &ctx,
        )
        .unwrap();
        let labels = Tensor1::<2, i64>::from_vec(vec![-100, -100], [2], &ctx).unwrap();
        let loss = logits.cross_entropy_ignore_index(&labels, -100).unwrap();
        assert_eq!(loss.as_dynamic().item().unwrap(), 0.0);
        let grads = loss.as_dynamic().backward().unwrap();
        assert_eq!(
            grads
                .wrt_input(logits.as_dynamic())
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            vec![0.0; 6]
        );
    }

    #[test]
    fn losses_reject_noncanonical_bindings_before_delegation() {
        let ctx = cpu();
        let runtime = Tensor::from_vec(vec![0.0f32], [1], &Device::Cpu).unwrap();
        let forged = <Tensor1<1> as SealedTypedTensor>::trusted_from_validated(
            runtime,
            Arc::new(DeviceBinding {
                device: Device::Cpu,
            }),
        );
        let valid = Tensor1::<1>::from_vec(vec![0.0], [1], &ctx).unwrap();
        assert!(matches!(
            forged.mse_loss(&valid),
            Err(Error::InvalidArg { op: "mse_loss", .. })
        ));
    }
}
