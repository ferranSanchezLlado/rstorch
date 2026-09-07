use super::{Conv2dOutput, Pool2dOutput, dynamic, wrap};
use crate::Result;
use crate::typed::const_check::assert_conv2d_channels;
use crate::typed::{NumericElement, Placement, Tensor4};

/// Expands one rank-4 NCHW pooling method; `$kind` names the reduction in its
/// doc line and `$method` is also the runtime method name and error op label.
macro_rules! pool2d {
    ($method:ident, $kind:literal) => {
        #[doc = concat!("Applies rank-4 NCHW ", $kind, " pooling through the runtime tensor kernel.")]
        pub fn $method(
            &self,
            kernel: (usize, usize),
            stride: (usize, usize),
            padding: (usize, usize),
        ) -> Result<<Self as Pool2dOutput>::Output> {
            let op = stringify!($method);
            wrap(
                self,
                dynamic(self, op)?.$method(kernel, stride, padding)?,
                op,
            )
        }
    };
}

impl<
    const BATCH: usize,
    const INPUT_CHANNELS: usize,
    const H: usize,
    const W: usize,
    E: NumericElement,
    P: Placement,
> Tensor4<BATCH, INPUT_CHANNELS, H, W, E, P>
{
    /// Applies a rank-4 NCHW convolution through the runtime tensor kernel.
    pub fn conv2d<const OUT: usize, const WEIGHT_INPUT: usize, const KH: usize, const KW: usize>(
        &self,
        weight: &Tensor4<OUT, WEIGHT_INPUT, KH, KW, E, P>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<<Self as Conv2dOutput<Tensor4<OUT, WEIGHT_INPUT, KH, KW, E, P>>>::Output> {
        const { assert_conv2d_channels(INPUT_CHANNELS, WEIGHT_INPUT) };
        let output = dynamic(self, "conv2d")?.conv2d(
            dynamic(weight, "conv2d")?,
            stride,
            padding,
            dilation,
        )?;
        wrap(self, output, "conv2d")
    }

    pool2d!(max_pool2d, "max");
    pool2d!(avg_pool2d, "average");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed::sealed::TypedTensor as SealedTypedTensor;
    use crate::typed::{Cpu, DYN, DeviceBinding, DeviceCtx};
    use crate::{Device, Error, Tensor};
    use std::sync::Arc;

    fn ctx() -> DeviceCtx<Cpu> {
        DeviceCtx::cpu().unwrap()
    }

    fn values(len: usize) -> Vec<f32> {
        (0..len).map(|i| i as f32 * 0.25 - 1.0).collect()
    }

    #[test]
    fn typed_forward_results_match_dynamic_delegates() {
        let ctx = ctx();
        let x = Tensor4::<2, 2, 4, 5>::from_vec(values(80), [2, 2, 4, 5], &ctx).unwrap();
        let w = Tensor4::<3, 2, 2, 3>::from_vec(values(36), [3, 2, 2, 3], &ctx).unwrap();

        let typed_conv = x.conv2d(&w, (2, 1), (1, 1), (1, 1)).unwrap();
        let dynamic_conv = x
            .as_dynamic()
            .conv2d(w.as_dynamic(), (2, 1), (1, 1), (1, 1))
            .unwrap();
        assert_eq!(typed_conv.dims(), [2, 3, 3, 5]);
        assert_eq!(
            typed_conv.as_dynamic().to_vec::<f32>().unwrap(),
            dynamic_conv.to_vec::<f32>().unwrap()
        );

        let typed_max = x.max_pool2d((2, 3), (2, 1), (0, 1)).unwrap();
        let dynamic_max = x.as_dynamic().max_pool2d((2, 3), (2, 1), (0, 1)).unwrap();
        assert_eq!(typed_max.dims(), [2, 2, 2, 5]);
        assert_eq!(
            typed_max.as_dynamic().to_vec::<f32>().unwrap(),
            dynamic_max.to_vec::<f32>().unwrap()
        );

        let typed_avg = x.avg_pool2d((2, 3), (2, 1), (0, 1)).unwrap();
        let dynamic_avg = x.as_dynamic().avg_pool2d((2, 3), (2, 1), (0, 1)).unwrap();
        assert_eq!(typed_avg.dims(), [2, 2, 2, 5]);
        assert_eq!(
            typed_avg.as_dynamic().to_vec::<f32>().unwrap(),
            dynamic_avg.to_vec::<f32>().unwrap()
        );
    }

    #[test]
    fn dynamic_channel_mismatch_is_a_runtime_shape_error() {
        let ctx = ctx();
        let x = Tensor4::<1, DYN, 3, 3>::from_vec(values(18), [1, 2, 3, 3], &ctx).unwrap();
        let w = Tensor4::<1, DYN, 2, 2>::from_vec(values(12), [1, 3, 2, 2], &ctx).unwrap();

        assert!(matches!(
            x.conv2d(&w, (1, 1), (0, 0), (1, 1)),
            Err(Error::ShapeMismatch { op: "conv2d", .. })
        ));
    }

    #[test]
    fn runtime_argument_errors_are_preserved() {
        let ctx = ctx();
        let x = Tensor4::<1, 1, 3, 3>::from_vec(values(9), [1, 1, 3, 3], &ctx).unwrap();
        let w = Tensor4::<1, 1, 2, 2>::from_vec(values(4), [1, 1, 2, 2], &ctx).unwrap();

        assert!(matches!(
            x.conv2d(&w, (0, 1), (0, 0), (1, 1)),
            Err(Error::InvalidArg { op: "conv2d", .. })
        ));
        assert!(matches!(
            x.avg_pool2d((2, 2), (1, 1), (2, 0)),
            Err(Error::InvalidArg {
                op: "avg_pool2d",
                ..
            })
        ));
    }

    #[test]
    fn operations_reject_noncanonical_bindings_before_delegation() {
        let ctx = ctx();
        let dynamic_input = Tensor::from_vec(values(9), [1, 1, 3, 3], &Device::Cpu).unwrap();
        let forged = Arc::new(DeviceBinding {
            device: Device::Cpu,
        });
        let x = <Tensor4<1, 1, 3, 3> as SealedTypedTensor>::trusted_from_validated(
            dynamic_input,
            Arc::clone(&forged),
        );
        let w = Tensor4::<1, 1, 2, 2>::from_vec(values(4), [1, 1, 2, 2], &ctx).unwrap();

        assert!(matches!(
            x.conv2d(&w, (1, 1), (0, 0), (1, 1)),
            Err(Error::InvalidArg { op: "conv2d", .. })
        ));
        assert!(matches!(
            x.max_pool2d((2, 2), (1, 1), (0, 0)),
            Err(Error::InvalidArg {
                op: "max_pool2d",
                ..
            })
        ));

        let canonical_x = Tensor4::<1, 1, 3, 3>::from_vec(values(9), [1, 1, 3, 3], &ctx).unwrap();
        let dynamic_weight = Tensor::from_vec(values(4), [1, 1, 2, 2], &Device::Cpu).unwrap();
        let forged_w = <Tensor4<1, 1, 2, 2> as SealedTypedTensor>::trusted_from_validated(
            dynamic_weight,
            forged,
        );
        assert!(matches!(
            canonical_x.conv2d(&forged_w, (1, 1), (0, 0), (1, 1)),
            Err(Error::InvalidArg { op: "conv2d", .. })
        ));
    }

    #[test]
    fn typed_delegation_preserves_conv_and_pool_gradients() {
        let ctx = ctx();
        let x = Tensor::from_vec(values(16), [1, 1, 4, 4], &Device::Cpu).unwrap();
        let w = Tensor::from_vec(values(9), [1, 1, 3, 3], &Device::Cpu).unwrap();
        let readout =
            Tensor::from_vec(vec![0.5f32, 1.0, 1.5, 2.0], [1, 1, 2, 2], &Device::Cpu).unwrap();
        crate::testing::check_grad(
            |xs| {
                let x = Tensor4::<1, 1, 4, 4>::try_from_dynamic(xs[0].clone(), &ctx)?;
                let w = Tensor4::<1, 1, 3, 3>::try_from_dynamic(xs[1].clone(), &ctx)?;
                x.conv2d(&w, (1, 1), (0, 0), (1, 1))?.into_dynamic().conv2d(
                    &readout,
                    (1, 1),
                    (0, 0),
                    (1, 1),
                )
            },
            &[x.clone(), w],
            1e-3,
            1e-3,
        )
        .unwrap();

        let pool_readout = readout.clone();
        crate::testing::check_grad(
            |xs| {
                Tensor4::<1, 1, 4, 4>::try_from_dynamic(xs[0].clone(), &ctx)?
                    .max_pool2d((2, 2), (2, 2), (0, 0))?
                    .into_dynamic()
                    .conv2d(&pool_readout, (1, 1), (0, 0), (1, 1))
            },
            std::slice::from_ref(&x),
            1e-3,
            1e-3,
        )
        .unwrap();

        crate::testing::check_grad(
            |xs| {
                Tensor4::<1, 1, 4, 4>::try_from_dynamic(xs[0].clone(), &ctx)?
                    .avg_pool2d((2, 2), (2, 2), (0, 0))?
                    .into_dynamic()
                    .conv2d(&readout, (1, 1), (0, 0), (1, 1))
            },
            &[x],
            1e-3,
            1e-3,
        )
        .unwrap();
    }

    #[test]
    fn numeric_i64_is_accepted() -> Result<()> {
        let ctx = ctx();
        let x = Tensor4::<1, 1, 2, 2, i64>::from_vec(vec![1, 2, 3, 4], [1, 1, 2, 2], &ctx).unwrap();
        let w = Tensor4::<1, 1, 1, 1, i64>::from_vec(vec![2], [1, 1, 1, 1], &ctx).unwrap();
        assert_eq!(
            x.conv2d(&w, (1, 1), (0, 0), (1, 1))?
                .as_dynamic()
                .to_vec::<i64>()?,
            vec![2, 4, 6, 8]
        );
        assert_eq!(
            x.max_pool2d((2, 2), (1, 1), (0, 0))?
                .as_dynamic()
                .to_vec::<i64>()?,
            vec![4]
        );
        Ok(())
    }
}
