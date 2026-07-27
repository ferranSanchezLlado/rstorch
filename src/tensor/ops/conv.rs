//! Convolution and pooling ops (**T26**): [`Tensor::conv2d`],
//! [`Tensor::max_pool2d`], [`Tensor::avg_pool2d`].
//!
//! All three take an NCHW image (`[batch, channels, height, width]`) and
//! **compute** their output spatial size from the window parameters — v3 has
//! no const-generic `OUT_H`/`OUT_W` to keep in sync (exploration §4.1).
//! Kernel semantics (cross-correlation, NaN-propagating max pooling,
//! `count_include_pad = true` average pooling) are documented on
//! `backend::cpu::conv`.
//!
//! Each op computes its forward output through the backend `conv` entry point
//! and then registers a backward closure with the frozen `record` seam. The
//! closures capture the operands in **detached** form (exploration §4.3) and
//! call the gradient kernels that live next to the forward ones.

use crate::autograd;
use crate::backend::cpu::conv::Conv2dGeometry;
use crate::backend::{Conv2dParams, ConvOp, dispatch};
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::tensor::Tensor;

impl Tensor {
    /// 2-D convolution (cross-correlation, as in PyTorch) of an NCHW image
    /// with an OIHW `weight`.
    ///
    /// `self` is `[batch, in_channels, height, width]` and `weight` is
    /// `[out_channels, in_channels, kernel_h, kernel_w]`. The output is
    /// `[batch, out_channels, out_h, out_w]` with the spatial size **computed**
    /// from the window parameters, one axis at a time:
    ///
    /// ```text
    /// out = (input + 2·padding − ((kernel − 1)·dilation + 1)) / stride + 1
    /// ```
    ///
    /// Every pair is `(height, width)`. `dilation` of `(1, 1)` is the dense
    /// case. There is no bias operand: a convolution bias is a per-channel
    /// broadcast add the caller (or the `nn` layer) performs on the result.
    ///
    /// # Errors
    ///
    /// - [`Error::RankMismatch`] if either operand is not rank 4.
    /// - [`Error::ShapeMismatch`] if the weight's `in_channels` axis does not
    ///   match the input's channel axis.
    /// - [`Error::DTypeMismatch`] / [`Error::DeviceMismatch`] if the operands
    ///   disagree (there is no implicit promotion or movement).
    /// - [`Error::InvalidArg`] for a zero stride/kernel/dilation or a padded
    ///   input smaller than the effective kernel.
    /// - [`Error::Unsupported`] for a dtype the backend has no conv kernel
    ///   for (notably [`DType::Bool`](crate::DType::Bool)).
    ///
    /// # Examples
    ///
    /// ```
    /// use rstorch::{DType, Device, Tensor};
    ///
    /// let dev = Device::Cpu;
    /// // A 1x1x3x3 image and a 1x1x2x2 kernel of ones: each output is the
    /// // sum of one 2x2 window.
    /// let x = Tensor::from_vec((1..=9).map(|v| v as f32).collect(), [1, 1, 3, 3], &dev)?;
    /// let w = Tensor::ones([1, 1, 2, 2], DType::F32, &dev)?;
    /// let y = x.conv2d(&w, (1, 1), (0, 0), (1, 1))?;
    /// assert_eq!(y.dims(), &[1, 1, 2, 2]);
    /// assert_eq!(y.to_vec::<f32>()?, vec![12.0, 16.0, 24.0, 28.0]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn conv2d(
        &self,
        weight: &Tensor,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<Tensor> {
        if self.device() != weight.device() {
            return Err(Error::DeviceMismatch {
                op: "conv2d",
                expected: self.device(),
                got: weight.device(),
            });
        }
        if self.dtype() != weight.dtype() {
            return Err(Error::DTypeMismatch {
                op: "conv2d",
                expected: self.dtype(),
                got: weight.dtype(),
            });
        }
        let weight_dims = weight.dims();
        if weight_dims.len() != 4 {
            return Err(Error::RankMismatch {
                op: "conv2d",
                expected: 4,
                got: weight_dims.len(),
            });
        }
        let params = Conv2dParams {
            kernel: (weight_dims[2], weight_dims[3]),
            stride,
            padding,
            dilation,
        };
        let geo = Conv2dGeometry::conv2d("conv2d", self.dims(), weight_dims, &params)?;
        let storage = dispatch::backend(self.device()).conv(
            ConvOp::Conv2d,
            &[self.view(), weight.view()],
            &params,
        )?;
        let out = Tensor::from_parts(storage, Layout::contiguous(geo.output_dims())?);

        // Detached captures: the closure must not hold an `Arc` back into the
        // graph it is attached to (exploration §4.3).
        let saved_input = self.detach();
        let saved_weight = weight.detach();
        Ok(autograd::record(
            "conv2d",
            out,
            &[self, weight],
            Box::new(move |g| {
                let backend = dispatch::backend(g.device());
                vec![
                    from_kernel(
                        backend.conv(
                            ConvOp::Conv2dInputGrad,
                            &[g.view(), saved_weight.view(), saved_input.view()],
                            &params,
                        ),
                        geo.input_dims(),
                    ),
                    from_kernel(
                        backend.conv(
                            ConvOp::Conv2dWeightGrad,
                            &[g.view(), saved_input.view(), saved_weight.view()],
                            &params,
                        ),
                        geo.weight_dims(),
                    ),
                ]
            }),
        ))
    }

    /// 2-D max pooling over an NCHW image.
    ///
    /// `kernel`, `stride` and `padding` are `(height, width)` pairs; the
    /// output spatial size follows the same formula as
    /// [`conv2d`](Tensor::conv2d) with a dilation of 1. Padding positions are
    /// **not** candidates for the maximum, and `padding` may not exceed half
    /// the window on either axis, so no window is ever empty.
    ///
    /// A window containing a NaN pools to NaN (as in PyTorch), and the
    /// gradient of a tie goes to the first winning position — the same
    /// first-wins rule `argmax` uses.
    ///
    /// # Errors
    ///
    /// As [`conv2d`](Tensor::conv2d) (minus the weight-specific cases), plus
    /// [`Error::InvalidArg`] when `2 · padding` exceeds the window.
    ///
    /// # Examples
    ///
    /// ```
    /// use rstorch::{Device, Tensor};
    ///
    /// let dev = Device::Cpu;
    /// let x = Tensor::from_vec((1..=16).map(|v| v as f32).collect(), [1, 1, 4, 4], &dev)?;
    /// let y = x.max_pool2d((2, 2), (2, 2), (0, 0))?;
    /// assert_eq!(y.to_vec::<f32>()?, vec![6.0, 8.0, 14.0, 16.0]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn max_pool2d(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Tensor> {
        let (geo, out) = self.pool("max_pool2d", ConvOp::MaxPool2d, kernel, stride, padding)?;
        let saved_input = self.detach();
        let params = Conv2dParams {
            kernel,
            stride,
            padding,
            dilation: (1, 1),
        };
        Ok(autograd::record(
            "max_pool2d",
            out,
            &[self],
            Box::new(move |g| {
                vec![from_kernel(
                    dispatch::backend(g.device()).conv(
                        ConvOp::MaxPool2dBackward,
                        &[g.view(), saved_input.view()],
                        &params,
                    ),
                    geo.input_dims(),
                )]
            }),
        ))
    }

    /// 2-D average pooling over an NCHW image.
    ///
    /// Geometry is exactly [`max_pool2d`](Tensor::max_pool2d)'s. The divisor
    /// is the **full window area** (`kernel.0 · kernel.1`), i.e. PyTorch's
    /// `count_include_pad = true` default: padding positions contribute zero
    /// to the sum but still count in the divisor.
    ///
    /// # Errors
    ///
    /// As [`max_pool2d`](Tensor::max_pool2d).
    ///
    /// # Examples
    ///
    /// ```
    /// use rstorch::{Device, Tensor};
    ///
    /// let dev = Device::Cpu;
    /// let x = Tensor::from_vec((1..=16).map(|v| v as f32).collect(), [1, 1, 4, 4], &dev)?;
    /// let y = x.avg_pool2d((2, 2), (2, 2), (0, 0))?;
    /// assert_eq!(y.to_vec::<f32>()?, vec![3.5, 5.5, 11.5, 13.5]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn avg_pool2d(
        &self,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Tensor> {
        let (geo, out) = self.pool("avg_pool2d", ConvOp::AvgPool2d, kernel, stride, padding)?;
        let saved_input = self.detach();
        let params = Conv2dParams {
            kernel,
            stride,
            padding,
            dilation: (1, 1),
        };
        Ok(autograd::record(
            "avg_pool2d",
            out,
            &[self],
            Box::new(move |g| {
                vec![from_kernel(
                    dispatch::backend(g.device()).conv(
                        ConvOp::AvgPool2dBackward,
                        &[g.view(), saved_input.view()],
                        &params,
                    ),
                    geo.input_dims(),
                )]
            }),
        ))
    }

    /// Shared forward half of the two pooling ops: resolve the geometry, run
    /// the kernel, and wrap the result in an untraced output tensor.
    fn pool(
        &self,
        op: &'static str,
        conv_op: ConvOp,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<(Conv2dGeometry, Tensor)> {
        let params = Conv2dParams {
            kernel,
            stride,
            padding,
            dilation: (1, 1),
        };
        let geo = Conv2dGeometry::pool(op, self.dims(), &params)?;
        let storage = dispatch::backend(self.device()).conv(conv_op, &[self.view()], &params)?;
        let out = Tensor::from_parts(storage, Layout::contiguous(geo.output_dims())?);
        Ok((geo, out))
    }
}

/// Turn a gradient-kernel result into the optional cotangent the backward
/// seam expects.
///
/// The seam is infallible (a `BackwardFn` yields `Option<Tensor>`, not
/// `Result`), so a kernel failure becomes "no gradient for this input". The
/// only reachable failures are a non-CPU cotangent — the forward pass would
/// have failed there first — and a layout allocation whose element count the
/// forward output already proved fits.
fn from_kernel(storage: Result<crate::storage::Storage>, dims: [usize; 4]) -> Option<Tensor> {
    let storage = storage.ok()?;
    let layout = Layout::contiguous(dims).ok()?;
    Some(Tensor::from_parts(storage, layout))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::dtype::DType;

    const CPU: Device = Device::Cpu;

    fn image(dims: [usize; 4]) -> Tensor {
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 - 3.0).collect();
        Tensor::from_vec(data, dims, &CPU).unwrap()
    }

    fn seq(dims: [usize; 4]) -> Tensor {
        let n: usize = dims.iter().product();
        let data: Vec<f32> = (1..=n).map(|i| i as f32).collect();
        Tensor::from_vec(data, dims, &CPU).unwrap()
    }

    // ----- forward ------------------------------------------------------

    #[test]
    fn conv2d_hand_computed() {
        // 1x1x3x3 = 1..9 with the 2x2 diagonal kernel [[1,0],[0,1]].
        let x = seq([1, 1, 3, 3]);
        let w = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], [1, 1, 2, 2], &CPU).unwrap();
        let y = x.conv2d(&w, (1, 1), (0, 0), (1, 1)).unwrap();
        assert_eq!(y.dims(), &[1, 1, 2, 2]);
        assert_eq!(y.to_vec::<f32>().unwrap(), vec![6.0, 8.0, 12.0, 14.0]);
        assert_eq!(y.dtype(), DType::F32);
        assert!(y.is_contiguous());
    }

    #[test]
    fn conv2d_multi_channel_and_batch() {
        // 2 images, 2 input channels, 3 output channels of ones: every output
        // is the sum of that image's 2x2x2 window.
        let x = seq([2, 2, 2, 2]);
        let w = Tensor::ones([3, 2, 2, 2], DType::F32, &CPU).unwrap();
        let y = x.conv2d(&w, (1, 1), (0, 0), (1, 1)).unwrap();
        assert_eq!(y.dims(), &[2, 3, 1, 1]);
        // image 0 = 1..8 -> 36 ; image 1 = 9..16 -> 100.
        assert_eq!(
            y.to_vec::<f32>().unwrap(),
            vec![36.0, 36.0, 36.0, 100.0, 100.0, 100.0]
        );
    }

    #[test]
    fn conv2d_computes_the_output_size() {
        let x = Tensor::zeros([1, 1, 7, 7], DType::F32, &CPU).unwrap();
        let w = Tensor::zeros([2, 1, 3, 3], DType::F32, &CPU).unwrap();
        let cases = [
            ((1, 1), (0, 0), (1, 1), [1, 2, 5, 5]),
            ((2, 2), (1, 1), (1, 1), [1, 2, 4, 4]),
            ((1, 1), (1, 1), (1, 1), [1, 2, 7, 7]),
            ((1, 1), (0, 0), (2, 2), [1, 2, 3, 3]),
            ((3, 1), (0, 0), (1, 1), [1, 2, 2, 5]),
        ];
        for (stride, padding, dilation, want) in cases {
            let y = x.conv2d(&w, stride, padding, dilation).unwrap();
            assert_eq!(y.dims(), want.as_slice(), "{stride:?}/{padding:?}");
        }
    }

    #[test]
    fn max_and_avg_pool_hand_computed() {
        let x = seq([1, 1, 4, 4]);
        let m = x.max_pool2d((2, 2), (2, 2), (0, 0)).unwrap();
        assert_eq!(m.dims(), &[1, 1, 2, 2]);
        assert_eq!(m.to_vec::<f32>().unwrap(), vec![6.0, 8.0, 14.0, 16.0]);

        let a = x.avg_pool2d((2, 2), (2, 2), (0, 0)).unwrap();
        assert_eq!(a.to_vec::<f32>().unwrap(), vec![3.5, 5.5, 11.5, 13.5]);
    }

    #[test]
    fn pooling_keeps_the_channel_count() {
        let x = seq([2, 3, 4, 4]);
        let y = x.max_pool2d((2, 2), (2, 2), (0, 0)).unwrap();
        assert_eq!(y.dims(), &[2, 3, 2, 2]);
    }

    #[test]
    fn overlapping_pool_windows() {
        // 3x3 window, stride 1 over 1x1x4x4 -> 2x2 output.
        let x = seq([1, 1, 4, 4]);
        let y = x.max_pool2d((3, 3), (1, 1), (0, 0)).unwrap();
        assert_eq!(y.dims(), &[1, 1, 2, 2]);
        assert_eq!(y.to_vec::<f32>().unwrap(), vec![11.0, 12.0, 15.0, 16.0]);
    }

    #[test]
    fn avg_pool_counts_padding_in_the_divisor() {
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [1, 1, 2, 2], &CPU).unwrap();
        let y = x.avg_pool2d((2, 2), (2, 2), (1, 1)).unwrap();
        assert_eq!(y.dims(), &[1, 1, 2, 2]);
        assert_eq!(y.to_vec::<f32>().unwrap(), vec![0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn conv2d_accepts_a_non_contiguous_input() {
        // Build a [1,1,3,3] view of the second image of a [2,1,3,3] tensor by
        // narrowing the batch axis (offset != 0, still stride-aware).
        let x = seq([2, 1, 3, 3]);
        let narrowed = Tensor::from_parts(x.storage().clone(), x.layout().narrow(0, 1, 1).unwrap());
        assert!(!narrowed.is_contiguous());
        let w = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], [1, 1, 2, 2], &CPU).unwrap();
        let y = narrowed.conv2d(&w, (1, 1), (0, 0), (1, 1)).unwrap();
        assert_eq!(y.to_vec::<f32>().unwrap(), vec![24.0, 26.0, 30.0, 32.0]);
    }

    // ----- error contracts ----------------------------------------------

    #[test]
    fn rank_must_be_four() {
        let x = Tensor::zeros([1, 3, 3], DType::F32, &CPU).unwrap();
        let w = Tensor::zeros([1, 1, 2, 2], DType::F32, &CPU).unwrap();
        assert!(matches!(
            x.conv2d(&w, (1, 1), (0, 0), (1, 1)),
            Err(Error::RankMismatch {
                op: "conv2d",
                expected: 4,
                got: 3
            })
        ));
        assert!(matches!(
            x.max_pool2d((2, 2), (2, 2), (0, 0)),
            Err(Error::RankMismatch {
                op: "max_pool2d",
                ..
            })
        ));
        let w3 = Tensor::zeros([1, 2, 2], DType::F32, &CPU).unwrap();
        let x4 = Tensor::zeros([1, 1, 3, 3], DType::F32, &CPU).unwrap();
        assert!(matches!(
            x4.conv2d(&w3, (1, 1), (0, 0), (1, 1)),
            Err(Error::RankMismatch { op: "conv2d", .. })
        ));
    }

    #[test]
    fn channel_mismatch_is_a_shape_error() {
        let x = Tensor::zeros([1, 3, 5, 5], DType::F32, &CPU).unwrap();
        let w = Tensor::zeros([2, 2, 3, 3], DType::F32, &CPU).unwrap();
        assert!(matches!(
            x.conv2d(&w, (1, 1), (0, 0), (1, 1)),
            Err(Error::ShapeMismatch { op: "conv2d", .. })
        ));
    }

    #[test]
    fn dtype_mismatch_is_loud() {
        let x = Tensor::zeros([1, 1, 3, 3], DType::F32, &CPU).unwrap();
        let w = Tensor::zeros([1, 1, 2, 2], DType::I64, &CPU).unwrap();
        assert!(matches!(
            x.conv2d(&w, (1, 1), (0, 0), (1, 1)),
            Err(Error::DTypeMismatch { op: "conv2d", .. })
        ));
    }

    #[test]
    fn bool_convolution_is_unsupported() {
        let x = Tensor::zeros([1, 1, 3, 3], DType::Bool, &CPU).unwrap();
        let w = Tensor::zeros([1, 1, 2, 2], DType::Bool, &CPU).unwrap();
        assert!(matches!(
            x.conv2d(&w, (1, 1), (0, 0), (1, 1)),
            Err(Error::Unsupported {
                op: "conv2d",
                dtype: DType::Bool,
                ..
            })
        ));
    }

    #[test]
    fn zero_stride_is_invalid() {
        let x = Tensor::zeros([1, 1, 3, 3], DType::F32, &CPU).unwrap();
        let w = Tensor::zeros([1, 1, 2, 2], DType::F32, &CPU).unwrap();
        assert!(matches!(
            x.conv2d(&w, (0, 1), (0, 0), (1, 1)),
            Err(Error::InvalidArg { op: "conv2d", .. })
        ));
    }

    #[test]
    fn kernel_larger_than_the_padded_input_is_invalid() {
        let x = Tensor::zeros([1, 1, 3, 3], DType::F32, &CPU).unwrap();
        let w = Tensor::zeros([1, 1, 5, 5], DType::F32, &CPU).unwrap();
        assert!(matches!(
            x.conv2d(&w, (1, 1), (0, 0), (1, 1)),
            Err(Error::InvalidArg { op: "conv2d", .. })
        ));
    }

    #[test]
    fn pool_padding_over_half_the_window_is_invalid() {
        let x = Tensor::zeros([1, 1, 4, 4], DType::F32, &CPU).unwrap();
        assert!(matches!(
            x.avg_pool2d((2, 2), (2, 2), (2, 2)),
            Err(Error::InvalidArg {
                op: "avg_pool2d",
                ..
            })
        ));
    }

    // ----- backward: finite differences ----------------------------------
    //
    // Written against the single `testing::check_grad` harness. The
    // implementation-plan §4 grid deferred T26's cases to milestone m4 on the
    // assumption that T26 would not be merged by m2; it was, and these pass,
    // so **T31** activates them as regression protection rather than leaving
    // a merged op family unguarded.
    //
    // `check_grad` needs a scalar-valued function, and reductions belong to
    // T23, so each case ends in a convolution with a one-element output —
    // that final convolution *is* the weighted readout.

    /// A `[1, 1, h, w]` kernel with pairwise-distinct weights, collapsing a
    /// `[1, 1, h, w]` feature map to one element.
    ///
    /// The weights are deliberately not all ones: a uniform readout gives
    /// every spatial position the same cotangent, so a backward that scattered
    /// the gradient to the wrong position within a window could still agree
    /// with finite differences. Distinct weights make placement observable.
    fn readout_kernel(h: usize, w: usize) -> Tensor {
        let data: Vec<f32> = (0..h * w).map(|i| 0.25 + 0.5 * (i as f32)).collect();
        Tensor::from_vec(data, [1, 1, h, w], &CPU).unwrap()
    }

    #[test]
    fn conv2d_backward_matches_finite_differences() {
        let x = image([1, 1, 4, 4]);
        let w = image([1, 1, 3, 3]);
        let sum = readout_kernel(2, 2);
        crate::testing::check_grad(
            move |xs| {
                let y = xs[0].conv2d(&xs[1], (1, 1), (0, 0), (1, 1))?;
                y.conv2d(&sum, (1, 1), (0, 0), (1, 1))
            },
            &[x, w],
            1e-3,
            1e-3,
        )
        .unwrap();
    }

    #[test]
    fn conv2d_backward_with_stride_padding_and_dilation() {
        // stride 2, padding 1, dilation 2 over 5x5 with a 2x2 kernel:
        // effective kernel 3, padded 7, so the feature map is 3x3.
        let x = image([1, 2, 5, 5]);
        let w = image([1, 2, 2, 2]);
        let sum = readout_kernel(3, 3);
        crate::testing::check_grad(
            move |xs| {
                let y = xs[0].conv2d(&xs[1], (2, 2), (1, 1), (2, 2))?;
                y.conv2d(&sum, (1, 1), (0, 0), (1, 1))
            },
            &[x, w],
            1e-3,
            1e-3,
        )
        .unwrap();
    }

    #[test]
    fn max_pool2d_backward_matches_finite_differences() {
        // Strictly increasing values: no ties, so the maximum is a locally
        // smooth function of the input.
        let x = image([1, 1, 4, 4]);
        let sum = readout_kernel(2, 2);
        crate::testing::check_grad(
            move |xs| {
                let y = xs[0].max_pool2d((2, 2), (2, 2), (0, 0))?;
                y.conv2d(&sum, (1, 1), (0, 0), (1, 1))
            },
            &[x],
            1e-3,
            1e-3,
        )
        .unwrap();
    }

    #[test]
    fn avg_pool2d_backward_matches_finite_differences() {
        // Padded pooling: 4x4 with a 2x2 window, stride 2, padding 1 -> 3x3.
        let x = image([1, 1, 4, 4]);
        let sum = readout_kernel(3, 3);
        crate::testing::check_grad(
            move |xs| {
                let y = xs[0].avg_pool2d((2, 2), (2, 2), (1, 1))?;
                y.conv2d(&sum, (1, 1), (0, 0), (1, 1))
            },
            &[x],
            1e-3,
            1e-3,
        )
        .unwrap();
    }
}
