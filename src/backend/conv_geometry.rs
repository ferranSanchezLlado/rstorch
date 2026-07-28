//! Device-independent convolution and pooling geometry.

use crate::backend::Conv2dParams;
use crate::error::{Error, Result};
use crate::shape::Shape;

/// The resolved, validated geometry of one conv or pool call: operand sizes,
/// the computed output spatial size, and the window parameters.
///
/// Device-independent shape math. The op layer resolves it once (to build the
/// output layout) and hands it back to the gradient kernels, so forward and
/// backward cannot disagree about the window.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Conv2dGeometry {
    /// Public op name, used in this type's error messages.
    pub(super) op: &'static str,
    pub(super) batch: usize,
    pub(super) in_channels: usize,
    pub(super) in_h: usize,
    pub(super) in_w: usize,
    pub(super) out_channels: usize,
    pub(super) kernel_h: usize,
    pub(super) kernel_w: usize,
    pub(super) out_h: usize,
    pub(super) out_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
}

impl Conv2dGeometry {
    /// Resolve a `conv2d` call from an NCHW input shape and an OIHW weight
    /// shape.
    ///
    /// # Errors
    ///
    /// - [`Error::RankMismatch`](crate::error::Error::RankMismatch) when
    ///   either operand is not rank 4.
    /// - [`Error::ShapeMismatch`](crate::error::Error::ShapeMismatch) when the
    ///   weight's `in_channels` axis does not match the input's channel axis.
    /// - [`Error::InvalidArg`](crate::error::Error::InvalidArg) for a zero
    ///   kernel/stride/dilation, an arithmetic overflow, a padded input
    ///   smaller than the effective kernel, or a `params.kernel` that
    ///   contradicts the weight's spatial size.
    pub(crate) fn conv2d(
        op: &'static str,
        input_dims: &[usize],
        weight_dims: &[usize],
        params: &Conv2dParams,
    ) -> Result<Conv2dGeometry> {
        let input = rank4(op, input_dims)?;
        let weight = rank4(op, weight_dims)?;
        if weight[1] != input[1] {
            return Err(Error::ShapeMismatch {
                op,
                lhs: Shape::from(input_dims.to_vec()),
                rhs: Shape::from(weight_dims.to_vec()),
            });
        }
        if params.kernel != (weight[2], weight[3]) {
            return Err(Error::InvalidArg {
                op,
                msg: format!(
                    "kernel {:?} does not match the weight's spatial size {:?}",
                    params.kernel,
                    (weight[2], weight[3])
                ),
            });
        }
        Conv2dGeometry::resolve(op, input, weight[0], weight[2], weight[3], params, false)
    }

    /// Resolve a pooling call from an NCHW input shape. The window comes from
    /// `Conv2dParams::kernel`; dilation is ignored (forced to 1), and the
    /// padding may not exceed half the window on either axis, which is what
    /// guarantees no window lies entirely inside the zero padding.
    ///
    /// # Errors
    ///
    /// As [`Conv2dGeometry::conv2d`], plus
    /// [`Error::InvalidArg`](crate::error::Error::InvalidArg) when
    /// `2 · padding > kernel` on either axis.
    pub(crate) fn pool(
        op: &'static str,
        input_dims: &[usize],
        params: &Conv2dParams,
    ) -> Result<Conv2dGeometry> {
        let input = rank4(op, input_dims)?;
        let (kernel_h, kernel_w) = params.kernel;
        Conv2dGeometry::resolve(op, input, input[1], kernel_h, kernel_w, params, true)
    }

    /// Shared body of the two constructors. `pool` selects the pooling rules
    /// (dilation forced to 1, padding capped at half the window).
    fn resolve(
        op: &'static str,
        input: [usize; 4],
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        params: &Conv2dParams,
        pool: bool,
    ) -> Result<Conv2dGeometry> {
        let (stride_h, stride_w) = params.stride;
        let (padding_h, padding_w) = params.padding;
        let (dilation_h, dilation_w) = if pool { (1, 1) } else { params.dilation };
        if pool {
            pool_padding_ok(op, "height", kernel_h, padding_h)?;
            pool_padding_ok(op, "width", kernel_w, padding_w)?;
        }
        let out_h = spatial_output_dim(
            op, "height", input[2], kernel_h, stride_h, padding_h, dilation_h,
        )?;
        let out_w = spatial_output_dim(
            op, "width", input[3], kernel_w, stride_w, padding_w, dilation_w,
        )?;
        Ok(Conv2dGeometry {
            op,
            batch: input[0],
            in_channels: input[1],
            in_h: input[2],
            in_w: input[3],
            out_channels,
            kernel_h,
            kernel_w,
            out_h,
            out_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
        })
    }

    /// The NCHW input shape.
    pub(crate) fn input_dims(&self) -> [usize; 4] {
        [self.batch, self.in_channels, self.in_h, self.in_w]
    }

    /// The OIHW weight shape (for pooling: the notional per-channel window).
    pub(crate) fn weight_dims(&self) -> [usize; 4] {
        [
            self.out_channels,
            self.in_channels,
            self.kernel_h,
            self.kernel_w,
        ]
    }

    /// The NCHW output shape.
    pub(crate) fn output_dims(&self) -> [usize; 4] {
        [self.batch, self.out_channels, self.out_h, self.out_w]
    }

    /// The window area, i.e. the `AvgPool2d` divisor.
    pub(super) fn window(&self) -> usize {
        self.kernel_h * self.kernel_w
    }

    /// Input row for output row `oh` and kernel row `kh`, or `None` when the
    /// position falls in the zero padding.
    pub(super) fn source_h(&self, oh: usize, kh: usize) -> Option<usize> {
        source(
            oh,
            kh,
            self.stride_h,
            self.dilation_h,
            self.padding_h,
            self.in_h,
        )
    }

    /// Input column for output column `ow` and kernel column `kw`, or `None`
    /// when the position falls in the zero padding.
    pub(super) fn source_w(&self, ow: usize, kw: usize) -> Option<usize> {
        source(
            ow,
            kw,
            self.stride_w,
            self.dilation_w,
            self.padding_w,
            self.in_w,
        )
    }
}

/// Assert a rank-4 shape, returning its dims as an array.
fn rank4(op: &'static str, dims: &[usize]) -> Result<[usize; 4]> {
    match dims {
        &[a, b, c, d] => Ok([a, b, c, d]),
        other => Err(Error::RankMismatch {
            op,
            expected: 4,
            got: other.len(),
        }),
    }
}

/// The output size along one spatial axis, validating the window parameters.
fn spatial_output_dim(
    op: &'static str,
    axis: &'static str,
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<usize> {
    let bad = |what: &str| Error::InvalidArg {
        op,
        msg: format!("{axis}: {what}"),
    };
    if kernel == 0 {
        return Err(bad("kernel size must be greater than 0"));
    }
    if stride == 0 {
        return Err(bad("stride must be greater than 0"));
    }
    if dilation == 0 {
        return Err(bad("dilation must be greater than 0"));
    }
    let effective = (kernel - 1)
        .checked_mul(dilation)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| bad("effective kernel size overflows usize"))?;
    let padded = padding
        .checked_mul(2)
        .and_then(|p| p.checked_add(input))
        .ok_or_else(|| bad("padded input size overflows usize"))?;
    if padded < effective {
        return Err(bad(&format!(
            "padded input {padded} is smaller than the effective kernel size {effective}"
        )));
    }
    Ok((padded - effective) / stride + 1)
}

/// Pooling rejects a padding larger than half the window: otherwise the first
/// (or last) window could lie entirely inside the zero padding, which has no
/// maximum and no meaningful average.
fn pool_padding_ok(
    op: &'static str,
    axis: &'static str,
    kernel: usize,
    padding: usize,
) -> Result<()> {
    if padding.saturating_mul(2) > kernel {
        return Err(Error::InvalidArg {
            op,
            msg: format!(
                "{axis}: padding {padding} exceeds half the pooling window {kernel}; \
                 a window would contain no input element"
            ),
        });
    }
    Ok(())
}

/// The input position a `(out_index, kernel_index)` pair reads, or `None` when
/// it lands in the zero padding (before the start or past the end).
fn source(
    out_index: usize,
    kernel_index: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
    size: usize,
) -> Option<usize> {
    // Validated geometry keeps every term far below `usize::MAX`: `out_index`
    // and `kernel_index` are bounded by the resolved output and kernel sizes,
    // whose overflow-checked combination already fit in `usize` inside
    // `spatial_output_dim`.
    let pos = out_index * stride + kernel_index * dilation;
    if pos < padding {
        return None;
    }
    let pos = pos - padding;
    (pos < size).then_some(pos)
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDENTITY: Conv2dParams = Conv2dParams {
        kernel: (2, 2),
        stride: (1, 1),
        padding: (0, 0),
        dilation: (1, 1),
    };

    fn params(
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Conv2dParams {
        Conv2dParams {
            kernel,
            stride,
            padding,
            dilation,
        }
    }

    #[test]
    fn output_size_formula() {
        let dims = [1usize, 1, 7, 7];
        let cases = [
            ((3, 3), (1, 1), (0, 0), (1, 1), 5, 5),
            ((3, 3), (2, 2), (1, 1), (1, 1), 4, 4),
            ((3, 3), (1, 1), (0, 0), (2, 2), 3, 3),
            ((1, 7), (1, 1), (0, 0), (1, 1), 7, 1),
        ];
        for (kernel, stride, padding, dilation, out_h, out_w) in cases {
            let p = params(kernel, stride, padding, dilation);
            let wdims = [2usize, 1, kernel.0, kernel.1];
            let geo = Conv2dGeometry::conv2d("conv2d", &dims, &wdims, &p).unwrap();
            assert_eq!(geo.output_dims(), [1, 2, out_h, out_w], "{kernel:?}");
        }
    }

    #[test]
    fn rank_and_channel_contracts_are_loud() {
        assert!(matches!(
            Conv2dGeometry::conv2d("conv2d", &[1, 3, 3], &[1, 1, 2, 2], &IDENTITY),
            Err(Error::RankMismatch {
                op: "conv2d",
                expected: 4,
                got: 3
            })
        ));
        assert!(matches!(
            Conv2dGeometry::conv2d("conv2d", &[1, 2, 3, 3], &[1, 1, 2, 2], &IDENTITY),
            Err(Error::ShapeMismatch { op: "conv2d", .. })
        ));
    }

    #[test]
    fn zero_window_parameters_are_invalid() {
        for p in [
            params((0, 2), (1, 1), (0, 0), (1, 1)),
            params((2, 2), (0, 1), (0, 0), (1, 1)),
            params((2, 2), (1, 1), (0, 0), (0, 1)),
        ] {
            let wdims = [1usize, 1, p.kernel.0, p.kernel.1];
            assert!(matches!(
                Conv2dGeometry::conv2d("conv2d", &[1, 1, 3, 3], &wdims, &p),
                Err(Error::InvalidArg { op: "conv2d", .. })
            ));
        }
    }

    #[test]
    fn kernel_larger_than_padded_input_is_invalid() {
        let p = params((4, 4), (1, 1), (0, 0), (1, 1));
        assert!(matches!(
            Conv2dGeometry::conv2d("conv2d", &[1, 1, 3, 3], &[1, 1, 4, 4], &p),
            Err(Error::InvalidArg { op: "conv2d", .. })
        ));
    }

    #[test]
    fn params_kernel_must_match_the_weight() {
        let p = params((3, 3), (1, 1), (0, 0), (1, 1));
        assert!(matches!(
            Conv2dGeometry::conv2d("conv2d", &[1, 1, 5, 5], &[1, 1, 2, 2], &p),
            Err(Error::InvalidArg { op: "conv2d", .. })
        ));
    }

    #[test]
    fn pool_padding_over_half_the_window_is_invalid() {
        let p = params((2, 2), (2, 2), (2, 2), (1, 1));
        assert!(matches!(
            Conv2dGeometry::pool("max_pool2d", &[1, 1, 4, 4], &p),
            Err(Error::InvalidArg {
                op: "max_pool2d",
                ..
            })
        ));
    }
}
