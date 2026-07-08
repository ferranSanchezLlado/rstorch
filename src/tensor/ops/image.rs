use super::super::autograd::{AnyTensor, raw_from_vec_like};
use super::super::{RawTensor, Tensor};
use super::ensure_same_device;
use crate::backend::{Backend, NativeBinaryOp};
use crate::dtype::{DType, FloatDType};
use crate::error::{Error, Result, ShapeError, const_check};
use crate::shape::{C, D2, D4, DimEntry, DimSpec, Shape, bind_and_check};

/// Symmetric zero-padding for NCHW rank-4 image tensors.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Padding2d {
    pub height: usize,
    pub width: usize,
}

impl Padding2d {
    pub const fn new(height: usize, width: usize) -> Self {
        Self { height, width }
    }
}

/// Runtime spatial parameters for [`Tensor::conv2d`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv2dOptions {
    pub stride_h: usize,
    pub stride_w: usize,
    pub padding_h: usize,
    pub padding_w: usize,
    pub dilation_h: usize,
    pub dilation_w: usize,
}

impl Conv2dOptions {
    pub const fn new(
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
        dilation_h: usize,
        dilation_w: usize,
    ) -> Self {
        Self {
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
        }
    }

    pub const fn with_stride(mut self, height: usize, width: usize) -> Self {
        self.stride_h = height;
        self.stride_w = width;
        self
    }

    pub const fn with_padding(mut self, height: usize, width: usize) -> Self {
        self.padding_h = height;
        self.padding_w = width;
        self
    }

    pub const fn with_dilation(mut self, height: usize, width: usize) -> Self {
        self.dilation_h = height;
        self.dilation_w = width;
        self
    }
}

impl Default for Conv2dOptions {
    fn default() -> Self {
        Self::new(1, 1, 0, 0, 1, 1)
    }
}

/// Runtime spatial parameters for rank-4 NCHW pooling.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pool2dOptions {
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub stride_h: usize,
    pub stride_w: usize,
    pub padding_h: usize,
    pub padding_w: usize,
}

impl Pool2dOptions {
    pub const fn new(kernel_h: usize, kernel_w: usize) -> Self {
        Self {
            kernel_h,
            kernel_w,
            stride_h: kernel_h,
            stride_w: kernel_w,
            padding_h: 0,
            padding_w: 0,
        }
    }

    pub const fn with_stride_padding(
        kernel_h: usize,
        kernel_w: usize,
        stride_h: usize,
        stride_w: usize,
        padding_h: usize,
        padding_w: usize,
    ) -> Self {
        Self {
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
        }
    }

    pub const fn with_stride(mut self, height: usize, width: usize) -> Self {
        self.stride_h = height;
        self.stride_w = width;
        self
    }

    pub const fn with_padding(mut self, height: usize, width: usize) -> Self {
        self.padding_h = height;
        self.padding_w = width;
        self
    }
}

impl Default for Pool2dOptions {
    fn default() -> Self {
        Self::new(1, 1)
    }
}

impl<A, BDim, Cc, Dd, E, BackendT> Tensor<D4<A, BDim, Cc, Dd>, E, BackendT>
where
    A: DimSpec,
    BDim: DimSpec,
    Cc: DimSpec,
    Dd: DimSpec,
    E: DType,
    BackendT: Backend<E>,
{
    pub fn transpose_middle2(&self) -> Result<Tensor<D4<A, Cc, BDim, Dd>, E, BackendT>> {
        let raw = self
            .raw()
            .view_with_layout(self.layout().transpose_axes(1, 2)?)?;
        Tensor::<D4<A, Cc, BDim, Dd>, E, BackendT>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self)],
            |grad| {
                Ok(vec![Some(
                    grad.view_with_layout(grad.layout().transpose_axes(1, 2)?)?,
                )])
            },
        )
    }

    /// Pads an NCHW image tensor with zeros on the height and width axes.
    pub fn pad2d<const OUT_H: usize, const OUT_W: usize>(
        &self,
        padding: Padding2d,
    ) -> Result<Tensor<D4<A, BDim, C<OUT_H>, C<OUT_W>>, E, BackendT>> {
        let dims = self.shape().dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let expected_h = padded_dim("pad2d", height, padding.height)?;
        let expected_w = padded_dim("pad2d", width, padding.width)?;
        if expected_h != OUT_H {
            return Err(ShapeError::DimMismatch {
                op: "pad2d",
                operand: 1,
                axis: 2,
                expected: expected_h,
                found: OUT_H,
            }
            .into());
        }
        if expected_w != OUT_W {
            return Err(ShapeError::DimMismatch {
                op: "pad2d",
                operand: 1,
                axis: 3,
                expected: expected_w,
                found: OUT_W,
            }
            .into());
        }

        let input = self.host_values()?;
        let mut values = vec![E::ZERO; batch * channels * OUT_H * OUT_W];
        for n in 0..batch {
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let src = nchw_index(n, c, h, w, channels, height, width);
                        let dst = nchw_index(
                            n,
                            c,
                            h + padding.height,
                            w + padding.width,
                            channels,
                            OUT_H,
                            OUT_W,
                        );
                        values[dst] = input[src];
                    }
                }
            }
        }
        let raw = RawTensor::from_vec_on(
            self.device().clone(),
            values,
            Shape::known([batch, channels, OUT_H, OUT_W]),
        )?;
        let input_raw = self.raw().clone();
        Tensor::<D4<A, BDim, C<OUT_H>, C<OUT_W>>, E, BackendT>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self)],
            move |grad| {
                let grad = grad.host_values()?;
                let mut input_grad = vec![E::ZERO; batch * channels * height * width];
                for n in 0..batch {
                    for c in 0..channels {
                        for h in 0..height {
                            for w in 0..width {
                                let src = nchw_index(
                                    n,
                                    c,
                                    h + padding.height,
                                    w + padding.width,
                                    channels,
                                    OUT_H,
                                    OUT_W,
                                );
                                let dst = nchw_index(n, c, h, w, channels, height, width);
                                input_grad[dst] = grad[src];
                            }
                        }
                    }
                }
                Ok(vec![Some(raw_from_vec_like(&input_raw, input_grad)?)])
            },
        )
    }
}

impl<Batch, Channels, Height, Width, E, B> Tensor<D4<Batch, Channels, Height, Width>, E, B>
where
    Batch: DimSpec,
    Channels: DimSpec,
    Height: DimSpec,
    Width: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    /// Flattens the NCHW `[channels, height, width]` tail into a dense feature axis.
    pub fn flatten_spatial<const OUT: usize>(&self) -> Result<Tensor<D2<Batch, C<OUT>>, E, B>> {
        const {
            const_check::known_size_eq(
                crate::shape::known_numel([Channels::KNOWN, Height::KNOWN, Width::KNOWN]),
                Some(OUT),
                "flatten_spatial",
                "channels * height * width",
                "OUT",
            );
        };

        let dims = self.shape().dims();
        let batch = dims[0];
        let features = checked_product("flatten_spatial", &[dims[1], dims[2], dims[3]])?;
        if features != OUT {
            return Err(ShapeError::LengthMismatch {
                op: "flatten_spatial",
                expected: OUT,
                found: features,
            }
            .into());
        }
        self.reshape_with_shape::<D2<Batch, C<OUT>>>([batch, OUT])
    }

    /// Adds a channel-shaped tensor across the batch and spatial axes.
    pub fn add_channel_dim(&self, rhs: &Tensor<D1<Channels>, E, B>) -> Result<Self> {
        ensure_same_device::<E, B>(self.device(), rhs.device(), "add_channel_dim")?;
        let dims = self.shape().dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        let rhs_channels = rhs.shape().dims()[0];
        bind_and_check(
            "add_channel_dim",
            [
                (DimEntry::of::<Channels>(0, 1), channels),
                (DimEntry::of::<Channels>(1, 0), rhs_channels),
            ],
        )?;
        if rhs_channels != channels {
            return Err(ShapeError::DimMismatch {
                op: "add_channel_dim",
                operand: 1,
                axis: 0,
                expected: channels,
                found: rhs_channels,
            }
            .into());
        }

        let lhs_input = self.contiguous()?;
        let rhs_input = rhs.contiguous()?;
        let raw = if let Some(storage) = B::try_broadcast_channel(
            lhs_input.device(),
            lhs_input.raw().storage(),
            rhs_input.raw().storage(),
            batch,
            channels,
            height,
            width,
            NativeBinaryOp::Add,
        )
        .map_err(Error::backend)?
        {
            RawTensor::from_storage_on(self.device().clone(), storage, self.shape().clone())?
        } else {
            crate::backend::record_reference_fall("broadcast_channel");
            let lhs_values = lhs_input.host_values()?;
            let rhs_values = rhs_input.host_values()?;
            let mut values = Vec::with_capacity(lhs_values.len());
            for n in 0..batch {
                for (c, &channel_value) in rhs_values.iter().enumerate().take(channels) {
                    for h in 0..height {
                        for w in 0..width {
                            let idx = nchw_index(n, c, h, w, channels, height, width);
                            values.push(lhs_values[idx] + channel_value);
                        }
                    }
                }
            }
            RawTensor::from_vec_on(self.device().clone(), values, self.shape().clone())?
        };
        let lhs_raw = self.raw().clone();
        let rhs_raw = rhs.raw().clone();
        Self::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(rhs)],
            move |grad| {
                let grad_values = grad.host_values()?.into_owned();
                let mut rhs_grad = vec![E::ZERO; channels];
                for n in 0..batch {
                    for (c, channel_grad) in rhs_grad.iter_mut().enumerate().take(channels) {
                        for h in 0..height {
                            for w in 0..width {
                                *channel_grad +=
                                    grad_values[nchw_index(n, c, h, w, channels, height, width)];
                            }
                        }
                    }
                }
                Ok(vec![
                    Some(raw_from_vec_like(&lhs_raw, grad_values)?),
                    Some(raw_from_vec_like(&rhs_raw, rhs_grad)?),
                ])
            },
        )
    }

    /// Computes 2D convolution over NCHW tensors using explicit output shape markers.
    pub fn conv2d<OutChannels, KernelH, KernelW, const OUT_H: usize, const OUT_W: usize>(
        &self,
        weight: &Tensor<D4<OutChannels, Channels, KernelH, KernelW>, E, B>,
        options: Conv2dOptions,
    ) -> Result<Tensor<D4<Batch, OutChannels, C<OUT_H>, C<OUT_W>>, E, B>>
    where
        OutChannels: DimSpec,
        KernelH: DimSpec,
        KernelW: DimSpec,
    {
        ensure_same_device::<E, B>(self.device(), weight.device(), "conv2d")?;
        let input_dims = self.shape().dims();
        let weight_dims = weight.shape().dims();
        let batch = input_dims[0];
        let in_channels = input_dims[1];
        let height = input_dims[2];
        let width = input_dims[3];
        let out_channels = weight_dims[0];
        let weight_in_channels = weight_dims[1];
        let kernel_h = weight_dims[2];
        let kernel_w = weight_dims[3];
        bind_and_check(
            "conv2d",
            [
                (DimEntry::of::<Batch>(0, 0), batch),
                (DimEntry::of::<Channels>(0, 1), in_channels),
                (DimEntry::of::<Height>(0, 2), height),
                (DimEntry::of::<Width>(0, 3), width),
                (DimEntry::of::<OutChannels>(1, 0), out_channels),
                (DimEntry::of::<Channels>(1, 1), weight_in_channels),
                (DimEntry::of::<KernelH>(1, 2), kernel_h),
                (DimEntry::of::<KernelW>(1, 3), kernel_w),
            ],
        )?;
        if weight_in_channels != in_channels {
            return Err(ShapeError::DimMismatch {
                op: "conv2d",
                operand: 1,
                axis: 1,
                expected: in_channels,
                found: weight_in_channels,
            }
            .into());
        }
        validate_spatial_output(
            "conv2d",
            height,
            width,
            kernel_h,
            kernel_w,
            OUT_H,
            OUT_W,
            options.stride_h,
            options.stride_w,
            options.padding_h,
            options.padding_w,
            options.dilation_h,
            options.dilation_w,
        )?;

        let input_values = self.host_values()?.into_owned();
        let weight_values = weight.host_values()?.into_owned();
        let values = conv2d_values(
            &input_values,
            &weight_values,
            batch,
            in_channels,
            height,
            width,
            out_channels,
            kernel_h,
            kernel_w,
            OUT_H,
            OUT_W,
            options,
        )?;
        let raw = RawTensor::from_vec_on(
            self.device().clone(),
            values,
            Shape::known([batch, out_channels, OUT_H, OUT_W]),
        )?;
        let input_raw = self.raw().clone();
        let weight_raw = weight.raw().clone();
        Tensor::<D4<Batch, OutChannels, C<OUT_H>, C<OUT_W>>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self), AnyTensor::from_shape(weight)],
            move |grad| {
                let (input_grad, weight_grad) = conv2d_backward_values(
                    &grad.host_values()?,
                    &input_values,
                    &weight_values,
                    batch,
                    in_channels,
                    height,
                    width,
                    out_channels,
                    kernel_h,
                    kernel_w,
                    OUT_H,
                    OUT_W,
                    options,
                )?;
                Ok(vec![
                    Some(raw_from_vec_like(&input_raw, input_grad)?),
                    Some(raw_from_vec_like(&weight_raw, weight_grad)?),
                ])
            },
        )
    }

    /// Max-pools NCHW image tensors. Tied maxima split gradient evenly.
    pub fn max_pool2d<const OUT_H: usize, const OUT_W: usize>(
        &self,
        options: Pool2dOptions,
    ) -> Result<Tensor<D4<Batch, Channels, C<OUT_H>, C<OUT_W>>, E, B>> {
        let dims = self.shape().dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        validate_pool_output("max_pool2d", height, width, OUT_H, OUT_W, options)?;

        let input_values = self.host_values()?.into_owned();
        let mut values = vec![E::ZERO; batch * channels * OUT_H * OUT_W];
        for n in 0..batch {
            for c in 0..channels {
                for oh in 0..OUT_H {
                    for ow in 0..OUT_W {
                        let mut best = None;
                        for kh in 0..options.kernel_h {
                            for kw in 0..options.kernel_w {
                                let Some(ih) = source_index(
                                    oh,
                                    kh,
                                    options.stride_h,
                                    options.padding_h,
                                    1,
                                    height,
                                )?
                                else {
                                    continue;
                                };
                                let Some(iw) = source_index(
                                    ow,
                                    kw,
                                    options.stride_w,
                                    options.padding_w,
                                    1,
                                    width,
                                )?
                                else {
                                    continue;
                                };
                                let value =
                                    input_values[nchw_index(n, c, ih, iw, channels, height, width)];
                                if match best {
                                    Some(best) => value > best,
                                    None => true,
                                } {
                                    best = Some(value);
                                }
                            }
                        }
                        let value = best.ok_or(ShapeError::InvalidSpatialParam {
                            op: "max_pool2d",
                            param: "padding",
                            value: options.padding_h.max(options.padding_w),
                            reason: "pooling window contains no input elements",
                        })?;
                        values[nchw_index(n, c, oh, ow, channels, OUT_H, OUT_W)] = value;
                    }
                }
            }
        }
        let raw = RawTensor::from_vec_on(
            self.device().clone(),
            values.clone(),
            Shape::known([batch, channels, OUT_H, OUT_W]),
        )?;
        let input_raw = self.raw().clone();
        Tensor::<D4<Batch, Channels, C<OUT_H>, C<OUT_W>>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self)],
            move |grad| {
                let grad = grad.host_values()?;
                let mut input_grad = vec![E::ZERO; batch * channels * height * width];
                for n in 0..batch {
                    for c in 0..channels {
                        for oh in 0..OUT_H {
                            for ow in 0..OUT_W {
                                let out_idx = nchw_index(n, c, oh, ow, channels, OUT_H, OUT_W);
                                let max = values[out_idx];
                                let mut ties = 0usize;
                                for kh in 0..options.kernel_h {
                                    for kw in 0..options.kernel_w {
                                        let Some(ih) = source_index(
                                            oh,
                                            kh,
                                            options.stride_h,
                                            options.padding_h,
                                            1,
                                            height,
                                        )?
                                        else {
                                            continue;
                                        };
                                        let Some(iw) = source_index(
                                            ow,
                                            kw,
                                            options.stride_w,
                                            options.padding_w,
                                            1,
                                            width,
                                        )?
                                        else {
                                            continue;
                                        };
                                        if input_values
                                            [nchw_index(n, c, ih, iw, channels, height, width)]
                                            == max
                                        {
                                            ties += 1;
                                        }
                                    }
                                }
                                let each = grad[out_idx] / E::from_usize(ties);
                                for kh in 0..options.kernel_h {
                                    for kw in 0..options.kernel_w {
                                        let Some(ih) = source_index(
                                            oh,
                                            kh,
                                            options.stride_h,
                                            options.padding_h,
                                            1,
                                            height,
                                        )?
                                        else {
                                            continue;
                                        };
                                        let Some(iw) = source_index(
                                            ow,
                                            kw,
                                            options.stride_w,
                                            options.padding_w,
                                            1,
                                            width,
                                        )?
                                        else {
                                            continue;
                                        };
                                        let input_idx =
                                            nchw_index(n, c, ih, iw, channels, height, width);
                                        if input_values[input_idx] == max {
                                            input_grad[input_idx] += each;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(vec![Some(raw_from_vec_like(&input_raw, input_grad)?)])
            },
        )
    }

    /// Average-pools NCHW image tensors over valid input positions only.
    pub fn avg_pool2d<const OUT_H: usize, const OUT_W: usize>(
        &self,
        options: Pool2dOptions,
    ) -> Result<Tensor<D4<Batch, Channels, C<OUT_H>, C<OUT_W>>, E, B>> {
        let dims = self.shape().dims();
        let batch = dims[0];
        let channels = dims[1];
        let height = dims[2];
        let width = dims[3];
        validate_pool_output("avg_pool2d", height, width, OUT_H, OUT_W, options)?;

        let input_values = self.host_values()?.into_owned();
        let mut counts = vec![0usize; batch * channels * OUT_H * OUT_W];
        let mut values = vec![<E::Acc as DType>::ZERO; batch * channels * OUT_H * OUT_W];
        for n in 0..batch {
            for c in 0..channels {
                for oh in 0..OUT_H {
                    for ow in 0..OUT_W {
                        let out_idx = nchw_index(n, c, oh, ow, channels, OUT_H, OUT_W);
                        for kh in 0..options.kernel_h {
                            for kw in 0..options.kernel_w {
                                let Some(ih) = source_index(
                                    oh,
                                    kh,
                                    options.stride_h,
                                    options.padding_h,
                                    1,
                                    height,
                                )?
                                else {
                                    continue;
                                };
                                let Some(iw) = source_index(
                                    ow,
                                    kw,
                                    options.stride_w,
                                    options.padding_w,
                                    1,
                                    width,
                                )?
                                else {
                                    continue;
                                };
                                values[out_idx] += E::Acc::from_f64(
                                    input_values[nchw_index(n, c, ih, iw, channels, height, width)]
                                        .to_f64(),
                                );
                                counts[out_idx] += 1;
                            }
                        }
                        if counts[out_idx] == 0 {
                            return Err(ShapeError::InvalidSpatialParam {
                                op: "avg_pool2d",
                                param: "padding",
                                value: options.padding_h.max(options.padding_w),
                                reason: "pooling window contains no input elements",
                            }
                            .into());
                        }
                        values[out_idx] /= E::Acc::from_usize(counts[out_idx]);
                    }
                }
            }
        }
        let values = values
            .into_iter()
            .map(|value| E::from_f64(value.to_f64()))
            .collect();
        let raw = RawTensor::from_vec_on(
            self.device().clone(),
            values,
            Shape::known([batch, channels, OUT_H, OUT_W]),
        )?;
        let input_raw = self.raw().clone();
        Tensor::<D4<Batch, Channels, C<OUT_H>, C<OUT_W>>, E, B>::autograd_output(
            raw,
            vec![AnyTensor::from_shape(self)],
            move |grad| {
                let grad = grad.host_values()?;
                let mut input_grad = vec![E::ZERO; batch * channels * height * width];
                for n in 0..batch {
                    for c in 0..channels {
                        for oh in 0..OUT_H {
                            for ow in 0..OUT_W {
                                let out_idx = nchw_index(n, c, oh, ow, channels, OUT_H, OUT_W);
                                let each = grad[out_idx] / E::from_usize(counts[out_idx]);
                                for kh in 0..options.kernel_h {
                                    for kw in 0..options.kernel_w {
                                        let Some(ih) = source_index(
                                            oh,
                                            kh,
                                            options.stride_h,
                                            options.padding_h,
                                            1,
                                            height,
                                        )?
                                        else {
                                            continue;
                                        };
                                        let Some(iw) = source_index(
                                            ow,
                                            kw,
                                            options.stride_w,
                                            options.padding_w,
                                            1,
                                            width,
                                        )?
                                        else {
                                            continue;
                                        };
                                        input_grad
                                            [nchw_index(n, c, ih, iw, channels, height, width)] +=
                                            each;
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(vec![Some(raw_from_vec_like(&input_raw, input_grad)?)])
            },
        )
    }
}

use crate::shape::D1;

impl<Batch, Channels, Height, Width, E, B> Tensor<D4<Batch, Channels, Height, Width>, E, B>
where
    Batch: DimSpec,
    Channels: DimSpec,
    Height: DimSpec,
    Width: DimSpec,
    E: FloatDType,
    B: Backend<E>,
{
    /// Batch normalization forward pass (training mode).
    ///
    /// Returns `(output, batch_mean, batch_var)`. The caller is responsible for
    /// updating running statistics with the returned per-channel statistics.
    pub fn batch_norm2d(
        &self,
        weight: &Tensor<D1<Channels>, E, B>,
        bias: &Tensor<D1<Channels>, E, B>,
        eps: f64,
        _momentum: f64,
    ) -> Result<(Self, Vec<E>, Vec<E>)> {
        ensure_same_device::<E, B>(self.device(), weight.device(), "batch_norm2d")?;
        ensure_same_device::<E, B>(self.device(), bias.device(), "batch_norm2d")?;
        let dims = self.shape().dims();
        let n = dims[0];
        let ch = dims[1];
        let h = dims[2];
        let w = dims[3];
        let items = n * h * w;

        let input_values = self.host_values()?.into_owned();
        let weight_values = weight.host_values()?.into_owned();
        let bias_values = bias.host_values()?.into_owned();

        let mut mean_f64 = vec![0f64; ch];
        let mut var_f64 = vec![0f64; ch];
        for c in 0..ch {
            let mut sum = 0f64;
            for ni in 0..n {
                for hi in 0..h {
                    for wi in 0..w {
                        sum += input_values[nchw_index(ni, c, hi, wi, ch, h, w)].to_f64();
                    }
                }
            }
            mean_f64[c] = sum / items as f64;
            let mut sq_sum = 0f64;
            for ni in 0..n {
                for hi in 0..h {
                    for wi in 0..w {
                        let diff = input_values[nchw_index(ni, c, hi, wi, ch, h, w)].to_f64()
                            - mean_f64[c];
                        sq_sum += diff * diff;
                    }
                }
            }
            var_f64[c] = sq_sum / items as f64;
        }

        let std_vals: Vec<f64> = var_f64.iter().map(|&v| (v + eps).sqrt()).collect();
        let total = n * ch * h * w;
        let mut x_hat = vec![0f64; total];
        let mut out_values = Vec::with_capacity(total);
        for ni in 0..n {
            for c in 0..ch {
                for hi in 0..h {
                    for wi in 0..w {
                        let idx = nchw_index(ni, c, hi, wi, ch, h, w);
                        let xh = (input_values[idx].to_f64() - mean_f64[c]) / std_vals[c];
                        x_hat[idx] = xh;
                        out_values.push(E::from_f64(
                            xh * weight_values[c].to_f64() + bias_values[c].to_f64(),
                        ));
                    }
                }
            }
        }

        let batch_mean: Vec<E> = mean_f64.iter().map(|&m| E::from_f64(m)).collect();
        let batch_var: Vec<E> = var_f64.iter().map(|&v| E::from_f64(v)).collect();

        let raw = RawTensor::from_vec_on(self.device().clone(), out_values, self.shape().clone())?;
        let input_raw = self.raw().clone();
        let weight_raw = weight.raw().clone();
        let bias_raw = bias.raw().clone();
        let weight_values_cap = weight_values;
        let x_hat_cap = x_hat;
        let std_vals_cap = std_vals;

        let result = Self::autograd_output(
            raw,
            vec![
                AnyTensor::from_shape(self),
                AnyTensor::from_shape(weight),
                AnyTensor::from_shape(bias),
            ],
            move |grad| {
                let dout = grad.host_values()?;
                let mut dx = vec![E::ZERO; n * ch * h * w];
                let mut dweight = vec![<E::Acc as DType>::ZERO; ch];
                let mut dbias = vec![<E::Acc as DType>::ZERO; ch];
                for c in 0..ch {
                    let mut sum_dxhat = 0f64;
                    let mut sum_dxhat_xhat = 0f64;
                    for ni in 0..n {
                        for hi in 0..h {
                            for wi in 0..w {
                                let idx = nchw_index(ni, c, hi, wi, ch, h, w);
                                let dxh = dout[idx].to_f64() * weight_values_cap[c].to_f64();
                                sum_dxhat += dxh;
                                sum_dxhat_xhat += dxh * x_hat_cap[idx];
                                dweight[c] += E::Acc::from_f64(dout[idx].to_f64() * x_hat_cap[idx]);
                                dbias[c] += E::Acc::from_f64(dout[idx].to_f64());
                            }
                        }
                    }
                    let m = items as f64;
                    let inv_std = 1.0 / std_vals_cap[c];
                    for ni in 0..n {
                        for hi in 0..h {
                            for wi in 0..w {
                                let idx = nchw_index(ni, c, hi, wi, ch, h, w);
                                let dxh = dout[idx].to_f64() * weight_values_cap[c].to_f64();
                                let val = inv_std
                                    * (dxh - sum_dxhat / m - x_hat_cap[idx] * sum_dxhat_xhat / m);
                                dx[idx] = E::from_f64(val);
                            }
                        }
                    }
                }
                let dweight_f: Vec<E> = dweight
                    .into_iter()
                    .map(|v| E::from_f64(v.to_f64()))
                    .collect();
                let dbias_f: Vec<E> = dbias.into_iter().map(|v| E::from_f64(v.to_f64())).collect();
                Ok(vec![
                    Some(raw_from_vec_like(&input_raw, dx)?),
                    Some(raw_from_vec_like(&weight_raw, dweight_f)?),
                    Some(raw_from_vec_like(&bias_raw, dbias_f)?),
                ])
            },
        )?;

        Ok((result, batch_mean, batch_var))
    }

    /// Batch normalization inference pass using stored running statistics.
    pub fn batch_norm2d_eval(
        &self,
        weight: &Tensor<D1<Channels>, E, B>,
        bias: &Tensor<D1<Channels>, E, B>,
        running_mean: &[E],
        running_var: &[E],
        eps: f64,
    ) -> Result<Self> {
        ensure_same_device::<E, B>(self.device(), weight.device(), "batch_norm2d_eval")?;
        ensure_same_device::<E, B>(self.device(), bias.device(), "batch_norm2d_eval")?;
        let dims = self.shape().dims();
        let n = dims[0];
        let ch = dims[1];
        let h = dims[2];
        let w = dims[3];

        let input_values = self.host_values()?.into_owned();
        let weight_values = weight.host_values()?.into_owned();
        let bias_values = bias.host_values()?.into_owned();

        let mut out_values = Vec::with_capacity(n * ch * h * w);
        for ni in 0..n {
            for c in 0..ch {
                let std = (running_var[c].to_f64() + eps).sqrt();
                for hi in 0..h {
                    for wi in 0..w {
                        let idx = nchw_index(ni, c, hi, wi, ch, h, w);
                        let xh = (input_values[idx].to_f64() - running_mean[c].to_f64()) / std;
                        out_values.push(E::from_f64(
                            xh * weight_values[c].to_f64() + bias_values[c].to_f64(),
                        ));
                    }
                }
            }
        }

        let raw = RawTensor::from_vec_on(
            self.device().clone(),
            out_values,
            Shape::known([n, ch, h, w]),
        )?;
        let input_raw = self.raw().clone();
        let weight_raw = weight.raw().clone();
        let bias_raw = bias.raw().clone();
        let running_mean_cap: Vec<E> = running_mean.to_vec();
        let running_var_cap: Vec<E> = running_var.to_vec();

        Self::autograd_output(
            raw,
            vec![
                AnyTensor::from_shape(self),
                AnyTensor::from_shape(weight),
                AnyTensor::from_shape(bias),
            ],
            move |grad| {
                let dout = grad.host_values()?;
                let mut dx = vec![E::ZERO; n * ch * h * w];
                let mut dweight = vec![<E::Acc as DType>::ZERO; ch];
                let mut dbias = vec![<E::Acc as DType>::ZERO; ch];
                for ni in 0..n {
                    for c in 0..ch {
                        let std = (running_var_cap[c].to_f64() + eps).sqrt();
                        let inv_std = 1.0 / std;
                        let xh_mean = running_mean_cap[c].to_f64();
                        for hi in 0..h {
                            for wi in 0..w {
                                let idx = nchw_index(ni, c, hi, wi, ch, h, w);
                                let xh = (input_values[idx].to_f64() - xh_mean) / std;
                                dweight[c] += E::Acc::from_f64(dout[idx].to_f64() * xh);
                                dbias[c] += E::Acc::from_f64(dout[idx].to_f64());
                                dx[idx] = E::from_f64(
                                    dout[idx].to_f64() * weight_values[c].to_f64() * inv_std,
                                );
                            }
                        }
                    }
                }
                let dweight_f: Vec<E> = dweight
                    .into_iter()
                    .map(|v| E::from_f64(v.to_f64()))
                    .collect();
                let dbias_f: Vec<E> = dbias.into_iter().map(|v| E::from_f64(v.to_f64())).collect();
                Ok(vec![
                    Some(raw_from_vec_like(&input_raw, dx)?),
                    Some(raw_from_vec_like(&weight_raw, dweight_f)?),
                    Some(raw_from_vec_like(&bias_raw, dbias_f)?),
                ])
            },
        )
    }
}

fn padded_dim(op: &'static str, input: usize, padding: usize) -> Result<usize> {
    let doubled = padding
        .checked_mul(2)
        .ok_or(ShapeError::InvalidSpatialParam {
            op,
            param: "padding",
            value: padding,
            reason: "padding overflow",
        })?;
    input.checked_add(doubled).ok_or_else(|| {
        ShapeError::InvalidSpatialParam {
            op,
            param: "padding",
            value: padding,
            reason: "padded dimension overflow",
        }
        .into()
    })
}

fn checked_product(op: &'static str, dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            ShapeError::InvalidSpatialParam {
                op,
                param: "shape",
                value: dim,
                reason: "spatial element count overflow",
            }
            .into()
        })
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_spatial_output(
    op: &'static str,
    input_h: usize,
    input_w: usize,
    kernel_h: usize,
    kernel_w: usize,
    out_h: usize,
    out_w: usize,
    stride_h: usize,
    stride_w: usize,
    padding_h: usize,
    padding_w: usize,
    dilation_h: usize,
    dilation_w: usize,
) -> Result<()> {
    let expected_h = spatial_output_dim(op, input_h, kernel_h, stride_h, padding_h, dilation_h)?;
    let expected_w = spatial_output_dim(op, input_w, kernel_w, stride_w, padding_w, dilation_w)?;
    if expected_h != out_h {
        return Err(ShapeError::DimMismatch {
            op,
            operand: 2,
            axis: 2,
            expected: expected_h,
            found: out_h,
        }
        .into());
    }
    if expected_w != out_w {
        return Err(ShapeError::DimMismatch {
            op,
            operand: 2,
            axis: 3,
            expected: expected_w,
            found: out_w,
        }
        .into());
    }
    Ok(())
}

fn validate_pool_output(
    op: &'static str,
    input_h: usize,
    input_w: usize,
    out_h: usize,
    out_w: usize,
    options: Pool2dOptions,
) -> Result<()> {
    validate_spatial_output(
        op,
        input_h,
        input_w,
        options.kernel_h,
        options.kernel_w,
        out_h,
        out_w,
        options.stride_h,
        options.stride_w,
        options.padding_h,
        options.padding_w,
        1,
        1,
    )
}

fn spatial_output_dim(
    op: &'static str,
    input: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<usize> {
    if kernel == 0 {
        return Err(ShapeError::InvalidSpatialParam {
            op,
            param: "kernel",
            value: kernel,
            reason: "kernel size must be greater than 0",
        }
        .into());
    }
    if stride == 0 {
        return Err(ShapeError::InvalidSpatialParam {
            op,
            param: "stride",
            value: stride,
            reason: "stride must be greater than 0",
        }
        .into());
    }
    if dilation == 0 {
        return Err(ShapeError::InvalidSpatialParam {
            op,
            param: "dilation",
            value: dilation,
            reason: "dilation must be greater than 0",
        }
        .into());
    }
    let effective_kernel = (kernel - 1)
        .checked_mul(dilation)
        .and_then(|value| value.checked_add(1))
        .ok_or(ShapeError::InvalidSpatialParam {
            op,
            param: "dilation",
            value: dilation,
            reason: "effective kernel size overflow",
        })?;
    let padded = padded_dim(op, input, padding)?;
    if padded < effective_kernel {
        return Err(ShapeError::InvalidSpatialParam {
            op,
            param: "padding",
            value: padding,
            reason: "padded input is smaller than effective kernel",
        }
        .into());
    }
    Ok((padded - effective_kernel) / stride + 1)
}

fn source_index(
    out_index: usize,
    kernel_index: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    input_size: usize,
) -> Result<Option<usize>> {
    let source = out_index
        .checked_mul(stride)
        .and_then(|value| {
            kernel_index
                .checked_mul(dilation)
                .and_then(|k| value.checked_add(k))
        })
        .ok_or(ShapeError::InvalidSpatialParam {
            op: "spatial_window",
            param: "index",
            value: out_index,
            reason: "window index overflow",
        })?;
    if source < padding {
        return Ok(None);
    }
    let source = source - padding;
    if source < input_size {
        Ok(Some(source))
    } else {
        Ok(None)
    }
}

fn nchw_index(
    batch: usize,
    channel: usize,
    height: usize,
    width: usize,
    channels: usize,
    height_size: usize,
    width_size: usize,
) -> usize {
    ((batch * channels + channel) * height_size + height) * width_size + width
}

fn oihw_index(
    out_channel: usize,
    in_channel: usize,
    height: usize,
    width: usize,
    in_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
) -> usize {
    ((out_channel * in_channels + in_channel) * kernel_h + height) * kernel_w + width
}

#[allow(clippy::too_many_arguments)]
fn conv2d_values<E: FloatDType>(
    input: &[E],
    weight: &[E],
    batch: usize,
    in_channels: usize,
    height: usize,
    width: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    out_h: usize,
    out_w: usize,
    options: Conv2dOptions,
) -> Result<Vec<E>> {
    let mut out = vec![E::ZERO; batch * out_channels * out_h * out_w];
    for n in 0..batch {
        for oc in 0..out_channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = <E::Acc as DType>::ZERO;
                    for ic in 0..in_channels {
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let Some(ih) = source_index(
                                    oh,
                                    kh,
                                    options.stride_h,
                                    options.padding_h,
                                    options.dilation_h,
                                    height,
                                )?
                                else {
                                    continue;
                                };
                                let Some(iw) = source_index(
                                    ow,
                                    kw,
                                    options.stride_w,
                                    options.padding_w,
                                    options.dilation_w,
                                    width,
                                )?
                                else {
                                    continue;
                                };
                                acc += E::Acc::from_f64(
                                    (input[nchw_index(n, ic, ih, iw, in_channels, height, width)]
                                        * weight[oihw_index(
                                            oc,
                                            ic,
                                            kh,
                                            kw,
                                            in_channels,
                                            kernel_h,
                                            kernel_w,
                                        )])
                                    .to_f64(),
                                );
                            }
                        }
                    }
                    out[nchw_index(n, oc, oh, ow, out_channels, out_h, out_w)] =
                        E::from_f64(acc.to_f64());
                }
            }
        }
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn conv2d_backward_values<E: FloatDType>(
    grad: &[E],
    input: &[E],
    weight: &[E],
    batch: usize,
    in_channels: usize,
    height: usize,
    width: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    out_h: usize,
    out_w: usize,
    options: Conv2dOptions,
) -> Result<(Vec<E>, Vec<E>)> {
    let mut input_grad = vec![<E::Acc as DType>::ZERO; batch * in_channels * height * width];
    let mut weight_grad =
        vec![<E::Acc as DType>::ZERO; out_channels * in_channels * kernel_h * kernel_w];
    for n in 0..batch {
        for oc in 0..out_channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let g = grad[nchw_index(n, oc, oh, ow, out_channels, out_h, out_w)];
                    for ic in 0..in_channels {
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let Some(ih) = source_index(
                                    oh,
                                    kh,
                                    options.stride_h,
                                    options.padding_h,
                                    options.dilation_h,
                                    height,
                                )?
                                else {
                                    continue;
                                };
                                let Some(iw) = source_index(
                                    ow,
                                    kw,
                                    options.stride_w,
                                    options.padding_w,
                                    options.dilation_w,
                                    width,
                                )?
                                else {
                                    continue;
                                };
                                let input_idx =
                                    nchw_index(n, ic, ih, iw, in_channels, height, width);
                                let weight_idx =
                                    oihw_index(oc, ic, kh, kw, in_channels, kernel_h, kernel_w);
                                input_grad[input_idx] +=
                                    E::Acc::from_f64((g * weight[weight_idx]).to_f64());
                                weight_grad[weight_idx] +=
                                    E::Acc::from_f64((g * input[input_idx]).to_f64());
                            }
                        }
                    }
                }
            }
        }
    }
    Ok((
        input_grad
            .into_iter()
            .map(|value| E::from_f64(value.to_f64()))
            .collect(),
        weight_grad
            .into_iter()
            .map(|value| E::from_f64(value.to_f64()))
            .collect(),
    ))
}
