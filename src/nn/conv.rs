//! [`Conv2d`] — the 2-D convolution layer over [`Tensor::conv2d`].

use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::nn::{Forward, Mode, Param, init};
use crate::rng::Rng;
use crate::tensor::Tensor;

/// A 2-D convolution over an NCHW image, wrapping [`Tensor::conv2d`] with
/// weight/bias [`Param`]s.
///
/// Parameter paths are part of the persistence contract: `weight`, shape
/// `[out_channels, in_channels, kernel_h, kernel_w]`, and, when present,
/// `bias`, shape `[out_channels]` — the same rank-1 spelling
/// [`Linear`](crate::nn::Linear) and [`BatchNorm2d`](crate::nn::BatchNorm2d)
/// use, and the one `PyTorch` writes, so a state dict moves between the two
/// without a reshape. [`forward`](Conv2d::forward) reshapes it to
/// `[out_channels, 1, 1]` to broadcast over the spatial axes; that is a view,
/// not a copy.
///
/// # Initialization
///
/// The weight is Kaiming-uniform for a `ReLU` nonlinearity — the same
/// `gain = √2` convention [`Linear`](crate::nn::Linear) uses, via
/// [`nn::init::kaiming_uniform`](crate::nn::init::kaiming_uniform) — and the
/// bias starts at zero, in the weight's own dtype and on the weight's own
/// device, so a [`with_init`](Conv2d::with_init) closure that returns an
/// `F64` or off-device weight still yields a layer that can run.
/// [`new`](Conv2d::new) always produces [`F32`](crate::DType::F32)
/// parameters; convert afterwards with
/// [`ModuleExt::to_dtype`](crate::nn::ModuleExt::to_dtype), or use
/// [`with_init`](Conv2d::with_init) for any other scheme, including any other
/// weight dtype.
///
/// ```
/// use rstorch::nn::{Conv2d, Forward, Mode};
/// use rstorch::{DType, Device, Rng, Tensor};
///
/// let dev = Device::Cpu;
/// let mut rng = Rng::seed(0);
/// let mut conv = Conv2d::new(1, 4, (3, 3), &dev, &mut rng)?.with_padding((1, 1));
/// let x = Tensor::zeros([2, 1, 8, 8], DType::F32, &dev)?;
/// assert_eq!(conv.forward(&x, Mode::EVAL)?.dims(), &[2, 4, 8, 8]);
/// # Ok::<(), rstorch::Error>(())
/// ```
#[derive(rstorch::Module)]
pub struct Conv2d {
    weight: Param,
    bias: Option<Param>,
    #[module(skip)]
    stride: (usize, usize),
    #[module(skip)]
    padding: (usize, usize),
    #[module(skip)]
    dilation: (usize, usize),
}

impl Conv2d {
    /// `kernel` is a `(height, width)` pair, matching [`Tensor::conv2d`].
    /// Stride defaults to `(1, 1)`, padding to `(0, 0)`, dilation to `(1, 1)`
    /// — the standard convolution — overridable with
    /// [`with_stride`](Conv2d::with_stride) / [`with_padding`](Conv2d::with_padding)
    /// / [`with_dilation`](Conv2d::with_dilation).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArg`] (`op: "Conv2d::new"`) if `in_channels`,
    /// `out_channels`, or either kernel axis is zero, or if the weight's
    /// element count overflows `usize`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize),
        device: &Device,
        rng: &mut Rng,
    ) -> Result<Conv2d> {
        const OP: &str = "Conv2d::new";
        check_dims(OP, in_channels, out_channels, kernel)?;
        let weight = init::kaiming_uniform(
            [out_channels, in_channels, kernel.0, kernel.1],
            2f64.sqrt(),
            DType::F32,
            device,
            rng,
        )?;
        Self::assemble(out_channels, weight)
    }

    /// [`new`](Conv2d::new) with the weight drawn by `init` instead of the
    /// hard-coded Kaiming-uniform scheme. `init` receives the weight's shape
    /// (`[out_channels, in_channels, kh, kw]`) and `device`, and must return
    /// a tensor of exactly that shape — the natural way to plug in
    /// [`nn::init`](crate::nn::init)'s other named initializers (Xavier, for
    /// instance, or Kaiming-normal). Stride/padding/dilation default exactly
    /// as [`new`](Conv2d::new)'s do.
    ///
    /// # Errors
    ///
    /// As [`new`](Conv2d::new), plus whatever `init` returns, or
    /// [`Error::ShapeMismatch`] if `init`'s output is not exactly
    /// `[out_channels, in_channels, kh, kw]`.
    pub fn with_init(
        in_channels: usize,
        out_channels: usize,
        kernel: (usize, usize),
        device: &Device,
        init: impl FnOnce(&[usize], &Device) -> Result<Tensor>,
    ) -> Result<Conv2d> {
        const OP: &str = "Conv2d::with_init";
        check_dims(OP, in_channels, out_channels, kernel)?;
        let expected = [out_channels, in_channels, kernel.0, kernel.1];
        let weight = init(&expected, device)?;
        if weight.dims() != expected {
            return Err(Error::shape_mismatch(OP, expected, weight.shape()));
        }
        Self::assemble(out_channels, weight)
    }

    /// The bias is drawn from the *weight*, not from the constructor's
    /// `device`/`DType::F32`: a `with_init` closure is free to return an `F64`
    /// weight or one on another device, and a bias that disagreed with it
    /// would make every `forward` fail on the broadcast add.
    fn assemble(out_channels: usize, weight: Tensor) -> Result<Conv2d> {
        let bias = Tensor::zeros([out_channels], weight.dtype(), &weight.device())?;
        Ok(Conv2d {
            weight: Param::new(weight),
            bias: Some(Param::new(bias)),
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
        })
    }

    /// Override the stride (default: `(1, 1)`).
    #[must_use]
    pub fn with_stride(mut self, stride: (usize, usize)) -> Conv2d {
        self.stride = stride;
        self
    }

    /// Override the padding (default: `(0, 0)`).
    #[must_use]
    pub fn with_padding(mut self, padding: (usize, usize)) -> Conv2d {
        self.padding = padding;
        self
    }

    /// Override the dilation (default: `(1, 1)`).
    #[must_use]
    pub fn with_dilation(mut self, dilation: (usize, usize)) -> Conv2d {
        self.dilation = dilation;
        self
    }

    /// Drop this layer's bias, as a convolution immediately followed by a
    /// normalization layer (whose own shift subsumes it) wants.
    ///
    /// Consuming, so it reads as part of construction. The bias is
    /// zero-initialized and therefore consumes no randomness, so removing it
    /// afterward leaves the weight — and the caller's `Rng` stream — exactly
    /// as it was.
    #[must_use]
    pub fn without_bias(mut self) -> Conv2d {
        self.bias = None;
        self
    }

    /// The input channel count (the weight's second axis).
    pub fn in_channels(&self) -> usize {
        self.weight.value().dims()[1]
    }

    /// The output channel count (the weight's first axis).
    pub fn out_channels(&self) -> usize {
        self.weight.value().dims()[0]
    }

    /// The `(height, width)` kernel extent (the weight's last two axes).
    pub fn kernel(&self) -> (usize, usize) {
        let dims = self.weight.value().dims();
        (dims[2], dims[3])
    }

    /// The configured stride.
    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }

    /// The configured padding.
    pub fn padding(&self) -> (usize, usize) {
        self.padding
    }

    /// The configured dilation.
    pub fn dilation(&self) -> (usize, usize) {
        self.dilation
    }

    /// The weight parameter, shape `[out_channels, in_channels, kh, kw]`.
    pub fn weight(&self) -> &Param {
        &self.weight
    }

    /// The bias parameter, shape `[out_channels]`, or `None` for a
    /// [bias-free](Conv2d::without_bias) layer.
    pub fn bias(&self) -> Option<&Param> {
        self.bias.as_ref()
    }
}

fn check_dims(
    op: &'static str,
    in_channels: usize,
    out_channels: usize,
    kernel: (usize, usize),
) -> Result<()> {
    if in_channels == 0 || out_channels == 0 || kernel.0 == 0 || kernel.1 == 0 {
        return Err(Error::invalid_arg(
            op,
            format!(
                "in_channels, out_channels, and both kernel axes must be non-zero \
                 (got in_channels {in_channels}, out_channels {out_channels}, kernel {kernel:?})"
            ),
        ));
    }
    // Caught downstream by `Layout::contiguous` regardless, but under
    // `op: "layout"` — which names the wrong operation to a caller who passed
    // the dimensions to this constructor.
    if out_channels
        .checked_mul(in_channels)
        .and_then(|n| n.checked_mul(kernel.0))
        .and_then(|n| n.checked_mul(kernel.1))
        .is_none()
    {
        return Err(Error::invalid_arg(
            op,
            format!(
                "{out_channels} * {in_channels} * {} * {} weights overflow usize",
                kernel.0, kernel.1
            ),
        ));
    }
    Ok(())
}

impl Forward for Conv2d {
    type Output = Tensor;

    /// # Errors
    ///
    /// As [`Tensor::conv2d`], plus a [`Error::ReshapeMismatch`] path that
    /// cannot trigger for a layer built by either constructor: the rank-1
    /// bias is reshaped to `[out_channels, 1, 1]` to broadcast over the
    /// spatial axes.
    fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        let y = x.conv2d(
            &self.weight.get(mode),
            self.stride,
            self.padding,
            self.dilation,
        )?;
        match &self.bias {
            Some(bias) => {
                let channels = bias.value().dims()[0];
                y.add(&bias.get(mode).reshape([channels, 1, 1])?)
            }
            None => Ok(y),
        }
    }
}

impl std::fmt::Debug for Conv2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dims = self.weight.value().dims();
        write!(
            f,
            "Conv2d({} -> {}, kernel {:?}, stride {:?}, padding {:?}, dilation {:?}{})",
            dims[1],
            dims[0],
            (dims[2], dims[3]),
            self.stride,
            self.padding,
            self.dilation,
            if self.bias.is_some() {
                ", bias"
            } else {
                ", no bias"
            }
        )
    }
}

#[cfg(test)]
mod tests;
