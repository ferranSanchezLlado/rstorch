//! [`MaxPool2d`], [`AvgPool2d`] over [`Tensor::max_pool2d`]/[`Tensor::avg_pool2d`],
//! plus [`Flatten`] and [`Identity`] — the shape-only layers a convolutional
//! network needs between its conv stack and its classifier head.

use crate::error::Result;
use crate::nn::{Forward, Mode};
use crate::tensor::Tensor;

/// 2-D max pooling over an NCHW image ([`Tensor::max_pool2d`]).
///
/// `stride` defaults to `kernel` (`PyTorch`'s convention: a non-overlapping
/// window unless told otherwise) and `padding` defaults to `(0, 0)`; override
/// either with [`with_stride`](MaxPool2d::with_stride) /
/// [`with_padding`](MaxPool2d::with_padding).
///
/// ```
/// use rstorch::nn::{Forward, MaxPool2d, Mode};
/// use rstorch::{DType, Device, Tensor};
///
/// let dev = Device::Cpu;
/// let mut pool = MaxPool2d::new((2, 2));
/// let x = Tensor::zeros([1, 3, 8, 8], DType::F32, &dev)?;
/// assert_eq!(pool.forward(&x, Mode::EVAL)?.dims(), &[1, 3, 4, 4]);
/// # Ok::<(), rstorch::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, rstorch::Module)]
pub struct MaxPool2d {
    #[module(skip)]
    kernel: (usize, usize),
    #[module(skip)]
    stride: (usize, usize),
    #[module(skip)]
    padding: (usize, usize),
}

impl MaxPool2d {
    /// A pooling window of `kernel`, stride equal to `kernel`, no padding.
    ///
    /// Infallible, so validation is deferred to
    /// [`forward`](Forward::forward): a zero kernel or stride, or a padding
    /// over half the window, surfaces there as
    /// [`Error::InvalidArg`](crate::Error::InvalidArg) from
    /// [`Tensor::max_pool2d`], not at construction the way
    /// [`Conv2d::new`](crate::nn::Conv2d::new) rejects a zero kernel. These
    /// are `Copy` value types whose builders return `Self`, not `Result`.
    #[must_use]
    pub fn new(kernel: (usize, usize)) -> MaxPool2d {
        MaxPool2d {
            kernel,
            stride: kernel,
            padding: (0, 0),
        }
    }

    /// Override the stride (default: equal to the kernel).
    #[must_use]
    pub fn with_stride(mut self, stride: (usize, usize)) -> MaxPool2d {
        self.stride = stride;
        self
    }

    /// Override the padding (default: `(0, 0)`).
    #[must_use]
    pub fn with_padding(mut self, padding: (usize, usize)) -> MaxPool2d {
        self.padding = padding;
        self
    }

    /// The `(height, width)` pooling window.
    #[must_use]
    pub fn kernel(&self) -> (usize, usize) {
        self.kernel
    }

    /// The configured stride (defaults to the kernel).
    #[must_use]
    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }

    /// The configured padding.
    #[must_use]
    pub fn padding(&self) -> (usize, usize) {
        self.padding
    }
}

impl Forward for MaxPool2d {
    type Output = Tensor;

    /// # Errors
    ///
    /// As [`Tensor::max_pool2d`].
    fn forward(&mut self, x: &Tensor, _mode: Mode) -> Result<Tensor> {
        x.max_pool2d(self.kernel, self.stride, self.padding)
    }
}

/// 2-D average pooling over an NCHW image ([`Tensor::avg_pool2d`]).
///
/// Same defaults as [`MaxPool2d`]: `stride` equal to `kernel`, `padding`
/// `(0, 0)` unless overridden.
///
/// The divisor is the **full window area**, not the count of real (unpadded)
/// elements — `PyTorch`'s `count_include_pad = true`, as
/// [`Tensor::avg_pool2d`] documents. With a non-zero `padding` that scales
/// border outputs down; [`MaxPool2d`] has no such asymmetry.
///
/// ```
/// use rstorch::nn::{AvgPool2d, Forward, Mode};
/// use rstorch::{DType, Device, Tensor};
///
/// let dev = Device::Cpu;
/// let mut pool = AvgPool2d::new((2, 2));
/// let x = Tensor::zeros([1, 3, 8, 8], DType::F32, &dev)?;
/// assert_eq!(pool.forward(&x, Mode::EVAL)?.dims(), &[1, 3, 4, 4]);
/// # Ok::<(), rstorch::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, rstorch::Module)]
pub struct AvgPool2d {
    #[module(skip)]
    kernel: (usize, usize),
    #[module(skip)]
    stride: (usize, usize),
    #[module(skip)]
    padding: (usize, usize),
}

impl AvgPool2d {
    /// A pooling window of `kernel`, stride equal to `kernel`, no padding.
    ///
    /// Validation is deferred to [`forward`](Forward::forward), as in
    /// [`MaxPool2d::new`]: a zero kernel or stride, or a padding over half the
    /// window, surfaces as
    /// [`Error::InvalidArg`](crate::Error::InvalidArg) from
    /// [`Tensor::avg_pool2d`].
    #[must_use]
    pub fn new(kernel: (usize, usize)) -> AvgPool2d {
        AvgPool2d {
            kernel,
            stride: kernel,
            padding: (0, 0),
        }
    }

    /// Override the stride (default: equal to the kernel).
    #[must_use]
    pub fn with_stride(mut self, stride: (usize, usize)) -> AvgPool2d {
        self.stride = stride;
        self
    }

    /// Override the padding (default: `(0, 0)`).
    #[must_use]
    pub fn with_padding(mut self, padding: (usize, usize)) -> AvgPool2d {
        self.padding = padding;
        self
    }

    /// The `(height, width)` pooling window.
    #[must_use]
    pub fn kernel(&self) -> (usize, usize) {
        self.kernel
    }

    /// The configured stride (defaults to the kernel).
    #[must_use]
    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }

    /// The configured padding.
    #[must_use]
    pub fn padding(&self) -> (usize, usize) {
        self.padding
    }
}

impl Forward for AvgPool2d {
    type Output = Tensor;

    /// # Errors
    ///
    /// As [`Tensor::avg_pool2d`].
    fn forward(&mut self, x: &Tensor, _mode: Mode) -> Result<Tensor> {
        x.avg_pool2d(self.kernel, self.stride, self.padding)
    }
}

/// Collapse every axis from `start_dim` onward into one — `PyTorch`'s
/// `nn.Flatten`. The default `start_dim` is `1`, so a `[batch, C, H, W]`
/// conv output becomes `[batch, C*H*W]`, ready for a
/// [`Linear`](crate::nn::Linear) classifier head.
///
/// ```
/// use rstorch::nn::{Flatten, Forward, Mode};
/// use rstorch::{DType, Device, Tensor};
///
/// let dev = Device::Cpu;
/// let mut flatten = Flatten::new();
/// let x = Tensor::zeros([2, 4, 3, 3], DType::F32, &dev)?;
/// assert_eq!(flatten.forward(&x, Mode::EVAL)?.dims(), &[2, 36]);
/// # Ok::<(), rstorch::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, rstorch::Module)]
pub struct Flatten {
    #[module(skip)]
    start_dim: isize,
}

impl Flatten {
    /// Flatten from axis `1` onward (every leading axis is a batch axis).
    #[must_use]
    pub fn new() -> Flatten {
        Flatten { start_dim: 1 }
    }

    /// Flatten from `start_dim` onward instead (negative indexing allowed).
    #[must_use]
    pub fn with_start_dim(mut self, start_dim: isize) -> Flatten {
        self.start_dim = start_dim;
        self
    }

    /// The axis this layer flattens from (negative values index from the end).
    #[must_use]
    pub fn start_dim(&self) -> isize {
        self.start_dim
    }
}

impl Default for Flatten {
    fn default() -> Flatten {
        Flatten::new()
    }
}

impl Forward for Flatten {
    type Output = Tensor;

    /// # Errors
    ///
    /// [`Error::InvalidAxis`](crate::Error::InvalidAxis) (`op:
    /// "Flatten::forward"`) if `start_dim` is out of range for `x`; the
    /// reshape itself cannot fail, since the flattened shape holds exactly
    /// the same element count.
    fn forward(&mut self, x: &Tensor, _mode: Mode) -> Result<Tensor> {
        let axis = x.shape().resolve_axis(self.start_dim, "Flatten::forward")?;
        let dims = x.dims();
        let mut flat = dims[..axis].to_vec();
        flat.push(dims[axis..].iter().product());
        x.reshape(flat)
    }
}

/// The identity layer: returns its input unchanged. Useful as the "no-op"
/// arm of a conditional shortcut (a `ResNet` block whose stride and channel
/// count do not change), or anywhere a [`Sequential`](crate::nn::Sequential)
/// needs a placeholder stage.
///
/// ```
/// use rstorch::nn::{Forward, Identity, Mode};
/// use rstorch::{DType, Device, Tensor};
///
/// let dev = Device::Cpu;
/// let mut id = Identity;
/// let x = Tensor::zeros([2, 3], DType::F32, &dev)?;
/// assert_eq!(id.forward(&x, Mode::EVAL)?.dims(), &[2, 3]);
/// # Ok::<(), rstorch::Error>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, rstorch::Module)]
pub struct Identity;

impl Forward for Identity {
    type Output = Tensor;

    /// Never fails: an `Arc` bump, no computation.
    fn forward(&mut self, x: &Tensor, _mode: Mode) -> Result<Tensor> {
        Ok(x.clone())
    }
}

#[cfg(test)]
mod tests;
