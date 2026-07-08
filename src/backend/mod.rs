mod cpu;
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
mod cuda;
#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal;
pub(crate) mod parallel;
#[cfg(feature = "wgpu")]
mod wgpu;

use crate::dtype::DType;
use std::borrow::Cow;

pub use cpu::{Cpu, CpuDevice, CpuError};
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
pub use cuda::{Cuda, CudaDevice, CudaError};
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use metal::{Metal, MetalDevice, MetalError};
#[cfg(feature = "wgpu")]
pub use wgpu::{Wgpu, WgpuDevice, WgpuError};

pub(crate) mod sealed {
    pub trait SealedBackend {}
}

/// Test-only counter of native-dispatch fallbacks.
///
/// Every typed-layer op that has a `try_*` native hook records here when it
/// takes the reference fallback path (the hook returned `None`). Parity tests
/// on the Metal hardware lane assert an op ran native by checking its fall
/// count is zero. The counter is thread-local so parallel tests do not
/// interfere, and it compiles out entirely outside `cfg(test)`.
#[cfg(test)]
pub(crate) mod fall_counter {
    use std::cell::RefCell;
    use std::collections::BTreeMap;

    thread_local! {
        static FALLS: RefCell<BTreeMap<&'static str, usize>> =
            const { RefCell::new(BTreeMap::new()) };
    }

    pub(crate) fn record(op: &'static str) {
        FALLS.with(|falls| *falls.borrow_mut().entry(op).or_insert(0) += 1);
    }

    pub(crate) fn reset() {
        FALLS.with(|falls| falls.borrow_mut().clear());
    }

    pub(crate) fn count(op: &'static str) -> usize {
        FALLS.with(|falls| falls.borrow().get(op).copied().unwrap_or(0))
    }
}

/// Records a reference-path fallback in test builds; a no-op otherwise.
#[inline]
pub(crate) fn record_reference_fall(_op: &'static str) {
    #[cfg(test)]
    fall_counter::record(_op);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativeUnaryOp {
    Relu,
    Neg,
    Exp,
    Ln,
    Tanh,
    Sigmoid,
    Sqrt,
    Abs,
    Gelu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativeRowOp {
    Softmax,
    LogSoftmax,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NativeBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Tensor storage and kernel backend.
///
/// Backend implementation is not a supported downstream extension point.
/// The trait is sealed through 1.0: the Epoch 16 backend claims decision
/// demoted every GPU backend to experimental, so no stable implementor
/// surface is justified yet. Unsealing (or a capability-trait split) is
/// reconsidered when a backend is promoted to a supported training path,
/// which requires the device-resident work recorded in
/// `docs/backend-dtype-support.md`.
pub trait Backend<E: DType>: sealed::SealedBackend + Clone + Send + Sync + 'static {
    type Device: Clone + Send + Sync + PartialEq + std::fmt::Debug + 'static;
    type Storage: Clone + Send + Sync + 'static;
    type Error: std::error::Error + Send + Sync + 'static;

    fn default_device() -> std::result::Result<Self::Device, Self::Error>;
    fn zeros(device: &Self::Device, len: usize) -> std::result::Result<Self::Storage, Self::Error>;
    fn ones(device: &Self::Device, len: usize) -> std::result::Result<Self::Storage, Self::Error>;
    fn from_vec(
        device: &Self::Device,
        data: Vec<E>,
    ) -> std::result::Result<Self::Storage, Self::Error>;
    fn to_vec(
        device: &Self::Device,
        storage: &Self::Storage,
    ) -> std::result::Result<Vec<E>, Self::Error>;
    fn host_access<'a>(
        device: &Self::Device,
        storage: &'a Self::Storage,
    ) -> std::result::Result<Cow<'a, [E]>, Self::Error> {
        Self::to_vec(device, storage).map(Cow::Owned)
    }
    fn storage_len(storage: &Self::Storage) -> usize;

    fn matmul(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        m: usize,
        k: usize,
        n: usize,
    ) -> std::result::Result<Self::Storage, Self::Error>;

    fn add(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error>;

    fn sub(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error>;

    fn mul(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error>;

    fn div(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error>;

    fn add_scalar(
        device: &Self::Device,
        input: &Self::Storage,
        rhs: E,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error>;

    fn sub_scalar(
        device: &Self::Device,
        input: &Self::Storage,
        rhs: E,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error>;

    fn mul_scalar(
        device: &Self::Device,
        input: &Self::Storage,
        rhs: E,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error>;

    fn div_scalar(
        device: &Self::Device,
        input: &Self::Storage,
        rhs: E,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error>;

    fn sum(
        device: &Self::Device,
        input: &Self::Storage,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error>;

    fn try_unary(
        _device: &Self::Device,
        _input: &Self::Storage,
        _len: usize,
        _op: NativeUnaryOp,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        Ok(None)
    }

    fn try_row_softmax(
        _device: &Self::Device,
        _input: &Self::Storage,
        _rows: usize,
        _cols: usize,
        _op: NativeRowOp,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        Ok(None)
    }

    fn try_sum_last(
        _device: &Self::Device,
        _input: &Self::Storage,
        _rows: usize,
        _cols: usize,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        Ok(None)
    }

    fn try_bmm(
        _device: &Self::Device,
        _lhs: &Self::Storage,
        _rhs: &Self::Storage,
        _batch: usize,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn try_strided_matmul(
        _device: &Self::Device,
        _lhs: &Self::Storage,
        _rhs: &Self::Storage,
        _m: usize,
        _k: usize,
        _n: usize,
        _lhs_offset: usize,
        _lhs_row_stride: usize,
        _lhs_col_stride: usize,
        _rhs_offset: usize,
        _rhs_row_stride: usize,
        _rhs_col_stride: usize,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        Ok(None)
    }

    fn try_broadcast_last(
        _device: &Self::Device,
        _lhs: &Self::Storage,
        _rhs: &Self::Storage,
        _rows: usize,
        _cols: usize,
        _op: NativeBinaryOp,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        Ok(None)
    }

    fn try_broadcast_leading(
        _device: &Self::Device,
        _lhs: &Self::Storage,
        _rhs: &Self::Storage,
        _leading: usize,
        _inner: usize,
        _op: NativeBinaryOp,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn try_broadcast_channel(
        _device: &Self::Device,
        _lhs: &Self::Storage,
        _rhs: &Self::Storage,
        _batch: usize,
        _channels: usize,
        _height: usize,
        _width: usize,
        _op: NativeBinaryOp,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        Ok(None)
    }

    fn try_masked_fill(
        _device: &Self::Device,
        _input: &Self::Storage,
        _mask: &[bool],
        _value: E,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        Ok(None)
    }

    fn try_where_mask(
        _device: &Self::Device,
        _lhs: &Self::Storage,
        _mask: &[bool],
        _rhs: &Self::Storage,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        Ok(None)
    }

    fn try_index_select_rows(
        _device: &Self::Device,
        _input: &Self::Storage,
        _indices: &[usize],
        _rows: usize,
        _cols: usize,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    fn try_cross_entropy(
        _device: &Self::Device,
        _logits: &Self::Storage,
        _targets: &[usize],
        _rows: usize,
        _cols: usize,
        _ignore_index: Option<usize>,
        _label_smoothing: f64,
        _mean_reduction: bool,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        Ok(None)
    }

    fn try_layer_norm(
        _device: &Self::Device,
        _input: &Self::Storage,
        _weight: &Self::Storage,
        _bias: &Self::Storage,
        _rows: usize,
        _cols: usize,
        _eps: f64,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        Ok(None)
    }

    fn try_rms_norm(
        _device: &Self::Device,
        _input: &Self::Storage,
        _weight: &Self::Storage,
        _rows: usize,
        _cols: usize,
        _eps: f64,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        Ok(None)
    }
}
