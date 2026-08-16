//! Tensor storage: device-tagged, dtype-tagged element buffers.
//!
//! Storage is crate-private. Buffers are `Arc`-shared:
//! cloning a `Storage` is a refcount bump, which is what makes zero-copy
//! views (`transpose`, `narrow`, `broadcast_to`) and detached captures
//! cheap. Tensors are immutable values, so shared buffers are never
//! written after construction.

use crate::device::Device;
use crate::dtype::DType;
use std::sync::Arc;

/// A CPU element buffer, one variant per [`DType`].
///
/// Also serves as the host interchange format for every backend: transfers
/// in/out of a device (`BackendOps::transfer_in` / `transfer_out`) speak
/// `CpuStorage`.
#[derive(Clone)]
pub(crate) enum CpuStorage {
    /// f16 buffer.
    F16(Arc<Vec<half::f16>>),
    /// bf16 buffer.
    BF16(Arc<Vec<half::bf16>>),
    /// f32 buffer.
    F32(Arc<Vec<f32>>),
    /// f64 buffer.
    F64(Arc<Vec<f64>>),
    /// i64 buffer.
    I64(Arc<Vec<i64>>),
    /// bool buffer.
    Bool(Arc<Vec<bool>>),
}

impl CpuStorage {
    /// The dtype tag of this buffer.
    pub(crate) fn dtype(&self) -> DType {
        match self {
            CpuStorage::F16(_) => DType::F16,
            CpuStorage::BF16(_) => DType::BF16,
            CpuStorage::F32(_) => DType::F32,
            CpuStorage::F64(_) => DType::F64,
            CpuStorage::I64(_) => DType::I64,
            CpuStorage::Bool(_) => DType::Bool,
        }
    }

    /// Number of elements in the underlying buffer (not the logical
    /// element count of any view over it).
    pub(crate) fn len(&self) -> usize {
        match self {
            CpuStorage::F16(v) => v.len(),
            CpuStorage::BF16(v) => v.len(),
            CpuStorage::F32(v) => v.len(),
            CpuStorage::F64(v) => v.len(),
            CpuStorage::I64(v) => v.len(),
            CpuStorage::Bool(v) => v.len(),
        }
    }
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
pub(crate) use crate::backend::cuda::CudaStorage;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) use crate::backend::metal::MetalStorage;
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
pub(crate) use crate::backend::wgpu::WgpuStorage;

/// A device-tagged element buffer.
#[derive(Clone)]
pub(crate) enum Storage {
    /// Host memory, dtype-tagged.
    Cpu(CpuStorage),
    /// Metal device buffer (see the `metal` feature).
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(MetalStorage),
    /// CUDA device buffer.
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    Cuda(CudaStorage),
    /// WebGPU device buffer.
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    Wgpu(WgpuStorage),
}

impl Storage {
    /// The dtype of the stored elements.
    pub(crate) fn dtype(&self) -> DType {
        match self {
            Storage::Cpu(s) => s.dtype(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Storage::Metal(s) => s.dtype(),
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            Storage::Cuda(s) => s.dtype(),
            #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
            Storage::Wgpu(s) => s.dtype(),
        }
    }

    /// The device this buffer lives on.
    pub(crate) fn device(&self) -> Device {
        match self {
            Storage::Cpu(_) => Device::Cpu,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Storage::Metal(s) => s.device(),
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            Storage::Cuda(s) => s.device(),
            #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
            Storage::Wgpu(s) => s.device(),
        }
    }

    /// Number of elements in the underlying buffer.
    pub(crate) fn len(&self) -> usize {
        match self {
            Storage::Cpu(s) => s.len(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Storage::Metal(s) => s.len(),
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            Storage::Cuda(s) => s.len(),
            #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
            Storage::Wgpu(s) => s.len(),
        }
    }
}
