//! Tensor storage: device-tagged, dtype-tagged element buffers.
//!
//! Storage is crate-private (exploration §4.5). Buffers are `Arc`-shared:
//! cloning a `Storage` is a refcount bump, which is what makes zero-copy
//! views (`transpose`, `narrow`, `broadcast_to`) and detached captures
//! cheap. Tensors are immutable values, so shared buffers are never
//! written after construction.

// Consumed by W2/W3 kernel and tensor tasks; the integrator removes this
// allow at v3-m1 once consumers exist.
#![allow(dead_code)]

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

/// Placeholder for the Metal buffer type; T61 replaces this with the real
/// device buffer + queue handle.
#[cfg(feature = "metal")]
#[derive(Clone)]
pub(crate) struct MetalStorage;

/// A device-tagged element buffer.
#[derive(Clone)]
pub(crate) enum Storage {
    /// Host memory, dtype-tagged.
    Cpu(CpuStorage),
    /// Metal device buffer (experimental; see the `metal` feature).
    #[cfg(feature = "metal")]
    Metal(MetalStorage),
}

impl Storage {
    /// The dtype of the stored elements.
    pub(crate) fn dtype(&self) -> DType {
        match self {
            Storage::Cpu(s) => s.dtype(),
            #[cfg(feature = "metal")]
            Storage::Metal(_) => unimplemented!("T61: Metal storage dtype"),
        }
    }

    /// The device this buffer lives on.
    pub(crate) fn device(&self) -> Device {
        match self {
            Storage::Cpu(_) => Device::Cpu,
            #[cfg(feature = "metal")]
            Storage::Metal(_) => unimplemented!("T61: Metal storage device"),
        }
    }

    /// Number of elements in the underlying buffer.
    pub(crate) fn len(&self) -> usize {
        match self {
            Storage::Cpu(s) => s.len(),
            #[cfg(feature = "metal")]
            Storage::Metal(_) => unimplemented!("T61: Metal storage len"),
        }
    }
}
