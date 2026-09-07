//! Tensor storage: device-tagged, dtype-tagged element buffers.
//!
//! Storage is crate-private. Buffers are `Arc`-shared:
//! cloning a `Storage` is a refcount bump, which is what makes zero-copy
//! views (`transpose`, `narrow`, `broadcast_to`) and detached captures
//! cheap. Tensors are immutable values, so shared buffers are never
//! written after construction.
//!
//! The only interior mutability permitted in [`Storage`] or [`Inner`](crate::tensor::Inner)
//! is the write-once memoization cell owned by a pending expression. It caches
//! the result of a pure computation; it is not module state or observable
//! mutation.

use crate::device::Device;
use crate::dtype::DType;
use crate::layout::Layout;
use std::mem::ManuallyDrop;
use std::sync::{Arc, OnceLock};

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

/// A deferred storage result.
///
/// The expression and its operands are immutable. `cache` is the sole
/// interior-mutable field in the storage/tensor representation and is filled
/// only after the expression's dependencies have been realized.
pub(crate) struct PendingStorage {
    pub(crate) dtype: DType,
    pub(crate) device: Device,
    /// Layout produced by the deferred expression.
    ///
    /// A zero-copy shape view can reuse the same pending storage with a
    /// different layout. The lazy CPU executor must not fuse through that
    /// view, because the expression's physical order no longer matches the
    /// layout consumed by its parent.
    pub(crate) layout: Layout,
    pub(crate) len: usize,
    /// An upper bound on unresolved nodes reachable from this expression.
    /// Straight-line chains use it to avoid rebuilding a graph walk for every
    /// appended operation; multi-operand DAGs may refine it conservatively.
    pub(crate) node_count: usize,
    pub(crate) expr: ManuallyDrop<crate::lazy::Expr>,
    pub(crate) cache: ManuallyDrop<OnceLock<Storage>>,
}

impl PendingStorage {
    pub(crate) fn new(
        dtype: DType,
        device: Device,
        layout: Layout,
        node_count: usize,
        expr: crate::lazy::Expr,
    ) -> PendingStorage {
        let len = layout.num_elements();
        PendingStorage {
            dtype,
            device,
            layout,
            len,
            node_count,
            expr: ManuallyDrop::new(expr),
            cache: ManuallyDrop::new(OnceLock::new()),
        }
    }

    /// Number of unresolved nodes reachable from this pending node.
    pub(crate) fn pending_nodes(&self) -> usize {
        if self.cache.get().is_some() {
            0
        } else {
            self.node_count
        }
    }
}

impl Drop for PendingStorage {
    fn drop(&mut self) {
        // Both fields are `ManuallyDrop` so the owned expression tree can be
        // dismantled without letting nested `Arc<PendingStorage>` drops
        // recurse through a long chain.
        let expr = unsafe { ManuallyDrop::take(&mut self.expr) };
        let cache = unsafe { ManuallyDrop::take(&mut self.cache) };
        let mut pending = Vec::new();
        let mut owned = Vec::new();
        expr.into_storages(&mut owned);
        if let Some(storage) = cache.into_inner() {
            owned.push(storage);
        }
        for storage in owned {
            collect_pending(storage, &mut pending);
        }
        while let Some(child) = pending.pop() {
            match Arc::try_unwrap(child) {
                Ok(node) => drain_owned_pending(node, &mut pending),
                Err(child) => drop(child),
            }
        }
    }
}

fn collect_pending(storage: Storage, pending: &mut Vec<Arc<PendingStorage>>) {
    if let Storage::Pending(child) = storage {
        pending.push(child);
    }
}

fn drain_owned_pending(node: PendingStorage, pending: &mut Vec<Arc<PendingStorage>>) {
    let mut node = ManuallyDrop::new(node);
    let expr = unsafe { ManuallyDrop::take(&mut node.expr) };
    let cache = unsafe { ManuallyDrop::take(&mut node.cache) };
    let mut owned = Vec::new();
    expr.into_storages(&mut owned);
    if let Some(storage) = cache.into_inner() {
        owned.push(storage);
    }
    for storage in owned {
        collect_pending(storage, pending);
    }
}

/// A device-tagged element buffer.
#[derive(Clone)]
pub(crate) enum Storage {
    /// Host memory, dtype-tagged.
    Cpu(CpuStorage),
    /// A deferred pure expression. The `Arc` makes cloning a pending storage
    /// share its memo cell rather than duplicating an empty cell.
    Pending(Arc<PendingStorage>),
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
    /// Return a ready backing storage, realizing a pending expression first.
    pub(crate) fn ready(&self) -> crate::error::Result<&Storage> {
        match self {
            Storage::Pending(pending) => {
                crate::lazy::realize::ensure_ready(pending.clone())?;
                pending
                    .cache
                    .get()
                    .ok_or_else(|| crate::error::Error::Backend {
                        op: pending.expr.op(),
                        msg: "realizer completed without publishing storage".to_string(),
                    })
            }
            _ => Ok(self),
        }
    }

    /// Whether this handle denotes a pending expression.
    pub(crate) fn is_pending(&self) -> bool {
        matches!(self, Storage::Pending(_))
    }
}

impl Storage {
    /// The dtype of the stored elements.
    pub(crate) fn dtype(&self) -> DType {
        match self {
            Storage::Cpu(s) => s.dtype(),
            Storage::Pending(s) => s.dtype,
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
            Storage::Pending(s) => s.device,
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
            Storage::Pending(s) => s.len,
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Storage::Metal(s) => s.len(),
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            Storage::Cuda(s) => s.len(),
            #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
            Storage::Wgpu(s) => s.len(),
        }
    }
}
