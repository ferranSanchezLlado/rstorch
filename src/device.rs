//! The [`Device`] enum is the entire public face of the backend layer. There is
//! no public backend trait; backend implementations remain crate-internal.

use crate::error::Result;

/// Where a tensor's storage lives and where its kernels run.
///
/// CPU is the reference implementation: every other backend is validated
/// against it by the table-driven conformance suite, and a missing kernel
/// on another device is a loud [`Unsupported`](crate::Error::Unsupported)
/// error — never a silent host round-trip.
///
/// # Examples
///
/// ```
/// use rstorch::{DType, Device, Tensor};
///
/// let device = Device::Cpu;
/// let x = Tensor::zeros([2, 2], DType::F32, &device)?;
/// assert_eq!(x.device(), device);
/// # Ok::<(), rstorch::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Device {
    /// The CPU reference backend. Always available.
    Cpu,
    /// Apple Metal GPU, identified by device ordinal.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(usize),
    /// NVIDIA CUDA GPU, identified by device ordinal.
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    Cuda(usize),
    /// Portable WebGPU adapter, identified by an ordinal in the current
    /// process's sorted adapter set; it is not a stable device identity.
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    Wgpu(usize),
}

impl Device {
    /// The best device available at runtime.
    ///
    /// Returns the first device that can initialize in this order: Metal
    /// ordinal 0 on macOS when the `metal` feature is enabled, CUDA on Linux
    /// or Windows when the `cuda` feature is enabled, the highest-ranked
    /// eligible WGPU adapter when the `wgpu` feature is enabled, and finally
    /// [`Device::Cpu`]. Metal availability includes runtime shader
    /// compilation, command-queue creation, and validation-buffer allocation;
    /// a Metal ordinal that fails any of those steps is skipped. WGPU adapter
    /// type and backend are driver-reported; automatic selection excludes
    /// adapters reported as `Cpu`, while other software classification is
    /// backend-dependent. WGPU ranking prefers a discrete GPU over an
    /// integrated one, then virtual, then other; ties break by backend,
    /// preferring DX12 or Metal over Vulkan over GL.
    ///
    /// Availability and performance are runtime properties; selecting a GPU
    /// does not promise that it is faster than CPU.
    pub fn best_available() -> Device {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if crate::backend::metal::is_available(0) {
            return Device::Metal(0);
        }
        #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
        if crate::backend::cuda::is_available(0) {
            return Device::Cuda(0);
        }
        #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
        if let Some(ordinal) = crate::backend::wgpu::best_adapter_ordinal() {
            return Device::Wgpu(ordinal);
        }
        Device::Cpu
    }

    /// Block until every operation already submitted on this device has
    /// completed.
    ///
    /// Kernel results are **device-ordered, not synchronous**: an op returns as
    /// soon as its work is queued, and a backend otherwise synchronizes only at
    /// a host boundary ([`Tensor::to_vec`](crate::Tensor::to_vec),
    /// [`to_scalar`](crate::Tensor::to_scalar), [`item`](crate::Tensor::item)).
    /// A GPU backend can therefore be many dispatches behind the calls that
    /// appear to have produced its tensors — Metal accumulates encoded work in
    /// one command buffer and commits it in batches. This is the explicit
    /// flush for the cases where the waiting *is* the point rather than a side
    /// effect of reading values back: timing a kernel, or forcing a deferred
    /// error to surface now.
    ///
    /// Per backend:
    ///
    /// - [`Cpu`](Device::Cpu): a **documented no-op** returning `Ok(())`. A CPU
    ///   kernel has finished by the time it returns, so nothing is ever queued.
    /// - Metal: commits the open command buffer and waits for every submitted
    ///   one, then collects deferred bounds verdicts.
    /// - CUDA: synchronizes the stream, then collects deferred bounds verdicts.
    /// - WGPU: polls the device until the queue has drained.
    ///
    /// A device-wide synchronize does not discover pending lazy tensors because
    /// those expressions are owned by their tensor handles; call
    /// [`Tensor::realize`](crate::Tensor::realize) on each root when lazy
    /// execution is enabled.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Backend`](crate::Error::Backend) if the device cannot
    /// be reached or the drained work failed.
    ///
    /// On Metal and CUDA it also returns
    /// [`Error::IndexOutOfBounds`](crate::Error::IndexOutOfBounds) for a
    /// deferred bounds check — an earlier `index_select`/`gather` with an
    /// out-of-range index, whose verdict this flush is the first to read.
    /// WGPU cannot do that: a pending verdict is attached to the storage that
    /// carries it rather than to the device, so there is no device-wide set to
    /// read here, and it surfaces at the host read of the tensor that carries
    /// it instead.
    ///
    /// # Examples
    ///
    /// ```
    /// use rstorch::{DType, Device, Tensor};
    ///
    /// let device = Device::best_available();
    /// let x = Tensor::zeros([256, 256], DType::F32, &device)?;
    /// let y = x.add(&x)?;
    /// // `y`'s kernel may still be queued; after this it has run.
    /// device.synchronize()?;
    /// assert_eq!(y.dims(), &[256, 256]);
    /// # Ok::<(), rstorch::Error>(())
    /// ```
    pub fn synchronize(&self) -> Result<()> {
        crate::backend::dispatch::backend(*self).synchronize()
    }

    /// Whether this is the CPU device.
    pub fn is_cpu(&self) -> bool {
        matches!(self, Device::Cpu)
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::Cpu => f.write_str("cpu"),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Device::Metal(idx) => write!(f, "metal:{idx}"),
            #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
            Device::Cuda(idx) => write!(f, "cuda:{idx}"),
            #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
            Device::Wgpu(idx) => write!(f, "wgpu:{idx}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn best_available_exists() {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if crate::backend::metal::is_available(0) {
            assert_eq!(Device::best_available(), Device::Metal(0));
            return;
        }
        #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
        if crate::backend::cuda::is_available(0) {
            assert_eq!(Device::best_available(), Device::Cuda(0));
            return;
        }
        #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
        if let Some(ordinal) = crate::backend::wgpu::best_adapter_ordinal() {
            assert_eq!(Device::best_available(), Device::Wgpu(ordinal));
            return;
        }
        assert_eq!(Device::best_available(), Device::Cpu);
    }

    #[test]
    fn display() {
        assert_eq!(Device::Cpu.to_string(), "cpu");
        #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
        assert_eq!(Device::Cuda(3).to_string(), "cuda:3");
        #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
        assert_eq!(Device::Wgpu(3).to_string(), "wgpu:3");
    }
}
