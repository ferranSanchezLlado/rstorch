//! The [`Device`] enum is the entire public face of the backend layer. There is
//! no public backend trait; backend implementations remain crate-internal.

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
    /// Portable WebGPU adapter, identified by deterministic adapter ordinal.
    #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
    Wgpu(usize),
}

impl Device {
    /// The best device available at runtime.
    ///
    /// Returns the first available device in this order: Metal on macOS when
    /// the `metal` feature is enabled, CUDA on Linux or Windows when the `cuda`
    /// feature is enabled, the first hardware WGPU adapter when the `wgpu`
    /// feature is enabled, and finally [`Device::Cpu`].
    ///
    /// Availability and performance are runtime properties; selecting a GPU
    /// does not promise that it is faster than CPU.
    pub fn best_available() -> Device {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if !objc2_metal::MTLCopyAllDevices().is_empty() {
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
        if !objc2_metal::MTLCopyAllDevices().is_empty() {
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
