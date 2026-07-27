//! The [`Device`] enum — the entire public face of the backend layer
//! (exploration §4.5). There is no public backend trait; adding a backend
//! is a crate-internal change with no semver event.

/// Where a tensor's storage lives and where its kernels run.
///
/// CPU is the reference implementation: every other backend is validated
/// against it by the table-driven conformance suite, and a missing kernel
/// on another device is a loud [`Unsupported`](crate::Error::Unsupported)
/// error — never a silent host round-trip.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Device {
    /// The CPU reference backend. Always available; the only supported
    /// training path until an accelerator backend passes its perf gate.
    Cpu,
    /// Apple Metal GPU, identified by device ordinal. Experimental until
    /// the T61 conformance + performance gate passes.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    Metal(usize),
}

impl Device {
    /// The best device available at runtime.
    ///
    /// Returns [`Device::Cpu`] until an accelerator backend is promoted
    /// past its performance gate (T61): a device that would train slower
    /// than CPU is not "best", whatever the marketing says.
    pub fn best_available() -> Device {
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn best_available_is_cpu_for_now() {
        assert_eq!(Device::best_available(), Device::Cpu);
        assert!(Device::best_available().is_cpu());
    }

    #[test]
    fn display() {
        assert_eq!(Device::Cpu.to_string(), "cpu");
    }
}
