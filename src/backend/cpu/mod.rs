//! CPU backend: the reference implementation every other backend is
//! conformance-tested against.
//!
//! One module per kernel family: `host`, `elementwise`, `reduce`, `matmul`,
//! `index`, `conv`, and `fused`. `acc` holds the wide-accumulator traits every
//! one of those families shares, and `dispatch` the single runtime-dtype
//! dispatch they all route through; `backend/parallel.rs` is the rayon switch.

use crate::backend::View;
use crate::storage::{CpuStorage, Storage};

pub(crate) mod acc;
pub(crate) mod conv;
pub(crate) mod dispatch;
pub(crate) mod elementwise;
pub(crate) mod fused;
pub(crate) mod host;
pub(crate) mod index;
pub(crate) mod matmul;
pub(crate) mod reduce;

/// Borrow the [`CpuStorage`] behind a view.
///
/// [`dispatch::backend`](crate::backend::dispatch::backend) routes each
/// device to its own backend, so a CPU kernel only ever receives
/// CPU-resident views. Any other storage here is an internal contract
/// violation, not a user error.
#[inline]
pub(super) fn cpu_storage(x: View<'_>) -> &CpuStorage {
    match x.storage() {
        Storage::Cpu(s) => s,
        Storage::Pending(_) => {
            unreachable!("CPU backend received pending storage; view must be ready")
        }
        #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
        Storage::Cuda(_) => {
            unreachable!("CPU backend received non-CPU storage; dispatcher invariant violated")
        }
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Storage::Metal(_) => {
            unreachable!("CPU backend received non-CPU storage; dispatcher invariant violated")
        }
        #[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
        Storage::Wgpu(_) => {
            unreachable!("CPU backend received non-CPU storage; dispatcher invariant violated")
        }
    }
}
