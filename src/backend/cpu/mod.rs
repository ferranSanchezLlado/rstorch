//! CPU backend: the reference implementation every other backend is
//! conformance-tested against.
//!
//! Per-family kernel modules (ownership per implementation-plan §3):
//! `host` (T10a), `elementwise` (T10b), `reduce`/`matmul` (T11),
//! `index` (T25), `conv` (T26), `fused` (T48). T10b additionally owns
//! this file and `backend/parallel.rs` (the rayon switch). `acc` holds the
//! wide-accumulator traits every one of those kernel families shares, and
//! `dispatch` the single runtime-dtype dispatch they all route through.

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
pub(super) fn cpu_storage<'a>(x: View<'a>) -> &'a CpuStorage {
    match x.storage() {
        Storage::Cpu(s) => s,
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Storage::Metal(_) => {
            unreachable!("CPU backend received non-CPU storage; dispatcher invariant violated")
        }
    }
}
