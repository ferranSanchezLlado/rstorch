#![warn(missing_docs)]

//! A safer PyTorch-inspired deep learning library: one concrete tensor type
//! with zero generic parameters, and linear gradients.
//!
//! The first-hour surface is re-exported at the crate root and, together
//! with the `nn`/`optim`/`data` types, through [`prelude`]:
//!
//! ```
//! use rstorch::prelude::*;
//!
//! let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3], &Device::Cpu)?;
//! assert_eq!(x.dims(), &[3]);
//! # Ok::<(), rstorch::Error>(())
//! ```
//!
//! # Feature flags
//!
//! Features are additive: enabling one never removes or changes an item that
//! was there without it. `metal` is enabled by default and only affects macOS.
//!
//! | Feature | What it adds |
//! |---|---|
//! | `typed` | The `typed` namespace: rank, dimensions, dtype and placement checked at compile time, as a wrapper over this same [`Tensor`]. |
//! | `rayon` | Multi-threaded CPU kernels. Results are bit-identical to the single-threaded ones: kernels partition by output element, so no float is accumulated across threads in a racing order. |
//! | `hub` | Downloads for the bundled datasets in [`data::hub`]. |
//! | `metal` | The default macOS GPU backend. [`Device::best_available`] selects it when present, then considers WGPU and CPU. |
//! | `wgpu` | Opt-in portable native GPU backend. Supports F32 compute plus lossless I64 index storage; unsupported dtypes fail loudly. |
//! | `testing` | The `testing` finite-difference gradient harness. The one public module outside the stability guarantee. |
//!
//! # Stability
//!
//! This crate follows semantic versioning from 1.0. The covered surface is
//! recorded per feature combination under `api/` in the repository and diffed
//! by CI on every run; `STABILITY.md` states the scope, backend and dtype
//! capability policy, checkpoint integrity limits, and MSRV policy.

// `#[derive(Module)]` (the `rstorch-derive` crate) generates paths
// rooted at `::rstorch`; this alias lets that expansion resolve when the
// derive is used *inside* this crate (the layer zoo from wave 4 on) exactly
// as it does downstream.
extern crate self as rstorch;

// ---- public namespaces (types also re-exported flat below) --------------
pub mod device;
pub mod dtype;
pub mod error;
pub mod shape;

// ---- flat vocabulary: private modules, root re-exports ------------------
mod autograd;
mod checkpoint;
mod rng;
mod tensor;

// ---- crate-internal foundations -----------------------------------------
mod backend;
pub(crate) mod layout;
pub(crate) mod storage;

// ---- subsystems ----------------------------------------------------------
pub mod data;
pub mod models;
pub mod nn;
pub mod optim;
pub mod persist;
pub mod prelude;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
pub mod text;
/// Compile-time checked tensor and neural-network APIs.
#[cfg(feature = "typed")]
pub mod typed;

// ---- root re-exports: the design's flat vocabulary ----------------------
pub use autograd::Grads;
pub use device::Device;
pub use dtype::{DType, Element};
pub use error::{Error, Result};
pub use rng::Rng;
pub use shape::Shape;
pub use tensor::Tensor;

/// Derive an implementation of [`nn::Module`] — see the [`rstorch_derive`]
/// crate docs for the loud-by-default field-classification rule and
/// `#[module(skip)]`.
pub use rstorch_derive::Module;
