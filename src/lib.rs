#![warn(missing_docs)]

//! A PyTorch-inspired deep-learning library with one dynamic tensor type and
//! linear gradients.
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
//! Features are additive. `metal` is enabled by default on macOS.
//!
//! | Feature | What it adds |
//! |---|---|
//! | `typed` | Experimental compile-time checked tensor wrappers |
//! | `rayon` | Parallel CPU kernels |
//! | `hub` | Dataset downloads |
//! | `metal` | The macOS Metal backend |
//! | `cuda` | Native NVIDIA CUDA on Linux and Windows |
//! | `wgpu` | Portable native WebGPU |
//! | `testing` | The finite-difference gradient helper |
//!
//! The `rstorch::lazy` namespace is an experimental deferred executor and is
//! disabled unless enabled at runtime.
//!
//! The dynamic API follows semantic versioning from 1.0. The `testing`,
//! `typed`, and `lazy` namespaces are outside that compatibility promise.

// `#[derive(Module)]` resolves the runtime crate through this alias when the
// derive is used inside `rstorch`, just as it does in downstream crates.
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
/// Experimental runtime control for deferred and fused element-wise execution.
pub mod lazy;
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
