#![warn(missing_docs)]

//! A safer PyTorch-inspired deep learning library: one concrete tensor type
//! with zero generic parameters, and linear gradients.
//!
//! This crate is being rebuilt from zero on the `restart-v3` branch. The
//! approved design lives in `docs/restart-v3/exploration.md`; the task
//! breakdown in `docs/restart-v3/implementation-plan.md`.
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

// `#[derive(Module)]` (the `rstorch-derive` crate, T13) generates paths
// rooted at `::rstorch`; this alias lets that expansion resolve when the
// derive is used *inside* this crate (the layer zoo from wave 4 on) exactly
// as it does downstream. Sanctioned by the T13 brief.
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
