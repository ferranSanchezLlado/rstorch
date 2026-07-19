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
//! ```ignore
//! use rstorch::prelude::*;
//! ```

// ---- public namespaces (types also re-exported flat below) --------------
pub mod device;
pub mod dtype;
pub mod error;
pub mod shape;

// ---- flat vocabulary: private modules, root re-exports ------------------
mod autograd;
mod rng;
mod tensor;

// ---- crate-internal foundations -----------------------------------------
mod backend;
pub(crate) mod layout;
pub(crate) mod storage;

// ---- subsystems ----------------------------------------------------------
pub mod data;
pub mod nn;
pub mod optim;
pub mod persist;
pub mod prelude;
pub mod testing;
pub mod text;

// ---- root re-exports: the design's flat vocabulary ----------------------
pub use autograd::Grads;
pub use device::Device;
pub use dtype::{DType, Element};
pub use error::{Error, Result};
pub use rng::Rng;
pub use shape::Shape;
pub use tensor::Tensor;
