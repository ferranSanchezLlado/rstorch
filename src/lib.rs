#![warn(missing_docs)]

//! A safer PyTorch-inspired deep learning library: one concrete tensor type
//! with zero generic parameters, and linear gradients.
//!
//! This crate is being rebuilt from zero on the `restart-v3` branch. The
//! approved design lives in `docs/restart-v3/exploration.md`; the task
//! breakdown in `docs/restart-v3/implementation-plan.md`.

mod backend;
pub mod data;
pub mod nn;
pub mod optim;
pub mod persist;
pub mod prelude;
pub mod tensor;
pub mod text;
