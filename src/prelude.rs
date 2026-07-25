//! The first-hour surface: `use rstorch::prelude::*;`.
//!
//! Exposes the eleven-item vocabulary of exploration §4.7. Edits here are
//! **append-only** re-exports: later tasks add their public types (the
//! `nn` layers, `Sgd`/`Adam`, `DataLoader`, `#[derive(Module)]`) without
//! reordering or removing.

pub use crate::{DType, Device, Error, Grads, Result, Rng, Shape, Tensor};

pub use crate::nn::{Forward, Mode, Module, Param, Sequential};

pub use crate::text::{BpeTokenizer, CharTokenizer, Tokenizer};

// The `Module` *derive macro* lives in the macro namespace, so it coexists
// with the `Module` trait above under the one name: a single
// `use rstorch::prelude::*;` brings both, and `#[derive(Module)]` resolves.
pub use crate::Module;
