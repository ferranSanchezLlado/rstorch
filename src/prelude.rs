//! The first-hour surface: `use rstorch::prelude::*;`.
//!
//! Exposes the eleven-item vocabulary of exploration §4.7. Edits here are
//! **append-only** re-exports: later tasks add their public types (the
//! `nn` layers, `Sgd`/`Adam`, `DataLoader`, `#[derive(Module)]`) without
//! reordering or removing.

pub use crate::{DType, Device, Error, Grads, Result, Rng, Shape, Tensor};

pub use crate::nn::{Forward, Mode, Module, Param, Sequential};

// The core layer zoo (T41).
pub use crate::nn::{Dropout, Gelu, Linear, Relu};

// The normalization layers (T42).
pub use crate::nn::{BatchNorm2d, LayerNorm, RMSNorm};

pub use crate::text::{BpeTokenizer, CharTokenizer, Tokenizer};

// The data pipeline (T45): the loader, the trait its batches come from, and
// the two provided datasets that cover the in-memory cases.
pub use crate::data::{DataLoader, Dataset, TensorDataset, VecDataset};

// The layer zoo (T43): the two layers a transformer cannot be written without.
pub use crate::nn::{Embedding, MultiHeadAttention};

// The `Module` *derive macro* lives in the macro namespace, so it coexists
// with the `Module` trait above under the one name: a single
// `use rstorch::prelude::*;` brings both, and `#[derive(Module)]` resolves.
pub use crate::Module;
