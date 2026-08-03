//! Neural-network modules: the [`Module`]/[`Forward`] traits, [`Param`],
//! [`Mode`], the parameter [`Visitor`]s, the model-level utilities
//! ([`state_dict`]/[`load_state_dict`]/[`to_device`]/[`to_dtype`]),
//! [`Sequential`], and (from wave 4 on) the layer zoo (exploration §4.4).
//!
//! Five public traits exist in the entire library; two of them —
//! [`Module`] and [`Forward`] — live here. Both are object-safe.
//!
//! # Parameters, buffers, and paths
//!
//! A [`Param`] is trainable and optimizer-visited; a plain `Tensor` field
//! declared as a buffer is non-trainable persistent state (BatchNorm running
//! statistics). Both are moved by [`to_device`]/[`to_dtype`] and both land in
//! [`state_dict`], so a checkpoint reconstructs a model — only `Param`s count
//! toward [`num_params`] and receive gradients. Every leaf is named by the
//! dotted path its walk emits (`fc1.weight`, `blocks.3.attn.qkv.weight`);
//! those names are the `state_dict` keys.
//!
//! # Replication (EMA, target networks, per-thread inference)
//!
//! [`Param`] is deliberately not `Clone`, so a model holding one is not
//! `Clone` either. The sanctioned way to obtain a second copy is to construct
//! a fresh model of the same shape and load the original's values into it —
//! in memory, no disk:
//!
//! ```
//! # use rstorch::nn::{self, Module, Param};
//! # use rstorch::{DType, Device, Tensor};
//! # #[derive(rstorch::Module)]
//! # struct Mlp { w: Param }
//! # impl Mlp {
//! #     fn new(dev: &Device) -> rstorch::Result<Mlp> {
//! #         Ok(Mlp { w: Param::new(Tensor::zeros([2, 2], DType::F32, dev)?) })
//! #     }
//! # }
//! # fn main() -> rstorch::Result<()> {
//! let dev = Device::Cpu;
//! let model = Mlp::new(&dev)?;
//! let mut target = Mlp::new(&dev)?;
//! nn::load_state_dict(&mut target, &nn::state_dict(&model))?;
//! # Ok(())
//! # }
//! ```
//!
//! Nothing is copied element-wise: values are `Arc`-backed and immutable, so
//! the replica shares storage until either side's parameters are replaced (an
//! optimizer step, another load), which always writes a *new* tensor.
//!
//! # Loading and conversion are all-or-nothing
//!
//! [`load_state_dict`], [`to_device`], and [`to_dtype`] validate (or convert)
//! the entire walk before swapping anything — the in-memory sibling of
//! [`persist::stage`](crate::persist::stage). A rejected load or a failed
//! conversion leaves the model exactly as it was, because a half-loaded model
//! that silently produces wrong results is the failure mode they exist to
//! prevent. Every rejection names the offending path.

mod activation;
mod attention;
mod dropout;
mod embedding;
mod linear;
mod mode;
mod norm;
mod param;
mod sequential;
mod util;
pub(crate) mod visit;

pub use activation::{Gelu, Relu};
pub use attention::{MultiHeadAttention, scaled_dot_product_attention};
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use linear::Linear;
pub use mode::Mode;
pub use norm::{BatchNorm2d, LayerNorm, RMSNorm};
#[allow(unused_imports)] // Consumed by typed normalization wrappers in CT44.
pub(crate) use norm::{
    batch_norm2d_forward, check_eps, check_normalized_shape, check_suffix, layer_norm_forward,
    rms_norm_forward,
};
pub use param::Param;
pub use sequential::Sequential;
// Model-level utilities exposed flat (exploration §4.4: `nn::to_device`).
pub use util::{load_state_dict, num_params, state_dict, to_device, to_dtype};
pub use visit::{Visitor, VisitorMut};

use crate::error::Result;
use crate::tensor::Tensor;

/// A module is anything with parameters to visit (exploration §4.4).
///
/// The two methods are symmetric — read-only and mutable walks over the
/// module's own [`Param`]s and child modules, emitting dotted parameter
/// paths. `#[derive(Module)]` (the `rstorch-derive` crate, T13) writes both;
/// the derive is **loud by default** — every non-whitelisted field must be a
/// child `Module` or bear `#[module(skip)]`, so a silently unvisited (and
/// therefore untrained) parameter is a compile error.
///
/// Object-safe: `&dyn Module` drives the model-level utilities and, via
/// stable dyn upcasting (Rust 1.86), the crate-private `Sequential` layer.
pub trait Module {
    /// Walk parameters read-only (see [`Visitor`]).
    fn visit(&self, visitor: &mut Visitor);

    /// Walk parameters mutably (see [`VisitorMut`]).
    fn visit_mut(&mut self, visitor: &mut VisitorMut);
}

/// A module that maps a tensor to a tensor under a [`Mode`] (exploration
/// §4.1). `&mut self` is honest about layer state (dropout RNG, BatchNorm
/// running stats as plain fields — no interior mutability, no mutexes).
pub trait Forward {
    /// Run the forward pass. `mode` selects layer behavior and whether the
    /// computation is recorded for autograd.
    fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor>;
}
