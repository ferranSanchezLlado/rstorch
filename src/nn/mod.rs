//! Neural-network modules: the [`Module`]/[`Forward`] traits, [`Param`],
//! [`Mode`], the parameter [`Visitor`]s, the model-level utilities
//! ([`ModuleExt`]: `state_dict`/`load_state_dict`/`to_device`/`to_dtype`), the
//! [`init`] initializers, [`Sequential`], and (from wave 4 on) the layer zoo —
//! [`Linear`], [`Embedding`], [`Dropout`], the activations, the normalizations,
//! [`MultiHeadAttention`], and the convolution/pooling set ([`Conv2d`],
//! [`MaxPool2d`], [`AvgPool2d`], [`Flatten`], [`Identity`]).
//!
//! The dynamic core has five foundational public traits; two of them —
//! [`Module`] and [`Forward`] — live here. The optional `typed` namespace adds
//! compile-time contract traits of its own. Both dynamic traits are object-safe.
//!
//! # Parameters, buffers, and paths
//!
//! A [`Param`] is trainable and optimizer-visited; a plain `Tensor` field
//! declared as a buffer is non-trainable persistent state (`BatchNorm` running
//! statistics). Both are moved by [`ModuleExt::to_device`]/[`ModuleExt::to_dtype`]
//! and both land in [`ModuleExt::state_dict`], so a checkpoint reconstructs a
//! model — only `Param`s count toward [`ModuleExt::num_params`] and receive
//! gradients. Every leaf is named by the dotted path its walk emits (for
//! example, `fc1.weight` or `blocks.3.attention.q_proj.weight`); those names
//! are the `state_dict` keys.
//!
//! # Replication (EMA, target networks, per-thread inference)
//!
//! [`Param`] is deliberately not `Clone`, so a model holding one is not
//! `Clone` either. The sanctioned way to obtain a second copy is to construct
//! a fresh model of the same shape and load the original's values into it —
//! in memory, no disk:
//!
//! ```
//! # use rstorch::nn::{Module, ModuleExt, Param};
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
//! target.load_state_dict(&model.state_dict()?)?;
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
//! [`ModuleExt::load_state_dict`], [`ModuleExt::to_device`], and
//! [`ModuleExt::to_dtype`] validate (or convert) the entire walk before
//! swapping anything — the in-memory sibling of
//! [`persist::stage`](crate::persist::stage). A rejected load or a failed
//! conversion leaves the model exactly as it was, because a half-loaded model
//! that silently produces wrong results is the failure mode they exist to
//! prevent. Every rejection names the offending path.

mod activation;
mod attention;
mod conv;
mod dropout;
mod embedding;
pub mod init;
mod linear;
mod mode;
mod norm;
mod param;
mod pool;
mod sequential;
mod util;
pub(crate) mod visit;

pub use activation::{Gelu, Relu};
pub use attention::{AttentionInput, MultiHeadAttention, scaled_dot_product_attention};
// The head axis motion, shared with the typed attention wrapper
// (`typed::nn::attention`), which splits and merges heads identically.
#[cfg(feature = "typed")]
pub(crate) use attention::{merge_heads, split_heads};
pub use conv::Conv2d;
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use linear::Linear;
// Shared with the typed `Linear` (`typed::nn::linear`), whose `Debug` line is
// this one.
#[cfg(feature = "typed")]
pub(crate) use linear::debug_linear;
pub use mode::Mode;
pub use norm::{BatchNorm2d, LayerNorm, RMSNorm};
pub use pool::{AvgPool2d, Flatten, Identity, MaxPool2d};
// Shared with the typed normalization wrappers (`typed::nn::norm`), the only
// consumers outside `nn::norm` itself.
#[cfg(feature = "typed")]
pub(crate) use norm::{
    batch_norm2d_forward, check_eps, check_momentum, check_normalized_shape, check_suffix,
    layer_norm_forward, rms_norm_forward,
};
pub use param::Param;
pub use sequential::Sequential;
// Model-level utilities, discoverable as methods via `ModuleExt`.
pub use util::{ModuleExt, StateDict};
pub use visit::{Visitor, VisitorMut};

use crate::error::Result;
use crate::tensor::Tensor;

/// A module is anything with parameters to visit.
///
/// The two methods are symmetric — read-only and mutable walks over the
/// module's own [`Param`]s and child modules, emitting dotted parameter
/// paths. `#[derive(Module)]` (the `rstorch-derive` crate) writes both;
/// the derive is **loud by default** — every non-whitelisted field must be a
/// child `Module` or bear `#[module(skip)]`, so a silently unvisited (and
/// therefore untrained) parameter is a compile error unless the caller has
/// explicitly opted that field out with `skip`.
///
/// A safe implementation must emit each parameter or buffer exactly once from
/// both walks, under the same dotted path and leaf kind. The model utilities
/// validate this contract before mutating state, but the trait itself remains
/// safe so custom implementations can report malformed walks as ordinary
/// errors rather than creating undefined behavior.
///
/// Object-safe: `&dyn Module` drives the model-level utilities and, via
/// stable dyn upcasting (Rust 1.86), the crate-private `Sequential` layer.
pub trait Module {
    /// Walk parameters read-only (see [`Visitor`]).
    fn visit(&self, visitor: &mut Visitor);

    /// Walk parameters mutably (see [`VisitorMut`]).
    fn visit_mut(&mut self, visitor: &mut VisitorMut);
}

/// A module that maps an `Input` to an [`Output`](Forward::Output) under a
/// [`Mode`] (exploration §4.1). `&mut self` is honest about layer state
/// (dropout RNG, `BatchNorm` running stats as plain fields — no interior
/// mutability, no mutexes).
///
/// # Why `Input` is a type parameter
///
/// `Mode` is the crate-owned axis set and stays closed — `record` is the
/// alternative to a global no-grad switch and must reach every
/// [`Param::get`], so every layer relies on every axis existing, which only
/// works if the crate owns the set (and lets rstorch add axes in a minor
/// release without breaking anyone). `Input` is therefore the *user's*
/// channel: a layer that needs more than one tensor — an attention mask, a
/// sequence-length vector, a conditioning embedding — declares a struct and
/// implements `Forward<ThatStruct>`, instead of smuggling the extra state
/// through `&mut self` in call order.
///
/// `Input` defaults to [`Tensor`], so the single-tensor spelling
/// `impl Forward for Relu` is unchanged. A trait object must still name the
/// associated type, so the tensor-to-tensor object is spelled
/// `dyn Forward<Tensor, Output = Tensor>`.
///
/// # What this is not
///
/// It does not produce a heterogeneous [`Sequential`] where some layers take a
/// mask and others do not: every member of one `Sequential<I>` shares `I`. A
/// context-carrying model destructures at the top and calls its inner layers
/// with tensors. The win is that multi-input layers live *inside* the trait
/// system and that user context has a statically checked home.
pub trait Forward<Input = Tensor> {
    /// What the forward pass produces. `Tensor` for every layer in the crate's
    /// own zoo; a tuple or a struct for a layer that returns more than one
    /// value.
    type Output;

    /// Run the forward pass.
    ///
    /// `mode` selects layer behavior and whether parameter access returns
    /// traced leaves. It is not a global no-grad switch: operations can still
    /// propagate an existing input graph, and layers without parameters (such
    /// as [`Dropout`]) follow the input's tracing state.
    ///
    /// # Errors
    ///
    /// Implementation-defined: propagates whatever error the layer's own
    /// tensor ops return, typically a shape or dtype mismatch against `input`.
    fn forward(&mut self, input: &Input, mode: Mode) -> Result<Self::Output>;
}
