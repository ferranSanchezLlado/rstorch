//! Neural-network modules: the [`Module`]/[`Forward`] traits, [`Param`],
//! [`Mode`], the parameter [`Visitor`]s, and (from wave 4 on) the layer zoo
//! (exploration §4.4).
//!
//! Five public traits exist in the entire library; two of them —
//! [`Module`] and [`Forward`] — live here. Both are object-safe.

mod mode;
mod param;
mod util;
pub(crate) mod visit;

pub use mode::Mode;
pub use param::Param;
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
