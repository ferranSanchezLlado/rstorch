//! Parameter/buffer visitors with dotted paths (exploration §4.4).
//!
//! [`Module::visit`](crate::nn::Module::visit) and
//! [`visit_mut`](crate::nn::Module::visit_mut) walk a module tree, calling
//! into a [`Visitor`]/[`VisitorMut`] which threads a dotted path prefix
//! (`fc1.weight`, `blocks.3.attn.qkv.weight`) and forwards each leaf to a
//! single sink. `#[derive(Module)]` (T13) generates the walk: one method
//! call per field — [`param`](Visitor::param) for a [`Param`],
//! [`buffer`](Visitor::buffer) for a whitelisted `Tensor` buffer (e.g.
//! BatchNorm `running_mean`), and [`module`](Visitor::module) for a child
//! module. These path semantics are the on-disk `state_dict` key format, so
//! they are frozen here (a §9 risk item).
//!
//! **Params vs buffers** (exploration §4.4 whitelist): a `Param` is trainable
//! and optimizer-visited; a `Tensor` buffer is non-trainable persistent state
//! (running statistics). Both are moved by `nn::to_device`/`to_dtype` and both
//! land in `state_dict` (so a checkpoint reconstructs a model, as PyTorch
//! does); only `Param`s count toward `num_params` and receive gradients.

use crate::nn::{Module, Param};
use crate::tensor::Tensor;

fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

/// A read-only leaf handed to a visitor sink: a trainable [`Param`] or a
/// non-trainable `Tensor` buffer, distinguished so sinks can treat them
/// differently (e.g. `num_params` counts only params; `state_dict` keeps
/// both). Crate-private — the public visitor API is the `param`/`buffer`
/// methods, not this enum.
pub(crate) enum Leaf<'a> {
    Param(&'a Param),
    Buffer(&'a Tensor),
}

/// The mutable counterpart of [`Leaf`]. The variants are destructured by
/// T40/T44 (the mutable-walk consumers); until then the fields are unread.
#[allow(dead_code)]
pub(crate) enum LeafMut<'a> {
    Param(&'a mut Param),
    Buffer(&'a mut Tensor),
}

/// The read-only leaf visitor threaded through
/// [`Module::visit`](crate::nn::Module::visit).
///
/// `#[derive(Module)]` emits, per field: [`param`](Visitor::param) for a
/// [`Param`], [`buffer`](Visitor::buffer) for a whitelisted `Tensor` buffer,
/// [`module`](Visitor::module) for a child module, and an indexed
/// [`module`](Visitor::module) call per element for `Vec<M>`/`Option<M>`
/// (child name `"3"`, producing paths like `blocks.3.weight`).
pub struct Visitor<'a> {
    path: String,
    sink: &'a mut dyn FnMut(&str, Leaf<'_>),
}

impl<'a> Visitor<'a> {
    pub(crate) fn new(sink: &'a mut dyn FnMut(&str, Leaf<'_>)) -> Visitor<'a> {
        Visitor {
            path: String::new(),
            sink,
        }
    }

    /// Emit a trainable parameter leaf named `name` at the current prefix.
    pub fn param(&mut self, name: &str, p: &Param) {
        let full = join(&self.path, name);
        (self.sink)(&full, Leaf::Param(p));
    }

    /// Emit a non-trainable `Tensor` buffer leaf named `name` at the current
    /// prefix (persistent state such as BatchNorm running statistics).
    pub fn buffer(&mut self, name: &str, t: &Tensor) {
        let full = join(&self.path, name);
        (self.sink)(&full, Leaf::Buffer(t));
    }

    /// Descend into child module `child` under segment `name`, prefixing all
    /// of its leaf paths with `name.`.
    pub fn module(&mut self, name: &str, child: &dyn Module) {
        let saved = self.path.len();
        if !self.path.is_empty() {
            self.path.push('.');
        }
        self.path.push_str(name);
        child.visit(self);
        self.path.truncate(saved);
    }
}

/// The mutable counterpart of [`Visitor`], threaded through
/// [`Module::visit_mut`](crate::nn::Module::visit_mut). Same path semantics;
/// the sink receives a `LeafMut` (optimizer step, `load_state_dict`,
/// `to_device`/`to_dtype`).
pub struct VisitorMut<'a> {
    path: String,
    sink: &'a mut dyn FnMut(&str, LeafMut<'_>),
}

impl<'a> VisitorMut<'a> {
    pub(crate) fn new(sink: &'a mut dyn FnMut(&str, LeafMut<'_>)) -> VisitorMut<'a> {
        VisitorMut {
            path: String::new(),
            sink,
        }
    }

    /// Emit a mutable parameter leaf named `name` at the current prefix.
    pub fn param(&mut self, name: &str, p: &mut Param) {
        let full = join(&self.path, name);
        (self.sink)(&full, LeafMut::Param(p));
    }

    /// Emit a mutable `Tensor` buffer leaf named `name` at the current prefix.
    pub fn buffer(&mut self, name: &str, t: &mut Tensor) {
        let full = join(&self.path, name);
        (self.sink)(&full, LeafMut::Buffer(t));
    }

    /// Descend into mutable child module `child` under segment `name`.
    pub fn module(&mut self, name: &str, child: &mut dyn Module) {
        let saved = self.path.len();
        if !self.path.is_empty() {
            self.path.push('.');
        }
        self.path.push_str(name);
        child.visit_mut(self);
        self.path.truncate(saved);
    }
}

/// Run `sink` over every `(dotted_path, Leaf)` in `module` — parameters and
/// buffers (the shared engine behind `num_params` and `state_dict`).
pub(crate) fn visit_all(module: &dyn Module, sink: &mut dyn FnMut(&str, Leaf<'_>)) {
    let mut v = Visitor::new(sink);
    module.visit(&mut v);
}

/// Run `sink` over every `(dotted_path, LeafMut)` in `module` (optimizer
/// step, `load_state_dict`, device/dtype conversion).
// Consumed by T40 (util) and T44 (optimizer); the integrator removes this
// allow once those land.
#[allow(dead_code)]
pub(crate) fn visit_all_mut(module: &mut dyn Module, sink: &mut dyn FnMut(&str, LeafMut<'_>)) {
    let mut v = VisitorMut::new(sink);
    module.visit_mut(&mut v);
}
