//! Compile-success ("expansion") tests: `#[derive(Module)]` produces a valid
//! `rstorch::nn::Module` impl for every supported field shape, checked against
//! the *real* frozen T01 contracts (trait, visitor method names/signatures,
//! object-safe `&dyn Module` coercion).
//!
//! These deliberately do not *run* the walk: every `Tensor` constructor is a
//! T20 `todo!()`, so no `Param`/`Tensor` value can be built yet. The behavior
//! that can be verified now — that the generated code type-checks against the
//! contracts — is verified here; the token-level path/leaf choices are unit-
//! tested in `rstorch-derive`'s `module` module.

// Whitelisted-primitive / skipped fields exist only to exercise
// classification; they are never read.
#![allow(dead_code)]

// The prelude brings the `Module` *trait* and the `Module` *derive macro*
// (different namespaces, one name) in one glob — the intended user spelling.
use rstorch::prelude::*;

/// If this compiles, `T`'s generated `impl Module` satisfies the real trait,
/// its visitor calls resolved against the real `Visitor`/`VisitorMut`, and `T`
/// is object-safe-usable through the trait.
fn assert_module<T: Module>() {}

/// A trivial hand-written leaf module so the "child module" cases have a real
/// `Module` to recurse into (no derive, no tensors).
struct Leaf;
impl Module for Leaf {
    fn visit(&self, _v: &mut rstorch::nn::Visitor) {}
    fn visit_mut(&mut self, _v: &mut rstorch::nn::VisitorMut) {}
}

#[derive(Module)]
struct WithParam {
    weight: Param,
    bias: Param,
}

#[derive(Module)]
struct WithBuffer {
    running_mean: Tensor,
    running_var: Tensor,
}

#[derive(Module)]
struct WithOptions {
    weight: Param,
    bias: Option<Param>,
    buf: Option<Tensor>,
    head: Option<Leaf>,
}

#[derive(Module)]
struct WithChild {
    fc: Leaf,
    tail: Leaf,
}

#[derive(Module)]
struct WithVec {
    blocks: Vec<Leaf>,
}

#[derive(Module)]
struct WithPrimitives {
    weight: Param,
    p: f32,
    n: usize,
    training: bool,
    name: String,
}

/// Some non-`Module` config type kept via `#[module(skip)]`.
struct Config {
    _lr: f64,
}

#[derive(Module)]
struct WithSkip {
    weight: Param,
    #[module(skip)]
    _cfg: Config,
}

/// Tuple struct: index segments.
#[derive(Module)]
struct TupleModule(Param, Tensor, Leaf);

/// Unit struct: valid, empty walks.
#[derive(Module)]
struct UnitModule;

/// Generic module: the derive threads generics/where-clauses through; the
/// child-module bound is the caller's responsibility.
#[derive(Module)]
struct Generic<M: Module> {
    inner: M,
}

/// A realistic composite mixing every field kind.
#[derive(Module)]
struct Composite {
    weight: Param,
    bias: Option<Param>,
    running_mean: Tensor,
    blocks: Vec<Leaf>,
    head: Leaf,
    #[module(skip)]
    _dropout_p: f32,
    n_layers: usize,
}

#[test]
fn all_supported_shapes_implement_module() {
    assert_module::<WithParam>();
    assert_module::<WithBuffer>();
    assert_module::<WithOptions>();
    assert_module::<WithChild>();
    assert_module::<WithVec>();
    assert_module::<WithPrimitives>();
    assert_module::<WithSkip>();
    assert_module::<TupleModule>();
    assert_module::<UnitModule>();
    assert_module::<Generic<Leaf>>();
    assert_module::<Composite>();
}

/// `&dyn Module` coercion (object safety of the generated impl in use).
#[test]
fn derived_module_is_object_safe() {
    fn takes_dyn(_m: &dyn Module) {}
    let m = UnitModule;
    takes_dyn(&m);
}
