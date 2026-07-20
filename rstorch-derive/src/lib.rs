#![warn(missing_docs)]

//! Derive macros for `rstorch`.
//!
//! Provides `#[derive(Module)]` (task T13): loud-by-default field
//! classification with `#[module(skip)]` as the explicit opt-out — see
//! `docs/restart-v3/exploration.md` §4.4.
//!
//! # `#[derive(Module)]`
//!
//! Generates both walks of the `rstorch::nn::Module` trait (`visit` /
//! `visit_mut`) for a `struct`, emitting one visitor call per field with the
//! field name as its dotted-path segment. Field handling is decided **by
//! token match** on the written type, and is **loud by default**: any field
//! whose type is not on the whitelist is treated as a child module, so a
//! non-`Module` field is a compile error rather than a silently untrained
//! parameter (the exact bug class v3 abolishes).
//!
//! | field type (as written) | emitted call |
//! |---|---|
//! | `Param` | `v.param("name", &self.name)` |
//! | `Option<Param>` | `if let Some(p) = &self.name { v.param("name", p) }` |
//! | `Tensor` | `v.buffer("name", &self.name)` (non-trainable buffer) |
//! | `Option<Tensor>` | `if let Some(t) = … { v.buffer("name", t) }` |
//! | `Vec<M>` | `v.module("0", &self.name[0])`, `…("1", …)`, … (indexed) |
//! | `Option<M>` | `if let Some(m) = &self.name { v.module("name", m) }` |
//! | primitive whitelist (`f32`, `f64`, `usize`, `bool`, …) | skipped |
//! | `#[module(skip)]` (any type) | skipped |
//! | **anything else** | `v.module("name", &self.name)` — child module |
//!
//! The `_mut` walk mirrors this against `visit_mut` / the `&mut` visitor
//! methods.
//!
//! ## Type aliases fail loudly
//!
//! Classification is a **syntactic** token match on the type as written, so a
//! type alias defeats it. `type Weights = rstorch::Param;` used as a field
//! type is **not** recognized as `Param`; it falls through to the default and
//! is treated as a child module, producing a `Module`-not-satisfied compile
//! error. Spell the whitelisted types out (`Param`, `Option<Param>`,
//! `Tensor`) — do not alias them. This is intentional: a silent
//! misclassification of a parameter as a non-visited field is precisely what
//! the loud rule exists to prevent.

mod module;

use proc_macro::TokenStream;

/// Derive the `rstorch::nn::Module` trait for a `struct`.
///
/// See the [crate docs](crate) for the full field-classification table and
/// the loud-by-default rule. Only named-field and tuple `struct`s are
/// supported; unit structs, `enum`s, and `union`s are a compile error.
///
/// # Attributes
///
/// - `#[module(skip)]` on a field opts it out of both walks. It is the only
///   sanctioned way to keep a non-whitelisted, non-`Module` field.
#[proc_macro_derive(Module, attributes(module))]
pub fn derive_module(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    module::expand(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
