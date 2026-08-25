#![warn(missing_docs)]

//! Derive macros for `rstorch`.
//!
//! Provides `#[derive(Module)]`: loud-by-default field classification, with
//! `#[module(param)]`, `#[module(buffer)]`, `#[module(child)]`, and
//! `#[module(skip)]` as explicit field controls.
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
//! | `Vec<M>` | `v.module("name.0", &self.name[0])`, `…("name.1", …)`, … (indexed) |
//! | `Option<M>` | `if let Some(m) = &self.name { v.module("name", m) }` |
//! | primitive whitelist (`f32`, `f64`, `usize`, `bool`, …) | skipped |
//! | `#[module(skip)]` (any type) | skipped |
//! | **anything else** | `v.module("name", &self.name)` — child module |
//!
//! The `_mut` walk mirrors this against `visit_mut` / the `&mut` visitor
//! methods.
//!
//! `#[module(skip)]` is an explicit escape hatch. On the dynamic derive it can
//! skip any field, including a `Param`; a skipped parameter is intentionally
//! absent from optimizer and state-dict walks. The typed derive applies
//! stricter validation to typed leaves and containers; see its differences
//! below.
//!
//! ## Type aliases and explicit overrides
//!
//! Classification is normally a syntactic token match on the type as written,
//! so a type alias defeats inference. Use an explicit field override when an
//! alias is intentional. A plain newtype does not coerce automatically; add a
//! suitable `Deref` conversion or hand-write its `Module` implementation.
//!
//! ```ignore
//! #[derive(rstorch::Module)]
//! struct Layer {
//!     #[module(param)]
//!     weight: Weights,
//!     #[module(buffer)]
//!     running_mean: Tensor,
//!     #[module(child)]
//!     encoder: Encoder,
//! }
//! ```
//!
//! An override is checked before inference and emits the selected visitor
//! call. `#[module(skip)]` remains the explicit opt-out for dynamic modules.
//!
//! # `#[derive(TypedModule)]`
//!
//! Generates both walks of the `rstorch::typed::nn::Module` trait for a
//! `struct`. Classification is again a syntactic token match, but the table
//! differs from the dynamic derive in one way that bites immediately:
//! **there is no primitive whitelist.**
//!
//! | field type (as written) | emitted call |
//! |---|---|
//! | `TypedParam<T>` | `v.param("name", &self.name)` |
//! | `Option<TypedParam<T>>` | `if let Some(p) = … { v.param("name", p) }` |
//! | `TypedBuffer<T>` | `v.buffer("name", &self.name)` |
//! | `Option<TypedBuffer<T>>` | `if let Some(b) = … { v.buffer("name", b) }` |
//! | `Vec<M>` | `v.module("name.0", …)`, `…("name.1", …)`, … (indexed) |
//! | `Option<M>` | `if let Some(m) = … { v.module("name", m) }` |
//! | `#[typed_module(skip)]` on an opaque direct field | skipped |
//! | **anything else** | `v.module("name", &self.name)` — child module |
//!
//! ## Differences from `#[derive(Module)]`
//!
//! - **No primitive whitelist.** `hidden: usize` is skipped by the dynamic
//!   derive but is a child module to this one, so it fails with
//!   `usize: Module is not satisfied`. Configuration fields need an explicit
//!   `#[typed_module(skip)]`.
//! - **`skip` is classified first, then rejected.** It cannot exclude a
//!   `TypedParam`, a `TypedBuffer`, or an `Option`/`Vec` container, because a
//!   container may hold module state. A plain configuration value therefore
//!   has no directly skippable spelling when it is written as `Vec<T>` or
//!   `Option<T>`: wrap it in a newtype (`struct Config(Vec<usize>)`) and skip
//!   that, or hand-write `impl Module`.
//! - **The helper attribute is field-only.** A struct-level
//!   `#[typed_module(...)]` is a compile error rather than being silently
//!   ignored.
//!
//! Type aliases defeat classification here too, and for the same reason: an
//! alias for `TypedParam<T>` is treated as a child module.

mod module;
mod shared;
mod typed_module;

use proc_macro::TokenStream;

/// Derive the `rstorch::nn::Module` trait for a `struct`.
///
/// See the [crate docs](crate) for the full field-classification table and
/// the loud-by-default rule. Named-field, tuple, and unit `struct`s are all
/// supported — a unit struct derives an empty walk, which is what a stateless
/// layer such as `rstorch::nn::Relu` wants. `enum`s and `union`s are a
/// compile error, because a module's field set must be statically known for
/// every parameter to be visited.
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

/// Derive the `rstorch::typed::nn::Module` trait for a `struct`.
///
/// `TypedParam<T>` and `TypedBuffer<T>` fields are emitted as typed leaves;
/// `Option` and `Vec` containers are walked, and every other field is treated
/// as a typed child module. Use `#[typed_module(skip)]` to explicitly exclude
/// an opaque direct configuration field that is not part of the module tree.
/// The attribute is rejected on typed leaves and on `Option` or `Vec` fields,
/// because those containers may hold module state. A proc macro cannot tell an
/// opaque configuration type from a concrete child module, so applying `skip`
/// to such a child is an explicit contract violation by the caller.
#[proc_macro_derive(TypedModule, attributes(typed_module))]
pub fn derive_typed_module(input: TokenStream) -> TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    typed_module::expand(input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
