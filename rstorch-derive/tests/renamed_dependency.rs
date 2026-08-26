//! Compile coverage for derives used through a renamed `rstorch` dependency.
//!
//! The dependency is declared as `rstorch_alias` in this package's manifest.
//! The derive must therefore resolve generated trait and visitor paths through
//! `::rstorch_alias`, rather than assuming the package name is an identifier
//! available as `::rstorch`.

use rstorch_alias::typed::nn::{Module, TypedModule};

#[derive(TypedModule)]
struct AliasedTypedModule;

fn assert_typed_module<T: Module>() {}

#[test]
fn typed_derive_resolves_renamed_runtime_dependency() {
    assert_typed_module::<AliasedTypedModule>();
}
