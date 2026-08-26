//! Expansion for `#[derive(TypedModule)]`.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{DeriveInput, Error, Result, Type, TypePath};

use crate::shared::{self, FieldKind};

/// The helper attribute's name, shared by the struct-level rejection and the
/// per-field parser so the two cannot drift apart.
const ATTR: &str = "typed_module";

#[cfg(test)]
/// Top-level entry used by unit tests and direct expansion tests. The proc
/// macro entry point uses [`expand_with_crate`] so downstream dependency
/// renames resolve to the name visible in the caller's manifest.
pub(crate) fn expand(input: DeriveInput) -> Result<TokenStream> {
    expand_with_crate(input, quote!(::rstorch))
}

/// Build the typed implementation using the caller-visible runtime crate
/// path. `rstorch_crate` is either `crate` for an in-crate expansion or an
/// absolute dependency path such as `::rstorch_alias` downstream.
pub(crate) fn expand_with_crate(
    input: DeriveInput,
    rstorch_crate: TokenStream,
) -> Result<TokenStream> {
    let spec = shared::Spec {
        derive_name: "TypedModule",
        parameter_noun: "typed parameter",
        field_only_attr: Some(ATTR),
        trait_path: quote!(#rstorch_crate::typed::nn::Module),
        visitor: quote!(#rstorch_crate::typed::nn::TypedVisitor<'_>),
        visitor_mut: quote!(#rstorch_crate::typed::nn::TypedVisitorMut<'_>),
    };
    shared::expand(input, &spec, classify)
}

fn classify(field: &syn::Field) -> Result<FieldKind> {
    // Classify first so an explicit opt-out cannot hide state whose spelling
    // the derive knows how to walk.
    let kind = classify_type(&field.ty);
    if !shared::has_skip_attr(field, ATTR)? {
        return Ok(kind);
    }

    match kind {
        FieldKind::Param | FieldKind::OptionParam => Err(Error::new_spanned(
            field,
            "#[typed_module(skip)] cannot exclude a TypedParam field; every typed parameter must be visited",
        )),
        FieldKind::Buffer | FieldKind::OptionBuffer => Err(Error::new_spanned(
            field,
            "#[typed_module(skip)] cannot exclude a TypedBuffer field; every typed buffer must be visited",
        )),
        // A proc macro sees only tokens, so it cannot tell `Option<Config>`
        // from `Option<Linear<2, 2>>`; refusing to skip either is what keeps a
        // child module from going unvisited. Name the escape hatch, because
        // otherwise a plain configuration field such as `Option<String>` has
        // no spelling that compiles: skipping is this error, and not skipping
        // is a `Module` bound error on the inner type.
        FieldKind::OptionModule => Err(Error::new_spanned(
            field,
            "#[typed_module(skip)] cannot exclude an Option field because it may contain a typed child module; \
             wrap the value in a skippable newtype (`struct Config(Option<T>)`) or hand-write `impl Module`",
        )),
        FieldKind::VecModule => Err(Error::new_spanned(
            field,
            "#[typed_module(skip)] cannot exclude a Vec field because it may contain typed child modules; \
             wrap the value in a skippable newtype (`struct Config(Vec<T>)`) or hand-write `impl Module`",
        )),
        // The proc macro cannot ask rustc whether an opaque type implements
        // Module. This is the explicit configuration escape hatch; applying it
        // to a child module intentionally violates the module-walk contract.
        FieldKind::Module => Ok(FieldKind::Skip),
        FieldKind::Skip => unreachable!("type classification never returns Skip"),
    }
}

fn classify_type(ty: &Type) -> FieldKind {
    let Type::Path(type_path) = ty else {
        return FieldKind::Module;
    };
    let Some(last) = type_path.path.segments.last() else {
        return FieldKind::Module;
    };

    match last.ident.to_string().as_str() {
        "TypedParam" if has_one_type_argument(last) => FieldKind::Param,
        "TypedBuffer" if has_one_type_argument(last) => FieldKind::Buffer,
        "Option" => match shared::inner_of_angle(last) {
            Some(inner) => classify_option_inner(inner),
            None => FieldKind::Module,
        },
        "Vec" if has_one_type_argument(last) => FieldKind::VecModule,
        _ => FieldKind::Module,
    }
}

fn classify_option_inner(inner: &Type) -> FieldKind {
    if let Type::Path(TypePath { path, .. }) = inner
        && let Some(segment) = path.segments.last()
        && has_one_type_argument(segment)
    {
        match segment.ident.to_string().as_str() {
            "TypedParam" => return FieldKind::OptionParam,
            "TypedBuffer" => return FieldKind::OptionBuffer,
            _ => {}
        }
    }
    FieldKind::OptionModule
}

fn has_one_type_argument(segment: &syn::PathSegment) -> bool {
    shared::inner_of_angle(segment).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expand_str(input: &str) -> String {
        let input = syn::parse_str(input).expect("parse");
        expand(input).expect("expand").to_string()
    }

    fn expand_err(input: &str) -> String {
        let input = syn::parse_str(input).expect("parse");
        expand(input).expect_err("expected error").to_string()
    }

    #[test]
    fn emits_both_typed_leaf_walks() {
        let out = expand_str(
            "struct Net<T> where T: Bound { weight: TypedParam<T>, state: TypedBuffer<T> }",
        );
        assert!(out.contains("impl < T > :: rstorch :: typed :: nn :: Module for Net < T >"));
        assert!(out.contains("visitor . param (\"weight\" , & self . weight)"));
        assert!(out.contains("visitor . param (\"weight\" , & mut self . weight)"));
        assert!(out.contains("visitor . buffer (\"state\" , & self . state)"));
        assert!(out.contains("visitor . buffer (\"state\" , & mut self . state)"));
    }

    #[test]
    fn option_leaves_and_modules_are_conditional() {
        let out = expand_str(
            "struct Net<T> { p: Option<TypedParam<T>>, b: Option<TypedBuffer<T>>, child: Option<Child> }",
        );
        assert!(out.contains("visitor . param (\"p\" , __param)"));
        assert!(out.contains("visitor . buffer (\"b\" , __buffer)"));
        assert!(out.contains("visitor . module (\"child\" , __module)"));
    }

    #[test]
    fn vec_paths_include_field_and_index() {
        let out = expand_str("struct Net { blocks: Vec<Block> }");
        assert!(out.contains("self . blocks . iter () . enumerate ()"));
        assert!(out.contains("self . blocks . iter_mut () . enumerate ()"));
        assert!(out.contains(r#"format ! ("{}.{}" , "blocks" , __index)"#));
    }

    #[test]
    fn aliases_and_unknown_fields_recurse_loudly() {
        let out = expand_str("struct Net { weight: WeightAlias, count: usize }");
        assert!(out.contains("visitor . module (\"weight\" , & self . weight)"));
        assert!(out.contains("visitor . module (\"count\" , & self . count)"));
    }

    #[test]
    fn skip_and_tuple_paths_are_supported() {
        let out = expand_str("struct Net(#[typed_module(skip)] Config, TypedParam<T>);");
        assert!(!out.contains("\"0\""));
        assert!(out.contains("visitor . param (\"1\" , & self . 1)"));
    }

    #[test]
    fn known_state_and_child_containers_cannot_be_skipped() {
        for input in [
            "struct Net<T> { #[typed_module(skip)] p: TypedParam<T> }",
            "struct Net<T> { #[typed_module(skip)] b: TypedBuffer<T> }",
            "struct Net { #[typed_module(skip)] child: Option<Child> }",
            "struct Net { #[typed_module(skip)] children: Vec<Child> }",
        ] {
            let error = expand_err(input);
            assert!(
                error.contains("cannot exclude"),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn bad_inputs_are_rejected() {
        assert!(expand_err("enum Net { A }").contains("structs only"));
        assert!(expand_err("union Net { a: u8 }").contains("structs only"));
        assert!(
            expand_err("struct Net { #[typed_module(rename = \"x\")] child: Child }")
                .contains("only supported option is `skip`")
        );
    }
}
