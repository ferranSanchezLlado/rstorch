//! Expansion for `#[derive(Module)]`.
//!
//! Field classification first honors explicit `#[module(param)]`,
//! `#[module(buffer)]`, `#[module(child)]`, and `#[module(skip)]` overrides,
//! then falls back to syntactic type matching. This keeps aliases loud by
//! default while providing an intentional escape hatch.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{DeriveInput, PathArguments, Result, Type, TypePath};

use crate::shared::{self, FieldKind};

/// The helper attribute's name, as registered by `#[proc_macro_derive]` and
/// as named by the `skip` parser's error message.
const ATTR: &str = "module";

/// Primitive scalar / small-type whitelist. Fields of
/// these types are configuration, not parameters, and are silently skipped.
/// Any type *not* on this list (and not `Param`/`Option<Param>`/`Tensor`) is
/// treated as a child module so it must implement `Module`.
const PRIMITIVE_WHITELIST: &[&str] = &[
    "f32", "f64", "i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64", "u128",
    "usize", "bool", "char", "str", "String",
];

#[cfg(test)]
/// Top-level entry used by unit tests and direct expansion tests. The proc
/// macro entry point uses [`expand_with_crate`] so downstream dependency
/// renames resolve to the name visible in the caller's manifest.
pub(crate) fn expand(input: DeriveInput) -> Result<TokenStream> {
    expand_with_crate(input, quote!(::rstorch))
}

/// Build the dynamic implementation using the caller-visible runtime crate
/// path. `rstorch_crate` is either `crate` for an in-crate expansion or an
/// absolute dependency path such as `::rstorch_alias` downstream.
pub(crate) fn expand_with_crate(
    input: DeriveInput,
    rstorch_crate: TokenStream,
) -> Result<TokenStream> {
    let spec = shared::Spec {
        derive_name: "Module",
        parameter_noun: "parameter",
        // `#[module(...)]` on the struct is not policed by this derive.
        field_only_attr: None,
        trait_path: quote!(#rstorch_crate::nn::Module),
        visitor: quote!(#rstorch_crate::nn::Visitor),
        visitor_mut: quote!(#rstorch_crate::nn::VisitorMut),
    };
    shared::expand(input, &spec, classify)
}

/// Classify a field by an explicit override first, then by a syntactic token
/// match on its type.
fn classify(field: &syn::Field) -> Result<FieldKind> {
    if let Some(override_kind) = shared::explicit_field_override(field, ATTR)? {
        return Ok(match override_kind {
            shared::FieldOverride::Param => FieldKind::Param,
            shared::FieldOverride::Buffer => FieldKind::Buffer,
            shared::FieldOverride::Child => FieldKind::Module,
            shared::FieldOverride::Skip => FieldKind::Skip,
        });
    }

    let Type::Path(type_path) = &field.ty else {
        // `&T`, `[T; N]`, `(A, B)`, `dyn Trait`, fn pointers, … are not on the
        // whitelist and are not `Path` types, so they default to child-module
        // recursion (loud). Emit that default; it will fail to compile with a
        // clear `Module` bound error if the type is not a module.
        return Ok(FieldKind::Module);
    };

    // The final path segment carries the ident and any `<...>` arguments.
    let Some(last) = type_path.path.segments.last() else {
        return Ok(FieldKind::Module);
    };
    let ident = last.ident.to_string();

    match ident.as_str() {
        "Param" if is_bare(last) => Ok(FieldKind::Param),
        "Tensor" if is_bare(last) => Ok(FieldKind::Buffer),
        "Vec" => {
            // `Vec<M>` is always indexed module recursion. `Vec<Param>` /
            // `Vec<Tensor>` are deliberately unsupported: their leaf paths
            // would collide with a bare-field convention, and the layer zoo
            // holds parameter lists as `Vec<Linear>` etc., never `Vec<Param>`.
            Ok(FieldKind::VecModule)
        }
        "Option" => match shared::inner_of_angle(last) {
            Some(inner) => Ok(classify_option_inner(inner)),
            // `Option` with no/odd arguments: treat as a child module (loud).
            None => Ok(FieldKind::Module),
        },
        prim if PRIMITIVE_WHITELIST.contains(&prim) && is_bare(last) => Ok(FieldKind::Skip),
        // Everything else — including a type alias for `Param`/`Tensor`, which
        // by design is not recognized — defaults to child-module recursion.
        _ => Ok(FieldKind::Module),
    }
}

/// Classify the inner type of an `Option<...>`.
fn classify_option_inner(inner: &Type) -> FieldKind {
    // let-chain (stable since Rust 1.88, the crate MSRV): recognize a bare
    // `Param`/`Tensor` inner; everything else is a child module.
    if let Type::Path(TypePath { path, .. }) = inner
        && let Some(seg) = path.segments.last()
        && is_bare(seg)
    {
        match seg.ident.to_string().as_str() {
            "Param" => return FieldKind::OptionParam,
            "Tensor" => return FieldKind::OptionBuffer,
            _ => {}
        }
    }
    // `Option<M>` for any other `M` (or a non-path inner type) recurses as a
    // child module when present.
    FieldKind::OptionModule
}

/// Whether a path segment has no generic arguments (`Param`, not `Param<T>`).
fn is_bare(seg: &syn::PathSegment) -> bool {
    matches!(seg.arguments, PathArguments::None)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expand `input` (a `struct` definition) and return the generated impl as
    /// a single whitespace-collapsed string for substring assertions.
    fn expand_str(input: &str) -> String {
        let parsed: DeriveInput = syn::parse_str(input).expect("parse");
        let tokens = expand(parsed).expect("expand");
        // `TokenStream` display already inserts single spaces between tokens,
        // which is exactly the normal form we assert against.
        tokens.to_string()
    }

    fn expand_err(input: &str) -> String {
        let parsed: DeriveInput = syn::parse_str(input).expect("parse");
        expand(parsed).expect_err("expected error").to_string()
    }

    #[test]
    fn param_field_emits_param_leaf() {
        let out = expand_str("struct M { weight: Param }");
        assert!(
            out.contains("visitor . param (\"weight\" , & self . weight)"),
            "visit body wrong: {out}"
        );
        assert!(
            out.contains("visitor . param (\"weight\" , & mut self . weight)"),
            "visit_mut body wrong: {out}"
        );
    }

    #[test]
    fn tensor_field_emits_buffer_leaf() {
        let out = expand_str("struct M { running_mean: Tensor }");
        assert!(out.contains("visitor . buffer (\"running_mean\" , & self . running_mean)"));
        assert!(out.contains("visitor . buffer (\"running_mean\" , & mut self . running_mean)"));
    }

    #[test]
    fn unrecognized_field_defaults_to_child_module() {
        // `Linear` is not on the whitelist -> child-module recursion (loud).
        let out = expand_str("struct M { fc: Linear }");
        assert!(out.contains("visitor . module (\"fc\" , & self . fc)"));
        assert!(out.contains("visitor . module (\"fc\" , & mut self . fc)"));
    }

    #[test]
    fn primitive_whitelist_is_skipped() {
        let out = expand_str("struct M { p: f32, n: usize, flag: bool, w: Param }");
        // Only the Param produces a visitor call; primitives emit nothing.
        assert!(out.contains("visitor . param (\"w\""));
        assert!(!out.contains("\"p\""));
        assert!(!out.contains("\"n\""));
        assert!(!out.contains("\"flag\""));
    }

    #[test]
    fn option_param_emits_conditional_param() {
        let out = expand_str("struct M { bias: Option<Param> }");
        assert!(out.contains("if let :: core :: option :: Option :: Some"));
        assert!(out.contains("visitor . param (\"bias\" , __param)"));
    }

    #[test]
    fn option_tensor_emits_conditional_buffer() {
        let out = expand_str("struct M { buf: Option<Tensor> }");
        assert!(out.contains("visitor . buffer (\"buf\" , __buffer)"));
    }

    #[test]
    fn option_module_emits_conditional_module() {
        let out = expand_str("struct M { head: Option<Linear> }");
        assert!(out.contains("visitor . module (\"head\" , __module)"));
    }

    #[test]
    fn vec_module_is_indexed() {
        let out = expand_str("struct M { blocks: Vec<Block> }");
        // Indexed child segment = the FIELD NAME plus the element index, so
        // paths read `blocks.3.attn.qkv.weight` per the frozen path format.
        assert!(out.contains("self . blocks . iter () . enumerate ()"));
        assert!(out.contains("self . blocks . iter_mut () . enumerate ()"));
        assert!(out.contains(r#"format ! ("{}.{}" , "blocks" , __index)"#));
    }

    #[test]
    fn module_skip_opts_out() {
        let out = expand_str("struct M { #[module(skip)] cfg: NotAModule, w: Param }");
        assert!(out.contains("visitor . param (\"w\""));
        assert!(!out.contains("\"cfg\""));
    }

    #[test]
    fn tuple_struct_uses_index_segments() {
        let out = expand_str("struct M(Param, Tensor);");
        assert!(out.contains("visitor . param (\"0\" , & self . 0)"));
        assert!(out.contains("visitor . buffer (\"1\" , & self . 1)"));
    }

    #[test]
    fn type_alias_is_not_recognized_as_param() {
        // A type alias for `Param` defeats the syntactic token match and falls
        // to the loud default (child module). We only assert the *macro's*
        // choice here; the `Module`-bound compile error is the trybuild UI
        // case `type_alias_param`.
        let out = expand_str("struct M { w: Weights }");
        assert!(out.contains("visitor . module (\"w\" , & self . w)"));
    }

    #[test]
    fn explicit_param_override_recognizes_an_alias() {
        let out = expand_str("struct M { #[module(param)] w: Weights }");
        assert!(out.contains("visitor . param (\"w\" , & self . w)"));
        assert!(out.contains("visitor . param (\"w\" , & mut self . w)"));
    }

    #[test]
    fn generic_struct_gets_bounds_from_where_clause() {
        // Generics flow through untouched; the caller supplies the bound.
        let out = expand_str("struct M<T> where T: ::rstorch::nn::Module { inner: T }");
        assert!(out.contains("impl < T > :: rstorch :: nn :: Module for M < T >"));
        assert!(out.contains("visitor . module (\"inner\" , & self . inner)"));
    }

    #[test]
    fn enum_is_rejected() {
        let err = expand_err("enum E { A, B }");
        assert!(err.contains("structs only"), "{err}");
    }

    #[test]
    fn union_is_rejected() {
        let err = expand_err("union U { a: f32, b: u32 }");
        assert!(err.contains("structs only"), "{err}");
    }

    #[test]
    fn unknown_module_option_is_rejected() {
        let err = expand_err("struct M { #[module(rename = \"x\")] w: Param }");
        assert!(
            err.contains("supported options are `param`, `buffer`, `child`, and `skip`"),
            "{err}"
        );
    }

    #[test]
    fn unit_struct_expands_to_empty_walks() {
        // No fields: valid, trivially a module.
        let out = expand_str("struct M;");
        assert!(out.contains("impl :: rstorch :: nn :: Module for M"));
        assert!(!out.contains("visitor ."));
    }
}
