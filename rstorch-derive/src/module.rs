//! Expansion for `#[derive(Module)]`.
//!
//! Field classification is a **syntactic token match** on the type as written
//! (see [`classify`]); this is what makes type aliases fail loudly and what
//! makes every unrecognized field default to child-module recursion.

use proc_macro2::TokenStream;
use quote::quote;
use syn::spanned::Spanned;
use syn::{Data, DeriveInput, Error, GenericArgument, PathArguments, Result, Type, TypePath};

/// How a single field participates in the parameter walks.
enum FieldKind {
    /// `Param` → `v.param(name, &self.field)`.
    Param,
    /// `Option<Param>` → emit `param` when `Some`.
    OptionParam,
    /// `Tensor` → `v.buffer(name, &self.field)`.
    Buffer,
    /// `Option<Tensor>` → emit `buffer` when `Some`.
    OptionBuffer,
    /// `Vec<M>` → indexed `v.module("i", &self.field[i])`.
    VecModule,
    /// `Option<M>` (M not `Param`/`Tensor`) → emit `module` when `Some`.
    OptionModule,
    /// A whitelisted primitive (or `#[module(skip)]`): not visited.
    Skip,
    /// Anything else: a child module (`v.module(name, &self.field)`). This is
    /// the loud default — a non-`Module` field is a compile error here.
    Module,
}

/// Primitive scalar / small-type whitelist (exploration §4.4). Fields of
/// these types are configuration, not parameters, and are silently skipped.
/// Any type *not* on this list (and not `Param`/`Option<Param>`/`Tensor`) is
/// treated as a child module so it must implement `Module`.
const PRIMITIVE_WHITELIST: &[&str] = &[
    "f32", "f64", "i8", "i16", "i32", "i64", "i128", "isize", "u8", "u16", "u32", "u64", "u128",
    "usize", "bool", "char", "str", "String",
];

/// Top-level entry: build the `impl Module` for `input`.
pub(crate) fn expand(input: DeriveInput) -> Result<TokenStream> {
    let fields = match &input.data {
        Data::Struct(data) => &data.fields,
        Data::Enum(_) => {
            return Err(Error::new_spanned(
                &input,
                "#[derive(Module)] supports structs only, not enums \
                 (a module's field set must be statically known so every \
                 parameter is visited)",
            ));
        }
        Data::Union(_) => {
            return Err(Error::new_spanned(
                &input,
                "#[derive(Module)] supports structs only, not unions",
            ));
        }
    };

    let mut visit_calls = Vec::new();
    let mut visit_mut_calls = Vec::new();

    for (index, field) in fields.iter().enumerate() {
        // The dotted-path segment and the member used to access the field.
        // Named field: `self.weight`, segment `"weight"`. Tuple field:
        // `self.0`, segment `"0"`.
        let (accessor, segment) = match &field.ident {
            Some(ident) => {
                let seg = ident.to_string();
                (quote!(#ident), seg)
            }
            None => {
                let member = syn::Index::from(index);
                (quote!(#member), index.to_string())
            }
        };

        let kind = classify(field)?;
        visit_calls.push(emit(&kind, &accessor, &segment, false));
        visit_mut_calls.push(emit(&kind, &accessor, &segment, true));
    }

    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    Ok(quote! {
        #[automatically_derived]
        impl #impl_generics ::rstorch::nn::Module for #name #ty_generics #where_clause {
            fn visit(&self, visitor: &mut ::rstorch::nn::Visitor) {
                #(#visit_calls)*
            }

            fn visit_mut(&mut self, visitor: &mut ::rstorch::nn::VisitorMut) {
                #(#visit_mut_calls)*
            }
        }
    })
}

/// Emit the per-field statement for one walk.
///
/// `is_mut` selects `&mut` accessors; the visitor method names are identical
/// across the two walks (`param`/`buffer`/`module`), so only the reference
/// mutability differs.
fn emit(kind: &FieldKind, accessor: &TokenStream, segment: &str, is_mut: bool) -> TokenStream {
    let borrow = if is_mut { quote!(&mut) } else { quote!(&) };
    match kind {
        FieldKind::Skip => quote!(),
        FieldKind::Param => quote! {
            visitor.param(#segment, #borrow self.#accessor);
        },
        FieldKind::Buffer => quote! {
            visitor.buffer(#segment, #borrow self.#accessor);
        },
        FieldKind::Module => quote! {
            visitor.module(#segment, #borrow self.#accessor);
        },
        FieldKind::OptionParam => quote! {
            if let ::core::option::Option::Some(__p) = #borrow self.#accessor {
                visitor.param(#segment, __p);
            }
        },
        FieldKind::OptionBuffer => quote! {
            if let ::core::option::Option::Some(__t) = #borrow self.#accessor {
                visitor.buffer(#segment, __t);
            }
        },
        FieldKind::OptionModule => quote! {
            if let ::core::option::Option::Some(__m) = #borrow self.#accessor {
                visitor.module(#segment, __m);
            }
        },
        FieldKind::VecModule => {
            // Indexed dotted paths: `blocks.0`, `blocks.1`, … The child
            // segment carries BOTH the field name and the index, because
            // nothing descends into the field itself — emitting the bare
            // index would drop `blocks` from every path and make two `Vec`
            // fields in one module collide.
            let iter = if is_mut {
                quote!(self.#accessor.iter_mut())
            } else {
                quote!(self.#accessor.iter())
            };
            quote! {
                for (__i, __m) in #iter.enumerate() {
                    visitor.module(&::std::format!("{}.{}", #segment, __i), __m);
                }
            }
        }
    }
}

/// Classify a field by a syntactic token match on its type, honoring
/// `#[module(skip)]`.
fn classify(field: &syn::Field) -> Result<FieldKind> {
    if has_skip_attr(field)? {
        return Ok(FieldKind::Skip);
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
        "Option" => match inner_of_angle(last) {
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

/// Extract the single type argument of `T<Inner>`, else `None`.
fn inner_of_angle(seg: &syn::PathSegment) -> Option<&Type> {
    let PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    let mut types = args.args.iter().filter_map(|arg| match arg {
        GenericArgument::Type(ty) => Some(ty),
        _ => None,
    });
    let first = types.next()?;
    // Exactly one type argument (reject `Option<A, B>`-style oddities).
    if types.next().is_some() {
        return None;
    }
    Some(first)
}

/// Read `#[module(skip)]`. Any other `#[module(...)]` content is a hard error
/// (a typo must not silently do nothing).
fn has_skip_attr(field: &syn::Field) -> Result<bool> {
    let mut skip = false;
    for attr in &field.attrs {
        if !attr.path().is_ident("module") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                skip = true;
                Ok(())
            } else {
                Err(Error::new(
                    meta.path.span(),
                    "unknown #[module(...)] option; the only supported option is `skip`",
                ))
            }
        })?;
    }
    Ok(skip)
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
        assert!(out.contains("visitor . param (\"bias\" , __p)"));
    }

    #[test]
    fn option_tensor_emits_conditional_buffer() {
        let out = expand_str("struct M { buf: Option<Tensor> }");
        assert!(out.contains("visitor . buffer (\"buf\" , __t)"));
    }

    #[test]
    fn option_module_emits_conditional_module() {
        let out = expand_str("struct M { head: Option<Linear> }");
        assert!(out.contains("visitor . module (\"head\" , __m)"));
    }

    #[test]
    fn vec_module_is_indexed() {
        let out = expand_str("struct M { blocks: Vec<Block> }");
        // Indexed child segment = the FIELD NAME plus the element index, so
        // paths read `blocks.3.attn.qkv.weight` per the frozen path format.
        assert!(out.contains("self . blocks . iter () . enumerate ()"));
        assert!(out.contains("self . blocks . iter_mut () . enumerate ()"));
        assert!(out.contains(r#"format ! ("{}.{}" , "blocks" , __i)"#));
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
        assert!(err.contains("only supported option is `skip`"), "{err}");
    }

    #[test]
    fn unit_struct_expands_to_empty_walks() {
        // No fields: valid, trivially a module.
        let out = expand_str("struct M;");
        assert!(out.contains("impl :: rstorch :: nn :: Module for M"));
        assert!(!out.contains("visitor ."));
    }
}
