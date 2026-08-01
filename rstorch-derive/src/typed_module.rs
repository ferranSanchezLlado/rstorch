//! Expansion for `#[derive(TypedModule)]`.

use proc_macro2::TokenStream;
use quote::quote;
use syn::spanned::Spanned;
use syn::{Data, DeriveInput, Error, GenericArgument, PathArguments, Result, Type, TypePath};

enum FieldKind {
    Param,
    OptionParam,
    Buffer,
    OptionBuffer,
    VecModule,
    OptionModule,
    Skip,
    Module,
}

pub(crate) fn expand(input: DeriveInput) -> Result<TokenStream> {
    let fields = match &input.data {
        Data::Struct(data) => &data.fields,
        Data::Enum(_) => {
            return Err(Error::new_spanned(
                &input,
                "#[derive(TypedModule)] supports structs only, not enums (a module's field set must be statically known so every typed parameter is visited)",
            ));
        }
        Data::Union(_) => {
            return Err(Error::new_spanned(
                &input,
                "#[derive(TypedModule)] supports structs only, not unions",
            ));
        }
    };

    let mut visit_calls = Vec::new();
    let mut visit_mut_calls = Vec::new();
    for (index, field) in fields.iter().enumerate() {
        let (accessor, segment) = match &field.ident {
            Some(ident) => (quote!(#ident), ident.to_string()),
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
        impl #impl_generics ::rstorch::typed::nn::Module for #name #ty_generics #where_clause {
            fn visit(&self, visitor: &mut ::rstorch::typed::nn::TypedVisitor<'_>) {
                #(#visit_calls)*
            }

            fn visit_mut(&mut self, visitor: &mut ::rstorch::typed::nn::TypedVisitorMut<'_>) {
                #(#visit_mut_calls)*
            }
        }
    })
}

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
            if let ::core::option::Option::Some(__param) = #borrow self.#accessor {
                visitor.param(#segment, __param);
            }
        },
        FieldKind::OptionBuffer => quote! {
            if let ::core::option::Option::Some(__buffer) = #borrow self.#accessor {
                visitor.buffer(#segment, __buffer);
            }
        },
        FieldKind::OptionModule => quote! {
            if let ::core::option::Option::Some(__module) = #borrow self.#accessor {
                visitor.module(#segment, __module);
            }
        },
        FieldKind::VecModule => {
            let iter = if is_mut {
                quote!(self.#accessor.iter_mut())
            } else {
                quote!(self.#accessor.iter())
            };
            quote! {
                for (__index, __module) in #iter.enumerate() {
                    visitor.module(&::std::format!("{}.{}", #segment, __index), __module);
                }
            }
        }
    }
}

fn classify(field: &syn::Field) -> Result<FieldKind> {
    // Classify first so an explicit opt-out cannot hide state whose spelling
    // the derive knows how to walk.
    let kind = classify_type(&field.ty);
    if !has_skip_attr(field)? {
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
        FieldKind::OptionModule => Err(Error::new_spanned(
            field,
            "#[typed_module(skip)] cannot exclude an Option field because it may contain a typed child module",
        )),
        FieldKind::VecModule => Err(Error::new_spanned(
            field,
            "#[typed_module(skip)] cannot exclude a Vec field because it may contain typed child modules",
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
        "Option" => match inner_of_angle(last) {
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
    inner_of_angle(segment).is_some()
}

fn inner_of_angle(segment: &syn::PathSegment) -> Option<&Type> {
    let PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return None;
    };
    if arguments.args.len() != 1 {
        return None;
    }
    match arguments.args.first()? {
        GenericArgument::Type(ty) => Some(ty),
        _ => None,
    }
}

fn has_skip_attr(field: &syn::Field) -> Result<bool> {
    let mut skip = false;
    for attr in &field.attrs {
        if !attr.path().is_ident("typed_module") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                skip = true;
                Ok(())
            } else {
                Err(Error::new(
                    meta.path.span(),
                    "unknown #[typed_module(...)] option; the only supported option is `skip`",
                ))
            }
        })?;
    }
    Ok(skip)
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
