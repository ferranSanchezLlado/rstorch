//! Machinery shared by `#[derive(Module)]` and `#[derive(TypedModule)]`.
//!
//! The two derives differ in exactly two places: how they *classify* a field
//! (each has its own `classify`, because the recognized spellings and the
//! `skip` rules genuinely differ) and which trait and visitor types the
//! generated impl names. Everything else — rejecting enums and unions,
//! walking the fields, the emitted visitor statements, and the
//! `#[...(skip)]` parser — is identical for both and lives here, so the two
//! walks cannot drift apart.

use proc_macro2::TokenStream;
use quote::quote;
use syn::spanned::Spanned;
use syn::{Data, DeriveInput, Error, GenericArgument, PathArguments, Result, Type};

/// How a single field participates in the parameter walks.
///
/// The two derives spell their leaves differently — `Param` / `Tensor` for
/// `#[derive(Module)]`, `TypedParam<T>` / `TypedBuffer<T>` for
/// `#[derive(TypedModule)]` — but both emit the same visitor calls, so one
/// set of kinds drives both.
pub(crate) enum FieldKind {
    /// A parameter leaf → `v.param(name, &self.field)`.
    Param,
    /// `Option<parameter leaf>` → emit `param` when `Some`.
    OptionParam,
    /// A buffer leaf → `v.buffer(name, &self.field)`.
    Buffer,
    /// `Option<buffer leaf>` → emit `buffer` when `Some`.
    OptionBuffer,
    /// `Vec<M>` → indexed `v.module("field.i", &self.field[i])`.
    VecModule,
    /// `Option<M>` (M not a leaf) → emit `module` when `Some`.
    OptionModule,
    /// A whitelisted primitive (or an explicit `skip`): not visited.
    Skip,
    /// Anything else: a child module (`v.module(name, &self.field)`). This is
    /// the loud default — a non-`Module` field is a compile error here.
    Module,
}

/// Everything that distinguishes one derive from the other inside [`expand`].
pub(crate) struct Spec {
    /// Derive name as written (`Module` / `TypedModule`), used in the
    /// enum/union rejection messages.
    pub(crate) derive_name: &'static str,
    /// How the enum rejection names a visited parameter: `"parameter"` for
    /// the dynamic derive, `"typed parameter"` for the typed one.
    pub(crate) parameter_noun: &'static str,
    /// Helper attribute that is field-only and must be rejected when written
    /// on the struct, or `None` when the derive does not police it.
    pub(crate) field_only_attr: Option<&'static str>,
    /// Trait the generated impl implements.
    pub(crate) trait_path: TokenStream,
    /// Visitor type taken by `visit`.
    pub(crate) visitor: TokenStream,
    /// Visitor type taken by `visit_mut`.
    pub(crate) visitor_mut: TokenStream,
}

/// Build the `impl` for `input`, classifying each field with `classify`.
pub(crate) fn expand(
    input: DeriveInput,
    spec: &Spec,
    classify: fn(&syn::Field) -> Result<FieldKind>,
) -> Result<TokenStream> {
    let derive_name = spec.derive_name;
    let fields = match &input.data {
        Data::Struct(data) => &data.fields,
        Data::Enum(_) => {
            return Err(Error::new_spanned(
                &input,
                format!(
                    "#[derive({derive_name})] supports structs only, not enums \
                     (a module's field set must be statically known so every \
                     {} is visited)",
                    spec.parameter_noun,
                ),
            ));
        }
        Data::Union(_) => {
            return Err(Error::new_spanned(
                &input,
                format!("#[derive({derive_name})] supports structs only, not unions"),
            ));
        }
    };

    // The helper attribute is only meaningful on a field. Registering it puts
    // `#[...(...)]` in scope on the struct too, where rustc accepts it
    // silently — so a misplaced or misspelled attribute would otherwise be
    // ignored rather than reported, exactly the silence this derive exists to
    // avoid.
    if let Some(name) = spec.field_only_attr
        && let Some(attr) = input.attrs.iter().find(|a| a.path().is_ident(name))
    {
        return Err(Error::new_spanned(
            attr,
            format!(
                "#[{name}(...)] applies to a field, not to the struct; \
                 remove it or move it onto the field it should govern"
            ),
        ));
    }

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
    let trait_path = &spec.trait_path;
    let visitor = &spec.visitor;
    let visitor_mut = &spec.visitor_mut;

    Ok(quote! {
        #[automatically_derived]
        impl #impl_generics #trait_path for #name #ty_generics #where_clause {
            fn visit(&self, visitor: &mut #visitor) {
                #(#visit_calls)*
            }

            fn visit_mut(&mut self, visitor: &mut #visitor_mut) {
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
                for (__index, __module) in #iter.enumerate() {
                    visitor.module(&::std::format!("{}.{}", #segment, __index), __module);
                }
            }
        }
    }
}

/// Extract the single type argument of `T<Inner>`, else `None`.
pub(crate) fn inner_of_angle(segment: &syn::PathSegment) -> Option<&Type> {
    let PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return None;
    };
    // Exactly one type argument (reject `Option<A, B>`-style oddities).
    if arguments.args.len() != 1 {
        return None;
    }
    match arguments.args.first()? {
        GenericArgument::Type(ty) => Some(ty),
        _ => None,
    }
}

/// Read `#[<attr>(skip)]`. Any other `#[<attr>(...)]` content is a hard error
/// (a typo must not silently do nothing).
pub(crate) fn has_skip_attr(field: &syn::Field, attr_name: &str) -> Result<bool> {
    let mut skip = false;
    for attr in &field.attrs {
        if !attr.path().is_ident(attr_name) {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                skip = true;
                Ok(())
            } else {
                Err(Error::new(
                    meta.path.span(),
                    format!(
                        "unknown #[{attr_name}(...)] option; the only supported option is `skip`"
                    ),
                ))
            }
        })?;
    }
    Ok(skip)
}
