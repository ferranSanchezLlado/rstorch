//! Registering `typed_module` as a helper attribute puts it in scope on the
//! struct as well as on its fields, where rustc accepts it without complaint.
//! A misplaced attribute must be reported, not silently ignored — a user who
//! writes it here would otherwise believe a field had been excluded.
use rstorch_derive::TypedModule;

#[derive(TypedModule)]
#[typed_module(skip)]
struct Net {
    child: Child,
}

struct Child;

fn main() {}
