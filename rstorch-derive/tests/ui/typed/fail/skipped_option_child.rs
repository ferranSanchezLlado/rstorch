use rstorch_derive::TypedModule;

struct Child;

#[derive(TypedModule)]
struct Net {
    #[typed_module(skip)]
    child: Option<Child>,
}

fn main() {}
