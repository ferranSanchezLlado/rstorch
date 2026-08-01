use rstorch_derive::TypedModule;

struct Child;

#[derive(TypedModule)]
struct Net {
    #[typed_module(skip)]
    children: Vec<Child>,
}

fn main() {}
