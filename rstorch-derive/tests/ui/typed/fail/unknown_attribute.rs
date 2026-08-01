use rstorch_derive::TypedModule;

#[derive(TypedModule)]
struct Net {
    #[typed_module(rename = "other")]
    child: Child,
}

struct Child;

fn main() {}
