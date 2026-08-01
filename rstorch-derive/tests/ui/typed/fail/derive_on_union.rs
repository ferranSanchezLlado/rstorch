use rstorch_derive::TypedModule;

#[derive(TypedModule)]
union Net {
    integer: u32,
    float: f32,
}

fn main() {}
