use rstorch_derive::TypedModule;

struct Config;

#[derive(TypedModule)]
struct Net {
    config: Config,
}

fn main() {}
