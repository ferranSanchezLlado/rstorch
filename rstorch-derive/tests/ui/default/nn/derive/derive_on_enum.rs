//! `#[derive(Module)]` supports `struct`s only. A module's field set must be
//! statically known so every parameter is visited; an `enum` has no single
//! field set, so it is rejected with a clear message.

use rstorch::prelude::*;

#[derive(Module)]
enum Net {
    A(Param),
    B,
}

fn main() {}
