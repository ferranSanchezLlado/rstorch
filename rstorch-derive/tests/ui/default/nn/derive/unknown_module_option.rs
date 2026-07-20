//! `#[module(...)]` accepts only `skip`. Any other option is a hard error so a
//! typo (e.g. `#[module(skpi)]` or `#[module(rename = "x")]`) cannot silently
//! do nothing and leave a parameter unexpectedly visited/unvisited.

use rstorch::prelude::*;

#[derive(Module)]
struct Net {
    #[module(rename = "w")]
    weight: Param,
}

fn main() {}
