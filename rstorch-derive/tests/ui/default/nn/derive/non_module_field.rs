//! Loud rule: a field whose type is not on the whitelist
//! (`Param`/`Option<Param>`/`Tensor`/`Option<Tensor>`/primitives) and is not
//! marked `#[module(skip)]` defaults to child-module recursion. If that type
//! does not implement `Module`, this is a compile error — the exact bug class
//! v3 abolishes (a silently unvisited, therefore untrained, field).

use rstorch::prelude::*;

/// A plain configuration type that is *not* a `Module`.
struct Config {
    _lr: f64,
}

#[derive(Module)]
struct Net {
    weight: Param,
    // No `#[module(skip)]`: `Config` is treated as a child module, so the
    // generated `visitor.module("cfg", &self.cfg)` requires `Config: Module`.
    cfg: Config,
}

fn main() {}
