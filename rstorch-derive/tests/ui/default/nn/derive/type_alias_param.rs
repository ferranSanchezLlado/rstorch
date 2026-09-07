//! Type aliases defeat the syntactic token match and fail loudly
//! Classification matches on the type *as written*, so a
//! `type Weights = Param;` field is NOT recognized as a `Param`; it falls to
//! the loud default (child module) and, because `Param` is not a `Module`,
//! produces a `Module`-not-satisfied compile error. Spell `Param` out.

use rstorch_alias::prelude::*;

/// An alias for `Param` — deliberately not recognized by the derive.
type Weights = Param;

#[derive(Module)]
struct Net {
    // Written as `Weights`, not `Param`: the token match sees `Weights`,
    // treats it as a child module, and `Param: Module` does not hold.
    weight: Weights,
}

fn main() {}
