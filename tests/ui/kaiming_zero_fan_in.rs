#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use rstorch::prelude::*;

fn main() {
    let mut rng = SmallRng::seed_from_u64(1);
    let _ = Linear::<0, 4>::kaiming_uniform(&mut rng);
}
