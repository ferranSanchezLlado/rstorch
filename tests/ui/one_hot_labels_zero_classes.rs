#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use rstorch::prelude::*;

fn main() {
    let _ = one_hot_labels::<1, 0, f32, Cpu>([0]);
}
