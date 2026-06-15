#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use rstorch::prelude::*;

fn main() {
    let x: Tensor2D<2, 2> = Tensor2D::ones();
    x.backward();
}
