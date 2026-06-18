#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use rstorch::prelude::*;

fn main() {
    let logits: Tensor2D<0, 2> = Tensor2D::zeros();
    let targets: Tensor2D<0, 2> = Tensor2D::zeros();
    let _ = cross_entropy_one_hot(&logits, &targets);
}
