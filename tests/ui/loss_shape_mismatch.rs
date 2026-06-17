#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use rstorch::prelude::*;

fn main() {
    let prediction: Tensor2D<4, 1> = Tensor2D::zeros();
    let target: Tensor2D<4, 2> = Tensor2D::zeros();
    let _ = mse_loss(&prediction, &target);
}
