#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use rstorch::prelude::*;

fn main() {
    let tensor: Tensor3D<2, 2, 3> = Tensor3D::zeros();
    let _: Tensor2D<2, 5> = tensor.reshape_2d();
}
