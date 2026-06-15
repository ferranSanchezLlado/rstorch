#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use rstorch::prelude::*;

fn main() {
    let a: Tensor2D<32, 784> = Tensor2D::zeros();
    let b: Tensor2D<128, 10> = Tensor2D::zeros();
    let _ = a.matmul(&b);
}
