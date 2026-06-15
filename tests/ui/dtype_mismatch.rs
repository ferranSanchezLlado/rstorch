#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use rstorch::prelude::*;

fn main() {
    let a: Tensor2D<32, 10, f32, Cpu> = Tensor2D::zeros();
    let b: Tensor2D<32, 10, f64, Cpu> = Tensor2D::zeros();
    let _ = a.add(&b);
}
