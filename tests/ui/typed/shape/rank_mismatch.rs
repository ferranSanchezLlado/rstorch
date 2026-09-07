use rstorch::typed::{Tensor1, Tensor2};

fn rejected(matrix: Tensor2<1, 3>) {
    let _: Tensor1<3> = matrix;
}

fn main() {}
