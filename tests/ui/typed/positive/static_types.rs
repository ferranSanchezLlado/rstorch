// rstorch-ui: pass

use rstorch::typed::{Tensor1, Tensor2};

fn rank_and_dimension(value: Tensor2<2, 3>) -> Tensor2<2, 3> {
    value
}

fn dtype(value: Tensor1<2, i64>) -> Tensor1<2, i64> {
    value
}

fn main() {}
