use rstorch::prelude::*;

fn main() {
    let lhs = Tensor2D::<2, 3>::zeros().unwrap();
    let rhs = Tensor2D::<2, 4>::zeros().unwrap();
    let _ = &lhs + &rhs;
}
