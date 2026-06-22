use rstorch::Tensor2D;

fn main() {
    let lhs = Tensor2D::<2, 3>::zeros().unwrap();
    let rhs = Tensor2D::<4, 2>::zeros().unwrap();

    let _ = lhs.matmul(&rhs);
}
