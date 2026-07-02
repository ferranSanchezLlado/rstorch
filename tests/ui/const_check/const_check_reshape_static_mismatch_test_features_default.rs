use rstorch::Tensor2D;

fn main() {
    let tensor = Tensor2D::<2, 6>::zeros().unwrap();

    let _ = tensor.reshape2::<5, 2>();
}
