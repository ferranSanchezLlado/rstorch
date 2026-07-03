use rstorch::Tensor1D;

fn main() {
    let tensor = Tensor1D::<2>::from_vec(vec![1.0, 2.0]).unwrap();
    tensor.backward().unwrap();
}
