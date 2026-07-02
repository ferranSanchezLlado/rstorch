use rstorch::Tensor2D;

fn main() {
    let tensor = Tensor2D::<2, 0>::from_vec(Vec::<f32>::new()).unwrap();

    let _ = tensor.softmax_axis1();
}
