// rstorch-ui: build

use rstorch::Tensor3D;

fn main() {
    let tensor = Tensor3D::<2, 3, 0>::from_vec(Vec::<f32>::new()).unwrap();

    let _ = tensor.softmax_axis2();
}
