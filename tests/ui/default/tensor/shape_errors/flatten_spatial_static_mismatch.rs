// rstorch-ui: build

use rstorch::Tensor4D;

fn main() {
    let tensor = Tensor4D::<1, 2, 3, 4>::zeros().unwrap();

    let _ = tensor.flatten_spatial::<23>();
}
