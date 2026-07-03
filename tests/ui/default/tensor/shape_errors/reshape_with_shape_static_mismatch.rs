// rstorch-ui: build

use rstorch::{C, D2, Tensor2D};

fn main() {
    let tensor = Tensor2D::<2, 6>::zeros().unwrap();

    let _ = tensor.reshape_with_shape::<D2<C<4>, C<2>>>([4, 2]);
}
