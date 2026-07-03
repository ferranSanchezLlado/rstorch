// rstorch-ui: build

use rstorch::{C, Tensor1D};

fn main() {
    let lhs = Tensor1D::<2>::zeros().unwrap();
    let rhs = Tensor1D::<3>::zeros().unwrap();

    let _ = lhs.cat1::<C<3>, 4>(&rhs);
}
