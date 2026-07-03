use rstorch::prelude::*;

fn main() {
    // The running output has 3 features, but the next layer expects 5, so the
    // chain is rejected where the mismatched layer is added.
    let _model = Sequential::<_, Tensor2D<2, 2>>::new(Linear::<2, 3>::zeros().unwrap(), Relu)
        .add_module(Linear::<5, 1>::zeros().unwrap());
}
