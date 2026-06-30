use rstorch::prelude::*;

fn main() {
    // `42` is not a module, so it cannot be appended to the chain.
    let _model = Sequential::<_, Tensor2D<2, 2>>::new(Linear::<2, 3>::zeros().unwrap(), Relu)
        .add_module(42);
}
