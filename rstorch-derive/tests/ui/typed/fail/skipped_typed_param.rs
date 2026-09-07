use rstorch::typed::nn::TypedParam;
use rstorch::typed::{Cpu, Tensor1};
use rstorch_derive::TypedModule;

#[derive(TypedModule)]
struct Net {
    #[typed_module(skip)]
    weight: TypedParam<Tensor1<2, f32, Cpu>>,
}

fn main() {}
