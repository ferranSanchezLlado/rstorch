use rstorch::typed::nn::TypedParam;
use rstorch::typed::{Cpu, Tensor1};
use rstorch_derive::TypedModule;

type Weight = TypedParam<Tensor1<2, f32, Cpu>>;

#[derive(TypedModule)]
struct Net {
    weight: Weight,
}

fn main() {}
