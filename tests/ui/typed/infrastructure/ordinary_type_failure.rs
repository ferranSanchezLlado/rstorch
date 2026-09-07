use rstorch::typed::Tensor1;

fn wrong_static_dimension(value: Tensor1<4>) {
    let _: Tensor1<3> = value;
}

fn main() {}
