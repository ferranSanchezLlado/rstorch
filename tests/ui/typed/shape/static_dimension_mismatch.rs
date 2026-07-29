use rstorch::typed::Tensor2;

fn rejected(mismatched: Tensor2<2, 4>) {
    let _: Tensor2<2, 3> = mismatched;
}

fn main() {}
