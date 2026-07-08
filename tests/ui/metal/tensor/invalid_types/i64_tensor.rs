use rstorch::{C, D1, Metal, Tensor};

fn main() {
    let _ = Tensor::<D1<C<1>>, i64, Metal>::zeros();
}
