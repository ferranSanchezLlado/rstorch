use rstorch::{C, D2, Sym, Tensor};

struct Batch;

fn main() {
    let _ = Tensor::<D2<Sym<Batch>, C<784>>>::zeros();
}
