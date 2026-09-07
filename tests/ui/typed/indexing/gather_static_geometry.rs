// rstorch-ui: build
//
// An index grid LARGER than the source on a non-selected dimension is what the
// runtime rejects, so it is what the const check must reject too. A *smaller*
// grid is a legal partial gather that the runtime accepts, and pinning that as
// an error here would make static markers change acceptance rather than merely
// checking it; it is covered by a positive test in `src/typed/ops/index.rs`.

use rstorch::typed::{DeviceCtx, Tensor2};

fn main() {
    let ctx = DeviceCtx::cpu().unwrap();
    let source = Tensor2::<2, 3>::from_vec(vec![0.0; 6], [2, 3], &ctx).unwrap();
    let indices = Tensor2::<3, 1, i64>::from_vec(vec![0; 3], [3, 1], &ctx).unwrap();
    let _ = source.gather::<1, _>(&indices);
}
