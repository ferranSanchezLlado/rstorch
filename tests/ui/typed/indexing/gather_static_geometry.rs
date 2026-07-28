// rstorch-ui: build

use rstorch::typed::{DeviceCtx, Tensor2};

fn main() {
    let ctx = DeviceCtx::cpu().unwrap();
    let source = Tensor2::<2, 3>::from_vec(vec![0.0; 6], [2, 3], &ctx).unwrap();
    let indices = Tensor2::<1, 1, i64>::from_vec(vec![0], [1, 1], &ctx).unwrap();
    let _ = source.gather::<1, _>(&indices);
}
