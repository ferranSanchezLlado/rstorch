// rstorch-ui: build

use rstorch::typed::{DeviceCtx, Tensor2};

fn main() {
    let ctx = DeviceCtx::cpu().unwrap();
    let row = Tensor2::<1, 3>::from_vec(vec![0.0f32; 3], [1, 3], &ctx).unwrap();
    let _ = row.broadcast_to::<Tensor2<2, 4>>([2, 4]);
}
