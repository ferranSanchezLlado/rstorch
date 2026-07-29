// rstorch-ui: build

use rstorch::typed::{DeviceCtx, Tensor2};

fn main() {
    let ctx = DeviceCtx::cpu().unwrap();
    let invalid = Tensor2::<2, 3>::from_vec(vec![0.0f32; 6], [2, 3], &ctx).unwrap();
    let _ = invalid.squeeze::<0>();
}
