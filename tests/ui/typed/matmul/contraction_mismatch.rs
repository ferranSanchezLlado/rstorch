// rstorch-ui: build

use rstorch::typed::{DeviceCtx, Tensor2};

fn main() {
    let ctx = DeviceCtx::cpu().unwrap();
    let lhs = Tensor2::<2, 3>::from_vec(vec![0.0f32; 6], [2, 3], &ctx).unwrap();
    let invalid = Tensor2::<4, 2>::from_vec(vec![0.0f32; 8], [4, 2], &ctx).unwrap();
    let _ = lhs.matmul(&invalid);
}
