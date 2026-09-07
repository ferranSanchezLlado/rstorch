// rstorch-ui: build

use rstorch::typed::{DeviceCtx, Tensor1, Tensor2};

fn main() {
    let ctx = DeviceCtx::cpu().unwrap();
    let source = Tensor2::<2, 3>::from_vec(vec![0.0f32; 6], [2, 3], &ctx).unwrap();
    let _ = source.reshape::<Tensor1<5>>([5]);
}
