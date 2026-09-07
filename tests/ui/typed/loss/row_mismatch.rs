// rstorch-ui: build

use rstorch::typed::{DeviceCtx, Tensor1, Tensor2};

fn main() {
    let ctx = DeviceCtx::cpu().unwrap();
    let logits = Tensor2::<2, 3>::from_vec(vec![0.0f32; 6], [2, 3], &ctx).unwrap();
    let invalid = Tensor1::<1, i64>::from_indices(vec![0], &ctx).unwrap();
    let _ = logits.cross_entropy(&invalid);
}
