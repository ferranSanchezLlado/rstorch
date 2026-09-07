use rstorch::typed::{DeviceCtx, Tensor2};

fn main() {
    let ctx = DeviceCtx::cpu().unwrap();
    let matrix = Tensor2::<2, 3>::from_vec(vec![1.0f32; 6], [2, 3], &ctx).unwrap();
    let row = Tensor2::<1, 3>::from_vec(vec![2.0f32; 3], [1, 3], &ctx).unwrap();
    let _ = matrix.add(&row);
}
