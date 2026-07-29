use rstorch::typed::{DeviceCtx, Tensor3};

fn main() {
    let ctx = DeviceCtx::cpu().unwrap();
    let lhs = Tensor3::<2, 2, 3>::from_vec(vec![0.0f32; 12], [2, 2, 3], &ctx).unwrap();
    let invalid = Tensor3::<1, 3, 2>::from_vec(vec![0.0f32; 6], [1, 3, 2], &ctx).unwrap();
    let _ = lhs.matmul(&invalid);
}
