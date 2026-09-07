use rstorch::typed::{DeviceCtx, Tensor2};

fn main() {
    let ctx = DeviceCtx::cpu().unwrap();
    let mask = Tensor2::<2, 3, bool>::from_vec(vec![true; 6], [2, 3], &ctx).unwrap();
    let on_true = Tensor2::<2, 3>::from_vec(vec![1.0f32; 6], [2, 3], &ctx).unwrap();
    let row = Tensor2::<1, 3>::from_vec(vec![0.0f32; 3], [1, 3], &ctx).unwrap();
    let _ = mask.where_cond(&on_true, &row);
}
