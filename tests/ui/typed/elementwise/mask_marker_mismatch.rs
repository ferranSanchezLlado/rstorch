use rstorch::typed::{DeviceCtx, Tensor2};

fn main() {
    let ctx = DeviceCtx::cpu().unwrap();
    let values = Tensor2::<2, 3>::from_vec(vec![1.0f32; 6], [2, 3], &ctx).unwrap();
    let row = Tensor2::<1, 3, bool>::from_vec(vec![true; 3], [1, 3], &ctx).unwrap();
    let _ = values.masked_fill(&row, 0.0);
}
