use rstorch::typed::{DeviceCtx, Tensor1};

fn main() {
    let ctx = DeviceCtx::cpu().unwrap();
    let values = Tensor1::<2>::from_vec(vec![1.0f32; 2], [2], &ctx).unwrap();
    let not_a_mask = Tensor1::<2, i64>::from_vec(vec![1; 2], [2], &ctx).unwrap();
    let _ = values.masked_fill(&not_a_mask, 0.0);
}
