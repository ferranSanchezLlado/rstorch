use rstorch::typed::{DeviceCtx, Tensor1};

fn main() {
    let ctx = DeviceCtx::cpu().unwrap();
    let floats = Tensor1::<2>::from_vec(vec![1.0f32; 2], [2], &ctx).unwrap();
    let integers = Tensor1::<2, i64>::from_vec(vec![2; 2], [2], &ctx).unwrap();
    let _ = floats.add(&integers);
}
