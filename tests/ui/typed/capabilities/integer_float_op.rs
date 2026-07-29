use rstorch::typed::{DeviceCtx, Tensor1};

fn main() {
    let ctx = DeviceCtx::cpu().unwrap();
    let value = Tensor1::<2, i64>::from_vec(vec![1, 2], [2], &ctx).unwrap();
    let _ = value.tanh();
}
