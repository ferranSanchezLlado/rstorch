use rstorch::typed::{DeviceCtx, Tensor1};

fn main() {
    let ctx = DeviceCtx::cpu().unwrap();
    let value = Tensor1::<2, bool>::from_vec(vec![true, false], [2], &ctx).unwrap();
    let _ = value.sum::<0>();
}
