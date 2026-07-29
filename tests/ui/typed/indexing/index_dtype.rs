use rstorch::typed::{DeviceCtx, Tensor1, Tensor2};

fn main() {
    let ctx = DeviceCtx::cpu().unwrap();
    let source = Tensor2::<2, 3>::from_vec(vec![0.0f32; 6], [2, 3], &ctx).unwrap();
    let floats = Tensor1::<1>::from_vec(vec![0.0f32], [1], &ctx).unwrap();
    let _ = source.index_select::<0, 1>(&floats);
}
