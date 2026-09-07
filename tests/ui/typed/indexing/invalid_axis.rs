use rstorch::typed::{DeviceCtx, Tensor1, Tensor2};

fn main() {
    let ctx = DeviceCtx::cpu().unwrap();
    let source = Tensor2::<2, 3>::from_vec(vec![0.0; 6], [2, 3], &ctx).unwrap();
    let indices = Tensor1::<1, i64>::from_indices(vec![0], &ctx).unwrap();
    let _ = source.index_select::<2, 1>(&indices);
}
