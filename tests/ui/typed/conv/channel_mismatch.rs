// rstorch-ui: build

use rstorch::typed::{DeviceCtx, Tensor4};

fn main() -> rstorch::Result<()> {
    let ctx = DeviceCtx::cpu()?;
    let input = Tensor4::<1, 2, 3, 3>::from_vec(vec![0.0f32; 18], [1, 2, 3, 3], &ctx)?;
    let weight = Tensor4::<1, 3, 1, 1>::from_vec(vec![0.0f32; 3], [1, 3, 1, 1], &ctx)?;
    let _ = input.conv2d(&weight, (1, 1), (0, 0), (1, 1))?;
    Ok(())
}
