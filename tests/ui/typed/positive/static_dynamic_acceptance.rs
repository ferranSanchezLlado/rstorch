// rstorch-ui: build
// rstorch-ui: pass

use rstorch::Result;
use rstorch::typed::{DYN, DeviceCtx, Tensor1, Tensor2, Tensor4};

fn main() -> Result<()> {
    let ctx = DeviceCtx::cpu()?;

    let lhs = Tensor2::<2, 3>::from_vec(vec![0.0f32; 6], [2, 3], &ctx)?;
    let rhs = Tensor2::<DYN, 2>::from_vec(vec![0.0f32; 6], [3, 2], &ctx)?;
    let _ = lhs.matmul(&rhs)?;

    let _ = lhs.reshape::<Tensor1<DYN>>([6])?;

    let input = Tensor4::<1, 2, 3, 3>::from_vec(vec![0.0f32; 18], [1, 2, 3, 3], &ctx)?;
    let weight = Tensor4::<1, DYN, 1, 1>::from_vec(vec![0.0f32; 2], [1, 2, 1, 1], &ctx)?;
    let _ = input.conv2d(&weight, (1, 1), (0, 0), (1, 1))?;

    let labels = Tensor1::<DYN, i64>::from_indices(vec![0, 1], &ctx)?;
    let _ = lhs.cross_entropy(&labels)?;
    Ok(())
}
