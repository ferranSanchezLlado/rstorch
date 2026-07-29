// rstorch-ui: build
// rstorch-ui: pass

use rstorch::{Device, Result};
use rstorch::typed::{DeviceCtx, Placement, Tensor1, Tensor2, Tensor3, Tensor4};

struct Main;
impl Placement for Main {}

fn main() -> Result<()> {
    let ctx = DeviceCtx::<Main>::bind(Device::Cpu)?;

    let matrix = Tensor2::<2, 3, f32, Main>::from_vec(vec![1.0; 6], [2, 3], &ctx)?;
    let same = Tensor2::<2, 3, f32, Main>::from_vec(vec![2.0; 6], [2, 3], &ctx)?;
    let mask = Tensor2::<2, 3, bool, Main>::from_vec(vec![true; 6], [2, 3], &ctx)?;
    let _ = matrix.add(&same)?;
    let _ = matrix.masked_fill(&mask, 0.0)?;
    let _ = mask.where_cond(&matrix, &same)?;

    let indices = Tensor1::<1, i64, Main>::from_indices(vec![0], &ctx)?;
    let _ = matrix.index_select::<1, 1>(&indices)?;
    let gather = Tensor2::<2, 1, i64, Main>::from_vec(vec![0, 0], [2, 1], &ctx)?;
    let _ = matrix.gather::<1, _>(&gather)?;

    let rhs = Tensor2::<3, 2, f32, Main>::from_vec(vec![0.0; 6], [3, 2], &ctx)?;
    let _ = matrix.matmul(&rhs)?;
    let batch_lhs = Tensor3::<2, 2, 3, f32, Main>::from_vec(vec![0.0; 12], [2, 2, 3], &ctx)?;
    let batch_rhs = Tensor3::<2, 3, 2, f32, Main>::from_vec(vec![0.0; 12], [2, 3, 2], &ctx)?;
    let _ = batch_lhs.matmul(&batch_rhs)?;

    let singleton = Tensor2::<1, 3, f32, Main>::from_vec(vec![0.0; 3], [1, 3], &ctx)?;
    let _ = singleton.squeeze::<0>()?;
    let _ = matrix.reshape::<Tensor1<6, f32, Main>>([6])?;
    let _ = singleton.broadcast_to::<Tensor2<2, 3, f32, Main>>([2, 3])?;

    let input = Tensor4::<1, 2, 3, 3, f32, Main>::from_vec(vec![0.0; 18], [1, 2, 3, 3], &ctx)?;
    let weight = Tensor4::<1, 2, 1, 1, f32, Main>::from_vec(vec![0.0; 2], [1, 2, 1, 1], &ctx)?;
    let _ = input.conv2d(&weight, (1, 1), (0, 0), (1, 1))?;

    let labels = Tensor1::<2, i64, Main>::from_indices(vec![0, 1], &ctx)?;
    let _ = matrix.cross_entropy(&labels)?;
    Ok(())
}
