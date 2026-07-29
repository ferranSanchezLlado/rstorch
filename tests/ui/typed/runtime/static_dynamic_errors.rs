// rstorch-ui: run

use rstorch::typed::{DYN, DeviceCtx, Tensor1, Tensor2, Tensor4};
use rstorch::{Error, Result};

fn main() -> Result<()> {
    let ctx = DeviceCtx::cpu()?;

    let lhs = Tensor2::<2, 3>::from_vec(vec![0.0f32; 6], [2, 3], &ctx)?;
    let rhs = Tensor2::<DYN, 2>::from_vec(vec![0.0f32; 8], [4, 2], &ctx)?;
    assert!(matches!(
        lhs.matmul(&rhs),
        Err(Error::ShapeMismatch { op: "matmul", .. })
    ));

    assert!(matches!(
        lhs.reshape::<Tensor1<DYN>>([5]),
        Err(Error::ReshapeMismatch { op: "reshape", .. })
    ));

    let input = Tensor4::<1, 2, 3, 3>::from_vec(vec![0.0f32; 18], [1, 2, 3, 3], &ctx)?;
    let weight = Tensor4::<1, DYN, 1, 1>::from_vec(vec![0.0f32; 3], [1, 3, 1, 1], &ctx)?;
    assert!(matches!(
        input.conv2d(&weight, (1, 1), (0, 0), (1, 1)),
        Err(Error::ShapeMismatch { op: "conv2d", .. })
    ));

    let labels = Tensor1::<DYN, i64>::from_indices(vec![0], &ctx)?;
    assert!(matches!(
        lhs.cross_entropy(&labels),
        Err(Error::ShapeMismatch { op: "cross_entropy", .. })
    ));
    Ok(())
}
