// rstorch-ui: run

use rstorch::typed::{DYN, DeviceCtx, Tensor1, Tensor2, Tensor3, Tensor4};
use rstorch::{Error, Result};

fn main() -> Result<()> {
    let ctx = DeviceCtx::cpu()?;

    let lhs = Tensor2::<2, DYN>::from_vec(vec![0.0f32; 6], [2, 3], &ctx)?;
    let rhs = Tensor2::<DYN, 2>::from_vec(vec![0.0f32; 8], [4, 2], &ctx)?;
    assert!(matches!(lhs.matmul(&rhs), Err(Error::ShapeMismatch { op: "matmul", .. })));

    let batch_lhs =
        Tensor3::<DYN, 2, 3>::from_vec(vec![0.0f32; 12], [2, 2, 3], &ctx)?;
    let batch_rhs = Tensor3::<DYN, 3, 2>::from_vec(vec![0.0f32; 6], [1, 3, 2], &ctx)?;
    assert!(matches!(
        batch_lhs.matmul(&batch_rhs),
        Err(Error::ShapeMismatch { op: "matmul", .. })
    ));

    let shape = Tensor2::<DYN, 3>::from_vec(vec![0.0f32; 6], [2, 3], &ctx)?;
    assert!(matches!(
        shape.reshape::<Tensor1<DYN>>([5]),
        Err(Error::ReshapeMismatch { op: "reshape", .. })
    ));
    assert!(matches!(
        shape.squeeze::<0>(),
        Err(Error::InvalidArg { op: "squeeze", .. })
    ));
    assert!(matches!(
        shape.broadcast_to::<Tensor2<DYN, 3>>([4, 3]),
        Err(Error::ShapeMismatch { op: "broadcast_to", .. })
    ));

    let gather = Tensor2::<DYN, 1, i64>::from_vec(vec![0, 0, 0], [3, 1], &ctx)?;
    assert!(matches!(
        shape.gather::<1, _>(&gather),
        Err(Error::ShapeMismatch { op: "gather", .. })
    ));
    let out_of_bounds = Tensor1::<1, i64>::from_indices(vec![2], &ctx)?;
    assert!(matches!(
        shape.index_select::<0, 1>(&out_of_bounds),
        Err(Error::IndexOutOfBounds { op: "index_select", .. })
    ));

    let conv_input =
        Tensor4::<1, DYN, 3, 3>::from_vec(vec![0.0f32; 18], [1, 2, 3, 3], &ctx)?;
    let conv_weight =
        Tensor4::<1, DYN, 1, 1>::from_vec(vec![0.0f32; 3], [1, 3, 1, 1], &ctx)?;
    assert!(matches!(
        conv_input.conv2d(&conv_weight, (1, 1), (0, 0), (1, 1)),
        Err(Error::ShapeMismatch { op: "conv2d", .. })
    ));

    let logits = Tensor2::<DYN, 3>::from_vec(vec![0.0f32; 6], [2, 3], &ctx)?;
    let labels = Tensor1::<DYN, i64>::from_indices(vec![0], &ctx)?;
    assert!(matches!(
        logits.cross_entropy(&labels),
        Err(Error::ShapeMismatch { op: "cross_entropy", .. })
    ));
    Ok(())
}
