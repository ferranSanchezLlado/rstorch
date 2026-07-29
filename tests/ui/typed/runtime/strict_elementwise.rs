// rstorch-ui: run

use rstorch::typed::{DYN, DeviceCtx, Tensor2};
use rstorch::{Error, Result};

fn main() -> Result<()> {
    let ctx = DeviceCtx::cpu()?;
    let matrix = Tensor2::<DYN, DYN>::from_vec(vec![1.0f32; 6], [2, 3], &ctx)?;
    let row = Tensor2::<DYN, DYN>::from_vec(vec![2.0f32; 3], [1, 3], &ctx)?;
    assert!(matches!(
        matrix.add(&row),
        Err(Error::ShapeMismatch { op: "add", .. })
    ));

    let row_mask = Tensor2::<DYN, DYN, bool>::from_vec(vec![true; 3], [1, 3], &ctx)?;
    assert!(matches!(
        matrix.masked_fill(&row_mask, 0.0),
        Err(Error::ShapeMismatch { op: "masked_fill", .. })
    ));

    let mask = Tensor2::<DYN, DYN, bool>::from_vec(vec![true; 6], [2, 3], &ctx)?;
    assert!(matches!(
        mask.where_cond(&matrix, &row),
        Err(Error::ShapeMismatch { op: "where", .. })
    ));
    Ok(())
}
