// rstorch-ui: run

use rstorch::typed::{DYN, DeviceCtx, Tensor2};
use rstorch::{Error, Result};

fn main() -> Result<()> {
    let ctx = DeviceCtx::cpu()?;
    let dynamic = Tensor2::<DYN, 3>::from_vec(vec![0.0f32; 6], [2, 3], &ctx)?;
    assert!(matches!(
        dynamic.refine::<Tensor2<4, 3>>(),
        Err(Error::ShapeMismatch { op: "refine", .. })
    ));
    Ok(())
}
