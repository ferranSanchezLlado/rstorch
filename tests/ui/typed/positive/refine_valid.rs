// rstorch-ui: build
// rstorch-ui: pass

use rstorch::Result;
use rstorch::typed::{DYN, DeviceCtx, Tensor2};

fn main() -> Result<()> {
    let ctx = DeviceCtx::cpu()?;
    let dynamic = Tensor2::<DYN, 3>::from_vec(vec![0.0f32; 6], [2, 3], &ctx)?;
    let _: Tensor2<2, 3> = dynamic.refine()?;
    let same = Tensor2::<2, 3>::from_vec(vec![0.0f32; 6], [2, 3], &ctx)?;
    let _: Tensor2<2, 3> = same.refine()?;
    Ok(())
}
