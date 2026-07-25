//! `Grads` is `#[must_use]`: computing gradients and then never stepping is a
//! silent no-op training loop in PyTorch and a diagnostic here (exploration
//! §5). The lint is a warning by default; this case denies it so the guarantee
//! is pinned rather than merely observed.

#![deny(unused_must_use)]

use rstorch::prelude::*;

fn main() -> Result<()> {
    let x = Tensor::from_vec(vec![1.0f32, 2.0], [2], &Device::Cpu)?;
    let xt = x.traced()?;

    xt.mul(&xt)?.sum_all()?.backward()?;

    Ok(())
}
