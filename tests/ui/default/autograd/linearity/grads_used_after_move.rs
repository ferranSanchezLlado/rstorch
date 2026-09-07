//! Every `Grads` consumer takes it **by move**, so
//! applying one gradient value twice — the optimizer-step bug PyTorch cannot
//! see — is a borrow-checker error instead.

use rstorch::prelude::*;

fn main() -> Result<()> {
    let x = Tensor::from_vec(vec![1.0f32, 2.0], [2], &Device::Cpu)?;
    let xt = x.traced()?;
    let first = xt.mul(&xt)?.sum_all()?.backward()?;
    let second = xt.mul(&xt)?.sum_all()?.backward()?;

    // `merge` consumes both operands…
    let merged = first.merge(second)?;

    // …so neither may be reused afterwards.
    println!("{} {}", first.len(), merged.len());

    Ok(())
}
