//! `Grads` is deliberately **not** `Clone`: duplicating a
//! gradient value is the first half of "the same gradients applied twice", so
//! it must not be expressible at all.

use rstorch::prelude::*;

fn assert_clone<T: Clone>(_: &T) {}

fn main() -> Result<()> {
    let x = Tensor::from_vec(vec![1.0f32, 2.0], [2], &Device::Cpu)?;
    let xt = x.traced()?;
    let grads = xt.mul(&xt)?.sum_all()?.backward()?;

    assert_clone(&grads);

    Ok(())
}
