//! Tensors, autograd, and the three rules that make rstorch different.
//!
//! Run it with `cargo run --example tensors`.

use rstorch::prelude::*;

fn main() -> Result<()> {
    let dev = Device::Cpu;

    // One concrete `Tensor`. Rank, dtype and device are values it carries, not
    // type parameters, so a 2x2 f32 and a 3-D f64 have the same Rust type.
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2], &dev)?;
    let b = Tensor::from_vec(vec![10.0f32, 20.0], [2], &dev)?;
    println!("a = {:?}, dims {:?}", a.to_vec::<f32>()?, a.dims());

    // Broadcasting follows the usual right-aligned rule: [2] against [2, 2].
    let sum = a.add(&b)?;
    println!("a + b = {:?}", sum.to_vec::<f32>()?);

    // Rule 1: no implicit dtype promotion. Mixing dtypes is an error that
    // tells you which cast to write, rather than a silent widening.
    let ints = Tensor::from_vec(vec![1i64, 2, 3, 4], [2, 2], &dev)?;
    match a.add(&ints) {
        Err(e) => println!("\nrefused: {e}"),
        Ok(_) => unreachable!("f32 + i64 is not a promotion this crate performs"),
    }
    let widened = a.add(&ints.to_dtype(DType::F32)?)?;
    println!("after an explicit cast: {:?}", widened.to_vec::<f32>()?);

    // Rule 2: every error names the operation that raised it and the values
    // that were wrong, so a failure is readable without a backtrace.
    match a.matmul(&Tensor::zeros([3, 3], DType::F32, &dev)?) {
        Err(e) => println!("refused: {e}"),
        Ok(_) => unreachable!("[2, 2] @ [3, 3] does not contract"),
    }

    // Rule 3: gradients are linear. `backward` yields one `Grads`, which is
    // not `Clone` and is `#[must_use]` — an optimizer step consumes it, so
    // reusing the moved value is a compile error, while ignoring it is a
    // warning. `tests/linearity_ui.rs` pins those diagnostics.
    // Tracing is explicit: `traced` returns the binding the tape records
    // against, and that binding — not the tensor it came from — is what
    // `wrt_input` answers for.
    let x = Tensor::from_vec(vec![3.0f32, 5.0], [2], &dev)?.traced()?;
    let y = x.mul(&x)?.sum_all()?; // y = sum(x^2), so dy/dx = 2x
    let grads = y.backward()?;
    println!("\ny = {}", y.item()?);
    println!("dy/dx = {:?}", grads.wrt_input(&x)?.to_vec::<f32>()?);

    // `wrt_input` only borrows, so `grads` is still live here; handing it to
    // an optimizer would move it, and there is no second step to be had.
    println!("gradients recorded: {}", grads.len());

    Ok(())
}
