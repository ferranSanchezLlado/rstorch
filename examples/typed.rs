//! The `typed` feature: shapes, dtypes and device placement in the type
//! system, checked when the code compiles rather than when it runs.
//!
//! Run it with `cargo run --example typed --features typed`.
//!
//! The typed layer is a wrapper. It owns no storage, no kernels and no
//! backward formulas — it delegates to the same [`rstorch::Tensor`], so a
//! typed program and its dynamic twin compute identical values.

use rstorch::typed::prelude::*;
use rstorch::{Device, Result, Tensor};

fn main() -> Result<()> {
    // A `Placement` marker is bound to a real device once; from then on the
    // type says which device a tensor is on, and mixing two of them is a type
    // error rather than a runtime `DeviceMismatch`.
    let ctx = DeviceCtx::cpu()?;

    // Dimensions are const generics. This is a 2x3 f32 tensor on `Cpu`, and
    // the compiler knows all three facts.
    let x = Tensor2::<2, 3>::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], &ctx)?;

    // Contraction is checked at compile time: [2, 3] @ [3, 4] -> [2, 4], and
    // the output type is computed, not asserted by the caller.
    let w = Tensor2::<3, 4>::from_vec(vec![0.1f32; 12], [3, 4], &ctx)?;
    let y: Tensor2<2, 4> = x.matmul(&w)?;
    println!("x @ w has dims {:?}", y.dims());

    // Writing `x.matmul(&Tensor2::<4, 4>::…)` here would not compile: the
    // mismatch is a type error, so it never reaches a test run. The compile
    // -fail cases in `tests/typed_ui.rs` pin those diagnostics.

    // `DYN` opts one axis out of static checking — the usual need being a
    // batch size that is not known until the loader hands one over. Everything
    // it touches is validated at runtime instead, by the same code the dynamic
    // API uses.
    let batch = Tensor2::<DYN, 3>::from_vec(vec![0.5f32; 15], [5, 3], &ctx)?;
    let projected: Tensor2<DYN, 4> = batch.matmul(&w)?;
    println!("a DYN batch projects to {:?}", projected.dims());

    // The boundary is zero-copy in both directions and does not fork the
    // autograd graph: the typed tensor and the dynamic one it wraps are the
    // same tensor.
    let runtime = Tensor::from_vec(vec![2.0f32, 4.0], [2], &Device::Cpu)?.traced()?;
    let typed = Tensor1::<2>::try_from_dynamic(runtime.clone(), &ctx)?;
    let loss = typed.mul(&typed)?.sum_all()?;
    let grads = loss.backward()?;
    println!("loss = {}", loss.to_scalar()?);
    println!(
        "d(loss)/d(typed) = {:?}  and through the dynamic handle = {:?}",
        grads.wrt_typed_input(&typed)?.to_vec()?,
        grads.wrt_input(&runtime)?.to_vec::<f32>()?,
    );

    Ok(())
}
