#![cfg(feature = "typed")]
//! The typed API's first-hour path, start to finish: bind a device context,
//! construct typed tensors, build and run a module, then save and reload it.
//!
//! This crate depends only on the `rstorch` public API, so passing here is
//! evidence a downstream user can actually walk the path in this order.
use rstorch::typed::nn::{
    Forward, Linear, Mode, TypedBuffer, TypedModule, TypedParam, load_state_dict, state_dict,
};
use rstorch::typed::prelude::*;
use rstorch::{Device, Rng, Tensor};

#[test]
fn guide_snippets() -> rstorch::Result<()> {
    // 1. context + construction
    let ctx = DeviceCtx::cpu()?;
    let x = Tensor2::<2, 3>::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], &ctx)?;
    assert_eq!(x.dims(), [2, 3]);

    // 2. wrapping an existing runtime tensor
    let runtime = Tensor::zeros([2, 3], rstorch::DType::F32, &Device::Cpu)?;
    let wrapped = Tensor2::<2, 3>::try_from_dynamic(runtime, &ctx)?;
    assert_eq!(wrapped.dims(), [2, 3]);

    // 3. DYN batch axis: one type, many batch sizes
    let b2 = Tensor2::<DYN, 3>::from_vec(vec![0.0f32; 6], [2, 3], &ctx)?;
    let b5 = Tensor2::<DYN, 3>::from_vec(vec![0.0f32; 15], [5, 3], &ctx)?;
    assert_eq!(b2.dims()[0], 2);
    assert_eq!(b5.dims()[0], 5);

    // 4. crossing the boundary
    let borrowed: &Tensor = x.as_dynamic();
    assert_eq!(borrowed.dims(), [2, 3]);
    let erased: Tensor2<DYN, DYN> = x.clone().erase_shape()?;
    let refined: Tensor2<2, 3> = erased.refine()?;
    assert_eq!(refined.dims(), [2, 3]);
    let owned: Tensor = refined.into_dynamic();
    assert_eq!(owned.dims(), [2, 3]);

    // 5. a layer + forward
    let mut rng = Rng::seed(0);
    let mut layer = Linear::<3, 4>::new(3, 4, &ctx, &mut rng)?;
    let out: Tensor2<2, 4> = layer.forward(&x, Mode::EVAL)?;
    assert_eq!(out.dims(), [2, 4]);
    Ok(())
}

// 6. the derive, including the config-field difference
type Bias = Tensor1<4, f32, Cpu>;

#[derive(TypedModule)]
struct Block {
    proj: Linear<3, 4>,
    scale: TypedParam<Bias>,
    seen: TypedBuffer<Bias>,
    // No primitive whitelist in the typed derive: without the skip this is
    // treated as a child module and fails with `usize: Module is not satisfied`.
    #[typed_module(skip)]
    hidden: usize,
    // A plain Vec config field cannot be skipped directly, so newtype it.
    #[typed_module(skip)]
    dims: Dims,
}

struct Dims(Vec<usize>);

#[test]
fn guide_derive_snippet() -> rstorch::Result<()> {
    let ctx = DeviceCtx::cpu()?;
    let mut rng = Rng::seed(1);
    let mut block = Block {
        proj: Linear::new(3, 4, &ctx, &mut rng)?,
        scale: TypedParam::new(Tensor1::from_vec(vec![1.0f32; 4], [4], &ctx)?)?,
        seen: TypedBuffer::new(Tensor1::from_vec(vec![0.0f32; 4], [4], &ctx)?)?,
        hidden: 32,
        dims: Dims(vec![3, 4]),
    };
    let snapshot = state_dict(&block)?;
    let paths: Vec<&str> = snapshot.paths().collect();
    assert_eq!(paths, ["proj.bias", "proj.weight", "scale", "seen"]);
    // round-trip through load
    load_state_dict(&mut block, &snapshot)?;
    // The skipped configuration fields are not state and survive untouched.
    assert_eq!(block.hidden, 32);
    assert_eq!(block.dims.0, vec![3, 4]);
    Ok(())
}
