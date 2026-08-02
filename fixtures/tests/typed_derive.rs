#![cfg(feature = "typed")]

//! Downstream reachability of `#[derive(TypedModule)]`.
//!
//! This crate depends only on `rstorch`, so it can reach the derive exactly
//! when `rstorch` re-exports it. Without `pub use rstorch_derive::TypedModule`
//! in `rstorch::typed::nn`, the import below is `error[E0432]: unresolved
//! import` and a downstream user is forced either to hand-write `impl Module`
//! or to add a second direct dependency on `rstorch-derive` — which
//! `#[derive(Module)]` never requires.

use rstorch::Rng;
use rstorch::typed::nn::{Linear, Mode, Module, TypedBuffer, TypedModule, TypedParam, state_dict};
use rstorch::typed::prelude::*;

type Bias = Tensor1<2, f32, Cpu>;
type Count = Tensor1<1, f32, Cpu>;

#[derive(TypedModule)]
struct Block {
    proj: Linear<3, 2>,
    scale: TypedParam<Bias>,
    seen: TypedBuffer<Count>,
}

impl Block {
    fn new(ctx: &DeviceCtx<Cpu>, rng: &mut Rng) -> rstorch::Result<Self> {
        Ok(Self {
            proj: Linear::new(3, 2, ctx, rng)?,
            scale: TypedParam::new(Tensor1::from_vec(vec![1.0, 1.0], [2], ctx)?)?,
            seen: TypedBuffer::new(Tensor1::from_vec(vec![0.0], [1], ctx)?)?,
        })
    }
}

#[derive(TypedModule)]
struct Net {
    blocks: Vec<Block>,
    head: Linear<2, 1>,
    tail: Option<Linear<1, 1>>,
    #[typed_module(skip)]
    label: String,
}

#[test]
fn derived_typed_module_is_reachable_through_rstorch_only() -> rstorch::Result<()> {
    let ctx = DeviceCtx::<Cpu>::cpu()?;
    let mut rng = Rng::seed(7);
    let model = Net {
        blocks: vec![Block::new(&ctx, &mut rng)?, Block::new(&ctx, &mut rng)?],
        head: Linear::new(2, 1, &ctx, &mut rng)?,
        tail: Some(Linear::new(1, 1, &ctx, &mut rng)?),
        label: "reachability".to_string(),
    };

    let state = state_dict(&model)?;
    assert_eq!(model.label, "reachability");
    assert_eq!(
        state.paths().collect::<Vec<_>>(),
        vec![
            "blocks.0.proj.bias",
            "blocks.0.proj.weight",
            "blocks.0.scale",
            "blocks.0.seen",
            "blocks.1.proj.bias",
            "blocks.1.proj.weight",
            "blocks.1.scale",
            "blocks.1.seen",
            "head.bias",
            "head.weight",
            "tail.bias",
            "tail.weight",
        ]
    );
    Ok(())
}

#[test]
fn derived_walk_matches_a_forward_capable_model() -> rstorch::Result<()> {
    use rstorch::typed::nn::Forward;

    let ctx = DeviceCtx::<Cpu>::cpu()?;
    let mut rng = Rng::seed(11);
    let mut model = Net {
        blocks: Vec::new(),
        head: Linear::new(2, 1, &ctx, &mut rng)?,
        tail: None,
        label: "no-children".to_string(),
    };

    // A skipped opaque field and an absent `Option` child contribute no paths.
    assert_eq!(model.label, "no-children");
    let state = state_dict(&model)?;
    assert_eq!(
        state.paths().collect::<Vec<_>>(),
        vec!["head.bias", "head.weight"]
    );
    assert_eq!(state.len(), 2);

    let input = Tensor2::<1, 2>::from_vec(vec![1.0, 2.0], [1, 2], &ctx)?;
    let output = model.head.forward(&input, Mode::EVAL)?;
    assert_eq!(output.dims(), [1, 1]);

    // The derived trait is the typed one, usable behind a `dyn` reference.
    let erased: &dyn Module = &model;
    assert_eq!(state_dict(erased)?.len(), 2);
    Ok(())
}
