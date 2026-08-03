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
use rstorch::typed::nn::{
    Linear, Mode, Module, TypedBuffer, TypedModule, TypedParam, load_state_dict, state_dict,
};
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

/// Every leaf of a derived model, flattened for comparison.
fn leaves(net: &Net) -> rstorch::Result<Vec<Vec<f32>>> {
    let mut out = Vec::new();
    for block in &net.blocks {
        out.push(block.proj.weight().value()?.to_vec()?);
        out.push(block.scale.value()?.to_vec()?);
        out.push(block.seen.value()?.to_vec()?);
    }
    out.push(net.head.weight().value()?.to_vec()?);
    if let Some(tail) = &net.tail {
        out.push(tail.weight().value()?.to_vec()?);
    }
    Ok(out)
}

/// The derive emits *two* walks, `visit` and `visit_mut`, and `state_dict`
/// exercises only the first. `load_state_dict` runs both and rejects a pair
/// that disagrees, so this is what pins the derived mutable walk: a
/// `visit_mut` that skipped a leaf, visited leaves in a different order, or
/// bound the wrong field would either fail the agreement check or leave a
/// destination leaf unwritten below.
#[test]
fn derived_mutable_walk_writes_every_leaf() -> rstorch::Result<()> {
    let ctx = DeviceCtx::<Cpu>::cpu()?;

    let mut source_rng = Rng::seed(7);
    let mut source = Net {
        blocks: vec![
            Block::new(&ctx, &mut source_rng)?,
            Block::new(&ctx, &mut source_rng)?,
        ],
        head: Linear::new(2, 1, &ctx, &mut source_rng)?,
        tail: Some(Linear::new(1, 1, &ctx, &mut source_rng)?),
        label: "source".to_string(),
    };
    // Give the buffers and the bare param values distinct from their
    // constructor defaults, so a `visit_mut` that silently skips a
    // `TypedBuffer` or `TypedParam` cannot pass by coincidence.
    for (i, block) in source.blocks.iter_mut().enumerate() {
        let n = i as f32;
        block
            .scale
            .set(Tensor1::from_vec(vec![2.0 + n, 3.0 + n], [2], &ctx)?)?;
        block.seen = TypedBuffer::new(Tensor1::from_vec(vec![5.0 + n], [1], &ctx)?)?;
    }

    let mut dest_rng = Rng::seed(99);
    let mut dest = Net {
        blocks: vec![
            Block::new(&ctx, &mut dest_rng)?,
            Block::new(&ctx, &mut dest_rng)?,
        ],
        head: Linear::new(2, 1, &ctx, &mut dest_rng)?,
        tail: Some(Linear::new(1, 1, &ctx, &mut dest_rng)?),
        label: "destination".to_string(),
    };

    // Independent seeds, so every comparable leaf starts different. Without
    // this the load below could not be observed at all.
    let before = leaves(&dest)?;
    let expected = leaves(&source)?;
    assert_eq!(before.len(), expected.len());
    for (i, (b, e)) in before.iter().zip(&expected).enumerate() {
        assert_ne!(b, e, "leaf {i} started equal, so the load is unobservable");
    }

    load_state_dict(&mut dest, &state_dict(&source)?)?;

    assert_eq!(
        leaves(&dest)?,
        expected,
        "the derived mutable walk did not write every leaf"
    );
    // A skipped field is not state and must survive the load untouched.
    assert_eq!(dest.label, "destination");
    Ok(())
}
