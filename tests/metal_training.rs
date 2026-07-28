#![cfg(all(feature = "metal", target_os = "macos"))]

use rstorch::models::{DecoderTransformer, TransformerConfig};
use rstorch::prelude::*;

const METAL: Device = Device::Metal(0);

#[derive(Module)]
struct Mlp {
    first: Linear,
    second: Linear,
}

impl Forward for Mlp {
    fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        self.second
            .forward(&self.first.forward(x, mode)?.gelu()?, mode)
    }
}

#[test]
fn seeded_mlp_and_transformer_losses_decrease_on_metal() -> Result<()> {
    let mut rng = Rng::seed(61);
    let mut mlp = Mlp {
        first: Linear::new(4, 8, &METAL, &mut rng)?,
        second: Linear::new(8, 3, &METAL, &mut rng)?,
    };
    let x = Tensor::from_vec(
        vec![
            1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 0.0, 0.0, -1.0,
        ],
        [3, 4],
        &METAL,
    )?;
    let labels = Tensor::from_vec(vec![0i64, 1, 2], [3], &METAL)?;
    let mut sgd = Sgd::new(0.05);
    let first = mlp
        .forward(&x, Mode::EVAL)?
        .cross_entropy(&labels)?
        .item()?;
    eprintln!("metal MLP initial loss: {first}");
    for _ in 0..30 {
        let loss = mlp.forward(&x, Mode::TRAIN)?.cross_entropy(&labels)?;
        sgd.step(&mut mlp, loss.backward()?)?;
    }
    let last = mlp
        .forward(&x, Mode::EVAL)?
        .cross_entropy(&labels)?
        .item()?;
    eprintln!("metal MLP final loss: {last}");
    assert!(last < first, "MLP loss did not decrease: {first} -> {last}");

    let config = TransformerConfig {
        vocab_size: 6,
        max_seq_len: 3,
        embed_dim: 4,
        num_heads: 2,
        num_layers: 1,
        feed_forward_dim: 8,
    };
    let mut transformer = DecoderTransformer::new(config, &METAL, &mut Rng::seed(7))?;
    let ids = Tensor::from_vec(vec![0i64, 1, 2, 1, 2, 3], [2, 3], &METAL)?;
    let targets = Tensor::from_vec(vec![1i64, 2, 3, 2, 3, 4], [6], &METAL)?;
    let loss = |model: &mut DecoderTransformer, mode| {
        model
            .logits(&ids, mode)?
            .reshape([6, 6])?
            .cross_entropy(&targets)
    };
    let mut adam = AdamW::new(0.02, 0.0);
    let first = loss(&mut transformer, Mode::EVAL)?.item()?;
    eprintln!("metal transformer initial loss: {first}");
    for _ in 0..20 {
        let step = loss(&mut transformer, Mode::TRAIN)?;
        adam.step(&mut transformer, step.backward()?)?;
    }
    let last = loss(&mut transformer, Mode::EVAL)?.item()?;
    eprintln!("metal transformer final loss: {last}");
    assert!(
        last < first,
        "transformer loss did not decrease: {first} -> {last}"
    );
    Ok(())
}

#[derive(Module)]
struct Cnn {
    weight: Param,
    head: Linear,
}

impl Forward for Cnn {
    fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        let features = x
            .conv2d(&self.weight.get(mode), (1, 1), (0, 0), (1, 1))?
            .relu()?
            .avg_pool2d((2, 2), (2, 2), (0, 0))?
            .reshape([x.dims()[0], 2])?;
        self.head.forward(&features, mode)
    }
}

#[test]
fn seeded_cnn_loss_decreases_on_metal() -> Result<()> {
    let mut rng = Rng::seed(9);
    let mut cnn = Cnn {
        weight: Param::new(Tensor::randn([2, 1, 2, 2], DType::F32, &METAL, &mut rng)?),
        head: Linear::new(2, 2, &METAL, &mut rng)?,
    };
    let x = Tensor::from_vec(
        vec![
            1.0f32, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
        [2, 1, 4, 4],
        &METAL,
    )?;
    let labels = Tensor::from_vec(vec![0i64, 1], [2], &METAL)?;
    let mut adam = Adam::new(0.02);
    let first = cnn
        .forward(&x, Mode::EVAL)?
        .cross_entropy(&labels)?
        .item()?;
    eprintln!("metal CNN initial loss: {first}");
    for _ in 0..20 {
        let loss = cnn.forward(&x, Mode::TRAIN)?.cross_entropy(&labels)?;
        adam.step(&mut cnn, loss.backward()?)?;
    }
    let last = cnn
        .forward(&x, Mode::EVAL)?
        .cross_entropy(&labels)?
        .item()?;
    eprintln!("metal CNN final loss: {last}");
    assert!(last < first, "CNN loss did not decrease: {first} -> {last}");
    Ok(())
}
