#![cfg(all(feature = "metal", target_os = "macos"))]

#[path = "common/metal.rs"]
mod gpu;

use rstorch::models::{DecoderTransformer, TransformerConfig};
use rstorch::prelude::*;

const METAL: Device = Device::Metal(0);

#[derive(Module)]
struct Mlp {
    first: Linear,
    second: Linear,
}

impl Forward for Mlp {
    type Output = Tensor;

    fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        self.second
            .forward(&self.first.forward(x, mode)?.gelu()?, mode)
    }
}

#[test]
fn seeded_mlp_and_transformer_losses_decrease_on_metal() -> Result<()> {
    if !gpu::available() {
        return Ok(());
    }
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

    let config = TransformerConfig::new(6, 3, 4, 2, 1).with_feed_forward_dim(8);
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
    type Output = Tensor;

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
    if !gpu::available() {
        return Ok(());
    }
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

/// Metal-vs-CPU parity for the public `LayerNorm` backward graph across the
/// supported input ranks. This supplements entry-point conformance.
#[test]
fn layer_norm_gradients_match_cpu_on_every_rank() -> Result<()> {
    if !gpu::available() {
        return Ok(());
    }
    // Deterministic, non-symmetric coefficients so an error cannot cancel.
    fn coefficients(n: usize) -> Vec<f32> {
        (0..n).map(|i| 0.25 + (i as f32) * 0.5).collect()
    }

    for dims in [
        vec![5usize],
        vec![6, 4],
        vec![2, 3, 4],
        vec![2, 2, 3, 4],
        vec![3, 1, 4],
    ] {
        let width = *dims.last().unwrap();
        let count: usize = dims.iter().product();
        let values = coefficients(count);
        let coef = coefficients(count).into_iter().rev().collect::<Vec<_>>();

        // Run the identical computation on both devices and compare gradients.
        let mut grads = Vec::new();
        for dev in [Device::Cpu, METAL] {
            let mut rng = Rng::seed(9);
            let mut norm = LayerNorm::new([width], &dev)?;
            let x = Tensor::from_vec(values.clone(), dims.clone(), &dev)?.traced()?;
            let c = Tensor::from_vec(coef.clone(), dims.clone(), &dev)?;
            let out = norm.forward(&x, Mode::TRAIN)?;
            let g = out.mul(&c)?.sum_all()?.backward()?;
            let dx = g.wrt_input(&x)?.to_device(&Device::Cpu)?.to_vec::<f32>()?;
            let _ = &mut rng;
            grads.push(dx);
        }

        let (cpu, metal) = (&grads[0], &grads[1]);
        assert_eq!(cpu.len(), metal.len(), "dims {dims:?}: length mismatch");
        let worst = cpu
            .iter()
            .zip(metal)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 1e-4,
            "dims {dims:?}: LayerNorm input gradient differs by {worst}\n cpu={cpu:?}\n metal={metal:?}"
        );
    }
    Ok(())
}

/// CPU/Metal parity for NaN handling in the extremum family.
///
/// The `conformance` op × dtype table carries no NaN in its fixture data, so
/// nothing else compares the two backends on the one input class where
/// `maximum`/`minimum`/`relu`/`argmax` have a real choice to make. All four
/// must *propagate* NaN (as `PyTorch` does), and `argmax`/`argmin` must select
/// the NaN so they agree with what `max`/`min` report.
#[test]
fn nan_semantics_match_between_cpu_and_metal() -> Result<()> {
    if !gpu::available() {
        return Ok(());
    }
    let a = vec![f32::NAN, 1.0, 2.0, -3.0];
    let b = vec![5.0f32, f32::NAN, 3.0, -4.0];

    for dev in [Device::Cpu, METAL] {
        let x = Tensor::from_vec(a.clone(), [4], &dev)?;
        let y = Tensor::from_vec(b.clone(), [4], &dev)?;
        let host = |t: Tensor| -> Result<Vec<f32>> { t.to_device(&Device::Cpu)?.to_vec::<f32>() };

        let mx = host(x.maximum(&y)?)?;
        let mn = host(x.minimum(&y)?)?;
        let rl = host(x.relu()?)?;
        assert!(
            mx[0].is_nan() && mx[1].is_nan(),
            "{dev}: maximum dropped NaN: {mx:?}"
        );
        assert!(
            mn[0].is_nan() && mn[1].is_nan(),
            "{dev}: minimum dropped NaN: {mn:?}"
        );
        assert_eq!(&mx[2..], &[3.0, -3.0], "{dev}: clean lanes wrong");
        assert_eq!(&mn[2..], &[2.0, -4.0], "{dev}: clean lanes wrong");
        assert!(rl[0].is_nan(), "{dev}: relu laundered NaN: {rl:?}");
        assert_eq!(&rl[1..], &[1.0, 2.0, 0.0], "{dev}: relu clean lanes wrong");

        // max/min propagate, so argmax/argmin must name the NaN's index.
        let m = Tensor::from_vec(vec![1.0f32, f32::NAN, 3.0], [3], &dev)?;
        assert!(
            m.max_all()?.item()?.is_nan() && m.min_all()?.item()?.is_nan(),
            "{dev}: max_all/min_all must propagate NaN"
        );
        assert_eq!(
            m.argmax(0)?.to_device(&Device::Cpu)?.to_vec::<i64>()?,
            vec![1],
            "{dev}: argmax must select the NaN"
        );
        assert_eq!(
            m.argmin(0)?.to_device(&Device::Cpu)?.to_vec::<i64>()?,
            vec![1],
            "{dev}: argmin must select the NaN"
        );
    }
    Ok(())
}

/// `i64::MIN / -1` overflows, which is distinct from division by zero: under
/// the documented wrapping contract it is `i64::MIN`, not `0`. Pinned on both
/// backends because they implement the guard separately.
#[test]
fn i64_division_overflow_matches_between_cpu_and_metal() -> Result<()> {
    if !gpu::available() {
        return Ok(());
    }
    for dev in [Device::Cpu, METAL] {
        let a = Tensor::from_vec(vec![i64::MIN, i64::MIN, -8, 7], [4], &dev)?;
        let b = Tensor::from_vec(vec![-1i64, 1, 2, 0], [4], &dev)?;
        let got = a.div(&b)?.to_device(&Device::Cpu)?.to_vec::<i64>()?;
        assert_eq!(
            got,
            vec![i64::MIN, i64::MIN, -4, 0],
            "{dev}: i64 division semantics wrong"
        );
    }
    Ok(())
}

/// `sum` over an empty axis is legal and returns the identity. Metal computed
/// its output length as `num_elements() / dims()[axis]`, which divides by zero
/// on an empty axis — a panic where the CPU backend returns zeros.
#[test]
fn empty_axis_sum_matches_cpu_instead_of_panicking() -> Result<()> {
    if !gpu::available() {
        return Ok(());
    }
    for dims in [vec![0usize, 3], vec![2, 0, 3], vec![3, 0]] {
        let axis = dims.iter().position(|&d| d == 0).unwrap();
        let mut results = Vec::new();
        for dev in [Device::Cpu, METAL] {
            let x = Tensor::zeros(dims.clone(), DType::F32, &dev)?;
            let summed = x.sum(axis as isize)?;
            results.push((
                summed.dims().to_vec(),
                summed.to_device(&Device::Cpu)?.to_vec::<f32>()?,
            ));
        }
        assert_eq!(
            results[0], results[1],
            "dims {dims:?} axis {axis}: Metal and CPU disagree on an empty-axis sum"
        );
    }
    Ok(())
}
