#![cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]

use rstorch::prelude::*;

#[path = "common/wgpu.rs"]
mod gpu;

const WGPU: Device = Device::Wgpu(0);

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

#[derive(Module)]
struct Transformer {
    attention: MultiHeadAttention,
    norm: LayerNorm,
    head: Linear,
}

impl Forward for Transformer {
    fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        let attended = self.attention.attend(x, None, mode)?.add(x)?;
        self.head
            .forward(&self.norm.forward(&attended, mode)?, mode)
    }
}

#[test]
fn seeded_mlp_and_transformer_losses_decrease_on_wgpu() -> Result<()> {
    if !gpu::available() {
        return Ok(());
    }

    let mut rng = Rng::seed(61);
    let mut mlp = Mlp {
        first: Linear::new(4, 8, &WGPU, &mut rng)?,
        second: Linear::new(8, 3, &WGPU, &mut rng)?,
    };
    let x = Tensor::from_vec(
        vec![
            1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 0.0, 0.0, -1.0,
        ],
        [3, 4],
        &WGPU,
    )?;
    let targets = Tensor::from_vec(
        vec![1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        [3, 3],
        &WGPU,
    )?;
    let mut sgd = Sgd::new(0.05);
    let first = mlp.forward(&x, Mode::EVAL)?.mse_loss(&targets)?.item()?;
    for _ in 0..20 {
        let loss = mlp.forward(&x, Mode::TRAIN)?.mse_loss(&targets)?;
        sgd.step(&mut mlp, loss.backward()?)?;
    }
    let last = mlp.forward(&x, Mode::EVAL)?.mse_loss(&targets)?.item()?;
    assert!(last < first, "MLP loss did not decrease: {first} -> {last}");

    let mut rng = Rng::seed(7);
    let mut transformer = Transformer {
        attention: MultiHeadAttention::new(4, 2, &WGPU, &mut rng)?,
        norm: LayerNorm::new([4], &WGPU)?,
        head: Linear::new(4, 6, &WGPU, &mut rng)?,
    };
    let tokens = Tensor::from_vec(
        vec![
            1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
        [2, 3, 4],
        &WGPU,
    )?;
    let targets = Tensor::from_vec(
        vec![
            0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0,
        ],
        [2, 3, 6],
        &WGPU,
    )?;
    let loss = |model: &mut Transformer, mode| model.forward(&tokens, mode)?.mse_loss(&targets);
    let mut adam = AdamW::new(0.02, 0.0);
    let first = loss(&mut transformer, Mode::EVAL)?.item()?;
    for _ in 0..12 {
        let step = loss(&mut transformer, Mode::TRAIN)?;
        adam.step(&mut transformer, step.backward()?)?;
    }
    let last = loss(&mut transformer, Mode::EVAL)?.item()?;
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
fn seeded_cnn_loss_decreases_on_wgpu() -> Result<()> {
    if !gpu::available() {
        return Ok(());
    }

    let mut rng = Rng::seed(9);
    let mut cnn = Cnn {
        weight: Param::new(Tensor::randn([2, 1, 2, 2], DType::F32, &WGPU, &mut rng)?),
        head: Linear::new(2, 2, &WGPU, &mut rng)?,
    };
    let x = Tensor::from_vec(
        vec![
            1.0f32, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
        ],
        [2, 1, 4, 4],
        &WGPU,
    )?;
    let targets = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], [2, 2], &WGPU)?;
    let mut adam = Adam::new(0.02);
    let first = cnn.forward(&x, Mode::EVAL)?.mse_loss(&targets)?.item()?;
    for _ in 0..15 {
        let loss = cnn.forward(&x, Mode::TRAIN)?.mse_loss(&targets)?;
        adam.step(&mut cnn, loss.backward()?)?;
    }
    let last = cnn.forward(&x, Mode::EVAL)?.mse_loss(&targets)?.item()?;
    assert!(last < first, "CNN loss did not decrease: {first} -> {last}");
    Ok(())
}

#[test]
fn nan_and_empty_axis_semantics_match_cpu() -> Result<()> {
    if !gpu::available() {
        return Ok(());
    }

    for device in [Device::Cpu, WGPU] {
        let x = Tensor::from_vec(vec![f32::NAN, 1.0, 2.0, -3.0], [4], &device)?;
        let y = Tensor::from_vec(vec![5.0f32, f32::NAN, 3.0, -4.0], [4], &device)?;
        let maximum = x.maximum(&y)?.to_device(&Device::Cpu)?.to_vec::<f32>()?;
        let minimum = x.minimum(&y)?.to_device(&Device::Cpu)?.to_vec::<f32>()?;
        let relu = x.relu()?.to_device(&Device::Cpu)?.to_vec::<f32>()?;
        assert!(maximum[0].is_nan() && maximum[1].is_nan(), "{device}");
        assert!(minimum[0].is_nan() && minimum[1].is_nan(), "{device}");
        assert!(relu[0].is_nan(), "{device}: relu dropped NaN");
        assert_eq!(&maximum[2..], &[3.0, -3.0]);
        assert_eq!(&minimum[2..], &[2.0, -4.0]);

        let extrema = Tensor::from_vec(vec![1.0f32, f32::NAN, 3.0], [3], &device)?;
        assert!(extrema.max_all()?.item()?.is_nan());
        assert!(extrema.min_all()?.item()?.is_nan());
    }

    for dims in [vec![0usize, 3], vec![2, 0, 3], vec![3, 0]] {
        let axis = dims.iter().position(|&dim| dim == 0).unwrap();
        let mut results = Vec::new();
        for device in [Device::Cpu, WGPU] {
            let sum = Tensor::zeros(dims.clone(), DType::F32, &device)?.sum(axis as isize)?;
            results.push((
                sum.dims().to_vec(),
                sum.to_device(&Device::Cpu)?.to_vec::<f32>()?,
            ));
        }
        assert_eq!(results[0], results[1], "dims {dims:?}, axis {axis}");
    }
    Ok(())
}

#[test]
fn empty_index_select_still_validates_indices_on_wgpu() -> Result<()> {
    if !gpu::available() {
        return Ok(());
    }

    let source = Tensor::from_vec(Vec::<f32>::new(), [3, 0], &WGPU)?;
    let indices = Tensor::from_vec(vec![3i64], [1], &WGPU)?;
    let selected = source.index_select(0, &indices)?;
    assert_eq!(selected.dims(), &[1, 0]);
    assert!(matches!(
        selected.to_vec::<f32>(),
        Err(Error::IndexOutOfBounds {
            op: "index_select",
            index: 3,
            axis: 0,
            size: 3,
        })
    ));
    Ok(())
}

#[test]
fn arg_reductions_and_max_pool_nonfinite_values_match_cpu() -> Result<()> {
    if !gpu::available() {
        return Ok(());
    }

    for values in [
        vec![1.0f32, f32::NAN, 3.0, f32::NAN],
        vec![2.0f32, 2.0, 1.0, 1.0],
    ] {
        let reference = Tensor::from_vec(values.clone(), [4], &Device::Cpu)?;
        let candidate = Tensor::from_vec(values, [4], &WGPU)?;
        assert_eq!(
            candidate.argmax(0)?.to_vec::<i64>()?,
            reference.argmax(0)?.to_vec::<i64>()?
        );
        assert_eq!(
            candidate.argmin(0)?.to_vec::<i64>()?,
            reference.argmin(0)?.to_vec::<i64>()?
        );
    }

    for values in [
        vec![f32::NEG_INFINITY; 4],
        vec![1.0f32, f32::NAN, 3.0, 4.0],
        vec![f32::NAN, f32::NAN, 3.0, 4.0],
    ] {
        let run = |device: &Device| -> Result<(Vec<f32>, Vec<f32>)> {
            let input = Param::new(Tensor::from_vec(values.clone(), [1, 1, 2, 2], device)?);
            let pooled = input.get(Mode::TRAIN).max_pool2d((2, 2), (2, 2), (0, 0))?;
            let output = pooled.to_vec::<f32>()?;
            let gradient = pooled.backward()?.wrt(&input)?.to_vec::<f32>()?;
            Ok((output, gradient))
        };
        let (cpu_output, cpu_gradient) = run(&Device::Cpu)?;
        let (wgpu_output, wgpu_gradient) = run(&WGPU)?;
        if cpu_output[0].is_nan() {
            assert!(wgpu_output[0].is_nan());
        } else {
            assert_eq!(wgpu_output, cpu_output);
        }
        assert_eq!(wgpu_gradient, cpu_gradient);
    }
    Ok(())
}
