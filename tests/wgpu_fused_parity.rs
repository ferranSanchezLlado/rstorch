#![cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]

use rstorch::nn;
use rstorch::prelude::*;

#[path = "common/wgpu.rs"]
mod gpu;

const WGPU: Device = Device::Wgpu(0);

fn values(seed: u64, len: usize) -> Vec<f32> {
    let mut rng = Rng::seed(seed);
    (0..len).map(|_| rng.uniform(-2.0, 2.0) as f32).collect()
}

fn assert_close(what: &str, cpu: &[f32], wgpu: &[f32]) {
    assert_eq!(cpu.len(), wgpu.len(), "{what}: length differs");
    for (index, (want, got)) in cpu.iter().zip(wgpu).enumerate() {
        let tolerance = 3e-4 * want.abs().max(1.0);
        assert!(
            (want - got).abs() <= tolerance,
            "{what}[{index}] = {got} on wgpu, {want} on CPU"
        );
    }
}

fn compare(what: &str, build: impl Fn(&Device) -> Vec<f32>) {
    assert_close(what, &build(&Device::Cpu), &build(&WGPU));
}

fn value_and_grad(
    device: &Device,
    dims: [usize; 2],
    seed: u64,
    f: impl Fn(&Tensor) -> Tensor,
) -> Vec<f32> {
    let x =
        Param::new(Tensor::from_vec(values(seed, dims.iter().product()), dims, device).unwrap());
    let out = f(&x.get(Mode::TRAIN));
    let cotangent = Tensor::from_vec(
        values(seed + 100, out.dims().iter().product()),
        out.dims().to_vec(),
        device,
    )
    .unwrap();
    let mut result = out.to_vec::<f32>().unwrap();
    let grads = out
        .mul(&cotangent)
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    result.extend(grads.wrt(&x).unwrap().to_vec::<f32>().unwrap());
    result
}

#[test]
fn softmax_forward_and_backward_match_cpu() {
    if !gpu::available() {
        return;
    }
    for dims in [[4usize, 8], [3, 33], [2, 129]] {
        compare(&format!("softmax{dims:?}"), |device| {
            value_and_grad(device, dims, 1, |x| x.softmax(1).unwrap())
        });
    }
    compare("softmax.eval", |device| {
        Tensor::from_vec(values(2, 5 * 17), [5, 17], device)
            .unwrap()
            .softmax(1)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
    });
}

#[test]
fn layer_norm_forward_backward_and_parameters_match_cpu() {
    if !gpu::available() {
        return;
    }
    for width in [4usize, 31, 128] {
        compare(&format!("layer_norm.{width}"), |device| {
            value_and_grad(device, [4, width], 3, |x| {
                LayerNorm::new([width], device)
                    .unwrap()
                    .forward(x, Mode::TRAIN)
                    .unwrap()
            })
        });
    }
    compare("layer_norm.eval", |device| {
        let mut layer = LayerNorm::new([9usize], device).unwrap();
        let x = Tensor::from_vec(values(4, 6 * 9), [6, 9], device).unwrap();
        layer
            .forward(&x, Mode::EVAL)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
    });
    compare("layer_norm.params", |device| {
        let mut layer = LayerNorm::new([9usize], device).unwrap();
        let x = Tensor::from_vec(values(5, 6 * 9), [6, 9], device).unwrap();
        let out = layer.forward(&x, Mode::TRAIN).unwrap();
        let cotangent = Tensor::from_vec(values(6, 6 * 9), [6, 9], device).unwrap();
        let grads = out
            .mul(&cotangent)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        Sgd::new(0.5).step(&mut layer, grads).unwrap();
        nn::state_dict(&layer)
            .into_values()
            .flat_map(|tensor| tensor.to_vec::<f32>().unwrap())
            .collect()
    });
}

#[test]
fn attention_and_indexing_backward_match_cpu() {
    if !gpu::available() {
        return;
    }
    compare("attention", |device| {
        let mut attention = MultiHeadAttention::new(12, 3, device, &mut Rng::seed(11)).unwrap();
        let x = Tensor::from_vec(values(12, 2 * 4 * 12), [2, 4, 12], device).unwrap();
        let out = attention.attend(&x, None, Mode::TRAIN).unwrap();
        let mut result = out.to_vec::<f32>().unwrap();
        let grads = out.sum_all().unwrap().backward().unwrap();
        Sgd::new(0.1).step(&mut attention, grads).unwrap();
        result.extend(
            nn::state_dict(&attention)
                .into_values()
                .flat_map(|tensor| tensor.to_vec::<f32>().unwrap()),
        );
        result
    });

    compare("index_select.backward", |device| {
        let table = Param::new(Tensor::from_vec(values(13, 19 * 8), [19, 8], device).unwrap());
        let ids = Tensor::from_vec(vec![3i64, 17, 3, 0, 18, 17], [6], device).unwrap();
        let selected = table.get(Mode::TRAIN).index_select(0, &ids).unwrap();
        let grads = selected.sum_all().unwrap().backward().unwrap();
        grads.wrt(&table).unwrap().to_vec::<f32>().unwrap()
    });
    compare("gather.backward", |device| {
        let logits = Param::new(Tensor::from_vec(values(14, 16 * 23), [16, 23], device).unwrap());
        let indices = Tensor::from_vec(
            (0..16).map(|row| ((row * 7) % 23) as i64).collect(),
            [16, 1],
            device,
        )
        .unwrap();
        let selected = logits.get(Mode::TRAIN).gather(1, &indices).unwrap();
        let grads = selected.sum_all().unwrap().backward().unwrap();
        grads.wrt(&logits).unwrap().to_vec::<f32>().unwrap()
    });
}

#[test]
fn cross_entropy_and_optimizers_match_cpu() {
    if !gpu::available() {
        return;
    }
    compare("cross_entropy", |device| {
        let logits = Param::new(Tensor::from_vec(values(15, 6 * 10), [6, 10], device).unwrap());
        let targets = Tensor::from_vec(vec![0i64, 3, 9, 5, 1, 7], [6], device).unwrap();
        let loss = logits.get(Mode::TRAIN).cross_entropy(&targets).unwrap();
        vec![loss.item().unwrap() as f32]
    });

    #[derive(Module)]
    struct Model {
        first: Linear,
        second: Linear,
    }
    impl Forward for Model {
        fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
            self.second
                .forward(&self.first.forward(x, mode)?.relu()?, mode)
        }
    }
    let run = |device: &Device, kind: &str| {
        let mut rng = Rng::seed(16);
        let mut model = Model {
            first: Linear::new(6, 10, device, &mut rng).unwrap(),
            second: Linear::new(10, 4, device, &mut rng).unwrap(),
        };
        let x = Tensor::from_vec(values(17, 8 * 6), [8, 6], device).unwrap();
        let targets = Tensor::from_vec(values(18, 8 * 4), [8, 4], device).unwrap();
        let mut sgd = Sgd::new(0.05).momentum(0.9).weight_decay(0.01);
        let mut adam = Adam::new(0.01).weight_decay(0.01);
        let mut adamw = AdamW::new(0.01, 0.01);
        for _ in 0..5 {
            let grads = model
                .forward(&x, Mode::TRAIN)
                .unwrap()
                .mse_loss(&targets)
                .unwrap()
                .backward()
                .unwrap();
            match kind {
                "sgd" => sgd.step(&mut model, grads).unwrap(),
                "adam" => adam.step(&mut model, grads).unwrap(),
                _ => adamw.step(&mut model, grads).unwrap(),
            }
        }
        nn::state_dict(&model)
            .into_values()
            .flat_map(|tensor| tensor.to_vec::<f32>().unwrap())
            .collect::<Vec<_>>()
    };
    for kind in ["sgd", "adam", "adamw"] {
        compare(&format!("optimizer.{kind}"), |device| run(device, kind));
    }
}
