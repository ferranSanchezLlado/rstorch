//! Metal **fused** kernels against the CPU reference: softmax, the
//! normalization layers, cross-entropy, and the optimizer steps — forward and
//! backward, contiguous and strided.
//!
//! The `backend::conformance` table deliberately excludes fused ops (their
//! multi-output encodings do not fit its single-output harness), and the
//! in-crate Metal fused test checks output *shapes and dtypes* for `LayerNorm`
//! plus hardcoded values for one SGD step. Nothing compared a fused Metal
//! **value** against the CPU reference, which is where a hand-written
//! threadgroup reduction or a mis-set eps hides: the numbers stay finite and
//! the model still trains, only worse.
//!
//! Every case runs the identical graph on both devices from identical host
//! bytes. `Device::Cpu` is the reference (see `backend::conformance`).

#![cfg(all(feature = "metal", target_os = "macos"))]

#[path = "common/metal.rs"]
mod gpu;

use rstorch::prelude::*;

const METAL: Device = Device::Metal(0);

/// Deterministic host values.
fn values(seed: u64, len: usize) -> Vec<f32> {
    let mut rng = Rng::seed(seed);
    (0..len).map(|_| rng.uniform(-2.0, 2.0) as f32).collect()
}

fn assert_close(what: &str, cpu: &[f32], metal: &[f32]) {
    assert_eq!(cpu.len(), metal.len(), "{what}: length differs");
    for (i, (want, got)) in cpu.iter().zip(metal).enumerate() {
        let tolerance = 2e-4 * want.abs().max(1.0);
        assert!(
            (want - got).abs() <= tolerance,
            "{what}[{i}] = {got} on Metal, {want} on the CPU reference"
        );
    }
}

/// Run `build` on both devices and diff the results elementwise.
fn compare(what: &str, build: impl Fn(&Device) -> Vec<f32>) {
    assert_close(what, &build(&Device::Cpu), &build(&METAL));
}

/// The value and the gradient of `sum(f(x) * cotangent)` with respect to `x`,
/// so one comparison covers both directions of a fused kernel.
fn value_and_grad(
    device: &Device,
    dims: [usize; 2],
    seed: u64,
    f: impl Fn(&Tensor) -> Tensor,
) -> Vec<f32> {
    let x = Param::new(Tensor::from_vec(values(seed, dims[0] * dims[1]), dims, device).unwrap());
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
fn softmax_and_log_softmax_match_the_cpu() {
    if !gpu::available() {
        return;
    }
    for dims in [[4usize, 8], [3, 33], [1, 129], [17, 5]] {
        compare(&format!("softmax{dims:?}"), |device| {
            value_and_grad(device, dims, 1, |x| x.softmax(1).unwrap())
        });
        compare(&format!("log_softmax{dims:?}"), |device| {
            value_and_grad(device, dims, 2, |x| x.log_softmax(1).unwrap())
        });
    }
}

/// A softmax over a transposed view: the last *logical* axis is not the
/// contiguous one, which is the case a fused row-wise kernel gets wrong when
/// it assumes a packed row.
#[test]
fn softmax_over_a_strided_view_matches_the_cpu() {
    if !gpu::available() {
        return;
    }
    compare("softmax.transposed", |device| {
        let x = Param::new(Tensor::from_vec(values(3, 24), [4, 6], device).unwrap());
        let transposed = x.get(Mode::TRAIN).transpose(0, 1).unwrap();
        let out = transposed.softmax(1).unwrap();
        let mut result = out.to_vec::<f32>().unwrap();
        let grads = out.sum_all().unwrap().backward().unwrap();
        result.extend(grads.wrt(&x).unwrap().to_vec::<f32>().unwrap());
        result
    });
}

#[test]
fn layer_norm_and_rms_norm_match_the_cpu() {
    if !gpu::available() {
        return;
    }
    for width in [4usize, 31, 128] {
        compare(&format!("layer_norm.{width}"), |device| {
            value_and_grad(device, [5, width], 4, |x| {
                LayerNorm::new([width], device)
                    .unwrap()
                    .forward(x, Mode::TRAIN)
                    .unwrap()
            })
        });
        compare(&format!("rms_norm.{width}"), |device| {
            value_and_grad(device, [5, width], 5, |x| {
                RMSNorm::new([width], device)
                    .unwrap()
                    .forward(x, Mode::TRAIN)
                    .unwrap()
            })
        });
    }
}

#[test]
fn wide_parallel_reductions_match_the_cpu_forward_and_backward() {
    if !gpu::available() {
        return;
    }
    compare("softmax.parallel.1024", |device| {
        value_and_grad(device, [7, 1024], 51, |x| x.softmax(1).unwrap())
    });
    compare("layer_norm.parallel.1024", |device| {
        value_and_grad(device, [7, 1024], 52, |x| {
            LayerNorm::new([1024], device)
                .unwrap()
                .forward(x, Mode::TRAIN)
                .unwrap()
        })
    });
}

#[test]
fn contiguous_index_backward_fast_paths_match_the_cpu_with_duplicates() {
    if !gpu::available() {
        return;
    }
    compare("index_select.axis0.backward", |device| {
        let table = Param::new(Tensor::from_vec(values(53, 257 * 64), [257, 64], device).unwrap());
        let ids = Tensor::from_vec(vec![3i64, 200, 3, 0, 256, 200, 17], [7], device).unwrap();
        let selected = table.get(Mode::TRAIN).index_select(0, &ids).unwrap();
        let grads = selected.sum_all().unwrap().backward().unwrap();
        grads.wrt(&table).unwrap().to_vec::<f32>().unwrap()
    });

    compare("gather.last.backward", |device| {
        let logits =
            Param::new(Tensor::from_vec(values(54, 128 * 257), [128, 257], device).unwrap());
        let indices = Tensor::from_vec(
            (0..128).map(|row| ((row * 43) % 257) as i64).collect(),
            [128, 1],
            device,
        )
        .unwrap();
        let selected = logits.get(Mode::TRAIN).gather(1, &indices).unwrap();
        let grads = selected.sum_all().unwrap().backward().unwrap();
        grads.wrt(&logits).unwrap().to_vec::<f32>().unwrap()
    });
}

/// The normalization layers' own parameters also receive gradients; a kernel
/// that gets the input gradient right can still get `weight`/`bias` wrong.
#[test]
fn layer_norm_parameter_gradients_match_the_cpu() {
    if !gpu::available() {
        return;
    }
    compare("layer_norm.params", |device| {
        let mut layer = LayerNorm::new([9usize], device).unwrap();
        let x = Tensor::from_vec(values(6, 7 * 9), [7, 9], device).unwrap();
        let out = layer.forward(&x, Mode::TRAIN).unwrap();
        let cotangent = Tensor::from_vec(values(7, 7 * 9), [7, 9], device).unwrap();
        let grads = out
            .mul(&cotangent)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let mut sgd = Sgd::new(0.5);
        sgd.step(&mut layer, grads).unwrap();
        layer
            .state_dict()
            .into_values()
            .flat_map(|tensor| tensor.to_vec::<f32>().unwrap())
            .collect()
    });
}

#[test]
fn cross_entropy_matches_the_cpu() {
    if !gpu::available() {
        return;
    }
    compare("cross_entropy", |device| {
        let logits = Param::new(Tensor::from_vec(values(8, 6 * 10), [6, 10], device).unwrap());
        let targets = Tensor::from_vec(vec![0i64, 3, 9, 5, 1, 7], [6], device).unwrap();
        let loss = logits.get(Mode::TRAIN).cross_entropy(&targets).unwrap();
        let mut result = vec![loss.item().unwrap() as f32];
        let grads = loss.backward().unwrap();
        result.extend(grads.wrt(&logits).unwrap().to_vec::<f32>().unwrap());
        result
    });
}

/// Ten steps of each optimizer configuration. One step can agree by accident
/// when a momentum or a bias-correction term is mishandled; ten cannot.
#[test]
fn optimizer_steps_match_the_cpu() {
    if !gpu::available() {
        return;
    }
    #[derive(rstorch::Module)]
    struct Model {
        first: Linear,
        second: Linear,
    }

    impl Forward for Model {
        type Output = Tensor;

        fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
            self.second
                .forward(&self.first.forward(x, mode)?.relu()?, mode)
        }
    }

    let run = |device: &Device, kind: &str| -> Vec<f32> {
        let mut rng = Rng::seed(9);
        let mut model = Model {
            first: Linear::new(6, 12, device, &mut rng).unwrap(),
            second: Linear::new(12, 4, device, &mut rng).unwrap(),
        };
        let x = Tensor::from_vec(values(10, 8 * 6), [8, 6], device).unwrap();
        let targets = Tensor::from_vec(vec![0i64, 1, 2, 3, 3, 2, 1, 0], [8], device).unwrap();

        let mut sgd = Sgd::new(0.05).momentum(0.9).weight_decay(0.01);
        let mut adam = Adam::new(0.01).weight_decay(0.01);
        let mut adamw = AdamW::new(0.01, 0.01);
        for _ in 0..10 {
            let grads = model
                .forward(&x, Mode::TRAIN)
                .unwrap()
                .cross_entropy(&targets)
                .unwrap()
                .backward()
                .unwrap();
            match kind {
                "sgd" => sgd.step(&mut model, grads).unwrap(),
                "adam" => adam.step(&mut model, grads).unwrap(),
                _ => adamw.step(&mut model, grads).unwrap(),
            }
        }
        model
            .state_dict()
            .into_values()
            .flat_map(|tensor| tensor.to_vec::<f32>().unwrap())
            .collect()
    };

    for kind in ["sgd", "adam", "adamw"] {
        compare(&format!("optimizer.{kind}"), |device| run(device, kind));
    }
}

/// Attention composes matmul, a scaled softmax and two transposes — the path
/// where a fused softmax meets non-contiguous operands.
#[test]
fn attention_matches_the_cpu() {
    if !gpu::available() {
        return;
    }
    compare("attention", |device| {
        let mut rng = Rng::seed(11);
        let mut attention = MultiHeadAttention::new(16, 4, device, &mut rng).unwrap();
        let x = Tensor::from_vec(values(12, 2 * 5 * 16), [2, 5, 16], device).unwrap();
        let out = attention.attend(&x, None, Mode::TRAIN).unwrap();
        let mut result = out.to_vec::<f32>().unwrap();
        let grads = out.sum_all().unwrap().backward().unwrap();
        let mut sgd = Sgd::new(0.1);
        sgd.step(&mut attention, grads).unwrap();
        result.extend(
            attention
                .state_dict()
                .into_values()
                .flat_map(|tensor| tensor.to_vec::<f32>().unwrap()),
        );
        result
    });
}

/// `BatchNorm` keeps running buffers, so a Metal divergence accumulates across
/// forwards instead of showing up in one.
#[test]
fn batch_norm_running_statistics_match_the_cpu() {
    if !gpu::available() {
        return;
    }
    compare("batch_norm", |device| {
        let mut layer = BatchNorm2d::new(3, device).unwrap();
        for step in 0..5 {
            let x =
                Tensor::from_vec(values(13 + step, 4 * 3 * 5 * 5), [4, 3, 5, 5], device).unwrap();
            let out = layer.forward(&x, Mode::TRAIN).unwrap();
            let grads = out.sum_all().unwrap().backward().unwrap();
            let mut sgd = Sgd::new(0.1);
            sgd.step(&mut layer, grads).unwrap();
        }
        let x = Tensor::from_vec(values(99, 4 * 3 * 5 * 5), [4, 3, 5, 5], device).unwrap();
        let mut result = layer
            .forward(&x, Mode::EVAL)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        result.extend(
            layer
                .state_dict()
                .into_values()
                .flat_map(|tensor| tensor.to_vec::<f32>().unwrap()),
        );
        result
    });
}
