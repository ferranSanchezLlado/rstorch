#![cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]

use half::f16;
use rstorch::prelude::*;

#[path = "common/wgpu.rs"]
mod gpu;

const WGPU: Device = Device::Wgpu(0);

fn values(seed: u64, len: usize) -> Vec<f16> {
    let mut rng = Rng::seed(seed);
    (0..len)
        .map(|_| f16::from_f32(rng.uniform(-1.0, 1.0) as f32))
        .collect()
}

fn f16_available_or_loud() -> bool {
    if !gpu::available() {
        return false;
    }
    match Tensor::from_vec(vec![f16::ONE], [1], &WGPU) {
        Ok(_) => true,
        Err(Error::Unsupported {
            device: Device::Wgpu(0),
            dtype: DType::F16,
            ..
        }) => {
            eprintln!(
                "skipping WGPU F16 test: this adapter does not support the SHADER_F16 capability"
            );
            false
        }
        Err(error) => panic!("unexpected F16 capability probe error: {error}"),
    }
}

fn assert_close(what: &str, want: &[f16], got: &[f16], tolerance: f32) {
    assert_eq!(want.len(), got.len(), "{what}: length differs");
    for (index, (want, got)) in want.iter().zip(got).enumerate() {
        let want = want.to_f32();
        let got = got.to_f32();
        assert!(
            (want - got).abs() <= tolerance * want.abs().max(1.0),
            "{what}[{index}]: expected {want}, got {got}"
        );
    }
}

fn on(device: &Device, data: &[f16], dims: impl Into<rstorch::Shape>) -> Tensor {
    Tensor::from_vec(data.to_vec(), dims, device).unwrap()
}

#[test]
fn reductions_matmul_and_mixed_index_backward_match_cpu() {
    if !f16_available_or_loud() {
        return;
    }
    let x = values(1, 7 * 257);
    let lhs = values(2, 9 * 33);
    let rhs = values(3, 33 * 11);
    for (what, run) in [
        (
            "sum",
            Box::new(|device: &Device| on(device, &x, [7, 257]).sum(1).unwrap())
                as Box<dyn Fn(&Device) -> Tensor>,
        ),
        (
            "matmul",
            Box::new(|device: &Device| {
                on(device, &lhs, [9, 33])
                    .matmul(&on(device, &rhs, [33, 11]))
                    .unwrap()
            }),
        ),
    ] {
        let cpu = run(&Device::Cpu).to_vec::<f16>().unwrap();
        let wgpu = run(&WGPU).to_vec::<f16>().unwrap();
        assert_close(what, &cpu, &wgpu, 3e-2);
    }

    let table = values(4, 19 * 8);
    let gradient = |device: &Device| {
        let table = Param::new(on(device, &table, [19, 8]));
        let ids = Tensor::from_vec(vec![3i64, 17, 3, 0, 18, 17], [6], device).unwrap();
        let selected = table.get(Mode::TRAIN).index_select(0, &ids).unwrap();
        selected
            .sum_all()
            .unwrap()
            .backward()
            .unwrap()
            .wrt(&table)
            .unwrap()
            .to_vec::<f16>()
            .unwrap()
    };
    assert_close(
        "index_select.backward",
        &gradient(&Device::Cpu),
        &gradient(&WGPU),
        0.0,
    );

    #[derive(Module)]
    struct OneParameter {
        parameter: Param,
    }
    let adam_step = |device: &Device| {
        let mut model = OneParameter {
            parameter: Param::new(on(device, &values(8, 32), [8, 4])),
        };
        let loss = model
            .parameter
            .get(Mode::TRAIN)
            .mul(&model.parameter.get(Mode::TRAIN))
            .unwrap()
            .sum_all()
            .unwrap();
        Adam::new(0.01)
            .step(&mut model, loss.backward().unwrap())
            .unwrap();
        model.parameter.get(Mode::EVAL).to_vec::<f16>().unwrap()
    };
    assert_close(
        "adam.composed_step",
        &adam_step(&Device::Cpu),
        &adam_step(&WGPU),
        2e-3,
    );
}

#[test]
fn conv_backward_softmax_and_eval_layer_norm_match_cpu() {
    if !f16_available_or_loud() {
        return;
    }
    let x = values(10, 2 * 2 * 7 * 7);
    let weight = values(11, 3 * 2 * 3 * 3);
    let conv_grads = |device: &Device| {
        let input = Param::new(on(device, &x, [2, 2, 7, 7]));
        let weight = Param::new(on(device, &weight, [3, 2, 3, 3]));
        let output = input
            .get(Mode::TRAIN)
            .conv2d(&weight.get(Mode::TRAIN), (2, 2), (1, 1), (1, 1))
            .unwrap();
        let scale = on(
            device,
            &values(12, output.dims().iter().product()),
            output.dims().to_vec(),
        );
        let grads = output
            .mul(&scale)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let mut result = grads.wrt(&input).unwrap().to_vec::<f16>().unwrap();
        result.extend(grads.wrt(&weight).unwrap().to_vec::<f16>().unwrap());
        result
    };
    assert_close(
        "conv2d.backward",
        &conv_grads(&Device::Cpu),
        &conv_grads(&WGPU),
        4e-2,
    );

    let rows = values(20, 5 * 129);
    let softmax = |device: &Device| {
        on(device, &rows, [5, 129])
            .softmax(1)
            .unwrap()
            .to_vec::<f16>()
            .unwrap()
    };
    assert_close("softmax", &softmax(&Device::Cpu), &softmax(&WGPU), 4e-3);

    let norm = |device: &Device| {
        let mut layer = LayerNorm::new([129], device).unwrap();
        layer.to_dtype(DType::F16).unwrap();
        layer
            .forward(&on(device, &rows, [5, 129]), Mode::EVAL)
            .unwrap()
            .to_vec::<f16>()
            .unwrap()
    };
    assert_close("layer_norm.eval", &norm(&Device::Cpu), &norm(&WGPU), 3e-2);

    let norm_backward = |device: &Device| {
        let input = Param::new(on(device, &rows, [5, 129]));
        let mut layer = LayerNorm::new([129], device).unwrap();
        layer.to_dtype(DType::F16).unwrap();
        layer
            .forward(&input.get(Mode::TRAIN), Mode::TRAIN)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap()
            .wrt(&input)
            .unwrap()
            .to_vec::<f16>()
            .unwrap()
    };
    assert_close(
        "layer_norm.backward",
        &norm_backward(&Device::Cpu),
        &norm_backward(&WGPU),
        4e-2,
    );
}
