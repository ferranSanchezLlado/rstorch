#![cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]

use rstorch::prelude::*;

const WGPU: Device = Device::Wgpu(0);

fn wgpu_available() -> bool {
    Tensor::from_vec(vec![1.0f32], [1], &WGPU)
        .and_then(|x| x.to_vec::<f32>())
        .is_ok()
}

struct Case {
    input: [usize; 4],
    weight: [usize; 4],
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
}

fn values(seed: u64, len: usize) -> Vec<f32> {
    let mut rng = Rng::seed(seed);
    (0..len).map(|_| rng.uniform(-1.0, 1.0) as f32).collect()
}

fn conv_grads(
    device: &Device,
    case: &Case,
    x: &[f32],
    weight: &[f32],
    scale: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let input = Param::new(Tensor::from_vec(x.to_vec(), case.input, device).unwrap());
    let weight = Param::new(Tensor::from_vec(weight.to_vec(), case.weight, device).unwrap());
    let output = input
        .get(Mode::TRAIN)
        .conv2d(
            &weight.get(Mode::TRAIN),
            case.stride,
            case.padding,
            case.dilation,
        )
        .unwrap();
    let cotangent = Tensor::from_vec(scale.to_vec(), output.dims().to_vec(), device).unwrap();
    let grads = output
        .mul(&cotangent)
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    (
        grads.wrt(&input).unwrap().to_vec::<f32>().unwrap(),
        grads.wrt(&weight).unwrap().to_vec::<f32>().unwrap(),
    )
}

fn assert_close(what: &str, case: usize, cpu: &[f32], wgpu: &[f32]) {
    assert_eq!(cpu.len(), wgpu.len(), "case {case}: {what} length differs");
    for (index, (want, got)) in cpu.iter().zip(wgpu).enumerate() {
        let tolerance = 3e-4 * want.abs().max(1.0);
        assert!(
            (want - got).abs() <= tolerance,
            "case {case}: {what}[{index}] = {got} on wgpu, {want} on CPU"
        );
    }
}

#[test]
fn conv2d_backward_matches_cpu_across_representative_geometries() {
    if !wgpu_available() {
        return;
    }
    let cases = [
        Case {
            input: [2, 2, 6, 6],
            weight: [3, 2, 3, 3],
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
        },
        Case {
            input: [2, 2, 7, 7],
            weight: [3, 2, 3, 3],
            stride: (1, 1),
            padding: (1, 1),
            dilation: (1, 1),
        },
        Case {
            input: [2, 2, 8, 8],
            weight: [3, 2, 3, 3],
            stride: (2, 2),
            padding: (1, 1),
            dilation: (1, 1),
        },
        Case {
            input: [1, 2, 9, 7],
            weight: [3, 2, 3, 2],
            stride: (2, 1),
            padding: (1, 0),
            dilation: (1, 2),
        },
    ];

    for (index, case) in cases.iter().enumerate() {
        let x = values(1 + index as u64, case.input.iter().product());
        let weight = values(11 + index as u64, case.weight.iter().product());
        let probe = Tensor::from_vec(x.clone(), case.input, &Device::Cpu)
            .unwrap()
            .conv2d(
                &Tensor::from_vec(weight.clone(), case.weight, &Device::Cpu).unwrap(),
                case.stride,
                case.padding,
                case.dilation,
            )
            .unwrap();
        let scale = values(21 + index as u64, probe.dims().iter().product());
        let (cpu_input, cpu_weight) = conv_grads(&Device::Cpu, case, &x, &weight, &scale);
        let (wgpu_input, wgpu_weight) = conv_grads(&WGPU, case, &x, &weight, &scale);
        assert_close("input gradient", index, &cpu_input, &wgpu_input);
        assert_close("weight gradient", index, &cpu_weight, &wgpu_weight);
    }
}

#[test]
fn pool2d_backward_matches_cpu_across_stride_and_padding() {
    if !wgpu_available() {
        return;
    }
    let dims = [2usize, 2, 7, 7];
    let x = values(31, dims.iter().product());

    for (case, (kernel, stride, padding)) in [
        ((2, 2), (2, 2), (0, 0)),
        ((2, 2), (1, 1), (0, 0)),
        ((3, 2), (2, 1), (1, 0)),
    ]
    .into_iter()
    .enumerate()
    {
        for max_pool in [true, false] {
            let grads_on = |device: &Device| {
                let input = Param::new(Tensor::from_vec(x.clone(), dims, device).unwrap());
                let value = input.get(Mode::TRAIN);
                let pooled = if max_pool {
                    value.max_pool2d(kernel, stride, padding).unwrap()
                } else {
                    value.avg_pool2d(kernel, stride, padding).unwrap()
                };
                let cotangent = Tensor::from_vec(
                    values(41 + case as u64, pooled.dims().iter().product()),
                    pooled.dims().to_vec(),
                    device,
                )
                .unwrap();
                let grads = pooled
                    .mul(&cotangent)
                    .unwrap()
                    .sum_all()
                    .unwrap()
                    .backward()
                    .unwrap();
                grads.wrt(&input).unwrap().to_vec::<f32>().unwrap()
            };
            let what = if max_pool { "max_pool2d" } else { "avg_pool2d" };
            assert_close(what, case, &grads_on(&Device::Cpu), &grads_on(&WGPU));
        }
    }
}
