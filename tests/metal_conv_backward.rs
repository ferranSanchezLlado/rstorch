//! Metal conv/pool **backward** against the CPU reference, across the
//! stride/padding/dilation grid.
//!
//! The in-crate Metal test (`conv_and_pool_backward_execute_on_metal`) pins
//! hand-computed values for the identity geometry only: stride 1, no padding,
//! no dilation. Those are precisely the parameters that make the gradient's
//! index arithmetic trivial, so they cannot catch an input-gradient kernel that
//! maps output positions back to input positions incorrectly under a stride or
//! a dilation — a defect that produces a *plausible* gradient and a model that
//! merely trains badly.
//!
//! CPU is the reference implementation (see `backend::conformance`), so every
//! case here runs the identical graph on both devices from identical host bytes
//! and diffs the resulting gradients.

#![cfg(all(feature = "metal", target_os = "macos"))]

use rstorch::prelude::*;

const METAL: Device = Device::Metal(0);

/// One geometry: input dims, weight dims, stride, padding, dilation.
struct Case {
    input: [usize; 4],
    weight: [usize; 4],
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
}

/// Deterministic host values, so both devices start from the same bytes.
fn values(seed: u64, len: usize) -> Vec<f32> {
    let mut rng = Rng::seed(seed);
    (0..len).map(|_| rng.uniform(-1.0, 1.0) as f32).collect()
}

/// `(input gradient, weight gradient)` of `sum(conv2d(x, w) * scale)` on
/// `device`, where `scale` makes the cotangent non-uniform: a uniform one
/// hides errors that permute output positions.
fn conv_grads(
    device: &Device,
    case: &Case,
    x: &[f32],
    w: &[f32],
    scale: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let input = Param::new(Tensor::from_vec(x.to_vec(), case.input, device).unwrap());
    let weight = Param::new(Tensor::from_vec(w.to_vec(), case.weight, device).unwrap());
    let out = input
        .get(Mode::TRAIN)
        .conv2d(
            &weight.get(Mode::TRAIN),
            case.stride,
            case.padding,
            case.dilation,
        )
        .unwrap();
    let cotangent = Tensor::from_vec(scale.to_vec(), out.dims().to_vec(), device).unwrap();
    let grads = out
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

fn assert_close(what: &str, case: usize, cpu: &[f32], metal: &[f32]) {
    assert_eq!(cpu.len(), metal.len(), "case {case}: {what} length differs");
    for (i, (want, got)) in cpu.iter().zip(metal).enumerate() {
        let tolerance = 1e-4 * want.abs().max(1.0);
        assert!(
            (want - got).abs() <= tolerance,
            "case {case}: {what}[{i}] = {got} on Metal, {want} on the CPU reference"
        );
    }
}

#[test]
fn conv2d_backward_matches_the_cpu_across_stride_padding_and_dilation() {
    let cases = [
        // The identity geometry the in-crate test already pins, as a control.
        Case {
            input: [2, 3, 7, 7],
            weight: [4, 3, 3, 3],
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
        },
        // Padding alone: every border input position feeds fewer windows.
        Case {
            input: [2, 3, 7, 7],
            weight: [4, 3, 3, 3],
            stride: (1, 1),
            padding: (1, 1),
            dilation: (1, 1),
        },
        // Stride alone: most input positions feed *no* window for a given tap,
        // which is the divisibility condition of the inverse mapping.
        Case {
            input: [2, 3, 7, 7],
            weight: [4, 3, 3, 3],
            stride: (2, 2),
            padding: (0, 0),
            dilation: (1, 1),
        },
        // Stride and padding together.
        Case {
            input: [2, 3, 8, 8],
            weight: [4, 3, 3, 3],
            stride: (2, 2),
            padding: (1, 1),
            dilation: (1, 1),
        },
        // Dilation, and asymmetric everything: the case where a transposed
        // index expression still happens to produce the right shape.
        Case {
            input: [2, 2, 9, 7],
            weight: [3, 2, 3, 2],
            stride: (2, 1),
            padding: (1, 0),
            dilation: (1, 2),
        },
        Case {
            input: [1, 2, 6, 6],
            weight: [2, 2, 2, 2],
            stride: (1, 2),
            padding: (0, 1),
            dilation: (2, 2),
        },
    ];

    for (index, case) in cases.iter().enumerate() {
        let x = values(1, case.input.iter().product());
        let w = values(2, case.weight.iter().product());
        // The output shape is whatever the geometry says; ask the CPU first.
        let probe = Tensor::from_vec(x.clone(), case.input, &Device::Cpu)
            .unwrap()
            .conv2d(
                &Tensor::from_vec(w.clone(), case.weight, &Device::Cpu).unwrap(),
                case.stride,
                case.padding,
                case.dilation,
            )
            .unwrap();
        let scale = values(3, probe.dims().iter().product());

        let (cpu_input, cpu_weight) = conv_grads(&Device::Cpu, case, &x, &w, &scale);
        let (metal_input, metal_weight) = conv_grads(&METAL, case, &x, &w, &scale);
        assert_close("input grad", index, &cpu_input, &metal_input);
        assert_close("weight grad", index, &cpu_weight, &metal_weight);
    }
}

#[test]
fn pool2d_backward_matches_the_cpu_across_stride_and_padding() {
    let dims = [2usize, 3, 7, 7];
    let x = values(4, dims.iter().product());

    for (kernel, stride, padding) in [
        ((2, 2), (2, 2), (0, 0)),
        ((2, 2), (1, 1), (0, 0)),
        ((3, 3), (2, 2), (1, 1)),
        ((3, 2), (1, 2), (1, 0)),
    ] {
        for max_pool in [true, false] {
            let grads_on = |device: &Device| -> Vec<f32> {
                let input = Param::new(Tensor::from_vec(x.clone(), dims, device).unwrap());
                let value = input.get(Mode::TRAIN);
                let pooled = if max_pool {
                    value.max_pool2d(kernel, stride, padding).unwrap()
                } else {
                    value.avg_pool2d(kernel, stride, padding).unwrap()
                };
                let scale = values(5, pooled.dims().iter().product());
                let cotangent = Tensor::from_vec(scale, pooled.dims().to_vec(), device).unwrap();
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
            assert_close(what, 0, &grads_on(&Device::Cpu), &grads_on(&METAL));
        }
    }
}
