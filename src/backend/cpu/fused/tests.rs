//! The fused CPU kernels against their composed equivalents — the two must
//! agree, since the op layer picks between them.

use super::*;
use crate::storage::CpuStorage;
use std::sync::Arc;

fn storage(values: Vec<f32>) -> Storage {
    Storage::Cpu(CpuStorage::F32(Arc::new(values)))
}

fn values(storage: Storage) -> Vec<f32> {
    match storage {
        Storage::Cpu(CpuStorage::F32(values)) => Arc::unwrap_or_clone(values),
        _ => panic!("expected f32 CPU storage"),
    }
}

fn reduced_storage(dtype: DType, values: &[f32]) -> Storage {
    match dtype {
        DType::F16 => Storage::Cpu(CpuStorage::F16(Arc::new(
            values.iter().copied().map(half::f16::from_f32).collect(),
        ))),
        DType::BF16 => Storage::Cpu(CpuStorage::BF16(Arc::new(
            values.iter().copied().map(half::bf16::from_f32).collect(),
        ))),
        _ => panic!("expected reduced dtype"),
    }
}

fn reduced_values(storage: Storage) -> Vec<f32> {
    match storage {
        Storage::Cpu(CpuStorage::F16(values)) => {
            values.iter().map(|value| value.to_f32()).collect()
        }
        Storage::Cpu(CpuStorage::BF16(values)) => {
            values.iter().map(|value| value.to_f32()).collect()
        }
        _ => panic!("expected reduced CPU storage"),
    }
}

fn close(got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len());
    for (&got, &expected) in got.iter().zip(expected) {
        assert!((got - expected).abs() < 1e-6, "{got} != {expected}");
    }
}

fn one(mut outputs: Vec<Storage>) -> Storage {
    assert_eq!(outputs.len(), 1);
    outputs.pop().unwrap()
}

#[test]
fn softmax_is_stable_and_handles_fully_masked_rows() {
    let x = storage(vec![
        1000.0,
        1001.0,
        1002.0,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    ]);
    let layout = Layout::contiguous([2, 3]).unwrap();
    let composed = crate::tensor::Tensor::from_parts(x.clone(), layout.clone())
        .softmax(-1)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let got = values(one(
        fused(FusedOp::Softmax, &[View::new(&x, &layout)], &[]).unwrap()
    ));
    close(&got, &composed);
    close(&got[..3], &[0.090_030_57, 0.244_728_48, 0.665_240_94]);
    assert_eq!(&got[3..], &[0.0, 0.0, 0.0]);
}

#[test]
fn reduced_softmax_matches_an_independent_f32_uniform_reference() {
    let width = 4096;
    let layout = Layout::contiguous([1, width]).unwrap();
    for dtype in [DType::F16, DType::BF16] {
        let x = reduced_storage(dtype, &vec![1.0; width]);
        let got = reduced_values(one(
            fused(FusedOp::Softmax, &[View::new(&x, &layout)], &[]).unwrap()
        ));
        let expected = match dtype {
            DType::F16 => half::f16::from_f32(1.0 / width as f32).to_f32(),
            DType::BF16 => half::bf16::from_f32(1.0 / width as f32).to_f32(),
            _ => unreachable!(),
        };
        assert!(got.iter().all(|&value| value == expected));
    }
}

#[test]
fn softmax_reads_a_strided_last_axis_and_writes_contiguous_output() {
    let x = storage(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
    let base = Layout::contiguous([3, 2]).unwrap();
    let transposed = base.transpose(0, 1).unwrap();
    let got = values(one(fused(
        FusedOp::Softmax,
        &[View::new(&x, &transposed)],
        &[],
    )
    .unwrap()));
    close(&got[..3], &[0.090_030_57, 0.244_728_48, 0.665_240_94]);
    assert!(got[3] < 1e-8);
    assert!(got[4] < 1e-4);
    assert!(got[5] > 0.9999);
}

#[test]
fn contiguous_f32_softmax_preserves_special_value_policy() {
    let layout = Layout::contiguous([3, 3]).unwrap();
    let x = storage(vec![
        1.0,
        2.0,
        3.0,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
        4.0,
        f32::NAN,
        5.0,
    ]);
    let got = values(one(
        fused(FusedOp::Softmax, &[View::new(&x, &layout)], &[]).unwrap()
    ));

    close(&got[..3], &[0.090_030_57, 0.244_728_48, 0.665_240_94]);
    assert_eq!(&got[3..6], &[0.0, 0.0, 0.0]);
    assert!(got[6..].iter().all(|value| value.is_nan()));
}

#[test]
fn layer_norm_applies_strided_affine_views() {
    let x = storage(vec![1.0, 2.0, 3.0, 4.0, 6.0, 8.0]);
    let x_layout = Layout::contiguous([2, 3]).unwrap();
    let weight = storage(vec![2.0, 99.0, 3.0, 99.0, 4.0]);
    let bias = storage(vec![1.0, 99.0, -1.0, 99.0, 0.5]);
    let affine_layout = Layout::from_parts(
        crate::shape::Shape::from([3]),
        vec![2].into_boxed_slice(),
        0,
    )
    .unwrap();
    let got = values(one(fused(
        FusedOp::LayerNorm,
        &[
            View::new(&x, &x_layout),
            View::new(&weight, &affine_layout),
            View::new(&bias, &affine_layout),
        ],
        &[1e-5],
    )
    .unwrap()));
    close(&got[..3], &[-1.449_471_2, -1.0, 5.398_942_5]);
    close(&got[3..], &[-1.449_485_3, -1.0, 5.398_970_6]);
}

#[test]
fn layer_norm_saved_stats_match_the_composed_backward_state() {
    let x = storage(vec![0.5, -1.5, 2.0, 0.25, -0.75, 1.25]);
    let affine = storage(vec![1.0, 1.0, 1.0]);
    let bias = storage(vec![0.0, 0.0, 0.0]);
    let layout = Layout::contiguous([2, 3]).unwrap();
    let affine_layout = Layout::contiguous([3]).unwrap();
    let input = crate::tensor::Tensor::from_parts(x.clone(), layout.clone());
    let mean = input.mean_keepdim(-1).unwrap();
    let centered = input.sub(&mean).unwrap();
    let variance = centered.mul(&centered).unwrap().mean_keepdim(-1).unwrap();
    let std = variance.add_scalar(1e-5).unwrap().sqrt().unwrap();
    let expected_xhat = centered.div(&std).unwrap().to_vec::<f32>().unwrap();
    let expected_inv_std = crate::tensor::Tensor::ones([2, 1], DType::F32, &crate::Device::Cpu)
        .unwrap()
        .div(&std)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();

    let outputs = fused(
        FusedOp::LayerNorm,
        &[
            View::new(&x, &layout),
            View::new(&affine, &affine_layout),
            View::new(&bias, &affine_layout),
        ],
        &[1e-5, 1.0],
    )
    .unwrap();
    assert_eq!(outputs.len(), 3);
    assert_eq!(values(outputs[1].clone()), expected_xhat);
    assert_eq!(values(outputs[2].clone()), expected_inv_std);
}

#[test]
fn layer_norm_fused_input_gradient_is_bit_exact_to_the_composed_formula() {
    let g = storage(vec![0.3, -0.7, 1.1, 0.25, 0.9, -1.3]);
    let xhat = storage(vec![-0.2, 1.4, -0.6, 0.8, -1.1, 0.35]);
    let inv_std = storage(vec![0.75, 1.25]);
    let weight = storage(vec![1.5, -0.5, 2.0]);
    let layout = Layout::contiguous([2, 3]).unwrap();
    let stat_layout = Layout::contiguous([2, 1]).unwrap();
    let weight_layout = Layout::contiguous([3]).unwrap();
    let tensor = |storage: &Storage, layout: &Layout| {
        crate::tensor::Tensor::from_parts(storage.clone(), layout.clone())
    };
    let (gt, xt, st, wt) = (
        tensor(&g, &layout),
        tensor(&xhat, &layout),
        tensor(&inv_std, &stat_layout),
        tensor(&weight, &weight_layout),
    );
    let weighted = gt.mul(&wt).unwrap();
    let sum = weighted.sum_to(&[2, 1]).unwrap();
    let projected = weighted.mul(&xt).unwrap().sum_to(&[2, 1]).unwrap();
    let expected = weighted
        .mul_scalar(3.0)
        .unwrap()
        .sub(&sum)
        .unwrap()
        .sub(&xt.mul(&projected).unwrap())
        .unwrap()
        .mul(&st)
        .unwrap()
        .div_scalar(3.0)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();

    let got = fused(
        FusedOp::LayerNorm,
        &[
            View::new(&g, &layout),
            View::new(&xhat, &layout),
            View::new(&inv_std, &stat_layout),
            View::new(&weight, &weight_layout),
        ],
        &[],
    )
    .unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(values(got[0].clone()), expected);
}

#[test]
fn reduced_layer_norm_accumulates_mean_in_f32() {
    for dtype in [DType::F16, DType::BF16] {
        let values = [2048.0, 1.0, -2048.0, -1.0];
        let (x, weight, bias) = match dtype {
            DType::F16 => (
                Storage::Cpu(CpuStorage::F16(Arc::new(
                    values.map(half::f16::from_f32).to_vec(),
                ))),
                Storage::Cpu(CpuStorage::F16(Arc::new(vec![half::f16::ONE; 4]))),
                Storage::Cpu(CpuStorage::F16(Arc::new(vec![half::f16::ZERO; 4]))),
            ),
            DType::BF16 => (
                Storage::Cpu(CpuStorage::BF16(Arc::new(
                    values.map(half::bf16::from_f32).to_vec(),
                ))),
                Storage::Cpu(CpuStorage::BF16(Arc::new(vec![half::bf16::ONE; 4]))),
                Storage::Cpu(CpuStorage::BF16(Arc::new(vec![half::bf16::ZERO; 4]))),
            ),
            _ => unreachable!(),
        };
        let layout = Layout::contiguous([4]).unwrap();
        let result = one(fused(
            FusedOp::LayerNorm,
            &[
                View::new(&x, &layout),
                View::new(&weight, &layout),
                View::new(&bias, &layout),
            ],
            &[1e-5],
        )
        .unwrap());
        let result: Vec<f32> = match result {
            Storage::Cpu(CpuStorage::F16(v)) => v.iter().map(|x| x.to_f32()).collect(),
            Storage::Cpu(CpuStorage::BF16(v)) => v.iter().map(|x| x.to_f32()).collect(),
            _ => panic!("expected reduced CPU storage"),
        };
        assert_eq!(result[0], -result[2]);
        assert_eq!(result[1], -result[3]);
    }
}

#[test]
fn reduced_layer_norm_outputs_and_saved_stats_match_an_independent_f32_reference() {
    let input = [2048.0f32, 1.0, -2048.0, -1.0];
    let width = input.len();
    let mean = input.iter().sum::<f32>() / width as f32;
    let variance = input
        .iter()
        .map(|&value| {
            let centered = value - mean;
            centered * centered
        })
        .sum::<f32>()
        / width as f32;
    let inverse = 1.0 / (variance + 1e-5).sqrt();
    let expected_xhat: Vec<f32> = input
        .iter()
        .map(|&value| (value - mean) * inverse)
        .collect();
    let layout = Layout::contiguous([width]).unwrap();
    for dtype in [DType::F16, DType::BF16] {
        let x = reduced_storage(dtype, &input);
        let weight = reduced_storage(dtype, &vec![1.0; width]);
        let bias = reduced_storage(dtype, &vec![0.0; width]);
        let outputs = fused(
            FusedOp::LayerNorm,
            &[
                View::new(&x, &layout),
                View::new(&weight, &layout),
                View::new(&bias, &layout),
            ],
            &[1e-5, 1.0],
        )
        .unwrap();
        assert_eq!(outputs[0].dtype(), dtype);
        assert_eq!(outputs[1].dtype(), DType::F32);
        assert_eq!(outputs[2].dtype(), DType::F32);
        let y = reduced_values(outputs[0].clone());
        let xhat = values(outputs[1].clone());
        let inv_std = values(outputs[2].clone());
        for ((&got_y, &got_xhat), &reference) in y.iter().zip(&xhat).zip(&expected_xhat) {
            let expected_y = match dtype {
                DType::F16 => half::f16::from_f32(reference).to_f32(),
                DType::BF16 => half::bf16::from_f32(reference).to_f32(),
                _ => unreachable!(),
            };
            assert_eq!(got_y, expected_y);
            assert!((got_xhat - reference).abs() < 1e-6);
        }
        assert!((inv_std[0] - inverse).abs() < 1e-9);
    }
}

#[test]
fn reduced_layer_norm_backward_uses_wide_stats_at_small_eps() {
    let layout = Layout::contiguous([1, 2]).unwrap();
    let stat_layout = Layout::contiguous([1, 1]).unwrap();
    let weight_layout = Layout::contiguous([2]).unwrap();
    for dtype in [DType::F16, DType::BF16] {
        let x = reduced_storage(dtype, &[1.0, 1.0]);
        let weight = reduced_storage(dtype, &[1.0, 1.0]);
        let bias = reduced_storage(dtype, &[0.0, 0.0]);
        let outputs = fused(
            FusedOp::LayerNorm,
            &[
                View::new(&x, &layout),
                View::new(&weight, &weight_layout),
                View::new(&bias, &weight_layout),
            ],
            &[1e-12, 1.0],
        )
        .unwrap();
        assert_eq!(outputs[1].dtype(), DType::F32);
        assert_eq!(outputs[2].dtype(), DType::F32);
        assert_eq!(values(outputs[1].clone()), vec![0.0, 0.0]);
        let inverse = values(outputs[2].clone())[0];
        assert_eq!(inverse, 1_000_000.0);

        let g = reduced_storage(dtype, &[0.0, 0.001]);
        let backward = fused(
            FusedOp::LayerNorm,
            &[
                View::new(&g, &layout),
                View::new(&outputs[1], &layout),
                View::new(&outputs[2], &stat_layout),
                View::new(&weight, &weight_layout),
            ],
            &[],
        )
        .unwrap();
        let got = reduced_values(backward[0].clone());
        let quantized_g = match dtype {
            DType::F16 => half::f16::from_f32(0.001).to_f32(),
            DType::BF16 => half::bf16::from_f32(0.001).to_f32(),
            _ => unreachable!(),
        };
        let expected = [-0.5 * quantized_g * inverse, 0.5 * quantized_g * inverse];
        for (&got, expected) in got.iter().zip(expected) {
            let expected = match dtype {
                DType::F16 => half::f16::from_f32(expected).to_f32(),
                DType::BF16 => half::bf16::from_f32(expected).to_f32(),
                _ => unreachable!(),
            };
            assert_eq!(got, expected);
        }

        let old_xhat = reduced_storage(dtype, &[0.0, 0.0]);
        let old_inv_std = reduced_storage(dtype, &[1.0]);
        assert!(matches!(
            fused(
                FusedOp::LayerNorm,
                &[
                    View::new(&g, &layout),
                    View::new(&old_xhat, &layout),
                    View::new(&old_inv_std, &stat_layout),
                    View::new(&weight, &weight_layout),
                ],
                &[],
            ),
            Err(Error::DTypeMismatch {
                expected: DType::F32,
                ..
            })
        ));
    }
}

#[test]
fn sgd_matches_scalar_reference_for_strided_coupled_decay() {
    let param_values = Arc::new(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
    let grad_values = Arc::new(vec![0.5, 5.0, -1.0, 6.0, 2.0, 7.0]);
    let param = Storage::Cpu(CpuStorage::F32(param_values.clone()));
    let grad = Storage::Cpu(CpuStorage::F32(grad_values.clone()));
    let layout = Layout::contiguous([3, 2]).unwrap().transpose(0, 1).unwrap();
    let got = values(
        fused(
            FusedOp::SgdStep,
            &[View::new(&param, &layout), View::new(&grad, &layout)],
            &[0.1, 0.0, 0.2],
        )
        .unwrap()
        .remove(0),
    );
    let logical_param = [1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
    let logical_grad = [0.5, -1.0, 2.0, 5.0, 6.0, 7.0];
    let expected: Vec<_> = logical_param
        .iter()
        .zip(logical_grad)
        .map(|(&p, g)| p - (g + p * 0.2) * 0.1)
        .collect();
    close(&got, &expected);
    assert_eq!(param_values.as_slice(), &[1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
    assert_eq!(grad_values.as_slice(), &[0.5, 5.0, -1.0, 6.0, 2.0, 7.0]);
}

#[test]
fn sgd_first_and_later_momentum_steps_preserve_wide_half_velocity() {
    let param = Storage::Cpu(CpuStorage::F16(Arc::new(vec![half::f16::from_f32(2048.0)])));
    let grad = Storage::Cpu(CpuStorage::F16(Arc::new(vec![half::f16::from_f32(1.0)])));
    let layout = Layout::contiguous([1]).unwrap();
    let first = fused(
        FusedOp::SgdStep,
        &[View::new(&param, &layout), View::new(&grad, &layout)],
        &[0.0, 0.999, 0.0],
    )
    .unwrap();
    assert_eq!(first.len(), 2);
    let Storage::Cpu(CpuStorage::F32(first_velocity)) = &first[1] else {
        panic!("expected wide f32 velocity")
    };
    assert_eq!(first_velocity.as_slice(), &[1.0]);

    let velocity = Storage::Cpu(CpuStorage::F32(Arc::new(vec![2048.0])));
    let later = fused(
        FusedOp::SgdStep,
        &[
            View::new(&param, &layout),
            View::new(&grad, &layout),
            View::new(&velocity, &layout),
        ],
        &[0.0, 0.999, 0.0],
    )
    .unwrap();
    let Storage::Cpu(CpuStorage::F32(next_velocity)) = &later[1] else {
        panic!("expected wide f32 velocity")
    };
    let expected = 2048.0f32 * 0.999 + 1.0;
    assert_eq!(next_velocity[0], expected);
    assert_ne!(next_velocity[0], half::f16::from_f32(expected).to_f32());
}

#[test]
fn adam_and_adamw_match_scalar_reference_with_strided_state() {
    let param = storage(vec![1.0, 10.0, -2.0, 20.0]);
    let grad = storage(vec![0.5, 5.0, -0.25, 6.0]);
    let m = storage(vec![0.1, 1.0, -0.2, 2.0]);
    let v = storage(vec![0.3, 3.0, 0.4, 4.0]);
    let layout = Layout::contiguous([2, 2]).unwrap().transpose(0, 1).unwrap();
    let lr = 0.01f32;
    let beta1 = 0.9f32;
    let beta2 = 0.99f32;
    let eps = 1e-6f32;
    let decay = 0.1f32;
    let correction1 = 0.19f32;
    let correction2 = 0.0199f32;
    for decoupled in [false, true] {
        let outputs = fused(
            FusedOp::AdamStep,
            &[
                View::new(&param, &layout),
                View::new(&grad, &layout),
                View::new(&m, &layout),
                View::new(&v, &layout),
            ],
            &[
                f64::from(lr),
                f64::from(beta1),
                f64::from(beta2),
                f64::from(eps),
                f64::from(decay),
                f64::from(correction1),
                f64::from(correction2),
                f64::from(u8::from(decoupled)),
            ],
        )
        .unwrap();
        let got_p = values(outputs[0].clone());
        let got_m = values(outputs[1].clone());
        let got_v = values(outputs[2].clone());
        let ps = [1.0f32, -2.0, 10.0, 20.0];
        let gs = [0.5f32, -0.25, 5.0, 6.0];
        let ms = [0.1f32, -0.2, 1.0, 2.0];
        let vs = [0.3f32, 0.4, 3.0, 4.0];
        for i in 0..ps.len() {
            let g = if decoupled {
                gs[i]
            } else {
                gs[i] + ps[i] * decay
            };
            let next_m = ms[i] * beta1 + g * (1.0 - beta1);
            let next_v = vs[i] * beta2 + (g * g) * (1.0 - beta2);
            let direction = (next_m / correction1) / ((next_v / correction2).sqrt() + eps);
            let mut next_p = ps[i];
            if decoupled {
                next_p *= 1.0 - lr * decay;
            }
            next_p -= direction * lr;
            assert_eq!(got_m[i], next_m);
            assert_eq!(got_v[i], next_v);
            assert_eq!(got_p[i], next_p);
        }
    }
}

#[test]
fn optimizer_kernels_support_bf16_and_f64_parameters() {
    let layout = Layout::contiguous([2]).unwrap();
    let bf16 = |values: &[f32]| {
        Storage::Cpu(CpuStorage::BF16(Arc::new(
            values.iter().copied().map(half::bf16::from_f32).collect(),
        )))
    };
    let p = bf16(&[1.0, -2.0]);
    let g = bf16(&[0.5, -0.25]);
    let bf16_out = fused(
        FusedOp::SgdStep,
        &[View::new(&p, &layout), View::new(&g, &layout)],
        &[0.1, 0.0, 0.0],
    )
    .unwrap();
    assert!(matches!(bf16_out[0], Storage::Cpu(CpuStorage::BF16(_))));

    let f64_storage = |values: Vec<f64>| Storage::Cpu(CpuStorage::F64(Arc::new(values)));
    let p = f64_storage(vec![1.0, -2.0]);
    let g = f64_storage(vec![0.5, -0.25]);
    let m = f64_storage(vec![0.0, 0.0]);
    let v = f64_storage(vec![0.0, 0.0]);
    let f64_out = fused(
        FusedOp::AdamStep,
        &[
            View::new(&p, &layout),
            View::new(&g, &layout),
            View::new(&m, &layout),
            View::new(&v, &layout),
        ],
        &[0.1, 0.9, 0.999, 1e-8, 0.0, 0.1, 0.001, 0.0],
    )
    .unwrap();
    assert!(matches!(f64_out[0], Storage::Cpu(CpuStorage::F64(_))));
}

#[test]
fn optimizer_invalid_encodings_hyperparameters_and_metadata_are_structured() {
    let x = storage(vec![1.0, 2.0]);
    let layout = Layout::contiguous([2]).unwrap();
    let view = View::new(&x, &layout);
    assert!(matches!(
        fused(FusedOp::Softmax, &[view], &[1.0]),
        Err(Error::InvalidArg {
            op: "fused_softmax",
            ..
        })
    ));
    assert!(matches!(
        fused(FusedOp::SgdStep, &[view], &[]),
        Err(Error::InvalidArg {
            op: "fused_sgd_step",
            ..
        })
    ));
    assert!(matches!(
        fused(FusedOp::AdamStep, &[view], &[]),
        Err(Error::InvalidArg {
            op: "fused_adam_step",
            ..
        })
    ));

    let wrong_shape = Layout::contiguous([1, 2]).unwrap();
    assert!(matches!(
        fused(
            FusedOp::SgdStep,
            &[view, View::new(&x, &wrong_shape)],
            &[0.1, 0.0, 0.0]
        ),
        Err(Error::ShapeMismatch { .. })
    ));
    let f64_grad = Storage::Cpu(CpuStorage::F64(Arc::new(vec![1.0, 2.0])));
    assert!(matches!(
        fused(
            FusedOp::SgdStep,
            &[view, View::new(&f64_grad, &layout)],
            &[0.1, 0.0, 0.0]
        ),
        Err(Error::DTypeMismatch { .. })
    ));
    assert!(matches!(
        fused(FusedOp::SgdStep, &[view, view], &[f64::NAN, 0.0, 0.0]),
        Err(Error::InvalidArg { .. })
    ));
    assert!(matches!(
        fused(FusedOp::SgdStep, &[view, view], &[0.1, 1.0, 0.0]),
        Err(Error::InvalidArg { .. })
    ));
    let half = Storage::Cpu(CpuStorage::F16(Arc::new(vec![half::f16::ZERO; 2])));
    let velocity = storage(vec![0.0, 0.0]);
    assert!(matches!(
        fused(
            FusedOp::SgdStep,
            &[
                View::new(&half, &layout),
                View::new(&half, &layout),
                View::new(&velocity, &layout),
            ],
            &[0.1, f64::MIN_POSITIVE, 0.0]
        ),
        Err(Error::InvalidArg { .. })
    ));
    let zero = storage(vec![0.0, 0.0]);
    assert!(matches!(
        fused(
            FusedOp::AdamStep,
            &[
                view,
                view,
                View::new(&zero, &layout),
                View::new(&zero, &layout)
            ],
            &[0.1, 0.9, 0.999, 0.0, 0.0, 0.1, 0.001, 2.0]
        ),
        Err(Error::InvalidArg { .. })
    ));

    assert!(matches!(
        fused(
            FusedOp::AdamStep,
            &[
                View::new(&half, &layout),
                View::new(&half, &layout),
                View::new(&half, &layout),
                View::new(&half, &layout),
            ],
            &[0.1, 0.9, 0.999, 1e-8, 0.0, 0.1, 0.001, 0.0]
        ),
        Err(Error::DTypeMismatch {
            expected: DType::F32,
            got: DType::F16,
            ..
        })
    ));
}

#[test]
fn non_float_and_out_of_bounds_views_are_rejected() {
    let integers = Storage::Cpu(CpuStorage::I64(Arc::new(vec![1, 2])));
    let layout = Layout::contiguous([2]).unwrap();
    assert!(matches!(
        fused(FusedOp::Softmax, &[View::new(&integers, &layout)], &[]),
        Err(Error::Unsupported {
            dtype: DType::I64,
            ..
        })
    ));

    let x = storage(vec![1.0]);
    assert!(matches!(
        fused(FusedOp::Softmax, &[View::new(&x, &layout)], &[]),
        Err(Error::InvalidArg {
            op: "fused_softmax",
            ..
        })
    ));
}
