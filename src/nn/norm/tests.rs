//! The three normalization layers against independently computed reference
//! statistics, plus gradcheck, mode behaviour and the running-buffer rules.

use super::*;
use crate::nn::{self, Module, Sequential};
use crate::testing::check_grad;
use std::collections::BTreeMap;

const CPU: Device = Device::Cpu;

fn t(data: &[f32], shape: impl Into<Shape>) -> Tensor {
    Tensor::from_vec(data.to_vec(), shape, &CPU).unwrap()
}

fn v(x: &Tensor) -> Vec<f32> {
    x.to_vec::<f32>().unwrap()
}

fn t64(data: &[f64], shape: impl Into<Shape>) -> Tensor {
    Tensor::from_vec(data.to_vec(), shape, &CPU).unwrap()
}

/// Assert element-wise closeness against hand-computed values.
#[track_caller]
fn close(got: &Tensor, want: &[f32], tol: f32) {
    close_all(&v(got), want, tol);
}

#[track_caller]
fn close_all(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "length: {got:?} vs {want:?}");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "element {i}: got {g}, want {w} (all: {got:?} vs {want:?})"
        );
    }
}

/// Install explicit values into a module's leaves by path — the sanctioned
/// way to give a norm layer non-default parameters (`load_state_dict`).
fn load(module: &mut dyn Module, values: &[(&str, Tensor)]) {
    let state: BTreeMap<String, Tensor> = values
        .iter()
        .map(|(k, t)| ((*k).to_string(), t.clone()))
        .collect();
    nn::load_state_dict(module, &state).unwrap();
}

// -- LayerNorm forward numerics -------------------------------------

#[test]
fn layer_norm_matches_hand_computed_values() {
    // Row [1,2,3,4]: mean 2.5, Σ(x−x̄)² = 5.
    // correction = 0 -> var 1.25, √1.25 = 1.1180340, out ±1.5/1.118, ±0.5/1.118.
    // (With correction = 1 the divisor would be √(5/3) = 1.2909944 and the
    // outer values −1.161895/1.161895 — the check below excludes that.)
    let mut norm = LayerNorm::new([4], &CPU).unwrap();
    let y = norm
        .forward(&t(&[1.0, 2.0, 3.0, 4.0], [1, 4]), Mode::EVAL)
        .unwrap();
    close(
        &y,
        &[-1.341_640_8, -0.447_213_6, 0.447_213_6, 1.341_640_8],
        1e-4,
    );
}

#[test]
fn layer_norm_normalizes_each_batch_row_independently() {
    // Second row is the first scaled by 10 and shifted by 100: the same
    // whitened output, which is the property the layer exists for.
    let mut norm = LayerNorm::new([4], &CPU).unwrap();
    let x = t(&[1.0, 2.0, 3.0, 4.0, 110.0, 120.0, 130.0, 140.0], [2, 4]);
    let got = v(&norm.forward(&x, Mode::EVAL).unwrap());
    for i in 0..4 {
        assert!((got[i] - got[4 + i]).abs() < 1e-3, "row {i}: {got:?}");
    }
}

#[test]
fn layer_norm_applies_weight_and_bias() {
    let mut norm = LayerNorm::with_eps([4], 0.0625, &CPU).unwrap();
    load(
        &mut norm,
        &[
            ("weight", t(&[2.0, 2.0, 2.0, 2.0], [4])),
            ("bias", t(&[1.0, 1.0, 1.0, 1.0], [4])),
        ],
    );
    // var 1.25 + eps 0.0625 = 1.3125? No: eps chosen so √(1.25+0.0625)
    // stays irrational — assert against the formula's own constants.
    let scale = (1.25f32 + 0.0625).sqrt();
    let y = norm
        .forward(&t(&[1.0, 2.0, 3.0, 4.0], [4]), Mode::EVAL)
        .unwrap();
    close(
        &y,
        &[
            2.0 * -1.5 / scale + 1.0,
            2.0 * -0.5 / scale + 1.0,
            2.0 * 0.5 / scale + 1.0,
            2.0 * 1.5 / scale + 1.0,
        ],
        1e-5,
    );
    assert_eq!(norm.eps(), 0.0625);
}

#[test]
fn layer_norm_over_several_trailing_axes() {
    // normalized_shape [2,2] normalizes all four elements of each sample.
    let mut norm = LayerNorm::new([2, 2], &CPU).unwrap();
    assert_eq!(norm.normalized_shape().dims(), &[2, 2]);
    let x = t(&[1.0, 2.0, 3.0, 4.0, 40.0, 30.0, 20.0, 10.0], [2, 2, 2]);
    let y = norm.forward(&x, Mode::EVAL).unwrap();
    assert_eq!(y.dims(), &[2, 2, 2]);
    close(
        &y,
        &[
            -1.341_640_8,
            -0.447_213_6,
            0.447_213_6,
            1.341_640_8,
            1.341_640_8,
            0.447_213_6,
            -0.447_213_6,
            -1.341_640_8,
        ],
        1e-3,
    );
}

#[test]
fn layer_norm_mode_does_not_change_the_numbers() {
    let mut norm = LayerNorm::new([4], &CPU).unwrap();
    let x = t(&[1.0, 2.0, 3.0, 4.0], [4]);
    let train = v(&norm.forward(&x, Mode::TRAIN).unwrap());
    let eval = v(&norm.forward(&x, Mode::EVAL).unwrap());
    assert_eq!(train, eval);
    // Only recording differs.
    assert!(norm.forward(&x, Mode::TRAIN).unwrap().backward().is_ok());
    assert!(matches!(
        norm.forward(&x, Mode::EVAL).unwrap().backward(),
        Err(Error::NotTraced { .. })
    ));
}

#[test]
fn layer_norm_rejects_a_bad_trailing_shape() {
    let mut norm = LayerNorm::new([4], &CPU).unwrap();
    for bad in [t(&[1.0, 2.0, 3.0], [3]), t(&[1.0, 2.0, 3.0, 4.0], [4, 1])] {
        assert!(matches!(
            norm.forward(&bad, Mode::EVAL),
            Err(Error::ShapeMismatch {
                op: "LayerNorm::forward",
                ..
            })
        ));
    }
    // A rank *lower* than the normalized shape is the same error.
    let mut deep = LayerNorm::new([2, 2], &CPU).unwrap();
    assert!(matches!(
        deep.forward(&t(&[1.0, 2.0], [2]), Mode::EVAL),
        Err(Error::ShapeMismatch { .. })
    ));
}

#[test]
fn norm_constructors_validate_their_arguments() {
    assert!(matches!(
        LayerNorm::new(Vec::<usize>::new(), &CPU),
        Err(Error::InvalidArg {
            op: "LayerNorm::new",
            ..
        })
    ));
    assert!(matches!(
        LayerNorm::new([0], &CPU),
        Err(Error::InvalidArg { .. })
    ));
    assert!(matches!(
        LayerNorm::with_eps([2], 0.0, &CPU),
        Err(Error::InvalidArg { .. })
    ));
    assert!(matches!(
        RMSNorm::with_eps([2], f64::NAN, &CPU),
        Err(Error::InvalidArg {
            op: "RMSNorm::new",
            ..
        })
    ));
    assert!(matches!(
        BatchNorm2d::new(0, &CPU),
        Err(Error::InvalidArg {
            op: "BatchNorm2d::new",
            ..
        })
    ));
    assert!(matches!(
        BatchNorm2d::with_params(2, 1e-5, 1.5, &CPU),
        Err(Error::InvalidArg { .. })
    ));
    assert!(BatchNorm2d::with_params(2, 1e-5, 0.0, &CPU).is_ok());
}

// -- RMSNorm forward numerics ---------------------------------------

#[test]
fn rms_norm_matches_hand_computed_values() {
    // mean(x²) for [1,2,3,4] is 30/4 = 7.5; √7.5 = 2.7386128.
    let mut norm = RMSNorm::new([4], &CPU).unwrap();
    assert_eq!(norm.eps(), RMSNorm::DEFAULT_EPS);
    let y = norm
        .forward(&t(&[1.0, 2.0, 3.0, 4.0], [4]), Mode::EVAL)
        .unwrap();
    close(
        &y,
        &[0.365_148_4, 0.730_296_8, 1.095_445_1, 1.460_593_5],
        1e-5,
    );
}

#[test]
fn rms_norm_does_not_center_and_has_no_bias() {
    // A constant row has zero variance: LayerNorm maps it to zeros, while
    // RMSNorm maps it to ±1 — the distinguishing property.
    let x = t(&[3.0, 3.0, 3.0, 3.0], [4]);
    let mut ln = LayerNorm::new([4], &CPU).unwrap();
    let mut rms = RMSNorm::new([4], &CPU).unwrap();
    close(&ln.forward(&x, Mode::EVAL).unwrap(), &[0.0; 4], 1e-6);
    close(&rms.forward(&x, Mode::EVAL).unwrap(), &[1.0; 4], 1e-6);
    assert_eq!(
        nn::state_dict(&rms).keys().collect::<Vec<_>>(),
        ["weight"],
        "RMSNorm has no bias"
    );
}

#[test]
fn rms_norm_applies_weight() {
    let mut norm = RMSNorm::new([4], &CPU).unwrap();
    load(&mut norm, &[("weight", t(&[1.0, 2.0, 3.0, 4.0], [4]))]);
    let y = norm
        .forward(&t(&[1.0, 2.0, 3.0, 4.0], [4]), Mode::EVAL)
        .unwrap();
    close(
        &y,
        &[0.365_148_4, 1.460_593_5, 3.286_335_3, 5.842_374],
        1e-5,
    );
}

#[test]
fn external_state_helpers_are_exactly_the_dynamic_paths() {
    let x = t(&[0.5, -1.5, 2.0, 0.25, -0.75, 1.25], [2, 3]);
    let weight = t(&[1.5, -0.5, 2.0], [3]);
    let bias = t(&[0.25, -0.5, 0.75], [3]);

    let mut ln = LayerNorm::new([3], &CPU).unwrap();
    load(
        &mut ln,
        &[("weight", weight.clone()), ("bias", bias.clone())],
    );
    let external = layer_norm_forward(&x, &weight, Some(&bias), LayerNorm::DEFAULT_EPS).unwrap();
    assert_eq!(v(&ln.forward(&x, Mode::EVAL).unwrap()), v(&external));

    let mut rms = RMSNorm::new([3], &CPU).unwrap();
    load(&mut rms, &[("weight", weight.clone())]);
    let external = rms_norm_forward(&x, &weight, RMSNorm::DEFAULT_EPS).unwrap();
    assert_eq!(v(&rms.forward(&x, Mode::EVAL).unwrap()), v(&external));

    let bn_x = bn_input();
    let bn_weight = t(&[1.5, -0.5], [2]);
    let bn_bias = t(&[0.25, 0.75], [2]);
    let mean = t(&[0.25, -0.5], [2]);
    let variance = t(&[1.5, 0.75], [2]);
    let mut bn = BatchNorm2d::new(2, &CPU).unwrap();
    load(
        &mut bn,
        &[
            ("weight", bn_weight.clone()),
            ("bias", bn_bias.clone()),
            ("running_mean", mean.clone()),
            ("running_var", variance.clone()),
        ],
    );
    let (external, replacements) = batch_norm2d_forward(
        &bn_x,
        &bn_weight,
        &bn_bias,
        &mean,
        &variance,
        bn.eps(),
        bn.momentum(),
        Mode::TRAIN.frozen(),
    )
    .unwrap();
    let dynamic = bn.forward(&bn_x, Mode::TRAIN.frozen()).unwrap();
    assert_eq!(v(&dynamic), v(&external));
    let (next_mean, next_variance) = replacements.unwrap();
    assert_eq!(v(bn.running_mean()), v(&next_mean));
    assert_eq!(v(bn.running_var()), v(&next_variance));
    for replacement in [&next_mean, &next_variance] {
        assert!(matches!(
            replacement.backward(),
            Err(Error::NotTraced { .. })
        ));
    }

    let before_mean = v(bn.running_mean());
    let before_variance = v(bn.running_var());
    let (external, replacements) = batch_norm2d_forward(
        &bn_x,
        &bn_weight,
        &bn_bias,
        bn.running_mean(),
        bn.running_var(),
        bn.eps(),
        bn.momentum(),
        Mode::EVAL,
    )
    .unwrap();
    assert!(replacements.is_none());
    assert_eq!(v(&bn.forward(&bn_x, Mode::EVAL).unwrap()), v(&external));
    assert_eq!(v(bn.running_mean()), before_mean);
    assert_eq!(v(bn.running_var()), before_variance);
}

#[test]
fn external_state_helpers_preserve_caller_gradient_identities() {
    let x = t(&[0.5, -1.5, 2.0, 0.25, -0.75, 1.25], [2, 3]);
    let weight = t(&[1.5, -0.5, 2.0], [3]).traced().unwrap();
    let bias = t(&[0.25, -0.5, 0.75], [3]).traced().unwrap();
    let grads = layer_norm_forward(&x, &weight, Some(&bias), LayerNorm::DEFAULT_EPS)
        .unwrap()
        .mul(&coef(&[2, 3]))
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    assert_eq!(grads.wrt_input(&weight).unwrap().dims(), &[3]);
    assert_eq!(grads.wrt_input(&bias).unwrap().dims(), &[3]);

    let rms_weight = t(&[1.5, -0.5, 2.0], [3]).traced().unwrap();
    let grads = rms_norm_forward(&x, &rms_weight, RMSNorm::DEFAULT_EPS)
        .unwrap()
        .mul(&coef(&[2, 3]))
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    assert_eq!(grads.wrt_input(&rms_weight).unwrap().dims(), &[3]);

    let bn_weight = t(&[1.5, -0.5], [2]).traced().unwrap();
    let bn_bias = t(&[0.25, 0.75], [2]).traced().unwrap();
    let (out, replacements) = batch_norm2d_forward(
        &bn_input(),
        &bn_weight,
        &bn_bias,
        &t(&[0.0, 0.0], [2]),
        &t(&[1.0, 1.0], [2]),
        BatchNorm2d::DEFAULT_EPS,
        BatchNorm2d::DEFAULT_MOMENTUM,
        Mode::TRAIN,
    )
    .unwrap();
    let grads = out
        .mul(&coef(&[2, 2, 1, 3]))
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    assert_eq!(grads.wrt_input(&bn_weight).unwrap().dims(), &[2]);
    assert_eq!(grads.wrt_input(&bn_bias).unwrap().dims(), &[2]);
    let (next_mean, next_variance) = replacements.unwrap();
    assert!(next_mean.node().is_none());
    assert!(next_variance.node().is_none());
}

// -- an independent reference implementation ---------------------------
//
// The tests above pick inputs whose statistics are exact in decimal, which
// keeps them readable but also keeps them small. These three replay a
// 24-element input through a from-the-definition reference implementation
// written separately (plain scalar loops, f64, no tensor library) and
// compare element for element. A reduction taken over the wrong axis, or a
// statistic broadcast one axis off, survives a symmetric toy input far more
// easily than it survives this.

/// The shared input: 24 values with no symmetry, no repeats, and no
/// structure a mistaken axis could hide behind.
const REF_X: [f32; 24] = [
    0.5, -1.5, 2.0, 0.25, -0.75, 1.25, 0.75, -0.25, 1.75, -1.25, 0.125, 2.5, -2.0, 0.375, 1.5,
    -0.5, 3.0, 0.625, -1.75, 0.875, 2.25, -0.125, 1.0, -2.5,
];

#[test]
fn layer_norm_matches_an_independent_reference() {
    // [2, 3, 4] with normalized_shape [4]: six independent blocks, and the
    // two leading axes must both be treated as batch axes.
    let mut norm = LayerNorm::new([4], &CPU).unwrap();
    load(
        &mut norm,
        &[
            ("weight", t(&[1.5, -0.5, 2.0, 0.25], [4])),
            ("bias", t(&[0.25, -0.5, 0.75, -1.0], [4])),
        ],
    );
    let y = norm.forward(&t(&REF_X, [2, 3, 4]), Mode::EVAL).unwrap();
    assert_eq!(y.dims(), &[2, 3, 4]);
    close(
        &y,
        &[
            0.476_418_8,
            0.229_571_8,
            3.467_026,
            -1.012_578_8,
            -1.647_351_4,
            -1.132_450_5,
            2.014_901,
            -1.158_112_6,
            1.249_824_2,
            0.198_801_9,
            -0.153_067_1,
            -0.704_353,
            -1.912_392_6,
            -0.707_687_4,
            3.339_984_3,
            -1.067_193,
            2.311_214_6,
            -0.481_430_5,
            -2.146_842_2,
            -0.972_145_7,
            2.046_072_3,
            -0.419_578_9,
            1.715_053_8,
            -1.379_766_5,
        ],
        1e-5,
    );
}

#[test]
fn fused_layer_norm_handles_f64_and_constant_rows() {
    let mut norm = LayerNorm::with_eps([3], 1e-4, &CPU).unwrap();
    norm.weight.set(t64(&[1.5, -0.5, 2.0], [3])).unwrap();
    norm.bias.set(t64(&[0.25, -0.75, 1.0], [3])).unwrap();
    let x = t64(&[2.0, 2.0, 2.0, 0.5, -1.5, 2.0], [2, 3]);
    let got = norm.forward(&x, Mode::EVAL).unwrap();
    let want = composed_layer_norm(
        &x,
        &t64(&[1.5, -0.5, 2.0], [3]),
        Some(&t64(&[0.25, -0.75, 1.0], [3])),
        1e-4,
    )
    .unwrap();
    let got = got.to_vec::<f64>().unwrap();
    let want = want.to_vec::<f64>().unwrap();
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-12, "got {got:?}, want {want:?}");
    }
    assert!(got.iter().all(|value| value.is_finite()));
}

#[test]
fn multi_axis_layer_norm_explicitly_uses_the_composed_fallback() {
    let mut norm = LayerNorm::with_eps([2, 2], 1e-4, &CPU).unwrap();
    let x = t(&[0.5, -1.5, 2.0, 0.25, -0.75, 1.25, 0.75, -0.25], [2, 2, 2]);
    let weight = t(&[1.5, -0.5, 2.0, 0.5], [2, 2]);
    let bias = t(&[0.25, -0.5, 0.75, 0.0], [2, 2]);
    load(
        &mut norm,
        &[("weight", weight.clone()), ("bias", bias.clone())],
    );
    let got = norm.forward(&x, Mode::EVAL).unwrap();
    let composed = composed_layer_norm(&x, &weight, Some(&bias), 1e-4).unwrap();
    assert_eq!(v(&got), v(&composed));
}

#[test]
fn reduced_precision_layer_norm_uses_the_fused_wide_path() {
    for dtype in [DType::F16, DType::BF16] {
        let make = |values: &[f32], dims: &[usize]| {
            match dtype {
                DType::F16 => Tensor::from_vec(
                    values.iter().copied().map(half::f16::from_f32).collect(),
                    dims.to_vec(),
                    &CPU,
                ),
                DType::BF16 => Tensor::from_vec(
                    values.iter().copied().map(half::bf16::from_f32).collect(),
                    dims.to_vec(),
                    &CPU,
                ),
                _ => unreachable!(),
            }
            .unwrap()
        };
        let mut norm = LayerNorm::with_eps([3], 1e-3, &CPU).unwrap();
        let x = make(&[0.5, -1.5, 2.0, 0.25, -0.75, 1.25], &[2, 3]);
        let weight = make(&[1.5, -0.5, 2.0], &[3]);
        let bias = make(&[0.25, -0.5, 0.75], &[3]);
        norm.weight.set(weight.clone()).unwrap();
        norm.bias.set(bias.clone()).unwrap();
        let got = norm.forward(&x, Mode::EVAL).unwrap();
        let inputs = [0.5f32, -1.5, 2.0, 0.25, -0.75, 1.25];
        let weights = [1.5f32, -0.5, 2.0];
        let biases = [0.25f32, -0.5, 0.75];
        let mut expected = Vec::with_capacity(inputs.len());
        for row in inputs.chunks_exact(3) {
            let row: Vec<f32> = row
                .iter()
                .map(|&value| match dtype {
                    DType::F16 => half::f16::from_f32(value).to_f32(),
                    DType::BF16 => half::bf16::from_f32(value).to_f32(),
                    _ => unreachable!(),
                })
                .collect();
            let mean = row.iter().sum::<f32>() / 3.0;
            let variance = row
                .iter()
                .map(|&value| (value - mean) * (value - mean))
                .sum::<f32>()
                / 3.0;
            let inverse = 1.0 / (variance + 1e-3).sqrt();
            for col in 0..3 {
                expected.push((row[col] - mean) * inverse * weights[col] + biases[col]);
            }
        }
        match dtype {
            DType::F16 => assert_eq!(
                got.to_vec::<half::f16>().unwrap(),
                expected
                    .into_iter()
                    .map(half::f16::from_f32)
                    .collect::<Vec<_>>()
            ),
            DType::BF16 => assert_eq!(
                got.to_vec::<half::bf16>().unwrap(),
                expected
                    .into_iter()
                    .map(half::bf16::from_f32)
                    .collect::<Vec<_>>()
            ),
            _ => unreachable!(),
        }
    }
}

#[test]
fn reduced_precision_layer_norm_backward_reaches_input_and_affine_params() {
    for dtype in [DType::F16, DType::BF16] {
        let mut norm = LayerNorm::with_eps([3], 1e-3, &CPU).unwrap();
        crate::nn::to_dtype(&mut norm, dtype).unwrap();
        norm.weight
            .set(t(&[1.5, -0.5, 2.0], [3]).to_dtype(dtype).unwrap())
            .unwrap();
        let x = t(&[0.5, -1.5, 2.0, 0.25, -0.75, 1.25], [2, 3])
            .to_dtype(dtype)
            .unwrap()
            .traced()
            .unwrap();
        let loss = norm.forward(&x, Mode::TRAIN).unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let reduced_grads = [
            grads.wrt_input(&x).unwrap(),
            grads.wrt(&norm.weight).unwrap(),
            grads.wrt(&norm.bias).unwrap(),
        ];
        for grad in &reduced_grads {
            assert_eq!(grad.dtype(), dtype);
            assert!(
                grad.to_dtype(DType::F32)
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .all(|value| value.is_finite())
            );
        }

        let mut reference = LayerNorm::with_eps([3], 1e-3, &CPU).unwrap();
        reference.weight.set(t(&[1.5, -0.5, 2.0], [3])).unwrap();
        let reference_x = x.detach().to_dtype(DType::F32).unwrap().traced().unwrap();
        let reference_loss = reference
            .forward(&reference_x, Mode::TRAIN)
            .unwrap()
            .sum_all()
            .unwrap();
        let reference_grads = reference_loss.backward().unwrap();
        let expected = [
            reference_grads.wrt_input(&reference_x).unwrap(),
            reference_grads.wrt(&reference.weight).unwrap(),
            reference_grads.wrt(&reference.bias).unwrap(),
        ];
        for (got, expected) in reduced_grads.iter().zip(expected) {
            match dtype {
                DType::F16 => assert_eq!(
                    got.to_vec::<half::f16>().unwrap(),
                    expected
                        .to_dtype(dtype)
                        .unwrap()
                        .to_vec::<half::f16>()
                        .unwrap()
                ),
                DType::BF16 => assert_eq!(
                    got.to_vec::<half::bf16>().unwrap(),
                    expected
                        .to_dtype(dtype)
                        .unwrap()
                        .to_vec::<half::bf16>()
                        .unwrap()
                ),
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn rms_norm_matches_an_independent_reference() {
    let mut norm = RMSNorm::new([4], &CPU).unwrap();
    load(&mut norm, &[("weight", t(&[1.5, -0.5, 2.0, 0.25], [4]))]);
    let y = norm.forward(&t(&REF_X, [2, 3, 4]), Mode::EVAL).unwrap();
    close(
        &y,
        &[
            0.585_539_9,
            0.585_539_9,
            3.122_879_3,
            0.048_795,
            -1.356_800_1,
            -0.753_777_8,
            1.809_066_8,
            -0.075_377_8,
            1.590_863_1,
            0.378_776_9,
            0.151_510_8,
            0.378_776_9,
            -2.328_341_3,
            -0.145_521_3,
            2.328_341_3,
            -0.097_014_2,
            2.475_410_6,
            -0.171_903_5,
            -1.925_319_4,
            0.120_332_5,
            1.922_450_6,
            0.035_600_9,
            1.139_23,
            -0.356_009_4,
        ],
        1e-5,
    );
}

#[test]
fn batch_norm_train_matches_an_independent_reference() {
    // [N=2, C=3, H=2, W=2]: N, C, H and W are pairwise distinguishable
    // enough that reducing the wrong axis set changes every output value.
    let mut bn = BatchNorm2d::new(3, &CPU).unwrap();
    load(
        &mut bn,
        &[
            ("weight", t(&[1.5, -0.5, 2.0], [3])),
            ("bias", t(&[0.25, -0.5, 0.75], [3])),
            ("running_mean", t(&[0.0, 0.0, 0.0], [3])),
            ("running_var", t(&[1.0, 1.0, 1.0], [3])),
        ],
    );
    let y = bn.forward(&t(&REF_X, [2, 3, 2, 2]), Mode::TRAIN).unwrap();
    close(
        &y,
        &[
            0.743_497_4,
            -1.596_046,
            2.498_155,
            0.451_054_5,
            -0.042_788_6,
            -0.793_084_2,
            -0.605_510_3,
            -0.230_362_5,
            2.314_529_3,
            -1.348_758_9,
            0.330_248_2,
            3.230_351_4,
            -2.180_931_8,
            0.597_276,
            1.913_269,
            -0.426_274_3,
            -1.449_593,
            -0.558_616_8,
            0.332_359_2,
            -0.652_403_8,
            2.925_077_4,
            0.024_974_2,
            1.398_707_3,
            -2.875_129,
        ],
        1e-5,
    );
    // The same reference's per-channel statistics, folded in at 0.1 (the
    // means are `0.1 · [0.078125, 0.46875, 0.46875]`, the variances
    // `0.9 + 0.1 · unbiased([1.6442871, 1.7763672, 2.6826172])`).
    close(
        bn.running_mean(),
        &[0.007_812_5, 0.046_875, 0.046_875],
        1e-6,
    );
    close(
        bn.running_var(),
        &[1.087_918_5, 1.103_013_4, 1.206_584_8],
        1e-5,
    );
}

// -- BatchNorm2d: statistics, buffers, both branches ------------------

/// A `[2, 2, 1, 3]` input — `C = 2` deliberately differs from `W = 3`, so a
/// per-channel statistic mistakenly broadcast against the width axis
/// cannot silently line up — with per-channel values chosen so the two
/// channels have *different* means and variances and the arithmetic below
/// is exact:
///
/// channel 0: 1, 3, 5, 7, 9, 11 -> mean 6, Σd² = 70, biased 70/6, unbiased 14
/// channel 1: 2, 2, 2, 2, 2, 14 -> mean 4, Σd² = 120, biased 20,   unbiased 24
fn bn_input() -> Tensor {
    t(
        &[1.0, 3.0, 5.0, 2.0, 2.0, 2.0, 7.0, 9.0, 11.0, 2.0, 2.0, 14.0],
        [2, 2, 1, 3],
    )
}

/// The biased (`correction = 0`) per-channel variances of [`bn_input`] —
/// what the *normalization* divides by.
const BN_VAR_BIASED: [f32; 2] = [70.0 / 6.0, 20.0];
/// The unbiased (`correction = 1`) ones — what `running_var` stores.
const BN_VAR_UNBIASED: [f32; 2] = [14.0, 24.0];

#[test]
fn batch_norm_train_matches_hand_computed_values() {
    let mut bn = BatchNorm2d::new(2, &CPU).unwrap();
    assert_eq!(bn.channels(), 2);
    let y = bn.forward(&bn_input(), Mode::TRAIN).unwrap();
    // The biased variance is the divisor: with correction = 1 (70/5 and
    // 24) the outer values below would be ~9% smaller.
    let s0 = (BN_VAR_BIASED[0] + 1e-5).sqrt();
    let s1 = (BN_VAR_BIASED[1] + 1e-5).sqrt();
    close(
        &y,
        &[
            (1.0 - 6.0) / s0,
            (3.0 - 6.0) / s0,
            (5.0 - 6.0) / s0,
            (2.0 - 4.0) / s1,
            (2.0 - 4.0) / s1,
            (2.0 - 4.0) / s1,
            (7.0 - 6.0) / s0,
            (9.0 - 6.0) / s0,
            (11.0 - 6.0) / s0,
            (2.0 - 4.0) / s1,
            (2.0 - 4.0) / s1,
            (14.0 - 4.0) / s1,
        ],
        1e-5,
    );
}

#[test]
fn batch_norm_train_updates_the_running_statistics() {
    let mut bn = BatchNorm2d::new(2, &CPU).unwrap();
    // Buffers start at mean 0 / var 1.
    close(bn.running_mean(), &[0.0, 0.0], 0.0);
    close(bn.running_var(), &[1.0, 1.0], 0.0);

    bn.forward(&bn_input(), Mode::TRAIN).unwrap();
    // mean: 0.9·0 + 0.1·[6, 4]
    close(bn.running_mean(), &[0.6, 0.4], 1e-6);
    // var: 0.9·1 + 0.1·[14, 24] — the *unbiased* batch variance, not the
    // [70/6, 20] the normalization above divided by.
    close(bn.running_var(), &[2.3, 3.3], 1e-5);

    // A second identical batch applies the same EMA again.
    bn.forward(&bn_input(), Mode::TRAIN).unwrap();
    close(bn.running_mean(), &[1.14, 0.76], 1e-5);
    close(bn.running_var(), &[3.47, 5.37], 1e-4);
}

#[test]
fn batch_norm_momentum_one_replaces_the_statistics_outright() {
    // momentum = 1 keeps nothing of the old estimate: the buffers become
    // this batch's statistics exactly. The clearest probe of both the EMA
    // weighting convention (momentum weights the *new* value) and the
    // unbiased-variance rule.
    let mut bn = BatchNorm2d::with_params(2, 1e-5, 1.0, &CPU).unwrap();
    bn.forward(&bn_input(), Mode::TRAIN).unwrap();
    close(bn.running_mean(), &[6.0, 4.0], 1e-5);
    close(bn.running_var(), &BN_VAR_UNBIASED, 1e-4);
    // momentum = 0 is the mirror image: the batch is ignored.
    let mut frozen = BatchNorm2d::with_params(2, 1e-5, 0.0, &CPU).unwrap();
    frozen.forward(&bn_input(), Mode::TRAIN).unwrap();
    close(frozen.running_mean(), &[0.0, 0.0], 0.0);
    close(frozen.running_var(), &[1.0, 1.0], 0.0);
}

#[test]
fn batch_norm_eval_uses_the_running_statistics_and_leaves_them_alone() {
    let mut bn = BatchNorm2d::new(2, &CPU).unwrap();
    load(
        &mut bn,
        &[
            ("weight", t(&[2.0, 3.0], [2])),
            ("bias", t(&[1.0, -1.0], [2])),
            ("running_mean", t(&[1.0, 2.0], [2])),
            ("running_var", t(&[4.0, 9.0], [2])),
        ],
    );
    let y = bn.forward(&bn_input(), Mode::EVAL).unwrap();
    let s0 = (4.0f32 + 1e-5).sqrt();
    let s1 = (9.0f32 + 1e-5).sqrt();
    close(
        &y,
        &[
            2.0 * (1.0 - 1.0) / s0 + 1.0,
            2.0 * (3.0 - 1.0) / s0 + 1.0,
            2.0 * (5.0 - 1.0) / s0 + 1.0,
            3.0 * (2.0 - 2.0) / s1 - 1.0,
            3.0 * (2.0 - 2.0) / s1 - 1.0,
            3.0 * (2.0 - 2.0) / s1 - 1.0,
            2.0 * (7.0 - 1.0) / s0 + 1.0,
            2.0 * (9.0 - 1.0) / s0 + 1.0,
            2.0 * (11.0 - 1.0) / s0 + 1.0,
            3.0 * (2.0 - 2.0) / s1 - 1.0,
            3.0 * (2.0 - 2.0) / s1 - 1.0,
            3.0 * (14.0 - 2.0) / s1 - 1.0,
        ],
        1e-5,
    );
    // Eval never touches the buffers.
    close(bn.running_mean(), &[1.0, 2.0], 0.0);
    close(bn.running_var(), &[4.0, 9.0], 0.0);
}

#[test]
fn batch_norm_train_and_eval_diverge_on_the_same_input() {
    let mut bn = BatchNorm2d::new(2, &CPU).unwrap();
    let x = bn_input();
    // Fresh buffers (mean 0, var 1) can only apply the identity affine.
    let eval = v(&bn.forward(&x, Mode::EVAL).unwrap());
    close_all(&eval, &v(&x), 1e-3);
    // The training branch whitens instead — a completely different result.
    let train = v(&bn.forward(&x, Mode::TRAIN).unwrap());
    assert!(
        train.iter().zip(&eval).any(|(a, b)| (a - b).abs() > 1.0),
        "train {train:?} should differ from eval {eval:?}"
    );
    // …and after enough training batches the eval branch converges on it.
    for _ in 0..200 {
        bn.forward(&x, Mode::TRAIN).unwrap();
    }
    let settled = v(&bn.forward(&x, Mode::EVAL).unwrap());
    for (i, (&a, &b)) in settled.iter().zip(&train).enumerate() {
        // The running var holds the *unbiased* variance (m/(m−1) = 6/5 of
        // the biased one), so the settled eval output is the train output
        // scaled by √(5/6) — a visible, deliberate 9% difference.
        assert!(
            (a - b * (5.0f32 / 6.0).sqrt()).abs() < 1e-3,
            "element {i}: eval {a} vs scaled train {b}"
        );
    }
}

#[test]
fn batch_norm_mode_axes_are_independent() {
    let x = bn_input();

    // TRAIN.frozen(): train behavior (buffers advance) with no graph.
    let mut a = BatchNorm2d::new(2, &CPU).unwrap();
    let out = a.forward(&x, Mode::TRAIN.frozen()).unwrap();
    close(a.running_mean(), &[0.6, 0.4], 1e-6);
    assert!(matches!(out.backward(), Err(Error::NotTraced { .. })));

    // EVAL.recorded(): eval behavior (buffers stand still) but recorded.
    let mut b = BatchNorm2d::new(2, &CPU).unwrap();
    let out = b.forward(&x, Mode::EVAL.recorded()).unwrap();
    close(b.running_mean(), &[0.0, 0.0], 0.0);
    assert!(out.sum_all().unwrap().backward().is_ok());
}

#[test]
fn batch_norm_survives_a_constant_channel() {
    // Zero variance is the degenerate case `eps` exists for: the output
    // must be finite (and, with default parameters, exactly the bias).
    let mut bn = BatchNorm2d::new(2, &CPU).unwrap();
    // [N=2, C=2, H=1, W=2]: channel 0 is all 5s, channel 1 is 1,2,3,4.
    let x = t(&[5.0, 5.0, 1.0, 2.0, 5.0, 5.0, 3.0, 4.0], [2, 2, 1, 2]);
    let y = v(&bn.forward(&x, Mode::TRAIN).unwrap());
    assert!(y.iter().all(|e| e.is_finite()), "{y:?}");
    close_all(&y[0..2], &[0.0, 0.0], 1e-3);
    // Channel 0 folds a zero variance into its buffer — the EMA keeps the
    // initial 1 alive, so a later eval forward is finite as well.
    // Channel 1: biased 1.25, unbiased 1.25 · 4/3 = 5/3.
    close(bn.running_var(), &[0.9, 0.9 + 0.1 * (5.0 / 3.0)], 1e-6);
    let e = v(&bn.forward(&x, Mode::EVAL).unwrap());
    assert!(e.iter().all(|q| q.is_finite()), "{e:?}");
}

#[test]
fn the_running_buffers_never_hold_a_graph() {
    // The folded-in statistics are detached: a buffer that kept the batch's
    // graph alive would retain every activation of the step that wrote it.
    let mut bn = BatchNorm2d::new(2, &CPU).unwrap();
    bn.forward(&bn_input(), Mode::TRAIN).unwrap();
    for buffer in [bn.running_mean(), bn.running_var()] {
        assert!(matches!(
            buffer.backward(),
            Err(Error::NotTraced { op: "backward" })
        ));
    }
    // And the buffers are absent from the gradients of a training step.
    let grads = bn
        .forward(&bn_input(), Mode::TRAIN)
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    assert_eq!(grads.len(), 2, "weight and bias only");
}

#[test]
fn non_contiguous_inputs_normalize_over_the_same_axes() {
    // A permuted NHWC->NCHW view is the realistic way an image batch
    // arrives, and a statistic taken over strided memory is exactly where
    // an axis mistake hides. Both layers must agree with the materialized
    // input element for element.
    let rows = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]);
    let view = rows.transpose(0, 1).unwrap();
    assert!(!view.is_contiguous());
    let mut ln = LayerNorm::new([3], &CPU).unwrap();
    assert_eq!(
        v(&ln.forward(&view, Mode::EVAL).unwrap()),
        v(&ln.forward(&view.contiguous().unwrap(), Mode::EVAL).unwrap())
    );

    let nhwc = t(
        &[1.0, 3.0, 5.0, 2.0, 2.0, 2.0, 7.0, 9.0, 11.0, 2.0, 2.0, 14.0],
        [2, 1, 3, 2],
    );
    // [N, H, W, C] -> [N, C, H, W].
    let nchw = nhwc.permute(&[0, 3, 1, 2]).unwrap();
    assert!(!nchw.is_contiguous());
    let mut a = BatchNorm2d::new(2, &CPU).unwrap();
    let mut b = BatchNorm2d::new(2, &CPU).unwrap();
    assert_eq!(
        v(&a.forward(&nchw, Mode::TRAIN).unwrap()),
        v(&b.forward(&nchw.contiguous().unwrap(), Mode::TRAIN).unwrap())
    );
    assert_eq!(v(a.running_mean()), v(b.running_mean()));
}

#[test]
fn batch_norm_rejects_bad_inputs() {
    let mut bn = BatchNorm2d::new(2, &CPU).unwrap();
    // Wrong rank.
    assert!(matches!(
        bn.forward(&t(&[1.0, 2.0], [1, 2]), Mode::EVAL),
        Err(Error::RankMismatch { expected: 4, .. })
    ));
    // Wrong channel count — the axis that must not be confused with W.
    assert!(matches!(
        bn.forward(&t(&[1.0, 2.0, 3.0], [1, 3, 1, 1]), Mode::EVAL),
        Err(Error::ShapeMismatch {
            op: "BatchNorm2d::forward",
            ..
        })
    ));
    // A single element per channel has no unbiased variance to store…
    let single = t(&[1.0, 2.0], [1, 2, 1, 1]);
    assert!(matches!(
        bn.forward(&single, Mode::TRAIN),
        Err(Error::InvalidArg {
            op: "BatchNorm2d::forward",
            ..
        })
    ));
    // …but eval reads the buffers and is perfectly happy with it.
    assert!(bn.forward(&single, Mode::EVAL).is_ok());
}

// -- module integration ----------------------------------------------

#[test]
fn state_dict_paths_are_the_documented_leaf_names() {
    let bn = BatchNorm2d::new(3, &CPU).unwrap();
    assert_eq!(
        nn::state_dict(&bn).keys().collect::<Vec<_>>(),
        ["bias", "running_mean", "running_var", "weight"]
    );
    // Buffers are not trainable, so they do not count as parameters.
    assert_eq!(nn::num_params(&bn), 6);

    let ln = LayerNorm::new([2, 3], &CPU).unwrap();
    assert_eq!(
        nn::state_dict(&ln).keys().collect::<Vec<_>>(),
        ["bias", "weight"]
    );
    assert_eq!(nn::num_params(&ln), 12);
}

#[test]
fn norms_nest_in_a_sequential_and_survive_replication() {
    let mut net = Sequential::new()
        .push(LayerNorm::new([4], &CPU).unwrap())
        .push(RMSNorm::new([4], &CPU).unwrap());
    assert_eq!(
        nn::state_dict(&net).keys().collect::<Vec<_>>(),
        ["0.bias", "0.weight", "1.weight"]
    );
    let x = t(&[1.0, 2.0, 3.0, 4.0], [4]);
    let want = v(&net.forward(&x, Mode::EVAL).unwrap());

    // Replication: construct + load_state_dict (no disk, no Clone).
    let mut replica = Sequential::new()
        .push(LayerNorm::new([4], &CPU).unwrap())
        .push(RMSNorm::new([4], &CPU).unwrap());
    nn::load_state_dict(&mut replica, &nn::state_dict(&net)).unwrap();
    assert_eq!(v(&replica.forward(&x, Mode::EVAL).unwrap()), want);
}

#[test]
fn a_batch_norm_checkpoint_restores_the_eval_branch() {
    let mut trained = BatchNorm2d::new(2, &CPU).unwrap();
    trained.forward(&bn_input(), Mode::TRAIN).unwrap();
    let want = v(&trained.forward(&bn_input(), Mode::EVAL).unwrap());

    let mut restored = BatchNorm2d::new(2, &CPU).unwrap();
    nn::load_state_dict(&mut restored, &nn::state_dict(&trained)).unwrap();
    assert_eq!(v(&restored.forward(&bn_input(), Mode::EVAL).unwrap()), want);
}

#[test]
fn a_device_move_carries_the_running_statistics() {
    // Buffers are structure, not just precision: a moved model must keep
    // its running statistics, or its eval branch silently changes meaning.
    // (Only `Device::Cpu` exists in a default build, so this is a round trip.)
    let mut bn = BatchNorm2d::new(2, &CPU).unwrap();
    bn.forward(&bn_input(), Mode::TRAIN).unwrap();
    let want = v(&bn.forward(&bn_input(), Mode::EVAL).unwrap());

    nn::to_device(&mut bn, &CPU).unwrap();
    assert_eq!(bn.running_mean().dtype(), DType::F32);
    close(bn.running_mean(), &[0.6, 0.4], 1e-6);
    assert_eq!(v(&bn.forward(&bn_input(), Mode::EVAL).unwrap()), want);
}

#[test]
fn a_dtype_conversion_visits_the_running_statistics() {
    // The buffers are float leaves, so `nn::to_dtype` converts them along
    // with the parameters and the eval branch still finds its statistics.
    // Only the identity float lane is exercisable today — the CPU backend's
    // F32↔F64 cast is deferred (`backend::cpu::host::cast`) — so this pins
    // that the buffers are *visited* as float leaves, not that a widening
    // conversion is numerically faithful.
    let mut bn = BatchNorm2d::new(2, &CPU).unwrap();
    bn.forward(&bn_input(), Mode::TRAIN).unwrap();
    let want = v(&bn.forward(&bn_input(), Mode::EVAL).unwrap());

    nn::to_dtype(&mut bn, DType::F32).unwrap();
    assert_eq!(bn.running_mean().dtype(), DType::F32);
    close(bn.running_mean(), &[0.6, 0.4], 1e-6);
    assert_eq!(v(&bn.forward(&bn_input(), Mode::EVAL).unwrap()), want);
}

// -- gradients: finite differences against the shared harness ---------

const EPS: f64 = 1e-3;
const TOL: f64 = 2e-3;

/// A fixed non-constant weighting for the scalar objective: without it the
/// sum of a normalized block is (nearly) constant and its gradient
/// vanishes, which would make the check vacuous.
fn coef(dims: &[usize]) -> Tensor {
    let n: usize = dims.iter().product();
    t(
        &(0..n)
            .map(|i| 0.3 + 0.7 * ((i % 5) as f32) - 0.2 * ((i % 3) as f32))
            .collect::<Vec<f32>>(),
        dims.to_vec(),
    )
}

#[test]
fn grad_layer_norm() {
    let x = t(&[0.5, -1.5, 2.0, 0.25, -0.75, 1.25], [2, 3]);
    let weight = t(&[1.5, -0.5, 2.0], [3]);
    let bias = t(&[0.25, -0.5, 0.75], [3]);
    let c = coef(&[2, 3]);
    check_grad(
        |xs| {
            composed_layer_norm(&xs[0], &xs[1], Some(&xs[2]), LayerNorm::DEFAULT_EPS)?
                .mul(&c)?
                .sum_all()
        },
        &[x, weight, bias],
        EPS,
        TOL,
    )
    .unwrap();
}

#[test]
fn grad_fused_layer_norm_wrt_input_weight_and_bias() {
    let x = t(&[0.5, -1.5, 2.0, 0.25, -0.75, 1.25], [2, 3]);
    let weight = t(&[1.5, -0.5, 2.0], [3]);
    let bias = t(&[0.25, -0.5, 0.75], [3]);
    let c = coef(&[2, 3]);
    check_grad(
        |xs| {
            fused_layer_norm(&xs[0], &xs[1], &xs[2], LayerNorm::DEFAULT_EPS)?
                .mul(&c)?
                .sum_all()
        },
        &[x, weight, bias],
        EPS,
        TOL,
    )
    .unwrap();
}

#[test]
fn fused_layer_norm_preserves_mode_and_parameter_freezing() {
    let x = t(&[0.5, -1.5, 2.0, 0.25, -0.75, 1.25], [2, 3]);
    let mut norm = LayerNorm::new([3], &CPU).unwrap();

    let eval = norm.forward(&x, Mode::EVAL).unwrap();
    assert!(matches!(eval.backward(), Err(Error::NotTraced { .. })));
    let frozen_mode = norm.forward(&x, Mode::TRAIN.frozen()).unwrap();
    assert!(matches!(
        frozen_mode.backward(),
        Err(Error::NotTraced { .. })
    ));

    norm.weight.freeze();
    let grads = norm
        .forward(&x, Mode::EVAL.recorded())
        .unwrap()
        .mul(&coef(&[2, 3]))
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    assert_eq!(grads.len(), 1, "only the unfrozen bias receives a gradient");

    let traced = x.traced().unwrap();
    let grads = norm
        .forward(&traced, Mode::TRAIN.frozen())
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    assert_eq!(grads.wrt_input(&traced).unwrap().dims(), &[2, 3]);
}

#[test]
fn grad_layer_norm_over_several_axes() {
    let x = t(&[0.5, -1.5, 2.0, 0.25, -0.75, 1.25, 0.75, -0.25], [2, 2, 2]);
    let weight = t(&[1.5, -0.5, 2.0, 0.5], [2, 2]);
    let bias = t(&[0.25, -0.5, 0.75, 0.0], [2, 2]);
    let c = coef(&[2, 2, 2]);
    check_grad(
        |xs| {
            composed_layer_norm(&xs[0], &xs[1], Some(&xs[2]), LayerNorm::DEFAULT_EPS)?
                .mul(&c)?
                .sum_all()
        },
        &[x, weight, bias],
        EPS,
        TOL,
    )
    .unwrap();
}

#[test]
fn grad_rms_norm() {
    let x = t(&[0.5, -1.5, 2.0, 0.25, -0.75, 1.25], [2, 3]);
    let weight = t(&[1.5, -0.5, 2.0], [3]);
    let c = coef(&[2, 3]);
    check_grad(
        |xs| {
            rms_norm(&xs[0], &xs[1], RMSNorm::DEFAULT_EPS)?
                .mul(&c)?
                .sum_all()
        },
        &[x, weight],
        EPS,
        TOL,
    )
    .unwrap();
}

#[test]
fn grad_batch_norm_train_branch() {
    // The important case: the gradient flows through the batch statistics
    // as well as through `x` directly.
    let x = t(
        &[
            0.5, -1.5, 2.0, 0.25, -0.75, 1.25, 0.75, -0.25, 1.75, -1.25, 0.125, 2.5,
        ],
        [2, 2, 1, 3],
    );
    let weight = t(&[1.5, -0.5], [2]);
    let bias = t(&[0.25, 0.75], [2]);
    let c = coef(&[2, 2, 1, 3]);
    check_grad(
        |xs| {
            let (mu, var) = batch_stats(&xs[0])?;
            batch_norm(&xs[0], &mu, &var, &xs[1], &xs[2], BatchNorm2d::DEFAULT_EPS)?
                .mul(&c)?
                .sum_all()
        },
        &[x, weight, bias],
        EPS,
        TOL,
    )
    .unwrap();
}

#[test]
fn grad_batch_norm_eval_branch() {
    // Fixed statistics: `x` only enters through the affine, so the
    // gradient must *not* pick up the through-statistics terms.
    let x = t(
        &[
            0.5, -1.5, 2.0, 0.25, -0.75, 1.25, 0.75, -0.25, 1.75, -1.25, 0.125, 2.5,
        ],
        [2, 2, 1, 3],
    );
    let weight = t(&[1.5, -0.5], [2]);
    let bias = t(&[0.25, 0.75], [2]);
    let mu = t(&[0.25, -0.5], [1, 2, 1, 1]);
    let var = t(&[1.5, 0.75], [1, 2, 1, 1]);
    let c = coef(&[2, 2, 1, 3]);
    check_grad(
        |xs| {
            batch_norm(&xs[0], &mu, &var, &xs[1], &xs[2], BatchNorm2d::DEFAULT_EPS)?
                .mul(&c)?
                .sum_all()
        },
        &[x, weight, bias],
        EPS,
        TOL,
    )
    .unwrap();
}

/// The input the two layer-level gradient checks differentiate.
fn bn_grad_input() -> Tensor {
    t(
        &[
            0.5, -1.5, 2.0, 0.25, -0.75, 1.25, 0.75, -0.25, 1.75, -1.25, 0.125, 2.5,
        ],
        [2, 2, 1, 3],
    )
}

#[test]
fn grad_through_the_batch_norm_layer_train_branch() {
    // Differentiating `forward` itself, not the free function: this is what
    // pins the *layer* onto the train-branch formula, statistics included.
    // (`RefCell` because `check_grad` takes an `Fn` while `forward` needs
    // `&mut self`; the repeated forwards only advance the buffers, which do
    // not enter the train branch's arithmetic.)
    let bn = std::cell::RefCell::new(BatchNorm2d::new(2, &CPU).unwrap());
    let c = coef(&[2, 2, 1, 3]);
    check_grad(
        |xs| {
            bn.borrow_mut()
                .forward(&xs[0], Mode::TRAIN)?
                .mul(&c)?
                .sum_all()
        },
        &[bn_grad_input()],
        EPS,
        TOL,
    )
    .unwrap();
}

#[test]
fn grad_through_the_batch_norm_layer_eval_branch() {
    // `Mode::EVAL.recorded()` is the fine-tuning flow: the buffers are
    // constants, so `x` reaches the output only through the affine.
    // (A finite-difference check compares the analytic gradient with the
    // derivative of whatever `forward` computes, so it cannot tell the two
    // *branches* apart — the value tests above do that. What it does pin is
    // that the graph the branch builds matches the arithmetic it performs,
    // the failure mode of a statistic detached on one path only.)
    let mut layer = BatchNorm2d::new(2, &CPU).unwrap();
    load(
        &mut layer,
        &[
            ("weight", t(&[1.5, -0.5], [2])),
            ("bias", t(&[0.25, 0.75], [2])),
            ("running_mean", t(&[0.25, -0.5], [2])),
            ("running_var", t(&[1.5, 0.75], [2])),
        ],
    );
    let bn = std::cell::RefCell::new(layer);
    let c = coef(&[2, 2, 1, 3]);
    check_grad(
        |xs| {
            bn.borrow_mut()
                .forward(&xs[0], Mode::EVAL.recorded())?
                .mul(&c)?
                .sum_all()
        },
        &[bn_grad_input()],
        EPS,
        TOL,
    )
    .unwrap();
}

#[test]
fn layer_gradients_reach_the_parameters_through_param_get() {
    // The free functions above carry the formulas; this checks the layers
    // actually route `Param::get(mode)` into them, by confirming a
    // gradient exists for every parameter path under Mode::TRAIN.
    let x = t(&[0.5, -1.5, 2.0, 0.25, -0.75, 1.25], [2, 3]);
    let mut ln = LayerNorm::new([3], &CPU).unwrap();
    let grads = ln
        .forward(&x, Mode::TRAIN)
        .unwrap()
        .mul(&coef(&[2, 3]))
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    assert_eq!(grads.len(), 2, "one gradient each for weight and bias");

    let mut rms = RMSNorm::new([3], &CPU).unwrap();
    let grads = rms
        .forward(&x, Mode::TRAIN)
        .unwrap()
        .mul(&coef(&[2, 3]))
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    assert_eq!(grads.len(), 1, "RMSNorm has only `weight`");

    let mut bn = BatchNorm2d::new(2, &CPU).unwrap();
    let grads = bn
        .forward(&bn_input(), Mode::TRAIN)
        .unwrap()
        .mul(&coef(&[2, 2, 1, 3]))
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    assert_eq!(grads.len(), 2, "weight and bias, never the buffers");
}
