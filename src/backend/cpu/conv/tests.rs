//! The convolution and pooling kernels, forward and backward, against
//! independently computed references.

use super::*;
use crate::dtype::DType;
use crate::storage::CpuStorage;
use std::sync::Arc;

const IDENTITY: Conv2dParams = Conv2dParams {
    kernel: (2, 2),
    stride: (1, 1),
    padding: (0, 0),
    dilation: (1, 1),
};

fn params(
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Conv2dParams {
    Conv2dParams {
        kernel,
        stride,
        padding,
        dilation,
    }
}

fn f32_storage(data: Vec<f32>) -> Storage {
    Storage::Cpu(CpuStorage::F32(Arc::new(data)))
}

fn as_f32(s: &Storage) -> Vec<f32> {
    match s {
        Storage::Cpu(CpuStorage::F32(v)) => v.as_ref().clone(),
        _ => panic!("expected f32 storage"),
    }
}

/// A tiny xorshift PRNG so the cross-check tests are deterministic without
/// depending on the crate `Rng`'s stream.
struct Prng(u64);

impl Prng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn f(&mut self) -> f32 {
        (self.next() % 2001) as f32 / 500.0 - 2.0
    }
    fn values(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f()).collect()
    }
}

// ----- forward goldens ----------------------------------------------

#[test]
fn conv2d_3x3_by_2x2_hand_computed() {
    // input 1x1x3x3 = 1..9, weight 1x1x2x2 = [[1,0],[0,1]] (the diagonal
    // pick), so each output is the sum of a 2x2 window's diagonal.
    let input = f32_storage((1..=9).map(|v| v as f32).collect());
    let weight = f32_storage(vec![1.0, 0.0, 0.0, 1.0]);
    let il = Layout::contiguous([1, 1, 3, 3]).unwrap();
    let wl = Layout::contiguous([1, 1, 2, 2]).unwrap();
    let out = conv(
        ConvOp::Conv2d,
        &[View::new(&input, &il), View::new(&weight, &wl)],
        &IDENTITY,
    )
    .unwrap();
    // 1+5, 2+6, 4+8, 5+9
    assert_eq!(as_f32(&out), vec![6.0, 8.0, 12.0, 14.0]);
}

#[test]
fn conv2d_multi_channel_hand_computed() {
    // input 1x2x2x2 = 1..8, weight 1x2x2x2 of ones -> the sum, 36.
    let input = f32_storage((1..=8).map(|v| v as f32).collect());
    let weight = f32_storage(vec![1.0; 8]);
    let il = Layout::contiguous([1, 2, 2, 2]).unwrap();
    let wl = Layout::contiguous([1, 2, 2, 2]).unwrap();
    let out = conv(
        ConvOp::Conv2d,
        &[View::new(&input, &il), View::new(&weight, &wl)],
        &IDENTITY,
    )
    .unwrap();
    assert_eq!(as_f32(&out), vec![36.0]);
}

#[test]
fn conv2d_padding_and_stride_hand_computed() {
    // input 1x1x2x2 = [[1,2],[3,4]], 2x2 kernel of ones, padding 1,
    // stride 2 -> padded 4x4, output 2x2, each window seeing one corner.
    let input = f32_storage(vec![1.0, 2.0, 3.0, 4.0]);
    let weight = f32_storage(vec![1.0; 4]);
    let il = Layout::contiguous([1, 1, 2, 2]).unwrap();
    let wl = Layout::contiguous([1, 1, 2, 2]).unwrap();
    let p = params((2, 2), (2, 2), (1, 1), (1, 1));
    let out = conv(
        ConvOp::Conv2d,
        &[View::new(&input, &il), View::new(&weight, &wl)],
        &p,
    )
    .unwrap();
    assert_eq!(as_f32(&out), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn conv2d_dilation_hand_computed() {
    // input 1x1x3x3 = 1..9, 2x2 kernel of ones dilated by 2 -> a single
    // output summing the four corners 1 + 3 + 7 + 9 = 20.
    let input = f32_storage((1..=9).map(|v| v as f32).collect());
    let weight = f32_storage(vec![1.0; 4]);
    let il = Layout::contiguous([1, 1, 3, 3]).unwrap();
    let wl = Layout::contiguous([1, 1, 2, 2]).unwrap();
    let p = params((2, 2), (1, 1), (0, 0), (2, 2));
    let out = conv(
        ConvOp::Conv2d,
        &[View::new(&input, &il), View::new(&weight, &wl)],
        &p,
    )
    .unwrap();
    assert_eq!(as_f32(&out), vec![20.0]);
}

#[test]
fn conv2d_reads_strided_input_views() {
    // Two images stored back to back; convolve only the second by
    // narrowing the batch axis (a non-zero-offset view).
    let input = f32_storage((1..=18).map(|v| v as f32).collect());
    let il = Layout::contiguous([2, 1, 3, 3]).unwrap();
    let second = il.narrow(0, 1, 1).unwrap();
    let weight = f32_storage(vec![1.0, 0.0, 0.0, 1.0]);
    let wl = Layout::contiguous([1, 1, 2, 2]).unwrap();
    let out = conv(
        ConvOp::Conv2d,
        &[View::new(&input, &second), View::new(&weight, &wl)],
        &IDENTITY,
    )
    .unwrap();
    // The second image is 10..18: 10+14, 11+15, 13+17, 14+18.
    assert_eq!(as_f32(&out), vec![24.0, 26.0, 30.0, 32.0]);
}

#[test]
fn conv2d_accumulates_f16_in_f32() {
    // 4096 channels of ones dotted with ones: exact in f32, saturating if
    // the kernel accumulated in f16.
    let k = 4096usize;
    let input = Storage::Cpu(CpuStorage::F16(Arc::new(vec![half::f16::from_f32(1.0); k])));
    let weight = Storage::Cpu(CpuStorage::F16(Arc::new(vec![half::f16::from_f32(1.0); k])));
    let il = Layout::contiguous([1, k, 1, 1]).unwrap();
    let wl = Layout::contiguous([1, k, 1, 1]).unwrap();
    let p = params((1, 1), (1, 1), (0, 0), (1, 1));
    let out = conv(
        ConvOp::Conv2d,
        &[View::new(&input, &il), View::new(&weight, &wl)],
        &p,
    )
    .unwrap();
    let got = match &out {
        Storage::Cpu(CpuStorage::F16(v)) => v.as_ref().clone(),
        _ => panic!("expected f16"),
    };
    assert_eq!(got[0].to_f32(), 4096.0);
}

#[test]
fn conv2d_accumulates_bf16_in_f32() {
    let k = 4096usize;
    let one = half::bf16::from_f32(1.0);
    let input = Storage::Cpu(CpuStorage::BF16(Arc::new(vec![one; k])));
    let weight = Storage::Cpu(CpuStorage::BF16(Arc::new(vec![one; k])));
    let il = Layout::contiguous([1, k, 1, 1]).unwrap();
    let wl = Layout::contiguous([1, k, 1, 1]).unwrap();
    let p = params((1, 1), (1, 1), (0, 0), (1, 1));
    let out = conv(
        ConvOp::Conv2d,
        &[View::new(&input, &il), View::new(&weight, &wl)],
        &p,
    )
    .unwrap();
    let Storage::Cpu(CpuStorage::BF16(got)) = out else {
        panic!("expected bf16")
    };
    assert_eq!(got[0].to_f32(), 4096.0);
}

#[test]
fn max_pool2d_hand_computed() {
    // 1x1x4x4 = 1..16, 2x2 window, stride 2.
    let input = f32_storage((1..=16).map(|v| v as f32).collect());
    let il = Layout::contiguous([1, 1, 4, 4]).unwrap();
    let p = params((2, 2), (2, 2), (0, 0), (1, 1));
    let out = conv(ConvOp::MaxPool2d, &[View::new(&input, &il)], &p).unwrap();
    assert_eq!(as_f32(&out), vec![6.0, 8.0, 14.0, 16.0]);
}

#[test]
fn max_pool2d_propagates_nan() {
    let input = f32_storage(vec![1.0, f32::NAN, 3.0, 4.0]);
    let il = Layout::contiguous([1, 1, 2, 2]).unwrap();
    let p = params((2, 2), (2, 2), (0, 0), (1, 1));
    let out = conv(ConvOp::MaxPool2d, &[View::new(&input, &il)], &p).unwrap();
    assert!(as_f32(&out)[0].is_nan());
}

#[test]
fn avg_pool2d_hand_computed() {
    let input = f32_storage((1..=16).map(|v| v as f32).collect());
    let il = Layout::contiguous([1, 1, 4, 4]).unwrap();
    let p = params((2, 2), (2, 2), (0, 0), (1, 1));
    let out = conv(ConvOp::AvgPool2d, &[View::new(&input, &il)], &p).unwrap();
    assert_eq!(as_f32(&out), vec![3.5, 5.5, 11.5, 13.5]);
}

#[test]
fn avg_pool2d_counts_padding_in_the_divisor() {
    // 1x1x2x2 = [[1,2],[3,4]], 2x2 window, stride 2, padding 1: each
    // output window holds exactly one real element, divided by 4.
    let input = f32_storage(vec![1.0, 2.0, 3.0, 4.0]);
    let il = Layout::contiguous([1, 1, 2, 2]).unwrap();
    let p = params((2, 2), (2, 2), (1, 1), (1, 1));
    let out = conv(ConvOp::AvgPool2d, &[View::new(&input, &il)], &p).unwrap();
    assert_eq!(as_f32(&out), vec![0.25, 0.5, 0.75, 1.0]);
}

#[test]
fn pooling_ignores_dilation() {
    let input = f32_storage((1..=16).map(|v| v as f32).collect());
    let il = Layout::contiguous([1, 1, 4, 4]).unwrap();
    let dense = params((2, 2), (2, 2), (0, 0), (1, 1));
    let dilated = params((2, 2), (2, 2), (0, 0), (3, 3));
    let a = conv(ConvOp::MaxPool2d, &[View::new(&input, &il)], &dense).unwrap();
    let b = conv(ConvOp::MaxPool2d, &[View::new(&input, &il)], &dilated).unwrap();
    assert_eq!(as_f32(&a), as_f32(&b));
}

// ----- cross-check against a naive reference ------------------------

/// Straightforward f64 reference convolution over contiguous NCHW/OIHW
/// buffers, written independently of the kernel above.
#[allow(clippy::too_many_arguments)]
fn reference_conv2d(
    input: &[f32],
    weight: &[f32],
    dims: [usize; 4],
    wdims: [usize; 4],
    out: [usize; 2],
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
) -> Vec<f32> {
    let [n, ic, h, w] = dims;
    let [oc, _, kh, kw] = wdims;
    let mut result = Vec::new();
    for b in 0..n {
        for o in 0..oc {
            for oh in 0..out[0] {
                for ow in 0..out[1] {
                    let mut acc = 0.0f64;
                    for c in 0..ic {
                        for i in 0..kh {
                            for j in 0..kw {
                                let sh =
                                    (oh * stride.0 + i * dilation.0) as isize - padding.0 as isize;
                                let sw =
                                    (ow * stride.1 + j * dilation.1) as isize - padding.1 as isize;
                                if sh < 0 || sw < 0 {
                                    continue;
                                }
                                let (sh, sw) = (sh as usize, sw as usize);
                                if sh >= h || sw >= w {
                                    continue;
                                }
                                let x = input[((b * ic + c) * h + sh) * w + sw] as f64;
                                let k = weight[((o * ic + c) * kh + i) * kw + j] as f64;
                                acc += x * k;
                            }
                        }
                    }
                    result.push(acc as f32);
                }
            }
        }
    }
    result
}

#[test]
fn conv2d_matches_naive_reference_over_a_parameter_grid() {
    let mut rng = Prng(0x5eed_1234_abcd_0001);
    for &stride in &[(1, 1), (2, 1), (2, 2)] {
        for &padding in &[(0, 0), (1, 0), (1, 1), (2, 2)] {
            for &dilation in &[(1, 1), (2, 1)] {
                let dims = [2usize, 3, 5, 6];
                let wdims = [4usize, 3, 2, 3];
                let p = params((wdims[2], wdims[3]), stride, padding, dilation);
                let geo = Conv2dGeometry::conv2d("conv2d", &dims, &wdims, &p).unwrap();
                let input = rng.values(dims.iter().product());
                let weight = rng.values(wdims.iter().product());
                let expected = reference_conv2d(
                    &input,
                    &weight,
                    dims,
                    wdims,
                    [geo.out_h, geo.out_w],
                    stride,
                    padding,
                    dilation,
                );
                let si = f32_storage(input);
                let sw = f32_storage(weight);
                let il = Layout::contiguous(dims).unwrap();
                let wl = Layout::contiguous(wdims).unwrap();
                let got = as_f32(
                    &conv(
                        ConvOp::Conv2d,
                        &[View::new(&si, &il), View::new(&sw, &wl)],
                        &p,
                    )
                    .unwrap(),
                );
                assert_eq!(got.len(), expected.len());
                for (g, e) in got.iter().zip(expected.iter()) {
                    assert!(
                        (g - e).abs() < 1e-3,
                        "conv mismatch {g} vs {e} at {stride:?}/{padding:?}/{dilation:?}"
                    );
                }
            }
        }
    }
}

// ----- gradients ----------------------------------------------------

/// Central finite differences of `⟨grad, forward(values)⟩` with respect to
/// `values`, where `forward` re-runs a kernel with a perturbed buffer.
fn finite_difference(
    values: &[f32],
    grad: &[f32],
    forward: impl Fn(&[f32]) -> Vec<f32>,
) -> Vec<f32> {
    const EPS: f32 = 1e-2;
    let dot = |v: &[f32]| -> f64 {
        forward(v)
            .iter()
            .zip(grad)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum()
    };
    (0..values.len())
        .map(|i| {
            let mut plus = values.to_vec();
            plus[i] += EPS;
            let mut minus = values.to_vec();
            minus[i] -= EPS;
            ((dot(&plus) - dot(&minus)) / f64::from(2.0 * EPS)) as f32
        })
        .collect()
}

fn assert_close(got: &[f32], expected: &[f32], what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: length");
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        assert!(
            (g - e).abs() <= 1e-2 * (1.0 + e.abs()),
            "{what}: element {i}: {g} vs {e}"
        );
    }
}

#[test]
fn conv2d_input_grad_hand_computed() {
    // 1x1x3x3 input, 1x1x2x2 weight [[1,2],[3,4]], unit cotangent: each
    // input position accumulates the weights of every window reading it.
    let grad = f32_storage(vec![1.0; 4]);
    let weight = f32_storage(vec![1.0, 2.0, 3.0, 4.0]);
    let gl = Layout::contiguous([1, 1, 2, 2]).unwrap();
    let wl = Layout::contiguous([1, 1, 2, 2]).unwrap();
    let geo = Conv2dGeometry::conv2d("conv2d", &[1, 1, 3, 3], &[1, 1, 2, 2], &IDENTITY).unwrap();
    let got =
        as_f32(&conv2d_input_grad(View::new(&grad, &gl), View::new(&weight, &wl), &geo).unwrap());
    assert_eq!(got, vec![1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0]);
}

#[test]
fn conv2d_weight_grad_hand_computed() {
    // 1x1x3x3 input 1..9, unit cotangent over the 2x2 output: each weight
    // gets the sum of the inputs it multiplies.
    let grad = f32_storage(vec![1.0; 4]);
    let input = f32_storage((1..=9).map(|v| v as f32).collect());
    let gl = Layout::contiguous([1, 1, 2, 2]).unwrap();
    let il = Layout::contiguous([1, 1, 3, 3]).unwrap();
    let geo = Conv2dGeometry::conv2d("conv2d", &[1, 1, 3, 3], &[1, 1, 2, 2], &IDENTITY).unwrap();
    let got =
        as_f32(&conv2d_weight_grad(View::new(&grad, &gl), View::new(&input, &il), &geo).unwrap());
    // w00 sees 1,2,4,5 = 12 ; w01: 2,3,5,6 = 16 ; w10: 4,5,7,8 = 24 ;
    // w11: 5,6,8,9 = 28.
    assert_eq!(got, vec![12.0, 16.0, 24.0, 28.0]);
}

#[test]
fn conv2d_gradients_match_finite_differences() {
    let mut rng = Prng(0x5eed_1234_abcd_0002);
    for &stride in &[(1, 1), (2, 2)] {
        for &padding in &[(0, 0), (1, 1)] {
            for &dilation in &[(1, 1), (2, 1)] {
                let dims = [2usize, 2, 5, 5];
                let wdims = [3usize, 2, 2, 3];
                let p = params((wdims[2], wdims[3]), stride, padding, dilation);
                let geo = Conv2dGeometry::conv2d("conv2d", &dims, &wdims, &p).unwrap();
                let input = rng.values(dims.iter().product());
                let weight = rng.values(wdims.iter().product());
                let out_len: usize = geo.output_dims().iter().product();
                let cotangent = rng.values(out_len);

                let il = Layout::contiguous(dims).unwrap();
                let wl = Layout::contiguous(wdims).unwrap();
                let gl = Layout::contiguous(geo.output_dims()).unwrap();
                let sg = f32_storage(cotangent.clone());
                let si = f32_storage(input.clone());
                let sw = f32_storage(weight.clone());

                let analytic = as_f32(
                    &conv2d_input_grad(View::new(&sg, &gl), View::new(&sw, &wl), &geo).unwrap(),
                );
                let numeric = finite_difference(&input, &cotangent, |x| {
                    let sx = f32_storage(x.to_vec());
                    as_f32(
                        &conv(
                            ConvOp::Conv2d,
                            &[View::new(&sx, &il), View::new(&sw, &wl)],
                            &p,
                        )
                        .unwrap(),
                    )
                });
                assert_close(&analytic, &numeric, "conv2d input grad");

                let analytic = as_f32(
                    &conv2d_weight_grad(View::new(&sg, &gl), View::new(&si, &il), &geo).unwrap(),
                );
                let numeric = finite_difference(&weight, &cotangent, |w| {
                    let sww = f32_storage(w.to_vec());
                    as_f32(
                        &conv(
                            ConvOp::Conv2d,
                            &[View::new(&si, &il), View::new(&sww, &wl)],
                            &p,
                        )
                        .unwrap(),
                    )
                });
                assert_close(&analytic, &numeric, "conv2d weight grad");
            }
        }
    }
}

#[test]
fn max_pool2d_backward_routes_to_the_winner() {
    // 2x2 window over 1x1x4x4 = 1..16: the winners are 6, 8, 14, 16.
    let input = f32_storage((1..=16).map(|v| v as f32).collect());
    let grad = f32_storage(vec![1.0, 2.0, 3.0, 4.0]);
    let il = Layout::contiguous([1, 1, 4, 4]).unwrap();
    let gl = Layout::contiguous([1, 1, 2, 2]).unwrap();
    let p = params((2, 2), (2, 2), (0, 0), (1, 1));
    let geo = Conv2dGeometry::pool("max_pool2d", &[1, 1, 4, 4], &p).unwrap();
    let got =
        as_f32(&max_pool2d_backward(View::new(&grad, &gl), View::new(&input, &il), &geo).unwrap());
    let mut expected = vec![0.0f32; 16];
    expected[5] = 1.0; // value 6
    expected[7] = 2.0; // value 8
    expected[13] = 3.0; // value 14
    expected[15] = 4.0; // value 16
    assert_eq!(got, expected);
}

#[test]
fn max_pool2d_backward_breaks_ties_by_first_position() {
    let input = f32_storage(vec![5.0, 5.0, 5.0, 5.0]);
    let grad = f32_storage(vec![7.0]);
    let il = Layout::contiguous([1, 1, 2, 2]).unwrap();
    let gl = Layout::contiguous([1, 1, 1, 1]).unwrap();
    let p = params((2, 2), (2, 2), (0, 0), (1, 1));
    let geo = Conv2dGeometry::pool("max_pool2d", &[1, 1, 2, 2], &p).unwrap();
    let got =
        as_f32(&max_pool2d_backward(View::new(&grad, &gl), View::new(&input, &il), &geo).unwrap());
    assert_eq!(got, vec![7.0, 0.0, 0.0, 0.0]);
}

#[test]
fn avg_pool2d_backward_spreads_over_the_window() {
    let grad = f32_storage(vec![4.0]);
    let gl = Layout::contiguous([1, 1, 1, 1]).unwrap();
    let p = params((2, 2), (2, 2), (0, 0), (1, 1));
    let geo = Conv2dGeometry::pool("avg_pool2d", &[1, 1, 2, 2], &p).unwrap();
    let got = as_f32(&avg_pool2d_backward(View::new(&grad, &gl), &geo).unwrap());
    assert_eq!(got, vec![1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn pool_gradients_match_finite_differences() {
    let mut rng = Prng(0x5eed_1234_abcd_0003);
    for &stride in &[(1, 1), (2, 2)] {
        for &padding in &[(0, 0), (1, 1)] {
            let dims = [2usize, 2, 5, 5];
            let p = params((2, 2), stride, padding, (1, 1));
            let il = Layout::contiguous(dims).unwrap();
            // Distinct, well-separated values keep the window maximum
            // unambiguous under the finite-difference perturbation.
            let input: Vec<f32> = (0..dims.iter().product::<usize>())
                .map(|i| i as f32 * 0.5 - 5.0)
                .collect();
            let si = f32_storage(input.clone());

            let geo = Conv2dGeometry::pool("max_pool2d", &dims, &p).unwrap();
            let gl = Layout::contiguous(geo.output_dims()).unwrap();
            let out_len: usize = geo.output_dims().iter().product();
            let cotangent = rng.values(out_len);
            let sg = f32_storage(cotangent.clone());

            let analytic = as_f32(
                &max_pool2d_backward(View::new(&sg, &gl), View::new(&si, &il), &geo).unwrap(),
            );
            let numeric = finite_difference(&input, &cotangent, |x| {
                let sx = f32_storage(x.to_vec());
                as_f32(&conv(ConvOp::MaxPool2d, &[View::new(&sx, &il)], &p).unwrap())
            });
            assert_close(&analytic, &numeric, "max_pool2d grad");

            let geo = Conv2dGeometry::pool("avg_pool2d", &dims, &p).unwrap();
            let analytic = as_f32(&avg_pool2d_backward(View::new(&sg, &gl), &geo).unwrap());
            let numeric = finite_difference(&input, &cotangent, |x| {
                let sx = f32_storage(x.to_vec());
                as_f32(&conv(ConvOp::AvgPool2d, &[View::new(&sx, &il)], &p).unwrap())
            });
            assert_close(&analytic, &numeric, "avg_pool2d grad");
        }
    }
}

#[test]
fn bool_is_unsupported() {
    let input = Storage::Cpu(CpuStorage::Bool(Arc::new(vec![true; 4])));
    let il = Layout::contiguous([1, 1, 2, 2]).unwrap();
    let p = params((2, 2), (2, 2), (0, 0), (1, 1));
    assert!(matches!(
        conv(ConvOp::MaxPool2d, &[View::new(&input, &il)], &p),
        Err(Error::Unsupported {
            op: "max_pool2d",
            dtype: DType::Bool,
            ..
        })
    ));
}

#[test]
fn dtype_mismatch_is_loud() {
    let input = f32_storage(vec![1.0; 9]);
    let weight = Storage::Cpu(CpuStorage::I64(Arc::new(vec![1; 4])));
    let il = Layout::contiguous([1, 1, 3, 3]).unwrap();
    let wl = Layout::contiguous([1, 1, 2, 2]).unwrap();
    assert!(matches!(
        conv(
            ConvOp::Conv2d,
            &[View::new(&input, &il), View::new(&weight, &wl)],
            &IDENTITY
        ),
        Err(Error::DTypeMismatch { op: "conv2d", .. })
    ));
}

#[test]
fn operand_arity_is_checked() {
    let input = f32_storage(vec![1.0; 9]);
    let il = Layout::contiguous([1, 1, 3, 3]).unwrap();
    assert!(matches!(
        conv(ConvOp::Conv2d, &[View::new(&input, &il)], &IDENTITY),
        Err(Error::InvalidArg { op: "conv2d", .. })
    ));
}

#[test]
fn gradient_kernels_check_operand_shapes() {
    let geo = Conv2dGeometry::conv2d("conv2d", &[1, 1, 3, 3], &[1, 1, 2, 2], &IDENTITY).unwrap();
    let grad = f32_storage(vec![1.0; 9]);
    let weight = f32_storage(vec![1.0; 4]);
    let bad = Layout::contiguous([1, 1, 3, 3]).unwrap();
    let wl = Layout::contiguous([1, 1, 2, 2]).unwrap();
    assert!(matches!(
        conv2d_input_grad(View::new(&grad, &bad), View::new(&weight, &wl), &geo),
        Err(Error::ShapeMismatch { op: "conv2d", .. })
    ));
}

#[test]
fn i64_pooling_is_supported() {
    let input = Storage::Cpu(CpuStorage::I64(Arc::new(vec![1, 5, 3, 2])));
    let il = Layout::contiguous([1, 1, 2, 2]).unwrap();
    let p = params((2, 2), (2, 2), (0, 0), (1, 1));
    let out = conv(ConvOp::MaxPool2d, &[View::new(&input, &il)], &p).unwrap();
    match &out {
        Storage::Cpu(CpuStorage::I64(v)) => assert_eq!(v.as_ref(), &vec![5]),
        _ => panic!("expected i64 storage"),
    }
}
