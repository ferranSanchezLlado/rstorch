//! [`Conv2d`] shapes, initialization, bias, and gradients.

use super::*;
use crate::device::Device;
use crate::nn::{self, ModuleExt};
use crate::testing::check_grad;

const CPU: Device = Device::Cpu;

fn t(data: &[f32], shape: impl Into<crate::shape::Shape>) -> Tensor {
    Tensor::from_vec(data.to_vec(), shape, &CPU).unwrap()
}

#[test]
fn parameter_paths_and_shapes_are_the_persistence_contract() {
    let conv = Conv2d::new(2, 4, (3, 3), &CPU, &mut Rng::seed(0)).unwrap();
    let state = conv.state_dict().unwrap();
    assert_eq!(state.keys().collect::<Vec<_>>(), ["bias", "weight"]);
    assert_eq!(state["weight"].dims(), &[4, 2, 3, 3]);
    assert_eq!(state["bias"].dims(), &[4]);
    assert_eq!(conv.num_params(), 4 * 2 * 3 * 3 + 4);
    assert_eq!((conv.in_channels(), conv.out_channels()), (2, 4));
    assert_eq!(conv.kernel(), (3, 3));
    assert_eq!(
        (conv.stride(), conv.padding(), conv.dilation()),
        ((1, 1), (0, 0), (1, 1))
    );
    assert!(conv.bias().is_some());
}

#[test]
fn without_bias_drops_the_bias_leaf() {
    let conv = Conv2d::new(1, 2, (3, 3), &CPU, &mut Rng::seed(0))
        .unwrap()
        .with_padding((1, 1))
        .without_bias();
    assert!(conv.bias().is_none());
    assert_eq!(
        conv.state_dict().unwrap().keys().collect::<Vec<_>>(),
        ["weight"]
    );
}

#[test]
fn forward_computes_the_convolution_by_hand() {
    let mut conv = Conv2d::new(1, 1, (2, 2), &CPU, &mut Rng::seed(0)).unwrap();
    let mut state = conv.state_dict().unwrap();
    state
        .insert("weight".to_string(), t(&[1.0, 1.0, 1.0, 1.0], [1, 1, 2, 2]))
        .unwrap();
    state.insert("bias".to_string(), t(&[10.0], [1])).unwrap();
    conv.load_state_dict(&state).unwrap();

    // 1..9 in a 3×3 image under a 2×2 all-ones kernel, stride 1, no padding:
    // the four windows sum to 12, 16, 24, 28, and the bias adds 10.
    let x = t(&(1..=9).map(|v| v as f32).collect::<Vec<_>>(), [1, 1, 3, 3]);
    let y = conv.forward(&x, nn::Mode::EVAL).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    assert_eq!(y.to_vec::<f32>().unwrap(), vec![22.0, 26.0, 34.0, 38.0]);
}

#[test]
fn with_init_keeps_the_bias_on_the_weight_dtype_and_device() {
    // The bias is built from the weight, so a non-`F32` `with_init` weight
    // still yields a layer whose `forward` runs instead of failing on the
    // broadcast add.
    let mut conv = Conv2d::with_init(1, 2, (2, 2), &CPU, |shape, device| {
        Tensor::ones(shape.to_vec(), DType::F64, device)
    })
    .unwrap();
    assert_eq!(conv.bias().unwrap().value().dtype(), DType::F64);
    let x = Tensor::ones([1, 1, 2, 2], DType::F64, &CPU).unwrap();
    let y = conv.forward(&x, nn::Mode::EVAL).unwrap();
    assert_eq!(y.to_vec::<f64>().unwrap(), vec![4.0, 4.0]);
}

#[test]
fn zero_dims_are_rejected() {
    assert!(Conv2d::new(0, 4, (3, 3), &CPU, &mut Rng::seed(0)).is_err());
    assert!(Conv2d::new(4, 0, (3, 3), &CPU, &mut Rng::seed(0)).is_err());
    assert!(Conv2d::new(4, 4, (0, 3), &CPU, &mut Rng::seed(0)).is_err());
}

#[test]
fn with_init_rejects_a_mismatched_weight_shape() {
    let err = Conv2d::with_init(1, 2, (3, 3), &CPU, |_, device| {
        Tensor::zeros([1, 1, 1, 1], DType::F32, device)
    })
    .unwrap_err();
    assert!(matches!(err, Error::ShapeMismatch { .. }));
}

#[test]
fn gradients_flow_through_weight_and_bias() {
    let mut conv = Conv2d::new(2, 1, (2, 2), &CPU, &mut Rng::seed(3)).unwrap();
    let x = t(
        &(0..8).map(|v| v as f32 * 0.1).collect::<Vec<_>>(),
        [1, 2, 2, 2],
    );
    let grads = conv
        .forward(&x, nn::Mode::TRAIN)
        .unwrap()
        .sum_all()
        .unwrap()
        .backward()
        .unwrap();
    assert_eq!(grads.len(), 2, "weight and bias");
    assert_eq!(grads.wrt(conv.weight()).unwrap().dims(), &[1, 2, 2, 2]);
    // Every output element receives the bias once, so ∂Σy/∂b is the output
    // element count — 1 here, for a single 1×1 spatial position.
    assert_eq!(
        grads
            .wrt(conv.bias().unwrap())
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        vec![1.0]
    );

    // The tensor-level composition the layer delegates to, checked against
    // finite differences.
    let weight = t(
        &(0..8).map(|v| v as f32 * 0.05).collect::<Vec<_>>(),
        [1, 2, 2, 2],
    );
    let bias = t(&[0.3], [1, 1, 1]);
    check_grad(
        |xs| {
            xs[0]
                .conv2d(&xs[1], (1, 1), (0, 0), (1, 1))?
                .add(&xs[2])?
                .sum_all()
        },
        &[x, weight, bias],
        1e-3,
        1e-2,
    )
    .unwrap();
}

#[test]
fn debug_reports_the_geometry() {
    let conv = Conv2d::new(3, 8, (3, 3), &CPU, &mut Rng::seed(0))
        .unwrap()
        .with_stride((2, 2))
        .with_padding((1, 1));
    let text = format!("{conv:?}");
    assert!(text.contains("3 -> 8"), "{text}");
    assert!(text.contains("stride (2, 2)"), "{text}");
    assert!(text.ends_with(", bias)"), "{text}");

    let bare = Conv2d::new(3, 8, (3, 3), &CPU, &mut Rng::seed(0))
        .unwrap()
        .without_bias();
    assert!(format!("{bare:?}").ends_with(", no bias)"), "{bare:?}");
}
