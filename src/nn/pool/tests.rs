//! [`MaxPool2d`]/[`AvgPool2d`] defaults and shapes, [`Flatten`]'s reshape,
//! and [`Identity`]'s pass-through.

use super::*;
use crate::device::Device;
use crate::testing::check_grad;

const CPU: Device = Device::Cpu;

fn t(data: &[f32], shape: impl Into<crate::shape::Shape>) -> Tensor {
    Tensor::from_vec(data.to_vec(), shape, &CPU).unwrap()
}

#[test]
fn max_pool2d_defaults_stride_to_the_kernel() {
    let mut pool = MaxPool2d::new((2, 2));
    let x = t(
        &(1..=16).map(|v| v as f32).collect::<Vec<_>>(),
        [1, 1, 4, 4],
    );
    let y = pool.forward(&x, Mode::EVAL).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    assert_eq!(y.to_vec::<f32>().unwrap(), vec![6.0, 8.0, 14.0, 16.0]);
}

#[test]
fn max_pool2d_with_stride_and_padding_override_the_defaults() {
    let pool = MaxPool2d::new((2, 2))
        .with_stride((1, 1))
        .with_padding((1, 1));
    assert_eq!(
        (pool.kernel(), pool.stride(), pool.padding()),
        ((2, 2), (1, 1), (1, 1))
    );

    // An overlapping stride is the observable half: a 2×2 window stepped by
    // one over 1..16 leaves a 3×3 map whose maxima are each window's
    // bottom-right corner.
    let mut overlapping = MaxPool2d::new((2, 2)).with_stride((1, 1));
    let x = t(
        &(1..=16).map(|v| v as f32).collect::<Vec<_>>(),
        [1, 1, 4, 4],
    );
    let y = overlapping.forward(&x, Mode::EVAL).unwrap();
    assert_eq!(y.dims(), &[1, 1, 3, 3]);
    assert_eq!(
        y.to_vec::<f32>().unwrap(),
        vec![6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 14.0, 15.0, 16.0]
    );
}

#[test]
fn avg_pool2d_divides_by_the_full_window_including_padding() {
    // `count_include_pad = true`: over the padded window grid of [[1, 2],
    // [3, 4]] every corner sum is divided by 4, not by the one real element
    // the window covers.
    let mut pool = AvgPool2d::new((2, 2)).with_padding((1, 1));
    let x = t(&[1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]);
    let y = pool.forward(&x, Mode::EVAL).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    assert_eq!(y.to_vec::<f32>().unwrap(), vec![0.25, 0.5, 0.75, 1.0]);
}

#[test]
fn avg_pool2d_defaults_stride_to_the_kernel() {
    let mut pool = AvgPool2d::new((2, 2));
    let x = t(
        &(1..=16).map(|v| v as f32).collect::<Vec<_>>(),
        [1, 1, 4, 4],
    );
    let y = pool.forward(&x, Mode::EVAL).unwrap();
    assert_eq!(y.dims(), &[1, 1, 2, 2]);
    // Window means of a row-major 1..16 4x4 grid, 2x2 non-overlapping tiles.
    assert_eq!(y.to_vec::<f32>().unwrap(), vec![3.5, 5.5, 11.5, 13.5]);
}

#[test]
fn flatten_default_start_dim_keeps_the_batch_axis() {
    let mut flatten = Flatten::new();
    let x = t(&(0..24).map(|v| v as f32).collect::<Vec<_>>(), [2, 3, 4]);
    let y = flatten.forward(&x, Mode::EVAL).unwrap();
    assert_eq!(y.dims(), &[2, 12]);
    assert_eq!(y.to_vec::<f32>().unwrap(), x.to_vec::<f32>().unwrap());
}

#[test]
fn flatten_with_start_dim_flattens_from_a_different_axis() {
    let mut flatten = Flatten::new().with_start_dim(2);
    let x = t(&(0..24).map(|v| v as f32).collect::<Vec<_>>(), [2, 3, 4]);
    assert_eq!(flatten.forward(&x, Mode::EVAL).unwrap().dims(), &[2, 3, 4]);

    let mut flatten_all = Flatten::new().with_start_dim(0);
    assert_eq!(flatten_all.forward(&x, Mode::EVAL).unwrap().dims(), &[24]);
}

#[test]
fn flatten_rejects_an_out_of_range_start_dim() {
    let mut flatten = Flatten::new().with_start_dim(5);
    let x = t(&[1.0, 2.0], [2]);
    assert!(flatten.forward(&x, Mode::EVAL).is_err());
}

#[test]
fn identity_returns_the_input_unchanged() {
    let mut id = Identity;
    let x = t(&[1.0, 2.0, 3.0], [3]);
    let y = id.forward(&x, Mode::EVAL).unwrap();
    assert_eq!(y.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    assert_eq!(y.dims(), x.dims());
}

#[test]
fn gradients_flow_through_max_and_avg_pool_layers() {
    let x = t(
        &(0..16).map(|v| v as f32 * 0.1).collect::<Vec<_>>(),
        [1, 1, 4, 4],
    );
    check_grad(
        |xs| {
            let mut pool = MaxPool2d::new((2, 2));
            pool.forward(&xs[0], Mode::EVAL)?.sum_all()
        },
        std::slice::from_ref(&x),
        1e-3,
        1e-2,
    )
    .unwrap();
    check_grad(
        |xs| {
            let mut pool = AvgPool2d::new((2, 2));
            pool.forward(&xs[0], Mode::EVAL)?.sum_all()
        },
        &[x],
        1e-3,
        1e-2,
    )
    .unwrap();
}

#[test]
fn gradients_flow_through_flatten() {
    let x = t(
        &(0..12).map(|v| v as f32 * 0.1).collect::<Vec<_>>(),
        [2, 2, 3],
    );
    check_grad(
        |xs| {
            let mut f = Flatten::new();
            f.forward(&xs[0], Mode::EVAL)?.sum_all()
        },
        &[x],
        1e-3,
        1e-2,
    )
    .unwrap();
}
