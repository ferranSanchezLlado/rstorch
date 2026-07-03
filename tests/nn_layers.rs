//! Behavior of neural-network layers (`LayerNorm`, `Dropout`).

use rstorch::prelude::*;
use rstorch::{Error, Optimizer, ShapeError, Tensor};

#[derive(Debug)]
struct Batch;

#[test]
fn layer_norm_centers_each_row() {
    let norm = LayerNorm::<3>::new(1e-5).unwrap();
    let input = Tensor::<D2<Sym<Batch>, C<3>>>::from_vec_with_shape(
        vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
        [2, 3],
    )
    .unwrap();
    let mut ctx = TrainContext::eval();

    let row_means = norm
        .forward(&input, &mut ctx)
        .unwrap()
        .mean_last()
        .unwrap()
        .to_vec()
        .unwrap();

    assert_close(row_means[0], 0.0, 1e-5);
    assert_close(row_means[1], 0.0, 1e-5);
}

#[test]
fn layer_norm_gradient_matches_finite_difference() {
    let input_data = vec![1.0, 2.0, 4.0, -1.0, 0.5, 3.0];
    let probe = Tensor2D::<2, 3, f64>::from_vec(vec![0.5, -0.2, 0.7, -0.4, 0.3, 0.1]).unwrap();
    let norm = LayerNorm::<3, f64>::new(1e-5).unwrap();
    let input = Tensor2D::<2, 3, f64>::from_vec(input_data.clone())
        .unwrap()
        .with_requires_grad(true);
    let mut ctx = TrainContext::eval();

    norm.forward(&input, &mut ctx)
        .unwrap()
        .mul(&probe)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();

    let grad = input.grad().unwrap().to_vec().unwrap();
    for (idx, &analytic) in grad.iter().enumerate() {
        let numerical = finite_difference(&input_data, idx, 1e-6, |values| {
            let mut ctx = TrainContext::eval();
            LayerNorm::<3, f64>::new(1e-5)
                .unwrap()
                .forward(
                    &Tensor2D::<2, 3, f64>::from_vec(values.to_vec()).unwrap(),
                    &mut ctx,
                )
                .unwrap()
                .mul(&probe)
                .unwrap()
                .sum()
                .unwrap()
                .to_vec()
                .unwrap()[0]
        });
        assert_close_f64(analytic, numerical, 1e-5);
    }
}

#[test]
fn dropout_scales_in_training_and_is_identity_in_eval() {
    let dropout = Dropout::new(0.5);
    let mut train_ctx = TrainContext::training(7);
    let dropped = dropout
        .forward(&Tensor1D::<4>::ones().unwrap(), &mut train_ctx)
        .unwrap()
        .to_vec()
        .unwrap();
    assert!(dropped.iter().all(|&value| value == 0.0 || value == 2.0));

    let mut eval_ctx = TrainContext::eval();
    assert_eq!(
        dropout
            .forward(&Tensor1D::<2>::ones().unwrap(), &mut eval_ctx)
            .unwrap()
            .to_vec()
            .unwrap(),
        vec![1.0, 1.0]
    );
}

#[test]
fn conv2d_forward_and_backward_cover_input_weight_and_bias() {
    let input =
        Tensor4D::<1, 1, 3, 3, f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap()
            .with_requires_grad(true);
    let weight = Tensor4D::<1, 1, 2, 2, f64>::from_vec(vec![1.0, 1.0, 1.0, 1.0])
        .unwrap()
        .with_requires_grad(true);
    let bias = Tensor1D::<1, f64>::from_vec(vec![0.5])
        .unwrap()
        .with_requires_grad(true);

    let out = input
        .conv2d::<C<1>, C<2>, C<2>, 2, 2>(&weight, Conv2dOptions::default())
        .unwrap()
        .add_channel_dim(&bias)
        .unwrap();

    assert_eq!(out.shape().dims(), &[1, 1, 2, 2]);
    assert_close_vec_f64(&out.to_vec().unwrap(), &[12.5, 16.5, 24.5, 28.5], 1e-12);

    out.sum().unwrap().backward().unwrap();
    assert_close_vec_f64(
        &input.grad().unwrap().to_vec().unwrap(),
        &[1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0],
        1e-12,
    );
    assert_close_vec_f64(
        &weight.grad().unwrap().to_vec().unwrap(),
        &[12.0, 16.0, 24.0, 28.0],
        1e-12,
    );
    assert_close_vec_f64(&bias.grad().unwrap().to_vec().unwrap(), &[4.0], 1e-12);
}

#[test]
fn conv2d_validates_stride_padding_dilation_kernel_and_output_shape() {
    let input = Tensor4D::<1, 1, 4, 4, f64>::from_vec((1..=16).map(|value| value as f64).collect())
        .unwrap();
    let weight = Tensor4D::<1, 1, 2, 2, f64>::from_vec(vec![1.0; 4]).unwrap();
    let options = Conv2dOptions::default()
        .with_stride(2, 2)
        .with_padding(1, 1)
        .with_dilation(2, 2);

    let out = input
        .conv2d::<C<1>, C<2>, C<2>, 2, 2>(&weight, options)
        .unwrap();
    assert_close_vec_f64(&out.to_vec().unwrap(), &[6.0, 14.0, 20.0, 44.0], 1e-12);

    let err = input
        .conv2d::<C<1>, C<2>, C<2>, 3, 2>(&weight, options)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Shape(ShapeError::DimMismatch {
            op: "conv2d",
            operand: 2,
            axis: 2,
            expected: 2,
            found: 3,
        })
    ));

    let err = input
        .conv2d::<C<1>, C<2>, C<2>, 2, 2>(&weight, Conv2dOptions::default().with_stride(0, 1))
        .unwrap_err();
    assert_invalid_spatial(err, "conv2d", "stride");

    let err = input
        .conv2d::<C<1>, C<2>, C<2>, 2, 2>(&weight, Conv2dOptions::default().with_dilation(0, 1))
        .unwrap_err();
    assert_invalid_spatial(err, "conv2d", "dilation");

    let zero_kernel = Tensor4D::<1, 1, 0, 1, f64>::from_vec(Vec::new()).unwrap();
    let err = input
        .conv2d::<C<1>, C<0>, C<1>, 1, 4>(&zero_kernel, Conv2dOptions::default())
        .unwrap_err();
    assert_invalid_spatial(err, "conv2d", "kernel");

    let big_kernel = Tensor4D::<1, 1, 5, 5, f64>::from_vec(vec![1.0; 25]).unwrap();
    let err = input
        .conv2d::<C<1>, C<5>, C<5>, 1, 1>(&big_kernel, Conv2dOptions::default())
        .unwrap_err();
    assert_invalid_spatial(err, "conv2d", "padding");

    let err = input
        .max_pool2d::<1, 1>(Pool2dOptions::new(2, 2).with_stride(0, 1))
        .unwrap_err();
    assert_invalid_spatial(err, "max_pool2d", "stride");
}

#[test]
fn pooling_forward_and_backward_define_max_ties_and_average_counts() {
    let max_input = Tensor4D::<1, 1, 2, 2, f64>::from_vec(vec![1.0, 2.0, 2.0, 0.0])
        .unwrap()
        .with_requires_grad(true);
    let max_out = max_input
        .max_pool2d::<1, 1>(Pool2dOptions::new(2, 2))
        .unwrap();
    assert_eq!(max_out.to_vec().unwrap(), vec![2.0]);
    max_out.sum().unwrap().backward().unwrap();
    assert_close_vec_f64(
        &max_input.grad().unwrap().to_vec().unwrap(),
        &[0.0, 0.5, 0.5, 0.0],
        1e-12,
    );

    let avg_input = Tensor4D::<1, 1, 2, 2, f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0])
        .unwrap()
        .with_requires_grad(true);
    let avg_out = avg_input
        .avg_pool2d::<1, 1>(Pool2dOptions::new(2, 2))
        .unwrap();
    assert_eq!(avg_out.to_vec().unwrap(), vec![2.5]);
    avg_out.sum().unwrap().backward().unwrap();
    assert_close_vec_f64(
        &avg_input.grad().unwrap().to_vec().unwrap(),
        &[0.25, 0.25, 0.25, 0.25],
        1e-12,
    );
}

#[test]
fn pad2d_uses_nchw_layout() {
    let input = Tensor4D::<1, 1, 1, 2>::from_vec(vec![1.0, 2.0])
        .unwrap()
        .with_requires_grad(true);
    let padded = input.pad2d::<3, 4>(Padding2d::new(1, 1)).unwrap();

    assert_eq!(padded.shape().dims(), &[1, 1, 3, 4]);
    assert_eq!(
        padded.to_vec().unwrap(),
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );

    padded.sum().unwrap().backward().unwrap();
    assert_eq!(input.grad().unwrap().to_vec().unwrap(), vec![1.0, 1.0]);
}

#[test]
fn visual_modules_compose_and_expose_parameters() {
    let model = Sequential::new(
        Conv2d::<1, 2, 2, 2, 1, 1>::zeros(Conv2dOptions::default()).unwrap(),
        Flatten::<2>,
    );
    let input = Tensor::<D4<Sym<Batch>, C<1>, C<2>, C<2>>>::from_vec_with_shape(
        vec![1.0, 2.0, 3.0, 4.0],
        [1, 1, 2, 2],
    )
    .unwrap();
    let mut ctx = TrainContext::eval();

    let out: Tensor<D2<Sym<Batch>, C<2>>> = model.forward(&input, &mut ctx).unwrap();
    assert_eq!(out.shape().dims(), &[1, 2]);

    let mut names = Vec::new();
    model.visit_parameters("cnn", &mut |name, _| names.push(name.to_owned()));
    assert_eq!(names, vec!["cnn.0.weight", "cnn.0.bias"]);
}

#[test]
fn image_collator_and_synthetic_cnn_training_loss_decrease() {
    let (images, labels): (ImageBatch<1, 4, 4>, Vec<usize>) = images::<1, 4, 4>()
        .collate(vec![
            (vec![0.0; 16], 0),
            (
                vec![
                    1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                ],
                1,
            ),
            (vec![0.0; 16], 0),
            (
                vec![
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                ],
                1,
            ),
        ])
        .unwrap();
    let normalized = normalize_image_sample::<1, 1, 1, f32>(vec![2.0], [1.0], [2.0]).unwrap();
    assert_eq!(normalized, vec![0.5]);

    let mut model: TinySpatialCnn = Sequential::new(
        Conv2d::<1, 2, 2, 2, 3, 3>::zeros(Conv2dOptions::default()).unwrap(),
        AvgPool2d::<2, 2, 1, 1>::new(),
    )
    .add_module(Flatten::<2>);
    let mut opt = Sgd::new(0.5);
    let first = cnn_loss(&model, &images, &labels);

    for _ in 0..20 {
        let mut params = Vec::new();
        model.parameters(&mut params);
        opt.zero_grad(&params);
        drop(params);

        let loss = cnn_loss_tensor(&model, &images, &labels);
        loss.backward().unwrap();
        let mut params = Vec::new();
        model.parameters_mut(&mut params);
        opt.step(&mut params).unwrap();
    }

    let last = cnn_loss(&model, &images, &labels);
    assert!(last < first, "expected {last} < {first}");
}

type TinySpatialCnn = Sequential<
    (
        (Conv2d<1, 2, 2, 2, 3, 3>, AvgPool2d<2, 2, 1, 1>),
        Flatten<2>,
    ),
    ImageBatch<1, 4, 4>,
>;

fn assert_close(actual: f32, expected: f32, tol: f32) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {actual} within {tol} of {expected}"
    );
}

fn assert_close_f64(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {actual} within {tol} of {expected}"
    );
}

fn assert_close_vec_f64(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert_close_f64(actual, expected, tol);
    }
}

fn assert_invalid_spatial(err: Error, expected_op: &'static str, expected_param: &'static str) {
    assert!(matches!(
        err,
        Error::Shape(ShapeError::InvalidSpatialParam { op, param, .. })
            if op == expected_op && param == expected_param
    ));
}

fn cnn_loss(model: &TinySpatialCnn, images: &ImageBatch<1, 4, 4>, labels: &[usize]) -> f32 {
    cnn_loss_tensor(model, images, labels).to_vec().unwrap()[0]
}

fn cnn_loss_tensor(
    model: &TinySpatialCnn,
    images: &ImageBatch<1, 4, 4>,
    labels: &[usize],
) -> Scalar {
    let mut ctx = TrainContext::eval();
    model
        .forward(images, &mut ctx)
        .unwrap()
        .cross_entropy(labels)
        .unwrap()
}

fn finite_difference(values: &[f64], idx: usize, eps: f64, f: impl Fn(&[f64]) -> f64) -> f64 {
    let mut plus = values.to_vec();
    plus[idx] += eps;
    let mut minus = values.to_vec();
    minus[idx] -= eps;
    (f(&plus) - f(&minus)) / (2.0 * eps)
}
