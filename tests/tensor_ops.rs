//! Forward and autograd behavior of the typed tensor op surface.

use rstorch::prelude::*;
use rstorch::{DTypeError, DTypeId, Error, ShapeError, Tensor};

#[test]
fn numeric_ops_are_differentiable() {
    let x = Tensor1D::<2>::from_vec(vec![-1.0, 0.5])
        .unwrap()
        .with_requires_grad(true);

    x.exp()
        .unwrap()
        .ln()
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();

    assert_close(x.grad().unwrap().to_vec().unwrap()[0], 1.0, 1e-6);
    assert_close(x.grad().unwrap().to_vec().unwrap()[1], 1.0, 1e-6);

    assert_eq!(x.neg().unwrap().to_vec().unwrap(), vec![1.0, -0.5]);
    assert_close(
        x.sigmoid().unwrap().to_vec().unwrap()[1],
        0.622_459_35,
        1e-6,
    );
    assert_close(x.tanh().unwrap().to_vec().unwrap()[1], 0.462_117_17, 1e-6);
}

#[test]
fn reductions_softmax_and_cross_entropy_work() {
    let logits = Tensor2D::<2, 3>::from_vec(vec![1.0, 2.0, 3.0, 1.0, 0.0, -1.0])
        .unwrap()
        .with_requires_grad(true);

    assert_eq!(
        logits.sum_leading().unwrap().to_vec().unwrap(),
        vec![2.0, 2.0, 2.0]
    );
    assert_eq!(
        logits.mean_last().unwrap().to_vec().unwrap(),
        vec![2.0, 0.0]
    );

    let softmax = logits.softmax_last().unwrap().to_vec().unwrap();
    assert_close(softmax[0] + softmax[1] + softmax[2], 1.0, 1e-6);
    assert_close(softmax[3] + softmax[4] + softmax[5], 1.0, 1e-6);

    assert_eq!(logits.argmax_last().unwrap(), vec![2, 0]);
    assert_eq!(logits.correct_count(&[2, 1]).unwrap(), 1);
    assert_close_f64(logits.accuracy(&[2, 0]).unwrap(), 1.0, 1e-12);

    let loss = logits.cross_entropy(&[2, 0]).unwrap();
    assert_close(loss.to_vec().unwrap()[0], 0.407_605_95, 1e-6);
    loss.backward().unwrap();

    let grad = logits.grad().unwrap().to_vec().unwrap();
    assert_close(grad[2], (softmax[2] - 1.0) / 2.0, 1e-6);
    assert_close(grad[3], (softmax[3] - 1.0) / 2.0, 1e-6);
}

#[test]
fn random_constructors_full_reductions_item_and_d2_cat_stack_work() {
    let mut rng_a = SmallRng::seed_from_u64(123);
    let mut rng_b = SmallRng::seed_from_u64(123);
    let rand_a = Tensor2D::<2, 2>::rand(&mut rng_a).unwrap();
    let rand_b = Tensor2D::<2, 2>::rand(&mut rng_b).unwrap();
    assert_eq!(rand_a.to_vec().unwrap(), rand_b.to_vec().unwrap());

    let normal = Tensor1D::<128>::randn(&mut rng_a).unwrap();
    assert_eq!(normal.shape().dims(), &[128]);
    let dynamic = Tensor::<D2<AnyDim, C<2>>>::rand_with_shape(&mut rng_a, [3, 2]).unwrap();
    assert_eq!(dynamic.shape().dims(), &[3, 2]);
    assert_eq!(
        Tensor1D::<4>::arange().unwrap().to_vec().unwrap(),
        vec![0.0, 1.0, 2.0, 3.0]
    );
    assert_eq!(
        Tensor2D::<2, 2>::full(3.0).unwrap().to_vec().unwrap(),
        vec![3.0; 4]
    );

    let values = Tensor2D::<2, 3>::from_vec(vec![1.0, -2.0, 3.0, 4.0, 5.0, -6.0]).unwrap();
    assert_eq!(
        values.abs().unwrap().to_vec().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    assert_eq!(values.min().unwrap().item().unwrap(), -6.0);
    assert_eq!(values.max().unwrap().item().unwrap(), 5.0);
    assert_close(values.mean().unwrap().item().unwrap(), 5.0 / 6.0, 1e-6);
    assert_close(
        values.var().unwrap().item().unwrap(),
        86.833_336 / 6.0,
        1e-5,
    );
    assert_eq!(values.var_last().unwrap().shape().dims(), &[2]);
    assert_eq!(values.std_last().unwrap().shape().dims(), &[2]);

    let lhs = Tensor2D::<2, 2>::from_vec(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let rhs_rows = Tensor2D::<1, 2>::from_vec(vec![5.0, 6.0]).unwrap();
    assert_eq!(
        lhs.cat_leading::<C<1>, 3>(&rhs_rows)
            .unwrap()
            .to_vec()
            .unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    let rhs_cols = Tensor2D::<2, 1>::from_vec(vec![7.0, 8.0]).unwrap();
    assert_eq!(
        lhs.cat_last::<C<1>, 3>(&rhs_cols)
            .unwrap()
            .to_vec()
            .unwrap(),
        vec![1.0, 2.0, 7.0, 3.0, 4.0, 8.0]
    );
    assert_eq!(lhs.stack(&lhs).unwrap().shape().dims(), &[2, 2, 2]);
}

#[test]
fn f16_full_mean_uses_wide_accumulator() {
    let input = Tensor1D::<4096, f16>::ones().unwrap();

    assert_eq!(input.mean().unwrap().item().unwrap(), f16::ONE);
}

#[test]
fn i64_tensors_cover_data_surface_and_loud_casts() {
    let ids = Tensor1D::<5, i64>::arange().unwrap();
    assert_eq!(ids.dtype(), DTypeId::I64);
    assert_eq!(ids.to_vec().unwrap(), vec![0, 1, 2, 3, 4]);

    let matrix = ids.reshape_with_shape::<D2<C<1>, C<5>>>([1, 5]).unwrap();
    assert_eq!(matrix.transpose().unwrap().shape().dims(), &[5, 1]);
    assert!(matrix.contiguous().unwrap().is_contiguous());
    assert_eq!(
        ids.gt_scalar(2).unwrap().to_vec().unwrap(),
        vec![false, false, false, true, true]
    );

    let more = Tensor1D::<2, i64>::from_vec(vec![5, 6]).unwrap();
    assert_eq!(
        ids.cat::<C<2>, 7>(&more).unwrap().to_vec().unwrap(),
        vec![0, 1, 2, 3, 4, 5, 6]
    );
    assert_eq!(
        more.stack(&more).unwrap().to_vec().unwrap(),
        vec![5, 6, 5, 6]
    );

    let floats: Tensor1D<4> = Tensor1D::<4, f32>::from_vec(vec![1.9, -2.2, 3.0, 0.0]).unwrap();
    let ints: Tensor1D<4, i64> = floats.cast().unwrap();
    assert_eq!(ints.to_vec().unwrap(), vec![1, -2, 3, 0]);
    let round_trip: Tensor1D<4, f32> = ints.cast().unwrap();
    assert_eq!(round_trip.to_vec().unwrap(), vec![1.0, -2.0, 3.0, 0.0]);

    let same: Tensor1D<1, i64> = Tensor1D::<1, i64>::from_vec(vec![9_007_199_254_740_993])
        .unwrap()
        .cast()
        .unwrap();
    assert_eq!(same.to_vec().unwrap(), vec![9_007_199_254_740_993]);

    let err = Tensor1D::<1>::from_vec(vec![f32::NAN])
        .unwrap()
        .cast::<i64>()
        .unwrap_err();
    assert!(matches!(err, Error::DType(DTypeError::InvalidCast { .. })));
    let err = Tensor1D::<1, f64>::from_vec(vec![9_223_372_036_854_775_808.0])
        .unwrap()
        .cast::<i64>()
        .unwrap_err();
    assert!(matches!(err, Error::DType(DTypeError::InvalidCast { .. })));

    let logits = Tensor2D::<2, 3>::from_vec(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0]).unwrap();
    assert_eq!(
        logits.argmax_last_tensor().unwrap().to_vec().unwrap(),
        vec![1, 2]
    );
}

#[test]
fn static_shape_operators_and_display_are_ergonomic() {
    let lhs = Tensor1D::<3>::from_vec(vec![1.0, 2.0, 3.0]).unwrap();
    let rhs = Tensor1D::<3>::from_vec(vec![4.0, 5.0, 6.0]).unwrap();

    assert_eq!(
        (&lhs + &rhs).unwrap().to_vec().unwrap(),
        vec![5.0, 7.0, 9.0]
    );
    assert_eq!(
        (&rhs - &lhs).unwrap().to_vec().unwrap(),
        vec![3.0, 3.0, 3.0]
    );
    assert_eq!((&lhs * 2.0).unwrap().to_vec().unwrap(), vec![2.0, 4.0, 6.0]);
    assert_eq!(
        (rhs.clone() / 2.0).unwrap().to_vec().unwrap(),
        vec![2.0, 2.5, 3.0]
    );
    assert_eq!((-&lhs).unwrap().to_vec().unwrap(), vec![-1.0, -2.0, -3.0]);

    let displayed = format!(
        "{}",
        Tensor1D::<12>::from_vec((0..12).map(|x| x as f32).collect()).unwrap()
    );
    assert_eq!(
        displayed,
        "Tensor(shape=[12], dtype=F32, values=[0, 1, 2, 3, 4, 5, ..., 10, 11])"
    );
}

#[test]
fn cross_entropy_and_log_softmax_stay_finite_on_extreme_logits() {
    let extreme = Tensor2D::<1, 2>::from_vec(vec![0.0, -1000.0]).unwrap();
    assert_close(
        extreme.cross_entropy(&[1]).unwrap().to_vec().unwrap()[0],
        1000.0,
        1e-3,
    );
    assert_close(
        extreme.log_softmax_last().unwrap().to_vec().unwrap()[1],
        -1000.0,
        1e-3,
    );
}

#[test]
fn broadcasts_masks_and_indexing_have_gradients() {
    let x = Tensor2D::<2, 3>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap()
        .with_requires_grad(true);
    let col = Tensor1D::<2>::from_vec(vec![10.0, 20.0]).unwrap();
    assert_eq!(
        x.add_leading_dim(&col).unwrap().to_vec().unwrap(),
        vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]
    );

    let row = Tensor1D::<3>::from_vec(vec![1.0, 2.0, 3.0])
        .unwrap()
        .with_requires_grad(true);
    x.mul_last_dim(&row)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    assert_eq!(row.grad().unwrap().to_vec().unwrap(), vec![5.0, 7.0, 9.0]);
    x.zero_grad();

    let mask = x.gt_scalar(3.0).unwrap();
    let mask2 = x.lt_scalar(6.0).unwrap();
    assert_eq!(
        mask.and(&mask2).unwrap().to_vec().unwrap(),
        vec![false, false, false, true, true, false]
    );
    assert_eq!(
        mask.or(&mask2.not()).unwrap().to_vec().unwrap(),
        vec![false, false, false, true, true, true]
    );
    assert_eq!(
        mask.expand_leading::<2>().unwrap().shape().dims(),
        &[2, 2, 3]
    );
    assert_eq!(
        x.masked_fill(&mask, -1.0).unwrap().to_vec().unwrap(),
        vec![1.0, 2.0, 3.0, -1.0, -1.0, -1.0]
    );

    let picked = x.index_select_rows::<C<3>>(&[1, 0, 1]).unwrap();
    picked.sum().unwrap().backward().unwrap();
    assert_eq!(
        x.grad().unwrap().to_vec().unwrap(),
        vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    );

    let id_indices = Tensor1D::<3, i64>::from_vec(vec![1, 0, 1]).unwrap();
    assert_eq!(
        x.index_select_rows_ids::<C<3>, C<3>>(&id_indices)
            .unwrap()
            .to_vec()
            .unwrap(),
        vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
}

#[test]
fn bmm_computes_batched_matmul() {
    let lhs = Tensor::<D3<C<2>, C<2>, C<3>>>::from_vec(vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
    ])
    .unwrap();
    let rhs = Tensor::<D3<C<2>, C<3>, C<2>>>::from_vec(vec![
        1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 2.0, 1.0, 1.0,
    ])
    .unwrap();
    assert_eq!(
        lhs.bmm(&rhs).unwrap().to_vec().unwrap(),
        vec![4.0, 5.0, 10.0, 11.0, 3.0, 1.0, 0.0, 2.0]
    );
}

#[test]
fn generic_axis_ops_cover_higher_ranks() {
    let x = Tensor::<D3<C<2>, C<2>, C<3>>>::from_vec((1..=12).map(|value| value as f32).collect())
        .unwrap();

    assert_eq!(
        x.sum_last().unwrap().to_vec().unwrap(),
        vec![6.0, 15.0, 24.0, 33.0]
    );
    assert_eq!(x.mean_last().unwrap().shape().dims(), &[2, 2]);
    assert_eq!(x.argmax_last().unwrap(), vec![2, 2, 2, 2]);

    let row = Tensor1D::<3>::from_vec(vec![1.0, 10.0, 100.0]).unwrap();
    assert_eq!(
        x.add_last_dim(&row).unwrap().to_vec().unwrap(),
        vec![
            2.0, 12.0, 103.0, 5.0, 15.0, 106.0, 8.0, 18.0, 109.0, 11.0, 21.0, 112.0
        ]
    );

    let y = Tensor::<D4<C<1>, C<2>, C<2>, C<3>>>::from_vec(
        (1..=12).map(|value| value as f32).collect(),
    )
    .unwrap();
    let softmax = y.softmax_last().unwrap().to_vec().unwrap();
    for row in softmax.chunks_exact(3) {
        assert_close(row.iter().sum(), 1.0, 1e-6);
    }
}

#[test]
fn generic_axis_broadcast_validates_symbols() {
    #[derive(Debug)]
    struct Hidden;

    let x = Tensor::<D3<C<1>, C<2>, Sym<Hidden>>>::from_vec_with_shape(vec![1.0; 6], [1, 2, 3])
        .unwrap();
    let row = Tensor::<D1<Sym<Hidden>>>::from_vec_with_shape(vec![1.0; 4], [4]).unwrap();
    let err = x.add_last_dim(&row).unwrap_err();

    assert!(matches!(
        err,
        Error::Shape(ShapeError::SymbolMismatch {
            op: "add_last_dim",
            ..
        })
    ));
}

#[test]
fn gelu_softmax_and_log_softmax_gradients_match_finite_difference() {
    let gelu_data = vec![-0.7, 0.2];
    let x = Tensor1D::<2, f64>::from_vec(gelu_data.clone())
        .unwrap()
        .with_requires_grad(true);
    x.gelu().unwrap().sum().unwrap().backward().unwrap();
    let grad = x.grad().unwrap().to_vec().unwrap();
    for (idx, &analytic) in grad.iter().enumerate() {
        let numerical = finite_difference(&gelu_data, idx, 1e-6, |values| {
            Tensor1D::<2, f64>::from_vec(values.to_vec())
                .unwrap()
                .gelu()
                .unwrap()
                .sum()
                .unwrap()
                .to_vec()
                .unwrap()[0]
        });
        assert_close_f64(analytic, numerical, 1e-6);
    }

    let data = vec![0.2, -0.5, 1.0, -1.0, 0.3, 0.7];
    let weights = Tensor2D::<2, 3, f64>::from_vec(vec![0.2, -0.4, 0.8, -0.3, 0.7, -0.1]).unwrap();

    let x = Tensor2D::<2, 3, f64>::from_vec(data.clone())
        .unwrap()
        .with_requires_grad(true);
    x.softmax_last()
        .unwrap()
        .mul(&weights)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    let grad = x.grad().unwrap().to_vec().unwrap();
    for (idx, &analytic) in grad.iter().enumerate() {
        let numerical = finite_difference(&data, idx, 1e-6, |values| {
            Tensor2D::<2, 3, f64>::from_vec(values.to_vec())
                .unwrap()
                .softmax_last()
                .unwrap()
                .mul(&weights)
                .unwrap()
                .sum()
                .unwrap()
                .to_vec()
                .unwrap()[0]
        });
        assert_close_f64(analytic, numerical, 1e-6);
    }

    let x = Tensor2D::<2, 3, f64>::from_vec(data.clone())
        .unwrap()
        .with_requires_grad(true);
    x.log_softmax_last()
        .unwrap()
        .mul(&weights)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    let grad = x.grad().unwrap().to_vec().unwrap();
    for (idx, &analytic) in grad.iter().enumerate() {
        let numerical = finite_difference(&data, idx, 1e-6, |values| {
            Tensor2D::<2, 3, f64>::from_vec(values.to_vec())
                .unwrap()
                .log_softmax_last()
                .unwrap()
                .mul(&weights)
                .unwrap()
                .sum()
                .unwrap()
                .to_vec()
                .unwrap()[0]
        });
        assert_close_f64(analytic, numerical, 1e-6);
    }
}

#[test]
fn bmm_gradients_match_finite_difference() {
    let lhs_data = vec![1.0, 2.0, -1.0, 0.5];
    let rhs_data = vec![0.3, -0.2, 0.7, 1.1];
    let weights = Tensor::<D3<C<1>, C<2>, C<2>>, f64>::from_vec(vec![0.4, -0.5, 0.2, 0.9]).unwrap();

    let lhs = Tensor::<D3<C<1>, C<2>, C<2>>, f64>::from_vec(lhs_data.clone())
        .unwrap()
        .with_requires_grad(true);
    let rhs = Tensor::<D3<C<1>, C<2>, C<2>>, f64>::from_vec(rhs_data.clone())
        .unwrap()
        .with_requires_grad(true);
    lhs.bmm(&rhs)
        .unwrap()
        .mul(&weights)
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();

    let lhs_grad = lhs.grad().unwrap().to_vec().unwrap();
    for (idx, &analytic) in lhs_grad.iter().enumerate() {
        let numerical = finite_difference(&lhs_data, idx, 1e-6, |values| {
            Tensor::<D3<C<1>, C<2>, C<2>>, f64>::from_vec(values.to_vec())
                .unwrap()
                .bmm(&Tensor::<D3<C<1>, C<2>, C<2>>, f64>::from_vec(rhs_data.clone()).unwrap())
                .unwrap()
                .mul(&weights)
                .unwrap()
                .sum()
                .unwrap()
                .to_vec()
                .unwrap()[0]
        });
        assert_close_f64(analytic, numerical, 1e-6);
    }
    let rhs_grad = rhs.grad().unwrap().to_vec().unwrap();
    for (idx, &analytic) in rhs_grad.iter().enumerate() {
        let numerical = finite_difference(&rhs_data, idx, 1e-6, |values| {
            Tensor::<D3<C<1>, C<2>, C<2>>, f64>::from_vec(lhs_data.clone())
                .unwrap()
                .bmm(&Tensor::<D3<C<1>, C<2>, C<2>>, f64>::from_vec(values.to_vec()).unwrap())
                .unwrap()
                .mul(&weights)
                .unwrap()
                .sum()
                .unwrap()
                .to_vec()
                .unwrap()[0]
        });
        assert_close_f64(analytic, numerical, 1e-6);
    }
}

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

fn finite_difference(values: &[f64], idx: usize, eps: f64, f: impl Fn(&[f64]) -> f64) -> f64 {
    let mut plus = values.to_vec();
    plus[idx] += eps;
    let mut minus = values.to_vec();
    minus[idx] -= eps;
    (f(&plus) - f(&minus)) / (2.0 * eps)
}
