use rstorch::prelude::*;

#[test]
fn embedding_lookup_accumulates_repeated_token_gradients() {
    let embedding = Embedding::<4, 2>::from_weight(
        Tensor2D::<4, 2>::from_vec(vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );

    let out = embedding.forward(&[[1, 1, 2]]).unwrap();
    assert_eq!(out.shape().dims(), &[1, 3, 2]);
    assert_eq!(out.to_vec().unwrap(), vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0]);

    out.sum().unwrap().backward().unwrap();
    assert_eq!(
        embedding.weight().grad().unwrap().to_vec().unwrap(),
        vec![0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 0.0, 0.0]
    );
}

#[test]
fn embedding_forward_ids_accepts_i64_tensor_ids() {
    let embedding = Embedding::<4, 2>::from_weight(
        Tensor2D::<4, 2>::from_vec(vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
    );
    let ids =
        Tensor::<D2<Sym<Batch>, C<3>>, i64>::from_vec_with_shape(vec![1, 1, 2], [1, 3]).unwrap();

    let out = embedding.forward_ids(&ids).unwrap();

    assert_eq!(out.shape().dims(), &[1, 3, 2]);
    assert_eq!(out.to_vec().unwrap(), vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn scaled_attention_causal_mask_blocks_future_positions() {
    let q = Tensor::<D3<AnyDim, C<3>, C<1>>>::from_vec_with_shape(vec![0.0, 0.0, 0.0], [1, 3, 1])
        .unwrap();
    let k = Tensor::<D3<AnyDim, C<3>, C<1>>>::from_vec_with_shape(vec![0.0, 0.0, 0.0], [1, 3, 1])
        .unwrap();
    let v =
        Tensor::<D3<AnyDim, C<3>, C<1>>>::from_vec_with_shape(vec![10.0, 20.0, 30.0], [1, 3, 1])
            .unwrap();
    let mask = causal_attention_mask::<3>(1).unwrap();

    let out = scaled_dot_product_attention(&q, &k, &v, Some(&mask)).unwrap();

    assert_close(out.to_vec().unwrap()[0], 10.0, 1e-5);
    assert_close(out.to_vec().unwrap()[1], 15.0, 1e-5);
    assert_close(out.to_vec().unwrap()[2], 20.0, 1e-5);
}

fn assert_close(actual: f32, expected: f32, tol: f32) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {actual} to be within {tol} of {expected}"
    );
}
