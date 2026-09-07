//! Scaled dot-product attention and `MultiHeadAttention`: mask polarity, head
//! splitting and merging, and gradients.

use super::*;
use crate::nn::ModuleExt;
use crate::testing::check_grad;

const CPU: Device = Device::Cpu;

fn seq(n: usize) -> Vec<f32> {
    // A deterministic, non-symmetric spread in [-1, 1): distinct values,
    // so no accidental cancellation hides a wrong axis.
    (0..n)
        .map(|i| ((i as f32 * 0.37).sin() * 0.9 + (i as f32) * 0.011).clamp(-1.0, 1.0))
        .collect()
}

fn t(shape: &[usize]) -> Tensor {
    Tensor::from_vec(seq(shape.iter().product()), shape.to_vec(), &CPU).unwrap()
}

fn v(t: &Tensor) -> Vec<f32> {
    t.to_vec::<f32>().unwrap()
}

fn mha(embed: usize, heads: usize) -> MultiHeadAttention {
    MultiHeadAttention::new(embed, heads, &CPU, &mut Rng::seed(17)).unwrap()
}

fn assert_close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length");
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol * (1.0 + x.abs().max(y.abs())),
            "element {i}: {x} vs {y}"
        );
    }
}

/// A fixed non-uniform weighting, so every output element contributes to
/// the scalar objective the gradient tests differentiate.
fn coefficients(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_vec(
        (0..n).map(|i| 0.3 + i as f32 * 0.17).collect(),
        shape.to_vec(),
        &CPU,
    )
    .unwrap()
}

// ------------------------------------------------------------------
// scaled_dot_product_attention
// ------------------------------------------------------------------

#[test]
fn sdpa_shapes_are_query_length_by_value_width() {
    // Batch axes broadcast; q_len and kv_len are independent.
    let q = t(&[2, 3, 4, 5]); // [batch, heads, q_len, head_dim]
    let k = t(&[2, 3, 7, 5]); // [batch, heads, kv_len, head_dim]
    let val = t(&[2, 3, 7, 6]); // a value width of its own
    let out = scaled_dot_product_attention(&q, &k, &val, None).unwrap();
    assert_eq!(out.dims(), &[2, 3, 4, 6]);

    // Unbatched: one bare head.
    let out = scaled_dot_product_attention(&t(&[4, 5]), &t(&[7, 5]), &t(&[7, 5]), None).unwrap();
    assert_eq!(out.dims(), &[4, 5]);

    // One query against a cached prefix — the decoding step.
    let out = scaled_dot_product_attention(&t(&[2, 3, 1, 5]), &k, &val, None).unwrap();
    assert_eq!(out.dims(), &[2, 3, 1, 6]);
}

#[test]
fn sdpa_is_the_scaled_softmax_weighted_average_of_the_values() {
    // Hand-rolled on paper: two queries, two keys of width 4.
    let q = t(&[2, 4]);
    let k = t(&[2, 4]);
    let val = t(&[2, 3]);
    let out = scaled_dot_product_attention(&q, &k, &val, None).unwrap();

    let (qv, kv, vv) = (v(&q), v(&k), v(&val));
    let scale = 4f32.sqrt();
    let mut expected = vec![0.0f32; 2 * 3];
    for i in 0..2 {
        let logits: Vec<f32> = (0..2)
            .map(|j| (0..4).map(|d| qv[i * 4 + d] * kv[j * 4 + d]).sum::<f32>() / scale)
            .collect();
        let peak = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = logits.iter().map(|l| (l - peak).exp()).collect();
        let denom: f32 = e.iter().sum();
        for c in 0..3 {
            expected[i * 3 + c] = (0..2).map(|j| e[j] / denom * vv[j * 3 + c]).sum();
        }
    }
    assert_close(&v(&out), &expected, 1e-6);
}

#[test]
fn attention_weights_are_a_distribution_over_the_keys() {
    // Attending with the identity as values recovers the weights, whose
    // rows must each sum to one.
    let identity = Tensor::from_vec(
        (0..9).map(|i| f32::from(i % 4 == 0)).collect::<Vec<f32>>(),
        [3, 3],
        &CPU,
    )
    .unwrap();
    let weights = scaled_dot_product_attention(&t(&[3, 4]), &t(&[3, 4]), &identity, None).unwrap();
    for row in v(&weights).chunks(3) {
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6, "{row:?}");
    }
}

#[test]
fn a_masked_key_gets_exactly_zero_weight() {
    // Blocking key 0 for every query must reproduce attention over the
    // remaining keys alone.
    let (q, k) = (t(&[2, 4]), t(&[3, 4]));
    let val = t(&[3, 5]);
    let blocked = Tensor::from_vec(vec![true, false, false], [1, 3], &CPU).unwrap();
    let masked = scaled_dot_product_attention(&q, &k, &val, Some(&blocked)).unwrap();

    let without = scaled_dot_product_attention(
        &q,
        &k.narrow(0, 1, 2).unwrap(),
        &val.narrow(0, 1, 2).unwrap(),
        None,
    )
    .unwrap();
    assert_close(&v(&masked), &v(&without), 1e-6);
}

#[test]
fn a_fully_masked_row_is_zero_rather_than_nan() {
    // The degenerate case softmax is built to survive: a query with
    // no visible key at all. It must not poison the batch.
    let mask = Tensor::from_vec(vec![true, true, false, false], [2, 2], &CPU).unwrap();
    let out =
        scaled_dot_product_attention(&t(&[2, 3]), &t(&[2, 3]), &t(&[2, 3]), Some(&mask)).unwrap();
    let out = v(&out);
    assert_eq!(&out[0..3], &[0.0, 0.0, 0.0]);
    assert!(out.iter().all(|x| !x.is_nan()), "{out:?}");
    assert!(out[3..].iter().any(|x| *x != 0.0));
}

#[test]
fn a_causal_mask_hides_the_future_from_every_position() {
    // The proof the gate asks for: perturbing position t+1 of the keys and
    // values cannot move the output at position t, while perturbing
    // position t-1 does.
    let (seq_len, dim) = (4, 3);
    let x = t(&[seq_len, dim]);
    let mask = Tensor::causal_mask(seq_len, &CPU).unwrap();
    let base = v(&scaled_dot_product_attention(&x, &x, &x, Some(&mask)).unwrap());

    for perturbed in 0..seq_len {
        let mut values = v(&x);
        values[perturbed * dim..(perturbed + 1) * dim]
            .iter_mut()
            .for_each(|value| *value += 10.0);
        let y = Tensor::from_vec(values, [seq_len, dim], &CPU).unwrap();
        let moved = v(&scaled_dot_product_attention(&y, &y, &y, Some(&mask)).unwrap());

        for query in 0..seq_len {
            let row = query * dim..(query + 1) * dim;
            let changed = base[row.clone()]
                .iter()
                .zip(&moved[row])
                .any(|(a, b)| (a - b).abs() > 1e-6);
            if query < perturbed {
                assert!(
                    !changed,
                    "query {query} saw the future at position {perturbed}"
                );
            } else {
                assert!(changed, "query {query} ignored position {perturbed}");
            }
        }
    }
}

#[test]
fn sdpa_gradients_match_finite_differences() {
    // Unmasked, batched over two heads.
    let coef = coefficients(&[2, 3, 5]);
    let f = |xs: &[Tensor]| {
        scaled_dot_product_attention(&xs[0], &xs[1], &xs[2], None)?
            .mul(&coef)?
            .sum_all()
    };
    check_grad(
        f,
        &[t(&[2, 3, 4]), t(&[2, 4, 4]), t(&[2, 4, 5])],
        1e-2,
        1e-3,
    )
    .unwrap();

    // Masked: the causal case, where some scores are -inf. The cotangent
    // must not flow through them (and must not become NaN).
    let mask = Tensor::causal_mask(3, &CPU).unwrap();
    let coef = coefficients(&[2, 3, 4]);
    let g = move |xs: &[Tensor]| {
        scaled_dot_product_attention(&xs[0], &xs[1], &xs[2], Some(&mask))?
            .mul(&coef)?
            .sum_all()
    };
    check_grad(
        g,
        &[t(&[2, 3, 4]), t(&[2, 3, 4]), t(&[2, 3, 4])],
        1e-2,
        1e-3,
    )
    .unwrap();
}

#[test]
fn sdpa_rejects_shapes_that_do_not_line_up() {
    let ok = t(&[3, 4]);
    // head_dim disagreement between q and k.
    assert!(matches!(
        scaled_dot_product_attention(&ok, &t(&[3, 5]), &ok, None),
        Err(Error::ShapeMismatch {
            op: "scaled_dot_product_attention",
            ..
        })
    ));
    // kv_len disagreement between k and v.
    assert!(matches!(
        scaled_dot_product_attention(&ok, &ok, &t(&[2, 4]), None),
        Err(Error::ShapeMismatch { .. })
    ));
    // Rank 1 is not a sequence of vectors.
    assert!(matches!(
        scaled_dot_product_attention(&t(&[4]), &ok, &ok, None),
        Err(Error::InvalidArg {
            op: "scaled_dot_product_attention",
            ..
        })
    ));
    // An empty axis has no softmax.
    let empty = Tensor::zeros([0, 4], DType::F32, &CPU).unwrap();
    assert!(matches!(
        scaled_dot_product_attention(&empty, &ok, &ok, None),
        Err(Error::InvalidArg { .. })
    ));
}

#[test]
fn sdpa_rejects_a_mask_that_is_not_a_broadcastable_bool() {
    let x = t(&[3, 4]);
    let floats = t(&[3, 3]);
    assert!(matches!(
        scaled_dot_product_attention(&x, &x, &x, Some(&floats)),
        Err(Error::DTypeMismatch {
            op: "scaled_dot_product_attention",
            expected: DType::Bool,
            got: DType::F32
        })
    ));
    // A mask that would *grow* the scores is a bug, not a broadcast.
    let too_wide = Tensor::causal_mask(5, &CPU).unwrap();
    assert!(matches!(
        scaled_dot_product_attention(&x, &x, &x, Some(&too_wide)),
        Err(Error::ShapeMismatch {
            op: "scaled_dot_product_attention",
            ..
        })
    ));
    // …and so is one of higher rank than the scores.
    let too_deep = Tensor::causal_mask(3, &CPU)
        .unwrap()
        .reshape([1, 3, 3])
        .unwrap();
    assert!(matches!(
        scaled_dot_product_attention(&x, &x, &x, Some(&too_deep)),
        Err(Error::ShapeMismatch { .. })
    ));
}

// ------------------------------------------------------------------
// MultiHeadAttention: shapes
// ------------------------------------------------------------------

#[test]
fn attend_preserves_the_input_shape_across_batch_heads_and_seq() {
    for &heads in &[1usize, 2, 4, 8] {
        let attn = mha(8, heads);
        assert_eq!(attn.num_heads(), heads);
        assert_eq!(attn.head_dim(), 8 / heads);
        for shape in [vec![3usize, 8], vec![2, 3, 8], vec![2, 3, 5, 8]] {
            let out = attn.attend(&t(&shape), None, Mode::EVAL).unwrap();
            assert_eq!(out.dims(), &shape[..], "heads {heads}, shape {shape:?}");
        }
    }
}

#[test]
fn attend_to_takes_independent_query_and_key_lengths() {
    let attn = mha(6, 3);
    // Cross-attention: 2 queries over 5 keys.
    let out = attn
        .attend_to(&t(&[2, 2, 6]), &t(&[2, 5, 6]), None, Mode::EVAL)
        .unwrap();
    assert_eq!(out.dims(), &[2, 2, 6]);

    // One new token over a cached prefix.
    let step = attn
        .attend_to(&t(&[2, 1, 6]), &t(&[2, 5, 6]), None, Mode::EVAL)
        .unwrap();
    assert_eq!(step.dims(), &[2, 1, 6]);
}

#[test]
fn the_projection_halves_round_trip_the_head_split() {
    let attn = mha(6, 3);
    let x = t(&[2, 5, 6]);
    let q = attn.project_query(&x, Mode::EVAL).unwrap();
    assert_eq!(q.dims(), &[2, 3, 5, 2]);
    let (k, val) = attn.project_keys_values(&x, Mode::EVAL).unwrap();
    assert_eq!(k.dims(), &[2, 3, 5, 2]);
    assert_eq!(v(&val), v(&attn.project_values(&x, Mode::EVAL).unwrap()));

    // Assembling the halves by hand reproduces `attend` exactly.
    let context = scaled_dot_product_attention(&q, &k, &val, None).unwrap();
    let out = attn.project_output(&context, Mode::EVAL).unwrap();
    assert_eq!(v(&out), v(&attn.attend(&x, None, Mode::EVAL).unwrap()));
}

#[test]
fn the_head_split_partitions_the_embedding_contiguously() {
    // The ordering no shape assertion can catch: head `h` of position `s`
    // must be the slice `[h * head_dim .. (h + 1) * head_dim]` of the
    // *unsplit* projection — PyTorch's
    // `reshape(batch, seq, heads, head_dim).transpose(1, 2)`. A permute
    // written the other way round has the same shape and different values.
    let attn = mha(6, 3);
    let (heads, head_dim, seq_len) = (3, 2, 4);
    let x = t(&[1, seq_len, 6]);
    let flat = v(&attn.q_proj.apply(&x, Mode::EVAL).unwrap());
    let split = v(&attn.project_query(&x, Mode::EVAL).unwrap());

    for h in 0..heads {
        for s in 0..seq_len {
            for d in 0..head_dim {
                let from_split = split[(h * seq_len + s) * head_dim + d];
                let from_flat = flat[s * (heads * head_dim) + h * head_dim + d];
                assert_eq!(from_split, from_flat, "head {h}, position {s}, lane {d}");
            }
        }
    }
}

#[test]
fn attending_over_a_concatenated_cache_equals_attending_over_the_whole_prefix() {
    // The KV-cache invariant, without a cache type: keys projected token by
    // token and concatenated are the keys projected in one go.
    let attn = mha(4, 2);
    let prefix = t(&[1, 3, 4]);
    let step = t(&[1, 1, 4]);
    let whole = Tensor::cat(&[&prefix, &step], 1).unwrap();

    let cached_k = Tensor::cat(
        &[
            &attn.project_keys(&prefix, Mode::EVAL).unwrap(),
            &attn.project_keys(&step, Mode::EVAL).unwrap(),
        ],
        -2,
    )
    .unwrap();
    let cached_v = Tensor::cat(
        &[
            &attn.project_values(&prefix, Mode::EVAL).unwrap(),
            &attn.project_values(&step, Mode::EVAL).unwrap(),
        ],
        -2,
    )
    .unwrap();
    let q = attn.project_query(&step, Mode::EVAL).unwrap();
    let cached = attn
        .project_output(
            &scaled_dot_product_attention(&q, &cached_k, &cached_v, None).unwrap(),
            Mode::EVAL,
        )
        .unwrap();

    // The same last position, attending over the same four tokens.
    let full = attn.attend(&whole, None, Mode::EVAL).unwrap();
    assert_close(&v(&cached), &v(&full.narrow(1, 3, 1).unwrap()), 1e-6);
}

// ------------------------------------------------------------------
// MultiHeadAttention: the single-head equivalence
// ------------------------------------------------------------------

/// Single-head attention written out with plain tensors: the reference the
/// module is measured against (and, in the gradient test below, the
/// function finite differences are taken of).
///
/// `xs` is `[x, w_q, w_k, w_v, w_out]`; biases are omitted, which is why
/// the equivalence test uses `new_without_bias`.
fn hand_rolled_single_head(xs: &[Tensor], mask: Option<&Tensor>) -> Result<Tensor> {
    let (x, embed) = (&xs[0], xs[0].dims()[xs[0].rank() - 1]);
    let project = |w: &Tensor| x.matmul(&w.transpose(0, 1)?);
    let (q, k, val) = (project(&xs[1])?, project(&xs[2])?, project(&xs[3])?);
    let scores = q
        .matmul(&k.transpose(-2, -1)?)?
        .div_scalar((embed as f64).sqrt())?;
    let scores = match mask {
        Some(mask) => scores.masked_fill(mask, f64::NEG_INFINITY)?,
        None => scores,
    };
    scores
        .softmax(-1)?
        .matmul(&val)?
        .matmul(&xs[4].transpose(0, 1)?)
}

/// `[x, w_q, w_k, w_v, w_out]` for a bias-free single-head module.
fn single_head_inputs(attn: &MultiHeadAttention, x: &Tensor) -> Vec<Tensor> {
    vec![
        x.clone(),
        attn.q_proj.weight.value().clone(),
        attn.k_proj.weight.value().clone(),
        attn.v_proj.weight.value().clone(),
        attn.out_proj.weight.value().clone(),
    ]
}

#[test]
fn single_head_attention_matches_a_hand_rolled_computation() {
    let attn = MultiHeadAttention::new_without_bias(4, 1, &CPU, &mut Rng::seed(5)).unwrap();
    let x = t(&[2, 3, 4]);
    let mask = Tensor::causal_mask(3, &CPU).unwrap();

    for m in [None, Some(&mask)] {
        let expected = hand_rolled_single_head(&single_head_inputs(&attn, &x), m).unwrap();
        let actual = attn.attend(&x, m, Mode::EVAL).unwrap();
        assert_eq!(actual.dims(), expected.dims());
        assert_close(&v(&actual), &v(&expected), 1e-6);
    }
}

// ------------------------------------------------------------------
// MultiHeadAttention: gradients
// ------------------------------------------------------------------

#[test]
fn attend_gradients_wrt_the_input_match_finite_differences() {
    let attn = mha(4, 2);
    let mask = Tensor::causal_mask(3, &CPU).unwrap();
    let coef = coefficients(&[2, 3, 4]);
    let f = |xs: &[Tensor]| {
        attn.attend(&xs[0], Some(&mask), Mode::TRAIN)?
            .mul(&coef)?
            .sum_all()
    };
    check_grad(f, &[t(&[2, 3, 4])], 1e-2, 1e-3).unwrap();
}

#[test]
fn parameter_gradients_are_the_finite_differences_of_the_same_function() {
    // Two chained claims. First: the hand-rolled single-head function
    // agrees with central finite differences in *all five* of its inputs,
    // including the four projection weights.
    let attn = MultiHeadAttention::new_without_bias(4, 1, &CPU, &mut Rng::seed(9)).unwrap();
    let x = t(&[2, 3, 4]);
    let mask = Tensor::causal_mask(3, &CPU).unwrap();
    let coef = coefficients(&[2, 3, 4]);
    let inputs = single_head_inputs(&attn, &x);

    let objective = {
        let (mask, coef) = (mask.clone(), coef.clone());
        move |xs: &[Tensor]| {
            hand_rolled_single_head(xs, Some(&mask))?
                .mul(&coef)?
                .sum_all()
        }
    };
    check_grad(&objective, &inputs, 1e-2, 1e-3).unwrap();

    // Second: the module's own parameter gradients equal that function's,
    // weight for weight — so the finite-difference evidence carries over to
    // the `Param` path.
    let traced: Vec<Tensor> = inputs.iter().map(|t| t.traced().unwrap()).collect();
    let reference = objective(&traced).unwrap().backward().unwrap();

    let loss = attn
        .attend(&x, Some(&mask), Mode::TRAIN)
        .unwrap()
        .mul(&coef)
        .unwrap()
        .sum_all()
        .unwrap();
    let grads = loss.backward().unwrap();
    for (i, param) in [
        &attn.q_proj.weight,
        &attn.k_proj.weight,
        &attn.v_proj.weight,
        &attn.out_proj.weight,
    ]
    .into_iter()
    .enumerate()
    {
        let expected = reference.wrt_input(&traced[i + 1]).unwrap();
        let actual = grads.wrt(param).unwrap();
        assert_eq!(actual.dims(), expected.dims());
        assert_close(&v(&actual), &v(&expected), 1e-5);
    }
}

#[test]
fn every_parameter_receives_a_gradient() {
    // The completeness property the optimizer enforces: with biases
    // on and several heads, all eight leaves are reached.
    let attn = mha(6, 3);
    let loss = attn
        .attend(&t(&[2, 4, 6]), None, Mode::TRAIN)
        .unwrap()
        .mul(&coefficients(&[2, 4, 6]))
        .unwrap()
        .sum_all()
        .unwrap();
    let grads = loss.backward().unwrap();
    assert_eq!(grads.len(), 8);
    for proj in [&attn.q_proj, &attn.k_proj, &attn.v_proj, &attn.out_proj] {
        for p in [Some(&proj.weight), proj.bias.as_ref()]
            .into_iter()
            .flatten()
        {
            let g = grads.wrt(p).unwrap();
            assert_eq!(g.dims(), p.value().dims());
            assert!(v(&g).iter().any(|x| *x != 0.0), "an all-zero gradient");
        }
    }
}

#[test]
fn the_layer_trains_under_plain_gradient_descent() {
    // End to end, without an optimizer: 60 hand-written SGD steps over
    // every visited parameter must drive a real objective down. This is the
    // property all the gradient algebra exists for, and the one a
    // sign error or a mis-scaled head would break while every shape test
    // still passed.
    use crate::nn::visit::{LeafMut, visit_all_mut};

    let mut attn = mha(4, 2);
    let x = t(&[1, 4, 4]);
    let target = coefficients(&[1, 4, 4]).mul_scalar(0.05).unwrap();
    let mask = Tensor::causal_mask(4, &CPU).unwrap();

    let loss_now = |attn: &MultiHeadAttention| {
        attn.attend(&x, Some(&mask), Mode::TRAIN)
            .unwrap()
            .mse_loss(&target)
            .unwrap()
    };

    let first = loss_now(&attn).item().unwrap();
    for _ in 0..60 {
        let grads = loss_now(&attn).backward().unwrap();
        visit_all_mut(&mut attn, &mut |path, leaf| {
            let LeafMut::Param(p) = leaf else {
                panic!("attention has no buffers, yet {path} is one");
            };
            let step = grads.wrt(p).unwrap().mul_scalar(0.5).unwrap();
            p.set(p.value().sub(&step).unwrap()).unwrap();
        });
    }
    let last = loss_now(&attn).item().unwrap();
    assert!(
        last < first * 0.1,
        "loss barely moved: {first} -> {last} in 60 steps"
    );
}

#[test]
fn eval_records_nothing_and_agrees_with_train_on_the_values() {
    let attn = mha(4, 2);
    let x = t(&[1, 3, 4]);
    let train = attn.attend(&x, None, Mode::TRAIN).unwrap();
    let eval = attn.attend(&x, None, Mode::EVAL).unwrap();
    assert_eq!(v(&train), v(&eval));
    assert!(matches!(
        eval.sum_all().unwrap().backward(),
        Err(Error::NotTraced { .. })
    ));
}

// ------------------------------------------------------------------
// MultiHeadAttention: module wiring and loud errors
// ------------------------------------------------------------------

#[test]
fn the_state_dict_paths_are_the_four_projections() {
    let attn = mha(4, 2);
    assert_eq!(
        attn.state_dict().unwrap().keys().collect::<Vec<_>>(),
        [
            "k_proj.bias",
            "k_proj.weight",
            "out_proj.bias",
            "out_proj.weight",
            "q_proj.bias",
            "q_proj.weight",
            "v_proj.bias",
            "v_proj.weight",
        ]
    );
    assert_eq!(attn.num_params(), 4 * (4 * 4 + 4));

    // Bias-free: the `bias` paths are simply absent.
    let bare = MultiHeadAttention::new_without_bias(4, 2, &CPU, &mut Rng::seed(1)).unwrap();
    assert_eq!(bare.state_dict().unwrap().len(), 4);
    assert_eq!(bare.num_params(), 4 * 4 * 4);
}

#[test]
fn a_checkpoint_round_trip_reproduces_the_outputs() {
    let trained = mha(4, 2);
    let mut fresh = MultiHeadAttention::new(4, 2, &CPU, &mut Rng::seed(999)).unwrap();
    let x = t(&[1, 3, 4]);
    assert_ne!(
        v(&trained.attend(&x, None, Mode::EVAL).unwrap()),
        v(&fresh.attend(&x, None, Mode::EVAL).unwrap())
    );
    fresh
        .load_state_dict(&trained.state_dict().unwrap())
        .unwrap();
    assert_eq!(
        v(&trained.attend(&x, None, Mode::EVAL).unwrap()),
        v(&fresh.attend(&x, None, Mode::EVAL).unwrap())
    );
}

#[test]
fn a_padding_mask_composes_with_a_causal_one() {
    // The recipe this module's docs promise: two blocked-position masks
    // OR-ed on device, no host round-trip. Key 2 is padding for the one
    // sample here, so no query may attend to it.
    let attn = mha(4, 2);
    let x = t(&[1, 3, 4]);
    let causal = Tensor::causal_mask(3, &CPU).unwrap();
    // [batch, kv_len] reshaped to broadcast over heads and queries.
    let padding = Tensor::from_vec(vec![false, false, true], [1, 1, 1, 3], &CPU).unwrap();
    let combined = causal.where_cond(&causal, &padding).unwrap();
    assert_eq!(combined.dims(), &[1, 1, 3, 3]);

    let masked = attn.attend(&x, Some(&combined), Mode::EVAL).unwrap();
    // Dropping the padded token entirely must give the same first two
    // positions, since nothing was allowed to see it.
    let short = attn
        .attend(
            &x.narrow(1, 0, 2).unwrap(),
            Some(&Tensor::causal_mask(2, &CPU).unwrap()),
            Mode::EVAL,
        )
        .unwrap();
    assert_close(&v(&masked)[..8], &v(&short), 1e-6);
}

#[test]
fn a_head_count_that_does_not_divide_the_embedding_is_loud() {
    let mut rng = Rng::seed(1);
    for (embed, heads) in [(6usize, 4usize), (0, 1), (4, 0)] {
        assert!(
            matches!(
                MultiHeadAttention::new(embed, heads, &CPU, &mut rng),
                Err(Error::InvalidArg {
                    op: "MultiHeadAttention::new",
                    ..
                })
            ),
            "embed {embed}, heads {heads}"
        );
    }
    assert!(matches!(
        MultiHeadAttention::new_without_bias(6, 4, &CPU, &mut rng),
        Err(Error::InvalidArg {
            op: "MultiHeadAttention::new_without_bias",
            ..
        })
    ));
}

#[test]
fn a_wrongly_shaped_input_names_this_layer_not_matmul() {
    let attn = mha(4, 2);
    assert!(matches!(
        attn.attend(&t(&[3, 5]), None, Mode::EVAL),
        Err(Error::ShapeMismatch {
            op: "MultiHeadAttention::project_query",
            ..
        })
    ));
    assert!(matches!(
        attn.attend_to(&t(&[3, 4]), &t(&[3, 5]), None, Mode::EVAL),
        Err(Error::ShapeMismatch {
            op: "MultiHeadAttention::project_keys",
            ..
        })
    ));
    assert!(matches!(
        attn.project_query(&t(&[4]), Mode::EVAL),
        Err(Error::InvalidArg {
            op: "MultiHeadAttention::project_query",
            ..
        })
    ));
}

#[test]
fn project_output_checks_the_head_axes() {
    let attn = mha(4, 2);
    // Rank 2 has no head axis at all.
    assert!(matches!(
        attn.project_output(&t(&[3, 2]), Mode::EVAL),
        Err(Error::InvalidArg {
            op: "MultiHeadAttention::project_output",
            ..
        })
    ));
    // Three heads where the layer has two.
    assert!(matches!(
        attn.project_output(&t(&[3, 3, 2]), Mode::EVAL),
        Err(Error::ShapeMismatch {
            op: "MultiHeadAttention::project_output",
            ..
        })
    ));
    // The right head count, the wrong head width.
    assert!(matches!(
        attn.project_output(&t(&[2, 3, 3]), Mode::EVAL),
        Err(Error::ShapeMismatch { .. })
    ));
}

#[test]
fn attention_is_reachable_through_the_forward_trait() {
    let mut rng = Rng::seed(11);
    let mut attn = MultiHeadAttention::new(4, 2, &CPU, &mut rng).unwrap();
    let x = t(&[2, 3, 4]);
    let mask = Tensor::causal_mask(3, &CPU).unwrap();

    // The trait spelling and the inherent spelling are the same computation,
    // mask included — not merely the same shape.
    let direct = attn.attend(&x, Some(&mask), Mode::EVAL).unwrap();
    let input = AttentionInput::new(x.clone(), Some(mask.clone()));
    let through_trait = attn.forward(&input, Mode::EVAL).unwrap();
    assert_eq!(
        through_trait.to_vec::<f32>().unwrap(),
        direct.to_vec::<f32>().unwrap()
    );

    // And the mask is genuinely carried: dropping it changes the result, so a
    // `Forward` caller cannot accidentally get bidirectional attention.
    let unmasked = attn
        .forward(&AttentionInput::new(x, None), Mode::EVAL)
        .unwrap();
    assert_ne!(
        unmasked.to_vec::<f32>().unwrap(),
        direct.to_vec::<f32>().unwrap()
    );
}

#[test]
fn a_generic_forward_caller_accepts_the_attention_layer() {
    // The point of `Forward<Input>`: code written against the trait, not
    // against a concrete layer, can drive a multi-input layer.
    fn run<L: Forward<AttentionInput, Output = Tensor>>(
        layer: &mut L,
        input: &AttentionInput,
    ) -> Tensor {
        layer.forward(input, Mode::EVAL).unwrap()
    }

    let mut rng = Rng::seed(12);
    let mut attn = MultiHeadAttention::new(4, 2, &CPU, &mut rng).unwrap();
    let input = AttentionInput::new(t(&[2, 3, 4]), Some(Tensor::causal_mask(3, &CPU).unwrap()));
    assert_eq!(run(&mut attn, &input).dims(), &[2, 3, 4]);
}
