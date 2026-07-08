use rstorch::prelude::*;
use rstorch::{AdamW, HasParameters, Mask, Optimizer};

#[test]
fn tiny_decoder_only_transformer_trains_on_cpu() {
    let mut rng = SmallRng::seed_from_u64(7);
    let mut model = DecoderOnlyTransformer::<6, 3, 4, 2, 2, 8, 2>::new(&mut rng).unwrap();
    let input = [[2, 4, 5], [4, 5, 3]];
    let target = [[4, 5, 3], [5, 3, 0]];
    let first = model.loss(&input, &target).unwrap().to_vec().unwrap()[0];
    let mut opt = AdamW::new(0.03, 0.0);

    for _ in 0..20 {
        model.loss(&input, &target).unwrap().backward().unwrap();
        let mut params = Vec::new();
        model.parameters_mut(&mut params);
        opt.step(&mut params).unwrap();
    }

    let last = model.loss(&input, &target).unwrap().to_vec().unwrap()[0];
    assert!(
        last < first,
        "expected final loss {last} to be below initial loss {first}"
    );

    let generated = model.generate(&[2, 4, 5], 2).unwrap();
    assert_eq!(generated.len(), 5);
    assert!(model.generate(&[], 1).is_err());
}

#[test]
fn same_transformer_instance_accepts_different_batch_sizes() {
    let mut rng = SmallRng::seed_from_u64(11);
    let model = DecoderOnlyTransformer::<6, 3, 4, 2, 2, 8>::new(&mut rng).unwrap();

    let one = model.forward(&[[2, 4, 5]]).unwrap();
    let two = model.forward(&[[2, 4, 5], [4, 5, 3]]).unwrap();

    assert_eq!(one.shape().dims(), &[1, 3, 6]);
    assert_eq!(two.shape().dims(), &[2, 3, 6]);
}

#[test]
fn transformer_config_customizes_construction_without_signature_changes() {
    let mut rng = SmallRng::seed_from_u64(12);
    let mut config = TransformerConfig::<f32>::default();
    config.norm_eps = 1e-4;
    config.init_std = 0.01;

    let model = DecoderOnlyTransformer::<6, 3, 4, 2, 2, 8>::with_config(&mut rng, config).unwrap();
    assert_eq!(
        model.forward(&[[2, 4, 5]]).unwrap().shape().dims(),
        &[1, 3, 6]
    );
}

#[test]
fn transformer_parameter_names_are_hierarchical_and_deterministic() {
    let mut rng = SmallRng::seed_from_u64(14);
    let model = DecoderOnlyTransformer::<6, 3, 4, 2, 2, 8>::new(&mut rng).unwrap();
    let mut names = Vec::new();
    model.visit_parameters("", &mut |name, _| names.push(name.to_owned()));

    assert_eq!(names[0], "token_embedding.weight");
    assert!(names.contains(&"blocks.0.attention.q_proj.weight".to_owned()));
    assert!(names.contains(&"lm_head.bias".to_owned()));
}

#[test]
fn transformer_block_composes_through_module_trait() {
    let mut rng = SmallRng::seed_from_u64(13);
    let block = TransformerBlock::<3, 4, 2, 2, 8>::new(&mut rng).unwrap();
    let input =
        Tensor::<D3<Sym<Batch>, C<3>, C<4>>>::from_vec_with_shape(vec![0.1; 2 * 3 * 4], [2, 3, 4])
            .unwrap();
    let mut ctx = ();

    let out = Module::forward(&block, &input, &mut ctx).unwrap();

    assert_eq!(out.shape().dims(), &[2, 3, 4]);
}

#[test]
fn forward_padded_accepts_padding_mask_and_produces_correct_shape() {
    let mut rng = SmallRng::seed_from_u64(8);
    let model = DecoderOnlyTransformer::<6, 3, 4, 2, 2, 8>::new(&mut rng).unwrap();
    let input = [[2usize, 4, 5], [4, 5, 0]];

    // padding_mask: true = padded position; batch 1 has no padding, batch 2 last token is pad
    let padding_mask = Mask::<D2<AnyDim, C<3>>>::from_vec_with_shape(
        vec![false, false, false, false, false, true],
        [2, 3],
    )
    .unwrap();

    let logits = model.forward_padded(&input, Some(&padding_mask)).unwrap();
    assert_eq!(logits.shape().dims(), &[2, 3, 6]);

    // None mask must also work (falls back to causal-only)
    let logits_no_mask = model.forward_padded(&input, None).unwrap();
    assert_eq!(logits_no_mask.shape().dims(), &[2, 3, 6]);

    // loss_padded with pad token=0 as ignore_index
    let targets = [[4usize, 5, 3], [5, 3, 0]];
    let loss = model
        .loss_padded(&input, &targets, Some(&padding_mask), 0)
        .unwrap();
    let loss_val = loss.to_vec().unwrap()[0];
    assert!(loss_val.is_finite() && loss_val > 0.0);
}

#[cfg(feature = "hub")]
#[test]
#[ignore]
fn tiny_shakespeare_character_model_overfits_one_batch() {
    let hub = DatasetHub::default_cache();
    let corpus = TinyShakespeare::<8>::load_text(&hub).unwrap();
    let tokenizer = CharTokenizer::from_text(&corpus);
    let dataset = TinyShakespeare::<8>::load(&hub, &tokenizer, false).unwrap();
    let samples = [dataset.get(0).unwrap(), dataset.get(1).unwrap()];
    let input = [samples[0].input, samples[1].input];
    let target = [samples[0].target, samples[1].target];
    let mut rng = SmallRng::seed_from_u64(19);
    let mut model = DecoderOnlyTransformer::<128, 8, 8, 2, 4, 16, 2>::new(&mut rng).unwrap();
    let mut opt = AdamW::new(0.01, 0.0);
    let first = model.loss(&input, &target).unwrap().to_vec().unwrap()[0];

    for _ in 0..5 {
        model.loss(&input, &target).unwrap().backward().unwrap();
        let mut params = Vec::new();
        model.parameters_mut(&mut params);
        opt.step(&mut params).unwrap();
    }

    let last = model.loss(&input, &target).unwrap().to_vec().unwrap()[0];
    assert!(last < first);
}
