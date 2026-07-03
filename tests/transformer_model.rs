use rstorch::prelude::*;
use rstorch::{AdamW, HasParameters, Optimizer};

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
