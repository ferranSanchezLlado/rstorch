use rstorch::prelude::*;
use rstorch::shape::AnyDim;
use rstorch::{
    AdamW, BatchNorm2d, Error, HasParameters, Optimizer, PersistenceError, StateDict, Tensor,
    TensorRecord,
};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
struct Batch;

static NEXT_PATH: AtomicUsize = AtomicUsize::new(0);

#[test]
fn tensor_files_round_trip_all_float_dtypes_bit_exactly() {
    round_trip_f16();
    round_trip_bf16();
    round_trip_f32();
    round_trip_f64();
    round_trip_i64();
}

#[test]
fn tensor_file_rejects_unknown_dtype_code() {
    let tensor = Tensor1D::<1, i64>::from_vec(vec![7]).unwrap();
    let path = test_path("unknown-dtype");
    save_tensor(&path, &tensor).unwrap();

    let mut bytes = fs::read(&path).unwrap();
    bytes[26] = 99;
    fs::write(&path, bytes).unwrap();

    let err = load_tensor::<D1<C<1>>, i64>(&path).unwrap_err();
    assert!(matches!(
        err,
        Error::Persistence(PersistenceError::InvalidDType { code: 99 })
    ));
}

#[test]
fn linear_state_dict_round_trips_all_float_dtypes() {
    round_trip_linear_state::<f16>();
    round_trip_linear_state::<bf16>();
    round_trip_linear_state::<f32>();
    round_trip_linear_state::<f64>();
}

#[test]
fn state_dict_strictly_loads_mlp_and_reports_mismatches() {
    type Mlp = Sequential<((Linear<4, 3>, Relu), Linear<3, 2>), Tensor<D2<Sym<Batch>, C<4>>>>;

    let mut rng = SmallRng::seed_from_u64(13);
    let mut model: Mlp = seq![
        Linear::<4, 3>::xavier_uniform(&mut rng).unwrap(),
        Relu,
        Linear::<3, 2>::xavier_uniform(&mut rng).unwrap(),
    ];
    let input = Tensor::<D2<Sym<Batch>, C<4>>>::from_vec_with_shape(
        vec![1.0, -2.0, 0.5, 3.0, -1.0, 0.25, 2.0, 0.75],
        [2, 4],
    )
    .unwrap();
    let mut ctx = TrainContext::eval();
    let target =
        Tensor::<D2<Sym<Batch>, C<2>>>::from_vec_with_shape(vec![0.25, -0.5, 1.5, 0.0], [2, 2])
            .unwrap();
    let mut opt = AdamW::new(0.02, 0.0);
    for _ in 0..3 {
        let mut params = Vec::new();
        model.parameters(&mut params);
        opt.zero_grad(&params);
        drop(params);

        let loss = mse_loss(&model.forward(&input, &mut ctx).unwrap(), &target).unwrap();
        loss.backward().unwrap();
        let mut params = Vec::new();
        model.parameters_mut(&mut params);
        opt.step(&mut params).unwrap();
    }
    let expected = model.forward(&input, &mut ctx).unwrap().to_vec().unwrap();

    let path = test_path("mlp-state");
    StateDict::from_module(&model)
        .unwrap()
        .save_to_path(&path)
        .unwrap();

    let mut restored: Mlp = seq![
        Linear::<4, 3>::zeros().unwrap(),
        Relu,
        Linear::<3, 2>::zeros().unwrap(),
    ];
    let state = StateDict::load_from_path(&path).unwrap();
    state.load_module(&mut restored).unwrap();
    let found = restored
        .forward(&input, &mut ctx)
        .unwrap()
        .to_vec()
        .unwrap();
    assert_eq!(found, expected);

    let mut missing_records = state.clone().into_records();
    missing_records.pop();
    let err = StateDict::new(missing_records)
        .unwrap()
        .load_module(&mut restored)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Persistence(PersistenceError::MissingTensor { .. })
    ));

    let mut unexpected_records = state.clone().into_records();
    unexpected_records.push(TensorRecord::from_values("extra", vec![1], &[0.0f32]).unwrap());
    let err = StateDict::new(unexpected_records)
        .unwrap()
        .load_module(&mut restored)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Persistence(PersistenceError::UnexpectedTensor { .. })
    ));

    let duplicate_err =
        StateDict::new(vec![state.records()[0].clone(), state.records()[0].clone()]).unwrap_err();
    assert!(matches!(
        duplicate_err,
        Error::Persistence(PersistenceError::DuplicateTensor { .. })
    ));

    let mut wrong_dims = state.clone().into_records();
    let first_name = wrong_dims[0].name().to_owned();
    let first_len = wrong_dims[0].dims().iter().product();
    wrong_dims[0] =
        TensorRecord::from_values(first_name, vec![first_len], &vec![0.0f32; first_len]).unwrap();
    let err = StateDict::new(wrong_dims)
        .unwrap()
        .load_module(&mut restored)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Persistence(PersistenceError::ShapeMismatch { .. })
    ));

    let mut wrong_dtype = state.into_records();
    let first_name = wrong_dtype[0].name().to_owned();
    let first_dims = wrong_dtype[0].dims().to_vec();
    let first_len = first_dims.iter().product();
    wrong_dtype[0] =
        TensorRecord::from_values(first_name, first_dims, &vec![0.0f64; first_len]).unwrap();
    let err = StateDict::new(wrong_dtype)
        .unwrap()
        .load_module(&mut restored)
        .unwrap_err();
    assert!(matches!(
        err,
        Error::Persistence(PersistenceError::DTypeMismatch { .. })
    ));
}

#[test]
fn conv2d_state_dict_handles_optional_bias_strictly() {
    let with_bias = Conv2d::<1, 2, 2, 2, 1, 1>::zeros(Conv2dOptions::default()).unwrap();
    let state = StateDict::from_module(&with_bias).unwrap();
    assert_eq!(
        state
            .records()
            .iter()
            .map(TensorRecord::name)
            .collect::<Vec<_>>(),
        vec!["weight", "bias"]
    );

    let mut restored = Conv2d::<1, 2, 2, 2, 1, 1>::zeros(Conv2dOptions::default()).unwrap();
    state.load_module(&mut restored).unwrap();

    let mut without_bias =
        Conv2d::<1, 2, 2, 2, 1, 1>::zeros_without_bias(Conv2dOptions::default()).unwrap();
    let err = state.load_module(&mut without_bias).unwrap_err();
    assert!(matches!(
        err,
        Error::Persistence(PersistenceError::UnexpectedTensor { .. })
    ));

    let without_bias_state = StateDict::from_module(&without_bias).unwrap();
    let err = without_bias_state.load_module(&mut restored).unwrap_err();
    assert!(matches!(
        err,
        Error::Persistence(PersistenceError::MissingTensor { .. })
    ));
}

#[test]
fn optimizer_state_rejects_wrong_kind_and_different_parameter_set() {
    let mut layer = Linear::<1, 1>::zeros().unwrap();
    let sgd_state = Sgd::with_momentum(0.1, 0.9).state_dict(&layer).unwrap();
    let mut adamw = AdamW::new(1.0, 1.0);
    let err = adamw.load_state_dict(&layer, &sgd_state).unwrap_err();
    assert!(matches!(
        err,
        Error::Persistence(PersistenceError::OptimizerMismatch { .. })
    ));
    assert_eq!(adamw.lr(), 1.0);

    let mut trained = AdamW::new(0.05, 0.0);
    train_step(&mut layer, &mut trained);
    let adamw_state = trained.state_dict(&layer).unwrap();
    let wider = Linear::<2, 1>::zeros().unwrap();
    let mut fresh = AdamW::new(0.25, 0.0);
    let err = fresh.load_state_dict(&wider, &adamw_state).unwrap_err();
    assert!(matches!(
        err,
        Error::Persistence(PersistenceError::ShapeMismatch { .. })
    ));
    assert_eq!(fresh.lr(), 0.25);
}

#[test]
fn transformer_state_dict_round_trip_preserves_loss() {
    let mut rng = SmallRng::seed_from_u64(21);
    let model = DecoderOnlyTransformer::<6, 3, 4, 2, 2, 8, 2>::new(&mut rng).unwrap();
    let input = [[2, 4, 5], [4, 5, 3]];
    let target = [[4, 5, 3], [5, 3, 0]];
    let expected = model.loss(&input, &target).unwrap().to_vec().unwrap();

    let state = StateDict::from_module(&model).unwrap();
    let mut other_rng = SmallRng::seed_from_u64(99);
    let mut restored = DecoderOnlyTransformer::<6, 3, 4, 2, 2, 8, 2>::new(&mut other_rng).unwrap();
    state.load_module(&mut restored).unwrap();

    assert_eq!(
        restored.loss(&input, &target).unwrap().to_vec().unwrap(),
        expected
    );
}

#[test]
fn checkpoint_resume_matches_uninterrupted_adamw_training() {
    let mut uninterrupted = Linear::<1, 1>::zeros().unwrap();
    let mut uninterrupted_opt = AdamW::new(0.05, 0.01);
    let mut interrupted = Linear::<1, 1>::zeros().unwrap();
    let mut interrupted_opt = AdamW::new(0.05, 0.01);

    for _ in 0..3 {
        train_step(&mut uninterrupted, &mut uninterrupted_opt);
        train_step(&mut interrupted, &mut interrupted_opt);
    }

    let path = test_path("checkpoint");
    let mut rng = SmallRng::seed_from_u64(123);
    let _: f32 = rng.uniform(0.0, 1.0);
    let saved_rng_state = rng.state();
    save_checkpoint(
        &path,
        &interrupted,
        &interrupted_opt,
        3,
        Some(&rng),
        BTreeMap::from([("epoch".to_owned(), "13".to_owned())]),
    )
    .unwrap();

    let mut restored = Linear::<1, 1>::zeros().unwrap();
    let mut restored_opt = AdamW::new(1.0, 1.0);
    let mut restored_rng = SmallRng::seed_from_u64(0);
    let checkpoint = load_checkpoint(
        &path,
        &mut restored,
        &mut restored_opt,
        Some(&mut restored_rng),
    )
    .unwrap();
    assert_eq!(checkpoint.scheduler_step(), 3);
    assert_eq!(restored_rng.state(), saved_rng_state);
    assert_eq!(checkpoint.metadata().get("epoch").unwrap(), "13");

    for _ in 0..4 {
        train_step(&mut uninterrupted, &mut uninterrupted_opt);
        train_step(&mut restored, &mut restored_opt);
        assert_eq!(
            scalar_loss(&restored).to_vec().unwrap()[0].to_bits(),
            scalar_loss(&uninterrupted).to_vec().unwrap()[0].to_bits()
        );
    }
}

#[test]
fn malformed_state_files_return_structured_errors() {
    let wrong_magic = test_path("wrong-magic");
    let mut bytes = Vec::from(&b"BADSIGN\0"[..]);
    bytes.extend_from_slice(&1u32.to_le_bytes());
    fs::write(&wrong_magic, bytes).unwrap();
    let err = StateDict::load_from_path(&wrong_magic).unwrap_err();
    assert!(matches!(
        err,
        Error::Persistence(PersistenceError::InvalidMagic { .. })
    ));

    let wrong_version = test_path("wrong-version");
    let mut bytes = Vec::from(&b"RSTSD13\0"[..]);
    bytes.extend_from_slice(&2u32.to_le_bytes());
    fs::write(&wrong_version, bytes).unwrap();
    let err = StateDict::load_from_path(&wrong_version).unwrap_err();
    assert!(matches!(
        err,
        Error::Persistence(PersistenceError::UnsupportedVersion { .. })
    ));

    let truncated = test_path("truncated");
    fs::write(&truncated, b"RSTSD13\0").unwrap();
    let err = StateDict::load_from_path(&truncated).unwrap_err();
    assert!(matches!(
        err,
        Error::Persistence(PersistenceError::Io { .. })
    ));

    let trailing = test_path("trailing");
    StateDict::from_module(&Linear::<1, 1>::zeros().unwrap())
        .unwrap()
        .save_to_path(&trailing)
        .unwrap();
    let mut bytes = fs::read(&trailing).unwrap();
    bytes.push(0);
    fs::write(&trailing, bytes).unwrap();
    let err = StateDict::load_from_path(&trailing).unwrap_err();
    assert!(matches!(
        err,
        Error::Persistence(PersistenceError::UnexpectedTrailingBytes)
    ));
}

#[test]
fn buffer_state_dict_round_trips_and_optimizer_skips_buffers() {
    let mut bn = BatchNorm2d::<2>::new().unwrap();

    // Train pass updates running_mean / running_var away from their initial values
    let input = Tensor::<D4<Sym<Batch>, C<2>, AnyDim, AnyDim>>::from_vec_with_shape(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        [2, 2, 1, 2],
    )
    .unwrap();
    let mut train_ctx = TrainContext::training(0);
    bn.forward(&input, &mut train_ctx).unwrap();

    // Collect buffer names from the state dict
    let state = StateDict::from_module(&bn).unwrap();
    let record_names: Vec<_> = state.records().iter().map(TensorRecord::name).collect();
    assert!(
        record_names.contains(&"running_mean"),
        "running_mean must appear in state dict; got {record_names:?}"
    );
    assert!(
        record_names.contains(&"running_var"),
        "running_var must appear in state dict; got {record_names:?}"
    );

    // Load into a fresh module and verify eval output matches
    let mut restored = BatchNorm2d::<2>::new().unwrap();
    state.load_module(&mut restored).unwrap();

    let mut eval_ctx = TrainContext::eval();
    let out_orig = bn.forward(&input, &mut eval_ctx).unwrap().to_vec().unwrap();
    let out_restored = restored
        .forward(&input, &mut eval_ctx)
        .unwrap()
        .to_vec()
        .unwrap();
    assert_eq!(
        out_orig, out_restored,
        "eval output must match after buffer round-trip"
    );

    // Optimizer must not see buffers — only weight and bias (2 parameters)
    let mut params = Vec::new();
    bn.parameters_mut(&mut params);
    assert_eq!(
        params.len(),
        2,
        "only weight and bias should be exposed to optimizer; buffers must be excluded"
    );

    // Name-mismatch: StateDict missing a buffer record → MissingTensor error
    let mut records = state.into_records();
    records.retain(|r| r.name() != "running_mean");
    let err = StateDict::new(records)
        .unwrap()
        .load_module(&mut restored)
        .unwrap_err();
    assert!(
        matches!(
            err,
            Error::Persistence(PersistenceError::MissingTensor { .. })
        ),
        "missing buffer record must yield MissingTensor error"
    );
}

fn round_trip_linear_state<E>()
where
    E: FloatDType + std::fmt::Debug,
{
    let mut rng = SmallRng::seed_from_u64(17);
    let model = Linear::<2, 1, E>::xavier_uniform(&mut rng).unwrap();
    let expected = parameter_data::<_, E>(&model);
    let mut restored = Linear::<2, 1, E>::zeros().unwrap();
    StateDict::from_module(&model)
        .unwrap()
        .load_module(&mut restored)
        .unwrap();
    assert_eq!(parameter_data::<_, E>(&restored), expected);
}

fn parameter_data<M, E>(module: &M) -> Vec<(String, Vec<E>)>
where
    M: HasParameters<E, Cpu>,
    E: FloatDType,
{
    let mut out = Vec::new();
    module.visit_parameters("", &mut |name, param| {
        out.push((name.to_owned(), param.data().unwrap()));
    });
    out
}

fn round_trip_f16() {
    let values = [
        f16::from_f32(1.5),
        f16::from_bits(0x8000),
        f16::from_f32(-2.25),
    ];
    let tensor = Tensor1D::<3, f16>::from_vec(values.to_vec()).unwrap();
    let path = test_path("f16-tensor");
    tensor.save(&path).unwrap();
    let loaded = Tensor1D::<3, f16>::load(&path).unwrap();
    assert_eq!(
        loaded
            .to_vec()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

fn round_trip_bf16() {
    let values = [
        bf16::from_f32(1.5),
        bf16::from_bits(0x8000),
        bf16::from_f32(-2.25),
    ];
    let tensor = Tensor1D::<3, bf16>::from_vec(values.to_vec()).unwrap();
    let path = test_path("bf16-tensor");
    tensor.save(&path).unwrap();
    let loaded = Tensor1D::<3, bf16>::load(&path).unwrap();
    assert_eq!(
        loaded
            .to_vec()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

fn round_trip_f32() {
    let values = [1.5f32, -0.0, -2.25];
    let tensor = Tensor1D::<3>::from_vec(values.to_vec()).unwrap();
    let path = test_path("f32-tensor");
    tensor.save(&path).unwrap();
    let loaded = Tensor1D::<3>::load(&path).unwrap();
    assert_eq!(
        loaded
            .to_vec()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

fn round_trip_f64() {
    let values = [1.5f64, -0.0, -2.25];
    let tensor = Tensor1D::<3, f64>::from_vec(values.to_vec()).unwrap();
    let path = test_path("f64-tensor");
    tensor.save(&path).unwrap();
    let loaded = Tensor1D::<3, f64>::load(&path).unwrap();
    assert_eq!(
        loaded
            .to_vec()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

fn round_trip_i64() {
    let values = [i64::MIN, 0, 9_007_199_254_740_993];
    let tensor = Tensor1D::<3, i64>::from_vec(values.to_vec()).unwrap();
    let path = test_path("i64-tensor");
    tensor.save(&path).unwrap();
    let loaded = Tensor1D::<3, i64>::load(&path).unwrap();
    assert_eq!(loaded.to_vec().unwrap(), values);
}

fn train_step(layer: &mut Linear<1, 1>, opt: &mut AdamW<f32>) {
    let mut params = Vec::new();
    layer.parameters(&mut params);
    opt.zero_grad(&params);
    drop(params);

    scalar_loss(layer).backward().unwrap();
    let mut params = Vec::new();
    layer.parameters_mut(&mut params);
    opt.step(&mut params).unwrap();
}

fn scalar_loss(layer: &Linear<1, 1>) -> Scalar {
    let input = Tensor2D::<1, 1>::from_vec(vec![2.0]).unwrap();
    let target = Tensor2D::<1, 1>::from_vec(vec![4.0]).unwrap();
    let mut ctx = TrainContext::eval();
    mse_loss(&layer.forward(&input, &mut ctx).unwrap(), &target).unwrap()
}

fn test_path(name: &str) -> PathBuf {
    let idx = NEXT_PATH.fetch_add(1, Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("rstorch-{name}-{}-{idx}.bin", std::process::id()));
    let _ = fs::remove_file(&path);
    path
}
