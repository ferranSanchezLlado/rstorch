use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use rstorch::nn::{self, Forward, Linear, Mode};
use rstorch::optim::{AdamW, Sgd};
use rstorch::persist::{Envelope, Limits};
use rstorch::testing::check_grad;
use rstorch::{DType, Device, Result, Rng, Tensor};

static NEXT_PATH: AtomicU64 = AtomicU64::new(0);

fn reduced(values: &[f32], dims: impl Into<rstorch::Shape>, dtype: DType) -> Result<Tensor> {
    match dtype {
        DType::F16 => Tensor::from_vec(
            values.iter().copied().map(half::f16::from_f32).collect(),
            dims,
            &Device::Cpu,
        ),
        DType::BF16 => Tensor::from_vec(
            values.iter().copied().map(half::bf16::from_f32).collect(),
            dims,
            &Device::Cpu,
        ),
        _ => unreachable!("reduced test helper requires F16 or BF16"),
    }
}

fn checkpoint_path(dtype: DType) -> PathBuf {
    let id = NEXT_PATH.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rstorch-t60-{}-{}-{id}.safetensors",
        std::process::id(),
        dtype
    ))
}

fn assert_same_model(lhs: &dyn nn::Module, rhs: &dyn nn::Module) {
    let lhs = nn::state_dict(lhs);
    let rhs = nn::state_dict(rhs);
    assert_eq!(
        lhs.keys().collect::<Vec<_>>(),
        rhs.keys().collect::<Vec<_>>()
    );
    for (path, left) in lhs {
        let right = &rhs[&path];
        assert_eq!(left.dtype(), right.dtype(), "{path}");
        match left.dtype() {
            DType::F16 => assert_eq!(
                left.to_vec::<half::f16>().unwrap(),
                right.to_vec::<half::f16>().unwrap(),
                "{path}"
            ),
            DType::BF16 => assert_eq!(
                left.to_vec::<half::bf16>().unwrap(),
                right.to_vec::<half::bf16>().unwrap(),
                "{path}"
            ),
            other => panic!("reduced model has unexpected dtype {other}"),
        }
    }
}

fn assert_wide_state_and_clocks(envelope: &Envelope, clock: u64) {
    assert!(!envelope.tensors().is_empty());
    assert!(
        envelope
            .tensors()
            .values()
            .all(|tensor| tensor.dtype() == DType::F32)
    );
    let clocks: Vec<_> = envelope
        .section("optimizer")
        .unwrap()
        .lines()
        .filter(|line| line.starts_with("clock."))
        .collect();
    assert!(!clocks.is_empty());
    assert!(
        clocks
            .iter()
            .all(|line| line.ends_with(&format!("={clock}"))),
        "{clocks:?}"
    );
}

#[test]
fn reduced_losses_and_autograd_match_finite_differences() {
    for dtype in [DType::F16, DType::BF16] {
        let x = reduced(&[0.5, -1.0, 1.5, 0.25], [2, 2], dtype).unwrap();
        let target = reduced(&[0.25, -0.5, 1.0, -0.25], [2, 2], dtype).unwrap();
        check_grad(
            |inputs| inputs[0].mse_loss(&inputs[1]),
            &[x, target],
            if dtype == DType::F16 { 0.05 } else { 0.25 },
            if dtype == DType::F16 { 0.03 } else { 0.12 },
        )
        .unwrap();

        let weights = reduced(&[0.5, -1.0, 0.25, 0.75], [2, 2], dtype).unwrap();
        check_grad(
            |inputs| inputs[0].softmax(-1)?.mul(&inputs[1])?.sum_all(),
            &[
                reduced(&[1.0, -0.5, 0.25, 2.0], [2, 2], dtype).unwrap(),
                weights,
            ],
            if dtype == DType::F16 { 0.05 } else { 0.25 },
            if dtype == DType::F16 { 0.04 } else { 0.15 },
        )
        .unwrap();

        let logits = reduced(&[0.5, -0.25, 1.0, -1.0, 0.75, 0.25], [2, 3], dtype).unwrap();
        let labels = Tensor::from_vec(vec![2i64, 1], [2], &Device::Cpu).unwrap();
        check_grad(
            |inputs| inputs[0].cross_entropy(&inputs[1]),
            &[logits, labels],
            if dtype == DType::F16 { 0.05 } else { 0.25 },
            if dtype == DType::F16 { 0.04 } else { 0.15 },
        )
        .unwrap();
    }
}

#[test]
fn reduced_loss_means_keep_large_reductions_wide() {
    for dtype in [DType::F16, DType::BF16] {
        let rows = 100_000;
        let prediction = Tensor::ones([rows], dtype, &Device::Cpu).unwrap();
        let target = Tensor::zeros([rows], dtype, &Device::Cpu).unwrap();
        assert_eq!(prediction.mse_loss(&target).unwrap().item().unwrap(), 1.0);

        let logits = Tensor::zeros([rows, 2], dtype, &Device::Cpu).unwrap();
        let labels = Tensor::zeros([rows], DType::I64, &Device::Cpu).unwrap();
        let one_row = Tensor::zeros([1, 2], dtype, &Device::Cpu)
            .unwrap()
            .cross_entropy(&Tensor::zeros([1], DType::I64, &Device::Cpu).unwrap())
            .unwrap()
            .item()
            .unwrap();
        let large = logits.cross_entropy(&labels).unwrap().item().unwrap();
        assert!(large.is_finite());
        assert!(
            (large - one_row).abs() < 0.002,
            "{dtype}: {large} vs {one_row}"
        );
    }
}

#[test]
fn reduced_float_casts_are_differentiable_in_both_directions() {
    for dtype in [DType::F16, DType::BF16] {
        let f32_input = Tensor::from_vec(vec![0.5f32, -1.5, 2.0], [3], &Device::Cpu)
            .unwrap()
            .traced()
            .unwrap();
        let f32_grad = f32_input
            .to_dtype(dtype)
            .unwrap()
            .mul_scalar(2.0)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap()
            .wrt_input(&f32_input)
            .unwrap();
        assert_eq!(f32_grad.dtype(), DType::F32);
        assert_eq!(f32_grad.to_vec::<f32>().unwrap(), vec![2.0; 3]);

        let reduced_input = reduced(&[0.5, -1.5, 2.0], [3], dtype)
            .unwrap()
            .traced()
            .unwrap();
        let reduced_grad = reduced_input
            .to_dtype(DType::F32)
            .unwrap()
            .mul_scalar(3.0)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap()
            .wrt_input(&reduced_input)
            .unwrap();
        assert_eq!(reduced_grad.dtype(), dtype);
        assert!(
            reduced_grad
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|&value| value == 3.0)
        );
    }
}

#[test]
fn reduced_mse_backward_scales_once_in_f32_and_preserves_target_sign() {
    let n = 1000;
    for dtype in [DType::F16, DType::BF16] {
        let prediction = Tensor::full([n], 0.1, dtype, &Device::Cpu)
            .unwrap()
            .traced()
            .unwrap();
        let target = Tensor::zeros([n], dtype, &Device::Cpu)
            .unwrap()
            .traced()
            .unwrap();
        let grads = prediction
            .mse_loss(&target)
            .unwrap()
            .mul_scalar(0.1)
            .unwrap()
            .backward()
            .unwrap();
        let prediction_grad = grads.wrt_input(&prediction).unwrap();
        let target_grad = grads.wrt_input(&target).unwrap();
        match dtype {
            DType::F16 => {
                let input = half::f16::from_f32(0.1).to_f32();
                let upstream = half::f16::from_f32(0.1).to_f32();
                let expected = half::f16::from_f32(input * (2.0 / n as f32) * upstream);
                assert!(
                    prediction_grad
                        .to_vec::<half::f16>()
                        .unwrap()
                        .iter()
                        .all(|&value| value == expected)
                );
                assert!(
                    target_grad
                        .to_vec::<half::f16>()
                        .unwrap()
                        .iter()
                        .all(|&value| value == -expected)
                );
            }
            DType::BF16 => {
                let input = half::bf16::from_f32(0.1).to_f32();
                let upstream = half::bf16::from_f32(0.1).to_f32();
                let expected = half::bf16::from_f32(input * (2.0 / n as f32) * upstream);
                assert!(
                    prediction_grad
                        .to_vec::<half::bf16>()
                        .unwrap()
                        .iter()
                        .all(|&value| value == expected)
                );
                assert!(
                    target_grad
                        .to_vec::<half::bf16>()
                        .unwrap()
                        .iter()
                        .all(|&value| value == -expected)
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn reduced_training_conversion_optimizer_state_and_resume_are_end_to_end() {
    for dtype in [DType::F16, DType::BF16] {
        let mut rng = Rng::seed(60);
        let mut model = Linear::new(1, 1, &Device::Cpu, &mut rng).unwrap();
        nn::to_dtype(&mut model, dtype).unwrap();
        assert!(
            nn::state_dict(&model)
                .values()
                .all(|value| value.dtype() == dtype)
        );

        let x = reduced(&[-1.0, -0.5, 0.5, 1.0], [4, 1], dtype).unwrap();
        let target = reduced(&[-2.0, -1.0, 1.0, 2.0], [4, 1], dtype).unwrap();
        let mut optimizer = AdamW::new(0.1, 0.01);
        let initial = model
            .forward(&x, Mode::EVAL)
            .unwrap()
            .mse_loss(&target)
            .unwrap()
            .item()
            .unwrap();
        for _ in 0..30 {
            let loss = model
                .forward(&x, Mode::TRAIN)
                .unwrap()
                .mse_loss(&target)
                .unwrap();
            optimizer
                .step(&mut model, loss.backward().unwrap())
                .unwrap();
        }
        let final_loss = model
            .forward(&x, Mode::EVAL)
            .unwrap()
            .mse_loss(&target)
            .unwrap()
            .item()
            .unwrap();
        assert!(
            final_loss < initial * 0.2,
            "{dtype}: {initial} -> {final_loss}"
        );

        let mut envelope = Envelope::new();
        optimizer.save_state(&model, &mut envelope).unwrap();
        assert!(
            envelope
                .tensors()
                .iter()
                .filter(|(path, _)| path.starts_with("optim."))
                .all(|(_, tensor)| tensor.dtype() == DType::F32)
        );
        let path = checkpoint_path(dtype);
        envelope.save(&path, &Limits::default()).unwrap();
        let loaded = Envelope::load(&path, &Limits::default()).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_wide_state_and_clocks(&loaded, 30);

        let mut resumed_model = Linear::new(1, 1, &Device::Cpu, &mut rng).unwrap();
        nn::to_dtype(&mut resumed_model, dtype).unwrap();
        nn::load_state_dict(&mut resumed_model, &nn::state_dict(&model)).unwrap();
        let mut resumed = AdamW::new(0.1, 0.01);
        resumed.load_state(&resumed_model, &loaded).unwrap();
        assert_eq!(resumed.steps(), optimizer.steps());

        let loss = resumed_model
            .forward(&x, Mode::TRAIN)
            .unwrap()
            .mse_loss(&target)
            .unwrap();
        resumed
            .step(&mut resumed_model, loss.backward().unwrap())
            .unwrap();
        assert!(
            resumed_model
                .forward(&x, Mode::EVAL)
                .unwrap()
                .mse_loss(&target)
                .unwrap()
                .item()
                .unwrap()
                .is_finite()
        );

        let mut sgd_model = Linear::new(1, 1, &Device::Cpu, &mut rng).unwrap();
        nn::to_dtype(&mut sgd_model, dtype).unwrap();
        let mut sgd = Sgd::new(0.05).momentum(0.9);
        let loss = sgd_model
            .forward(&x, Mode::TRAIN)
            .unwrap()
            .mse_loss(&target)
            .unwrap();
        sgd.step(&mut sgd_model, loss.backward().unwrap()).unwrap();
        let mut sgd_state = Envelope::new();
        sgd.save_state(&sgd_model, &mut sgd_state).unwrap();
        assert!(
            sgd_state
                .tensors()
                .values()
                .all(|tensor| tensor.dtype() == DType::F32)
        );
        let mut resumed_sgd = Sgd::new(1.0);
        resumed_sgd.load_state(&sgd_model, &sgd_state).unwrap();
        assert_eq!(resumed_sgd.steps(), 1);
    }
}

#[test]
fn reduced_optimizer_disk_resume_matches_uninterrupted_trajectory() {
    for dtype in [DType::F16, DType::BF16] {
        let mut rng = Rng::seed(600);
        let mut initial = Linear::new(1, 1, &Device::Cpu, &mut rng).unwrap();
        nn::to_dtype(&mut initial, dtype).unwrap();
        let initial_state = nn::state_dict(&initial);
        let x = reduced(&[-1.0, -0.5, 0.5, 1.0], [4, 1], dtype).unwrap();
        let target = reduced(&[-2.0, -1.0, 1.0, 2.0], [4, 1], dtype).unwrap();

        let make_model = |rng: &mut Rng| {
            let mut model = Linear::new(1, 1, &Device::Cpu, rng).unwrap();
            nn::to_dtype(&mut model, dtype).unwrap();
            nn::load_state_dict(&mut model, &initial_state).unwrap();
            model
        };

        let mut reference = make_model(&mut rng);
        let mut reference_opt = AdamW::new(0.05, 0.01);
        for _ in 0..8 {
            let loss = reference
                .forward(&x, Mode::TRAIN)
                .unwrap()
                .mse_loss(&target)
                .unwrap();
            reference_opt
                .step(&mut reference, loss.backward().unwrap())
                .unwrap();
        }

        let mut staged = make_model(&mut rng);
        let mut staged_opt = AdamW::new(0.05, 0.01);
        for _ in 0..4 {
            let loss = staged
                .forward(&x, Mode::TRAIN)
                .unwrap()
                .mse_loss(&target)
                .unwrap();
            staged_opt
                .step(&mut staged, loss.backward().unwrap())
                .unwrap();
        }
        let staged_model_state = nn::state_dict(&staged);
        let mut envelope = Envelope::new();
        staged_opt.save_state(&staged, &mut envelope).unwrap();
        assert_wide_state_and_clocks(&envelope, 4);
        let path = checkpoint_path(dtype);
        envelope.save(&path, &Limits::default()).unwrap();
        let loaded = Envelope::load(&path, &Limits::default()).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_wide_state_and_clocks(&loaded, 4);

        let mut resumed_model = make_model(&mut rng);
        nn::load_state_dict(&mut resumed_model, &staged_model_state).unwrap();
        let mut resumed_opt = AdamW::new(999.0, 0.01);
        resumed_opt.load_state(&resumed_model, &loaded).unwrap();
        assert_eq!(resumed_opt.steps(), 4);
        for _ in 0..4 {
            let loss = resumed_model
                .forward(&x, Mode::TRAIN)
                .unwrap()
                .mse_loss(&target)
                .unwrap();
            resumed_opt
                .step(&mut resumed_model, loss.backward().unwrap())
                .unwrap();
        }
        assert_same_model(&reference, &resumed_model);
        let mut reference_state = Envelope::new();
        reference_opt
            .save_state(&reference, &mut reference_state)
            .unwrap();
        let mut resumed_state = Envelope::new();
        resumed_opt
            .save_state(&resumed_model, &mut resumed_state)
            .unwrap();
        assert_wide_state_and_clocks(&resumed_state, 8);
        assert_eq!(reference_state.tensors(), resumed_state.tensors());
        assert_eq!(
            reference_state.section("optimizer"),
            resumed_state.section("optimizer")
        );

        let mut reference = make_model(&mut rng);
        let mut reference_opt = Sgd::new(0.05).momentum(0.9).weight_decay(0.01);
        for _ in 0..8 {
            let loss = reference
                .forward(&x, Mode::TRAIN)
                .unwrap()
                .mse_loss(&target)
                .unwrap();
            reference_opt
                .step(&mut reference, loss.backward().unwrap())
                .unwrap();
        }

        let mut staged = make_model(&mut rng);
        let mut staged_opt = Sgd::new(0.05).momentum(0.9).weight_decay(0.01);
        for _ in 0..4 {
            let loss = staged
                .forward(&x, Mode::TRAIN)
                .unwrap()
                .mse_loss(&target)
                .unwrap();
            staged_opt
                .step(&mut staged, loss.backward().unwrap())
                .unwrap();
        }
        let staged_model_state = nn::state_dict(&staged);
        let mut envelope = Envelope::new();
        staged_opt.save_state(&staged, &mut envelope).unwrap();
        assert_wide_state_and_clocks(&envelope, 4);
        let path = checkpoint_path(dtype);
        envelope.save(&path, &Limits::default()).unwrap();
        let loaded = Envelope::load(&path, &Limits::default()).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_wide_state_and_clocks(&loaded, 4);

        let mut resumed_model = make_model(&mut rng);
        nn::load_state_dict(&mut resumed_model, &staged_model_state).unwrap();
        let mut resumed_opt = Sgd::new(999.0);
        resumed_opt.load_state(&resumed_model, &loaded).unwrap();
        assert_eq!(resumed_opt.steps(), 4);
        for _ in 0..4 {
            let loss = resumed_model
                .forward(&x, Mode::TRAIN)
                .unwrap()
                .mse_loss(&target)
                .unwrap();
            resumed_opt
                .step(&mut resumed_model, loss.backward().unwrap())
                .unwrap();
        }
        assert_same_model(&reference, &resumed_model);
        let mut reference_state = Envelope::new();
        reference_opt
            .save_state(&reference, &mut reference_state)
            .unwrap();
        let mut resumed_state = Envelope::new();
        resumed_opt
            .save_state(&resumed_model, &mut resumed_state)
            .unwrap();
        assert_wide_state_and_clocks(&resumed_state, 8);
        assert_eq!(reference_state.tensors(), resumed_state.tensors());
        assert_eq!(
            reference_state.section("optimizer"),
            resumed_state.section("optimizer")
        );
    }
}
