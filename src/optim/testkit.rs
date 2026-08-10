//! Fixtures shared by the optimizer test suites (test builds only).
//!
//! The layer zoo (`Linear`, `LayerNorm`, ...) lives elsewhere, so these
//! models spell their arithmetic out by hand. Their *field names* matter: the
//! group-predicate tests match on `bias` and `norm`, which is the standard
//! transformer recipe the design calls out.

use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::nn::{Mode, Module, Param};
use crate::persist::Envelope;
use crate::tensor::Tensor;

use super::engine::{Engine, Rule};

pub(crate) const CPU: Device = Device::Cpu;

/// A 1-D `f32` tensor on the CPU.
pub(crate) fn t(values: &[f32]) -> Tensor {
    Tensor::from_vec(values.to_vec(), [values.len()], &CPU).unwrap()
}

/// The host values of a tensor.
pub(crate) fn values(tensor: &Tensor) -> Vec<f32> {
    tensor.to_vec::<f32>().unwrap()
}

/// A scalar tensor's value.
pub(crate) fn scalar(tensor: &Tensor) -> f64 {
    tensor.item().unwrap()
}

/// The affine map `y = x·w + b`, the smallest thing that can actually converge.
///
/// `weight` is `[1, 1]` and `bias` is `[1]`, so a `[n, 1]` batch of inputs maps
/// to a `[n, 1]` batch of predictions and the whole model is two numbers.
#[derive(rstorch::Module)]
pub(crate) struct Affine {
    pub(crate) weight: Param,
    pub(crate) bias: Param,
}

/// The four inputs the convergence tests fit `y = 3x + 2` on.
pub(crate) const REGRESSION_XS: [f32; 4] = [-1.0, 0.0, 1.0, 2.0];

impl Affine {
    /// Start at `w = b = 0`, far from the target the tests fit.
    pub(crate) fn zeros() -> Affine {
        Affine {
            weight: Param::new(Tensor::zeros([1, 1], DType::F32, &CPU).unwrap()),
            bias: Param::new(Tensor::zeros([1], DType::F32, &CPU).unwrap()),
        }
    }

    /// Mean-squared error of this model's predictions on `xs` against
    /// `3·x + 2`, the line the convergence tests fit.
    pub(crate) fn loss(&self, xs: &[f32], mode: Mode) -> Result<Tensor> {
        let n = xs.len();
        let x = Tensor::from_vec(xs.to_vec(), [n, 1], &CPU)?;
        let targets: Vec<f32> = xs.iter().map(|v| 3.0 * v + 2.0).collect();
        let y = Tensor::from_vec(targets, [n, 1], &CPU)?;
        let pred = x
            .matmul(&self.weight.get(mode))?
            .add(&self.bias.get(mode))?;
        pred.mse_loss(&y)
    }
}

/// A single parameter under the loss `Σ w²`, whose gradient `2w` **changes** as
/// the parameter moves.
///
/// That is the point: an update formula pinned against a *constant* gradient can
/// be satisfied by several wrong formulas (a mis-ordered bias correction, decay
/// applied to the wrong operand), because every step looks the same. With `2w`
/// the trajectory is sensitive to all of it, so a test can compare it against an
/// independent scalar implementation of the algorithm and mean it.
#[derive(rstorch::Module)]
pub(crate) struct Solo {
    pub(crate) w: Param,
}

impl Solo {
    /// One parameter holding `value`.
    pub(crate) fn new(value: f32) -> Solo {
        Solo {
            w: Param::new(t(&[value])),
        }
    }

    /// `Σ w²`, so `dL/dw = 2w`.
    pub(crate) fn square_loss(&self, mode: Mode) -> Result<Tensor> {
        let w = self.w.get(mode);
        w.mul(&w)?.sum_all()
    }

    /// The parameter's current scalar value.
    pub(crate) fn value(&self) -> f64 {
        f64::from(values(self.w.value())[0])
    }
}

/// A two-level model whose dotted paths cover what group predicates select on:
/// `trunk.weight`, `trunk.bias`, `norm.weight`, `head.weight`, `head.bias`.
#[derive(rstorch::Module)]
pub(crate) struct Net {
    pub(crate) trunk: Block,
    pub(crate) norm: Gain,
    pub(crate) head: Block,
}

/// A weight/bias pair.
#[derive(rstorch::Module)]
pub(crate) struct Block {
    pub(crate) weight: Param,
    pub(crate) bias: Param,
}

/// A lone gain, standing in for a normalization layer's scale.
#[derive(rstorch::Module)]
pub(crate) struct Gain {
    pub(crate) weight: Param,
}

impl Net {
    /// Every parameter starts at `1.0`, so a multiplicative or additive update
    /// is visible in the value itself.
    pub(crate) fn ones() -> Net {
        let p = || Param::new(t(&[1.0]));
        Net {
            trunk: Block {
                weight: p(),
                bias: p(),
            },
            norm: Gain { weight: p() },
            head: Block {
                weight: p(),
                bias: p(),
            },
        }
    }

    /// A loss whose gradient is exactly `1.0` at every parameter: the sum of
    /// all five. That makes each optimizer's update formula readable straight
    /// off the resulting values.
    pub(crate) fn unit_grad_loss(&self, mode: Mode) -> Result<Tensor> {
        let mut acc = self.trunk.weight.get(mode);
        for p in [
            &self.trunk.bias,
            &self.norm.weight,
            &self.head.weight,
            &self.head.bias,
        ] {
            acc = acc.add(&p.get(mode))?;
        }
        acc.sum_all()
    }

    /// The **untraced-weight scenario**: a forward that reads
    /// [`Param::value`](crate::nn::Param::value) for `head.bias` instead of
    /// [`Param::get`](crate::nn::Param::get), which is exactly how a real model
    /// stops training one weight without any other symptom. Its gradient is
    /// therefore absent from the resulting `Grads`.
    pub(crate) fn untraced_head_bias_loss(&self, mode: Mode) -> Result<Tensor> {
        let mut acc = self.trunk.weight.get(mode);
        for p in [&self.trunk.bias, &self.norm.weight, &self.head.weight] {
            acc = acc.add(&p.get(mode))?;
        }
        // The bug: the value, not the traced leaf.
        acc.add(self.head.bias.value())?.sum_all()
    }

    /// Every parameter's value, in `state_dict` (sorted-path) order.
    pub(crate) fn snapshot(&self) -> Vec<(String, f32)> {
        crate::nn::state_dict(self)
            .into_iter()
            .map(|(path, tensor)| (path, values(&tensor)[0]))
            .collect()
    }

    /// One parameter's scalar value, by dotted path.
    pub(crate) fn at(&self, path: &str) -> f32 {
        self.snapshot()
            .into_iter()
            .find(|(p, _)| p == path)
            .unwrap_or_else(|| panic!("no parameter at `{path}`"))
            .1
    }
}

/// `assert!` on `|a - b| < tol`, reporting both sides.
pub(crate) fn close(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() < tol, "{a} vs {b} (tolerance {tol})");
}

/// One successful update of `model` under the unit-gradient loss.
pub(crate) fn step_once<R: Rule>(opt: &mut Engine<R>, model: &mut Net) {
    let loss = model.unit_grad_loss(Mode::TRAIN).unwrap();
    opt.step(model, loss.backward().unwrap()).unwrap();
}

/// The error one update of `model` is refused with.
fn refuse<R: Rule>(opt: &mut Engine<R>, model: &mut Net) -> Error {
    let loss = model.unit_grad_loss(Mode::TRAIN).unwrap();
    opt.step(model, loss.backward().unwrap()).unwrap_err()
}

/// Every byte of an optimizer's state, as one value two snapshots can compare:
/// moment buffers and per-parameter clocks both ride in here.
fn checkpoint<R: Rule>(opt: &Engine<R>, model: &dyn Module) -> Envelope {
    let mut envelope = Envelope::new();
    opt.save(model, &mut envelope).unwrap();
    envelope
}

/// The step `opt` is about to take must be refused with a message containing
/// `expected`, and refusing it must leave the model and every byte of optimizer
/// state exactly as they are.
pub(crate) fn assert_next_step_is_refused<R: Rule>(
    opt: &mut Engine<R>,
    model: &mut Net,
    expected: &str,
) {
    let params = model.snapshot();
    let before = checkpoint(opt, model);
    let err = refuse(opt, model);
    assert!(matches!(err, Error::InvalidArg { op: "step", .. }), "{err}");
    assert!(err.to_string().contains(expected), "{err}");
    assert_eq!(model.snapshot(), params, "{err}");
    assert_eq!(checkpoint(opt, model), before, "{err}");
}

/// Every rejection this layer can diagnose is raised before the first
/// `Param::set`, as `optim`'s module docs state normatively: seed a run of
/// `build()` with two updates, arrange each rejection in turn, and assert the
/// refused step moved nothing at all — not a parameter, not a moment buffer,
/// not a clock, not `steps`.
pub(crate) fn assert_every_rejection_is_atomic<R: Rule>(build: impl Fn() -> Engine<R>) {
    fn check<R: Rule>(
        mut opt: Engine<R>,
        derail: impl FnOnce(&mut Engine<R>, &mut Net),
        attempt: impl FnOnce(&mut Engine<R>, &mut Net) -> Error,
    ) {
        let mut model = Net::ones();
        for _ in 0..2 {
            step_once(&mut opt, &mut model);
        }
        derail(&mut opt, &mut model);
        let params = model.snapshot();
        let before = checkpoint(&opt, &model);

        let err = attempt(&mut opt, &mut model);

        assert_eq!(model.snapshot(), params, "{err}");
        assert_eq!(checkpoint(&opt, &model), before, "{err}");
        assert_eq!(opt.steps(), 2, "{err}");
    }

    // A missing gradient (the untraced-weight bug)…
    check(
        build(),
        |_, _| {},
        |opt, model| {
            let loss = model.untraced_head_bias_loss(Mode::TRAIN).unwrap();
            opt.step(model, loss.backward().unwrap()).unwrap_err()
        },
    );
    // …a per-parameter step clock with no room left…
    check(
        build(),
        |opt, model| opt.set_clock(&model.head.bias, u64::MAX),
        refuse,
    );
    // …and an out-of-range hyperparameter.
    check(build(), |opt, _| opt.set_lr(-1.0), refuse);
}

/// The pre-pass's two bailouts are *ordered*, and which one a caller sees is
/// observable: an exhausted clock outranks an out-of-range hyperparameter even
/// though the walk meets the bad hyperparameter first (`trunk.weight` leads,
/// `head.bias` trails), because the whole walk runs before either is reported.
pub(crate) fn assert_an_exhausted_clock_outranks_an_invalid_hyperparameter<R: Rule>(
    mut opt: Engine<R>,
) {
    let mut model = Net::ones();
    step_once(&mut opt, &mut model);
    // Invalid for every parameter, including the first one visited…
    opt.set_lr(-1.0);
    // …but the last one visited has no clock left.
    opt.set_clock(&model.head.bias, u64::MAX);

    let err = refuse(&mut opt, &mut model);
    assert!(
        err.to_string()
            .contains(&format!("{} step clock for parameter `head.bias`", R::NAME)),
        "{err}"
    );
}

/// A run checkpointed halfway and resumed by a *different* optimizer value must
/// land exactly where an uninterrupted run of six steps would have.
///
/// `resumed` is built with deliberately wrong hyperparameters, so the assertion
/// also pins that the checkpoint's win. `tag` names the temp directory.
pub(crate) fn assert_state_round_trips<R: Rule>(
    tag: &str,
    build: impl Fn() -> Engine<R>,
    mut resumed: Engine<R>,
) {
    let dir = tmpdir(tag);
    let path = dir.join("run.rstorch");

    // The reference: six uninterrupted steps.
    let mut reference_model = Net::ones();
    let mut reference = build();
    for _ in 0..6 {
        step_once(&mut reference, &mut reference_model);
    }

    // The resumed run: three steps, a checkpoint, then three more.
    let mut model = Net::ones();
    let mut opt = build();
    for _ in 0..3 {
        step_once(&mut opt, &mut model);
    }
    checkpoint(&opt, &model)
        .save(&path, &crate::persist::Limits::defaults())
        .unwrap();

    let loaded = Envelope::load(&path, &crate::persist::Limits::defaults()).unwrap();
    resumed.load(&model, &loaded).unwrap();
    close(resumed.lr(), build().lr(), 1e-12);
    assert_eq!(resumed.steps(), 3);
    assert_eq!(resumed.param_steps(&model.trunk.weight), 3);
    for _ in 0..3 {
        step_once(&mut resumed, &mut model);
    }

    assert_eq!(model.snapshot(), reference_model.snapshot());
    std::fs::remove_dir_all(&dir).unwrap();
}

/// The gate every optimizer has to clear: `iterations` of `step` on an
/// [`Affine`] starting from zero must drive the loss down and find `y = 3x + 2`.
pub(crate) fn assert_converges_on_a_tiny_regression(
    iterations: usize,
    mut step: impl FnMut(&mut Affine),
) {
    let mut model = Affine::zeros();
    let loss = |model: &Affine| scalar(&model.loss(&REGRESSION_XS, Mode::TRAIN).unwrap());

    let first = loss(&model);
    let mut latest = first;
    for _ in 0..iterations {
        step(&mut model);
        latest = loss(&model);
    }
    assert!(latest < first, "loss rose: {first} -> {latest}");
    assert!(latest < 1e-4, "did not converge: {latest}");
    // It found the line: y = 3x + 2.
    close(f64::from(values(model.weight.value())[0]), 3.0, 1e-2);
    close(f64::from(values(model.bias.value())[0]), 2.0, 1e-2);
}

/// A copy of `envelope` whose global `steps` and every per-parameter clock have
/// been rewritten, so a test can present a checkpoint no honest run could reach
/// without taking 2⁶⁴ steps to get there.
pub(crate) fn with_clocks(envelope: &Envelope, steps: u64, clocks: u64) -> Envelope {
    let mut rewritten = Envelope::new();
    let section = envelope
        .section("optimizer")
        .unwrap()
        .lines()
        .map(|line| {
            if line.starts_with("steps=") {
                format!("steps={steps}")
            } else if let Some((key, _)) = line.split_once('=')
                && key.starts_with("clock.")
            {
                format!("{key}={clocks}")
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    rewritten.set_section("optimizer", section).unwrap();
    for (key, tensor) in envelope.tensors() {
        rewritten.insert_tensor(key.clone(), tensor.clone());
    }
    rewritten
}

/// A private temp directory for a checkpoint round-trip, removed by the caller.
pub(crate) fn tmpdir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rstorch-optim-{}-{}-{}",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}
