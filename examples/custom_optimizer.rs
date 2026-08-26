//! Write your own optimizer, from public items only.
//!
//! `Rule`/`Engine` — the machinery `Sgd`/`Adam` share — are `pub(crate)`
//! (`src/optim/engine.rs`), so a third-party optimizer cannot plug into that
//! machinery. The generic parameter walk that machinery is built on is only
//! half-public: `Module::visit_mut` and `VisitorMut` are exported, but the
//! walk's sink type `LeafMut` and `VisitorMut::new` are `pub(crate)`
//! (`src/nn/visit.rs`), so a third-party type can *be* visited without being
//! able to start a visit. Neither omission closes the door: an
//! optimizer for **your own model** never needed a generic walk, because you already
//! wrote the struct and can name its `Param` fields directly.
//!
//! This example builds RMSprop for a hand-rolled one-layer model
//! (`y = w·x + b`) using only:
//!
//! - [`Param::get`]/[`Param::set`]/[`Param::is_frozen`] to read, write, and
//!   respect a frozen parameter,
//! - [`Grads::wrt`] to pull one parameter's cotangent out of the linear
//!   result [`Tensor::backward`] returns,
//! - [`Envelope::set_section`]/[`insert_tensor`](Envelope::insert_tensor) to
//!   persist the optimizer's own second-moment state alongside a model
//!   checkpoint, in the same file.
//!
//! # What this recipe does **not** inherit
//!
//! Two things `Sgd`/`Adam::step` give you for free that this hand-written
//! loop does not:
//!
//! 1. **The all-or-nothing pre-pass.** The built-in optimizers validate
//!    *every* parameter — a missing gradient, a shape/dtype/device mismatch —
//!    before touching any of them, so a rejected step leaves the model
//!    untouched. This example updates each parameter as it visits it; a
//!    failure partway through leaves the earlier parameters already stepped.
//!    For one parameter that distinction is invisible; for a real model with
//!    many, reproduce the pre-pass yourself if you need it (compute every
//!    update into a side buffer first, then commit all of them).
//! 2. **Path-predicate parameter groups.** `AdamW::new(..).group(|path| .., |g|
//!    ..)` lets one optimizer instance carry different hyperparameters for
//!    different parameters, matched by their dotted `state_dict` path. A
//!    hand-written optimizer has no such path space unless it builds one; this
//!    example applies one flat set of hyperparameters to every field.

use rstorch::nn::{Mode, Param};
use rstorch::persist::{Envelope, HostTensor, Limits};
use rstorch::{DType, Device, Error, Grads, Result, Rng, Tensor};

/// `y = w·x + b`, with `Param`s named as plain struct fields so the recipe
/// can update them directly — no generic walk needed.
struct Linear1d {
    weight: Param,
    bias: Param,
}

impl Linear1d {
    fn new(device: &Device, rng: &mut Rng) -> Result<Linear1d> {
        Ok(Linear1d {
            weight: Param::new(Tensor::from_vec(
                vec![rng.uniform(-1.0, 1.0) as f32],
                [1],
                device,
            )?),
            bias: Param::new(Tensor::zeros([1], DType::F32, device)?),
        })
    }

    fn forward(&self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        x.mul(&self.weight.get(mode))?.add(&self.bias.get(mode))
    }
}

/// RMSprop's per-parameter state: a running average of the squared gradient.
/// Keyed by name so it survives a save/reload round trip through the same
/// dotted-path convention `state_dict` uses.
struct RmspropState {
    v_weight: Tensor,
    v_bias: Tensor,
}

impl RmspropState {
    fn zeros(device: &Device) -> Result<RmspropState> {
        Ok(RmspropState {
            v_weight: Tensor::zeros([1], DType::F32, device)?,
            v_bias: Tensor::zeros([1], DType::F32, device)?,
        })
    }
}

/// One RMSprop update: `v ← α·v + (1−α)·g²`, `param ← param − lr·g/(√v + ε)`.
///
/// Skips a frozen parameter entirely — reading [`Param::is_frozen`] the same
/// way the built-in optimizers do, so freezing a hand-rolled model's
/// parameter still works with a hand-rolled optimizer.
///
/// # Errors
///
/// [`rstorch::Error::NotTraced`] if `grads` carries no cotangent for `param`
/// (the parameter was never used under a recording [`Mode`]), or whatever
/// the tensor arithmetic reports.
fn rmsprop_update(
    param: &mut Param,
    v: &mut Tensor,
    grads: &Grads,
    lr: f64,
    alpha: f64,
    eps: f64,
) -> Result<()> {
    if param.is_frozen() {
        return Ok(());
    }
    let g = grads.wrt(param)?;
    let g_squared = g.mul(&g)?;
    *v = v
        .mul_scalar(alpha)?
        .add(&g_squared.mul_scalar(1.0 - alpha)?)?;
    let denom = v.sqrt()?.add_scalar(eps)?;
    let step = g.div(&denom)?.mul_scalar(lr)?;
    let current = param.get(Mode::EVAL);
    param.set(current.sub(&step)?)
}

fn rmsprop_step(
    model: &mut Linear1d,
    state: &mut RmspropState,
    grads: &Grads,
    lr: f64,
    alpha: f64,
    eps: f64,
) -> Result<()> {
    rmsprop_update(
        &mut model.weight,
        &mut state.v_weight,
        grads,
        lr,
        alpha,
        eps,
    )?;
    rmsprop_update(&mut model.bias, &mut state.v_bias, grads, lr, alpha, eps)
}

/// A `[n]` `F32` tensor as raw little-endian bytes — the manual half of the
/// `Tensor` → [`HostTensor`] bridge the crate's own checkpoint path performs
/// internally (`src/checkpoint.rs`) but does not export, so a public
/// implementation goes through [`Tensor::to_vec`] and
/// [`HostTensor::from_bytes`] instead.
fn tensor_to_host(t: &Tensor) -> Result<HostTensor> {
    let values = t.to_vec::<f32>()?;
    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    HostTensor::from_bytes(DType::F32, t.dims().to_vec(), bytes)
}

fn host_to_tensor(host: &HostTensor, device: &Device) -> Result<Tensor> {
    if host.dtype() != DType::F32 {
        return Err(Error::persistence(format!(
            "expected an F32 tensor, found {}; this reader reinterprets raw bytes as f32 and \
             cannot convert",
            host.dtype()
        )));
    }
    let values: Vec<f32> = host
        .bytes()
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunks_exact(4)")))
        .collect();
    Tensor::from_vec(values, host.dims().to_vec(), device)
}

/// Look one tensor up in a loaded envelope. A checkpoint read off disk is
/// untrusted input — a missing key is a malformed file, not a bug in this
/// program, so it becomes an [`Error`] rather than a panic.
fn load_tensor(envelope: &Envelope, path: &str, device: &Device) -> Result<Tensor> {
    let host = envelope
        .tensor(path)
        .ok_or_else(|| Error::persistence(format!("checkpoint is missing tensor `{path}`")))?;
    host_to_tensor(host, device)
}

/// Save the optimizer's own state into `envelope`, under an `rmsprop.`
/// prefix so it cannot collide with the model's own tensor keys.
fn save_rmsprop_state(state: &RmspropState, envelope: &mut Envelope) -> Result<()> {
    envelope.insert_tensor("rmsprop.weight.v", tensor_to_host(&state.v_weight)?);
    envelope.insert_tensor("rmsprop.bias.v", tensor_to_host(&state.v_bias)?);
    envelope.set_section("rmsprop", "format=1".to_string())?;
    Ok(())
}

/// Save the model's parameters into the same `envelope`, under their bare
/// field names — the flat keys `state_dict` would produce for a one-layer
/// model, and disjoint from the optimizer's `rmsprop.` prefix.
fn save_model(model: &Linear1d, envelope: &mut Envelope) -> Result<()> {
    envelope.insert_tensor("weight", tensor_to_host(model.weight.value())?);
    envelope.insert_tensor("bias", tensor_to_host(model.bias.value())?);
    Ok(())
}

fn load_rmsprop_state(envelope: &Envelope, device: &Device) -> Result<RmspropState> {
    Ok(RmspropState {
        v_weight: load_tensor(envelope, "rmsprop.weight.v", device)?,
        v_bias: load_tensor(envelope, "rmsprop.bias.v", device)?,
    })
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let mut rng = Rng::seed(0);

    // Synthetic data for y = 3x - 1, so the fitted weight/bias are checkable
    // by eye in the printed trajectory.
    let n = 64;
    let xs: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) * 4.0 - 2.0).collect();
    let ys: Vec<f32> = xs.iter().map(|&x| 3.0 * x - 1.0).collect();
    let x = Tensor::from_vec(xs, [n], &device)?;
    let y = Tensor::from_vec(ys, [n], &device)?;

    let mut model = Linear1d::new(&device, &mut rng)?;
    let mut state = RmspropState::zeros(&device)?;
    let (lr, alpha, eps) = (0.1, 0.9, 1e-8);

    println!("training y = 3x - 1 with a hand-rolled RMSprop:");
    let mut first_loss = None;
    let mut last_loss = 0.0;
    for step in 0..200 {
        let traced_x = x.traced()?;
        let prediction = model.forward(&traced_x, Mode::TRAIN)?;
        let residual = prediction.sub(&y)?;
        let loss = residual.mul(&residual)?.mean_all()?;
        let loss_value = loss.item()?;
        first_loss.get_or_insert(loss_value);
        last_loss = loss_value;
        let grads = loss.backward()?;
        rmsprop_step(&mut model, &mut state, &grads, lr, alpha, eps)?;
        if step % 40 == 0 {
            println!(
                "  step {step:>3}: loss {loss_value:.6}  w = {:.4}  b = {:.4}",
                model.weight.value().to_vec::<f32>()?[0],
                model.bias.value().to_vec::<f32>()?[0]
            );
        }
    }
    println!(
        "  final: loss {last_loss:.6}  w = {:.4}  b = {:.4}",
        model.weight.value().to_vec::<f32>()?[0],
        model.bias.value().to_vec::<f32>()?[0]
    );
    assert!(
        last_loss < first_loss.unwrap() * 0.01,
        "RMSprop did not converge: {:.6} -> {last_loss:.6}",
        first_loss.unwrap()
    );

    // Round-trip both halves of the checkpoint through one `Envelope`: the
    // model's parameters under their bare names and the optimizer's own
    // second moments under `rmsprop.`, so the file demonstrates the two
    // sharing a container without colliding and both coming back exact.
    let mut envelope = Envelope::new();
    save_model(&model, &mut envelope)?;
    save_rmsprop_state(&state, &mut envelope)?;
    let path = std::env::temp_dir().join(format!(
        "rstorch-custom-optimizer-{}.rstorch",
        std::process::id()
    ));
    envelope.save(&path, &Limits::default())?;

    let reloaded_envelope = Envelope::load(&path, &Limits::default())?;
    let restored_weight = load_tensor(&reloaded_envelope, "weight", &device)?;
    let restored_bias = load_tensor(&reloaded_envelope, "bias", &device)?;
    let restored = load_rmsprop_state(&reloaded_envelope, &device)?;
    assert_eq!(
        restored_weight.to_vec::<f32>()?,
        model.weight.value().to_vec::<f32>()?,
        "restored model weight must match exactly"
    );
    assert_eq!(
        restored_bias.to_vec::<f32>()?,
        model.bias.value().to_vec::<f32>()?,
        "restored model bias must match exactly"
    );
    assert_eq!(
        restored.v_weight.to_vec::<f32>()?,
        state.v_weight.to_vec::<f32>()?,
        "restored weight second moment must match exactly"
    );
    assert_eq!(
        restored.v_bias.to_vec::<f32>()?,
        state.v_bias.to_vec::<f32>()?,
        "restored bias second moment must match exactly"
    );
    std::fs::remove_file(&path).ok();
    println!("model and optimizer state round-tripped through {path:?} exactly.");

    Ok(())
}
