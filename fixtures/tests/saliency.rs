//! **The m2 acceptance fixture**: gradients as a linear value, and
//! gradient-with-respect-to-*input* done the way the design says to do it.
//!
//! Like every fixture this file is a downstream consumer — it imports nothing
//! but `rstorch::prelude::*`, so each step below has to be expressible in the
//! public first-hour vocabulary. What it pins:
//!
//! - **Saliency**: `let xt = x.traced()?; let y = f(&xt)?; let g =
//!   y.backward()?; g.wrt_input(&xt)?` — the traced binding is the one used in
//!   the computation *and* in the lookup. Getting that wrong is loud, not
//!   silently zero, and the misuse cases below assert exactly that.
//! - **Parameter gradients** alongside input gradients, from the same
//!   `backward()`, keyed by `Param` identity rather than by a `.grad` slot.
//! - **Weight tying**: one `Param` read twice accumulates both contributions.
//! - **The linear surface**: `merge` / `scale` / `clip_norm` as explicit
//!   pipelines. There is no `zero_grad()` to forget, because gradients are
//!   return values.
//! - **No ambient mode**: `Mode::EVAL` records nothing, so an eval forward
//!   cannot be differentiated at all.

use rstorch::prelude::*;

/// The device under test; `best_available()` is the spelling a user writes.
fn dev() -> Device {
    Device::best_available()
}

/// Assert two float slices agree to a small relative tolerance.
#[track_caller]
fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "length: {got:?} vs {want:?}");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (f64::from(g) - f64::from(w)).abs() <= 1e-5 * f64::from(w).abs().max(1.0),
            "element {i}: got {g}, want {w} (in {got:?})"
        );
    }
}

/// A tiny fixed classifier: `logits = relu(x·W₁ᵀ + b₁)·W₂ᵀ + b₂`, four
/// features in, three classes out. Built from `Param`s so the same forward
/// pass yields both input and parameter gradients.
struct Scorer {
    w1: Param,
    b1: Param,
    w2: Param,
    b2: Param,
}

impl Scorer {
    fn new(device: &Device) -> Result<Scorer> {
        // Deterministic weights: no RNG dependency in an acceptance fixture.
        let w1 = Tensor::from_vec(
            vec![
                0.5f32, -0.25, 0.75, 0.1, //
                -0.6, 0.4, 0.2, -0.3, //
                0.15, 0.35, -0.45, 0.55,
            ],
            [3, 4],
            device,
        )?;
        let b1 = Tensor::from_vec(vec![0.05f32, -0.1, 0.2], [3], device)?;
        let w2 = Tensor::from_vec(
            vec![
                1.0f32, -0.5, 0.25, //
                0.3, 0.8, -0.2, //
                -0.7, 0.1, 0.9,
            ],
            [3, 3],
            device,
        )?;
        let b2 = Tensor::from_vec(vec![0.0f32, 0.1, -0.1], [3], device)?;
        Ok(Scorer {
            w1: Param::new(w1),
            b1: Param::new(b1),
            w2: Param::new(w2),
            b2: Param::new(b2),
        })
    }

    /// `x` is `[batch, 4]`; the result is `[batch, 3]`.
    fn logits(&self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        let h = x
            .matmul(&self.w1.get(mode).transpose(-2, -1)?)?
            .add(&self.b1.get(mode))?
            .relu()?;
        h.matmul(&self.w2.get(mode).transpose(-2, -1)?)?
            .add(&self.b2.get(mode))
    }
}

/// The score of `class` for a single-row `logits`, as a rank-0 tensor.
fn class_score(logits: &Tensor, class: usize) -> Result<Tensor> {
    logits.narrow(1, class, 1)?.reshape(())
}

// ---------------------------------------------------------------------------
// The headline: a saliency map
// ---------------------------------------------------------------------------

#[test]
fn saliency_map_of_a_class_score() -> Result<()> {
    let dev = dev();
    let model = Scorer::new(&dev)?;
    let x = Tensor::from_vec(vec![0.9f32, -0.4, 0.2, 1.1], [1, 4], &dev)?;

    // The three-line recipe.
    let xt = x.traced()?;
    let score = class_score(&model.logits(&xt, Mode::EVAL.recorded())?, 2)?;
    let grads = score.backward()?;
    let saliency = grads.wrt_input(&xt)?;

    assert_eq!(saliency.dims(), &[1, 4]);
    assert_eq!(saliency.dtype(), DType::F32);
    // A saliency map is data, not graph: it can be fed straight back into
    // ordinary ops (here, "which feature mattered most?").
    assert_eq!(saliency.abs()?.argmax(1)?.to_vec::<i64>()?.len(), 1);

    // Checked against a central difference through the public API — no
    // internal harness, just the forward pass twice per feature.
    let eps = 1e-2f32;
    let base = x.to_vec::<f32>()?;
    let mut numeric = Vec::new();
    for i in 0..base.len() {
        let bump = |delta: f32| -> Result<f64> {
            let mut v = base.clone();
            v[i] += delta;
            let xp = Tensor::from_vec(v, [1, 4], &dev)?;
            class_score(&model.logits(&xp, Mode::EVAL)?, 2)?.item()
        };
        numeric.push(((bump(eps)? - bump(-eps)?) / (2.0 * f64::from(eps))) as f32);
    }
    assert_close(&saliency.to_vec::<f32>()?, &numeric);
    Ok(())
}

#[test]
fn one_backward_yields_both_input_and_parameter_gradients() -> Result<()> {
    let dev = dev();
    let model = Scorer::new(&dev)?;
    let x = Tensor::from_vec(vec![0.3f32, 0.7, -0.2, 0.5], [1, 4], &dev)?;

    let xt = x.traced()?;
    let loss = class_score(&model.logits(&xt, Mode::TRAIN)?, 0)?;
    let grads = loss.backward()?;

    // Four parameters plus the traced input.
    assert_eq!(grads.len(), 5);
    assert_eq!(grads.wrt(&model.w1)?.dims(), &[3, 4]);
    assert_eq!(grads.wrt(&model.b1)?.dims(), &[3]);
    assert_eq!(grads.wrt(&model.w2)?.dims(), &[3, 3]);
    assert_eq!(grads.wrt(&model.b2)?.dims(), &[3]);
    assert_eq!(grads.wrt_input(&xt)?.dims(), &[1, 4]);

    // `b2` feeds the output directly, so its gradient is the one-hot cotangent
    // of the class we scored.
    assert_close(&grads.wrt(&model.b2)?.to_vec::<f32>()?, &[1.0, 0.0, 0.0]);
    Ok(())
}

// ---------------------------------------------------------------------------
// A linear scorer has an exactly known saliency
// ---------------------------------------------------------------------------

#[test]
fn a_linear_scores_saliency_is_its_weight_row() -> Result<()> {
    let dev = dev();
    let w = Tensor::from_vec(
        vec![
            1.0f32, 2.0, 3.0, //
            -4.0, 5.0, -6.0,
        ],
        [2, 3],
        &dev,
    )?;
    let x = Tensor::from_vec(vec![0.1f32, 0.2, 0.3], [1, 3], &dev)?;

    let xt = x.traced()?;
    let score = class_score(&xt.matmul(&w.transpose(-2, -1)?)?, 1)?;
    let saliency = score.backward()?.wrt_input(&xt)?;

    assert_eq!(saliency.dims(), &[1, 3]);
    assert_close(&saliency.to_vec::<f32>()?, &[-4.0, 5.0, -6.0]);
    Ok(())
}

// ---------------------------------------------------------------------------
// The misuse cases are loud
// ---------------------------------------------------------------------------

#[test]
fn looking_up_the_wrong_binding_is_an_error_not_a_zero() -> Result<()> {
    let dev = dev();
    let x = Tensor::from_vec(vec![1.0f32, 2.0], [2], &dev)?;
    let xt = x.traced()?;
    let grads = xt.mul(&xt)?.sum_all()?.backward()?;

    // The original tensor carries no graph: PyTorch would hand back `None`.
    assert!(grads.wrt_input(&x).is_err());
    // An interior value is not a leaf.
    assert!(grads.wrt_input(&xt.relu()?).is_err());
    // A traced leaf that took no part in the computation.
    let unrelated = Tensor::ones([2], DType::F32, &dev)?.traced()?;
    assert!(grads.wrt_input(&unrelated).is_err());
    // …while the right binding works.
    assert_close(&grads.wrt_input(&xt)?.to_vec::<f32>()?, &[2.0, 4.0]);

    // Double-tracing is a bug, and says so.
    assert!(xt.traced().is_err());
    Ok(())
}

#[test]
fn tracing_is_data_flow_not_an_ambient_mode() -> Result<()> {
    let dev = dev();
    let model = Scorer::new(&dev)?;
    let x = Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], [1, 4], &dev)?;

    // No traced input, no recording mode: nothing is retained, and asking for
    // gradients is a structured error rather than an empty `Grads`.
    assert!(model.logits(&x, Mode::EVAL)?.backward().is_err());

    // Recording follows the *data*, not a mode flag: a traced input records
    // even under `Mode::EVAL`, but the parameters — read through
    // `Param::get(Mode::EVAL)` — contribute nothing, which is exactly what a
    // saliency map wants and what `Mode::EVAL.recorded()` opts out of.
    let xt = x.traced()?;
    let eval_grads = class_score(&model.logits(&xt, Mode::EVAL)?, 0)?.backward()?;
    assert_eq!(eval_grads.len(), 1);
    assert!(eval_grads.wrt_input(&xt).is_ok());
    assert!(eval_grads.wrt(&model.w1).is_err());

    let recorded = class_score(&model.logits(&xt, Mode::EVAL.recorded())?, 0)?.backward()?;
    assert_eq!(recorded.len(), 5);

    // A frozen parameter opts out per-parameter, explicitly.
    let mut model = model;
    model.w1.freeze();
    let frozen = class_score(&model.logits(&xt, Mode::TRAIN)?, 0)?.backward()?;
    assert_eq!(frozen.len(), 4);
    assert!(frozen.wrt(&model.w1).is_err());

    // `detach` is the explicit scissors mid-graph.
    assert!(
        class_score(&model.logits(&xt, Mode::TRAIN)?, 0)?
            .detach()
            .backward()
            .is_err()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Weight tying and the linear `Grads` surface
// ---------------------------------------------------------------------------

#[test]
fn a_tied_parameter_accumulates_both_uses() -> Result<()> {
    let dev = dev();
    // The classic tie: the same embedding matrix used to embed and to score.
    let e = Param::new(Tensor::from_vec(vec![2.0f32, -3.0], [2], &dev)?);
    let loss = e.get(Mode::TRAIN).mul(&e.get(Mode::TRAIN))?.sum_all()?;
    let grads = loss.backward()?;

    // One key, not two: d(Σ e²)/de = 2e.
    assert_eq!(grads.len(), 1);
    assert_close(&grads.wrt(&e)?.to_vec::<f32>()?, &[4.0, -6.0]);
    Ok(())
}

#[test]
fn micro_batch_accumulation_is_an_explicit_pipeline() -> Result<()> {
    let dev = dev();
    let model = Scorer::new(&dev)?;
    let rows = [
        vec![0.9f32, -0.4, 0.2, 1.1],
        vec![-0.3f32, 0.8, 0.6, -0.7],
        vec![0.4f32, 0.4, -0.9, 0.2],
    ];

    // No `zero_grad()` exists to forget: each step's gradients are a fresh
    // value, and accumulation is a fold the compiler checks is linear.
    let mut accumulated: Option<Grads> = None;
    for row in &rows {
        let x = Tensor::from_vec(row.clone(), [1, 4], &dev)?;
        let step = class_score(&model.logits(&x, Mode::TRAIN)?, 1)?.backward()?;
        accumulated = Some(match accumulated {
            Some(acc) => acc.merge(step)?,
            None => step,
        });
    }
    let averaged = accumulated
        .expect("three micro-batches")
        .scale(1.0 / rows.len() as f64)?;
    assert_eq!(averaged.len(), 4);

    // Clipping is the same shape of pipeline; a budget above the current norm
    // is a no-op, one below it rescales every entry together.
    let before = averaged.wrt(&model.b2)?.to_vec::<f32>()?;
    let clipped = averaged.clip_norm(1e6)?;
    assert_close(&clipped.wrt(&model.b2)?.to_vec::<f32>()?, &before);

    let clipped = clipped.clip_norm(1e-3)?;
    let after = clipped.wrt(&model.b2)?.to_vec::<f32>()?;
    assert!(after.iter().zip(&before).all(|(a, b)| a.abs() <= b.abs()));
    Ok(())
}

#[test]
fn a_batched_saliency_map_keeps_the_batch_axis() -> Result<()> {
    let dev = dev();
    let model = Scorer::new(&dev)?;
    let x = Tensor::from_vec(
        vec![
            0.9f32, -0.4, 0.2, 1.1, //
            -0.3, 0.8, 0.6, -0.7,
        ],
        [2, 4],
        &dev,
    )?;

    let xt = x.traced()?;
    // Summing the per-row top scores gives every row an independent seed of 1,
    // so one backward produces the whole batch's saliency at once.
    let objective = model
        .logits(&xt, Mode::EVAL.recorded())?
        .max(1)?
        .sum_all()?;
    let saliency = objective.backward()?.wrt_input(&xt)?;
    assert_eq!(saliency.dims(), &[2, 4]);

    // Row 0's saliency must equal the saliency of row 0 taken on its own.
    let row0 = Tensor::from_vec(vec![0.9f32, -0.4, 0.2, 1.1], [1, 4], &dev)?;
    let row0t = row0.traced()?;
    let alone = model
        .logits(&row0t, Mode::EVAL.recorded())?
        .max(1)?
        .sum_all()?
        .backward()?
        .wrt_input(&row0t)?;
    assert_close(
        &saliency.narrow(0, 0, 1)?.contiguous()?.to_vec::<f32>()?,
        &alone.to_vec::<f32>()?,
    );
    Ok(())
}
