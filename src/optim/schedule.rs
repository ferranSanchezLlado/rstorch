//! Learning-rate schedules as **plain `f64` functions**.
//!
//! A schedule is not a trait, an object, or a piece of optimizer state: it is
//! arithmetic on the step count, and the optimizer learns about it through
//! `set_lr`. That keeps the schedule out of the checkpoint (a resumed run
//! recomputes it from the step count, so changing the schedule between runs is
//! an ordinary code edit) and makes composition trivial — the functions take a
//! base learning rate, so they nest:
//!
//! ```
//! use rstorch::optim::{Sgd, schedule};
//!
//! let mut opt = Sgd::new(0.1);
//! // 100 warmup steps in front of a 10k-step cosine decay.
//! for _ in 0..3 {
//!     let step = opt.steps();
//!     opt.set_lr(schedule::warmup(
//!         schedule::cosine(0.1, 0.0, 10_000, step),
//!         100,
//!         step,
//!     ));
//!     # assert!(opt.lr() > 0.0);
//!     // ... forward, backward, opt.step(&mut model, grads)? ...
//!     # break;
//! }
//! ```
//!
//! Every function is total: a zero-length ramp or horizon degenerates
//! sensibly rather than dividing by zero. `step` is the number of *completed*
//! steps (`Sgd::steps`/`Adam::steps`), so the first iteration of a loop sees
//! `step == 0`.

/// Staircase decay: multiply by `gamma` every `step_size` steps.
///
/// `step_size == 0` disables the decay (returns `base_lr`).
pub fn step_decay(base_lr: f64, gamma: f64, step_size: u64, step: u64) -> f64 {
    if step_size == 0 {
        return base_lr;
    }
    // Saturate rather than cast: `as i32` wraps at 2³¹ exponents, and a negative
    // exponent turns `gamma < 1` decay into unbounded *growth*. Clamping is
    // exact here — `gamma^i32::MAX` has already underflowed to 0.
    let exponent = (step / step_size).min(i32::MAX as u64) as i32;
    base_lr * gamma.powi(exponent)
}

/// Cosine decay from `base_lr` down to `min_lr` over `total_steps`, staying at
/// `min_lr` afterwards.
///
/// `total_steps == 0` means "the horizon is already reached" (returns
/// `min_lr`).
pub fn cosine(base_lr: f64, min_lr: f64, total_steps: u64, step: u64) -> f64 {
    if total_steps == 0 {
        return min_lr;
    }
    let progress = step.min(total_steps) as f64 / total_steps as f64;
    let factor = 0.5 * (1.0 + (std::f64::consts::PI * progress).cos());
    min_lr + (base_lr - min_lr) * factor
}

/// Linear warmup into `target_lr` over `warmup_steps`, then `target_lr`.
///
/// The first step gets `target_lr / warmup_steps` (never zero — a step with a
/// zero learning rate is a wasted batch), and step `warmup_steps - 1` gets the
/// full rate. `warmup_steps == 0` disables the ramp.
pub fn warmup(target_lr: f64, warmup_steps: u64, step: u64) -> f64 {
    if warmup_steps == 0 || step >= warmup_steps {
        return target_lr;
    }
    target_lr * (step + 1) as f64 / warmup_steps as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }

    #[test]
    fn step_decay_drops_on_the_boundary() {
        close(step_decay(1.0, 0.1, 10, 0), 1.0);
        close(step_decay(1.0, 0.1, 10, 9), 1.0);
        close(step_decay(1.0, 0.1, 10, 10), 0.1);
        close(step_decay(1.0, 0.1, 10, 25), 0.01);
        // Degenerate step size is a no-op, not a division by zero.
        close(step_decay(1.0, 0.1, 0, 99), 1.0);
    }

    #[test]
    fn cosine_spans_base_to_min_and_holds() {
        close(cosine(1.0, 0.0, 100, 0), 1.0);
        close(cosine(1.0, 0.0, 100, 50), 0.5);
        close(cosine(1.0, 0.0, 100, 100), 0.0);
        // Past the horizon it holds at the floor.
        close(cosine(1.0, 0.2, 100, 1_000), 0.2);
        close(cosine(1.0, 0.2, 0, 0), 0.2);
    }

    #[test]
    fn warmup_ramps_then_passes_through() {
        close(warmup(1.0, 4, 0), 0.25);
        close(warmup(1.0, 4, 3), 1.0);
        close(warmup(1.0, 4, 9), 1.0);
        close(warmup(1.0, 0, 0), 1.0);
    }

    #[test]
    fn schedules_compose_because_they_are_functions() {
        // Warmup in front of cosine: during the ramp the cosine value is the
        // target being ramped into.
        let step = 1;
        let inner = cosine(1.0, 0.0, 100, step);
        close(warmup(inner, 4, step), inner * 0.5);
        // After the ramp, composition is the inner schedule unchanged.
        close(warmup(cosine(1.0, 0.0, 100, 50), 4, 50), 0.5);
    }
}
