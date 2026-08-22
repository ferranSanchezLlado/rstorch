//! Weight initializers, and a tree-wide way to apply one.
//!
//! `Linear::new` (and `Proj::new` inside [`MultiHeadAttention`](crate::nn::MultiHeadAttention))
//! hard-code the scheme they draw. The `with_init` constructors
//! ([`Linear::with_init`](crate::nn::Linear::with_init),
//! [`Conv2d::with_init`](crate::nn::Conv2d::with_init)) take a closure instead;
//! this module is what you pass them: four named random initializers over
//! `&mut Rng`, the two constant fills, and [`apply`] to re-initialize every
//! [`Param`](crate::nn::Param) in an existing tree.
//!
//! # Fan inference
//!
//! Every initializer that needs a fan-in/fan-out infers it from the target
//! shape rather than taking it as a parameter, so a caller building a
//! `Conv2d` weight (`[out_channels, in_channels, kh, kw]`) never has to
//! compute it by hand:
//!
//! - **Rank 2** `[out, in]` (the `Linear` weight convention):
//!   `fan_in = in`, `fan_out = out`.
//! - **Rank 4** `[out_c, in_c, kh, kw]` (the `Conv2d` weight convention):
//!   the receptive field is `kh * kw`, so `fan_in = in_c * kh * kw` and
//!   `fan_out = out_c * kh * kw`.
//!
//! Any other rank is a loud [`Error::InvalidArg`] — there is no universally
//! agreed fan convention for, say, a rank-3 or rank-5 weight, so guessing one
//! would be worse than refusing.
//!
//! # Kaiming is `fan_in`, and stays that way
//!
//! [`kaiming_uniform`]/[`kaiming_normal`] divide by `fan_in` and take no
//! `mode` argument. That is a decision, not an omission: the two names are
//! frozen at 1.0.0, so a `mode` parameter cannot be added later without
//! breaking every call site, and a fan-out variant would therefore arrive
//! under its own name rather than as an argument. `fan_in` is the mode worth
//! freezing — it holds forward-pass variance steady, it is what `Linear::new`
//! and `Conv2d::new` already draw, and `benches/support/resnet.rs` records the
//! same choice for its `ResNet` on the grounds that it trains at that depth
//! without a warm-up schedule. A caller who wants `PyTorch`'s
//! `mode="fan_out"` today gets it by transposing the first two dimensions of
//! the shape passed in (`[out, in]` → `[in, out]`, `[out_c, in_c, kh, kw]` →
//! `[in_c, out_c, kh, kw]`), since the fans swap with them.
//!
//! # Reproducing the crate's own layers
//!
//! `kaiming_uniform(shape, 2.0f64.sqrt(), ..)` computes the same bound
//! `Linear::new` derives by hand (`gain = √2` is the standard ReLU gain, and
//! `Linear`'s Kaiming-uniform bound is exactly `gain·√(3/fan_in)` — verified
//! algebraically: `√(6/fan_in) = √2·√(3/fan_in)`). `xavier_uniform(shape,
//! 1.0, ..)` is `Proj::new`'s scheme exactly, `gain = 1`.
//!
//! Verified **not** bit-identical, and here is why: `Linear::new` and
//! `Proj::new` each draw in their own way (a hand-rolled host loop over
//! `Rng::uniform`, or one `Tensor::rand` call folded through an affine
//! transform), and this module draws through [`Tensor::rand`]/[`Tensor::randn`]
//! plus a chain of `Tensor` ops. Every draw consumes `Rng` in the same
//! row-major order and produces the same *distribution*, but a value narrowed
//! to `dtype` before an affine transform (this module's [`Tensor::rand`] path)
//! is not obligated to round identically to the same arithmetic performed in
//! `f64` and narrowed once at the end (`Linear::new`'s hand-rolled path).
//! `xavier_uniform`'s formula is written to match `Proj::new`'s exact
//! operation order (`mul_scalar` then an additive shift), so that one *is*
//! bit-identical; `kaiming_uniform`/`kaiming_normal` are not verified to be.

use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::nn::Module;
use crate::nn::visit::{LeafMut, visit_all_mut};
use crate::rng::Rng;
use crate::shape::Shape;
use crate::tensor::Tensor;

const OP_KAIMING_UNIFORM: &str = "nn::init::kaiming_uniform";
const OP_KAIMING_NORMAL: &str = "nn::init::kaiming_normal";
const OP_XAVIER_UNIFORM: &str = "nn::init::xavier_uniform";
const OP_XAVIER_NORMAL: &str = "nn::init::xavier_normal";

/// Infer `(fan_in, fan_out)` from `shape`; see the [module docs](self) for
/// the two supported ranks and the reasoning.
fn fan_in_out(op: &'static str, shape: &Shape) -> Result<(usize, usize)> {
    let dims = shape.dims();
    match dims.len() {
        2 => Ok((dims[1], dims[0])),
        4 => {
            let receptive_field = dims[2] * dims[3];
            Ok((dims[1] * receptive_field, dims[0] * receptive_field))
        }
        other => Err(Error::invalid_arg(
            op,
            format!(
                "cannot infer fan-in/fan-out for rank {other} (shape {shape}); only rank 2 \
                 [out, in] and rank 4 [out_channels, in_channels, kh, kw] are supported"
            ),
        )),
    }
}

/// A zero fan has no finite Kaiming/Xavier bound, but each initializer checks
/// only what its own formula divides by: Kaiming reads `fan_in` alone, so
/// [`require_nonzero_fan_in`] is what it calls, while Xavier divides by
/// `fan_in + fan_out` and so rejects only a zero *sum*. Rejecting a degenerate
/// direction the formula never reads would be a spurious error — Kaiming on
/// `[0, 4]` and Xavier on `[0, 4]` both have a finite bound, and both are
/// accepted.
fn require_nonzero_fan_sum(
    op: &'static str,
    shape: &Shape,
    fan_in: usize,
    fan_out: usize,
) -> Result<()> {
    if fan_in + fan_out == 0 {
        return Err(Error::invalid_arg(
            op,
            format!("shape {shape} has a zero fan sum (fan_in={fan_in}, fan_out={fan_out})"),
        ));
    }
    Ok(())
}

/// See [`require_nonzero_fan_sum`] for why this is separate.
fn require_nonzero_fan_in(op: &'static str, shape: &Shape, fan_in: usize) -> Result<()> {
    if fan_in == 0 {
        return Err(Error::invalid_arg(
            op,
            format!("shape {shape} has a zero fan-in"),
        ));
    }
    Ok(())
}

/// `Tensor::rand` folded from `[0, 1)` into `[low, high)` — the same affine
/// shape `Proj::new` (`src/nn/attention.rs`) already uses for its
/// Xavier-uniform draw.
fn affine_uniform(
    shape: Shape,
    low: f64,
    high: f64,
    dtype: DType,
    device: &Device,
    rng: &mut Rng,
) -> Result<Tensor> {
    Tensor::rand(shape, dtype, device, rng)?
        .mul_scalar(high - low)?
        .add_scalar(low)
}

/// Kaiming/He-uniform: `U(-bound, bound)`, `bound = gain·√(3 / fan_in)`.
///
/// `gain = √2` (the standard ReLU gain) reproduces the bound
/// [`Linear::new`](crate::nn::Linear::new) derives by hand; see the
/// [module docs](self) for why the two are not bit-identical.
///
/// # Errors
/// [`Error::InvalidArg`] if `shape`'s rank is not 2 or 4, if the inferred
/// fan-in is zero, or if `dtype` is not a float ([`Tensor::rand`] and
/// [`Tensor::randn`] draw floats only).
pub fn kaiming_uniform(
    shape: impl Into<Shape>,
    gain: f64,
    dtype: DType,
    device: &Device,
    rng: &mut Rng,
) -> Result<Tensor> {
    let shape = shape.into();
    let (fan_in, _) = fan_in_out(OP_KAIMING_UNIFORM, &shape)?;
    require_nonzero_fan_in(OP_KAIMING_UNIFORM, &shape, fan_in)?;
    let bound = gain * (3.0 / fan_in as f64).sqrt();
    affine_uniform(shape, -bound, bound, dtype, device, rng)
}

/// Kaiming/He-normal: `N(0, std²)`, `std = gain / √fan_in`.
///
/// # Errors
/// As [`kaiming_uniform`].
pub fn kaiming_normal(
    shape: impl Into<Shape>,
    gain: f64,
    dtype: DType,
    device: &Device,
    rng: &mut Rng,
) -> Result<Tensor> {
    let shape = shape.into();
    let (fan_in, _) = fan_in_out(OP_KAIMING_NORMAL, &shape)?;
    require_nonzero_fan_in(OP_KAIMING_NORMAL, &shape, fan_in)?;
    let std = gain / (fan_in as f64).sqrt();
    Tensor::randn(shape, dtype, device, rng)?.mul_scalar(std)
}

/// Glorot/Xavier-uniform: `U(-bound, bound)`,
/// `bound = gain·√(6 / (fan_in + fan_out))`.
///
/// `gain = 1.0` reproduces
/// [`MultiHeadAttention`](crate::nn::MultiHeadAttention)'s internal
/// `Proj::new` scheme **bit-for-bit** — verified: both draw one
/// [`Tensor::rand`] call and fold it through `mul_scalar` then an additive
/// shift, in the same order.
///
/// # Errors
/// As [`kaiming_uniform`], except that the formula divides by
/// `fan_in + fan_out`, so only a zero *sum* is rejected — a zero fan in one
/// direction alone leaves the bound finite.
pub fn xavier_uniform(
    shape: impl Into<Shape>,
    gain: f64,
    dtype: DType,
    device: &Device,
    rng: &mut Rng,
) -> Result<Tensor> {
    let shape = shape.into();
    let (fan_in, fan_out) = fan_in_out(OP_XAVIER_UNIFORM, &shape)?;
    require_nonzero_fan_sum(OP_XAVIER_UNIFORM, &shape, fan_in, fan_out)?;
    let bound = gain * (6.0 / (fan_in + fan_out) as f64).sqrt();
    affine_uniform(shape, -bound, bound, dtype, device, rng)
}

/// Glorot/Xavier-normal: `N(0, std²)`, `std = gain·√(2 / (fan_in + fan_out))`.
///
/// # Errors
/// As [`xavier_uniform`].
pub fn xavier_normal(
    shape: impl Into<Shape>,
    gain: f64,
    dtype: DType,
    device: &Device,
    rng: &mut Rng,
) -> Result<Tensor> {
    let shape = shape.into();
    let (fan_in, fan_out) = fan_in_out(OP_XAVIER_NORMAL, &shape)?;
    require_nonzero_fan_sum(OP_XAVIER_NORMAL, &shape, fan_in, fan_out)?;
    let std = gain * (2.0 / (fan_in + fan_out) as f64).sqrt();
    Tensor::randn(shape, dtype, device, rng)?.mul_scalar(std)
}

/// All-zeros, for parity with the four random names — `Tensor::zeros` under
/// this module's `(shape, dtype, device)` spelling. No randomness consumed.
///
/// # Errors
/// [`Error::InvalidArg`] if the element count overflows `usize`.
pub fn zeros(shape: impl Into<Shape>, dtype: DType, device: &Device) -> Result<Tensor> {
    Tensor::zeros(shape, dtype, device)
}

/// All-ones, for parity with the four random names — `Tensor::ones` under
/// this module's `(shape, dtype, device)` spelling. No randomness consumed.
///
/// # Errors
/// As [`zeros`].
pub fn ones(shape: impl Into<Shape>, dtype: DType, device: &Device) -> Result<Tensor> {
    Tensor::ones(shape, dtype, device)
}

/// Re-initialize every [`Param`](crate::nn::Param) in `model`'s tree in place, in the walk
/// order `Module::visit_mut` emits (deterministic, so `init` consuming an
/// `&mut Rng` internally draws in that same order every call).
///
/// `init` receives each parameter's dotted path (`"fc1.weight"`) and
/// **current** value (for its shape, dtype, and device) and returns its
/// replacement — `Tensor` has no in-place ops, so "re-init" means building a
/// fresh tensor and [`Param::set`](crate::nn::Param::set)ting it, which this
/// function does for you. Buffers (`BatchNorm` running statistics) are left
/// untouched: they are not trainable parameters and re-initializing them
/// would erase real state rather than pick a starting point for training.
///
/// The path is what makes "Kaiming the convs, zeros the norm weights, leave
/// `head.*` alone" expressible: dispatching on shape alone cannot tell two
/// same-shaped parameters apart, and the path already exists in the walk that
/// drives this function, so it costs nothing to pass through.
///
/// The first error is kept and every leaf after it is left untouched (the
/// walk itself runs to completion; `init` is simply not called again).
/// Because `Param::set` validates shape, dtype, and device before swapping, a
/// rejected replacement leaves that one parameter — and everything visited
/// before it — already changed: this is a convenience walker, not a
/// transaction; use it before training starts, not on a model already in use.
///
/// # Errors
/// Whatever `init` returns, or [`Error::InvalidArg`]/shape-mismatch errors
/// from [`Param::set`](crate::nn::Param::set) if `init`'s output does not
/// match the parameter it is replacing.
///
/// ```
/// # use rstorch::nn::{self, Linear, ModuleExt};
/// # use rstorch::{Device, Rng};
/// # fn main() -> rstorch::Result<()> {
/// let dev = Device::Cpu;
/// let mut rng = Rng::seed(0);
/// let mut fc = Linear::new(4, 3, &dev, &mut rng)?;
/// let before = fc.weight().value().clone();
/// nn::init::apply(&mut fc, |path, t| {
///     // Only the weight has a fan; leave the bias alone by name, not shape.
///     if path.ends_with(".weight") || path == "weight" {
///         nn::init::kaiming_normal(t.dims().to_vec(), 2f64.sqrt(), t.dtype(), &t.device(), &mut rng)
///     } else {
///         Ok(t.clone())
///     }
/// })?;
/// assert_ne!(before.to_vec::<f32>()?, fc.weight().value().to_vec::<f32>()?);
/// # Ok(())
/// # }
/// ```
pub fn apply<M: Module + ?Sized>(
    model: &mut M,
    mut init: impl FnMut(&str, &Tensor) -> Result<Tensor>,
) -> Result<()> {
    let mut failure: Option<Error> = None;
    visit_all_mut(model, &mut |path, leaf| {
        if failure.is_some() {
            return;
        }
        if let LeafMut::Param(p) = leaf {
            match init(path, p.value()).and_then(|fresh| p.set(fresh)) {
                Ok(()) => {}
                Err(e) => failure = Some(e),
            }
        }
    });
    match failure {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{BatchNorm2d, Linear, ModuleExt};

    const CPU: Device = Device::Cpu;

    fn mean_std(data: &[f32]) -> (f64, f64) {
        let n = data.len() as f64;
        let mean = data.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
        let var = data
            .iter()
            .map(|&x| (f64::from(x) - mean).powi(2))
            .sum::<f64>()
            / n;
        (mean, var.sqrt())
    }

    #[test]
    fn fan_in_out_infers_from_rank_2_and_rank_4() {
        assert_eq!(fan_in_out("t", &Shape::from(vec![8, 4])).unwrap(), (4, 8));
        // [out_c, in_c, kh, kw] = [16, 3, 5, 5]: receptive field 25.
        assert_eq!(
            fan_in_out("t", &Shape::from(vec![16, 3, 5, 5])).unwrap(),
            (75, 400)
        );
        assert!(fan_in_out("t", &Shape::from(vec![4])).is_err());
        assert!(fan_in_out("t", &Shape::from(vec![2, 3, 4])).is_err());
    }

    #[test]
    fn each_initializer_rejects_only_the_fans_its_formula_divides_by() {
        // [out, in] = [0, 4]: fan_out is zero, fan_in is not. Neither formula
        // divides by a zero here — Kaiming reads fan_in, Xavier reads the sum
        // — so both accept it.
        let shape = Shape::from(vec![0, 4]);
        let (fan_in, fan_out) = fan_in_out("t", &shape).unwrap();
        assert_eq!((fan_in, fan_out), (4, 0));
        assert!(require_nonzero_fan_in("t", &shape, fan_in).is_ok());
        assert!(require_nonzero_fan_sum("t", &shape, fan_in, fan_out).is_ok());
        // [out, in] = [4, 0]: Kaiming divides by the zero fan_in and rejects;
        // Xavier's sum is still 4, so it does not.
        let shape = Shape::from(vec![4, 0]);
        assert!(require_nonzero_fan_in("t", &shape, 0).is_err());
        assert!(require_nonzero_fan_sum("t", &shape, 0, 4).is_ok());
        // Only a zero sum has no finite Xavier bound.
        let shape = Shape::from(vec![0, 0]);
        assert!(require_nonzero_fan_sum("t", &shape, 0, 0).is_err());
    }

    #[test]
    fn kaiming_uniform_stays_within_its_analytic_bound() {
        let mut rng = Rng::seed(1);
        let fan_in = 256;
        let gain = 2f64.sqrt();
        let t = kaiming_uniform([64, fan_in], gain, DType::F32, &CPU, &mut rng).unwrap();
        let bound = gain * (3.0 / fan_in as f64).sqrt();
        let data = t.to_vec::<f32>().unwrap();
        assert!(
            data.iter()
                .all(|&x| (-bound..bound).contains(&f64::from(x))),
            "every sample must fall inside the analytic bound {bound}"
        );
        // Uniform(-b, b) has std = b/sqrt(3); loose tolerance for a finite sample.
        let (mean, std) = mean_std(&data);
        assert!(mean.abs() < 0.02, "mean {mean} should be near zero");
        let want_std = bound / 3f64.sqrt();
        assert!(
            (std - want_std).abs() / want_std < 0.1,
            "std {std} should be near the analytic {want_std}"
        );
    }

    #[test]
    fn kaiming_normal_matches_its_analytic_std() {
        let mut rng = Rng::seed(2);
        let fan_in = 512;
        let gain = 2f64.sqrt();
        let t = kaiming_normal([128, fan_in], gain, DType::F32, &CPU, &mut rng).unwrap();
        let want_std = gain / (fan_in as f64).sqrt();
        let (mean, std) = mean_std(&t.to_vec::<f32>().unwrap());
        assert!(mean.abs() < 0.01, "mean {mean} should be near zero");
        assert!(
            (std - want_std).abs() / want_std < 0.1,
            "std {std} should be near the analytic {want_std}"
        );
    }

    #[test]
    fn xavier_uniform_stays_within_its_analytic_bound() {
        let mut rng = Rng::seed(3);
        let (fan_in, fan_out) = (100, 50);
        let t = xavier_uniform([fan_out, fan_in], 1.0, DType::F32, &CPU, &mut rng).unwrap();
        let bound = (6.0 / (fan_in + fan_out) as f64).sqrt();
        let data = t.to_vec::<f32>().unwrap();
        assert!(
            data.iter()
                .all(|&x| (-bound..bound).contains(&f64::from(x)))
        );
    }

    #[test]
    fn xavier_normal_matches_its_analytic_std() {
        let mut rng = Rng::seed(4);
        let (fan_in, fan_out) = (200, 100);
        let t = xavier_normal([fan_out, fan_in], 1.0, DType::F32, &CPU, &mut rng).unwrap();
        let want_std = (2.0 / (fan_in + fan_out) as f64).sqrt();
        let (_, std) = mean_std(&t.to_vec::<f32>().unwrap());
        assert!((std - want_std).abs() / want_std < 0.1);
    }

    #[test]
    fn zeros_and_ones_are_exact() {
        let z = zeros([4], DType::F32, &CPU).unwrap();
        assert_eq!(z.to_vec::<f32>().unwrap(), vec![0.0; 4]);
        let o = ones([4], DType::F32, &CPU).unwrap();
        assert_eq!(o.to_vec::<f32>().unwrap(), vec![1.0; 4]);
    }

    #[test]
    fn xavier_uniform_gain_one_reproduces_attention_proj_scheme() {
        // `Proj::new` (src/nn/attention.rs) draws its Xavier-uniform weight
        // with an identical Tensor::rand-plus-affine formula in the same
        // order, so the two must agree bit for bit from the same seed.
        let mut a = Rng::seed(42);
        let mut b = Rng::seed(42);
        let attn = crate::nn::MultiHeadAttention::new(8, 2, &CPU, &mut a).unwrap();
        let via_init = xavier_uniform([8, 8], 1.0, DType::F32, &CPU, &mut b).unwrap();
        // `Proj::new` draws q/k/v/out in order; q_proj's weight is the first draw.
        let q_weight = attn.state_dict()["q_proj.weight"].to_vec::<f32>().unwrap();
        assert_eq!(q_weight, via_init.to_vec::<f32>().unwrap());
    }

    fn linear_new_weight(rng: &mut Rng) -> Vec<f32> {
        crate::nn::Linear::new(4, 3, &CPU, rng)
            .unwrap()
            .weight()
            .value()
            .to_vec::<f32>()
            .unwrap()
    }

    #[test]
    fn with_init_changes_the_weight_versus_new() {
        let mut rng = Rng::seed(5);
        let default = linear_new_weight(&mut rng);
        let mut rng2 = Rng::seed(5);
        let custom = Linear::with_init(4, 3, &CPU, |shape, device| {
            xavier_uniform(shape.to_vec(), 1.0, DType::F32, device, &mut rng2)
        })
        .unwrap();
        assert_ne!(default, custom.weight().value().to_vec::<f32>().unwrap());
    }

    #[test]
    fn apply_reaches_nested_params_and_skips_buffers() {
        #[derive(rstorch::Module)]
        struct Net {
            inner: crate::nn::Linear,
            norm: BatchNorm2d,
        }
        let mut rng = Rng::seed(6);
        let mut net = Net {
            inner: crate::nn::Linear::new(4, 4, &CPU, &mut rng).unwrap(),
            norm: BatchNorm2d::new(4, &CPU).unwrap(),
        };
        let running_mean_before = net.norm.state_dict()["running_mean"]
            .to_vec::<f32>()
            .unwrap();
        let weight_before = net.inner.weight().value().to_vec::<f32>().unwrap();
        let mut apply_rng = Rng::seed(7);
        apply(&mut net, |path, t| {
            if path == "inner.weight" {
                kaiming_uniform(
                    t.dims().to_vec(),
                    1.0,
                    t.dtype(),
                    &t.device(),
                    &mut apply_rng,
                )
            } else {
                Ok(t.clone())
            }
        })
        .unwrap();
        // The nested Linear's weight (rank 2) changed...
        assert_ne!(
            weight_before,
            net.inner.weight().value().to_vec::<f32>().unwrap()
        );
        // ...but BatchNorm2d's running_mean buffer was never touched, because
        // `apply` only visits `Param`s.
        assert_eq!(
            running_mean_before,
            net.norm.state_dict()["running_mean"]
                .to_vec::<f32>()
                .unwrap()
        );
    }
}
