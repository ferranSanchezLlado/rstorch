//! Compile-time checked normalization layers.
//!
//! [`LayerNorm`] and [`RMSNorm`] use a typed tensor as their suffix
//! specification. For example, `LayerNorm<Tensor2<3, 4>>` normalizes the last
//! two axes and owns `weight` and `bias` parameters of shape `[3, 4]`.
//! [`BatchNorm2d`] uses its channel count directly as a const marker.

use super::{
    Forward, Mode, Module, ToDType, ToDevice, TypedBuffer, TypedParam, TypedVisitor,
    TypedVisitorMut,
};
use crate::nn::{check_eps, check_normalized_shape, check_suffix};
use crate::typed::device::validate_binding;
use crate::typed::ops::{WithElement, WithPlacement};
use crate::typed::sealed::TypedTensor as SealedTypedTensor;
use crate::typed::tensor::checked_wrap;
use crate::typed::{DYN, DeviceCtx, FloatElement, Placement, Tensor1, Tensor4, TypedTensor};
use crate::{Element, Error, Result, Shape, Tensor};
use std::sync::Arc;

const fn assert_suffix(input: &[usize], suffix: &[usize]) {
    assert!(
        input.len() >= suffix.len(),
        "typed normalization suffix rank exceeds input rank"
    );
    let offset = input.len() - suffix.len();
    let mut axis = 0;
    while axis < suffix.len() {
        assert!(
            input[offset + axis] == DYN
                || suffix[axis] == DYN
                || input[offset + axis] == suffix[axis],
            "typed normalization suffix dimensions are incompatible"
        );
        axis += 1;
    }
}

const fn assert_channels(input: usize, channels: usize) {
    assert!(
        input == DYN || channels == DYN || input == channels,
        "typed BatchNorm2d channel dimensions are incompatible"
    );
}

fn check_channels(op: &'static str, input: &Tensor, channels: usize, leaf: &Tensor) -> Result<()> {
    if leaf.dims1()? != channels {
        return Err(Error::ShapeMismatch {
            op,
            lhs: input.shape().clone(),
            rhs: leaf.shape().clone(),
        });
    }
    Ok(())
}

fn initialized<S>(
    shape: impl Into<Shape>,
    value: f64,
    ctx: &DeviceCtx<S::Placement>,
    op: &'static str,
) -> Result<S>
where
    S: TypedTensor,
    S::Elem: FloatElement,
{
    validate_binding::<S::Placement>(ctx.binding(), op)?;
    let shape = shape.into();
    check_normalized_shape(op, &shape)?;
    let tensor = Tensor::full(shape, value, S::Elem::DTYPE, &ctx.device())?;
    checked_wrap(tensor, Arc::clone(ctx.binding()), op)
}

/// Layer normalization over the suffix represented by `S`.
///
/// Fully static incompatible input suffixes fail during monomorphization. A
/// relation containing [`DYN`] is checked against runtime dimensions.
///
/// ```compile_fail
/// use rstorch::typed::{nn::{Forward, LayerNorm, Mode}, Cpu, DeviceCtx, Tensor1, Tensor2};
/// let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
/// let mut norm = LayerNorm::<Tensor1<4>>::new([4], &ctx).unwrap();
/// let input = Tensor2::<2, 3>::from_vec(vec![0.0; 6], [2, 3], &ctx).unwrap();
/// let _ = norm.forward(&input, Mode::EVAL);
/// ```
pub struct LayerNorm<S>
where
    S: TypedTensor,
    S::Elem: FloatElement,
{
    weight: TypedParam<S>,
    bias: TypedParam<S>,
    eps: f64,
}

impl<S> LayerNorm<S>
where
    S: TypedTensor,
    S::Elem: FloatElement,
{
    /// Default epsilon, matching [`crate::nn::LayerNorm`].
    pub const DEFAULT_EPS: f64 = 1e-5;

    /// Constructs a typed layer with ones for weight and zeros for bias.
    pub fn new(shape: impl Into<Shape>, ctx: &DeviceCtx<S::Placement>) -> Result<Self> {
        Self::with_eps(shape, Self::DEFAULT_EPS, ctx)
    }

    /// Constructs a typed layer with an explicit positive epsilon.
    pub fn with_eps(
        shape: impl Into<Shape>,
        eps: f64,
        ctx: &DeviceCtx<S::Placement>,
    ) -> Result<Self> {
        const OP: &str = "LayerNorm::new";
        check_eps(OP, eps)?;
        let shape = shape.into();
        Ok(Self {
            weight: TypedParam::new(initialized::<S>(shape.clone(), 1.0, ctx, OP)?)?,
            bias: TypedParam::new(initialized::<S>(shape, 0.0, ctx, OP)?)?,
            eps,
        })
    }

    /// Returns the epsilon added to the variance.
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Returns the typed scale parameter.
    pub fn weight(&self) -> &TypedParam<S> {
        &self.weight
    }

    /// Returns the typed scale parameter mutably, for freeze or replacement.
    pub fn weight_mut(&mut self) -> &mut TypedParam<S> {
        &mut self.weight
    }

    /// Returns the typed bias parameter.
    pub fn bias(&self) -> &TypedParam<S> {
        &self.bias
    }

    /// Returns the typed bias parameter mutably, for freeze or replacement.
    pub fn bias_mut(&mut self) -> &mut TypedParam<S> {
        &mut self.bias
    }
}

impl<I, S> Forward<I> for LayerNorm<S>
where
    I: TypedTensor<Elem = S::Elem, Placement = S::Placement>,
    S: TypedTensor,
    S::Elem: FloatElement,
{
    type Output = I;

    fn forward(&mut self, input: &I, mode: Mode) -> Result<I> {
        const {
            assert_suffix(
                <I as SealedTypedTensor>::MARKERS,
                <S as SealedTypedTensor>::MARKERS,
            )
        };
        validate_binding::<I::Placement>(input.binding(), "LayerNorm::forward")?;
        let weight = self.weight.get(mode)?;
        let bias = self.bias.get(mode)?;
        // `layer_norm_forward` checks the weight's suffix itself with this
        // exact op and payload; the bias's is the one relation it does not
        // check, and without this the rejection would name `fused_layer_norm`.
        check_suffix(
            "LayerNorm::forward",
            input.dynamic(),
            bias.dynamic().shape(),
        )?;
        checked_wrap(
            crate::nn::layer_norm_forward(
                input.dynamic(),
                weight.dynamic(),
                Some(bias.dynamic()),
                self.eps,
            )?,
            Arc::clone(input.binding()),
            "LayerNorm::forward",
        )
    }
}

impl<S> Module for LayerNorm<S>
where
    S: TypedTensor,
    S::Elem: FloatElement,
{
    fn visit(&self, visitor: &mut TypedVisitor<'_>) {
        visitor.param("weight", &self.weight);
        visitor.param("bias", &self.bias);
    }
    fn visit_mut(&mut self, visitor: &mut TypedVisitorMut<'_>) {
        visitor.param("weight", &mut self.weight);
        visitor.param("bias", &mut self.bias);
    }
}

impl<S, Q> ToDevice<Q> for LayerNorm<S>
where
    S: TypedTensor + WithPlacement<Q>,
    S::Elem: FloatElement,
    Q: Placement,
    <S as WithPlacement<Q>>::Output: TypedTensor<Elem = S::Elem, Placement = Q>,
    <<S as WithPlacement<Q>>::Output as TypedTensor>::Elem: FloatElement,
{
    type Output = LayerNorm<<S as WithPlacement<Q>>::Output>;
    fn to_device(self, target: &DeviceCtx<Q>) -> Result<Self::Output> {
        Ok(LayerNorm {
            weight: self.weight.to_device(target)?,
            bias: self.bias.to_device(target)?,
            eps: self.eps,
        })
    }
}

impl<S, F> ToDType<F> for LayerNorm<S>
where
    S: TypedTensor + WithElement<F>,
    S::Elem: FloatElement,
    F: FloatElement,
    <S as WithElement<F>>::Output: TypedTensor<Elem = F, Placement = S::Placement>,
{
    type Output = LayerNorm<<S as WithElement<F>>::Output>;
    fn to_dtype(self) -> Result<Self::Output> {
        Ok(LayerNorm {
            weight: self.weight.to_dtype()?,
            bias: self.bias.to_dtype()?,
            eps: self.eps,
        })
    }
}

/// Root-mean-square normalization over the suffix represented by `S`.
///
/// It has one parameter named `weight` and preserves the exact input type.
///
/// ```compile_fail
/// use rstorch::typed::{nn::{Forward, Mode, RMSNorm}, Cpu, DeviceCtx, Tensor1, Tensor3};
/// let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
/// let mut norm = RMSNorm::<Tensor1<4>>::new([4], &ctx).unwrap();
/// let input = Tensor3::<2, 3, 5>::from_vec(vec![0.0; 30], [2, 3, 5], &ctx).unwrap();
/// let _ = norm.forward(&input, Mode::EVAL);
/// ```
pub struct RMSNorm<S>
where
    S: TypedTensor,
    S::Elem: FloatElement,
{
    weight: TypedParam<S>,
    eps: f64,
}

impl<S> RMSNorm<S>
where
    S: TypedTensor,
    S::Elem: FloatElement,
{
    /// Default epsilon, matching [`crate::nn::RMSNorm`].
    pub const DEFAULT_EPS: f64 = 1e-6;

    /// Constructs a typed RMSNorm with a unit scale.
    pub fn new(shape: impl Into<Shape>, ctx: &DeviceCtx<S::Placement>) -> Result<Self> {
        Self::with_eps(shape, Self::DEFAULT_EPS, ctx)
    }

    /// Constructs a typed RMSNorm with an explicit positive epsilon.
    pub fn with_eps(
        shape: impl Into<Shape>,
        eps: f64,
        ctx: &DeviceCtx<S::Placement>,
    ) -> Result<Self> {
        const OP: &str = "RMSNorm::new";
        check_eps(OP, eps)?;
        Ok(Self {
            weight: TypedParam::new(initialized::<S>(shape, 1.0, ctx, OP)?)?,
            eps,
        })
    }

    /// Returns the epsilon added to the mean square.
    pub fn eps(&self) -> f64 {
        self.eps
    }

    /// Returns the typed scale parameter.
    pub fn weight(&self) -> &TypedParam<S> {
        &self.weight
    }

    /// Returns the typed scale parameter mutably.
    pub fn weight_mut(&mut self) -> &mut TypedParam<S> {
        &mut self.weight
    }
}

impl<I, S> Forward<I> for RMSNorm<S>
where
    I: TypedTensor<Elem = S::Elem, Placement = S::Placement>,
    S: TypedTensor,
    S::Elem: FloatElement,
{
    type Output = I;
    fn forward(&mut self, input: &I, mode: Mode) -> Result<I> {
        const {
            assert_suffix(
                <I as SealedTypedTensor>::MARKERS,
                <S as SealedTypedTensor>::MARKERS,
            )
        };
        validate_binding::<I::Placement>(input.binding(), "RMSNorm::forward")?;
        let weight = self.weight.get(mode)?;
        // `rms_norm_forward` checks the suffix itself, with this op and payload.
        checked_wrap(
            crate::nn::rms_norm_forward(input.dynamic(), weight.dynamic(), self.eps)?,
            Arc::clone(input.binding()),
            "RMSNorm::forward",
        )
    }
}

impl<S> Module for RMSNorm<S>
where
    S: TypedTensor,
    S::Elem: FloatElement,
{
    fn visit(&self, visitor: &mut TypedVisitor<'_>) {
        visitor.param("weight", &self.weight);
    }
    fn visit_mut(&mut self, visitor: &mut TypedVisitorMut<'_>) {
        visitor.param("weight", &mut self.weight);
    }
}

impl<S, Q> ToDevice<Q> for RMSNorm<S>
where
    S: TypedTensor + WithPlacement<Q>,
    S::Elem: FloatElement,
    Q: Placement,
    <S as WithPlacement<Q>>::Output: TypedTensor<Elem = S::Elem, Placement = Q>,
    <<S as WithPlacement<Q>>::Output as TypedTensor>::Elem: FloatElement,
{
    type Output = RMSNorm<<S as WithPlacement<Q>>::Output>;
    fn to_device(self, target: &DeviceCtx<Q>) -> Result<Self::Output> {
        Ok(RMSNorm {
            weight: self.weight.to_device(target)?,
            eps: self.eps,
        })
    }
}

impl<S, F> ToDType<F> for RMSNorm<S>
where
    S: TypedTensor + WithElement<F>,
    S::Elem: FloatElement,
    F: FloatElement,
    <S as WithElement<F>>::Output: TypedTensor<Elem = F, Placement = S::Placement>,
{
    type Output = RMSNorm<<S as WithElement<F>>::Output>;
    fn to_dtype(self) -> Result<Self::Output> {
        Ok(RMSNorm {
            weight: self.weight.to_dtype()?,
            eps: self.eps,
        })
    }
}

/// Batch normalization for rank-four `NCHW` tensors.
///
/// Static channel mismatches fail during monomorphization. `C = DYN` permits a
/// runtime channel count supplied to the constructor and checked on forward.
///
/// ```compile_fail
/// use rstorch::typed::{nn::{BatchNorm2d, Forward, Mode}, Cpu, DeviceCtx, Tensor4};
/// let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
/// let mut norm = BatchNorm2d::<3>::new(3, &ctx).unwrap();
/// let input = Tensor4::<1, 2, 4, 4>::from_vec(vec![0.0; 32], [1, 2, 4, 4], &ctx).unwrap();
/// let _ = norm.forward(&input, Mode::EVAL);
/// ```
pub struct BatchNorm2d<const C: usize, E: FloatElement = f32, P: Placement = crate::typed::Cpu> {
    weight: TypedParam<Tensor1<C, E, P>>,
    bias: TypedParam<Tensor1<C, E, P>>,
    running_mean: TypedBuffer<Tensor1<C, E, P>>,
    running_var: TypedBuffer<Tensor1<C, E, P>>,
    eps: f64,
    momentum: f64,
}

impl<const C: usize, E: FloatElement, P: Placement> BatchNorm2d<C, E, P> {
    /// Default epsilon, matching [`crate::nn::BatchNorm2d`].
    pub const DEFAULT_EPS: f64 = 1e-5;
    /// Default running-statistic momentum, matching [`crate::nn::BatchNorm2d`].
    pub const DEFAULT_MOMENTUM: f64 = 0.1;

    /// Constructs a typed BatchNorm2d for `channels` actual channels.
    pub fn new(channels: usize, ctx: &DeviceCtx<P>) -> Result<Self> {
        Self::with_params(channels, Self::DEFAULT_EPS, Self::DEFAULT_MOMENTUM, ctx)
    }

    /// Constructs a typed BatchNorm2d with explicit epsilon and momentum.
    pub fn with_params(
        channels: usize,
        eps: f64,
        momentum: f64,
        ctx: &DeviceCtx<P>,
    ) -> Result<Self> {
        const OP: &str = "BatchNorm2d::new";
        if channels == 0 || (C != DYN && C != channels) {
            return Err(Error::InvalidArg {
                op: OP,
                msg: format!("channels must be non-zero and satisfy marker {C}, got {channels}"),
            });
        }
        check_eps(OP, eps)?;
        if !(momentum.is_finite() && (0.0..=1.0).contains(&momentum)) {
            return Err(Error::InvalidArg {
                op: OP,
                msg: format!("momentum must lie in [0, 1], got {momentum}"),
            });
        }
        let make = |value| {
            checked_wrap::<Tensor1<C, E, P>>(
                Tensor::full([channels], value, E::DTYPE, &ctx.device())?,
                Arc::clone(ctx.binding()),
                OP,
            )
        };
        Ok(Self {
            weight: TypedParam::new(make(1.0)?)?,
            bias: TypedParam::new(make(0.0)?)?,
            running_mean: TypedBuffer::new(make(0.0)?)?,
            running_var: TypedBuffer::new(make(1.0)?)?,
            eps,
            momentum,
        })
    }

    /// Returns the actual channel count.
    pub fn channels(&self) -> usize {
        self.running_mean
            .value()
            .expect("typed BatchNorm2d buffer invariant")
            .dims()[0]
    }
    /// Returns epsilon.
    pub fn eps(&self) -> f64 {
        self.eps
    }
    /// Returns running-statistic momentum.
    pub fn momentum(&self) -> f64 {
        self.momentum
    }
    /// Returns the typed scale parameter.
    pub fn weight(&self) -> &TypedParam<Tensor1<C, E, P>> {
        &self.weight
    }
    /// Returns the typed scale parameter mutably.
    pub fn weight_mut(&mut self) -> &mut TypedParam<Tensor1<C, E, P>> {
        &mut self.weight
    }
    /// Returns the typed bias parameter.
    pub fn bias(&self) -> &TypedParam<Tensor1<C, E, P>> {
        &self.bias
    }
    /// Returns the typed bias parameter mutably.
    pub fn bias_mut(&mut self) -> &mut TypedParam<Tensor1<C, E, P>> {
        &mut self.bias
    }
    /// Returns the running mean buffer value.
    pub fn running_mean(&self) -> Result<Tensor1<C, E, P>> {
        self.running_mean.value()
    }
    /// Returns the running variance buffer value.
    pub fn running_var(&self) -> Result<Tensor1<C, E, P>> {
        self.running_var.value()
    }
}

impl<const N: usize, const IC: usize, const H: usize, const W: usize, const C: usize, E, P>
    Forward<Tensor4<N, IC, H, W, E, P>> for BatchNorm2d<C, E, P>
where
    E: FloatElement,
    P: Placement,
{
    type Output = Tensor4<N, IC, H, W, E, P>;
    fn forward(&mut self, input: &Self::Output, mode: Mode) -> Result<Self::Output> {
        const { assert_channels(IC, C) };
        const OP: &str = "BatchNorm2d::forward";
        validate_binding::<P>(input.binding(), OP)?;
        let [_, channels, _, _] = input.dims();
        let weight = self.weight.get(mode)?;
        let bias = self.bias.get(mode)?;
        let running_mean = self.running_mean.value()?;
        let running_var = self.running_var.value()?;
        for leaf in [
            weight.dynamic(),
            bias.dynamic(),
            running_mean.dynamic(),
            running_var.dynamic(),
        ] {
            check_channels(OP, input.dynamic(), channels, leaf)?;
        }

        let (output, replacements) = crate::nn::batch_norm2d_forward(
            input.dynamic(),
            weight.dynamic(),
            bias.dynamic(),
            running_mean.dynamic(),
            running_var.dynamic(),
            self.eps,
            self.momentum,
            mode,
        )?;
        if let Some((next_mean, next_var)) = replacements {
            let next_mean = checked_wrap(next_mean, Arc::clone(input.binding()), OP)?;
            let next_var = checked_wrap(next_var, Arc::clone(input.binding()), OP)?;
            self.running_mean.set(next_mean)?;
            self.running_var.set(next_var)?;
        }
        checked_wrap(output, Arc::clone(input.binding()), OP)
    }
}

impl<const C: usize, E: FloatElement, P: Placement> Module for BatchNorm2d<C, E, P> {
    fn visit(&self, visitor: &mut TypedVisitor<'_>) {
        visitor.param("weight", &self.weight);
        visitor.param("bias", &self.bias);
        visitor.buffer("running_mean", &self.running_mean);
        visitor.buffer("running_var", &self.running_var);
    }
    fn visit_mut(&mut self, visitor: &mut TypedVisitorMut<'_>) {
        visitor.param("weight", &mut self.weight);
        visitor.param("bias", &mut self.bias);
        visitor.buffer("running_mean", &mut self.running_mean);
        visitor.buffer("running_var", &mut self.running_var);
    }
}

impl<const C: usize, E, P, Q> ToDevice<Q> for BatchNorm2d<C, E, P>
where
    E: FloatElement,
    P: Placement,
    Q: Placement,
{
    type Output = BatchNorm2d<C, E, Q>;
    fn to_device(self, target: &DeviceCtx<Q>) -> Result<Self::Output> {
        Ok(BatchNorm2d {
            weight: self.weight.to_device(target)?,
            bias: self.bias.to_device(target)?,
            running_mean: self.running_mean.to_device(target)?,
            running_var: self.running_var.to_device(target)?,
            eps: self.eps,
            momentum: self.momentum,
        })
    }
}

impl<const C: usize, E, P, F> ToDType<F> for BatchNorm2d<C, E, P>
where
    E: FloatElement,
    P: Placement,
    F: FloatElement,
{
    type Output = BatchNorm2d<C, F, P>;
    fn to_dtype(self) -> Result<Self::Output> {
        Ok(BatchNorm2d {
            weight: self.weight.to_dtype()?,
            bias: self.bias.to_dtype()?,
            running_mean: self.running_mean.to_dtype()?,
            running_var: self.running_var.to_dtype()?,
            eps: self.eps,
            momentum: self.momentum,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Forward as RuntimeForward;
    use crate::typed::sealed::DeviceBinding;
    use crate::typed::{Cpu, Tensor1, Tensor2};
    use crate::{DType, Device};

    trait Same<T> {}
    impl<T> Same<T> for T {}
    fn exact<T: Same<U>, U>(_: &T) {}

    /// The rendered message of an expected rejection. The typed layers are
    /// deliberately not `Debug` (their parameter values are reached through
    /// `state_dict`), so `unwrap_err` is unavailable on their constructors.
    #[track_caller]
    fn rejection<T>(result: Result<T>) -> String {
        match result {
            Ok(_) => panic!("expected a rejection"),
            Err(error) => error.to_string(),
        }
    }

    #[test]
    fn layer_and_rms_are_shape_preserving_and_match_runtime_values() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let input =
            Tensor2::<2, 4>::from_vec((1..=8).map(|x| x as f32).collect(), [2, 4], &ctx).unwrap();
        let mut typed = LayerNorm::<Tensor1<4>>::new([4], &ctx).unwrap();
        let output = typed.forward(&input, Mode::EVAL).unwrap();
        exact::<_, Tensor2<2, 4>>(&output);
        let mut runtime = crate::nn::LayerNorm::new([4], &ctx.device()).unwrap();
        let expected = RuntimeForward::forward(&mut runtime, input.dynamic(), Mode::EVAL).unwrap();
        assert_eq!(output.to_vec().unwrap(), expected.to_vec::<f32>().unwrap());

        let mut typed = RMSNorm::<Tensor1<4>>::new([4], &ctx).unwrap();
        let output = typed.forward(&input, Mode::EVAL).unwrap();
        let mut runtime = crate::nn::RMSNorm::new([4], &ctx.device()).unwrap();
        let expected = RuntimeForward::forward(&mut runtime, input.dynamic(), Mode::EVAL).unwrap();
        assert_eq!(output.to_vec().unwrap(), expected.to_vec::<f32>().unwrap());
    }

    /// Every constructor rejection, with the exact message — the typed layers
    /// route these through the runtime's own validators, so a reworded runtime
    /// message cannot drift away from the typed layer's silently.
    #[test]
    fn constructor_rejections_and_reported_configuration_match_the_runtime_rules() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();

        for eps in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let expected = format!("invalid argument: eps must be finite and positive, got {eps}");
            assert_eq!(
                rejection(LayerNorm::<Tensor1<4>>::with_eps([4], eps, &ctx)),
                format!("LayerNorm::new: {expected}")
            );
            assert_eq!(
                rejection(RMSNorm::<Tensor1<4>>::with_eps([4], eps, &ctx)),
                format!("RMSNorm::new: {expected}")
            );
            assert_eq!(
                rejection(BatchNorm2d::<2>::with_params(2, eps, 0.1, &ctx)),
                format!("BatchNorm2d::new: {expected}")
            );
            // The same rule as spelled by the runtime layer, verbatim.
            assert_eq!(
                rejection(crate::nn::LayerNorm::with_eps([4], eps, &ctx.device())),
                format!("LayerNorm::new: {expected}")
            );
        }

        for momentum in [1.5, -0.1, f64::NAN] {
            assert_eq!(
                rejection(BatchNorm2d::<2>::with_params(2, 1e-5, momentum, &ctx)),
                format!(
                    "BatchNorm2d::new: invalid argument: momentum must lie in [0, 1], \
                     got {momentum}"
                )
            );
        }
        assert!(BatchNorm2d::<2>::with_params(2, 1e-5, 0.0, &ctx).is_ok());
        assert!(BatchNorm2d::<2>::with_params(2, 1e-5, 1.0, &ctx).is_ok());
        for channels in [0, 3] {
            assert_eq!(
                rejection(BatchNorm2d::<2>::new(channels, &ctx)),
                format!(
                    "BatchNorm2d::new: invalid argument: channels must be non-zero and \
                     satisfy marker 2, got {channels}"
                )
            );
        }

        // A `normalized_shape` contradicting a static marker is a wrap failure;
        // a degenerate one is the runtime's `check_normalized_shape` rule, so it
        // must be reported exactly as the runtime layer reports it.
        assert_eq!(
            rejection(LayerNorm::<Tensor1<4>>::new([5], &ctx)),
            "LayerNorm::new: shape mismatch: lhs [5] vs rhs [4]"
        );
        for shape in [vec![], vec![0], vec![2, 0]] {
            let message = rejection(LayerNorm::<Tensor1<DYN>>::new(shape.clone(), &ctx));
            assert_eq!(
                message,
                rejection(crate::nn::LayerNorm::new(shape.clone(), &ctx.device())),
                "typed and runtime disagree for {shape:?}"
            );
            assert!(
                message.starts_with(
                    "LayerNorm::new: invalid argument: normalized_shape must have rank >= 1"
                ),
                "{message}"
            );
        }

        // The accessors report the configuration actually installed, and the
        // typed defaults are the runtime defaults.
        assert_eq!(
            LayerNorm::<Tensor1<4>>::with_eps([4], 1e-3, &ctx)
                .unwrap()
                .eps(),
            1e-3
        );
        assert_eq!(LayerNorm::<Tensor1<4>>::new([4], &ctx).unwrap().eps(), 1e-5);
        assert_eq!(
            LayerNorm::<Tensor1<4>>::DEFAULT_EPS,
            crate::nn::LayerNorm::DEFAULT_EPS
        );

        assert_eq!(
            RMSNorm::<Tensor1<4>>::with_eps([4], 1e-2, &ctx)
                .unwrap()
                .eps(),
            1e-2
        );
        assert_eq!(RMSNorm::<Tensor1<4>>::new([4], &ctx).unwrap().eps(), 1e-6);
        assert_eq!(
            RMSNorm::<Tensor1<4>>::DEFAULT_EPS,
            crate::nn::RMSNorm::DEFAULT_EPS
        );

        let dynamic = BatchNorm2d::<DYN>::with_params(3, 1e-4, 0.25, &ctx).unwrap();
        assert_eq!(
            (dynamic.channels(), dynamic.eps(), dynamic.momentum()),
            (3, 1e-4, 0.25)
        );
        let bn = BatchNorm2d::<2>::new(2, &ctx).unwrap();
        assert_eq!((bn.channels(), bn.eps(), bn.momentum()), (2, 1e-5, 0.1));
        assert_eq!(
            (
                BatchNorm2d::<2>::DEFAULT_EPS,
                BatchNorm2d::<2>::DEFAULT_MOMENTUM
            ),
            (
                crate::nn::BatchNorm2d::DEFAULT_EPS,
                crate::nn::BatchNorm2d::DEFAULT_MOMENTUM
            )
        );
    }

    #[test]
    fn dynamic_suffix_and_channels_report_runtime_mismatches() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let input = Tensor2::<DYN, DYN>::from_vec(vec![0.0; 6], [2, 3], &ctx).unwrap();
        let mut norm = LayerNorm::<Tensor1<DYN>>::new([4], &ctx).unwrap();
        assert!(matches!(
            norm.forward(&input, Mode::EVAL),
            Err(Error::ShapeMismatch { .. })
        ));
        let mut norm = RMSNorm::<Tensor1<DYN>>::new([4], &ctx).unwrap();
        assert!(matches!(
            norm.forward(&input, Mode::EVAL),
            Err(Error::ShapeMismatch { .. })
        ));

        let input = Tensor4::<1, DYN, 2, 2>::from_vec(vec![0.0; 8], [1, 2, 2, 2], &ctx).unwrap();
        let mut norm = BatchNorm2d::<DYN>::new(3, &ctx).unwrap();
        assert!(matches!(
            norm.forward(&input, Mode::EVAL),
            Err(Error::ShapeMismatch { .. })
        ));
    }

    fn dyn_vector(values: &[f32], ctx: &DeviceCtx<Cpu>) -> Tensor1<DYN> {
        Tensor1::from_vec(values.to_vec(), [values.len()], ctx).unwrap()
    }

    #[test]
    fn dynamic_affine_and_buffer_inconsistency_rejects_before_batch_updates() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let row = Tensor2::<1, DYN>::from_vec(vec![1.0, 2.0], [1, 2], &ctx).unwrap();

        // Both rejections must name `LayerNorm::forward` rather than a fused or
        // composed internal: the message is the whole payload, and a bare
        // `ShapeMismatch { .. }` match would pass either way — including with
        // the bias guard deleted, since the delegate then reports
        // `fused_layer_norm`.
        let mut layer = LayerNorm::<Tensor1<DYN>>::new([2], &ctx).unwrap();
        layer.weight = TypedParam::new(dyn_vector(&[1.0, 1.0, 1.0], &ctx)).unwrap();
        assert_eq!(
            layer.forward(&row, Mode::TRAIN).unwrap_err().to_string(),
            "LayerNorm::forward: shape mismatch: lhs [1, 2] vs rhs [3]"
        );
        let mut layer = LayerNorm::<Tensor1<DYN>>::new([2], &ctx).unwrap();
        layer.bias = TypedParam::new(dyn_vector(&[0.0, 0.0, 0.0], &ctx)).unwrap();
        assert_eq!(
            layer.forward(&row, Mode::TRAIN).unwrap_err().to_string(),
            "LayerNorm::forward: shape mismatch: lhs [1, 2] vs rhs [3]"
        );

        let image = Tensor4::<1, DYN, 1, 2>::from_vec(vec![1.0, 3.0, 2.0, 6.0], [1, 2, 1, 2], &ctx)
            .unwrap();
        macro_rules! rejects_leaf {
            ($field:ident, $replacement:expr) => {{
                let mut norm = BatchNorm2d::<DYN>::new(2, &ctx).unwrap();
                norm.$field = $replacement;
                let mean_before = norm.running_mean().unwrap().to_vec().unwrap();
                let var_before = norm.running_var().unwrap().to_vec().unwrap();
                assert!(matches!(
                    norm.forward(&image, Mode::TRAIN),
                    Err(Error::ShapeMismatch { .. })
                ));
                assert_eq!(norm.running_mean().unwrap().to_vec().unwrap(), mean_before);
                assert_eq!(norm.running_var().unwrap().to_vec().unwrap(), var_before);
            }};
        }
        rejects_leaf!(
            weight,
            TypedParam::new(dyn_vector(&[1.0, 1.0, 1.0], &ctx)).unwrap()
        );
        rejects_leaf!(
            bias,
            TypedParam::new(dyn_vector(&[0.0, 0.0, 0.0], &ctx)).unwrap()
        );
        rejects_leaf!(
            running_mean,
            TypedBuffer::new(dyn_vector(&[0.0, 0.0, 0.0], &ctx)).unwrap()
        );
        rejects_leaf!(
            running_var,
            TypedBuffer::new(dyn_vector(&[1.0, 1.0, 1.0], &ctx)).unwrap()
        );
    }

    #[test]
    fn gradients_and_exact_state_paths_include_parameters_and_buffers() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let input =
            Tensor2::<2, 3>::from_vec(vec![0.5, -1.5, 2.0, 0.25, -0.75, 1.25], [2, 3], &ctx)
                .unwrap()
                .traced()
                .unwrap();
        let mut norm = LayerNorm::<Tensor1<3>>::new([3], &ctx).unwrap();
        let loss = norm
            .forward(&input, Mode::TRAIN)
            .unwrap()
            .sum_all()
            .unwrap();
        let grads = loss.backward().unwrap();
        assert_eq!(norm.weight().grad_from(&grads).unwrap().dims(), [3]);
        assert_eq!(norm.bias().grad_from(&grads).unwrap().dims(), [3]);
        assert_eq!(grads.wrt_input(input.dynamic()).unwrap().dims(), [2, 3]);
        assert_eq!(
            super::super::state_dict(&norm)
                .unwrap()
                .paths()
                .collect::<Vec<_>>(),
            ["bias", "weight"]
        );

        let bn = BatchNorm2d::<2>::new(2, &ctx).unwrap();
        assert_eq!(
            super::super::state_dict(&bn)
                .unwrap()
                .paths()
                .collect::<Vec<_>>(),
            ["bias", "running_mean", "running_var", "weight"]
        );

        let mut rms = RMSNorm::<Tensor1<3>>::new([3], &ctx).unwrap();
        let rms_loss = rms.forward(&input, Mode::TRAIN).unwrap().sum_all().unwrap();
        let rms_grads = rms_loss.backward().unwrap();
        assert_eq!(rms.weight().grad_from(&rms_grads).unwrap().dims(), [3]);
        assert_eq!(rms_grads.wrt_input(input.dynamic()).unwrap().dims(), [2, 3]);

        let image = Tensor4::<1, 2, 1, 2>::from_vec(vec![1.0, 3.0, 2.0, 6.0], [1, 2, 1, 2], &ctx)
            .unwrap()
            .traced()
            .unwrap();
        let mut bn = BatchNorm2d::<2>::new(2, &ctx).unwrap();
        let bn_loss = bn
            .forward(&image, Mode::TRAIN)
            .unwrap()
            .square()
            .unwrap()
            .sum_all()
            .unwrap();
        let bn_grads = bn_loss.backward().unwrap();
        assert_eq!(bn.weight().grad_from(&bn_grads).unwrap().dims(), [2]);
        assert_eq!(bn.bias().grad_from(&bn_grads).unwrap().dims(), [2]);
        assert_eq!(
            bn_grads.wrt_input(image.dynamic()).unwrap().dims(),
            [1, 2, 1, 2]
        );
    }

    #[test]
    fn forged_noncanonical_input_binding_is_rejected_before_arithmetic() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let dynamic = Tensor::from_vec(vec![1.0f32, 2.0], [1, 2], &Device::Cpu).unwrap();
        let forged = Arc::new(DeviceBinding {
            device: Device::Cpu,
        });
        let input = <Tensor2<1, 2> as SealedTypedTensor>::trusted_from_validated(dynamic, forged);
        let mut norm = LayerNorm::<Tensor1<2>>::new([2], &ctx).unwrap();
        assert!(matches!(
            norm.forward(&input, Mode::EVAL),
            Err(Error::InvalidArg {
                op: "LayerNorm::forward",
                ..
            })
        ));
    }

    #[test]
    fn batch_norm_train_eval_and_state_movement_match_runtime() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let input = Tensor4::<2, 2, 2, 2>::from_vec(
            (0..16).map(|x| x as f32).collect(),
            [2, 2, 2, 2],
            &ctx,
        )
        .unwrap();
        let mut typed = BatchNorm2d::<2>::new(2, &ctx).unwrap();
        let mut runtime = crate::nn::BatchNorm2d::new(2, &ctx.device()).unwrap();
        let got = typed.forward(&input, Mode::TRAIN).unwrap();
        let want = RuntimeForward::forward(&mut runtime, input.dynamic(), Mode::TRAIN).unwrap();
        assert_eq!(got.to_vec().unwrap(), want.to_vec::<f32>().unwrap());
        assert_eq!(
            typed.running_mean().unwrap().to_vec().unwrap(),
            runtime.running_mean().to_vec::<f32>().unwrap()
        );
        assert_eq!(
            typed.running_var().unwrap().to_vec().unwrap(),
            runtime.running_var().to_vec::<f32>().unwrap()
        );

        let before = typed.running_mean().unwrap().to_vec().unwrap();
        let got = typed.forward(&input, Mode::EVAL.recorded()).unwrap();
        let want =
            RuntimeForward::forward(&mut runtime, input.dynamic(), Mode::EVAL.recorded()).unwrap();
        assert_eq!(got.to_vec().unwrap(), want.to_vec::<f32>().unwrap());
        assert_eq!(typed.running_mean().unwrap().to_vec().unwrap(), before);

        let moved = <BatchNorm2d<2> as ToDType<half::f16>>::to_dtype(typed).unwrap();
        assert_eq!(
            moved.running_mean().unwrap().to_vec().unwrap(),
            before
                .iter()
                .copied()
                .map(half::f16::from_f32)
                .collect::<Vec<_>>()
        );
        let moved = moved.to_device(&ctx).unwrap();
        assert_eq!(moved.running_mean().unwrap().to_vec().unwrap().len(), 2);

        let state = super::super::state_dict(&moved).unwrap();
        let mut restored = <BatchNorm2d<2> as ToDType<half::f16>>::to_dtype(
            BatchNorm2d::<2>::new(2, &ctx).unwrap(),
        )
        .unwrap();
        super::super::load_state_dict(&mut restored, &state).unwrap();
        assert_eq!(
            restored.running_mean().unwrap().to_vec().unwrap(),
            moved.running_mean().unwrap().to_vec().unwrap()
        );
    }

    #[test]
    fn reduced_precision_layer_norm_matches_runtime_and_keeps_gradients_typed() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let values = vec![0.5f32, -1.5, 2.0, 0.25, -0.75, 1.25];
        let input = Tensor2::<2, 3>::from_vec(values, [2, 3], &ctx)
            .unwrap()
            .to_dtype::<half::f16>()
            .unwrap()
            .traced()
            .unwrap();
        let mut typed = LayerNorm::<Tensor1<3, half::f16>>::new([3], &ctx).unwrap();
        let output = typed.forward(&input, Mode::TRAIN).unwrap();

        let mut runtime = crate::nn::LayerNorm::new([3], &ctx.device()).unwrap();
        crate::nn::to_dtype(&mut runtime, DType::F16).unwrap();
        let expected = RuntimeForward::forward(&mut runtime, input.dynamic(), Mode::TRAIN).unwrap();
        assert_eq!(
            output.to_vec().unwrap(),
            expected.to_vec::<half::f16>().unwrap()
        );

        let reference_input = input.dynamic().detach().traced().unwrap();
        let reference_weight = typed
            .weight()
            .value()
            .unwrap()
            .dynamic()
            .detach()
            .traced()
            .unwrap();
        let reference_bias = typed
            .bias()
            .value()
            .unwrap()
            .dynamic()
            .detach()
            .traced()
            .unwrap();
        let reference = crate::nn::layer_norm_forward(
            &reference_input,
            &reference_weight,
            Some(&reference_bias),
            LayerNorm::<Tensor1<3, half::f16>>::DEFAULT_EPS,
        )
        .unwrap();
        let grads = output
            .square()
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let reference_grads = reference
            .mul(&reference)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        assert_eq!(
            grads
                .wrt_input(input.dynamic())
                .unwrap()
                .to_vec::<half::f16>()
                .unwrap(),
            reference_grads
                .wrt_input(&reference_input)
                .unwrap()
                .to_vec::<half::f16>()
                .unwrap()
        );
        assert_eq!(
            typed.weight().grad_from(&grads).unwrap().to_vec().unwrap(),
            reference_grads
                .wrt_input(&reference_weight)
                .unwrap()
                .to_vec::<half::f16>()
                .unwrap()
        );
        assert_eq!(
            typed.bias().grad_from(&grads).unwrap().to_vec().unwrap(),
            reference_grads
                .wrt_input(&reference_bias)
                .unwrap()
                .to_vec::<half::f16>()
                .unwrap()
        );
    }

    #[test]
    fn reduced_precision_multi_axis_values_and_gradients_match_runtime_helper_exactly() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let values = vec![0.5f32, -1.5, 2.0, 0.25, -0.75, 1.25, 0.75, -0.25];
        let input = crate::typed::Tensor3::<2, 2, 2>::from_vec(values, [2, 2, 2], &ctx)
            .unwrap()
            .to_dtype::<half::f16>()
            .unwrap()
            .traced()
            .unwrap();
        let weight = Tensor2::<2, 2>::from_vec(vec![1.5, -0.5, 2.0, 0.25], [2, 2], &ctx)
            .unwrap()
            .to_dtype::<half::f16>()
            .unwrap();
        let bias = Tensor2::<2, 2>::from_vec(vec![0.25, -0.5, 0.75, -1.0], [2, 2], &ctx)
            .unwrap()
            .to_dtype::<half::f16>()
            .unwrap();
        let mut typed =
            LayerNorm::<Tensor2<2, 2, half::f16>>::with_eps([2, 2], 1e-3, &ctx).unwrap();
        typed.weight_mut().set(weight.clone()).unwrap();
        typed.bias_mut().set(bias.clone()).unwrap();
        let output = typed.forward(&input, Mode::TRAIN).unwrap();

        let reference_input = input.dynamic().detach().traced().unwrap();
        let reference_weight = weight.dynamic().detach().traced().unwrap();
        let reference_bias = bias.dynamic().detach().traced().unwrap();
        let reference = crate::nn::layer_norm_forward(
            &reference_input,
            &reference_weight,
            Some(&reference_bias),
            1e-3,
        )
        .unwrap();
        assert_eq!(
            output.to_vec().unwrap(),
            reference.to_vec::<half::f16>().unwrap()
        );

        let grads = output
            .square()
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let reference_grads = reference
            .mul(&reference)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        assert_eq!(
            grads
                .wrt_input(input.dynamic())
                .unwrap()
                .to_vec::<half::f16>()
                .unwrap(),
            reference_grads
                .wrt_input(&reference_input)
                .unwrap()
                .to_vec::<half::f16>()
                .unwrap()
        );
        assert_eq!(
            typed.weight().grad_from(&grads).unwrap().to_vec().unwrap(),
            reference_grads
                .wrt_input(&reference_weight)
                .unwrap()
                .to_vec::<half::f16>()
                .unwrap()
        );
        assert_eq!(
            typed.bias().grad_from(&grads).unwrap().to_vec().unwrap(),
            reference_grads
                .wrt_input(&reference_bias)
                .unwrap()
                .to_vec::<half::f16>()
                .unwrap()
        );
    }

    #[test]
    fn frozen_parameters_and_training_behavior_are_independent() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let input =
            Tensor4::<1, 2, 1, 2>::from_vec(vec![1.0, 3.0, 2.0, 6.0], [1, 2, 1, 2], &ctx).unwrap();
        let mut norm = BatchNorm2d::<2>::new(2, &ctx).unwrap();
        norm.weight_mut().freeze();
        norm.bias_mut().freeze();
        norm.forward(&input, Mode::TRAIN).unwrap();
        assert_ne!(
            norm.running_mean().unwrap().to_vec().unwrap(),
            vec![0.0, 0.0]
        );
        assert!(norm.weight().is_frozen());
    }
}
