use super::{Forward, Mode, ToDType, ToDevice, TypedParam};
use crate::typed::sealed::TypedTensor as SealedTypedTensor;
use crate::typed::tensor::checked_wrap;
use crate::typed::{
    DYN, DeviceCtx, FloatElement, NumericElement, Placement, Tensor1, Tensor2, Tensor3, Tensor4,
    Tensor5, Tensor6, Tensor7, Tensor8,
};
use crate::{DType, Error, Result, Rng};
use std::sync::Arc;

/// A typed affine layer with runtime state paths `weight` and optional `bias`.
///
/// `IN` and `OUT` describe the parameter geometry. Inputs may use `DYN` for
/// their trailing marker; known incompatible widths fail during
/// monomorphization, while any relationship involving `DYN` is checked by the
/// runtime layer contract.
///
/// ```compile_fail
/// use rstorch::typed::{DeviceCtx, Tensor2, nn::{Forward, Linear, Mode}};
/// use rstorch::Rng;
/// let ctx = DeviceCtx::cpu().unwrap();
/// let mut layer = Linear::<3, 2>::new(3, 2, &ctx, &mut Rng::seed(0)).unwrap();
/// let input = Tensor2::<4, 4>::from_vec(vec![0.0; 16], [4, 4], &ctx).unwrap();
/// let _ = layer.forward(&input, Mode::EVAL);
/// ```
///
/// Dtype and placement are part of the input type:
///
/// ```compile_fail
/// use rstorch::typed::{DeviceCtx, Tensor2, nn::{Forward, Linear, Mode}};
/// use rstorch::Rng;
/// let ctx = DeviceCtx::cpu().unwrap();
/// let mut layer = Linear::<3, 2, f32>::new(3, 2, &ctx, &mut Rng::seed(0)).unwrap();
/// let input = Tensor2::<1, 3, f64>::from_vec(vec![0.0; 3], [1, 3], &ctx).unwrap();
/// let _ = layer.forward(&input, Mode::EVAL);
/// ```
///
/// ```compile_fail
/// use rstorch::typed::{Cpu, DeviceCtx, Placement, Tensor2, nn::{Forward, Linear, Mode}};
/// use rstorch::{Device, Rng};
/// struct Auxiliary;
/// impl Placement for Auxiliary {}
/// let cpu = DeviceCtx::<Cpu>::cpu().unwrap();
/// let auxiliary = DeviceCtx::<Auxiliary>::bind(Device::Cpu).unwrap();
/// let mut layer = Linear::<3, 2>::new(3, 2, &cpu, &mut Rng::seed(0)).unwrap();
/// let input = Tensor2::<1, 3, f32, Auxiliary>::from_vec(vec![0.0; 3], [1, 3], &auxiliary).unwrap();
/// let _ = layer.forward(&input, Mode::EVAL);
/// ```
///
/// Runtime layers with observable parameter identities cannot be converted:
///
/// ```compile_fail
/// use rstorch::typed::{DeviceCtx, nn::Linear};
/// use rstorch::{Device, Rng};
/// let ctx = DeviceCtx::cpu().unwrap();
/// let runtime = rstorch::nn::Linear::new(3, 2, &Device::Cpu, &mut Rng::seed(0)).unwrap();
/// let _ = Linear::<3, 2>::from_runtime(runtime, &ctx);
/// ```
#[derive(rstorch::typed::nn::TypedModule)]
pub struct Linear<
    const IN: usize,
    const OUT: usize,
    E: FloatElement = f32,
    P: Placement = crate::typed::Cpu,
> {
    weight: TypedParam<Tensor2<OUT, IN, E, P>>,
    bias: Option<TypedParam<Tensor1<OUT, E, P>>>,
}

impl<const IN: usize, const OUT: usize, E: FloatElement, P: Placement> Linear<IN, OUT, E, P> {
    /// Initializes through runtime `Linear`, then validates and seals its state.
    pub fn new(
        in_features: usize,
        out_features: usize,
        ctx: &DeviceCtx<P>,
        rng: &mut Rng,
    ) -> Result<Self> {
        validate_marker(IN, in_features, "in_features", "typed::nn::Linear::new")?;
        validate_marker(OUT, out_features, "out_features", "typed::nn::Linear::new")?;
        let mut runtime = crate::nn::Linear::new(in_features, out_features, &ctx.device(), rng)?;
        if E::DTYPE != DType::F32 {
            crate::nn::to_dtype(&mut runtime, E::DTYPE)?;
        }
        Self::seal_fresh_runtime(runtime, ctx)
    }

    // This accepts only the unobserved temporary created in `new`. Exposing it
    // publicly would falsely imply that rebuilding TypedParams preserves an
    // existing runtime layer's GradKeys and live graph identity.
    fn seal_fresh_runtime(runtime: crate::nn::Linear, ctx: &DeviceCtx<P>) -> Result<Self> {
        let weight =
            Tensor2::<OUT, IN, E, P>::try_from_dynamic(runtime.weight().value().clone(), ctx)?;
        let bias = runtime
            .bias()
            .map(|bias| Tensor1::<OUT, E, P>::try_from_dynamic(bias.value().clone(), ctx))
            .transpose()?;
        let weight = TypedParam::new(weight)?;
        let bias = bias.map(TypedParam::new).transpose()?;
        Ok(Self { weight, bias })
    }

    /// Removes the bias without changing the weight or any RNG stream.
    #[must_use]
    pub fn without_bias(mut self) -> Self {
        self.bias = None;
        self
    }

    /// Returns the actual runtime input width.
    pub fn in_features(&self) -> usize {
        self.weight
            .value()
            .expect("sealed typed Linear weight")
            .dims()[1]
    }

    /// Returns the actual runtime output width.
    pub fn out_features(&self) -> usize {
        self.weight
            .value()
            .expect("sealed typed Linear weight")
            .dims()[0]
    }

    /// Returns the typed `[OUT, IN]` weight parameter.
    pub fn weight(&self) -> &TypedParam<Tensor2<OUT, IN, E, P>> {
        &self.weight
    }

    /// Returns the typed `[OUT]` bias parameter when present.
    pub fn bias(&self) -> Option<&TypedParam<Tensor1<OUT, E, P>>> {
        self.bias.as_ref()
    }
}

const fn assert_width(layer: usize, input: usize) {
    assert!(
        layer == DYN || input == DYN || layer == input,
        "typed Linear input width mismatch"
    );
}

fn validate_marker(marker: usize, actual: usize, name: &str, op: &'static str) -> Result<()> {
    if marker != DYN && marker != actual {
        return Err(Error::InvalidArg {
            op,
            msg: format!("{name} {actual} does not match static marker {marker}"),
        });
    }
    Ok(())
}

macro_rules! impl_linear_forward {
    ($name:ident, [$($leading:ident),+], $input:ident) => {
        impl<
            const IN: usize,
            const OUT: usize,
            const $input: usize,
            $(const $leading: usize,)+
            E: FloatElement + NumericElement,
            P: Placement,
        > Forward<$name<$($leading,)+ $input, E, P>> for Linear<IN, OUT, E, P>
        {
            type Output = $name<$($leading,)+ OUT, E, P>;

            fn forward(
                &mut self,
                input: &$name<$($leading,)+ $input, E, P>,
                mode: Mode,
            ) -> Result<Self::Output> {
                const { assert_width(IN, $input) };
                let weight = self.weight.get(mode)?;
                if input.dims().as_ref().last() != Some(&weight.dims()[1]) {
                    return Err(Error::ShapeMismatch {
                        op: "Linear::forward",
                        lhs: input.as_dynamic().shape().clone(),
                        rhs: weight.as_dynamic().shape().clone(),
                    });
                }
                let transposed = weight.transpose::<0, 1>()?;
                let output: Self::Output = input.matmul(&transposed)?;
                match &self.bias {
                    None => Ok(output),
                    Some(bias) => {
                        let dynamic = output.as_dynamic().add(bias.get(mode)?.as_dynamic())?;
                        checked_wrap(
                            dynamic,
                            Arc::clone(input.binding()),
                            "typed::nn::Linear::forward",
                        )
                    }
                }
            }
        }
    };
}

impl_linear_forward!(Tensor2, [D0], INPUT);
impl_linear_forward!(Tensor3, [D0, D1], INPUT);
impl_linear_forward!(Tensor4, [D0, D1, D2], INPUT);
impl_linear_forward!(Tensor5, [D0, D1, D2, D3], INPUT);
impl_linear_forward!(Tensor6, [D0, D1, D2, D3, D4], INPUT);
impl_linear_forward!(Tensor7, [D0, D1, D2, D3, D4, D5], INPUT);
impl_linear_forward!(Tensor8, [D0, D1, D2, D3, D4, D5, D6], INPUT);

impl<const IN: usize, const OUT: usize, E: FloatElement, P: Placement, Q: Placement> ToDevice<Q>
    for Linear<IN, OUT, E, P>
{
    type Output = Linear<IN, OUT, E, Q>;

    fn to_device(self, target: &DeviceCtx<Q>) -> Result<Self::Output> {
        Ok(Linear {
            weight: self.weight.to_device(target)?,
            bias: self.bias.map(|bias| bias.to_device(target)).transpose()?,
        })
    }
}

impl<const IN: usize, const OUT: usize, E: FloatElement, P: Placement, F: FloatElement> ToDType<F>
    for Linear<IN, OUT, E, P>
{
    type Output = Linear<IN, OUT, F, P>;

    fn to_dtype(self) -> Result<Self::Output> {
        Ok(Linear {
            weight: self.weight.to_dtype()?,
            bias: self.bias.map(TypedParam::to_dtype).transpose()?,
        })
    }
}

impl<const IN: usize, const OUT: usize, E: FloatElement, P: Placement> std::fmt::Debug
    for Linear<IN, OUT, E, P>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        crate::nn::debug_linear(
            f,
            self.in_features(),
            self.out_features(),
            self.bias.is_some(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Forward as RuntimeForward;
    use crate::typed::{Cpu, Tensor2, Tensor8};

    #[test]
    fn initialization_values_errors_and_rng_match_runtime() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let mut a = Rng::seed(12);
        let mut b = Rng::seed(12);
        let typed = Linear::<3, 2>::new(3, 2, &ctx, &mut a).unwrap();
        let runtime = crate::nn::Linear::new(3, 2, &ctx.device(), &mut b).unwrap();
        assert_eq!(
            typed.weight.value().unwrap().to_vec().unwrap(),
            runtime.weight().value().to_vec::<f32>().unwrap()
        );
        assert_eq!(a.state(), b.state());
        assert!(Linear::<4, 2>::new(3, 2, &ctx, &mut Rng::seed(0)).is_err());
    }

    #[test]
    fn ordinary_construction_values_state_paths_and_gradients_match_runtime() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let mut rng_a = Rng::seed(3);
        let mut rng_b = Rng::seed(3);
        let mut typed = Linear::<3, 2>::new(3, 2, &ctx, &mut rng_a).unwrap();
        let mut runtime = crate::nn::Linear::new(3, 2, &ctx.device(), &mut rng_b).unwrap();
        let input =
            Tensor2::<2, 3>::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], &ctx).unwrap();
        let typed_output = typed.forward(&input, Mode::TRAIN).unwrap();
        let runtime_output = runtime.forward(input.as_dynamic(), Mode::TRAIN).unwrap();
        assert_eq!(
            typed_output.to_vec().unwrap(),
            runtime_output.to_vec::<f32>().unwrap()
        );
        let grads = typed_output
            .as_dynamic()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let runtime_grads = runtime_output.sum_all().unwrap().backward().unwrap();
        assert_eq!(
            typed.weight.grad_from(&grads).unwrap().to_vec().unwrap(),
            runtime_grads
                .wrt(runtime.weight())
                .unwrap()
                .to_vec::<f32>()
                .unwrap()
        );
        assert_eq!(
            typed
                .bias
                .as_ref()
                .unwrap()
                .grad_from(&grads)
                .unwrap()
                .to_vec()
                .unwrap(),
            vec![2.0, 2.0]
        );
        assert_eq!(
            typed
                .bias
                .as_ref()
                .unwrap()
                .grad_from(&grads)
                .unwrap()
                .to_vec()
                .unwrap(),
            runtime_grads
                .wrt(runtime.bias().unwrap())
                .unwrap()
                .to_vec::<f32>()
                .unwrap()
        );
        assert_eq!(
            super::super::state_dict(&typed)
                .unwrap()
                .paths()
                .collect::<Vec<_>>(),
            vec!["bias", "weight"]
        );
    }

    #[test]
    fn dynamic_batch_can_change_and_dynamic_width_fails_at_runtime() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let mut layer = Linear::<3, 2>::new(3, 2, &ctx, &mut Rng::seed(1)).unwrap();
        let one = Tensor2::<DYN, 3>::from_vec(vec![1.0f32; 3], [1, 3], &ctx).unwrap();
        let five = Tensor2::<DYN, 3>::from_vec(vec![1.0f32; 15], [5, 3], &ctx).unwrap();
        assert_eq!(layer.forward(&one, Mode::EVAL).unwrap().dims(), [1, 2]);
        assert_eq!(layer.forward(&five, Mode::EVAL).unwrap().dims(), [5, 2]);

        let bad = Tensor2::<1, DYN>::from_vec(vec![1.0f32; 4], [1, 4], &ctx).unwrap();
        assert!(matches!(
            layer.forward(&bad, Mode::EVAL),
            Err(Error::ShapeMismatch {
                op: "Linear::forward",
                ..
            })
        ));
    }

    #[test]
    fn rank_two_and_rank_eight_preserve_all_leading_markers() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let mut layer = Linear::<1, 2>::new(1, 2, &ctx, &mut Rng::seed(5)).unwrap();
        let low = Tensor2::<3, 1>::from_vec(vec![1.0f32; 3], [3, 1], &ctx).unwrap();
        let _: Tensor2<3, 2> = layer.forward(&low, Mode::EVAL).unwrap();
        let high = Tensor8::<1, 1, 1, 1, 1, 1, 3, 1>::from_vec(
            vec![1.0f32; 3],
            [1, 1, 1, 1, 1, 1, 3, 1],
            &ctx,
        )
        .unwrap();
        let output: Tensor8<1, 1, 1, 1, 1, 1, 3, 2> = layer.forward(&high, Mode::EVAL).unwrap();
        assert_eq!(output.dims(), [1, 1, 1, 1, 1, 1, 3, 2]);
    }

    #[test]
    fn without_bias_sheds_the_bias_leaf_and_leaves_the_weight_intact() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let biased = Linear::<3, 2>::new(3, 2, &ctx, &mut Rng::seed(4)).unwrap();
        let weight = biased.weight.value().unwrap().to_vec().unwrap();
        let input =
            Tensor2::<2, 3>::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], &ctx).unwrap();

        let mut bare = biased.without_bias();
        assert!(bare.bias().is_none());
        assert_eq!(bare.weight.value().unwrap().to_vec().unwrap(), weight);
        let output = bare.forward(&input, Mode::EVAL).unwrap();
        assert_eq!(output.dims(), [2, 2]);
        assert_eq!(
            super::super::state_dict(&bare)
                .unwrap()
                .paths()
                .collect::<Vec<_>>(),
            vec!["weight"]
        );

        // The runtime sibling agrees on the values and on the shed path, and
        // dropping a zero-initialized bias consumes no randomness either side.
        let mut runtime = crate::nn::Linear::new(3, 2, &ctx.device(), &mut Rng::seed(4))
            .unwrap()
            .without_bias();
        assert_eq!(
            output.to_vec().unwrap(),
            runtime
                .forward(input.as_dynamic(), Mode::EVAL)
                .unwrap()
                .to_vec::<f32>()
                .unwrap()
        );
        assert_eq!(
            crate::nn::state_dict(&runtime).keys().collect::<Vec<_>>(),
            vec!["weight"]
        );
    }

    #[test]
    fn reported_geometry_and_debug_describe_the_actual_runtime_widths() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let layer = Linear::<3, 2>::new(3, 2, &ctx, &mut Rng::seed(6)).unwrap();
        assert_eq!((layer.in_features(), layer.out_features()), (3, 2));
        assert_eq!(format!("{layer:?}"), "Linear(3 -> 2, bias)");
        assert_eq!(
            format!("{:?}", layer.without_bias()),
            "Linear(3 -> 2, no bias)"
        );

        // `DYN` markers report the widths the constructor was given, not the
        // markers themselves.
        let dynamic = Linear::<DYN, DYN>::new(5, 7, &ctx, &mut Rng::seed(6)).unwrap();
        assert_eq!((dynamic.in_features(), dynamic.out_features()), (5, 7));
        assert_eq!(format!("{dynamic:?}"), "Linear(5 -> 7, bias)");
    }

    #[test]
    fn construction_rejects_widths_that_contradict_a_static_marker() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        assert_eq!(
            Linear::<4, 2>::new(3, 2, &ctx, &mut Rng::seed(0))
                .unwrap_err()
                .to_string(),
            "typed::nn::Linear::new: invalid argument: \
             in_features 3 does not match static marker 4"
        );
        assert_eq!(
            Linear::<3, 5>::new(3, 2, &ctx, &mut Rng::seed(0))
                .unwrap_err()
                .to_string(),
            "typed::nn::Linear::new: invalid argument: \
             out_features 2 does not match static marker 5"
        );
        // A `DYN` marker defers to the runtime layer's own non-zero rule, so the
        // message must be the runtime's verbatim.
        assert_eq!(
            Linear::<DYN, DYN>::new(0, 2, &ctx, &mut Rng::seed(0))
                .unwrap_err()
                .to_string(),
            crate::nn::Linear::new(0, 2, &ctx.device(), &mut Rng::seed(0))
                .unwrap_err()
                .to_string()
        );
    }

    #[test]
    fn mode_and_consuming_retyping_preserve_parameter_behavior() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let mut layer = Linear::<2, 1>::new(2, 1, &ctx, &mut Rng::seed(8)).unwrap();
        let input = Tensor2::<1, 2>::from_vec(vec![1.0f32, 2.0], [1, 2], &ctx).unwrap();
        assert!(
            layer
                .forward(&input, Mode::EVAL)
                .unwrap()
                .backward()
                .is_err()
        );
        let recorded = layer.forward(&input, Mode::EVAL.recorded()).unwrap();
        assert!(recorded.backward().is_ok());
        layer.weight.freeze();
        let layer = <Linear<2, 1> as ToDType<half::f16>>::to_dtype(layer).unwrap();
        assert!(layer.weight.is_frozen());
    }
}
