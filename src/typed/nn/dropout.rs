use super::{Forward, Mode, Module, ToDType, ToDevice, TypedVisitor, TypedVisitorMut};
use crate::nn::Forward as RuntimeForward;
use crate::typed::{
    FloatElement, Placement, Tensor0, Tensor1, Tensor2, Tensor3, Tensor4, Tensor5, Tensor6,
    Tensor7, Tensor8,
};
use crate::{Result, Rng};

/// A shape-preserving typed adapter over runtime inverted dropout.
///
/// The adapter owns exactly the runtime layer's split RNG stream. It has no
/// parameters or tensor buffers, and reads training behavior independently of
/// recording through [`Mode`].
pub struct Dropout {
    runtime: crate::nn::Dropout,
}

impl Dropout {
    /// Creates dropout by splitting `rng` exactly once in the runtime layer.
    pub fn new(p: f64, rng: &mut Rng) -> Result<Self> {
        Ok(Self {
            runtime: crate::nn::Dropout::new(p, rng)?,
        })
    }

    /// Moves an existing runtime dropout layer without changing its RNG state.
    ///
    /// Unlike parameterized layers, dropout has no parameter identity or tensor
    /// state to reconstruct: the complete runtime object is retained privately.
    pub fn from_runtime(runtime: crate::nn::Dropout) -> Self {
        Self { runtime }
    }

    /// Returns the drop probability.
    pub fn p(&self) -> f64 {
        self.runtime.p()
    }
}

macro_rules! impl_dropout_forward {
    ($(($name:ident, $rank:literal, [$($dim:ident),*])),+ $(,)?) => {
        $(
            impl<$(const $dim: usize,)* E: FloatElement, P: Placement>
                Forward<$name<$($dim,)* E, P>> for Dropout
            {
                type Output = $name<$($dim,)* E, P>;

                fn forward(
                    &mut self,
                    input: &$name<$($dim,)* E, P>,
                    mode: Mode,
                ) -> Result<Self::Output> {
                    let output = self.runtime.forward(input.as_dynamic(), mode)?;
                    $name::try_from_dynamic(
                        output,
                        &super::linear::context_from(input, "typed::nn::Dropout::forward")?,
                    )
                }
            }
        )+
    };
}

crate::typed::typed_rank_table!(impl_dropout_forward);

impl Module for Dropout {
    fn visit(&self, _visitor: &mut TypedVisitor<'_>) {}
    fn visit_mut(&mut self, _visitor: &mut TypedVisitorMut<'_>) {}
}

impl<Q: Placement> ToDevice<Q> for Dropout {
    type Output = Self;

    fn to_device(self, _target: &crate::typed::DeviceCtx<Q>) -> Result<Self> {
        Ok(self)
    }
}

impl<F: FloatElement> ToDType<F> for Dropout {
    type Output = Self;

    fn to_dtype(self) -> Result<Self> {
        Ok(self)
    }
}

impl std::fmt::Debug for Dropout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.runtime.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed::{DeviceCtx, Tensor2};

    fn pair(seed: u64) -> (Dropout, crate::nn::Dropout) {
        let mut a = Rng::seed(seed);
        let mut b = Rng::seed(seed);
        (
            Dropout::new(0.5, &mut a).unwrap(),
            crate::nn::Dropout::new(0.5, &mut b).unwrap(),
        )
    }

    #[test]
    fn rng_and_all_mode_behaviors_match_runtime_exactly() {
        let ctx = DeviceCtx::cpu().unwrap();
        let input = Tensor2::<2, 8>::from_vec(vec![1.0f32; 16], [2, 8], &ctx).unwrap();
        let (mut typed, mut runtime) = pair(42);

        for mode in [
            Mode::EVAL,
            Mode::TRAIN,
            Mode::TRAIN.frozen(),
            Mode::EVAL.recorded(),
            Mode::TRAIN,
        ] {
            let actual = typed.forward(&input, mode).unwrap().to_vec().unwrap();
            let expected = runtime
                .forward(input.as_dynamic(), mode)
                .unwrap()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(actual, expected);
        }
        assert!(super::super::state_dict(&typed).unwrap().is_empty());
    }

    #[test]
    fn traced_input_keeps_runtime_mask_gradients() {
        let ctx = DeviceCtx::cpu().unwrap();
        let input = Tensor2::<1, 16>::from_vec(vec![1.0f32; 16], [1, 16], &ctx)
            .unwrap()
            .traced()
            .unwrap();
        let mut layer = Dropout::new(0.5, &mut Rng::seed(7)).unwrap();
        let output = layer.forward(&input, Mode::TRAIN.frozen()).unwrap();
        let grads = output.as_dynamic().sum_all().unwrap().backward().unwrap();
        let gradient = grads
            .wrt_input(input.as_dynamic())
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        for (value, grad) in output.to_vec().unwrap().into_iter().zip(gradient) {
            assert_eq!(grad, if value == 0.0 { 0.0 } else { 2.0 });
        }
    }

    #[test]
    fn from_runtime_moves_the_existing_rng_stream_without_reinitializing_it() {
        let ctx = DeviceCtx::cpu().unwrap();
        let input = Tensor2::<1, 32>::from_vec(vec![1.0f32; 32], [1, 32], &ctx).unwrap();
        let mut a = crate::nn::Dropout::new(0.5, &mut Rng::seed(19)).unwrap();
        let mut b = crate::nn::Dropout::new(0.5, &mut Rng::seed(19)).unwrap();

        let _ = a.forward(input.as_dynamic(), Mode::TRAIN).unwrap();
        let _ = b.forward(input.as_dynamic(), Mode::TRAIN).unwrap();
        let mut typed = Dropout::from_runtime(a);

        assert_eq!(
            typed
                .forward(&input, Mode::TRAIN)
                .unwrap()
                .to_vec()
                .unwrap(),
            b.forward(input.as_dynamic(), Mode::TRAIN)
                .unwrap()
                .to_vec::<f32>()
                .unwrap()
        );
    }
}
