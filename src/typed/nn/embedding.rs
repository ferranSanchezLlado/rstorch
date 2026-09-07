use super::{Forward, Mode, ToDType, ToDevice, TypedParam};
use crate::typed::device::validate_binding;
use crate::typed::sealed::TypedTensor as SealedTypedTensor;
use crate::typed::tensor::checked_wrap;
use crate::typed::{DeviceCtx, FloatElement, Placement, TypedTensor};
use crate::{Error, Result, Rng};
use std::sync::Arc;

const fn assert_configuration(vocab: usize, width: usize) {
    assert!(
        vocab != crate::typed::DYN && vocab > 0,
        "typed embedding VOCAB must be static and non-zero"
    );
    assert!(
        width != crate::typed::DYN && width > 0,
        "typed embedding WIDTH must be static and non-zero"
    );
}

/// Rank-preserving embedding lookup output with one trailing width axis.
pub trait EmbeddingInput<const WIDTH: usize, E: FloatElement, P: Placement>:
    TypedTensor<Elem = i64, Placement = P>
{
    /// The index geometry followed by `WIDTH`.
    type Output: TypedTensor<Elem = E, Placement = P>;
}

macro_rules! embedding_inputs {
    ($(($input:ident, [$($dim:ident),*], $output:ident)),+ $(,)?) => {
        $(
            impl<$(const $dim: usize,)* const WIDTH: usize, E: FloatElement, P: Placement>
                EmbeddingInput<WIDTH, E, P> for crate::typed::$input<$($dim,)* i64, P>
            {
                type Output = crate::typed::$output<$($dim,)* WIDTH, E, P>;
            }
        )+
    };
}

embedding_inputs! {
    (Tensor0, [], Tensor1),
    (Tensor1, [D0], Tensor2),
    (Tensor2, [D0, D1], Tensor3),
    (Tensor3, [D0, D1, D2], Tensor4),
    (Tensor4, [D0, D1, D2, D3], Tensor5),
    (Tensor5, [D0, D1, D2, D3, D4], Tensor6),
    (Tensor6, [D0, D1, D2, D3, D4, D5], Tensor7),
    (Tensor7, [D0, D1, D2, D3, D4, D5, D6], Tensor8),
}

/// A typed learned lookup table with a static vocabulary and row width.
///
/// # Examples
///
/// ```
/// use rstorch::Rng;
/// use rstorch::typed::{DeviceCtx, Tensor1};
/// use rstorch::typed::nn::{Embedding, Mode};
///
/// # fn main() -> rstorch::Result<()> {
/// let ctx = DeviceCtx::cpu()?;
/// let embedding = Embedding::<100, 8>::new(&ctx, &mut Rng::seed(0))?;
/// let ids = Tensor1::<3, i64>::from_vec(vec![5, 0, 99], [3], &ctx)?;
/// let rows = embedding.lookup(&ids, Mode::EVAL)?;
/// assert_eq!(rows.dims(), [3, 8]);
/// # Ok(())
/// # }
/// ```
#[derive(rstorch::typed::nn::TypedModule)]
pub struct Embedding<
    const VOCAB: usize,
    const WIDTH: usize,
    E: FloatElement = f32,
    P: Placement = crate::typed::Cpu,
> {
    weight: TypedParam<crate::typed::Tensor2<VOCAB, WIDTH, E, P>>,
}

impl<const VOCAB: usize, const WIDTH: usize, E: FloatElement, P: Placement>
    Embedding<VOCAB, WIDTH, E, P>
{
    /// Initializes the runtime embedding distribution and seals its typed state.
    ///
    /// `VOCAB` and `WIDTH` configure persistent state and must both be static
    /// and non-zero.
    ///
    /// ```compile_fail
    /// use rstorch::Rng;
    /// use rstorch::typed::{DYN, DeviceCtx};
    /// use rstorch::typed::nn::Embedding;
    /// let ctx = DeviceCtx::cpu().unwrap();
    /// let _ = Embedding::<DYN, 3>::new(&ctx, &mut Rng::seed(1));
    /// ```
    ///
    /// ```compile_fail
    /// use rstorch::Rng;
    /// use rstorch::typed::DeviceCtx;
    /// use rstorch::typed::nn::Embedding;
    /// let ctx = DeviceCtx::cpu().unwrap();
    /// let _ = Embedding::<4, 0>::new(&ctx, &mut Rng::seed(1));
    /// ```
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArg`] if `ctx`'s binding is not the canonical one for
    /// `P`; otherwise propagates the runtime embedding's own initialization
    /// error and the cast/conversion into the typed table.
    pub fn new(ctx: &DeviceCtx<P>, rng: &mut Rng) -> Result<Self> {
        const { assert_configuration(VOCAB, WIDTH) };
        validate_binding::<P>(ctx.binding(), "typed::nn::Embedding::new")?;
        let runtime = crate::nn::Embedding::new(VOCAB, WIDTH, &ctx.device(), rng)?;
        let weight = runtime.weight().value().to_dtype(E::DTYPE)?;
        Self::from_weight(crate::typed::Tensor2::try_from_dynamic(weight, ctx)?)
    }

    /// Creates an embedding from an exact typed table.
    ///
    /// # Examples
    ///
    /// ```
    /// use rstorch::typed::{DeviceCtx, Tensor2};
    /// use rstorch::typed::nn::Embedding;
    ///
    /// # fn main() -> rstorch::Result<()> {
    /// let ctx = DeviceCtx::cpu()?;
    /// let table = Tensor2::<100, 8>::from_vec(vec![0.0f32; 800], [100, 8], &ctx)?;
    /// let embedding = Embedding::<100, 8>::from_weight(table)?;
    /// assert_eq!(embedding.weight().value()?.dims(), [100, 8]);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// As [`TypedParam::new`]: [`Error::InvalidArg`] if `weight`'s binding is
    /// not the canonical one for `P`.
    pub fn from_weight(weight: crate::typed::Tensor2<VOCAB, WIDTH, E, P>) -> Result<Self> {
        const { assert_configuration(VOCAB, WIDTH) };
        Ok(Self {
            weight: TypedParam::new(weight)?,
        })
    }

    /// Returns the typed table parameter.
    pub fn weight(&self) -> &TypedParam<crate::typed::Tensor2<VOCAB, WIDTH, E, P>> {
        &self.weight
    }

    /// Selects rows and appends the configured width to ranks zero through seven.
    ///
    /// Index dtype, placement, and the rank ceiling are type-level contracts.
    ///
    /// ```compile_fail
    /// use rstorch::Rng;
    /// use rstorch::typed::{DeviceCtx, Tensor1};
    /// use rstorch::typed::nn::{Embedding, Mode};
    /// let ctx = DeviceCtx::cpu().unwrap();
    /// let embedding = Embedding::<4, 3>::new(&ctx, &mut Rng::seed(1)).unwrap();
    /// let float_ids = Tensor1::<1>::from_vec(vec![1.0], [1], &ctx).unwrap();
    /// let _ = embedding.lookup(&float_ids, Mode::EVAL);
    /// ```
    ///
    /// ```compile_fail
    /// use rstorch::{Device, Rng};
    /// use rstorch::typed::{DeviceCtx, Placement, Tensor1};
    /// use rstorch::typed::nn::{Embedding, Mode};
    /// struct TokenDevice;
    /// impl Placement for TokenDevice {}
    /// let ctx = DeviceCtx::cpu().unwrap();
    /// let token_ctx = DeviceCtx::<TokenDevice>::bind(Device::Cpu).unwrap();
    /// let embedding = Embedding::<4, 3>::new(&ctx, &mut Rng::seed(1)).unwrap();
    /// let ids = Tensor1::<1, i64, TokenDevice>::from_vec(vec![1], [1], &token_ctx).unwrap();
    /// let _ = embedding.lookup(&ids, Mode::EVAL);
    /// ```
    ///
    /// ```compile_fail
    /// use rstorch::Rng;
    /// use rstorch::typed::{DeviceCtx, Tensor8};
    /// use rstorch::typed::nn::{Embedding, Mode};
    /// let ctx = DeviceCtx::cpu().unwrap();
    /// let embedding = Embedding::<4, 3>::new(&ctx, &mut Rng::seed(1)).unwrap();
    /// let ids = Tensor8::<1, 1, 1, 1, 1, 1, 1, 1, i64>::from_vec(
    ///     vec![1], [1, 1, 1, 1, 1, 1, 1, 1], &ctx,
    /// ).unwrap();
    /// let _ = embedding.lookup(&ids, Mode::EVAL);
    /// ```
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArg`] if `ids`' binding is not the canonical one for
    /// `P`, or if `ids` was not built through the same [`DeviceCtx`] as this
    /// embedding's weight; [`Error::IndexOutOfBounds`] if an index is outside
    /// `[0, VOCAB)`.
    pub fn lookup<I>(&self, ids: &I, mode: Mode) -> Result<I::Output>
    where
        I: EmbeddingInput<WIDTH, E, P>,
    {
        const { assert_configuration(VOCAB, WIDTH) };
        validate_binding::<P>(ids.binding(), "typed::nn::Embedding::lookup")?;
        let table = self.weight.get(mode)?;
        if !Arc::ptr_eq(table.binding(), ids.binding()) {
            return Err(Error::InvalidArg {
                op: "typed::nn::Embedding::lookup",
                msg: "indices and weight do not share the canonical placement binding".into(),
            });
        }

        // This is the runtime Embedding lookup path, retaining the typed
        // parameter leaf rather than constructing a detached runtime Param.
        let flat = ids.dynamic().reshape([ids.dynamic().num_elements()])?;
        let rows = table.dynamic().index_select(0, &flat)?;
        let mut dims = ids.dynamic().dims().to_vec();
        dims.push(WIDTH);
        checked_wrap::<I::Output>(
            rows.reshape(dims)?,
            Arc::clone(table.binding()),
            "typed::nn::Embedding::lookup",
        )
    }
}

impl<
    const VOCAB: usize,
    const WIDTH: usize,
    E: FloatElement,
    P: Placement,
    I: EmbeddingInput<WIDTH, E, P>,
> Forward<I> for Embedding<VOCAB, WIDTH, E, P>
{
    type Output = I::Output;

    fn forward(&mut self, input: &I, mode: Mode) -> Result<Self::Output> {
        self.lookup(input, mode)
    }
}

impl<const VOCAB: usize, const WIDTH: usize, E: FloatElement, P: Placement, Q: Placement>
    ToDevice<Q> for Embedding<VOCAB, WIDTH, E, P>
{
    type Output = Embedding<VOCAB, WIDTH, E, Q>;

    fn to_device(self, target: &DeviceCtx<Q>) -> Result<Self::Output> {
        Ok(Embedding {
            weight: self.weight.to_device(target)?,
        })
    }
}

impl<const VOCAB: usize, const WIDTH: usize, E: FloatElement, P: Placement, F: FloatElement>
    ToDType<F> for Embedding<VOCAB, WIDTH, E, P>
{
    type Output = Embedding<VOCAB, WIDTH, F, P>;

    fn to_dtype(self) -> Result<Self::Output> {
        Ok(Embedding {
            weight: self.weight.to_dtype()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed::{Cpu, DYN, Tensor0, Tensor2, Tensor7};

    fn table(ctx: &DeviceCtx<Cpu>) -> crate::typed::Tensor2<4, 3> {
        crate::typed::Tensor2::from_vec((0..12).map(|value| value as f32).collect(), [4, 3], ctx)
            .unwrap()
    }

    #[test]
    fn lookup_covers_rank_edges_state_mode_and_gradients() {
        let ctx = DeviceCtx::cpu().unwrap();
        let embedding = Embedding::from_weight(table(&ctx)).unwrap();
        let scalar = Tensor0::<i64>::from_vec(vec![2], [], &ctx).unwrap();
        assert_eq!(embedding.lookup(&scalar, Mode::EVAL).unwrap().dims(), [3]);

        let deep =
            Tensor7::<1, 1, 1, 1, 1, 1, 2, i64>::from_vec(vec![1, 1], [1, 1, 1, 1, 1, 1, 2], &ctx)
                .unwrap();
        assert_eq!(
            embedding.lookup(&deep, Mode::EVAL).unwrap().dims(),
            [1, 1, 1, 1, 1, 1, 2, 3]
        );

        let ids = Tensor2::<DYN, DYN, i64>::from_vec(vec![1, 1], [1, 2], &ctx).unwrap();
        let loss = embedding
            .lookup(&ids, Mode::TRAIN)
            .unwrap()
            .as_dynamic()
            .sum_all()
            .unwrap();
        let grad = embedding
            .weight
            .grad_from(&loss.backward().unwrap())
            .unwrap();
        assert_eq!(grad.to_vec().unwrap()[3..6], [2.0, 2.0, 2.0]);
        assert!(
            embedding
                .lookup(&ids, Mode::EVAL)
                .unwrap()
                .as_dynamic()
                .backward()
                .is_err()
        );
        assert_eq!(
            super::super::state_dict(&embedding)
                .unwrap()
                .paths()
                .collect::<Vec<_>>(),
            ["weight"]
        );
    }

    #[test]
    fn seeded_initialization_matches_the_runtime_embedding() {
        let ctx = DeviceCtx::cpu().unwrap();
        let typed = Embedding::<4, 3>::new(&ctx, &mut Rng::seed(19)).unwrap();
        let runtime = crate::nn::Embedding::new(4, 3, &ctx.device(), &mut Rng::seed(19)).unwrap();
        assert_eq!(
            typed.weight().value().unwrap().to_vec().unwrap(),
            runtime.weight().value().to_vec::<f32>().unwrap()
        );
    }

    #[test]
    fn dynamic_index_bounds_and_reduced_precision_delegate_to_runtime() {
        let ctx = DeviceCtx::cpu().unwrap();
        let weight = crate::typed::Tensor2::<4, 3, f64>::from_vec(
            (0..12).map(|value| value as f64).collect(),
            [4, 3],
            &ctx,
        )
        .unwrap();
        let embedding = Embedding::from_weight(weight).unwrap();
        let bad = crate::typed::Tensor1::<DYN, i64>::from_vec(vec![4], [1], &ctx).unwrap();
        assert!(matches!(
            embedding.lookup(&bad, Mode::EVAL),
            Err(Error::IndexOutOfBounds { .. })
        ));
        let good = crate::typed::Tensor1::<DYN, i64>::from_vec(vec![0], [1], &ctx).unwrap();
        assert_eq!(
            embedding
                .lookup(&good, Mode::EVAL)
                .unwrap()
                .as_dynamic()
                .dtype(),
            E64
        );
    }

    #[test]
    fn consuming_movement_and_retyping_preserve_embedding_state() {
        struct Main;
        impl Placement for Main {}

        let source = DeviceCtx::<Main>::bind(crate::Device::Cpu).unwrap();
        let target = DeviceCtx::cpu().unwrap();
        let weight = crate::typed::Tensor2::<4, 3, f32, Main>::from_vec(
            (0..12).map(|value| value as f32).collect(),
            [4, 3],
            &source,
        )
        .unwrap();
        let embedding = Embedding::from_weight(weight).unwrap();
        let embedding = ToDevice::to_device(embedding, &target).unwrap();
        let embedding: Embedding<4, 3> = ToDType::to_dtype(embedding).unwrap();
        let ids = crate::typed::Tensor1::<2, i64>::from_vec(vec![1, 3], [2], &target).unwrap();
        assert_eq!(
            embedding
                .lookup(&ids, Mode::EVAL)
                .unwrap()
                .to_vec()
                .unwrap(),
            vec![3.0, 4.0, 5.0, 9.0, 10.0, 11.0]
        );
        assert_eq!(
            super::super::state_dict(&embedding)
                .unwrap()
                .paths()
                .collect::<Vec<_>>(),
            ["weight"]
        );
    }

    const E64: crate::DType = crate::DType::F64;
}
