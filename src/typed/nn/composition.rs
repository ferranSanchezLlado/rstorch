//! Heterogeneous, compile-time checked module composition.

use super::{Forward, Mode, Module, ToDType, ToDevice, TypedVisitor, TypedVisitorMut};
use crate::Result;
use crate::typed::ops::WithPlacement;
use crate::typed::{
    DYN, DeviceCtx, FloatElement, NumericElement, Placement, Tensor0, Tensor1, Tensor2, Tensor3,
    Tensor4, Tensor5, Tensor6, Tensor7, Tensor8, TypedTensor,
};
use std::marker::PhantomData;

mod sealed {
    pub trait IntoSequential<Input> {}
    pub trait SequentialInputDType<F> {}
    pub trait SequentialLayer<Input> {}
}

/// Sealed dtype projection for a [`Sequential`]'s external input tensor.
///
/// Floating inputs are retyped to `F`; structural `i64` and `bool` inputs keep
/// their element type. Geometry and placement are always preserved.
#[doc(hidden)]
pub trait SequentialInputDType<F: FloatElement>: sealed::SequentialInputDType<F> {
    /// The external input type accepted after model conversion.
    type Output: TypedTensor;
}

macro_rules! impl_sequential_input_dtype {
    ($name:ident, [$($dim:ident),*]) => {
        macro_rules! floating {
            ($element:ty) => {
                impl<$(const $dim: usize,)* F: FloatElement, P: Placement>
                    sealed::SequentialInputDType<F> for $name<$($dim,)* $element, P>
                {}

                impl<$(const $dim: usize,)* F: FloatElement, P: Placement>
                    SequentialInputDType<F> for $name<$($dim,)* $element, P>
                {
                    type Output = $name<$($dim,)* F, P>;
                }
            };
        }

        floating!(f32);
        floating!(f64);
        floating!(half::f16);
        floating!(half::bf16);

        macro_rules! structural {
            ($element:ty) => {
                impl<$(const $dim: usize,)* F: FloatElement, P: Placement>
                    sealed::SequentialInputDType<F> for $name<$($dim,)* $element, P>
                {}

                impl<$(const $dim: usize,)* F: FloatElement, P: Placement>
                    SequentialInputDType<F> for $name<$($dim,)* $element, P>
                {
                    type Output = Self;
                }
            };
        }

        structural!(i64);
        structural!(bool);
    };
}

impl_sequential_input_dtype!(Tensor0, []);
impl_sequential_input_dtype!(Tensor1, [D0]);
impl_sequential_input_dtype!(Tensor2, [D0, D1]);
impl_sequential_input_dtype!(Tensor3, [D0, D1, D2]);
impl_sequential_input_dtype!(Tensor4, [D0, D1, D2, D3]);
impl_sequential_input_dtype!(Tensor5, [D0, D1, D2, D3, D4]);
impl_sequential_input_dtype!(Tensor6, [D0, D1, D2, D3, D4, D5]);
impl_sequential_input_dtype!(Tensor7, [D0, D1, D2, D3, D4, D5, D6]);
impl_sequential_input_dtype!(Tensor8, [D0, D1, D2, D3, D4, D5, D6, D7]);

/// A sealed built-in layer relationship accepted by [`Sequential::push`].
///
/// The trait is public because it appears in `push`'s signature, but downstream
/// crates cannot implement it. This keeps the associated check authoritative:
/// every implementation is audited alongside its built-in [`Forward`]
/// relation. Custom composition belongs in a typed module with an explicit
/// `Forward` implementation rather than an unchecked opt-in wrapper.
pub trait SequentialLayer<Input>: Forward<Input> + sealed::SequentialLayer<Input> {
    /// Performs any additional compile-time adjacency checks for this layer.
    #[doc(hidden)]
    const ADJACENCY: () = ();
}

fn enforce_adjacency(_: ()) {}

macro_rules! impl_shape_preserving_sequential_layer {
    ($($layer:ty),+ $(,)?) => {
        $(
            impl<Input> sealed::SequentialLayer<Input> for $layer where $layer: Forward<Input> {}
            impl<Input> SequentialLayer<Input> for $layer where $layer: Forward<Input> {}
        )+
    };
}

impl_shape_preserving_sequential_layer!(super::Relu, super::Gelu, super::Dropout);

const fn assert_linear_adjacency(layer: usize, input: usize) {
    assert!(
        layer == DYN || input == DYN || layer == input,
        "typed Sequential adjacent Linear width mismatch"
    );
}

macro_rules! impl_linear_sequential_layer {
    ($name:ident, [$($leading:ident),+], $input:ident) => {
        impl<
            const IN: usize,
            const OUT: usize,
            const $input: usize,
            $(const $leading: usize,)+
            E: FloatElement + NumericElement,
            P: Placement,
        > SequentialLayer<$name<$($leading,)+ $input, E, P>> for super::Linear<IN, OUT, E, P>
        {
            const ADJACENCY: () = assert_linear_adjacency(IN, $input);
        }

        impl<
            const IN: usize,
            const OUT: usize,
            const $input: usize,
            $(const $leading: usize,)+
            E: FloatElement + NumericElement,
            P: Placement,
        > sealed::SequentialLayer<$name<$($leading,)+ $input, E, P>>
            for super::Linear<IN, OUT, E, P>
        {}
    };
}

impl_linear_sequential_layer!(Tensor2, [D0], INPUT);
impl_linear_sequential_layer!(Tensor3, [D0, D1], INPUT);
impl_linear_sequential_layer!(Tensor4, [D0, D1, D2], INPUT);
impl_linear_sequential_layer!(Tensor5, [D0, D1, D2, D3], INPUT);
impl_linear_sequential_layer!(Tensor6, [D0, D1, D2, D3, D4], INPUT);
impl_linear_sequential_layer!(Tensor7, [D0, D1, D2, D3, D4, D5], INPUT);
impl_linear_sequential_layer!(Tensor8, [D0, D1, D2, D3, D4, D5, D6], INPUT);

const fn assert_suffix_adjacency(input: &[usize], suffix: &[usize]) {
    assert!(
        input.len() >= suffix.len(),
        "typed Sequential normalization suffix rank mismatch"
    );
    let offset = input.len() - suffix.len();
    let mut index = 0;
    while index < suffix.len() {
        let actual = input[offset + index];
        let expected = suffix[index];
        assert!(
            actual == DYN || expected == DYN || actual == expected,
            "typed Sequential normalization suffix mismatch"
        );
        index += 1;
    }
}

macro_rules! impl_suffix_sequential_layer {
    ($layer:ident) => {
        impl<I, S> sealed::SequentialLayer<I> for super::$layer<S>
        where
            I: TypedTensor<Elem = S::Elem, Placement = S::Placement>,
            S: TypedTensor,
            S::Elem: FloatElement,
        {
        }

        impl<I, S> SequentialLayer<I> for super::$layer<S>
        where
            I: TypedTensor<Elem = S::Elem, Placement = S::Placement>,
            S: TypedTensor,
            S::Elem: FloatElement,
        {
            const ADJACENCY: () = assert_suffix_adjacency(I::MARKERS, S::MARKERS);
        }
    };
}

impl_suffix_sequential_layer!(LayerNorm);
impl_suffix_sequential_layer!(RMSNorm);

const fn assert_channel_adjacency(layer: usize, input: usize) {
    assert!(
        layer == DYN || input == DYN || layer == input,
        "typed Sequential adjacent BatchNorm2d channel mismatch"
    );
}

impl<
    const N: usize,
    const IC: usize,
    const H: usize,
    const W: usize,
    const C: usize,
    E: FloatElement,
    P: Placement,
> sealed::SequentialLayer<Tensor4<N, IC, H, W, E, P>> for super::BatchNorm2d<C, E, P>
{
}

impl<
    const N: usize,
    const IC: usize,
    const H: usize,
    const W: usize,
    const C: usize,
    E: FloatElement,
    P: Placement,
> SequentialLayer<Tensor4<N, IC, H, W, E, P>> for super::BatchNorm2d<C, E, P>
{
    const ADJACENCY: () = assert_channel_adjacency(C, IC);
}

impl<const VOCAB: usize, const WIDTH: usize, E, P, Input> sealed::SequentialLayer<Input>
    for super::Embedding<VOCAB, WIDTH, E, P>
where
    E: FloatElement,
    P: Placement,
    Input: super::EmbeddingInput<WIDTH, E, P>,
{
}

impl<const VOCAB: usize, const WIDTH: usize, E, P, Input> SequentialLayer<Input>
    for super::Embedding<VOCAB, WIDTH, E, P>
where
    E: FloatElement,
    P: Placement,
    Input: super::EmbeddingInput<WIDTH, E, P>,
{
}

/// A heterogeneous chain whose input and every adjacent layer are checked.
///
/// The input type is named at construction. Each consuming [`push`](Self::push)
/// checks that the new layer accepts the output associated with the complete
/// preceding chain.
///
/// [`super::MultiHeadAttention`] is not a sequential layer because it has no
/// [`Forward`] implementation: self/cross attention must name mask and context
/// policy. Put it in an explicit typed module whose `Forward` binds those
/// choices, rather than silently discarding them in a sequential adapter.
///
/// ```
/// use rstorch::Rng;
/// use rstorch::typed::{DYN, DeviceCtx, Tensor2};
/// use rstorch::typed::nn::{Forward, Linear, Mode, Relu, Sequential};
///
/// let ctx = DeviceCtx::cpu().unwrap();
/// let mut net = Sequential::<Tensor2<DYN, 4>>::new()
///     .push(Linear::<4, 3>::new(4, 3, &ctx, &mut Rng::seed(1)).unwrap())
///     .push(Relu);
/// let input = Tensor2::from_vec(vec![1.0f32; 8], [2, 4], &ctx).unwrap();
/// let output: Tensor2<DYN, 3> = net.forward(&input, Mode::EVAL).unwrap();
/// assert_eq!(output.dims(), [2, 3]);
/// ```
///
/// The returned type is nameable for fields, aliases, and function signatures:
///
/// ```
/// use rstorch::typed::Tensor2;
/// use rstorch::typed::nn::{Linear, Sequential, SequentialCons};
/// type OneLayer = Sequential<
///     Tensor2<1, 2>,
///     SequentialCons<Sequential<Tensor2<1, 2>>, Linear<2, 3>>,
/// >;
/// struct Model { layers: OneLayer }
/// fn wrap(layers: OneLayer) -> Model { Model { layers } }
/// ```
///
/// An incompatible layer is rejected by the `push` that introduces it:
///
/// ```compile_fail
/// use rstorch::Rng;
/// use rstorch::typed::{DeviceCtx, Tensor2};
/// use rstorch::typed::nn::{Linear, Sequential};
///
/// let ctx = DeviceCtx::cpu().unwrap();
/// let net = Sequential::<Tensor2<8, 4>>::new()
///     .push(Linear::<4, 3>::new(4, 3, &ctx, &mut Rng::seed(1)).unwrap());
/// // The preceding output is Tensor2<8, 3>, not an input accepted by Linear<4, 2>.
/// let _ = net.push(Linear::<4, 2>::new(4, 2, &ctx, &mut Rng::seed(2)).unwrap());
/// ```
///
/// Normalization relations are checked at the same boundary:
///
/// ```compile_fail
/// use rstorch::typed::{DeviceCtx, Tensor1, Tensor2};
/// use rstorch::typed::nn::{LayerNorm, Sequential};
/// let ctx = DeviceCtx::cpu().unwrap();
/// let net = Sequential::<Tensor2<2, 3>>::new();
/// let _ = net.push(LayerNorm::<Tensor1<4>>::new([4], &ctx).unwrap());
/// ```
///
/// ```compile_fail
/// use rstorch::typed::{DeviceCtx, Tensor4};
/// use rstorch::typed::nn::{BatchNorm2d, Sequential};
/// let ctx = DeviceCtx::cpu().unwrap();
/// let net = Sequential::<Tensor4<1, 2, 4, 4>>::new();
/// let _ = net.push(BatchNorm2d::<3>::new(3, &ctx).unwrap());
/// ```
pub struct Sequential<Input, Layers = ()> {
    layers: Layers,
    len: usize,
    input: PhantomData<fn(&Input)>,
}

/// The append node used in [`Sequential`]'s public return type.
///
/// Its fields are private so construction always goes through `push` and its
/// adjacency check. State walks deliberately do not expose this representation.
pub struct SequentialCons<Previous, Layer> {
    previous: Previous,
    layer: Layer,
    index: usize,
}

/// Nameable return type for a one-layer tuple composition.
pub type Sequential1<Input, L1> = Sequential<Input, SequentialCons<Sequential<Input>, L1>>;

/// Nameable return type for a two-layer tuple composition.
pub type Sequential2<Input, L1, L2> = Sequential<Input, SequentialCons<Sequential1<Input, L1>, L2>>;

/// Nameable return type for a three-layer tuple composition.
pub type Sequential3<Input, L1, L2, L3> =
    Sequential<Input, SequentialCons<Sequential2<Input, L1, L2>, L3>>;

/// Nameable return type for a four-layer tuple composition.
pub type Sequential4<Input, L1, L2, L3, L4> =
    Sequential<Input, SequentialCons<Sequential3<Input, L1, L2, L3>, L4>>;

/// A sealed bounded tuple conversion into a typed [`Sequential`].
///
/// Implemented for tuple arities one through four. Every implementation calls
/// [`Sequential::push`] for each element, retaining the same immediate
/// adjacency checks as builder construction.
pub trait IntoSequential<Input>: sealed::IntoSequential<Input> {
    /// The nameable heterogeneous sequence type.
    type Output: Forward<Input> + Module;

    /// Consumes the tuple and checks each adjacent layer in order.
    fn into_sequential(self) -> Self::Output;
}

/// Builds a typed sequence from a tuple of one through four layers.
///
/// ```
/// use rstorch::Rng;
/// use rstorch::typed::{DeviceCtx, Tensor2};
/// use rstorch::typed::nn::{Forward, Linear, Mode, Relu, Sequential2, sequential};
/// let ctx = DeviceCtx::cpu().unwrap();
/// let layers = (
///     Linear::<2, 3>::new(2, 3, &ctx, &mut Rng::seed(1)).unwrap(),
///     Relu,
/// );
/// let mut net: Sequential2<Tensor2<1, 2>, Linear<2, 3>, Relu> =
///     sequential::<Tensor2<1, 2>, _>(layers);
/// let input = Tensor2::from_vec(vec![1.0f32, 2.0], [1, 2], &ctx).unwrap();
/// assert_eq!(net.forward(&input, Mode::EVAL).unwrap().dims(), [1, 3]);
/// ```
///
/// Tuple construction rejects the same incompatible adjacency as `push`:
///
/// ```compile_fail
/// use rstorch::Rng;
/// use rstorch::typed::{DeviceCtx, Tensor2};
/// use rstorch::typed::nn::{Linear, sequential};
/// let ctx = DeviceCtx::cpu().unwrap();
/// let _ = sequential::<Tensor2<1, 2>, _>((
///     Linear::<2, 3>::new(2, 3, &ctx, &mut Rng::seed(1)).unwrap(),
///     Linear::<4, 1>::new(4, 1, &ctx, &mut Rng::seed(2)).unwrap(),
/// ));
/// ```
pub fn sequential<Input, Layers>(layers: Layers) -> Layers::Output
where
    Layers: IntoSequential<Input>,
{
    layers.into_sequential()
}

impl<Input, L1> sealed::IntoSequential<Input> for (L1,) {}
impl<Input: Clone, L1> IntoSequential<Input> for (L1,)
where
    L1: SequentialLayer<Input> + Module,
{
    type Output = Sequential1<Input, L1>;

    fn into_sequential(self) -> Self::Output {
        Sequential::new().push(self.0)
    }
}

impl<Input, L1, L2> sealed::IntoSequential<Input> for (L1, L2) {}
impl<Input: Clone, L1, L2> IntoSequential<Input> for (L1, L2)
where
    L1: SequentialLayer<Input> + Module,
    L2: SequentialLayer<L1::Output> + Module,
{
    type Output = Sequential2<Input, L1, L2>;

    fn into_sequential(self) -> Self::Output {
        Sequential::new().push(self.0).push(self.1)
    }
}

impl<Input, L1, L2, L3> sealed::IntoSequential<Input> for (L1, L2, L3) {}
impl<Input: Clone, L1, L2, L3> IntoSequential<Input> for (L1, L2, L3)
where
    L1: SequentialLayer<Input> + Module,
    L2: SequentialLayer<L1::Output> + Module,
    L3: SequentialLayer<L2::Output> + Module,
{
    type Output = Sequential3<Input, L1, L2, L3>;

    fn into_sequential(self) -> Self::Output {
        Sequential::new().push(self.0).push(self.1).push(self.2)
    }
}

impl<Input, L1, L2, L3, L4> sealed::IntoSequential<Input> for (L1, L2, L3, L4) {}
impl<Input: Clone, L1, L2, L3, L4> IntoSequential<Input> for (L1, L2, L3, L4)
where
    L1: SequentialLayer<Input> + Module,
    L2: SequentialLayer<L1::Output> + Module,
    L3: SequentialLayer<L2::Output> + Module,
    L4: SequentialLayer<L3::Output> + Module,
{
    type Output = Sequential4<Input, L1, L2, L3, L4>;

    fn into_sequential(self) -> Self::Output {
        Sequential::new()
            .push(self.0)
            .push(self.1)
            .push(self.2)
            .push(self.3)
    }
}

impl<Input, Layers> sealed::SequentialLayer<Input> for Sequential<Input, Layers> where
    Self: Forward<Input>
{
}

impl<Input, Layers> SequentialLayer<Input> for Sequential<Input, Layers> where Self: Forward<Input> {}

impl<Input> Sequential<Input> {
    /// Creates an empty identity chain for `Input`.
    pub fn new() -> Self {
        Self {
            layers: (),
            len: 0,
            input: PhantomData,
        }
    }
}

impl<Input, Layers> Sequential<Input, Layers> {
    /// Appends a layer, checking its input against the preceding output now.
    #[must_use]
    pub fn push<Layer>(self, layer: Layer) -> Sequential<Input, SequentialCons<Self, Layer>>
    where
        Self: Forward<Input>,
        Layer: SequentialLayer<<Self as Forward<Input>>::Output> + Module,
    {
        enforce_adjacency(<Layer as SequentialLayer<<Self as Forward<Input>>::Output>>::ADJACENCY);
        let len = self.len + 1;
        let index = self.len;
        Sequential {
            layers: SequentialCons {
                previous: self,
                layer,
                index,
            },
            len,
            input: PhantomData,
        }
    }

    /// Returns the number of layers.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether this is the empty identity chain.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<Input: Clone> Forward<Input> for Sequential<Input> {
    type Output = Input;

    fn forward(&mut self, input: &Input, _mode: Mode) -> Result<Self::Output> {
        Ok(input.clone())
    }
}

impl<Input, Previous, Layer> Forward<Input> for Sequential<Input, SequentialCons<Previous, Layer>>
where
    Previous: Forward<Input>,
    Layer: Forward<Previous::Output>,
{
    type Output = Layer::Output;

    fn forward(&mut self, input: &Input, mode: Mode) -> Result<Self::Output> {
        let current = self.layers.previous.forward(input, mode)?;
        self.layers.layer.forward(&current, mode)
    }
}

impl<Input> Module for Sequential<Input> {
    fn visit(&self, _visitor: &mut TypedVisitor<'_>) {}

    fn visit_mut(&mut self, _visitor: &mut TypedVisitorMut<'_>) {}
}

impl<Input, Previous: Module, Layer: Module> Module
    for Sequential<Input, SequentialCons<Previous, Layer>>
{
    fn visit(&self, visitor: &mut TypedVisitor<'_>) {
        self.layers.previous.visit(visitor);
        visitor.module(&self.layers.index.to_string(), &self.layers.layer);
    }

    fn visit_mut(&mut self, visitor: &mut TypedVisitorMut<'_>) {
        let index = self.layers.index.to_string();
        self.layers.previous.visit_mut(visitor);
        visitor.module(&index, &mut self.layers.layer);
    }
}

impl<Input, Q> ToDevice<Q> for Sequential<Input>
where
    Input: TypedTensor + WithPlacement<Q>,
    Q: Placement,
{
    type Output = Sequential<<Input as WithPlacement<Q>>::Output>;

    fn to_device(self, _target: &DeviceCtx<Q>) -> Result<Self::Output> {
        Ok(Sequential::new())
    }
}

impl<Input, Previous, Layer, Q> ToDevice<Q> for Sequential<Input, SequentialCons<Previous, Layer>>
where
    Input: TypedTensor + WithPlacement<Q>,
    Q: Placement,
    Previous: Module + ToDevice<Q>,
    Layer: Module + ToDevice<Q>,
    Previous::Output: Forward<<Input as WithPlacement<Q>>::Output>,
    Layer::Output:
        Forward<<Previous::Output as Forward<<Input as WithPlacement<Q>>::Output>>::Output>,
{
    type Output = Sequential<
        <Input as WithPlacement<Q>>::Output,
        SequentialCons<Previous::Output, Layer::Output>,
    >;

    fn to_device(self, target: &DeviceCtx<Q>) -> Result<Self::Output> {
        Ok(Sequential {
            layers: SequentialCons {
                previous: self.layers.previous.to_device(target)?,
                layer: self.layers.layer.to_device(target)?,
                index: self.layers.index,
            },
            len: self.len,
            input: PhantomData,
        })
    }
}

impl<Input, F> ToDType<F> for Sequential<Input>
where
    Input: TypedTensor + SequentialInputDType<F>,
    F: FloatElement,
{
    type Output = Sequential<<Input as SequentialInputDType<F>>::Output>;

    fn to_dtype(self) -> Result<Self::Output> {
        Ok(Sequential::new())
    }
}

impl<Input, Previous, Layer, F> ToDType<F> for Sequential<Input, SequentialCons<Previous, Layer>>
where
    Input: TypedTensor + SequentialInputDType<F>,
    F: FloatElement,
    Previous: Module + ToDType<F>,
    Layer: Module + ToDType<F>,
    Previous::Output: Forward<<Input as SequentialInputDType<F>>::Output>,
    Layer::Output:
        Forward<<Previous::Output as Forward<<Input as SequentialInputDType<F>>::Output>>::Output>,
{
    type Output = Sequential<
        <Input as SequentialInputDType<F>>::Output,
        SequentialCons<Previous::Output, Layer::Output>,
    >;

    fn to_dtype(self) -> Result<Self::Output> {
        Ok(Sequential {
            layers: SequentialCons {
                previous: self.layers.previous.to_dtype()?,
                layer: self.layers.layer.to_dtype()?,
                index: self.layers.index,
            },
            len: self.len,
            input: PhantomData,
        })
    }
}

impl<Input, Layers> std::fmt::Debug for Sequential<Input, Layers> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sequential({} layers)", self.len)
    }
}

impl<Input> Default for Sequential<Input> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Forward as DynamicForward;
    use crate::typed::nn::{BatchNorm2d, Embedding, LayerNorm, Linear, RMSNorm};
    use crate::typed::{Cpu, DYN, Tensor1, Tensor2, Tensor4};
    use crate::{Rng, Tensor};

    type NamedLinear =
        Sequential<Tensor2<1, 2>, SequentialCons<Sequential<Tensor2<1, 2>>, Linear<2, 3>>>;

    struct NamedField {
        layers: NamedLinear,
    }

    fn named_linear(ctx: &DeviceCtx<Cpu>) -> NamedLinear {
        Sequential::new().push(Linear::new(2, 3, ctx, &mut Rng::seed(1)).unwrap())
    }

    #[test]
    fn dynamic_batch_output_and_flat_state_paths_match_dynamic_sequential() {
        let ctx = DeviceCtx::cpu().unwrap();
        let mut typed_rng = Rng::seed(17);
        let mut dynamic_rng = Rng::seed(17);
        let mut typed = Sequential::<Tensor2<DYN, 4>>::new()
            .push(Linear::<4, 3>::new(4, 3, &ctx, &mut typed_rng).unwrap())
            .push(super::super::Relu)
            .push(Linear::<3, 2>::new(3, 2, &ctx, &mut typed_rng).unwrap());
        let mut dynamic = crate::nn::Sequential::new()
            .push(crate::nn::Linear::new(4, 3, &ctx.device(), &mut dynamic_rng).unwrap())
            .push(crate::nn::Relu)
            .push(crate::nn::Linear::new(3, 2, &ctx.device(), &mut dynamic_rng).unwrap());

        assert_eq!(typed.len(), 3);
        assert_eq!(
            super::super::state_dict(&typed)
                .unwrap()
                .paths()
                .collect::<Vec<_>>(),
            crate::nn::state_dict(&dynamic)
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>()
        );
        let state = super::super::state_dict(&typed).unwrap();
        let mut replica_rng = Rng::seed(99);
        let mut replica = Sequential::<Tensor2<DYN, 4>>::new()
            .push(Linear::<4, 3>::new(4, 3, &ctx, &mut replica_rng).unwrap())
            .push(super::super::Relu)
            .push(Linear::<3, 2>::new(3, 2, &ctx, &mut replica_rng).unwrap());
        super::super::load_state_dict(&mut replica, &state).unwrap();

        for batch in [1, 5] {
            let values = vec![0.25f32; batch * 4];
            let input = Tensor2::<DYN, 4>::from_vec(values, [batch, 4], &ctx).unwrap();
            let output: Tensor2<DYN, 2> = typed.forward(&input, Mode::EVAL).unwrap();
            let replicated = replica.forward(&input, Mode::EVAL).unwrap();
            let expected = dynamic.forward(input.as_dynamic(), Mode::EVAL).unwrap();
            assert_eq!(output.dims(), [batch, 2]);
            assert_eq!(output.to_vec().unwrap(), expected.to_vec::<f32>().unwrap());
            assert_eq!(replicated.to_vec().unwrap(), output.to_vec().unwrap());
        }
    }

    #[test]
    fn gradients_and_mode_match_the_runtime_chain() {
        let ctx = DeviceCtx::cpu().unwrap();
        let mut typed_rng = Rng::seed(9);
        let mut dynamic_rng = Rng::seed(9);
        let mut typed = Sequential::<Tensor2<2, 3>>::new()
            .push(Linear::<3, 4>::new(3, 4, &ctx, &mut typed_rng).unwrap())
            .push(super::super::Gelu)
            .push(super::super::Dropout::new(0.25, &mut typed_rng).unwrap());
        let mut dynamic = crate::nn::Sequential::new()
            .push(crate::nn::Linear::new(3, 4, &ctx.device(), &mut dynamic_rng).unwrap())
            .push(crate::nn::Gelu)
            .push(crate::nn::Dropout::new(0.25, &mut dynamic_rng).unwrap());
        let input = Tensor2::from_vec(vec![0.5f32; 6], [2, 3], &ctx)
            .unwrap()
            .traced()
            .unwrap();
        let dynamic_input = Tensor::from_vec(vec![0.5f32; 6], [2, 3], &ctx.device())
            .unwrap()
            .traced()
            .unwrap();
        let output = typed.forward(&input, Mode::TRAIN).unwrap();
        let expected = dynamic.forward(&dynamic_input, Mode::TRAIN).unwrap();
        assert_eq!(output.to_vec().unwrap(), expected.to_vec::<f32>().unwrap());
        let grads = output.as_dynamic().sum_all().unwrap().backward().unwrap();
        let expected_grads = expected.sum_all().unwrap().backward().unwrap();
        assert_eq!(
            grads
                .wrt_input(input.as_dynamic())
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            expected_grads
                .wrt_input(&dynamic_input)
                .unwrap()
                .to_vec::<f32>()
                .unwrap()
        );
    }

    #[test]
    fn empty_chain_and_consuming_retyping_preserve_structure() {
        let ctx = DeviceCtx::cpu().unwrap();
        let empty = Sequential::<Tensor2<1, 2>>::new();
        assert!(empty.is_empty());

        let model = empty
            .push(Linear::<2, 3>::new(2, 3, &ctx, &mut Rng::seed(3)).unwrap())
            .push(super::super::Relu);
        let mut model = <_ as ToDevice<Cpu>>::to_device(model, &ctx).unwrap();
        let input = Tensor2::<1, 2>::from_vec(vec![1.0, 2.0], [1, 2], &ctx).unwrap();
        let output: Tensor2<1, 3> = model.forward(&input, Mode::EVAL).unwrap();
        assert_eq!(output.dims(), [1, 3]);
        assert_eq!(model.len(), 2);
        assert_eq!(format!("{model:?}"), "Sequential(2 layers)");
    }

    #[test]
    fn returned_cons_type_is_nameable_in_fields_and_signatures() {
        let ctx = DeviceCtx::cpu().unwrap();
        let mut model = NamedField {
            layers: named_linear(&ctx),
        };
        let input = Tensor2::from_vec(vec![1.0f32, 2.0], [1, 2], &ctx).unwrap();
        assert_eq!(
            model.layers.forward(&input, Mode::EVAL).unwrap().dims(),
            [1, 3]
        );
    }

    #[test]
    fn normalization_and_cnn_layers_compose_with_exact_paths_and_values() {
        let ctx = DeviceCtx::cpu().unwrap();
        let input =
            Tensor2::<2, 3>::from_vec(vec![1.0f32, 2.0, 4.0, 3.0, 5.0, 8.0], [2, 3], &ctx).unwrap();
        let mut chain = Sequential::<Tensor2<2, 3>>::new()
            .push(LayerNorm::<Tensor1<3>>::new([3], &ctx).unwrap())
            .push(RMSNorm::<Tensor1<3>>::new([3], &ctx).unwrap());
        let mut layer = LayerNorm::<Tensor1<3>>::new([3], &ctx).unwrap();
        let mut rms = RMSNorm::<Tensor1<3>>::new([3], &ctx).unwrap();
        let expected = rms
            .forward(&layer.forward(&input, Mode::EVAL).unwrap(), Mode::EVAL)
            .unwrap();
        let output = chain.forward(&input, Mode::EVAL).unwrap();
        assert_eq!(output.to_vec().unwrap(), expected.to_vec().unwrap());
        assert_eq!(
            super::super::state_dict(&chain)
                .unwrap()
                .paths()
                .collect::<Vec<_>>(),
            ["0.bias", "0.weight", "1.weight"]
        );

        let image = Tensor4::<1, 2, 2, 2>::from_vec(
            vec![-2.0f32, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [1, 2, 2, 2],
            &ctx,
        )
        .unwrap();
        let mut cnn = Sequential::<Tensor4<1, 2, 2, 2>>::new()
            .push(BatchNorm2d::<2>::new(2, &ctx).unwrap())
            .push(super::super::Relu);
        let mut batch_norm = BatchNorm2d::<2>::new(2, &ctx).unwrap();
        let expected = super::super::Relu
            .forward(
                &batch_norm.forward(&image, Mode::TRAIN).unwrap(),
                Mode::TRAIN,
            )
            .unwrap();
        let output = cnn.forward(&image, Mode::TRAIN).unwrap();
        assert_eq!(output.to_vec().unwrap(), expected.to_vec().unwrap());
        assert_eq!(
            super::super::state_dict(&cnn)
                .unwrap()
                .paths()
                .collect::<Vec<_>>(),
            ["0.bias", "0.running_mean", "0.running_var", "0.weight"]
        );
    }

    #[test]
    fn embedding_composes_into_linear_and_matches_runtime_values() {
        let ctx = DeviceCtx::cpu().unwrap();
        let mut typed_rng = Rng::seed(31);
        let mut dynamic_rng = Rng::seed(31);
        let mut typed = Sequential::<Tensor1<4, i64>>::new()
            .push(Embedding::<7, 3>::new(&ctx, &mut typed_rng).unwrap())
            .push(Linear::<3, 2>::new(3, 2, &ctx, &mut typed_rng).unwrap());
        let mut dynamic = crate::nn::Sequential::new()
            .push(crate::nn::Embedding::new(7, 3, &ctx.device(), &mut dynamic_rng).unwrap())
            .push(crate::nn::Linear::new(3, 2, &ctx.device(), &mut dynamic_rng).unwrap());
        let ids = Tensor1::from_vec(vec![0_i64, 2, 6, 1], [4], &ctx).unwrap();
        let output: Tensor2<4, 2> = typed.forward(&ids, Mode::TRAIN).unwrap();
        let expected = dynamic.forward(ids.as_dynamic(), Mode::TRAIN).unwrap();
        assert_eq!(output.to_vec().unwrap(), expected.to_vec::<f32>().unwrap());
        assert_eq!(
            super::super::state_dict(&typed)
                .unwrap()
                .paths()
                .collect::<Vec<_>>(),
            ["0.weight", "1.bias", "1.weight"]
        );
    }

    #[test]
    fn nested_sequence_flattens_only_each_numeric_boundary_and_matches_runtime() {
        let ctx = DeviceCtx::cpu().unwrap();
        let mut typed_rng = Rng::seed(44);
        let mut dynamic_rng = Rng::seed(44);
        let inner = Sequential::<Tensor2<1, 2>>::new()
            .push(Linear::<2, 3>::new(2, 3, &ctx, &mut typed_rng).unwrap())
            .push(super::super::Relu);
        let mut typed = Sequential::<Tensor2<1, 2>>::new()
            .push(inner)
            .push(Linear::<3, 1>::new(3, 1, &ctx, &mut typed_rng).unwrap());
        let dynamic_inner = crate::nn::Sequential::new()
            .push(crate::nn::Linear::new(2, 3, &ctx.device(), &mut dynamic_rng).unwrap())
            .push(crate::nn::Relu);
        let mut dynamic = crate::nn::Sequential::new()
            .push(dynamic_inner)
            .push(crate::nn::Linear::new(3, 1, &ctx.device(), &mut dynamic_rng).unwrap());
        let input = Tensor2::from_vec(vec![1.0f32, -2.0], [1, 2], &ctx).unwrap();
        assert_eq!(
            typed.forward(&input, Mode::EVAL).unwrap().to_vec().unwrap(),
            dynamic
                .forward(input.as_dynamic(), Mode::EVAL)
                .unwrap()
                .to_vec::<f32>()
                .unwrap()
        );
        assert_eq!(
            super::super::state_dict(&typed)
                .unwrap()
                .paths()
                .collect::<Vec<_>>(),
            ["0.0.bias", "0.0.weight", "1.bias", "1.weight"]
        );
    }

    #[test]
    fn consuming_device_and_dtype_conversion_retype_the_complete_chain() {
        let ctx = DeviceCtx::cpu().unwrap();
        let build = || {
            Sequential::<Tensor2<1, 2>>::new()
                .push(Linear::<2, 3>::new(2, 3, &ctx, &mut Rng::seed(52)).unwrap())
                .push(super::super::Relu)
        };
        let input = Tensor2::from_vec(vec![1.0f32, 2.0], [1, 2], &ctx).unwrap();
        let expected = build().forward(&input, Mode::EVAL).unwrap();
        let mut moved = <_ as ToDevice<Cpu>>::to_device(build(), &ctx).unwrap();
        assert_eq!(
            moved.forward(&input, Mode::EVAL).unwrap().to_vec().unwrap(),
            expected.to_vec().unwrap()
        );

        let converted = <_ as ToDType<half::f16>>::to_dtype(build()).unwrap();
        assert_eq!(
            super::super::state_dict(&converted)
                .unwrap()
                .paths()
                .collect::<Vec<_>>(),
            ["0.bias", "0.weight"]
        );
    }

    #[test]
    fn embedding_chain_dtype_conversion_keeps_structural_input_and_forwards() {
        let ctx = DeviceCtx::cpu().unwrap();
        let mut rng = Rng::seed(61);
        let model = Sequential::<Tensor1<4, i64>>::new()
            .push(Embedding::<8, 3>::new(&ctx, &mut rng).unwrap())
            .push(Linear::<3, 2>::new(3, 2, &ctx, &mut rng).unwrap());
        let mut converted = <_ as ToDType<half::f16>>::to_dtype(model).unwrap();
        let ids: Tensor1<4, i64> = Tensor1::from_vec(vec![0_i64, 3, 7, 1], [4], &ctx).unwrap();
        let output: Tensor2<4, 2, half::f16> = converted.forward(&ids, Mode::EVAL).unwrap();
        assert_eq!(output.dims(), [4, 2]);
        assert_eq!(output.as_dynamic().dtype(), crate::DType::F16);
        assert_eq!(
            super::super::state_dict(&converted)
                .unwrap()
                .paths()
                .collect::<Vec<_>>(),
            ["0.weight", "1.bias", "1.weight"]
        );
    }

    #[test]
    fn float_chain_dtype_conversion_retypes_external_input_and_output() {
        let ctx = DeviceCtx::cpu().unwrap();
        let model = Sequential::<Tensor2<1, 2>>::new()
            .push(Linear::<2, 3>::new(2, 3, &ctx, &mut Rng::seed(62)).unwrap())
            .push(super::super::Relu);
        let mut converted = <_ as ToDType<half::f16>>::to_dtype(model).unwrap();
        let input: Tensor2<1, 2, half::f16> =
            Tensor2::from_vec(vec![half::f16::ONE, half::f16::from_f32(2.0)], [1, 2], &ctx)
                .unwrap();
        let output: Tensor2<1, 3, half::f16> = converted.forward(&input, Mode::EVAL).unwrap();
        assert_eq!(output.as_dynamic().dtype(), crate::DType::F16);
        assert_eq!(output.dims(), [1, 3]);
    }

    #[test]
    fn bounded_tuple_helper_uses_nameable_types_and_flat_paths() {
        let ctx = DeviceCtx::cpu().unwrap();
        let layers = (
            Linear::<2, 3>::new(2, 3, &ctx, &mut Rng::seed(70)).unwrap(),
            super::super::Relu,
            super::super::Gelu,
            Linear::<3, 1>::new(3, 1, &ctx, &mut Rng::seed(71)).unwrap(),
        );
        let mut model: Sequential4<
            Tensor2<1, 2>,
            Linear<2, 3>,
            super::super::Relu,
            super::super::Gelu,
            Linear<3, 1>,
        > = sequential(layers);
        let input = Tensor2::from_vec(vec![1.0f32, -1.0], [1, 2], &ctx).unwrap();
        let output: Tensor2<1, 1> = model.forward(&input, Mode::EVAL).unwrap();
        assert_eq!(output.dims(), [1, 1]);
        assert_eq!(model.len(), 4);
        assert_eq!(
            super::super::state_dict(&model)
                .unwrap()
                .paths()
                .collect::<Vec<_>>(),
            ["0.bias", "0.weight", "3.bias", "3.weight"]
        );

        let one: Sequential1<Tensor2<1, 2>, super::super::Relu> = sequential((super::super::Relu,));
        assert_eq!(one.len(), 1);
    }

    /// The 1-, 2-, 3- and 4-arity `IntoSequential` impls are hand-written rather
    /// than macro-generated, and 3 was the one arity with no test at all.
    ///
    /// This is coverage that the 3-arity path works, NOT a guard against a
    /// mis-indexed push: the generic bounds already make that impossible.
    /// `push(self.1)` twice is a use-after-move, and reordering to
    /// `push(self.1).push(self.0)` fails to compile because `L2: SequentialLayer
    /// <L1::Output>` no longer holds (verified: `E0277` on that bound plus
    /// `E0308`). Two same-typed `Linear<2, 2>` layers with different seeds are
    /// used anyway so the asserted value depends on which layer ran first,
    /// leaving the check meaningful if those bounds are ever relaxed.
    #[test]
    fn the_three_tuple_helper_pushes_its_layers_in_order() {
        let ctx = DeviceCtx::cpu().unwrap();
        let mut first = Linear::<2, 2>::new(2, 2, &ctx, &mut Rng::seed(80)).unwrap();
        let mut second = Linear::<2, 2>::new(2, 2, &ctx, &mut Rng::seed(81)).unwrap();

        // The same two layers applied by hand, in the intended order.
        let input = Tensor2::<1, 2>::from_vec(vec![1.0f32, -1.0], [1, 2], &ctx).unwrap();
        let expected = second
            .forward(&first.forward(&input, Mode::EVAL).unwrap(), Mode::EVAL)
            .unwrap()
            .relu()
            .unwrap();

        let mut model: Sequential3<Tensor2<1, 2>, Linear<2, 2>, Linear<2, 2>, super::super::Relu> =
            sequential((first, second, super::super::Relu));

        assert_eq!(model.len(), 3);
        let output: Tensor2<1, 2> = model.forward(&input, Mode::EVAL).unwrap();
        assert_eq!(
            output.as_dynamic().to_vec::<f32>().unwrap(),
            expected.as_dynamic().to_vec::<f32>().unwrap(),
            "a swapped or repeated tuple index would change this value"
        );
        assert_eq!(
            super::super::state_dict(&model)
                .unwrap()
                .paths()
                .collect::<Vec<_>>(),
            ["0.bias", "0.weight", "1.bias", "1.weight"]
        );
    }

    /// `Default` is part of the public surface and had no test.
    #[test]
    fn default_sequential_is_empty() {
        let model: Sequential<Tensor2<1, 2>> = Sequential::default();
        assert_eq!(model.len(), 0);
        assert!(
            super::super::state_dict(&model)
                .unwrap()
                .paths()
                .next()
                .is_none()
        );
    }
}
