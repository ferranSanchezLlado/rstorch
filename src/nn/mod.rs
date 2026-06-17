//! Minimal neural network building blocks.
//!
//! `Sequential` uses a compile-time typed chain for batched `Tensor2D` MLPs:
//! every `.add_module(...)` requires the previous output feature count to match the
//! next layer's input feature count. This intentionally targets the common MLP
//! path instead of a fully general rank-N module abstraction.

pub mod loss;

use crate::backend::{Backend, Cpu};
use crate::dtype::FloatElement;
use crate::optim::OptimParameter;
use crate::rng::SmallRng;
use crate::shape::{D1, D2, Shape};
use crate::tensor::{Tensor, Tensor1D, Tensor2D};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_PARAMETER_ID: AtomicUsize = AtomicUsize::new(1);

/// Trainable tensor wrapper.
pub struct Parameter<S, E = f32, B = Cpu>
where
    S: Shape,
    E: FloatElement,
    B: Backend<E>,
{
    id: usize,
    tensor: Tensor<S, E, B>,
}

impl<S, E, B> Parameter<S, E, B>
where
    S: Shape,
    E: FloatElement,
    B: Backend<E>,
{
    pub fn new(tensor: Tensor<S, E, B>) -> Self {
        Self {
            id: NEXT_PARAMETER_ID.fetch_add(1, Ordering::Relaxed),
            tensor: tensor.with_requires_grad(true),
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn tensor(&self) -> &Tensor<S, E, B> {
        &self.tensor
    }

    pub fn grad(&self) -> Option<Tensor<S, E, B>> {
        self.tensor.grad()
    }

    pub fn zero_grad(&self) {
        self.tensor.zero_grad();
    }
}

impl<S, E, B> OptimParameter<E, B> for Parameter<S, E, B>
where
    S: Shape,
    E: FloatElement,
    B: Backend<E>,
{
    fn param_id(&self) -> usize {
        self.id
    }

    fn values(&self) -> Vec<E> {
        self.tensor.to_vec()
    }

    fn grad_values(&self) -> Option<Vec<E>> {
        self.tensor.grad().map(|grad| grad.to_vec())
    }

    fn set_values(&mut self, values: Vec<E>) {
        self.tensor.replace_data_as_leaf(values, true);
    }

    fn zero_grad(&self) {
        Parameter::zero_grad(self);
    }
}

/// Lightweight module trait for mutable parameter collection.
pub trait Module<E, B>
where
    E: FloatElement,
    B: Backend<E>,
{
    fn parameters_mut(&mut self) -> Vec<&mut dyn OptimParameter<E, B>>;

    fn zero_grad(&mut self) {
        for parameter in self.parameters_mut() {
            parameter.zero_grad();
        }
    }
}

/// Typed batched-2D forward pass used by composable MLP layers.
pub trait Layer<const IN: usize, const OUT: usize, E, B>: Module<E, B>
where
    E: FloatElement,
    B: Backend<E>,
{
    fn forward<const BATCH: usize>(
        &self,
        input: &Tensor2D<BATCH, IN, E, B>,
    ) -> Tensor2D<BATCH, OUT, E, B>;
}

/// ReLU activation layer.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReLU;

/// Hyperbolic tangent activation layer.
#[derive(Clone, Copy, Debug, Default)]
pub struct Tanh;

/// Sigmoid activation layer.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sigmoid;

impl<E, B> Module<E, B> for ReLU
where
    E: FloatElement,
    B: Backend<E>,
{
    fn parameters_mut(&mut self) -> Vec<&mut dyn OptimParameter<E, B>> {
        Vec::new()
    }
}

impl<E, B> Module<E, B> for Tanh
where
    E: FloatElement,
    B: Backend<E>,
{
    fn parameters_mut(&mut self) -> Vec<&mut dyn OptimParameter<E, B>> {
        Vec::new()
    }
}

impl<E, B> Module<E, B> for Sigmoid
where
    E: FloatElement,
    B: Backend<E>,
{
    fn parameters_mut(&mut self) -> Vec<&mut dyn OptimParameter<E, B>> {
        Vec::new()
    }
}

impl<const N: usize, E, B> Layer<N, N, E, B> for ReLU
where
    E: FloatElement,
    B: Backend<E>,
{
    fn forward<const BATCH: usize>(
        &self,
        input: &Tensor2D<BATCH, N, E, B>,
    ) -> Tensor2D<BATCH, N, E, B> {
        input.relu()
    }
}

impl<const N: usize, E, B> Layer<N, N, E, B> for Tanh
where
    E: FloatElement,
    B: Backend<E>,
{
    fn forward<const BATCH: usize>(
        &self,
        input: &Tensor2D<BATCH, N, E, B>,
    ) -> Tensor2D<BATCH, N, E, B> {
        input.tanh()
    }
}

impl<const N: usize, E, B> Layer<N, N, E, B> for Sigmoid
where
    E: FloatElement,
    B: Backend<E>,
{
    fn forward<const BATCH: usize>(
        &self,
        input: &Tensor2D<BATCH, N, E, B>,
    ) -> Tensor2D<BATCH, N, E, B> {
        input.sigmoid()
    }
}

#[doc(hidden)]
pub struct Chain<Prev, Next, const MID: usize> {
    prev: Prev,
    next: Next,
}

impl<Prev, Next, const MID: usize, E, B> Module<E, B> for Chain<Prev, Next, MID>
where
    E: FloatElement,
    B: Backend<E>,
    Prev: Module<E, B>,
    Next: Module<E, B>,
{
    fn parameters_mut(&mut self) -> Vec<&mut dyn OptimParameter<E, B>> {
        let mut parameters = self.prev.parameters_mut();
        parameters.extend(self.next.parameters_mut());
        parameters
    }
}

impl<Prev, Next, const IN: usize, const MID: usize, const OUT: usize, E, B> Layer<IN, OUT, E, B>
    for Chain<Prev, Next, MID>
where
    E: FloatElement,
    B: Backend<E>,
    Prev: Layer<IN, MID, E, B>,
    Next: Layer<MID, OUT, E, B>,
{
    fn forward<const BATCH: usize>(
        &self,
        input: &Tensor2D<BATCH, IN, E, B>,
    ) -> Tensor2D<BATCH, OUT, E, B> {
        self.next.forward(&self.prev.forward(input))
    }
}

/// Builder returned by `Sequential::new()` before the first layer fixes shapes.
pub struct SequentialBuilder<E = f32, B = Cpu>
where
    E: FloatElement,
    B: Backend<E>,
{
    marker: PhantomData<(E, B)>,
}

/// Compile-time typed sequence of batched-2D layers.
pub struct Sequential<Stack = (), const IN: usize = 0, const OUT: usize = 0, E = f32, B = Cpu>
where
    E: FloatElement,
    B: Backend<E>,
{
    stack: Stack,
    marker: PhantomData<(E, B)>,
}

impl<E, B> Sequential<(), 0, 0, E, B>
where
    E: FloatElement,
    B: Backend<E>,
{
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> SequentialBuilder<E, B> {
        SequentialBuilder {
            marker: PhantomData,
        }
    }
}

impl<E, B> SequentialBuilder<E, B>
where
    E: FloatElement,
    B: Backend<E>,
{
    pub fn add_module<const IN: usize, const OUT: usize, L>(
        self,
        layer: L,
    ) -> Sequential<L, IN, OUT, E, B>
    where
        L: Layer<IN, OUT, E, B>,
    {
        Sequential {
            stack: layer,
            marker: PhantomData,
        }
    }
}

impl<Stack, const IN: usize, const OUT: usize, E, B> Sequential<Stack, IN, OUT, E, B>
where
    E: FloatElement,
    B: Backend<E>,
    Stack: Layer<IN, OUT, E, B>,
{
    pub fn add_module<const NEXT_OUT: usize, Next>(
        self,
        next: Next,
    ) -> Sequential<Chain<Stack, Next, OUT>, IN, NEXT_OUT, E, B>
    where
        Next: Layer<OUT, NEXT_OUT, E, B>,
    {
        Sequential {
            stack: Chain {
                prev: self.stack,
                next,
            },
            marker: PhantomData,
        }
    }

    pub fn forward<const BATCH: usize>(
        &self,
        input: &Tensor2D<BATCH, IN, E, B>,
    ) -> Tensor2D<BATCH, OUT, E, B> {
        self.stack.forward(input)
    }
}

impl<Stack, const IN: usize, const OUT: usize, E, B> Module<E, B>
    for Sequential<Stack, IN, OUT, E, B>
where
    E: FloatElement,
    B: Backend<E>,
    Stack: Layer<IN, OUT, E, B>,
{
    fn parameters_mut(&mut self) -> Vec<&mut dyn OptimParameter<E, B>> {
        self.stack.parameters_mut()
    }
}

/// Fully connected layer with compile-time input and output dimensions.
pub struct Linear<const IN: usize, const OUT: usize, E = f32, B = Cpu>
where
    E: FloatElement,
    B: Backend<E>,
{
    weight: Parameter<D2<IN, OUT>, E, B>,
    bias: Parameter<D1<OUT>, E, B>,
}

impl<const IN: usize, const OUT: usize, E, B> Linear<IN, OUT, E, B>
where
    E: FloatElement,
    B: Backend<E>,
{
    /// Returns a zero-initialized layer.
    ///
    /// This constructor is deterministic and useful for exact-value tests. Prefer
    /// `kaiming_uniform` or `xavier_uniform` for trainable models.
    pub fn new() -> Self {
        Self::zeros()
    }

    /// Returns a zero-initialized layer for exact-value tests.
    pub fn zeros() -> Self {
        Self {
            weight: Parameter::new(Tensor2D::<IN, OUT, E, B>::zeros()),
            bias: Parameter::new(Tensor1D::<OUT, E, B>::zeros()),
        }
    }

    /// Returns a layer with Kaiming-uniform weights and zero bias.
    pub fn kaiming_uniform(rng: &mut SmallRng) -> Self {
        assert!(IN > 0, "fan_in must be greater than zero");
        let bound = (6.0 / IN as f64).sqrt();
        Self::uniform_weights_zero_bias(rng, bound)
    }

    /// Returns a layer with Xavier-uniform weights and zero bias.
    pub fn xavier_uniform(rng: &mut SmallRng) -> Self {
        assert!(IN + OUT > 0, "fan_in + fan_out must be greater than zero");
        let bound = (6.0 / (IN + OUT) as f64).sqrt();
        Self::uniform_weights_zero_bias(rng, bound)
    }

    fn uniform_weights_zero_bias(rng: &mut SmallRng, bound: f64) -> Self {
        let weight = (0..IN * OUT)
            .map(|_| E::from_f64(rng.uniform_f64(-bound, bound)))
            .collect();

        Self {
            weight: Parameter::new(Tensor2D::<IN, OUT, E, B>::from_vec(weight).unwrap()),
            bias: Parameter::new(Tensor1D::<OUT, E, B>::zeros()),
        }
    }

    pub fn from_parameters(
        weight: Parameter<D2<IN, OUT>, E, B>,
        bias: Parameter<D1<OUT>, E, B>,
    ) -> Self {
        Self { weight, bias }
    }

    pub fn weight(&self) -> &Parameter<D2<IN, OUT>, E, B> {
        &self.weight
    }

    pub fn bias(&self) -> &Parameter<D1<OUT>, E, B> {
        &self.bias
    }

    pub fn forward<const BATCH: usize>(
        &self,
        x: &Tensor2D<BATCH, IN, E, B>,
    ) -> Tensor2D<BATCH, OUT, E, B> {
        x.matmul(self.weight.tensor()).add_row(self.bias.tensor())
    }
}

impl<const IN: usize, const OUT: usize, E, B> Default for Linear<IN, OUT, E, B>
where
    E: FloatElement,
    B: Backend<E>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const IN: usize, const OUT: usize, E, B> Module<E, B> for Linear<IN, OUT, E, B>
where
    E: FloatElement,
    B: Backend<E>,
{
    fn parameters_mut(&mut self) -> Vec<&mut dyn OptimParameter<E, B>> {
        vec![&mut self.weight, &mut self.bias]
    }
}

impl<const IN: usize, const OUT: usize, E, B> Layer<IN, OUT, E, B> for Linear<IN, OUT, E, B>
where
    E: FloatElement,
    B: Backend<E>,
{
    fn forward<const BATCH: usize>(
        &self,
        input: &Tensor2D<BATCH, IN, E, B>,
    ) -> Tensor2D<BATCH, OUT, E, B> {
        Linear::forward(self, input)
    }
}

#[cfg(test)]
mod tests {
    use super::loss::mse_loss;
    use super::{Linear, Module, Parameter, ReLU, Sequential, Sigmoid, Tanh};
    use crate::optim::{Adam, OptimParameter, SGD};
    use crate::tensor::{Tensor1D, Tensor2D};

    fn assert_loss_decreased(initial: f32, final_loss: f32) {
        assert!(
            final_loss < initial,
            "loss did not decrease: initial={initial}, final={final_loss}"
        );
    }

    #[test]
    fn parameter_new_enables_gradients() {
        let parameter = Parameter::new(Tensor1D::<2>::from_array([1.0, 2.0]));

        assert_ne!(parameter.id(), 0);
        assert!(parameter.tensor().requires_grad_enabled());
        assert!(parameter.tensor().is_leaf());
    }

    #[test]
    fn linear_forward_has_static_output_shape() {
        let layer = Linear::<3, 2>::new();
        let input = Tensor2D::<4, 3>::ones();

        let output: Tensor2D<4, 2> = layer.forward(&input);

        assert_eq!(output.shape(), &[4, 2]);
    }

    #[test]
    fn module_zero_grad_clears_parameter_gradients() {
        let mut layer = Linear::<1, 1>::new();
        let input = Tensor2D::<1, 1>::from_array([[2.0]]);

        layer.forward(&input).sum().backward();
        assert!(layer.weight().tensor().grad().is_some());
        assert!(layer.bias().tensor().grad().is_some());

        layer.zero_grad();

        assert!(layer.weight().tensor().grad().is_none());
        assert!(layer.bias().tensor().grad().is_none());
    }

    #[test]
    fn tiny_linear_regression_loss_decreases() {
        let mut layer = Linear::<1, 1>::new();
        let input = Tensor2D::<4, 1>::from_array([[0.0], [1.0], [2.0], [3.0]]);
        let target = Tensor2D::<4, 1>::from_array([[1.0], [3.0], [5.0], [7.0]]);
        let mut optimizer = SGD::new(0.1);

        let initial = mse_loss(&layer.forward(&input), &target).to_vec()[0];

        for _ in 0..80 {
            layer.zero_grad();
            let loss = mse_loss(&layer.forward(&input), &target);
            loss.backward();
            optimizer.step(layer.parameters_mut());
        }

        let final_loss = mse_loss(&layer.forward(&input), &target).to_vec()[0];
        assert_loss_decreased(initial, final_loss);
        assert!(final_loss < 0.1, "final loss too high: {final_loss}");
    }

    #[test]
    fn parameters_mut_can_feed_sgd() {
        let mut layer = Linear::<1, 1>::new();
        let input = Tensor2D::<1, 1>::from_array([[2.0]]);
        let mut optimizer = SGD::new(0.25);

        layer.forward(&input).sum().backward();
        optimizer.step(layer.parameters_mut());

        assert_eq!(layer.weight().tensor().to_vec(), vec![-0.5]);
        assert_eq!(layer.bias().tensor().to_vec(), vec![-0.25]);
    }

    #[test]
    fn sgd_keeps_updated_parameters_as_leaf_tensors() {
        let mut parameter = Parameter::new(Tensor1D::<1>::from_array([2.0]));
        let parameter_id = parameter.id();
        parameter.tensor().mul_scalar(3.0).sum().backward();

        let mut optimizer = SGD::new(0.5);
        optimizer.step(vec![
            &mut parameter as &mut dyn OptimParameter<f32, crate::backend::Cpu>,
        ]);

        assert_eq!(parameter.tensor().to_vec(), vec![0.5]);
        assert_eq!(parameter.grad().unwrap().to_vec(), vec![3.0]);
        assert_eq!(parameter.id(), parameter_id);
        assert!(parameter.tensor().requires_grad_enabled());
        assert!(parameter.tensor().is_leaf());
    }

    #[test]
    fn activation_layers_match_tensor_ops() {
        let input = Tensor2D::<2, 3>::from_array([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]]);

        assert_eq!(
            <ReLU as super::Layer<3, 3, f32, crate::backend::Cpu>>::forward(&ReLU, &input).to_vec(),
            input.relu().to_vec()
        );
        assert_eq!(
            <Tanh as super::Layer<3, 3, f32, crate::backend::Cpu>>::forward(&Tanh, &input).to_vec(),
            input.tanh().to_vec()
        );
        assert_eq!(
            <Sigmoid as super::Layer<3, 3, f32, crate::backend::Cpu>>::forward(&Sigmoid, &input)
                .to_vec(),
            input.sigmoid().to_vec()
        );
    }

    #[test]
    fn activation_layers_have_no_parameters() {
        let mut relu = ReLU;
        let mut tanh = Tanh;
        let mut sigmoid = Sigmoid;

        assert!(<ReLU as Module<f32, crate::backend::Cpu>>::parameters_mut(&mut relu).is_empty());
        assert!(<Tanh as Module<f32, crate::backend::Cpu>>::parameters_mut(&mut tanh).is_empty());
        assert!(
            <Sigmoid as Module<f32, crate::backend::Cpu>>::parameters_mut(&mut sigmoid).is_empty()
        );
    }

    #[test]
    fn sequential_forward_has_static_output_shape() {
        let model = Sequential::new()
            .add_module(Linear::<3, 4>::new())
            .add_module(ReLU)
            .add_module(Linear::<4, 2>::new());
        let input = Tensor2D::<5, 3>::ones();

        let output: Tensor2D<5, 2> = model.forward(&input);

        assert_eq!(output.shape(), &[5, 2]);
    }

    #[test]
    fn sequential_parameters_mut_collects_contained_parameters() {
        let layer1 = Linear::<3, 4>::new();
        let layer2 = Linear::<4, 2>::new();
        let expected_ids = vec![
            layer1.weight().id(),
            layer1.bias().id(),
            layer2.weight().id(),
            layer2.bias().id(),
        ];
        let mut model = Sequential::new()
            .add_module(layer1)
            .add_module(ReLU)
            .add_module(layer2);

        let actual_ids: Vec<_> = model
            .parameters_mut()
            .into_iter()
            .map(|parameter| parameter.param_id())
            .collect();

        assert_eq!(actual_ids, expected_ids);
    }

    #[test]
    fn sequential_zero_grad_clears_all_contained_parameters() {
        let mut model = Sequential::new()
            .add_module(Linear::<3, 4>::new())
            .add_module(ReLU)
            .add_module(Linear::<4, 2>::new());
        let input = Tensor2D::<5, 3>::ones();

        model.forward(&input).sum().backward();
        assert!(
            model
                .parameters_mut()
                .into_iter()
                .all(|parameter| parameter.grad_values().is_some())
        );

        model.zero_grad();

        assert!(
            model
                .parameters_mut()
                .into_iter()
                .all(|parameter| parameter.grad_values().is_none())
        );
    }

    #[test]
    fn sequential_tiny_mlp_loss_decreases_with_adam() {
        let mut rng = crate::rng::SmallRng::seed_from_u64(42);
        let mut model = Sequential::new()
            .add_module(Linear::<2, 4>::kaiming_uniform(&mut rng))
            .add_module(Tanh)
            .add_module(Linear::<4, 1>::xavier_uniform(&mut rng));
        let input = Tensor2D::<4, 2>::from_array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
        let target = Tensor2D::<4, 1>::from_array([[0.0], [1.0], [1.0], [2.0]]);
        let mut optimizer = Adam::new(0.05);

        let initial = mse_loss(&model.forward(&input), &target).to_vec()[0];

        for _ in 0..400 {
            model.zero_grad();
            let prediction = model.forward(&input);
            let loss = mse_loss(&prediction, &target);
            loss.backward();
            optimizer.step(model.parameters_mut());
        }

        let final_loss = mse_loss(&model.forward(&input), &target).to_vec()[0];
        assert_loss_decreased(initial, final_loss);
        assert!(final_loss < 0.1, "final loss too high: {final_loss}");
    }
}
