//! Minimal neural network building blocks.

pub mod loss;

use crate::backend::{Backend, Cpu};
use crate::dtype::FloatElement;
use crate::optim::OptimParameter;
use crate::rng::SmallRng;
use crate::shape::{D1, D2, Shape};
use crate::tensor::{Tensor, Tensor1D, Tensor2D};
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

#[cfg(test)]
mod tests {
    use super::loss::mse_loss;
    use super::{Linear, Module, Parameter};
    use crate::optim::{OptimParameter, SGD};
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
}
