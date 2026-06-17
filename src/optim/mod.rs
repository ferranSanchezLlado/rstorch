//! Minimal optimizers.

use crate::backend::Backend;
use crate::dtype::FloatElement;
use crate::tensor::autograd::no_grad;
use std::collections::HashMap;

/// Type-erased parameter interface used by optimizers.
pub trait OptimParameter<E, B>
where
    E: FloatElement,
    B: Backend<E>,
{
    fn param_id(&self) -> usize;
    fn values(&self) -> Vec<E>;
    fn grad_values(&self) -> Option<Vec<E>>;
    fn set_values(&mut self, values: Vec<E>);
    fn zero_grad(&self);
}

/// Stochastic gradient descent without momentum.
pub struct SGD<E = f32>
where
    E: FloatElement,
{
    lr: E,
}

impl<E> SGD<E>
where
    E: FloatElement,
{
    pub fn new(lr: E) -> Self {
        Self { lr }
    }

    pub fn step<'a, B, P>(&mut self, params: P)
    where
        B: Backend<E> + 'a,
        E: 'a,
        P: IntoIterator<Item = &'a mut dyn OptimParameter<E, B>>,
    {
        let _guard = no_grad();
        for parameter in params {
            let Some(grad) = parameter.grad_values() else {
                continue;
            };

            let updated = parameter
                .values()
                .into_iter()
                .zip(grad)
                .map(|(value, grad)| value - self.lr * grad)
                .collect();
            parameter.set_values(updated);
        }
    }
}

/// Stochastic gradient descent with momentum.
pub struct SGDMomentum<E = f32>
where
    E: FloatElement,
{
    lr: E,
    momentum: E,
    velocity: HashMap<usize, Vec<E>>,
}

impl<E> SGDMomentum<E>
where
    E: FloatElement,
{
    pub fn new(lr: E, momentum: E) -> Self {
        Self {
            lr,
            momentum,
            velocity: HashMap::new(),
        }
    }

    pub fn step<'a, B, P>(&mut self, params: P)
    where
        B: Backend<E> + 'a,
        E: 'a,
        P: IntoIterator<Item = &'a mut dyn OptimParameter<E, B>>,
    {
        let _guard = no_grad();
        for parameter in params {
            let Some(grad) = parameter.grad_values() else {
                continue;
            };

            let values = parameter.values();
            assert_eq!(
                values.len(),
                grad.len(),
                "gradient length must match parameter length"
            );

            let velocity = self
                .velocity
                .entry(parameter.param_id())
                .or_insert_with(|| vec![E::zero(); values.len()]);
            if velocity.len() != values.len() {
                *velocity = vec![E::zero(); values.len()];
            }

            let updated = values
                .into_iter()
                .zip(grad)
                .zip(velocity.iter_mut())
                .map(|((value, grad), velocity)| {
                    *velocity = self.momentum * *velocity + grad;
                    value - self.lr * *velocity
                })
                .collect();
            parameter.set_values(updated);
        }
    }
}

struct AdamState<E>
where
    E: FloatElement,
{
    m: Vec<E>,
    v: Vec<E>,
}

/// Adam optimizer with bias correction.
pub struct Adam<E = f32>
where
    E: FloatElement,
{
    lr: E,
    beta1: E,
    beta2: E,
    eps: E,
    step: usize,
    state: HashMap<usize, AdamState<E>>,
}

impl<E> Adam<E>
where
    E: FloatElement,
{
    pub fn new(lr: E) -> Self {
        Self {
            lr,
            beta1: E::from_f64(0.9),
            beta2: E::from_f64(0.999),
            eps: E::from_f64(1e-8),
            step: 0,
            state: HashMap::new(),
        }
    }

    pub fn with_betas(mut self, beta1: E, beta2: E) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn with_eps(mut self, eps: E) -> Self {
        self.eps = eps;
        self
    }

    pub fn step<'a, B, P>(&mut self, params: P)
    where
        B: Backend<E> + 'a,
        E: 'a,
        P: IntoIterator<Item = &'a mut dyn OptimParameter<E, B>>,
    {
        let _guard = no_grad();
        let mut step = self.step;
        let mut advanced_step = false;

        for parameter in params {
            let Some(grad) = parameter.grad_values() else {
                continue;
            };

            if !advanced_step {
                step += 1;
                advanced_step = true;
            }

            let values = parameter.values();
            assert_eq!(
                values.len(),
                grad.len(),
                "gradient length must match parameter length"
            );

            let state = self
                .state
                .entry(parameter.param_id())
                .or_insert_with(|| AdamState {
                    m: vec![E::zero(); values.len()],
                    v: vec![E::zero(); values.len()],
                });
            if state.m.len() != values.len() || state.v.len() != values.len() {
                *state = AdamState {
                    m: vec![E::zero(); values.len()],
                    v: vec![E::zero(); values.len()],
                };
            }

            let one = E::one();
            let bias_correction1 = one - self.beta1.powf(E::from_usize(step));
            let bias_correction2 = one - self.beta2.powf(E::from_usize(step));
            let updated = values
                .into_iter()
                .zip(grad)
                .zip(state.m.iter_mut().zip(state.v.iter_mut()))
                .map(|((value, grad), (m, v))| {
                    *m = self.beta1 * *m + (one - self.beta1) * grad;
                    *v = self.beta2 * *v + (one - self.beta2) * grad * grad;
                    let m_hat = *m / bias_correction1;
                    let v_hat = *v / bias_correction2;
                    value - self.lr * m_hat / (v_hat.sqrt() + self.eps)
                })
                .collect();
            parameter.set_values(updated);
        }

        if advanced_step {
            self.step = step;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Adam, OptimParameter, SGD, SGDMomentum};
    use crate::backend::Cpu;
    use crate::nn::Parameter;
    use crate::tensor::Tensor1D;

    fn assert_close_slice(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    struct SignDescent<E> {
        lr: E,
    }

    impl<E> SignDescent<E>
    where
        E: crate::dtype::FloatElement,
    {
        fn step<'a, B, P>(&mut self, params: P)
        where
            B: crate::backend::Backend<E> + 'a,
            E: 'a,
            P: IntoIterator<Item = &'a mut dyn OptimParameter<E, B>>,
        {
            for parameter in params {
                let Some(grad) = parameter.grad_values() else {
                    continue;
                };

                let updated = parameter
                    .values()
                    .into_iter()
                    .zip(grad)
                    .map(|(value, grad)| {
                        if grad > E::zero() {
                            value - self.lr
                        } else if grad < E::zero() {
                            value + self.lr
                        } else {
                            value
                        }
                    })
                    .collect();
                parameter.set_values(updated);
            }
        }
    }

    #[test]
    fn sgd_updates_parameter_values() {
        let mut parameter = Parameter::new(Tensor1D::<2>::from_array([1.0, 2.0]));
        parameter.tensor().sum().backward();

        let mut optimizer = SGD::new(0.5);
        optimizer.step(vec![&mut parameter as &mut dyn OptimParameter<f32, Cpu>]);

        assert_eq!(parameter.tensor().to_vec(), vec![0.5, 1.5]);
    }

    #[test]
    fn sgd_skips_parameters_without_gradients() {
        let mut parameter = Parameter::new(Tensor1D::<2>::from_array([1.0, 2.0]));

        let mut optimizer = SGD::new(0.5);
        optimizer.step(vec![&mut parameter as &mut dyn OptimParameter<f32, Cpu>]);

        assert_eq!(parameter.tensor().to_vec(), vec![1.0, 2.0]);
    }

    #[test]
    fn optim_parameter_trait_supports_new_update_rules() {
        let mut parameter = Parameter::new(Tensor1D::<2>::from_array([1.0, 2.0]));
        parameter
            .tensor()
            .mul(&Tensor1D::<2>::from_array([2.0, -3.0]))
            .sum()
            .backward();

        let mut optimizer = SignDescent { lr: 0.25 };
        optimizer.step(vec![&mut parameter as &mut dyn OptimParameter<f32, Cpu>]);

        assert_eq!(parameter.tensor().to_vec(), vec![0.75, 2.25]);
    }

    #[test]
    fn sgd_momentum_matches_hand_calculated_two_step_update() {
        let mut parameter = Parameter::new(Tensor1D::<2>::from_array([1.0, 2.0]));
        let mut optimizer = SGDMomentum::new(0.1, 0.5);

        parameter
            .tensor()
            .mul(&Tensor1D::<2>::from_array([2.0, -4.0]))
            .sum()
            .backward();
        optimizer.step(vec![&mut parameter as &mut dyn OptimParameter<f32, Cpu>]);

        parameter.zero_grad();
        parameter.tensor().sum().backward();
        optimizer.step(vec![&mut parameter as &mut dyn OptimParameter<f32, Cpu>]);

        assert_close_slice(&parameter.tensor().to_vec(), &[0.6, 2.5]);
    }

    #[test]
    fn sgd_momentum_state_is_keyed_by_parameter_id() {
        let mut first = Parameter::new(Tensor1D::<1>::from_array([1.0]));
        let mut second = Parameter::new(Tensor1D::<1>::from_array([1.0]));
        let mut optimizer = SGDMomentum::new(0.1, 0.5);

        first.tensor().mul_scalar(2.0).sum().backward();
        optimizer.step(vec![&mut first as &mut dyn OptimParameter<f32, Cpu>]);

        second.tensor().sum().backward();
        optimizer.step(vec![&mut second as &mut dyn OptimParameter<f32, Cpu>]);

        assert_close_slice(&first.tensor().to_vec(), &[0.8]);
        assert_close_slice(&second.tensor().to_vec(), &[0.9]);
        assert!(optimizer.velocity.contains_key(&first.id()));
        assert!(optimizer.velocity.contains_key(&second.id()));
    }

    #[test]
    fn adam_first_step_matches_hand_calculated_update() {
        let mut parameter = Parameter::new(Tensor1D::<2>::from_array([1.0, 2.0]));
        parameter
            .tensor()
            .mul(&Tensor1D::<2>::from_array([2.0, -4.0]))
            .sum()
            .backward();

        let mut optimizer = Adam::new(0.1).with_eps(0.0);
        optimizer.step(vec![&mut parameter as &mut dyn OptimParameter<f32, Cpu>]);

        assert_close_slice(&parameter.tensor().to_vec(), &[0.9, 2.1]);
    }

    #[test]
    fn adam_skips_parameters_without_gradients() {
        let mut parameter = Parameter::new(Tensor1D::<2>::from_array([1.0, 2.0]));
        let mut optimizer = Adam::new(0.1);

        optimizer.step(vec![&mut parameter as &mut dyn OptimParameter<f32, Cpu>]);

        assert_eq!(parameter.tensor().to_vec(), vec![1.0, 2.0]);
        assert!(optimizer.state.is_empty());
        assert_eq!(optimizer.step, 0);
    }

    #[test]
    fn adam_reuses_state_and_preserves_parameter_id() {
        let mut parameter = Parameter::new(Tensor1D::<1>::from_array([1.0]));
        let parameter_id = parameter.id();
        let mut optimizer = Adam::new(0.1);

        parameter.tensor().sum().backward();
        optimizer.step(vec![&mut parameter as &mut dyn OptimParameter<f32, Cpu>]);
        let state_count = optimizer.state.len();

        parameter.zero_grad();
        parameter.tensor().sum().backward();
        optimizer.step(vec![&mut parameter as &mut dyn OptimParameter<f32, Cpu>]);

        assert_eq!(optimizer.state.len(), state_count);
        assert_eq!(parameter.id(), parameter_id);
    }

    #[test]
    fn stateful_optimizers_keep_updated_parameters_as_leaf_tensors() {
        let mut momentum_parameter = Parameter::new(Tensor1D::<1>::from_array([2.0]));
        momentum_parameter.tensor().mul_scalar(3.0).sum().backward();
        let mut momentum = SGDMomentum::new(0.5, 0.9);
        momentum.step(vec![
            &mut momentum_parameter as &mut dyn OptimParameter<f32, Cpu>,
        ]);

        let mut adam_parameter = Parameter::new(Tensor1D::<1>::from_array([2.0]));
        adam_parameter.tensor().mul_scalar(3.0).sum().backward();
        let mut adam = Adam::new(0.5);
        adam.step(vec![
            &mut adam_parameter as &mut dyn OptimParameter<f32, Cpu>,
        ]);

        assert!(momentum_parameter.tensor().requires_grad_enabled());
        assert!(momentum_parameter.tensor().is_leaf());
        assert!(adam_parameter.tensor().requires_grad_enabled());
        assert!(adam_parameter.tensor().is_leaf());
    }
}
