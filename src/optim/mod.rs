//! Minimal optimizers.

use crate::backend::Backend;
use crate::dtype::FloatElement;
use crate::tensor::autograd::no_grad;

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

#[cfg(test)]
mod tests {
    use super::{OptimParameter, SGD};
    use crate::backend::Cpu;
    use crate::nn::Parameter;
    use crate::tensor::Tensor1D;

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
}
