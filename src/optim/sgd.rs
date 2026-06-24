use super::Optimizer;
use crate::backend::Backend;
use crate::dtype::FloatDType;
use crate::error::Result;
use crate::nn::{ParameterId, ParameterRefMut};
use crate::no_grad;
use std::collections::HashMap;

pub struct Sgd<E> {
    lr: E,
    momentum: Option<E>,
    velocity: HashMap<ParameterId, Vec<E>>,
}

impl<E> Sgd<E>
where
    E: FloatDType,
{
    pub fn new(lr: E) -> Self {
        Self {
            lr,
            momentum: None,
            velocity: HashMap::new(),
        }
    }

    pub fn with_momentum(lr: E, momentum: E) -> Self {
        Self {
            lr,
            momentum: Some(momentum),
            velocity: HashMap::new(),
        }
    }
}

impl<E, B> Optimizer<E, B> for Sgd<E>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn step(&mut self, params: &mut [ParameterRefMut<'_, E, B>]) -> Result<()> {
        let _guard = no_grad();
        for param in params {
            let Some(grad) = param.grad()? else {
                continue;
            };
            let data = param.data()?;
            let update = if let Some(momentum) = self.momentum {
                let velocity = self
                    .velocity
                    .entry(param.id())
                    .or_insert_with(|| vec![E::zero(); grad.len()]);
                if velocity.len() != grad.len() {
                    *velocity = vec![E::zero(); grad.len()];
                }
                for (v, &g) in velocity.iter_mut().zip(&grad) {
                    *v = *v * momentum + g;
                }
                velocity.clone()
            } else {
                grad
            };
            let next = data
                .into_iter()
                .zip(update)
                .map(|(value, grad)| value - self.lr * grad)
                .collect();
            param.set_data(next)?;
        }
        Ok(())
    }
}
