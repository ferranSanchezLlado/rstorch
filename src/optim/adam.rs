use super::Optimizer;
use crate::backend::Backend;
use crate::dtype::FloatDType;
use crate::error::Result;
use crate::nn::{ParameterId, ParameterRefMut};
use crate::no_grad;
use std::collections::HashMap;

pub struct Adam<E> {
    lr: E,
    beta1: E,
    beta2: E,
    eps: E,
    step: usize,
    state: HashMap<ParameterId, AdamState<E>>,
}

struct AdamState<E> {
    m: Vec<E>,
    v: Vec<E>,
}

impl<E> Adam<E>
where
    E: FloatDType,
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
}

impl<E, B> Optimizer<E, B> for Adam<E>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn step(&mut self, params: &mut [ParameterRefMut<'_, E, B>]) -> Result<()> {
        let _guard = no_grad();
        self.step += 1;
        let one = E::one();
        let beta1_pow = pow(self.beta1, self.step);
        let beta2_pow = pow(self.beta2, self.step);
        for param in params {
            let Some(grad) = param.grad()? else {
                continue;
            };
            let data = param.data()?;
            let state = self.state.entry(param.id()).or_insert_with(|| AdamState {
                m: vec![E::zero(); grad.len()],
                v: vec![E::zero(); grad.len()],
            });
            if state.m.len() != grad.len() || state.v.len() != grad.len() {
                state.m = vec![E::zero(); grad.len()];
                state.v = vec![E::zero(); grad.len()];
            }

            let mut next = Vec::with_capacity(data.len());
            for ((value, &g), (m, v)) in data
                .into_iter()
                .zip(&grad)
                .zip(state.m.iter_mut().zip(state.v.iter_mut()))
            {
                *m = self.beta1 * *m + (one - self.beta1) * g;
                *v = self.beta2 * *v + (one - self.beta2) * g * g;
                let m_hat = *m / (one - beta1_pow);
                let v_hat = *v / (one - beta2_pow);
                next.push(value - self.lr * m_hat / (v_hat.sqrt() + self.eps));
            }
            param.set_data(next)?;
        }
        Ok(())
    }
}

fn pow<E: FloatDType>(value: E, n: usize) -> E {
    (0..n).fold(E::one(), |acc, _| acc * value)
}
