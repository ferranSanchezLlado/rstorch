use super::Optimizer;
use crate::backend::{Backend, Cpu};
use crate::dtype::FloatDType;
use crate::error::{Error, PersistenceError, Result};
use crate::nn::{HasParameters, ParameterId, ParameterRefMut};
use crate::no_grad;
use crate::persistence::{
    OptimizerKind, OptimizerState, OptimizerStateDict, TensorRecord, collect_parameter_snapshots,
    optimizer_parameter_state, scalar_record, validate_optimizer_kind,
    validate_optimizer_parameters,
};
use std::collections::HashMap;
use std::marker::PhantomData;

pub struct Adam<E, B = Cpu>
where
    E: FloatDType,
    B: Backend<E>,
{
    lr: E,
    beta1: E,
    beta2: E,
    eps: E,
    weight_decay: E,
    step: usize,
    beta1_pow: E,
    beta2_pow: E,
    state: HashMap<ParameterId, AdamState<E, B>>,
}

struct AdamState<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    device: B::Device,
    m: B::Storage,
    v: B::Storage,
    _dtype: PhantomData<E>,
}

/// Hyperparameters shared by the Adam and AdamW update.
///
/// `weight_decay` is the decoupled term: plain Adam passes zero, AdamW passes
/// its configured decay.
struct AdamConfig<E> {
    lr: E,
    beta1: E,
    beta2: E,
    eps: E,
    weight_decay: E,
}

/// Applies one moment-estimate update step to every parameter that has a
/// gradient, shared by [`Adam`] and [`AdamW`].
fn adam_step<E, B>(
    config: &AdamConfig<E>,
    beta1_pow: E,
    beta2_pow: E,
    state: &mut HashMap<ParameterId, AdamState<E, B>>,
    params: &mut [ParameterRefMut<'_, E, B>],
) -> Result<()>
where
    E: FloatDType,
    B: Backend<E>,
{
    let _guard = no_grad();
    for param in params {
        let id = param.id();
        let previous_m = state.get(&id).map(|state| state.m.clone());
        let previous_v = state.get(&id).map(|state| state.v.clone());
        // `next_state` is populated only when the closure runs, i.e. when the
        // parameter has a gradient; a gradient-less parameter keeps prior state.
        let mut next_state = None;
        param.update_storage(&mut |device, data, grad, len| {
            let (next, m, v) = B::adam_step(
                device,
                data,
                grad,
                previous_m.as_ref(),
                previous_v.as_ref(),
                len,
                config.lr,
                config.beta1,
                config.beta2,
                config.eps,
                config.weight_decay,
                beta1_pow,
                beta2_pow,
            )
            .map_err(Error::backend)?;
            next_state = Some(AdamState {
                device: device.clone(),
                m,
                v,
                _dtype: PhantomData,
            });
            Ok(next)
        })?;
        if let Some(next_state) = next_state {
            state.insert(id, next_state);
        }
    }
    Ok(())
}

impl<E, B> Adam<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn new(lr: E) -> Self {
        Self {
            lr,
            beta1: E::from_f64(0.9),
            beta2: E::from_f64(0.999),
            eps: E::from_f64(1e-8),
            weight_decay: E::ZERO,
            step: 0,
            beta1_pow: E::ONE,
            beta2_pow: E::ONE,
            state: HashMap::new(),
        }
    }

    pub fn with_weight_decay(lr: E, weight_decay: E) -> Self {
        Self {
            weight_decay,
            ..Self::new(lr)
        }
    }

    pub fn lr(&self) -> E {
        self.lr
    }

    pub fn set_lr(&mut self, lr: E) {
        self.lr = lr;
    }

    pub fn weight_decay(&self) -> E {
        self.weight_decay
    }
}

impl<E, B> Optimizer<E, B> for Adam<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn step(&mut self, params: &mut [ParameterRefMut<'_, E, B>]) -> Result<()> {
        self.step += 1;
        self.beta1_pow *= self.beta1;
        self.beta2_pow *= self.beta2;
        let config = AdamConfig {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
        };
        adam_step(
            &config,
            self.beta1_pow,
            self.beta2_pow,
            &mut self.state,
            params,
        )
    }
}

impl<E, B> OptimizerState<E, B> for Adam<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn state_dict<M>(&self, module: &M) -> Result<OptimizerStateDict>
    where
        M: HasParameters<E, B>,
    {
        OptimizerStateDict::new(
            OptimizerKind::Adam,
            E::ID,
            self.step,
            vec![
                scalar_record("lr", self.lr),
                scalar_record("beta1", self.beta1),
                scalar_record("beta2", self.beta2),
                scalar_record("eps", self.eps),
                scalar_record("weight_decay", self.weight_decay),
            ],
            adam_parameter_states::<E, B, M>(&self.state, module)?,
        )
    }

    fn load_state_dict<M>(&mut self, module: &M, state: &OptimizerStateDict) -> Result<()>
    where
        M: HasParameters<E, B>,
    {
        validate_optimizer_kind(state, OptimizerKind::Adam)?;
        let lr = state.hyper_value("lr")?;
        let beta1 = state.hyper_value("beta1")?;
        let beta2 = state.hyper_value("beta2")?;
        let eps = state.hyper_value("eps")?;
        let weight_decay = state
            .optional_hyper_value("weight_decay")?
            .unwrap_or(E::ZERO);
        let step = state.step();
        let moments = load_adam_parameter_states::<E, B, M>(module, state)?;

        self.lr = lr;
        self.beta1 = beta1;
        self.beta2 = beta2;
        self.eps = eps;
        self.weight_decay = weight_decay;
        self.step = step;
        self.beta1_pow = pow(beta1, step);
        self.beta2_pow = pow(beta2, step);
        self.state = moments;
        Ok(())
    }
}

fn pow<E: FloatDType>(value: E, n: usize) -> E {
    (0..n).fold(E::ONE, |acc, _| acc * value)
}

pub struct AdamW<E, B = Cpu>
where
    E: FloatDType,
    B: Backend<E>,
{
    lr: E,
    beta1: E,
    beta2: E,
    eps: E,
    weight_decay: E,
    step: usize,
    beta1_pow: E,
    beta2_pow: E,
    state: HashMap<ParameterId, AdamState<E, B>>,
}

impl<E, B> AdamW<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn new(lr: E, weight_decay: E) -> Self {
        Self {
            lr,
            beta1: E::from_f64(0.9),
            beta2: E::from_f64(0.999),
            eps: E::from_f64(1e-8),
            weight_decay,
            step: 0,
            beta1_pow: E::ONE,
            beta2_pow: E::ONE,
            state: HashMap::new(),
        }
    }

    pub fn lr(&self) -> E {
        self.lr
    }

    pub fn set_lr(&mut self, lr: E) {
        self.lr = lr;
    }

    pub fn weight_decay(&self) -> E {
        self.weight_decay
    }
}

impl<E, B> Optimizer<E, B> for AdamW<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn step(&mut self, params: &mut [ParameterRefMut<'_, E, B>]) -> Result<()> {
        self.step += 1;
        self.beta1_pow *= self.beta1;
        self.beta2_pow *= self.beta2;
        let config = AdamConfig {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
        };
        adam_step(
            &config,
            self.beta1_pow,
            self.beta2_pow,
            &mut self.state,
            params,
        )
    }
}

impl<E, B> OptimizerState<E, B> for AdamW<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn state_dict<M>(&self, module: &M) -> Result<OptimizerStateDict>
    where
        M: HasParameters<E, B>,
    {
        OptimizerStateDict::new(
            OptimizerKind::AdamW,
            E::ID,
            self.step,
            vec![
                scalar_record("lr", self.lr),
                scalar_record("beta1", self.beta1),
                scalar_record("beta2", self.beta2),
                scalar_record("eps", self.eps),
                scalar_record("weight_decay", self.weight_decay),
            ],
            adam_parameter_states::<E, B, M>(&self.state, module)?,
        )
    }

    fn load_state_dict<M>(&mut self, module: &M, state: &OptimizerStateDict) -> Result<()>
    where
        M: HasParameters<E, B>,
    {
        validate_optimizer_kind(state, OptimizerKind::AdamW)?;
        let lr = state.hyper_value("lr")?;
        let beta1 = state.hyper_value("beta1")?;
        let beta2 = state.hyper_value("beta2")?;
        let eps = state.hyper_value("eps")?;
        let weight_decay = state.hyper_value("weight_decay")?;
        let step = state.step();
        let moments = load_adam_parameter_states::<E, B, M>(module, state)?;

        self.lr = lr;
        self.beta1 = beta1;
        self.beta2 = beta2;
        self.eps = eps;
        self.weight_decay = weight_decay;
        self.step = step;
        self.beta1_pow = pow(beta1, step);
        self.beta2_pow = pow(beta2, step);
        self.state = moments;
        Ok(())
    }
}

fn adam_parameter_states<E, B, M>(
    state: &HashMap<ParameterId, AdamState<E, B>>,
    module: &M,
) -> Result<Vec<crate::persistence::OptimizerParameterState>>
where
    E: FloatDType,
    B: Backend<E>,
    M: HasParameters<E, B>,
{
    collect_parameter_snapshots(module)?
        .into_iter()
        .map(|meta| {
            let tensors = match state.get(&meta.id) {
                Some(moments) => {
                    let m = B::to_vec(&moments.device, &moments.m).map_err(Error::backend)?;
                    let v = B::to_vec(&moments.device, &moments.v).map_err(Error::backend)?;
                    vec![
                        TensorRecord::from_values("m", meta.dims.clone(), &m)?,
                        TensorRecord::from_values("v", meta.dims.clone(), &v)?,
                    ]
                }
                None => Vec::new(),
            };
            optimizer_parameter_state::<E>(&meta, tensors)
        })
        .collect()
}

fn load_adam_parameter_states<E, B, M>(
    module: &M,
    state: &OptimizerStateDict,
) -> Result<HashMap<ParameterId, AdamState<E, B>>>
where
    E: FloatDType,
    B: Backend<E>,
    M: HasParameters<E, B>,
{
    let snapshots = validate_optimizer_parameters::<E, B, M>(module, state)?;
    let mut out = HashMap::new();
    for (snapshot, saved) in snapshots.into_iter().zip(state.parameters()) {
        let mut m = None;
        let mut v = None;
        for tensor in saved.tensors() {
            match tensor.name() {
                "m" => m = Some(tensor.to_values::<E>()?),
                "v" => v = Some(tensor.to_values::<E>()?),
                name => {
                    return Err(PersistenceError::UnexpectedTensor {
                        name: format!("{}.{}", snapshot.name, name),
                    }
                    .into());
                }
            }
        }
        match (m, v) {
            (Some(m), Some(v)) => {
                let device = B::default_device().map_err(Error::backend)?;
                let m = B::from_vec(&device, m).map_err(Error::backend)?;
                let v = B::from_vec(&device, v).map_err(Error::backend)?;
                out.insert(
                    snapshot.id,
                    AdamState {
                        device,
                        m,
                        v,
                        _dtype: PhantomData,
                    },
                );
            }
            (None, None) => {}
            (Some(_), None) => {
                return Err(PersistenceError::MissingTensor {
                    name: format!("{}.v", snapshot.name),
                }
                .into());
            }
            (None, Some(_)) => {
                return Err(PersistenceError::MissingTensor {
                    name: format!("{}.m", snapshot.name),
                }
                .into());
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::pow;

    #[test]
    fn pow_matches_repeated_multiply_for_small_steps() {
        for step in 0..32 {
            let expected = (0..step).fold(1.0f64, |acc, _| acc * 0.9);
            assert_eq!(pow(0.9f64, step), expected);
        }
    }

    #[test]
    fn pow_preserves_checkpoint_reconstruction_order() {
        let mut expected = 1.0f64;
        for _ in 0..257 {
            expected *= 0.999;
        }
        assert_eq!(pow(0.999f64, 257), expected);
    }
}
