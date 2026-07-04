use super::Optimizer;
use crate::backend::Backend;
use crate::dtype::FloatDType;
use crate::error::{PersistenceError, Result};
use crate::nn::{HasParameters, ParameterId, ParameterRefMut};
use crate::no_grad;
use crate::persistence::{
    OptimizerKind, OptimizerState, OptimizerStateDict, TensorRecord, collect_parameter_snapshots,
    optimizer_parameter_state, scalar_record, validate_optimizer_kind,
    validate_optimizer_parameters,
};
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
    step: usize,
    state: &mut HashMap<ParameterId, AdamState<E>>,
    params: &mut [ParameterRefMut<'_, E, B>],
) -> Result<()>
where
    E: FloatDType,
    B: Backend<E>,
{
    let _guard = no_grad();
    let one = E::ONE;
    let beta1_pow = pow(config.beta1, step);
    let beta2_pow = pow(config.beta2, step);
    for param in params {
        let Some(grad) = param.grad()? else {
            continue;
        };
        let mut data = param.data()?;
        let moments = state.entry(param.id()).or_insert_with(|| AdamState {
            m: vec![E::ZERO; grad.len()],
            v: vec![E::ZERO; grad.len()],
        });
        if moments.m.len() != grad.len() || moments.v.len() != grad.len() {
            moments.m = vec![E::ZERO; grad.len()];
            moments.v = vec![E::ZERO; grad.len()];
        }

        for ((value, &g), (m, v)) in data
            .iter_mut()
            .zip(&grad)
            .zip(moments.m.iter_mut().zip(moments.v.iter_mut()))
        {
            *m = config.beta1 * *m + (one - config.beta1) * g;
            *v = config.beta2 * *v + (one - config.beta2) * g * g;
            let m_hat = *m / (one - beta1_pow);
            let v_hat = *v / (one - beta2_pow);
            let decayed = *value - config.lr * config.weight_decay * *value;
            *value = decayed - config.lr * m_hat / (v_hat.sqrt() + config.eps);
        }
        param.set_data(data)?;
    }
    Ok(())
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

    pub fn lr(&self) -> E {
        self.lr
    }

    pub fn set_lr(&mut self, lr: E) {
        self.lr = lr;
    }
}

impl<E, B> Optimizer<E, B> for Adam<E>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn step(&mut self, params: &mut [ParameterRefMut<'_, E, B>]) -> Result<()> {
        self.step += 1;
        let config = AdamConfig {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: E::ZERO,
        };
        adam_step(&config, self.step, &mut self.state, params)
    }
}

impl<E, B> OptimizerState<E, B> for Adam<E>
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
        let step = state.step();
        let moments = load_adam_parameter_states::<E, B, M>(module, state)?;

        self.lr = lr;
        self.beta1 = beta1;
        self.beta2 = beta2;
        self.eps = eps;
        self.step = step;
        self.state = moments;
        Ok(())
    }
}

fn pow<E: FloatDType>(value: E, n: usize) -> E {
    (0..n).fold(E::ONE, |acc, _| acc * value)
}

pub struct AdamW<E> {
    lr: E,
    beta1: E,
    beta2: E,
    eps: E,
    weight_decay: E,
    step: usize,
    state: HashMap<ParameterId, AdamState<E>>,
}

impl<E> AdamW<E>
where
    E: FloatDType,
{
    pub fn new(lr: E, weight_decay: E) -> Self {
        Self {
            lr,
            beta1: E::from_f64(0.9),
            beta2: E::from_f64(0.999),
            eps: E::from_f64(1e-8),
            weight_decay,
            step: 0,
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

impl<E, B> Optimizer<E, B> for AdamW<E>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn step(&mut self, params: &mut [ParameterRefMut<'_, E, B>]) -> Result<()> {
        self.step += 1;
        let config = AdamConfig {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
        };
        adam_step(&config, self.step, &mut self.state, params)
    }
}

impl<E, B> OptimizerState<E, B> for AdamW<E>
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
        self.state = moments;
        Ok(())
    }
}

fn adam_parameter_states<E, B, M>(
    state: &HashMap<ParameterId, AdamState<E>>,
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
                Some(moments) => vec![
                    TensorRecord::from_values("m", meta.dims.clone(), &moments.m)?,
                    TensorRecord::from_values("v", meta.dims.clone(), &moments.v)?,
                ],
                None => Vec::new(),
            };
            optimizer_parameter_state::<E>(&meta, tensors)
        })
        .collect()
}

fn load_adam_parameter_states<E, B, M>(
    module: &M,
    state: &OptimizerStateDict,
) -> Result<HashMap<ParameterId, AdamState<E>>>
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
                out.insert(snapshot.id, AdamState { m, v });
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
