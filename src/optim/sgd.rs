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

pub struct Sgd<E> {
    lr: E,
    momentum: Option<E>,
    weight_decay: E,
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
            weight_decay: E::ZERO,
            velocity: HashMap::new(),
        }
    }

    pub fn with_weight_decay(lr: E, weight_decay: E) -> Self {
        Self {
            lr,
            momentum: None,
            weight_decay,
            velocity: HashMap::new(),
        }
    }

    pub fn with_momentum(lr: E, momentum: E) -> Self {
        Self {
            lr,
            momentum: Some(momentum),
            weight_decay: E::ZERO,
            velocity: HashMap::new(),
        }
    }

    pub fn with_momentum_and_weight_decay(lr: E, momentum: E, weight_decay: E) -> Self {
        Self {
            lr,
            momentum: Some(momentum),
            weight_decay,
            velocity: HashMap::new(),
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

impl<E, B> Optimizer<E, B> for Sgd<E>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn step(&mut self, params: &mut [ParameterRefMut<'_, E, B>]) -> Result<()> {
        let _guard = no_grad();
        for param in params {
            let id = param.id();
            let lr = self.lr;
            let momentum = self.momentum;
            let weight_decay = self.weight_decay;
            let velocity = &mut self.velocity;
            param.update_data(&mut |data, grad| {
                let mut next = data.to_vec();
                if weight_decay != E::ZERO {
                    for value in &mut next {
                        *value -= lr * weight_decay * *value;
                    }
                }
                if let Some(momentum) = momentum {
                    let velocity = velocity
                        .entry(id)
                        .or_insert_with(|| vec![E::ZERO; grad.len()]);
                    if velocity.len() != grad.len() {
                        *velocity = vec![E::ZERO; grad.len()];
                    }
                    for (v, &g) in velocity.iter_mut().zip(grad) {
                        *v = *v * momentum + g;
                    }
                    for (value, &v) in next.iter_mut().zip(velocity.iter()) {
                        *value -= lr * v;
                    }
                } else {
                    for (value, &g) in next.iter_mut().zip(grad) {
                        *value -= lr * g;
                    }
                }
                next
            })?;
        }
        Ok(())
    }
}

impl<E, B> OptimizerState<E, B> for Sgd<E>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn state_dict<M>(&self, module: &M) -> Result<OptimizerStateDict>
    where
        M: HasParameters<E, B>,
    {
        let parameters = collect_parameter_snapshots(module)?
            .into_iter()
            .map(|meta| {
                let tensors = match self.velocity.get(&meta.id) {
                    Some(velocity) => {
                        vec![TensorRecord::from_values(
                            "velocity",
                            meta.dims.clone(),
                            velocity,
                        )?]
                    }
                    None => Vec::new(),
                };
                optimizer_parameter_state::<E>(&meta, tensors)
            })
            .collect::<Result<Vec<_>>>()?;

        let mut hyperparameters = vec![scalar_record("lr", self.lr)];
        if let Some(momentum) = self.momentum {
            hyperparameters.push(scalar_record("momentum", momentum));
        }
        hyperparameters.push(scalar_record("weight_decay", self.weight_decay));

        OptimizerStateDict::new(OptimizerKind::Sgd, E::ID, 0, hyperparameters, parameters)
    }

    fn load_state_dict<M>(&mut self, module: &M, state: &OptimizerStateDict) -> Result<()>
    where
        M: HasParameters<E, B>,
    {
        validate_optimizer_kind(state, OptimizerKind::Sgd)?;
        let snapshots = validate_optimizer_parameters::<E, B, M>(module, state)?;
        let lr = state.hyper_value("lr")?;
        let momentum = state.optional_hyper_value("momentum")?;
        let weight_decay = state
            .optional_hyper_value("weight_decay")?
            .unwrap_or(E::ZERO);
        let mut velocity_by_id = HashMap::new();

        for (snapshot, saved) in snapshots.into_iter().zip(state.parameters()) {
            let mut velocity = None;
            for tensor in saved.tensors() {
                match tensor.name() {
                    "velocity" => velocity = Some(tensor.to_values::<E>()?),
                    name => {
                        return Err(PersistenceError::UnexpectedTensor {
                            name: format!("{}.{}", snapshot.name, name),
                        }
                        .into());
                    }
                }
            }
            if let Some(velocity) = velocity {
                velocity_by_id.insert(snapshot.id, velocity);
            }
        }

        self.lr = lr;
        self.momentum = momentum;
        self.weight_decay = weight_decay;
        self.velocity = velocity_by_id;
        Ok(())
    }
}
