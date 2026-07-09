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

pub struct Sgd<E, B = Cpu>
where
    E: FloatDType,
    B: Backend<E>,
{
    lr: E,
    momentum: Option<E>,
    weight_decay: E,
    velocity: HashMap<ParameterId, OptimizerStorage<E, B>>,
    _backend: PhantomData<B>,
}

struct OptimizerStorage<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    device: B::Device,
    storage: B::Storage,
    _dtype: PhantomData<E>,
}

impl<E, B> Sgd<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn new(lr: E) -> Self {
        Self {
            lr,
            momentum: None,
            weight_decay: E::ZERO,
            velocity: HashMap::new(),
            _backend: PhantomData,
        }
    }

    pub fn with_weight_decay(lr: E, weight_decay: E) -> Self {
        Self {
            lr,
            momentum: None,
            weight_decay,
            velocity: HashMap::new(),
            _backend: PhantomData,
        }
    }

    pub fn with_momentum(lr: E, momentum: E) -> Self {
        Self {
            lr,
            momentum: Some(momentum),
            weight_decay: E::ZERO,
            velocity: HashMap::new(),
            _backend: PhantomData,
        }
    }

    pub fn with_momentum_and_weight_decay(lr: E, momentum: E, weight_decay: E) -> Self {
        Self {
            lr,
            momentum: Some(momentum),
            weight_decay,
            velocity: HashMap::new(),
            _backend: PhantomData,
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

impl<E, B> Optimizer<E, B> for Sgd<E, B>
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
            let previous_velocity = self
                .velocity
                .get(&id)
                .map(|velocity| velocity.storage.clone());
            let mut next_velocity = None;
            let updated = param.update_storage(&mut |device, data, grad, len| {
                let (next, velocity) = B::sgd_step(
                    device,
                    data,
                    grad,
                    previous_velocity.as_ref(),
                    len,
                    lr,
                    momentum,
                    weight_decay,
                )
                .map_err(Error::backend)?;
                next_velocity = velocity.map(|storage| OptimizerStorage {
                    device: device.clone(),
                    storage,
                    _dtype: PhantomData,
                });
                Ok(next)
            })?;
            if updated {
                if let Some(velocity) = next_velocity {
                    self.velocity.insert(id, velocity);
                } else {
                    self.velocity.remove(&id);
                }
            }
        }
        Ok(())
    }
}

impl<E, B> OptimizerState<E, B> for Sgd<E, B>
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
                        let values = B::to_vec(&velocity.device, &velocity.storage)
                            .map_err(Error::backend)?;
                        vec![TensorRecord::from_values(
                            "velocity",
                            meta.dims.clone(),
                            &values,
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
                    "velocity" => {
                        let values = tensor.to_values::<E>()?;
                        let device = B::default_device().map_err(Error::backend)?;
                        let storage = B::from_vec(&device, values).map_err(Error::backend)?;
                        velocity = Some(OptimizerStorage {
                            device,
                            storage,
                            _dtype: PhantomData,
                        });
                    }
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
