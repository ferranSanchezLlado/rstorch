//! Versioned persistence for tensors, module state dicts, optimizer state, and
//! checkpoints.
//!
//! Epoch 13 deliberately uses a small in-house little-endian format instead of
//! adding `serde` or `safetensors`: the current crate needs strict CPU-first
//! validation and bit-exact round trips, but does not yet need mmap or external
//! interop. Every artifact starts with an 8-byte magic plus a `u32` version.
//! Tensor records store a UTF-8 name, dtype code, dimensions, byte length, and
//! raw dtype bytes. Module and optimizer state use stable parameter names from
//! `HasParameters`; `ParameterId` values are never serialized.

use crate::backend::{Backend, Cpu};
use crate::dtype::{DType, DTypeId, FloatDType};
use crate::error::{PersistenceError, Result};
use crate::nn::{HasParameters, ParameterId, ParameterRefMut};
use crate::random::SmallRng;
use crate::shape::ShapeSpec;
use crate::tensor::Tensor;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const VERSION: u32 = 1;
const STATE_MAGIC: &[u8; 8] = b"RSTSD13\0";
const TENSOR_MAGIC: &[u8; 8] = b"RSTTN13\0";
const OPTIMIZER_MAGIC: &[u8; 8] = b"RSTOP13\0";
const CHECKPOINT_MAGIC: &[u8; 8] = b"RSTCK13\0";
const MAX_STRING_BYTES: u64 = 1 << 20;
const MAX_RANK: u64 = 64;
const MAX_RECORDS: u64 = 1_000_000;
const MAX_TENSOR_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorRecord {
    name: String,
    dtype: DTypeId,
    dims: Vec<usize>,
    data: Vec<u8>,
}

impl TensorRecord {
    pub fn from_tensor<S, E, B>(name: impl Into<String>, tensor: &Tensor<S, E, B>) -> Result<Self>
    where
        S: ShapeSpec,
        E: DType,
        B: Backend<E>,
    {
        Self::from_values(name, tensor.shape().dims().to_vec(), &tensor.to_vec()?)
    }

    pub fn from_values<E>(
        name: impl Into<String>,
        dims: impl Into<Vec<usize>>,
        values: &[E],
    ) -> Result<Self>
    where
        E: DType,
    {
        let name = name.into();
        let dims = dims.into();
        let expected_len = checked_numel(&name, &dims)?;
        if expected_len != values.len() {
            return Err(PersistenceError::LengthMismatch {
                name,
                expected: expected_len,
                found: values.len(),
            }
            .into());
        }

        let mut data = Vec::with_capacity(values.len().saturating_mul(E::BYTE_SIZE));
        for &value in values {
            value.write_le_bytes(&mut data);
        }
        Self::from_bytes(name, E::ID, dims, data)
    }

    pub fn from_bytes(
        name: impl Into<String>,
        dtype: DTypeId,
        dims: impl Into<Vec<usize>>,
        data: Vec<u8>,
    ) -> Result<Self> {
        let name = name.into();
        let dims = dims.into();
        let expected = checked_data_len(&name, dtype, &dims)?;
        if expected != data.len() {
            return Err(PersistenceError::LengthMismatch {
                name,
                expected,
                found: data.len(),
            }
            .into());
        }
        Ok(Self {
            name,
            dtype,
            dims,
            data,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dtype(&self) -> DTypeId {
        self.dtype
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn to_values<E>(&self) -> Result<Vec<E>>
    where
        E: DType,
    {
        if self.dtype != E::ID {
            return Err(PersistenceError::DTypeMismatch {
                name: self.name.clone(),
                expected: E::ID,
                found: self.dtype,
            }
            .into());
        }
        let expected = checked_data_len(&self.name, self.dtype, &self.dims)?;
        if expected != self.data.len() {
            return Err(PersistenceError::LengthMismatch {
                name: self.name.clone(),
                expected,
                found: self.data.len(),
            }
            .into());
        }

        let mut out = Vec::with_capacity(self.data.len() / E::BYTE_SIZE);
        for chunk in self.data.chunks_exact(E::BYTE_SIZE) {
            let Some(value) = E::read_le_bytes(chunk) else {
                return Err(PersistenceError::InvalidFormat {
                    reason: "invalid dtype byte width",
                }
                .into());
            };
            out.push(value);
        }
        Ok(out)
    }

    pub fn to_tensor<S, E>(&self) -> Result<Tensor<S, E, Cpu>>
    where
        S: ShapeSpec,
        E: DType,
    {
        Tensor::from_vec_with_shape(self.to_values()?, self.dims.clone())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StateDict {
    records: Vec<TensorRecord>,
}

impl StateDict {
    pub fn new(records: Vec<TensorRecord>) -> Result<Self> {
        validate_unique_tensor_names(&records)?;
        Ok(Self { records })
    }

    pub fn from_module<M, E, B>(module: &M) -> Result<Self>
    where
        M: HasParameters<E, B>,
        E: FloatDType,
        B: Backend<E>,
    {
        let mut records = Vec::new();
        let mut seen = HashSet::new();
        let mut err = None;
        module.visit_parameters("", &mut |name, param| {
            if err.is_some() {
                return;
            }
            if !seen.insert(name.to_owned()) {
                err = Some(
                    PersistenceError::DuplicateTensor {
                        name: name.to_owned(),
                    }
                    .into(),
                );
                return;
            }
            let data = match param.data() {
                Ok(data) => data,
                Err(source) => {
                    err = Some(source);
                    return;
                }
            };
            match TensorRecord::from_values(name, param.dims(), &data) {
                Ok(record) => records.push(record),
                Err(source) => err = Some(source),
            }
        });
        if let Some(err) = err {
            return Err(err);
        }
        module.visit_buffers("", &mut |name, buf| {
            if err.is_some() {
                return;
            }
            if !seen.insert(name.to_owned()) {
                err = Some(
                    PersistenceError::DuplicateTensor {
                        name: name.to_owned(),
                    }
                    .into(),
                );
                return;
            }
            let data = match buf.data() {
                Ok(data) => data,
                Err(source) => {
                    err = Some(source);
                    return;
                }
            };
            match TensorRecord::from_values(name, buf.dims(), &data) {
                Ok(record) => records.push(record),
                Err(source) => err = Some(source),
            }
        });
        if let Some(err) = err {
            return Err(err);
        }
        Ok(Self { records })
    }

    pub fn records(&self) -> &[TensorRecord] {
        &self.records
    }

    pub fn into_records(self) -> Vec<TensorRecord> {
        self.records
    }

    pub fn get(&self, name: &str) -> Option<&TensorRecord> {
        self.records.iter().find(|record| record.name == name)
    }

    pub fn load_module<M, E, B>(&self, module: &mut M) -> Result<()>
    where
        M: HasParameters<E, B>,
        E: FloatDType,
        B: Backend<E>,
    {
        validate_unique_tensor_names(&self.records)?;

        // Collect metadata only (no stored refs) so param and buffer passes don't
        // produce overlapping mutable/immutable borrows of `module`.
        let mut param_metas: Vec<(String, DTypeId, Vec<usize>)> = Vec::new();
        module.visit_parameters("", &mut |name, param| {
            param_metas.push((name.to_owned(), param.dtype(), param.dims()));
        });
        validate_unique_parameter_names(param_metas.iter().map(|(n, _, _)| n.as_str()))?;

        let mut buf_metas: Vec<(String, DTypeId, Vec<usize>)> = Vec::new();
        module.visit_buffers("", &mut |name, buf| {
            buf_metas.push((name.to_owned(), buf.dtype(), buf.dims()));
        });
        validate_unique_parameter_names(buf_metas.iter().map(|(n, _, _)| n.as_str()))?;

        let records = self.record_map()?;
        let expected: HashSet<_> = param_metas
            .iter()
            .map(|(n, _, _)| n.as_str())
            .chain(buf_metas.iter().map(|(n, _, _)| n.as_str()))
            .collect();
        for record in &self.records {
            if !expected.contains(record.name.as_str()) {
                return Err(PersistenceError::UnexpectedTensor {
                    name: record.name.clone(),
                }
                .into());
            }
        }

        for (name, dtype, dims) in &param_metas {
            let Some(record) = records.get(name.as_str()) else {
                return Err(PersistenceError::MissingTensor { name: name.clone() }.into());
            };
            validate_record_matches(name.as_str(), *dtype, dims, record)?;
        }

        for (name, dtype, dims) in &buf_metas {
            let Some(record) = records.get(name.as_str()) else {
                return Err(PersistenceError::MissingTensor { name: name.clone() }.into());
            };
            validate_record_matches(name.as_str(), *dtype, dims, record)?;
        }

        // Mutable pass to load params; enclosed in a block so the borrows are
        // released before the immutable buffer pass below.
        {
            let mut params = Vec::new();
            module.visit_parameters_mut("", &mut |name, param| {
                params.push(NamedParameterMut {
                    name: name.to_owned(),
                    param,
                });
            });
            for mut param in params {
                let record = records
                    .get(param.name.as_str())
                    .expect("validated module state record disappeared");
                param.param.set_data(record.to_values::<E>()?)?;
            }
        }

        // Buffer set_data uses Mutex interior mutability, so only &self is needed.
        let mut err: Option<crate::error::Error> = None;
        module.visit_buffers("", &mut |name, buf| {
            if err.is_some() {
                return;
            }
            let record = records
                .get(name)
                .expect("validated module state record disappeared");
            match record.to_values::<E>() {
                Ok(values) => {
                    if let Err(e) = buf.set_data(values) {
                        err = Some(e);
                    }
                }
                Err(e) => err = Some(e),
            }
        });
        if let Some(e) = err {
            return Err(e);
        }

        Ok(())
    }

    pub fn save_to_path(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path).map_err(|source| PersistenceError::Io { source })?;
        let mut writer = BufWriter::new(file);
        write_header(&mut writer, STATE_MAGIC)?;
        write_state_dict_body(&mut writer, self)?;
        writer
            .flush()
            .map_err(|source| PersistenceError::Io { source })?;
        Ok(())
    }

    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path).map_err(|source| PersistenceError::Io { source })?;
        let mut reader = BufReader::new(file);
        read_header(&mut reader, STATE_MAGIC, "RSTSD13")?;
        let state = read_state_dict_body(&mut reader)?;
        ensure_eof(&mut reader)?;
        Ok(state)
    }

    fn record_map(&self) -> Result<HashMap<&str, &TensorRecord>> {
        let mut records = HashMap::new();
        for record in &self.records {
            if records.insert(record.name.as_str(), record).is_some() {
                return Err(PersistenceError::DuplicateTensor {
                    name: record.name.clone(),
                }
                .into());
            }
        }
        Ok(records)
    }
}

pub fn save_tensor<S, E, B>(path: impl AsRef<Path>, tensor: &Tensor<S, E, B>) -> Result<()>
where
    S: ShapeSpec,
    E: DType,
    B: Backend<E>,
{
    let record = TensorRecord::from_tensor("tensor", tensor)?;
    let file = File::create(path).map_err(|source| PersistenceError::Io { source })?;
    let mut writer = BufWriter::new(file);
    write_header(&mut writer, TENSOR_MAGIC)?;
    write_tensor_record(&mut writer, &record)?;
    writer
        .flush()
        .map_err(|source| PersistenceError::Io { source })?;
    Ok(())
}

pub fn load_tensor<S, E>(path: impl AsRef<Path>) -> Result<Tensor<S, E, Cpu>>
where
    S: ShapeSpec,
    E: DType,
{
    let file = File::open(path).map_err(|source| PersistenceError::Io { source })?;
    let mut reader = BufReader::new(file);
    read_header(&mut reader, TENSOR_MAGIC, "RSTTN13")?;
    let record = read_tensor_record(&mut reader)?;
    ensure_eof(&mut reader)?;
    if record.name != "tensor" {
        return Err(PersistenceError::UnexpectedTensor { name: record.name }.into());
    }
    record.to_tensor()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum OptimizerKind {
    Sgd,
    Adam,
    AdamW,
}

impl OptimizerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sgd => "sgd",
            Self::Adam => "adam",
            Self::AdamW => "adamw",
        }
    }

    fn code(self) -> u8 {
        match self {
            Self::Sgd => 1,
            Self::Adam => 2,
            Self::AdamW => 3,
        }
    }

    fn from_code(code: u8) -> Result<Self> {
        match code {
            1 => Ok(Self::Sgd),
            2 => Ok(Self::Adam),
            3 => Ok(Self::AdamW),
            _ => Err(PersistenceError::UnknownOptimizer { code }.into()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalarRecord {
    name: String,
    dtype: DTypeId,
    data: Vec<u8>,
}

impl ScalarRecord {
    pub fn from_value<E>(name: impl Into<String>, value: E) -> Self
    where
        E: DType,
    {
        let mut data = Vec::with_capacity(E::BYTE_SIZE);
        value.write_le_bytes(&mut data);
        Self {
            name: name.into(),
            dtype: E::ID,
            data,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dtype(&self) -> DTypeId {
        self.dtype
    }

    pub fn to_value<E>(&self) -> Result<E>
    where
        E: DType,
    {
        if self.dtype != E::ID {
            return Err(PersistenceError::DTypeMismatch {
                name: self.name.clone(),
                expected: E::ID,
                found: self.dtype,
            }
            .into());
        }
        if self.data.len() != E::BYTE_SIZE {
            return Err(PersistenceError::LengthMismatch {
                name: self.name.clone(),
                expected: E::BYTE_SIZE,
                found: self.data.len(),
            }
            .into());
        }
        E::read_le_bytes(&self.data).ok_or_else(|| {
            PersistenceError::InvalidFormat {
                reason: "invalid scalar byte width",
            }
            .into()
        })
    }

    fn from_bytes(name: String, dtype: DTypeId, data: Vec<u8>) -> Result<Self> {
        let expected = dtype_size(dtype);
        if data.len() != expected {
            return Err(PersistenceError::LengthMismatch {
                name,
                expected,
                found: data.len(),
            }
            .into());
        }
        Ok(Self { name, dtype, data })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptimizerParameterState {
    name: String,
    dtype: DTypeId,
    dims: Vec<usize>,
    tensors: Vec<TensorRecord>,
}

impl OptimizerParameterState {
    pub fn new(
        name: impl Into<String>,
        dtype: DTypeId,
        dims: impl Into<Vec<usize>>,
        tensors: Vec<TensorRecord>,
    ) -> Result<Self> {
        validate_unique_tensor_names(&tensors)?;
        Ok(Self {
            name: name.into(),
            dtype,
            dims: dims.into(),
            tensors,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dtype(&self) -> DTypeId {
        self.dtype
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn tensors(&self) -> &[TensorRecord] {
        &self.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorRecord> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptimizerStateDict {
    kind: OptimizerKind,
    dtype: DTypeId,
    step: usize,
    hyperparameters: Vec<ScalarRecord>,
    parameters: Vec<OptimizerParameterState>,
}

impl OptimizerStateDict {
    pub fn new(
        kind: OptimizerKind,
        dtype: DTypeId,
        step: usize,
        hyperparameters: Vec<ScalarRecord>,
        parameters: Vec<OptimizerParameterState>,
    ) -> Result<Self> {
        let mut hyper_names = HashSet::new();
        for hyper in &hyperparameters {
            if hyper.dtype != dtype {
                return Err(PersistenceError::DTypeMismatch {
                    name: hyper.name.clone(),
                    expected: dtype,
                    found: hyper.dtype,
                }
                .into());
            }
            if !hyper_names.insert(hyper.name.clone()) {
                return Err(PersistenceError::DuplicateTensor {
                    name: hyper.name.clone(),
                }
                .into());
            }
        }

        let mut parameter_names = HashSet::new();
        for parameter in &parameters {
            if parameter.dtype != dtype {
                return Err(PersistenceError::DTypeMismatch {
                    name: parameter.name.clone(),
                    expected: dtype,
                    found: parameter.dtype,
                }
                .into());
            }
            if !parameter_names.insert(parameter.name.clone()) {
                return Err(PersistenceError::DuplicateTensor {
                    name: parameter.name.clone(),
                }
                .into());
            }
            for tensor in &parameter.tensors {
                validate_record_matches(&parameter.name, dtype, &parameter.dims, tensor)?;
            }
        }

        Ok(Self {
            kind,
            dtype,
            step,
            hyperparameters,
            parameters,
        })
    }

    pub fn kind(&self) -> OptimizerKind {
        self.kind
    }

    pub fn dtype(&self) -> DTypeId {
        self.dtype
    }

    pub fn step(&self) -> usize {
        self.step
    }

    pub fn hyperparameters(&self) -> &[ScalarRecord] {
        &self.hyperparameters
    }

    pub fn parameters(&self) -> &[OptimizerParameterState] {
        &self.parameters
    }

    pub fn hyper_value<E>(&self, name: &str) -> Result<E>
    where
        E: DType,
    {
        self.optional_hyper_value(name)?.ok_or_else(|| {
            PersistenceError::MissingTensor {
                name: name.to_owned(),
            }
            .into()
        })
    }

    pub fn optional_hyper_value<E>(&self, name: &str) -> Result<Option<E>>
    where
        E: DType,
    {
        self.hyperparameters
            .iter()
            .find(|record| record.name == name)
            .map(ScalarRecord::to_value)
            .transpose()
    }

    pub fn save_to_path(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path).map_err(|source| PersistenceError::Io { source })?;
        let mut writer = BufWriter::new(file);
        write_header(&mut writer, OPTIMIZER_MAGIC)?;
        write_optimizer_state_body(&mut writer, self)?;
        writer
            .flush()
            .map_err(|source| PersistenceError::Io { source })?;
        Ok(())
    }

    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path).map_err(|source| PersistenceError::Io { source })?;
        let mut reader = BufReader::new(file);
        read_header(&mut reader, OPTIMIZER_MAGIC, "RSTOP13")?;
        let state = read_optimizer_state_body(&mut reader)?;
        ensure_eof(&mut reader)?;
        Ok(state)
    }
}

pub trait OptimizerState<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn state_dict<M>(&self, module: &M) -> Result<OptimizerStateDict>
    where
        M: HasParameters<E, B>;

    fn load_state_dict<M>(&mut self, module: &M, state: &OptimizerStateDict) -> Result<()>
    where
        M: HasParameters<E, B>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Checkpoint {
    model: StateDict,
    optimizer: Option<OptimizerStateDict>,
    scheduler_step: usize,
    rng_state: Option<u64>,
    metadata: BTreeMap<String, String>,
}

impl Checkpoint {
    pub fn new(model: StateDict) -> Self {
        Self {
            model,
            optimizer: None,
            scheduler_step: 0,
            rng_state: None,
            metadata: BTreeMap::new(),
        }
    }

    pub fn from_training<M, O, E, B>(
        model: &M,
        optimizer: &O,
        scheduler_step: usize,
        rng: Option<&SmallRng>,
        metadata: BTreeMap<String, String>,
    ) -> Result<Self>
    where
        M: HasParameters<E, B>,
        O: OptimizerState<E, B>,
        E: FloatDType,
        B: Backend<E>,
    {
        Ok(Self {
            model: StateDict::from_module(model)?,
            optimizer: Some(optimizer.state_dict(model)?),
            scheduler_step,
            rng_state: rng.map(SmallRng::state),
            metadata,
        })
    }

    pub fn model(&self) -> &StateDict {
        &self.model
    }

    pub fn optimizer(&self) -> Option<&OptimizerStateDict> {
        self.optimizer.as_ref()
    }

    pub fn scheduler_step(&self) -> usize {
        self.scheduler_step
    }

    pub fn rng_state(&self) -> Option<u64> {
        self.rng_state
    }

    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    pub fn restore_model<M, E, B>(&self, model: &mut M) -> Result<()>
    where
        M: HasParameters<E, B>,
        E: FloatDType,
        B: Backend<E>,
    {
        self.model.load_module(model)
    }

    pub fn restore_training<M, O, E, B>(
        &self,
        model: &mut M,
        optimizer: &mut O,
        rng: Option<&mut SmallRng>,
    ) -> Result<()>
    where
        M: HasParameters<E, B>,
        O: OptimizerState<E, B>,
        E: FloatDType,
        B: Backend<E>,
    {
        self.model.load_module(model)?;
        let Some(state) = &self.optimizer else {
            return Err(PersistenceError::MissingOptimizerState.into());
        };
        optimizer.load_state_dict(model, state)?;
        if let (Some(rng), Some(state)) = (rng, self.rng_state) {
            rng.set_state(state);
        }
        Ok(())
    }

    pub fn save_to_path(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path).map_err(|source| PersistenceError::Io { source })?;
        let mut writer = BufWriter::new(file);
        write_header(&mut writer, CHECKPOINT_MAGIC)?;
        write_checkpoint_body(&mut writer, self)?;
        writer
            .flush()
            .map_err(|source| PersistenceError::Io { source })?;
        Ok(())
    }

    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path).map_err(|source| PersistenceError::Io { source })?;
        let mut reader = BufReader::new(file);
        read_header(&mut reader, CHECKPOINT_MAGIC, "RSTCK13")?;
        let checkpoint = read_checkpoint_body(&mut reader)?;
        ensure_eof(&mut reader)?;
        Ok(checkpoint)
    }
}

pub fn save_checkpoint<M, O, E, B>(
    path: impl AsRef<Path>,
    model: &M,
    optimizer: &O,
    scheduler_step: usize,
    rng: Option<&SmallRng>,
    metadata: BTreeMap<String, String>,
) -> Result<()>
where
    M: HasParameters<E, B>,
    O: OptimizerState<E, B>,
    E: FloatDType,
    B: Backend<E>,
{
    Checkpoint::from_training(model, optimizer, scheduler_step, rng, metadata)?.save_to_path(path)
}

pub fn load_checkpoint<M, O, E, B>(
    path: impl AsRef<Path>,
    model: &mut M,
    optimizer: &mut O,
    rng: Option<&mut SmallRng>,
) -> Result<Checkpoint>
where
    M: HasParameters<E, B>,
    O: OptimizerState<E, B>,
    E: FloatDType,
    B: Backend<E>,
{
    let checkpoint = Checkpoint::load_from_path(path)?;
    checkpoint.restore_training(model, optimizer, rng)?;
    Ok(checkpoint)
}

pub(crate) fn scalar_record<E>(name: &'static str, value: E) -> ScalarRecord
where
    E: DType,
{
    ScalarRecord::from_value(name, value)
}

pub(crate) fn optimizer_parameter_state<E>(
    meta: &ParameterSnapshot,
    tensors: Vec<TensorRecord>,
) -> Result<OptimizerParameterState>
where
    E: DType,
{
    OptimizerParameterState::new(meta.name.clone(), E::ID, meta.dims.clone(), tensors)
}

pub(crate) fn validate_optimizer_kind(
    state: &OptimizerStateDict,
    expected: OptimizerKind,
) -> Result<()> {
    if state.kind != expected {
        return Err(PersistenceError::OptimizerMismatch {
            expected: expected.as_str(),
            found: state.kind.as_str(),
        }
        .into());
    }
    Ok(())
}

pub(crate) fn collect_parameter_snapshots<M, E, B>(module: &M) -> Result<Vec<ParameterSnapshot>>
where
    M: HasParameters<E, B>,
    E: FloatDType,
    B: Backend<E>,
{
    let mut snapshots = Vec::new();
    module.visit_parameters("", &mut |name, param| {
        snapshots.push(ParameterSnapshot {
            name: name.to_owned(),
            id: param.id(),
            dtype: param.dtype(),
            dims: param.dims(),
        });
    });
    validate_unique_parameter_names(snapshots.iter().map(|param| param.name.as_str()))?;
    Ok(snapshots)
}

pub(crate) fn validate_optimizer_parameters<E, B, M>(
    module: &M,
    state: &OptimizerStateDict,
) -> Result<Vec<ParameterSnapshot>>
where
    M: HasParameters<E, B>,
    E: FloatDType,
    B: Backend<E>,
{
    if state.dtype != E::ID {
        return Err(PersistenceError::DTypeMismatch {
            name: "optimizer".to_owned(),
            expected: E::ID,
            found: state.dtype,
        }
        .into());
    }
    let snapshots = collect_parameter_snapshots(module)?;
    if snapshots.len() != state.parameters.len() {
        return Err(PersistenceError::LengthMismatch {
            name: "optimizer.parameters".to_owned(),
            expected: snapshots.len(),
            found: state.parameters.len(),
        }
        .into());
    }
    for (snapshot, saved) in snapshots.iter().zip(&state.parameters) {
        if snapshot.name != saved.name {
            return Err(PersistenceError::MissingTensor {
                name: snapshot.name.clone(),
            }
            .into());
        }
        if snapshot.dtype != saved.dtype {
            return Err(PersistenceError::DTypeMismatch {
                name: snapshot.name.clone(),
                expected: snapshot.dtype,
                found: saved.dtype,
            }
            .into());
        }
        if snapshot.dims != saved.dims {
            return Err(PersistenceError::ShapeMismatch {
                name: snapshot.name.clone(),
                expected: snapshot.dims.clone(),
                found: saved.dims.clone(),
            }
            .into());
        }
    }
    Ok(snapshots)
}

#[derive(Clone, Debug)]
pub(crate) struct ParameterSnapshot {
    pub(crate) name: String,
    pub(crate) id: ParameterId,
    pub(crate) dtype: DTypeId,
    pub(crate) dims: Vec<usize>,
}

struct NamedParameterMut<'a, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    name: String,
    param: ParameterRefMut<'a, E, B>,
}

fn validate_unique_tensor_names(records: &[TensorRecord]) -> Result<()> {
    let mut seen = HashSet::new();
    for record in records {
        if !seen.insert(record.name.as_str()) {
            return Err(PersistenceError::DuplicateTensor {
                name: record.name.clone(),
            }
            .into());
        }
    }
    Ok(())
}

fn validate_unique_parameter_names<'a>(names: impl IntoIterator<Item = &'a str>) -> Result<()> {
    let mut seen = HashSet::new();
    for name in names {
        if !seen.insert(name) {
            return Err(PersistenceError::DuplicateTensor {
                name: name.to_owned(),
            }
            .into());
        }
    }
    Ok(())
}

fn validate_record_matches(
    name: &str,
    dtype: DTypeId,
    dims: &[usize],
    record: &TensorRecord,
) -> Result<()> {
    if record.dtype != dtype {
        return Err(PersistenceError::DTypeMismatch {
            name: name.to_owned(),
            expected: dtype,
            found: record.dtype,
        }
        .into());
    }
    if record.dims != dims {
        return Err(PersistenceError::ShapeMismatch {
            name: name.to_owned(),
            expected: dims.to_vec(),
            found: record.dims.clone(),
        }
        .into());
    }
    let expected = checked_data_len(name, dtype, dims)?;
    if expected != record.data.len() {
        return Err(PersistenceError::LengthMismatch {
            name: name.to_owned(),
            expected,
            found: record.data.len(),
        }
        .into());
    }
    Ok(())
}

fn checked_numel(name: &str, dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1usize, |acc, &dim| {
        acc.checked_mul(dim).ok_or_else(|| {
            PersistenceError::SizeOverflow {
                name: name.to_owned(),
            }
            .into()
        })
    })
}

fn checked_data_len(name: &str, dtype: DTypeId, dims: &[usize]) -> Result<usize> {
    checked_numel(name, dims)?
        .checked_mul(dtype_size(dtype))
        .ok_or_else(|| {
            PersistenceError::SizeOverflow {
                name: name.to_owned(),
            }
            .into()
        })
}

fn dtype_size(dtype: DTypeId) -> usize {
    match dtype {
        DTypeId::F16 | DTypeId::BF16 => 2,
        DTypeId::F32 => 4,
        DTypeId::F64 | DTypeId::I64 => 8,
    }
}

fn dtype_code(dtype: DTypeId) -> u8 {
    match dtype {
        DTypeId::F16 => 1,
        DTypeId::BF16 => 2,
        DTypeId::F32 => 3,
        DTypeId::F64 => 4,
        DTypeId::I64 => 5,
    }
}

fn dtype_from_code(code: u8) -> Result<DTypeId> {
    match code {
        1 => Ok(DTypeId::F16),
        2 => Ok(DTypeId::BF16),
        3 => Ok(DTypeId::F32),
        4 => Ok(DTypeId::F64),
        5 => Ok(DTypeId::I64),
        _ => Err(PersistenceError::InvalidDType { code }.into()),
    }
}

fn write_header<W: Write>(writer: &mut W, magic: &[u8; 8]) -> Result<()> {
    writer
        .write_all(magic)
        .map_err(|source| PersistenceError::Io { source })?;
    write_u32(writer, VERSION)
}

fn read_header<R: Read>(reader: &mut R, magic: &[u8; 8], expected: &'static str) -> Result<()> {
    let mut found = [0u8; 8];
    reader
        .read_exact(&mut found)
        .map_err(|source| PersistenceError::Io { source })?;
    if &found != magic {
        return Err(PersistenceError::InvalidMagic {
            expected,
            found: found.to_vec(),
        }
        .into());
    }
    let version = read_u32(reader)?;
    if version != VERSION {
        return Err(PersistenceError::UnsupportedVersion { version }.into());
    }
    Ok(())
}

fn write_state_dict_body<W: Write>(writer: &mut W, state: &StateDict) -> Result<()> {
    write_len(writer, state.records.len())?;
    for record in &state.records {
        write_tensor_record(writer, record)?;
    }
    Ok(())
}

fn read_state_dict_body<R: Read>(reader: &mut R) -> Result<StateDict> {
    let count = read_count(reader, "tensor record count", MAX_RECORDS)?;
    let mut records = Vec::with_capacity(count);
    for _ in 0..count {
        records.push(read_tensor_record(reader)?);
    }
    StateDict::new(records)
}

fn write_tensor_record<W: Write>(writer: &mut W, record: &TensorRecord) -> Result<()> {
    write_string(writer, &record.name)?;
    write_u8(writer, dtype_code(record.dtype))?;
    write_len(writer, record.dims.len())?;
    for &dim in &record.dims {
        write_len(writer, dim)?;
    }
    write_len(writer, record.data.len())?;
    writer
        .write_all(&record.data)
        .map_err(|source| PersistenceError::Io { source })?;
    Ok(())
}

fn read_tensor_record<R: Read>(reader: &mut R) -> Result<TensorRecord> {
    let name = read_string(reader)?;
    let dtype = dtype_from_code(read_u8(reader)?)?;
    let rank = read_count(reader, "rank", MAX_RANK)?;
    let mut dims = Vec::with_capacity(rank);
    for _ in 0..rank {
        dims.push(read_count(reader, "dimension", usize::MAX as u64)?);
    }
    let data_len = read_count(reader, "tensor data", MAX_TENSOR_BYTES)?;
    let expected = checked_data_len(&name, dtype, &dims)?;
    if data_len != expected {
        return Err(PersistenceError::LengthMismatch {
            name,
            expected,
            found: data_len,
        }
        .into());
    }
    let mut data = vec![0u8; data_len];
    reader
        .read_exact(&mut data)
        .map_err(|source| PersistenceError::Io { source })?;
    TensorRecord::from_bytes(name, dtype, dims, data)
}

fn write_optimizer_state_body<W: Write>(writer: &mut W, state: &OptimizerStateDict) -> Result<()> {
    write_u8(writer, state.kind.code())?;
    write_u8(writer, dtype_code(state.dtype))?;
    write_len(writer, state.step)?;
    write_len(writer, state.hyperparameters.len())?;
    for hyper in &state.hyperparameters {
        write_string(writer, &hyper.name)?;
        write_u8(writer, dtype_code(hyper.dtype))?;
        write_len(writer, hyper.data.len())?;
        writer
            .write_all(&hyper.data)
            .map_err(|source| PersistenceError::Io { source })?;
    }

    write_len(writer, state.parameters.len())?;
    for parameter in &state.parameters {
        write_string(writer, &parameter.name)?;
        write_u8(writer, dtype_code(parameter.dtype))?;
        write_len(writer, parameter.dims.len())?;
        for &dim in &parameter.dims {
            write_len(writer, dim)?;
        }
        write_len(writer, parameter.tensors.len())?;
        for tensor in &parameter.tensors {
            write_tensor_record(writer, tensor)?;
        }
    }
    Ok(())
}

fn read_optimizer_state_body<R: Read>(reader: &mut R) -> Result<OptimizerStateDict> {
    let kind = OptimizerKind::from_code(read_u8(reader)?)?;
    let dtype = dtype_from_code(read_u8(reader)?)?;
    let step = read_count(reader, "optimizer step", usize::MAX as u64)?;
    let hyper_count = read_count(reader, "optimizer hyperparameter count", MAX_RECORDS)?;
    let mut hyperparameters = Vec::with_capacity(hyper_count);
    for _ in 0..hyper_count {
        let name = read_string(reader)?;
        let hyper_dtype = dtype_from_code(read_u8(reader)?)?;
        let len = read_count(reader, "hyperparameter", dtype_size(hyper_dtype) as u64)?;
        let mut data = vec![0u8; len];
        reader
            .read_exact(&mut data)
            .map_err(|source| PersistenceError::Io { source })?;
        hyperparameters.push(ScalarRecord::from_bytes(name, hyper_dtype, data)?);
    }

    let parameter_count = read_count(reader, "optimizer parameter count", MAX_RECORDS)?;
    let mut parameters = Vec::with_capacity(parameter_count);
    for _ in 0..parameter_count {
        let name = read_string(reader)?;
        let parameter_dtype = dtype_from_code(read_u8(reader)?)?;
        let rank = read_count(reader, "rank", MAX_RANK)?;
        let mut dims = Vec::with_capacity(rank);
        for _ in 0..rank {
            dims.push(read_count(reader, "dimension", usize::MAX as u64)?);
        }
        let tensor_count = read_count(reader, "optimizer tensor count", MAX_RECORDS)?;
        let mut tensors = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            tensors.push(read_tensor_record(reader)?);
        }
        parameters.push(OptimizerParameterState::new(
            name,
            parameter_dtype,
            dims,
            tensors,
        )?);
    }

    OptimizerStateDict::new(kind, dtype, step, hyperparameters, parameters)
}

fn write_checkpoint_body<W: Write>(writer: &mut W, checkpoint: &Checkpoint) -> Result<()> {
    write_len(writer, checkpoint.metadata.len())?;
    for (key, value) in &checkpoint.metadata {
        write_string(writer, key)?;
        write_string(writer, value)?;
    }
    write_len(writer, checkpoint.scheduler_step)?;
    match checkpoint.rng_state {
        Some(state) => {
            write_u8(writer, 1)?;
            write_u64(writer, state)?;
        }
        None => write_u8(writer, 0)?,
    }
    write_state_dict_body(writer, &checkpoint.model)?;
    match &checkpoint.optimizer {
        Some(state) => {
            write_u8(writer, 1)?;
            write_optimizer_state_body(writer, state)?;
        }
        None => write_u8(writer, 0)?,
    }
    Ok(())
}

fn read_checkpoint_body<R: Read>(reader: &mut R) -> Result<Checkpoint> {
    let metadata_count = read_count(reader, "metadata count", MAX_RECORDS)?;
    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = read_string(reader)?;
        let value = read_string(reader)?;
        metadata.insert(key, value);
    }
    let scheduler_step = read_count(reader, "scheduler step", usize::MAX as u64)?;
    let rng_state = match read_u8(reader)? {
        0 => None,
        1 => Some(read_u64(reader)?),
        _ => {
            return Err(PersistenceError::InvalidFormat {
                reason: "invalid rng-state flag",
            }
            .into());
        }
    };
    let model = read_state_dict_body(reader)?;
    let optimizer = match read_u8(reader)? {
        0 => None,
        1 => Some(read_optimizer_state_body(reader)?),
        _ => {
            return Err(PersistenceError::InvalidFormat {
                reason: "invalid optimizer-state flag",
            }
            .into());
        }
    };
    Ok(Checkpoint {
        model,
        optimizer,
        scheduler_step,
        rng_state,
        metadata,
    })
}

fn write_string<W: Write>(writer: &mut W, value: &str) -> Result<()> {
    write_len(writer, value.len())?;
    writer
        .write_all(value.as_bytes())
        .map_err(|source| PersistenceError::Io { source })?;
    Ok(())
}

fn read_string<R: Read>(reader: &mut R) -> Result<String> {
    let len = read_count(reader, "string", MAX_STRING_BYTES)?;
    let mut bytes = vec![0u8; len];
    reader
        .read_exact(&mut bytes)
        .map_err(|source| PersistenceError::Io { source })?;
    String::from_utf8(bytes).map_err(|source| PersistenceError::InvalidUtf8 { source }.into())
}

fn write_len<W: Write>(writer: &mut W, value: usize) -> Result<()> {
    write_u64(writer, value as u64)
}

fn write_u8<W: Write>(writer: &mut W, value: u8) -> Result<()> {
    writer
        .write_all(&[value])
        .map_err(|source| PersistenceError::Io { source })?;
    Ok(())
}

fn write_u32<W: Write>(writer: &mut W, value: u32) -> Result<()> {
    writer
        .write_all(&value.to_le_bytes())
        .map_err(|source| PersistenceError::Io { source })?;
    Ok(())
}

fn write_u64<W: Write>(writer: &mut W, value: u64) -> Result<()> {
    writer
        .write_all(&value.to_le_bytes())
        .map_err(|source| PersistenceError::Io { source })?;
    Ok(())
}

fn read_u8<R: Read>(reader: &mut R) -> Result<u8> {
    let mut bytes = [0u8; 1];
    reader
        .read_exact(&mut bytes)
        .map_err(|source| PersistenceError::Io { source })?;
    Ok(bytes[0])
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32> {
    let mut bytes = [0u8; 4];
    reader
        .read_exact(&mut bytes)
        .map_err(|source| PersistenceError::Io { source })?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64> {
    let mut bytes = [0u8; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(|source| PersistenceError::Io { source })?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_count<R: Read>(reader: &mut R, field: &'static str, max: u64) -> Result<usize> {
    let len = read_u64(reader)?;
    if len > max {
        return Err(PersistenceError::AllocationTooLarge { field, len, max }.into());
    }
    usize::try_from(len).map_err(|_| {
        PersistenceError::AllocationTooLarge {
            field,
            len,
            max: usize::MAX as u64,
        }
        .into()
    })
}

fn ensure_eof<R: Read>(reader: &mut R) -> Result<()> {
    let mut byte = [0u8; 1];
    match reader.read(&mut byte) {
        Ok(0) => Ok(()),
        Ok(_) => Err(PersistenceError::UnexpectedTrailingBytes.into()),
        Err(source) => Err(PersistenceError::Io { source }.into()),
    }
}

impl<S, E, B> Tensor<S, E, B>
where
    S: ShapeSpec,
    E: DType,
    B: Backend<E>,
{
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        save_tensor(path, self)
    }
}

impl<S, E> Tensor<S, E, Cpu>
where
    S: ShapeSpec,
    E: DType,
{
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        load_tensor(path)
    }
}
