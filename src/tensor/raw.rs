use crate::backend::{Backend, Cpu};
use crate::dtype::{DType, DTypeId};
use crate::error::{Error, Result, ShapeError};
use crate::shape::{Layout, Shape};
use std::marker::PhantomData;

pub(crate) struct RawTensor<E = f32, B = Cpu>
where
    E: DType,
    B: Backend<E>,
{
    storage: B::Storage,
    device: B::Device,
    layout: Layout,
    numel: usize,
    _dtype: PhantomData<E>,
    _backend: PhantomData<B>,
}

impl<E, B> RawTensor<E, B>
where
    E: DType,
    B: Backend<E>,
{
    pub(crate) fn zeros(shape: Shape) -> Result<Self> {
        Self::zeros_on(B::default_device(), shape)
    }

    pub(crate) fn ones(shape: Shape) -> Result<Self> {
        Self::ones_on(B::default_device(), shape)
    }

    pub(crate) fn from_vec(data: Vec<E>, shape: Shape) -> Result<Self> {
        Self::from_vec_on(B::default_device(), data, shape)
    }

    pub(crate) fn zeros_on(device: B::Device, shape: Shape) -> Result<Self> {
        let numel = shape.numel()?;
        let storage = B::zeros(&device, numel).map_err(Error::backend)?;
        Self::from_storage_on(device, storage, shape)
    }

    pub(crate) fn ones_on(device: B::Device, shape: Shape) -> Result<Self> {
        let numel = shape.numel()?;
        let storage = B::ones(&device, numel).map_err(Error::backend)?;
        Self::from_storage_on(device, storage, shape)
    }

    pub(crate) fn from_vec_on(device: B::Device, data: Vec<E>, shape: Shape) -> Result<Self> {
        let expected = shape.numel()?;
        if data.len() != expected {
            return Err(ShapeError::LengthMismatch {
                expected,
                found: data.len(),
            }
            .into());
        }

        let storage = B::from_vec(&device, data).map_err(Error::backend)?;
        Self::from_storage_on(device, storage, shape)
    }

    pub(crate) fn from_storage_on(
        device: B::Device,
        storage: B::Storage,
        shape: Shape,
    ) -> Result<Self> {
        let numel = shape.numel()?;
        let found = B::storage_len(&storage);
        if found != numel {
            return Err(ShapeError::LengthMismatch {
                expected: numel,
                found,
            }
            .into());
        }

        Ok(Self {
            storage,
            device,
            layout: Layout::contiguous(shape),
            numel,
            _dtype: PhantomData,
            _backend: PhantomData,
        })
    }

    pub(crate) fn dtype(&self) -> DTypeId {
        E::ID
    }

    pub(crate) fn device(&self) -> &B::Device {
        &self.device
    }

    pub(crate) fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    pub(crate) fn layout(&self) -> &Layout {
        &self.layout
    }

    pub(crate) fn rank(&self) -> usize {
        self.shape().rank()
    }

    pub(crate) fn numel(&self) -> usize {
        self.numel
    }

    pub(crate) fn storage(&self) -> &B::Storage {
        &self.storage
    }

    pub(crate) fn to_vec(&self) -> Result<Vec<E>> {
        B::to_vec(&self.device, &self.storage).map_err(Error::backend)
    }
}
