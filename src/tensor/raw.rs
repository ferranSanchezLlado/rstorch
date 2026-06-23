use crate::backend::{Backend, Cpu};
use crate::dtype::{DType, DTypeId};
use crate::error::{Error, Result, ShapeError};
use crate::shape::{Layout, Shape};
use std::marker::PhantomData;
use std::sync::Arc;

pub(crate) struct RawTensor<E = f32, B = Cpu>
where
    E: DType,
    B: Backend<E>,
{
    storage: Arc<B::Storage>,
    device: B::Device,
    layout: Layout,
    _dtype: PhantomData<E>,
    _backend: PhantomData<B>,
}

impl<E, B> RawTensor<E, B>
where
    E: DType,
    B: Backend<E>,
{
    pub(crate) fn zeros(shape: Shape) -> Result<Self> {
        let device = B::default_device().map_err(Error::backend)?;
        Self::zeros_on(device, shape)
    }

    pub(crate) fn ones(shape: Shape) -> Result<Self> {
        let device = B::default_device().map_err(Error::backend)?;
        Self::ones_on(device, shape)
    }

    pub(crate) fn from_vec(data: Vec<E>, shape: Shape) -> Result<Self> {
        let device = B::default_device().map_err(Error::backend)?;
        Self::from_vec_on(device, data, shape)
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
            storage: Arc::new(storage),
            device,
            layout: Layout::contiguous(shape)?,
            _dtype: PhantomData,
            _backend: PhantomData,
        })
    }

    pub(crate) fn from_shared_storage(
        storage: Arc<B::Storage>,
        device: B::Device,
        layout: Layout,
    ) -> Result<Self> {
        layout.validate_in_storage(B::storage_len(&storage))?;
        Ok(Self {
            storage,
            device,
            layout,
            _dtype: PhantomData,
            _backend: PhantomData,
        })
    }

    pub(crate) fn view_with_layout(&self, layout: Layout) -> Result<Self> {
        Self::from_shared_storage(Arc::clone(&self.storage), self.device.clone(), layout)
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
        self.layout.numel()
    }

    pub(crate) fn storage(&self) -> &B::Storage {
        &self.storage
    }

    pub(crate) fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    pub(crate) fn shares_storage_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }

    pub(crate) fn has_storage_view(&self) -> bool {
        Arc::strong_count(&self.storage) > 1
    }

    pub(crate) fn to_vec(&self) -> Result<Vec<E>> {
        let physical = B::to_vec(&self.device, &self.storage).map_err(Error::backend)?;
        self.layout
            .storage_positions()?
            .into_iter()
            .map(|position| {
                physical.get(position).copied().ok_or_else(|| {
                    ShapeError::LayoutOutOfBounds {
                        offset: position,
                        storage_len: physical.len(),
                    }
                    .into()
                })
            })
            .collect()
    }
}
