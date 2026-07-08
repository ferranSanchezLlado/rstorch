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

impl<E, B> Clone for RawTensor<E, B>
where
    E: DType,
    B: Backend<E>,
{
    fn clone(&self) -> Self {
        Self {
            storage: Arc::clone(&self.storage),
            device: self.device.clone(),
            layout: self.layout.clone(),
            _dtype: PhantomData,
            _backend: PhantomData,
        }
    }
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
                op: "raw_from_vec_on",
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
                op: "raw_from_storage_on",
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
        let mut physical = B::to_vec(&self.device, &self.storage).map_err(Error::backend)?;

        // Contiguous layouts read storage positions `0..numel` in order, so
        // the physical buffer already is the logical value order and the
        // per-position gather below would only re-copy it.
        if self.layout.is_contiguous() {
            let numel = self.layout.numel();
            if physical.len() < numel {
                return Err(ShapeError::LayoutOutOfBounds {
                    offset: physical.len(),
                    storage_len: physical.len(),
                }
                .into());
            }
            physical.truncate(numel);
            return Ok(physical);
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor2D;
    use crate::backend::Cpu;
    use crate::error::Error;
    use crate::tensor::test_support::FailingBackend;

    #[test]
    fn backend_built_storage_lengths_match_numel() {
        let zeros = Tensor2D::<2, 3>::zeros().unwrap();
        assert_eq!(<Cpu as Backend<f32>>::storage_len(zeros.raw().storage()), 6);
        assert_eq!(zeros.numel(), 6);

        let ones = Tensor2D::<2, 3>::ones().unwrap();
        assert_eq!(<Cpu as Backend<f32>>::storage_len(ones.raw().storage()), 6);
        assert_eq!(ones.numel(), 6);
    }

    #[test]
    fn backend_errors_are_boxed_sources() {
        let err = match RawTensor::<f32, FailingBackend>::zeros(Shape::known([2])) {
            Ok(_) => panic!("expected backend error"),
            Err(err) => err,
        };
        assert!(matches!(err, Error::Backend(_)));
        assert!(std::error::Error::source(&err).is_some());
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    mod metal_tests {
        use super::*;
        use crate::backend::{Metal, MetalDevice};

        fn device() -> Option<MetalDevice> {
            <Metal as Backend<f32>>::default_device().ok()
        }

        #[test]
        fn metal_zeros_and_ones_have_expected_values_and_lengths() {
            let Some(device) = device() else {
                return;
            };

            let zeros =
                RawTensor::<f32, Metal>::zeros_on(device.clone(), Shape::known([3])).unwrap();
            assert_eq!(<Metal as Backend<f32>>::storage_len(zeros.storage()), 3);
            assert_eq!(zeros.to_vec().unwrap(), vec![0.0, 0.0, 0.0]);

            let ones = RawTensor::<f32, Metal>::ones_on(device, Shape::known([3])).unwrap();
            assert_eq!(<Metal as Backend<f32>>::storage_len(ones.storage()), 3);
            assert_eq!(ones.to_vec().unwrap(), vec![1.0, 1.0, 1.0]);
        }
    }
}
