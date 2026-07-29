use super::{LeafContract, LeafKind, TypedBuffer, TypedParam};
use crate::typed::device::validate_binding;
use crate::typed::ops::{WithElement, WithPlacement};
use crate::typed::sealed::TypedTensor as SealedTypedTensor;
use crate::typed::tensor::checked_wrap;
use crate::typed::{DeviceCtx, FloatElement, Placement, TypedTensor};
use crate::{Element, Grads, Result};
use std::any::TypeId;
use std::marker::PhantomData;
use std::sync::Arc;

impl<T> TypedParam<T>
where
    T: TypedTensor,
    T::Elem: FloatElement,
{
    /// Creates a parameter from a detached, canonically bound typed value.
    pub fn new(value: T) -> Result<Self> {
        validate_binding::<T::Placement>(value.binding(), "TypedParam::new")?;
        let binding = Arc::clone(value.binding());
        let value = value.dynamic().detach();
        checked_wrap::<T>(value.clone(), Arc::clone(&binding), "TypedParam::new")?;
        Ok(Self {
            runtime: crate::nn::Param::new(value),
            binding,
            marker: PhantomData,
        })
    }

    /// Returns the cached traced leaf when `mode` records and the parameter is unfrozen.
    pub fn get(&self, mode: super::Mode) -> Result<T> {
        checked_wrap::<T>(
            self.runtime.get(mode),
            Arc::clone(&self.binding),
            "TypedParam::get",
        )
    }

    /// Returns the detached parameter value.
    pub fn value(&self) -> Result<T> {
        checked_wrap::<T>(
            self.runtime.value().clone(),
            Arc::clone(&self.binding),
            "TypedParam::value",
        )
    }

    /// Replaces the value without changing gradient identity or freeze state.
    pub fn set(&mut self, value: T) -> Result<()> {
        validate_binding::<T::Placement>(value.binding(), "TypedParam::set")?;
        if !Arc::ptr_eq(value.binding(), &self.binding) {
            return Err(crate::Error::InvalidArg {
                op: "TypedParam::set",
                msg: "value does not carry the parameter's canonical placement binding".into(),
            });
        }
        let value = value.dynamic().detach();
        checked_wrap::<T>(value.clone(), Arc::clone(&self.binding), "TypedParam::set")?;
        self.runtime.set(value)
    }

    /// Returns this parameter's gradient, checked and wrapped as `T`.
    pub fn grad_from(&self, grads: &Grads) -> Result<T> {
        checked_wrap::<T>(
            grads.wrt(&self.runtime)?,
            Arc::clone(&self.binding),
            "TypedParam::grad_from",
        )
    }

    /// Excludes this parameter from tracing and optimizer completeness checks.
    pub fn freeze(&mut self) {
        self.runtime.freeze();
    }

    /// Restores tracing and optimizer participation.
    pub fn unfreeze(&mut self) {
        self.runtime.unfreeze();
    }

    /// Returns whether this parameter is frozen.
    pub fn is_frozen(&self) -> bool {
        self.runtime.is_frozen()
    }

    /// Consumes and moves this parameter while preserving its runtime identity.
    pub fn to_device<Q>(
        self,
        target: &DeviceCtx<Q>,
    ) -> Result<TypedParam<<T as WithPlacement<Q>>::Output>>
    where
        Q: Placement,
        T: WithPlacement<Q>,
        <<T as WithPlacement<Q>>::Output as TypedTensor>::Elem: FloatElement,
    {
        validate_binding::<T::Placement>(&self.binding, "TypedParam::to_device")?;
        validate_binding::<Q>(target.binding(), "TypedParam::to_device")?;
        let replacement = self.runtime.value().to_device(&target.device())?.detach();
        checked_wrap::<<T as WithPlacement<Q>>::Output>(
            replacement.clone(),
            Arc::clone(target.binding()),
            "TypedParam::to_device",
        )?;
        let mut runtime = self.runtime;
        runtime.set(replacement)?;
        Ok(TypedParam {
            runtime,
            binding: Arc::clone(target.binding()),
            marker: PhantomData,
        })
    }

    /// Consumes and converts this parameter while preserving its runtime identity.
    pub fn to_dtype<F>(self) -> Result<TypedParam<<T as WithElement<F>>::Output>>
    where
        F: FloatElement,
        T: WithElement<F>,
    {
        validate_binding::<T::Placement>(&self.binding, "TypedParam::to_dtype")?;
        let replacement = self.runtime.value().to_dtype(F::DTYPE)?.detach();
        checked_wrap::<<T as WithElement<F>>::Output>(
            replacement.clone(),
            Arc::clone(&self.binding),
            "TypedParam::to_dtype",
        )?;
        let mut runtime = self.runtime;
        runtime.set(replacement)?;
        Ok(TypedParam {
            runtime,
            binding: self.binding,
            marker: PhantomData,
        })
    }

    pub(super) fn validate(&self, op: &'static str) -> Result<()> {
        checked_wrap::<T>(self.runtime.value().clone(), Arc::clone(&self.binding), op).map(drop)
    }

    pub(super) fn contract(&self) -> LeafContract {
        contract::<T>(LeafKind::Param, Arc::clone(&self.binding))
    }
}

impl<T: TypedTensor> TypedBuffer<T> {
    /// Creates persistent state from a detached, canonically bound value.
    pub fn new(value: T) -> Result<Self> {
        validate_binding::<T::Placement>(value.binding(), "TypedBuffer::new")?;
        let binding = Arc::clone(value.binding());
        let runtime = value.dynamic().detach();
        checked_wrap::<T>(runtime.clone(), Arc::clone(&binding), "TypedBuffer::new")?;
        Ok(Self {
            runtime,
            binding,
            marker: PhantomData,
        })
    }

    /// Returns the persistent value as its exact typed tensor.
    pub fn value(&self) -> Result<T> {
        checked_wrap::<T>(
            self.runtime.clone(),
            Arc::clone(&self.binding),
            "TypedBuffer::value",
        )
    }

    /// Replaces this buffer with a detached value of the exact same type.
    pub fn set(&mut self, value: T) -> Result<()> {
        validate_binding::<T::Placement>(value.binding(), "TypedBuffer::set")?;
        if !Arc::ptr_eq(value.binding(), &self.binding) {
            return Err(crate::Error::InvalidArg {
                op: "TypedBuffer::set",
                msg: "value does not carry the buffer's canonical placement binding".into(),
            });
        }
        let runtime = value.dynamic().detach();
        checked_wrap::<T>(
            runtime.clone(),
            Arc::clone(&self.binding),
            "TypedBuffer::set",
        )?;
        self.runtime = runtime;
        Ok(())
    }

    /// Consumes and moves this buffer to `Q`.
    pub fn to_device<Q>(
        self,
        target: &DeviceCtx<Q>,
    ) -> Result<TypedBuffer<<T as WithPlacement<Q>>::Output>>
    where
        Q: Placement,
        T: WithPlacement<Q>,
    {
        validate_binding::<T::Placement>(&self.binding, "TypedBuffer::to_device")?;
        validate_binding::<Q>(target.binding(), "TypedBuffer::to_device")?;
        let runtime = self.runtime.to_device(&target.device())?.detach();
        checked_wrap::<<T as WithPlacement<Q>>::Output>(
            runtime.clone(),
            Arc::clone(target.binding()),
            "TypedBuffer::to_device",
        )?;
        Ok(TypedBuffer {
            runtime,
            binding: Arc::clone(target.binding()),
            marker: PhantomData,
        })
    }

    /// Consumes and converts this buffer's element type.
    pub fn to_dtype<F>(self) -> Result<TypedBuffer<<T as WithElement<F>>::Output>>
    where
        F: Element,
        T: WithElement<F>,
    {
        validate_binding::<T::Placement>(&self.binding, "TypedBuffer::to_dtype")?;
        let runtime = self.runtime.to_dtype(F::DTYPE)?.detach();
        checked_wrap::<<T as WithElement<F>>::Output>(
            runtime.clone(),
            Arc::clone(&self.binding),
            "TypedBuffer::to_dtype",
        )?;
        Ok(TypedBuffer {
            runtime,
            binding: self.binding,
            marker: PhantomData,
        })
    }

    pub(super) fn validate(&self, op: &'static str) -> Result<()> {
        checked_wrap::<T>(self.runtime.clone(), Arc::clone(&self.binding), op).map(drop)
    }

    pub(super) fn contract(&self) -> LeafContract {
        contract::<T>(LeafKind::Buffer, Arc::clone(&self.binding))
    }
}

fn contract<T: TypedTensor>(kind: LeafKind, binding: Arc<super::DeviceBinding>) -> LeafContract {
    LeafContract {
        kind,
        rank: T::RANK,
        markers: <T as SealedTypedTensor>::MARKERS,
        dtype: T::Elem::DTYPE,
        placement: TypeId::of::<T::Placement>(),
        binding,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed::sealed::{DeviceBinding, TypedTensor as SealedTypedTensor};
    use crate::typed::{Cpu, Tensor1};
    use crate::{Device, Error};

    fn value(data: [f32; 2]) -> Tensor1<2> {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        Tensor1::from_vec(data.to_vec(), [2], &ctx).unwrap()
    }

    fn square_sum(param: &TypedParam<Tensor1<2>>) -> crate::Tensor {
        let value = param.get(super::super::Mode::TRAIN).unwrap();
        value
            .as_dynamic()
            .mul(value.as_dynamic())
            .unwrap()
            .sum_all()
            .unwrap()
    }

    #[test]
    fn parameter_detaches_and_grad_from_uses_the_stable_runtime_key() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let traced = value([1.0, 2.0]).traced().unwrap();
        let mut param = TypedParam::new(traced).unwrap();
        assert!(param.value().unwrap().as_dynamic().backward().is_err());

        let old_loss = square_sum(&param);
        param.set(value([3.0, 4.0])).unwrap();
        let old_grads = old_loss.backward().unwrap();
        assert_eq!(
            param.grad_from(&old_grads).unwrap().to_vec().unwrap(),
            vec![2.0, 4.0]
        );

        param.freeze();
        let param = param.to_dtype::<f32>().unwrap();
        assert!(param.is_frozen());
        let param = param.to_device(&ctx).unwrap();
        assert!(param.is_frozen());
        assert_eq!(param.value().unwrap().to_vec().unwrap(), vec![3.0, 4.0]);
        assert_eq!(
            param.grad_from(&old_grads).unwrap().to_vec().unwrap(),
            vec![2.0, 4.0],
            "consuming conversions must preserve the runtime gradient key"
        );
    }

    #[test]
    fn set_rebuilds_the_cached_leaf_and_freeze_state_is_explicit() {
        let mut param = TypedParam::new(value([1.0, 2.0])).unwrap();
        let first = param.get(super::super::Mode::TRAIN).unwrap();
        param.set(value([5.0, 6.0])).unwrap();
        let second = param.get(super::super::Mode::TRAIN).unwrap();
        assert_eq!(first.to_vec().unwrap(), vec![1.0, 2.0]);
        assert_eq!(second.to_vec().unwrap(), vec![5.0, 6.0]);
        param.freeze();
        assert!(param.is_frozen());
        assert!(
            param
                .get(super::super::Mode::TRAIN)
                .unwrap()
                .as_dynamic()
                .backward()
                .is_err()
        );
        param.unfreeze();
        assert!(!param.is_frozen());
    }

    #[test]
    fn forged_noncanonical_bindings_are_rejected_at_every_entry_boundary() {
        let runtime = crate::Tensor::from_vec(vec![1.0, 2.0], [2], &Device::Cpu).unwrap();
        let forged = Arc::new(DeviceBinding {
            device: Device::Cpu,
        });
        let typed =
            <Tensor1<2> as SealedTypedTensor>::trusted_from_validated(runtime, Arc::clone(&forged));
        assert!(matches!(
            TypedParam::new(typed),
            Err(Error::InvalidArg {
                op: "TypedParam::new",
                ..
            })
        ));

        let runtime = crate::Tensor::from_vec(vec![1.0, 2.0], [2], &Device::Cpu).unwrap();
        let typed = <Tensor1<2> as SealedTypedTensor>::trusted_from_validated(runtime, forged);
        assert!(matches!(
            TypedBuffer::new(typed),
            Err(Error::InvalidArg {
                op: "TypedBuffer::new",
                ..
            })
        ));
    }

    #[test]
    fn buffers_detach_set_and_consume_retype() {
        let traced = value([1.0, 2.0]).traced().unwrap();
        let mut buffer = TypedBuffer::new(traced).unwrap();
        assert!(buffer.value().unwrap().as_dynamic().backward().is_err());
        buffer.set(value([7.0, 8.0])).unwrap();
        let buffer = buffer.to_dtype::<f32>().unwrap();
        assert_eq!(buffer.value().unwrap().to_vec().unwrap(), vec![7.0, 8.0]);
    }
}
