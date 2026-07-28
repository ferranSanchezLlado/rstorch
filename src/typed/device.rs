use super::{DeviceBinding, DeviceCtx, Placement};
use crate::{DType, Device, Error, Result, Tensor};
use std::any::TypeId;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

type Registry = HashMap<TypeId, Arc<DeviceBinding>>;

fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn lock_registry() -> MutexGuard<'static, Registry> {
    registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn invalid_binding(op: &'static str, marker: &'static str) -> Error {
    Error::InvalidArg {
        op,
        msg: format!("placement marker {marker} does not carry its canonical device binding"),
    }
}

impl<P: Placement> DeviceCtx<P> {
    /// Binds this logical placement to one process-lifetime runtime device.
    ///
    /// The first successful call fixes `P` to `device`. Repeating that binding
    /// is idempotent, while attempting to bind `P` to another device returns
    /// [`Error::InvalidArg`]. Marker policy and device availability are checked
    /// before the binding is committed.
    pub fn bind(device: Device) -> Result<Self> {
        let marker = TypeId::of::<P>();
        {
            let bindings = lock_registry();
            if let Some(binding) = bindings.get(&marker) {
                return context_for_existing::<P>(binding, device);
            }
        }

        // Placement policies are downstream code. Never invoke one while the
        // registry lock is held: a policy may bind another marker.
        P::validate_device(device)?;
        let _probe = Tensor::zeros((), DType::F32, &device)?;

        let mut bindings = lock_registry();
        if let Some(binding) = bindings.get(&marker) {
            return context_for_existing::<P>(binding, device);
        }

        let binding = Arc::new(DeviceBinding { device });
        bindings.insert(marker, Arc::clone(&binding));
        Ok(Self {
            binding,
            marker: PhantomData,
        })
    }

    /// Returns the runtime device fixed for this logical placement.
    pub fn device(&self) -> Device {
        self.binding.device
    }

    pub(crate) fn binding(&self) -> &Arc<DeviceBinding> {
        &self.binding
    }
}

fn context_for_existing<P: Placement>(
    binding: &Arc<DeviceBinding>,
    device: Device,
) -> Result<DeviceCtx<P>> {
    if binding.device != device {
        return Err(Error::InvalidArg {
            op: "DeviceCtx::bind",
            msg: format!(
                "placement marker {} is already bound to {}, not {device}",
                std::any::type_name::<P>(),
                binding.device
            ),
        });
    }

    Ok(DeviceCtx {
        binding: Arc::clone(binding),
        marker: PhantomData,
    })
}

impl DeviceCtx<super::Cpu> {
    /// Returns the canonical context for the built-in CPU placement.
    pub fn cpu() -> Result<Self> {
        Self::bind(Device::Cpu)
    }
}

pub(crate) fn validate_binding<P: Placement>(
    binding: &Arc<DeviceBinding>,
    op: &'static str,
) -> Result<()> {
    let bindings = lock_registry();
    let canonical = bindings
        .get(&TypeId::of::<P>())
        .ok_or_else(|| invalid_binding(op, std::any::type_name::<P>()))?;

    if !Arc::ptr_eq(canonical, binding) {
        return Err(invalid_binding(op, std::any::type_name::<P>()));
    }
    Ok(())
}

pub(crate) fn checked_relabel_binding<P: Placement, Q: Placement>(
    source: Arc<DeviceBinding>,
    target: &DeviceCtx<Q>,
) -> Result<Arc<DeviceBinding>> {
    validate_binding::<P>(&source, "relabel")?;
    validate_binding::<Q>(target.binding(), "relabel")?;

    if source.device != target.device() {
        return Err(Error::DeviceMismatch {
            op: "relabel",
            expected: source.device,
            got: target.device(),
        });
    }

    Ok(Arc::clone(target.binding()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed::{Cpu, Placement};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    struct SameDevice;
    impl Placement for SameDevice {}

    #[test]
    fn same_marker_same_device_returns_canonical_binding() {
        let first = DeviceCtx::<SameDevice>::bind(Device::Cpu).unwrap();
        let second = DeviceCtx::<SameDevice>::bind(Device::Cpu).unwrap();

        assert_eq!(first.device(), Device::Cpu);
        assert!(Arc::ptr_eq(first.binding(), second.binding()));
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn same_marker_different_device_is_rejected_before_probe() {
        struct DifferentDevice;
        impl Placement for DifferentDevice {}

        DeviceCtx::<DifferentDevice>::bind(Device::Cpu).unwrap();
        assert!(matches!(
            DeviceCtx::<DifferentDevice>::bind(Device::Metal(usize::MAX)),
            Err(Error::InvalidArg {
                op: "DeviceCtx::bind",
                ..
            })
        ));
    }

    static ACCEPT_FAILED_BIND: AtomicBool = AtomicBool::new(false);

    struct FailedFirstBind;
    impl Placement for FailedFirstBind {
        fn validate_device(_device: Device) -> Result<()> {
            if ACCEPT_FAILED_BIND.load(Ordering::SeqCst) {
                Ok(())
            } else {
                Err(Error::InvalidArg {
                    op: "DeviceCtx::bind",
                    msg: "intentional first-bind failure".into(),
                })
            }
        }
    }

    #[test]
    fn failed_first_bind_does_not_insert_an_entry() {
        ACCEPT_FAILED_BIND.store(false, Ordering::SeqCst);
        assert!(DeviceCtx::<FailedFirstBind>::bind(Device::Cpu).is_err());

        ACCEPT_FAILED_BIND.store(true, Ordering::SeqCst);
        assert!(DeviceCtx::<FailedFirstBind>::bind(Device::Cpu).is_ok());
    }

    struct NestedBind;
    impl Placement for NestedBind {}

    struct RecursivePolicy;
    impl Placement for RecursivePolicy {
        fn validate_device(device: Device) -> Result<()> {
            DeviceCtx::<NestedBind>::bind(device).map(|_| ())
        }
    }

    #[test]
    fn placement_policy_can_bind_another_marker_without_deadlock() {
        let outer = DeviceCtx::<RecursivePolicy>::bind(Device::Cpu).unwrap();
        let nested = DeviceCtx::<NestedBind>::bind(Device::Cpu).unwrap();
        assert_eq!(outer.device(), nested.device());
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn failed_availability_probe_does_not_insert_an_entry() {
        struct UnavailableFirst;
        impl Placement for UnavailableFirst {}

        assert!(DeviceCtx::<UnavailableFirst>::bind(Device::Metal(usize::MAX)).is_err());
        assert!(DeviceCtx::<UnavailableFirst>::bind(Device::Cpu).is_ok());
    }

    struct ConcurrentBind;
    impl Placement for ConcurrentBind {}

    #[test]
    fn concurrent_bind_is_atomic_and_canonical() {
        let threads: Vec<_> = (0..16)
            .map(|_| thread::spawn(|| DeviceCtx::<ConcurrentBind>::bind(Device::Cpu).unwrap()))
            .collect();
        let contexts: Vec<_> = threads
            .into_iter()
            .map(|join| join.join().unwrap())
            .collect();

        assert!(
            contexts
                .iter()
                .all(|ctx| Arc::ptr_eq(contexts[0].binding(), ctx.binding()))
        );
    }

    struct MarkerA;
    struct MarkerB;
    impl Placement for MarkerA {}
    impl Placement for MarkerB {}

    #[test]
    fn multiple_markers_have_distinct_bindings_on_the_same_device() {
        let a = DeviceCtx::<MarkerA>::bind(Device::Cpu).unwrap();
        let b = DeviceCtx::<MarkerB>::bind(Device::Cpu).unwrap();

        assert_eq!(a.device(), b.device());
        assert!(!Arc::ptr_eq(a.binding(), b.binding()));
        let relabeled =
            checked_relabel_binding::<MarkerA, MarkerB>(Arc::clone(a.binding()), &b).unwrap();
        assert!(Arc::ptr_eq(&relabeled, b.binding()));
    }

    #[test]
    fn canonical_identity_corruption_is_rejected() {
        struct CanonicalIdentity;
        impl Placement for CanonicalIdentity {}

        let ctx = DeviceCtx::<CanonicalIdentity>::bind(Device::Cpu).unwrap();
        let forged = Arc::new(DeviceBinding {
            device: ctx.device(),
        });
        assert!(matches!(
            validate_binding::<CanonicalIdentity>(&forged, "test"),
            Err(Error::InvalidArg { op: "test", .. })
        ));
    }

    #[test]
    fn fixed_cpu_marker_and_convenience_constructor() {
        let cpu = DeviceCtx::<Cpu>::cpu().unwrap();
        assert_eq!(cpu.device(), Device::Cpu);

        #[cfg(all(feature = "metal", target_os = "macos"))]
        assert!(matches!(
            DeviceCtx::<Cpu>::bind(Device::Metal(0)),
            Err(Error::InvalidArg { .. })
        ));
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn fixed_metal_marker_enforces_its_ordinal() {
        use crate::typed::Metal;

        assert!(Metal::<2>::validate_device(Device::Metal(2)).is_ok());
        assert!(Metal::<2>::validate_device(Device::Metal(1)).is_err());
        assert!(Metal::<2>::validate_device(Device::Cpu).is_err());
    }

    struct BestAvailable;
    impl Placement for BestAvailable {}

    #[test]
    fn best_available_device_can_be_bound() {
        let best = Device::best_available();
        let ctx = DeviceCtx::<BestAvailable>::bind(best).unwrap();
        assert_eq!(ctx.device(), best);
    }
}
