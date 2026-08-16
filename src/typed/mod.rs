//! Compile-time checked tensor APIs.
//!
//! This opt-in namespace wraps [`crate::Tensor`]. It never owns another
//! storage, execution, dispatch, layout, or autograd implementation. Rank,
//! element type, and logical placement are Rust types; each
//! [`crate::typed::DYN`] occurrence
//! independently retains its value only in the wrapped runtime tensor.
//!
//! # Failure boundary
//!
//! Rank, dtype, placement marker, strict-shape, and invalid const-axis choices
//! rejected by an associated-output implementation are ordinary trait/type
//! errors. Body-level const assertions are reserved for all-known relational
//! failures: matmul contraction, reshape element count, squeeze's selected
//! dimension, broadcast compatibility, convolution channels, and loss rows.
//! Any such relation involving [`crate::typed::DYN`] is checked at runtime
//! instead.
//!
//! A body-level assertion reports `E0080` when the generic method is
//! monomorphized. `cargo check` is not guaranteed to instantiate it, which is
//! why the UI suite (`tests/typed_ui.rs`) runs build-required cases alongside
//! check-only ones. Allocation, backend support, device availability, index
//! values, and all deferred relationships return structured [`crate::Error`]
//! values.
//!
//! # Placement contract
//!
//! A [`Placement`](crate::typed::Placement) marker is bound to a runtime
//! [`Device`] through [`DeviceCtx`](crate::typed::DeviceCtx). The registry is process-lifetime and
//! keyed by `TypeId::of::<P>()`. Binding the same marker/device pair is
//! idempotent; binding a marker to a *different* device is
//! [`Error::InvalidArg`], because a marker that silently re-pointed would make
//! two tensors' types claim they are co-located when they are not. A binding is
//! inserted only after `P`'s policy accepts it and availability is probed with
//! `Tensor::zeros((), DType::F32, &device)`.
//!
//! Tensors retain the canonical `Arc` and operations compare its identity
//! defensively, so a forged binding is caught rather than trusted. Re-labeling
//! is zero-copy and succeeds only when both canonical bindings name the same
//! runtime device.
//!
//! # Tensor boundary
//!
//! Every constructor validates rank, each static marker, dtype, runtime device,
//! and canonical binding before reaching a crate-private trusted wrap seam
//! (`tensor::checked_wrap`). That seam is never public: there is no
//! `Deref<Target = Tensor>` and no unchecked public constructor, so a wrapper's
//! type is always a claim the crate has checked.
//!
//! Erasure, re-entry, refinement, and relabeling all preserve storage and
//! autograd identity — they are re-descriptions of the same tensor, never
//! copies.
//!
//! # Core and autograd
//!
//! `detach` is deliberately fallible even though the runtime detach cannot
//! fail, so that every named typed operation has the same `Result` shape and
//! callers never learn which ones happen to be infallible today.
//! Shape-preserving methods return `Self`; movement and casts go through the
//! associated output contracts in [`ops`](crate::typed::ops).
//!
//! [`TypedGradsExt`](crate::typed::TypedGradsExt) is implemented for
//! [`crate::Grads`]; its deliberately unique method name avoids
//! inherent-method overloading. It reuses the input's canonical binding and
//! wraps without copying or changing graph identity.

use crate::{Device, Element, Error, Result, Tensor};
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

mod autograd;
pub(crate) mod const_check;
mod device;
mod dim;
#[doc(hidden)]
pub mod ops;
pub mod prelude;
pub(in crate::typed) mod tensor;

pub use autograd::TypedGradsExt;
pub use dim::DYN;

pub(in crate::typed) mod sealed {
    use super::{Arc, Device, Tensor};

    #[derive(Debug)]
    pub struct DeviceBinding {
        pub(in crate::typed) device: Device,
    }

    pub trait ElementCapability {}
    pub trait TypedTensor: Sized {
        const MARKERS: &'static [usize];

        fn trusted_from_validated(tensor: Tensor, binding: Arc<DeviceBinding>) -> Self;
        fn dynamic(&self) -> &Tensor;
        fn binding(&self) -> &Arc<DeviceBinding>;
    }
}

pub(in crate::typed) use sealed::DeviceBinding;

/// An element type accepted by floating-point-only typed operations.
pub trait FloatElement: Element + sealed::ElementCapability {}

/// An element type accepted by typed arithmetic operations.
pub trait NumericElement: Element + sealed::ElementCapability {}

/// The element type accepted by typed index operations.
pub trait IndexElement: Element + sealed::ElementCapability {}

macro_rules! element_capability_table {
    ($(($ty:ty: $($cap:ident)*)),+ $(,)?) => {
        $(
            impl sealed::ElementCapability for $ty {}
            element_capability_table!(@caps $ty; $($cap)*);
        )+
    };
    (@caps $ty:ty;) => {};
    (@caps $ty:ty; float $($rest:ident)*) => {
        impl FloatElement for $ty {}
        element_capability_table!(@caps $ty; $($rest)*);
    };
    (@caps $ty:ty; numeric $($rest:ident)*) => {
        impl NumericElement for $ty {}
        element_capability_table!(@caps $ty; $($rest)*);
    };
    (@caps $ty:ty; index $($rest:ident)*) => {
        impl IndexElement for $ty {}
        element_capability_table!(@caps $ty; $($rest)*);
    };
}

element_capability_table! {
    (f32: float numeric),
    (f64: float numeric),
    (half::f16: float numeric),
    (half::bf16: float numeric),
    (i64: numeric index),
    (bool:),
}

/// A logical placement marker used by typed tensors and [`DeviceCtx`].
///
/// This trait is intentionally open but is not blanket-implemented. A custom
/// unconstrained marker opts in with an empty implementation:
///
/// ```
/// use rstorch::typed::Placement;
/// struct Main;
/// impl Placement for Main {}
/// ```
pub trait Placement: Send + Sync + 'static {
    /// Validates marker-specific device restrictions before registry insertion.
    #[doc(hidden)]
    fn validate_device(_device: Device) -> Result<()> {
        Ok(())
    }
}

/// The built-in logical placement constrained to [`Device::Cpu`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Cpu;

impl Placement for Cpu {
    fn validate_device(device: Device) -> Result<()> {
        if device == Device::Cpu {
            Ok(())
        } else {
            Err(Error::InvalidArg {
                op: "DeviceCtx::bind",
                msg: format!("Cpu placement requires cpu, got {device}"),
            })
        }
    }
}

/// A fixed Apple Metal logical placement with device ordinal `N`.
#[cfg(all(feature = "metal", target_os = "macos"))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Metal<const N: usize>;

#[cfg(all(feature = "metal", target_os = "macos"))]
impl<const N: usize> Placement for Metal<N> {
    fn validate_device(device: Device) -> Result<()> {
        if device == Device::Metal(N) {
            Ok(())
        } else {
            Err(Error::InvalidArg {
                op: "DeviceCtx::bind",
                msg: format!("Metal<{N}> placement requires metal:{N}, got {device}"),
            })
        }
    }
}

/// A fixed NVIDIA CUDA logical placement with device ordinal `N`.
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Cuda<const N: usize>;

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
impl<const N: usize> Placement for Cuda<N> {
    fn validate_device(device: Device) -> Result<()> {
        if device == Device::Cuda(N) {
            Ok(())
        } else {
            Err(Error::InvalidArg {
                op: "DeviceCtx::bind",
                msg: format!("Cuda<{N}> placement requires cuda:{N}, got {device}"),
            })
        }
    }
}

/// A fixed portable WebGPU logical placement with adapter ordinal `N`.
///
/// WGPU currently provides F32 compute and lossless I64/Bool storage for the
/// operations that support those dtypes. Adapters exposing `SHADER_F16` also
/// provide native F16 operations; unsupported dtype/operation pairs return
/// [`Error::Unsupported`].
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Wgpu<const N: usize>;

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
impl<const N: usize> Placement for Wgpu<N> {
    fn validate_device(device: Device) -> Result<()> {
        if device == Device::Wgpu(N) {
            Ok(())
        } else {
            Err(Error::InvalidArg {
                op: "DeviceCtx::bind",
                msg: format!("Wgpu<{N}> placement requires wgpu:{N}, got {device}"),
            })
        }
    }
}

/// A canonical process-lifetime binding from `P` to one runtime device.
pub struct DeviceCtx<P: Placement> {
    pub(in crate::typed) binding: Arc<DeviceBinding>,
    pub(in crate::typed) marker: PhantomData<P>,
}

impl<P: Placement> Clone for DeviceCtx<P> {
    fn clone(&self) -> Self {
        Self {
            binding: Arc::clone(&self.binding),
            marker: PhantomData,
        }
    }
}

impl<P: Placement> fmt::Debug for DeviceCtx<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceCtx")
            .field("marker", &std::any::type_name::<P>())
            .field("device", &self.binding.device)
            .finish()
    }
}

/// Compile-time metadata shared by all typed rank wrappers.
///
/// The trait is sealed. Runtime boundary methods are inherent methods added by
/// inherent methods, not trait methods, so leaf owner modules can implement them without
/// editing this contract file.
pub trait TypedTensor: sealed::TypedTensor + Sized {
    /// Exact Rust element type carried by the tensor.
    type Elem: Element;

    /// Exact logical placement marker carried by the tensor.
    type Placement: Placement;

    /// Runtime dimension container (`[usize; RANK]`).
    type Dims: AsRef<[usize]> + Copy + fmt::Debug + Eq;

    /// Compile-time rank of this wrapper.
    const RANK: usize;
}

macro_rules! define_typed_tensors {
    ($(($name:ident, $rank:literal, [$($dim:ident),*])),+ $(,)?) => {
        $(
            /// An opaque compile-time checked tensor wrapper.
            pub struct $name<$(const $dim: usize,)* E: Element = f32, P: Placement = Cpu> {
                inner: Tensor,
                binding: Arc<DeviceBinding>,
                marker: PhantomData<(E, P)>,
            }

            impl<$(const $dim: usize,)* E: Element, P: Placement> Clone
                for $name<$($dim,)* E, P>
            {
                fn clone(&self) -> Self {
                    Self {
                        inner: self.inner.clone(),
                        binding: Arc::clone(&self.binding),
                        marker: PhantomData,
                    }
                }
            }

            impl<$(const $dim: usize,)* E: Element, P: Placement> sealed::TypedTensor
                for $name<$($dim,)* E, P>
            {
                const MARKERS: &'static [usize] = &[$($dim),*];

                fn trusted_from_validated(
                    tensor: Tensor,
                    binding: Arc<DeviceBinding>,
                ) -> Self {
                    Self {
                        inner: tensor,
                        binding,
                        marker: PhantomData,
                    }
                }

                fn dynamic(&self) -> &Tensor {
                    &self.inner
                }

                fn binding(&self) -> &Arc<DeviceBinding> {
                    &self.binding
                }
            }

            impl<$(const $dim: usize,)* E: Element, P: Placement> TypedTensor
                for $name<$($dim,)* E, P>
            {
                type Elem = E;
                type Placement = P;
                type Dims = [usize; $rank];
                const RANK: usize = $rank;
            }

            impl<$(const $dim: usize,)* E: Element, P: Placement> fmt::Debug
                for $name<$($dim,)* E, P>
            {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    self.inner.fmt(f)
                }
            }

            impl<$(const $dim: usize,)* E: Element, P: Placement> fmt::Display
                for $name<$($dim,)* E, P>
            {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    self.inner.fmt(f)
                }
            }
        )+
    };
}

// The wrappers invoke this same table when generating rank metadata and complete
// associated-output implementations. Rank eight is the hard typed ceiling.
macro_rules! typed_rank_table {
    ($callback:ident) => {
        $callback! {
            (Tensor0, 0, []),
            (Tensor1, 1, [D0]),
            (Tensor2, 2, [D0, D1]),
            (Tensor3, 3, [D0, D1, D2]),
            (Tensor4, 4, [D0, D1, D2, D3]),
            (Tensor5, 5, [D0, D1, D2, D3, D4]),
            (Tensor6, 6, [D0, D1, D2, D3, D4, D5]),
            (Tensor7, 7, [D0, D1, D2, D3, D4, D5, D6]),
            (Tensor8, 8, [D0, D1, D2, D3, D4, D5, D6, D7]),
        }
    };
}

pub(crate) use typed_rank_table;
typed_rank_table!(define_typed_tensors);

pub mod data;
pub mod nn;
pub mod optim;
pub mod persist;

#[cfg(test)]
mod tests {
    use super::*;

    struct Main;
    impl Placement for Main {}

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn rank_capability_and_placement_contracts_compile_on_msrv() {
        assert_eq!(<Tensor0 as TypedTensor>::RANK, 0);
        assert_eq!(<Tensor8<1, 2, 3, 4, 5, 6, 7, 8> as TypedTensor>::RANK, 8);
        assert_send_sync::<Tensor2<DYN, 7, f32, Main>>();
        assert!(Cpu::validate_device(Device::Cpu).is_ok());
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn fixed_placement_policies_reject_other_devices() {
        assert!(Cpu::validate_device(Device::Metal(0)).is_err());
        assert!(Metal::<0>::validate_device(Device::Metal(0)).is_ok());
        assert!(Metal::<0>::validate_device(Device::Metal(1)).is_err());
        assert!(Metal::<0>::validate_device(Device::Cpu).is_err());
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    #[test]
    fn fixed_cuda_placement_policy_rejects_other_devices() {
        assert!(Cuda::<0>::validate_device(Device::Cuda(0)).is_ok());
        assert!(Cuda::<0>::validate_device(Device::Cuda(1)).is_err());
        assert!(Cuda::<0>::validate_device(Device::Cpu).is_err());
    }
}
