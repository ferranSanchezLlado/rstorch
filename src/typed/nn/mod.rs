//! Neural-network interfaces for compile-time checked tensors.
//!
//! Typed modules form a separate tree from [`crate::nn::Module`]. In
//! particular, a typed model is never adapted to the runtime module API by a
//! public trait implementation: runtime mutation could otherwise change a
//! leaf's dtype or device while its Rust type continued to claim the old
//! element or placement marker.
//!
//! The only runtime-module bridge is the crate-private, short-lived
//! `RuntimeModuleAdapter`. Dedicated typed wrappers validate metadata around
//! its use without changing the wrapped runtime operation's failure semantics.
//! Public model conversion is instead consuming and retyped through
//! [`ToDevice`] or [`ToDType`]; constructing a fresh target and checked-loading
//! an opaque [`TypedStateDict`] is the other supported route.
//!
//! # Parameters and buffers
//!
//! Parameter `new` detaches `value`, validates its canonical binding, and mints
//! one runtime identity. `get` returns the cached traced leaf exactly when
//! `mode.records() && !is_frozen()`; `value` always returns the plain value.
//! `get`, `value`, and `grad_from` validate the retained canonical binding
//! before wrapping a runtime tensor. `set` accepts only the same exact `T`,
//! detaches it, and preserves the parameter's runtime gradient identity and
//! freeze state. Buffer `new` and `set` likewise detach their values.
//!
//! The two `set` methods deliberately disagree about reshaping a leaf whose
//! type carries [`crate::typed::DYN`], because only one of them has a runtime
//! authority to defer to. `TypedParam::set` delegates to
//! [`crate::nn::Param::set`], whose fixed-shape rule the optimizer's moment
//! buffers depend on, so `TypedParam<Tensor1<DYN>>::set` rejects a value of a
//! different length with [`crate::Error::ShapeMismatch`] (`op: "Param::set"`)
//! even though `T` admits it. `TypedBuffer::set` owns a bare runtime tensor
//! with no such rule and therefore accepts the new length. Neither can install
//! a value contradicting `T`. Consequently a `DYN` buffer, unlike a `DYN`
//! parameter, has no runtime shape backstop during a state load, which is why
//! `load_state_dict` checks staged dimensions against the target walk itself.
//!
//! A consuming parameter conversion first computes and validates the replacement
//! runtime tensor, then uses [`crate::nn::Param::set`] on the moved runtime
//! parameter before sealing the output type. Thus conversion preserves the
//! existing key, cached-leaf behavior, and freeze flag without a new runtime
//! API. A failed conversion returns no stale typed owner.
//!
//! Visitor names are literal path segments. Descent joins non-empty segments
//! with one `.` and restores the previous prefix afterward, exactly matching
//! dynamic visitor spelling. Duplicate paths and duplicate leaf identities are
//! malformed rather than silently deduplicated, including a manually written
//! walk that emits one owner twice. Weight tying owns and visits one parameter
//! once; repeated forward uses share that parameter's runtime gradient key.
//!
//! `state_dict` is fallible. It rejects duplicate paths, duplicate identities,
//! and other detectable malformation in the read-only walk. A `TypedStateDict`
//! retains each value's rank, markers, dtype, placement `TypeId`, and exact
//! canonical binding `Arc` identity. `load_state_dict` compares both walks and
//! rejects missing, extra, or duplicate entries; leaf-kind or typed-contract
//! mismatches; malformed or disagreeing walks; and noncanonical or different
//! binding identities before mutating any leaf. Two logical placements bound
//! to the same physical device still have different identities and do not load
//! into one another. Values are detached during staging. A complete mutable
//! pre-commit walk captures detached originals, and the commit walk rechecks
//! each path, leaf identity, kind, dimensions, full contract, duplicates, and
//! final length. A late malformed commit walk never receives a value at the
//! mismatching leaf and triggers a checked rollback of earlier replacements.
//!
//! These stable-walk rules are semantic obligations of a safe [`Module`]
//! implementation, not Rust safety invariants. `Module` remains a safe trait;
//! violating the obligations is never undefined behavior. Detectable malformed
//! walks return [`crate::Error`]. Because safe callback borrows cannot be held
//! across walks, a stateful implementation can keep changing during rollback;
//! rollback is strongest when its subsequent walks are stable. Even if it is
//! not, assignments occur only to leaves that still exactly match the validated
//! identity and contract, so malformed behavior cannot stale typed metadata.
//!
//! The private runtime adapter does not add rollback to dynamic optimizers or
//! other runtime operations; it preserves their existing failure semantics. The
//! typed optimizer wrappers therefore validate typed metadata *around* the
//! short-lived adapter and do not claim a transactionality the runtime optimizer
//! does not provide. The all-or-nothing behavior described above applies
//! specifically to typed state staging and commit.
//!
//! # Imports
//!
//! Typed neural-network items are namespace-only under `rstorch::typed::nn`;
//! they add nothing to [`crate::typed::prelude`].
//!
//! The [`TypedModule`] derive is re-exported here for the same reason, so a
//! crate depending only on `rstorch` reaches it as
//! `rstorch::typed::nn::TypedModule` and never needs a second direct
//! dependency on `rstorch_derive`.

use super::{DeviceBinding, DeviceCtx, FloatElement, Placement, TypedTensor};
use crate::{DType, Result, Tensor};
use std::any::TypeId;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;

pub use crate::nn::Mode;

/// Derive an implementation of [`Module`].
///
/// See the `#[derive(TypedModule)]` section of the [`rstorch_derive`] crate
/// docs for the field-classification table, `#[typed_module(skip)]`, and the
/// three ways this derive deliberately differs from `#[derive(Module)]` — most
/// importantly that it has no primitive whitelist, so a configuration field
/// needs an explicit skip.
pub use rstorch_derive::TypedModule;

mod activation;
mod attention;
mod composition;
mod dropout;
mod embedding;
mod linear;
mod norm;
mod param;
mod visit;

pub use activation::{Gelu, Relu};
pub use attention::{
    AttentionContext, AttentionInput, MultiHeadAttention, scaled_dot_product_attention,
};
pub use composition::{
    IntoSequential, Sequential, Sequential1, Sequential2, Sequential3, Sequential4, SequentialCons,
    SequentialInputDType, SequentialLayer, sequential,
};
pub use dropout::Dropout;
pub use embedding::{Embedding, EmbeddingInput};
pub use linear::Linear;
pub use norm::{BatchNorm2d, LayerNorm, RMSNorm};
pub use visit::{load_state_dict, state_dict};

/// A typed module that maps `Input` to an associated typed output.
///
/// The mutable receiver is intentional: dropout RNG and running-statistic
/// buffers are ordinary model state, not hidden interior mutability.
pub trait Forward<Input> {
    /// The compile-time checked output type.
    type Output;

    /// Runs the module under `mode`.
    ///
    /// # Errors
    ///
    /// Implementation-defined; conforming implementations only fail from a
    /// canonical-binding mismatch between `self` and `input`, or from the
    /// underlying runtime op's own error.
    fn forward(&mut self, input: &Input, mode: Mode) -> Result<Self::Output>;
}

/// A compile-time checked module tree.
///
/// The two walks must emit the same leaves in the same order with byte-for-byte
/// identical dotted paths. A parameter or buffer has one owning location and
/// is emitted once; using a parameter more than once during `forward` does not
/// visit or update it more than once. Repeated walks must remain stable and the
/// walk methods must not mutate leaf ownership or contracts as a side effect.
///
/// This trait deliberately has no relationship to [`crate::nn::Module`].
pub trait Module {
    /// Walks typed parameters and persistent buffers read-only.
    fn visit(&self, visitor: &mut TypedVisitor<'_>);

    /// Walks the same typed parameters and buffers mutably.
    fn visit_mut(&mut self, visitor: &mut TypedVisitorMut<'_>);
}

/// A trainable parameter whose runtime value always satisfies `T`.
///
/// The runtime [`crate::nn::Param`] and canonical placement binding are private.
/// `TypedParam` intentionally does not implement `Clone`: replication is
/// construct-plus-checked-load, preserving the runtime parameter's stable
/// gradient identity, cached leaf, and freeze state within each model.
///
/// Its value-preserving constructors and accessors return `T`, never a mutable
/// runtime parameter or tensor.
pub struct TypedParam<T>
where
    T: TypedTensor,
    T::Elem: FloatElement,
{
    runtime: crate::nn::Param,
    binding: Arc<DeviceBinding>,
    marker: PhantomData<T>,
}

/// A persistent, non-trainable state leaf whose runtime value satisfies `T`.
///
/// A dedicated owner, rather than a bare runtime tensor, lets the private
/// mutable adapter revalidate a replacement before the typed model is used
/// again. Public access remains typed and cannot mutate the runtime tensor.
pub struct TypedBuffer<T: TypedTensor> {
    runtime: Tensor,
    binding: Arc<DeviceBinding>,
    marker: PhantomData<T>,
}

/// A read-only visitor over typed parameters and persistent buffers.
///
/// The `param`, `buffer`, and `module` entry points. Construction and
/// the erased sink remain crate-private so safe public code cannot obtain a
/// runtime parameter or mutable runtime tensor from a typed leaf.
pub struct TypedVisitor<'a> {
    path: String,
    sink: &'a mut dyn FnMut(&str, TypedLeaf<'_>),
    error: Option<crate::Error>,
}

/// The mutable counterpart of [`TypedVisitor`].
///
/// Mutable erased leaves are private and are only valid for the duration of one
/// walk. State loading reaches this visitor only after complete staging.
pub struct TypedVisitorMut<'a> {
    path: String,
    sink: &'a mut dyn FnMut(&str, TypedLeafMut<'_>),
    error: Option<crate::Error>,
}

/// An opaque, ordered snapshot of typed module state.
///
/// Entries retain the runtime value plus the exact typed leaf contract and
/// canonical [`DeviceCtx`] binding identity. Checkpoint or dynamic tensors do
/// not acquire this authority: they must be checked against a target module and
/// its canonical context before a transaction commits.
pub struct TypedStateDict {
    entries: BTreeMap<String, StateEntry>,
}

impl TypedStateDict {
    /// Returns the number of uniquely named parameters and buffers.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns whether the snapshot contains no state leaves.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the snapshot's stable dotted paths in bytewise order.
    pub fn paths(&self) -> impl ExactSizeIterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }
}

/// Consuming model movement that changes every placement-bearing output type.
///
/// Implementations consume the old model and construct `Output` from consuming
/// leaf conversions. Parameter conversions preserve the moved runtime
/// parameter's identity and freeze state through `Param::set`; buffers preserve
/// their state role and all paths remain byte-identical. An error yields no
/// model under stale placement markers.
pub trait ToDevice<Q: Placement>: Module + Sized {
    /// The same model structure with all placement-bearing leaves retyped to
    /// `Q`.
    type Output: Module;

    /// Moves the complete model to the target canonical binding.
    ///
    /// # Errors
    ///
    /// Implementation-defined; conforming implementations only fail from a
    /// canonical-binding mismatch or the underlying transfer's own error.
    fn to_device(self, target: &DeviceCtx<Q>) -> Result<Self::Output>;
}

/// Consuming floating-point precision conversion for a typed model.
///
/// Integer and boolean structural buffers retain their element types. Every
/// floating-point leaf represented by the model's precision parameter must be
/// retyped to `F`; no in-place conversion under the old element marker is
/// permitted.
pub trait ToDType<F: FloatElement>: Module + Sized {
    /// The same model structure with floating-point leaves retyped to `F`.
    type Output: Module;

    /// Consumes the complete model and returns one whose precision-bearing
    /// leaves have been retyped to `F`.
    ///
    /// # Errors
    ///
    /// Implementation-defined; conforming implementations only fail from the
    /// underlying cast's own error.
    fn to_dtype(self) -> Result<Self::Output>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LeafKind {
    Param,
    Buffer,
}

/// Runtime-erased metadata that is still authoritative enough to revalidate a
/// typed leaf. `binding` identity, not only its physical device, is part of the
/// contract.
#[derive(Clone)]
struct LeafContract {
    kind: LeafKind,
    /// Cross-check only, never an independent authority: `markers.len()` is
    /// always this value, because [`crate::typed::sealed::TypedTensor::MARKERS`]
    /// and [`TypedTensor::RANK`] are emitted by one macro from one dimension
    /// list and `TypedTensor` is sealed. Comparing it as well as `markers`
    /// therefore cannot reject anything `markers` does not already reject; it
    /// exists so a hand-built [`StateEntry`] cannot claim a rank its marker
    /// list contradicts. Do not read its survival of a mutation as evidence
    /// that `markers` is unchecked — see
    /// `every_erased_contract_field_and_binding_identity_is_checked`.
    rank: usize,
    markers: &'static [usize],
    dtype: DType,
    placement: TypeId,
    binding: Arc<DeviceBinding>,
}

enum TypedLeaf<'a> {
    Param(&'a crate::nn::Param, LeafContract),
    Buffer(&'a Tensor, LeafContract),
}

enum TypedLeafMut<'a> {
    Param(&'a mut crate::nn::Param, LeafContract),
    Buffer(&'a mut Tensor, LeafContract),
}

struct StateEntry {
    value: Tensor,
    contract: LeafContract,
}

/// Walks `model` twice and returns the first state dict only if both walks
/// produced the same paths.
///
/// Stable repeated read-only walks. `typed::optim` and
/// `typed::persist` both preflight a model this way before handing it to the
/// runtime adapter, so the walk-comparison lives here once.
pub(in crate::typed) fn stable_state<M: Module + ?Sized>(
    model: &M,
    op: &'static str,
) -> Result<TypedStateDict> {
    let first = state_dict(model)?;
    let second = state_dict(model)?;
    if first.paths().ne(second.paths()) {
        return Err(crate::Error::InvalidArg {
            op,
            msg: "typed Module paths changed between preflight walks; Module requires stable repeated walks"
                .into(),
        });
    }
    Ok(first)
}

/// The only seam through which runtime optimizers and persistence code may
/// temporarily view a typed tree as a runtime module.
///
/// The runtime trait is implemented privately. The adapter borrows the typed
/// module mutably for its entire lifetime and must not escape the dedicated
/// typed wrapper that created it. It does not alter the runtime operation's
/// failure or rollback semantics.
pub(crate) struct RuntimeModuleAdapter<'a, M: Module + ?Sized> {
    module: &'a mut M,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed::{Cpu, Tensor1};

    struct Identity;

    impl Module for Identity {
        fn visit(&self, _visitor: &mut TypedVisitor<'_>) {}

        fn visit_mut(&mut self, _visitor: &mut TypedVisitorMut<'_>) {}
    }

    impl<T: Clone> Forward<T> for Identity {
        type Output = T;

        fn forward(&mut self, input: &T, _mode: Mode) -> Result<T> {
            Ok(input.clone())
        }
    }

    fn assert_module_object_safe(_module: &dyn Module) {}

    fn assert_forward_object_safe(_module: &mut dyn Forward<Tensor1<1>, Output = Tensor1<1>>) {}

    impl<Q: Placement> ToDevice<Q> for Identity {
        type Output = Self;

        fn to_device(self, _target: &DeviceCtx<Q>) -> Result<Self::Output> {
            Ok(self)
        }
    }

    impl<F: FloatElement> ToDType<F> for Identity {
        type Output = Self;

        fn to_dtype(self) -> Result<Self::Output> {
            Ok(self)
        }
    }

    #[test]
    fn forward_and_consuming_retype_contracts_are_implementable() {
        let ctx = DeviceCtx::cpu().unwrap();
        let input = Tensor1::<1>::from_vec(vec![3.0], [1], &ctx).unwrap();
        let mut module = Identity;
        assert_module_object_safe(&module);
        assert_forward_object_safe(&mut module);
        let output = module.forward(&input, Mode::EVAL).unwrap();
        assert_eq!(output.to_vec().unwrap(), vec![3.0]);

        let module = <Identity as ToDevice<Cpu>>::to_device(module, &ctx).unwrap();
        let _module = <Identity as ToDType<f32>>::to_dtype(module).unwrap();
    }

    #[test]
    fn state_paths_are_byte_preserving_and_sorted() {
        let state = TypedStateDict {
            entries: BTreeMap::from([
                ("head.weight".to_string(), state_entry()),
                ("blocks.0.weight".to_string(), state_entry()),
            ]),
        };

        assert_eq!(
            state.paths().collect::<Vec<_>>(),
            vec!["blocks.0.weight", "head.weight"]
        );
    }

    fn state_entry() -> StateEntry {
        let ctx = DeviceCtx::cpu().unwrap();
        let value = Tensor1::<1>::from_vec(vec![0.0], [1], &ctx).unwrap();
        StateEntry {
            value: value.as_dynamic().clone(),
            contract: LeafContract {
                kind: LeafKind::Buffer,
                rank: 1,
                markers: &[1],
                dtype: DType::F32,
                placement: TypeId::of::<Cpu>(),
                binding: Arc::clone(ctx.binding()),
            },
        }
    }
}
