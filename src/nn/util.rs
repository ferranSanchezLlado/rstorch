//! Model-level utilities: visitor-based helpers that operate on any
//! [`Module`], erased or not.
//!
//! The public spelling is one extension trait, [`ModuleExt`], blanket-
//! implemented for every `Module + ?Sized`. Its methods delegate to the
//! `pub(crate)` free functions below, which are the actual implementations
//! and the only form the rest of the crate calls. There is deliberately no
//! public free-function spelling: two ways to write `state_dict` would be two
//! ways forever.
//!
//! Two halves: the read-only walks ([`num_params`], [`state_dict`]) and the
//! mutating conversions ([`load_state_dict`], [`to_device`], [`to_dtype`]),
//! which validate the whole walk before swapping anything.
//!
//! `state_dict` uses an opaque [`StateDict`] with validated paths and
//! controlled mutation. Values are `Arc`-cheap clones. The representation is
//! the sanctioned replication path (construct + `load_state_dict`, no disk).
//! Persistence to safetensors is a separate concern owned by
//! [`persist`](crate::persist).
//!
//! **Params vs buffers**: `num_params` counts trainable [`Param`](crate::nn::Param)
//! elements only; `state_dict`/`load_state_dict`/`to_device`/`to_dtype`
//! operate on both parameters and non-trainable `Tensor` buffers (`BatchNorm`
//! running stats), so a moved or checkpointed model stays complete.
//!
//! The user-facing prose — the replication recipe and the all-or-nothing
//! guarantee — lives in the [`nn`](crate::nn) module docs, which is where
//! rustdoc renders it (this module is private; only `ModuleExt` is
//! re-exported).
//!
//! Every mutating helper validates the whole walk before it swaps anything
//! (the in-memory sibling of [`persist::stage`](crate::persist::stage)), so a
//! rejected load or a failed conversion leaves the model exactly as it was.

use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::nn::Module;
use crate::nn::visit::{Leaf, LeafMut, visit_all, visit_all_mut};
use crate::tensor::Tensor;
use std::collections::BTreeMap;

/// Total number of scalar elements across all **trainable parameters** of
/// `module` (buffers are excluded).
pub(crate) fn num_params<M: Module + ?Sized>(module: &M) -> usize {
    let mut total = 0usize;
    visit_all(module, &mut |_path, leaf| {
        if let Leaf::Param(p) = leaf {
            total += p.value().num_elements();
        }
    });
    total
}

/// Collect the module's parameters **and buffers** into a validated,
/// deterministic [`StateDict`]. Values are `Arc`-cheap clones. Buffers are
/// included so a checkpoint reconstructs a model (running stats survive),
/// matching `PyTorch` `state_dict` semantics.
///
/// # Errors
///
/// [`Error::InvalidArg`] (`op: "state_dict"`) if a hand-written
/// [`Module`] emits the same path more than once, emits one leaf under
/// multiple paths, or emits a path with an empty dotted segment. Such a walk
/// cannot represent a lossless state dictionary; derive-generated modules
/// cannot produce either condition.
pub(crate) fn state_dict<M: Module + ?Sized>(module: &M) -> Result<StateDict> {
    let mut out = BTreeMap::new();
    let mut seen = BTreeMap::new();
    let mut failure = None;
    visit_all(module, &mut |path, leaf| {
        if failure.is_some() {
            return;
        }
        if let Err(error) = validate_path(path, "state_dict") {
            failure = Some(error);
            return;
        }
        let leaf_identity = identity(&leaf);
        if seen.insert(leaf_identity, path.to_string()).is_some() {
            failure = Some(Error::invalid_arg(
                "state_dict",
                "module visits one leaf under multiple state-dict paths",
            ));
            return;
        }
        let kind = leaf_identity.kind;
        let value = match leaf {
            Leaf::Param(p) => p.value().detach(),
            Leaf::Buffer(t) => t.detach(),
        };
        if out
            .insert(
                path.to_string(),
                StateEntry {
                    value,
                    kind: Some(kind),
                },
            )
            .is_some()
        {
            failure = Some(Error::invalid_arg(
                "state_dict",
                format!("module emits the state-dict path `{path}` twice"),
            ));
        }
    });
    match failure {
        Some(error) => Err(error),
        None => Ok(StateDict { entries: out }),
    }
}

/// Load values from `state` into `module` by matching dotted paths — both
/// parameters and buffers (the in-memory half of the replication/checkpoint
/// path; see the [`nn`](crate::nn) module docs for the replication recipe).
///
/// The match is **exact and loud**. `state` must carry one entry for every
/// path the module's walk emits, and no others, each with the same dtype,
/// dims, and device as the value it replaces. A silent partial load is the
/// failure mode this rejects: every check names the offending path.
///
/// Incoming values are [`detach`](crate::Tensor::detach)ed, so a state dict
/// captured from a live graph can never smuggle autograd history into a
/// parameter.
///
/// # Errors
///
/// [`Error::InvalidArg`] (`op: "load_state_dict"`), naming the path, when
///
/// - `state` is missing a path the module expects,
/// - `state` carries a path the module does not have,
/// - a value's dims, dtype, or device differ from the current value's — for a
///   device or dtype difference, convert the source explicitly first
///   ([`to_device`], [`to_dtype`], or [`Tensor::to_device`]), because a
///   parameter silently changing device or precision is exactly the kind of
///   surprise this API refuses, or
/// - the module emits one path twice, or its two walks disagree (a path
///   emitted by `visit_mut` that `visit` never emitted) — both are `Module`
///   implementation bugs that would make a state dict lossy.
///
/// Nothing is swapped unless every check passes.
pub(crate) fn load_state_dict<M: Module + ?Sized>(module: &mut M, state: &StateDict) -> Result<()> {
    const OP: &str = "load_state_dict";
    // The current values double as the schema every incoming value must match.
    let current = mapped(OP, module, |t| Ok(t.clone()))?;

    // Unexpected paths first — the loudest signal that the state dict came
    // from a different model (mirrors `persist::stage`).
    for path in state.entries.keys() {
        if !current.contains_key(path) {
            return Err(Error::invalid_arg(
                OP,
                format!(
                    "unexpected key `{path}` in state dict: the target module has \
                     no parameter or buffer at that path"
                ),
            ));
        }
    }

    for (path, target) in &current {
        let Some(entry) = state.entries.get(path) else {
            return Err(Error::invalid_arg(
                OP,
                format!("missing key `{path}` in state dict (expected by the target module)"),
            ));
        };
        if let Some(kind) = entry.kind
            && kind != target.identity.kind
        {
            return Err(Error::invalid_arg(
                OP,
                format!("`{path}` leaf kind does not match the target module"),
            ));
        }
        let value = &entry.value;
        if value.dims() != target.value.dims() {
            return Err(Error::invalid_arg(
                OP,
                format!(
                    "`{path}` shape mismatch: state dict has {}, target expects {}",
                    value.shape(),
                    target.value.shape()
                ),
            ));
        }
        if value.dtype() != target.value.dtype() {
            return Err(Error::invalid_arg(
                OP,
                format!(
                    "`{path}` dtype mismatch: state dict has {}, target expects {} \
                     (cast explicitly with to_dtype)",
                    value.dtype(),
                    target.value.dtype()
                ),
            ));
        }
        if value.device() != target.value.device() {
            return Err(Error::invalid_arg(
                OP,
                format!(
                    "`{path}` device mismatch: state dict has {}, target expects {} \
                     (move it explicitly with to_device)",
                    value.device(),
                    target.value.device()
                ),
            ));
        }
    }

    let values = current
        .into_iter()
        .map(|(path, target)| {
            let value = state
                .entries
                .get(&path)
                .expect("load_state_dict validated every current path")
                .value
                .clone();
            (
                path,
                Mapped {
                    value,
                    identity: target.identity,
                },
            )
        })
        .collect();
    commit(OP, module, &values)
}

/// Move every parameter and buffer of `module` to `device` (constructors
/// initialize on their given device; this converts after).
///
/// All-or-nothing: every value is converted first, and only a fully converted
/// module is committed, so a failure never leaves parameters split across two
/// devices.
///
/// This helper does not own or convert an optimizer's moment buffers. Move the
/// model before creating optimizer state, or rebuild/reload the optimizer for
/// the new device before stepping it again.
///
/// # Errors
///
/// Whatever [`Tensor::to_device`] reports for a leaf (for a device with no
/// backend, [`Error::Unsupported`]), or [`Error::InvalidArg`]
/// (`op: "to_device"`) if the module's two walks disagree.
pub(crate) fn to_device<M: Module + ?Sized>(module: &mut M, device: &Device) -> Result<()> {
    const OP: &str = "to_device";
    let values = mapped(OP, module, |t| t.to_device(device))?;
    commit(OP, module, &values)
}

/// Cast every **floating-point** parameter and buffer of `module` to `dtype`
/// (constructors initialize F32, convert after).
///
/// Integer and boolean leaves are left untouched, as in `PyTorch`: an `I64`
/// index buffer or a `Bool` mask is structure, not precision, and casting it
/// to a float would corrupt the model rather than convert it.
///
/// All-or-nothing: every value is converted first, and only a fully converted
/// module is committed, so a failure never leaves a model in mixed precision.
///
/// This helper does not own or convert an optimizer's moment buffers. Cast the
/// model before creating optimizer state, or rebuild/reload the optimizer for
/// the new dtype before stepping it again.
///
/// # Errors
///
/// - [`Error::InvalidArg`] (`op: "to_dtype"`) if `dtype` is not a float dtype
///   (a whole model cast to `I64`/`Bool` is a bug, not a request), or if the
///   module's two walks disagree.
/// - Whatever [`Tensor::to_dtype`] reports for a leaf — notably
///   [`Error::Unsupported`] for a cast lane the backend does not implement.
pub(crate) fn to_dtype<M: Module + ?Sized>(module: &mut M, dtype: DType) -> Result<()> {
    const OP: &str = "to_dtype";
    if !dtype.is_float() {
        return Err(Error::invalid_arg(
            OP,
            format!(
                "cannot cast a module to {dtype}: nn::to_dtype converts precision, \
                 so the target must be a float dtype"
            ),
        ));
    }
    let values = mapped(OP, module, |t| {
        if t.dtype().is_float() {
            t.to_dtype(dtype)
        } else {
            Ok(t.clone())
        }
    })?;
    commit(OP, module, &values)
}

/// The discoverable, method-call spelling of this module's five model-level
/// utilities. Blanket-implemented for every [`Module`], so
/// `use rstorch::prelude::*; model.num_params()` resolves without naming
/// this trait — the whole reason it exists: the underlying functions used to
/// be free functions in an internal module that `prelude` never exported,
/// which made them undiscoverable from a bare `use rstorch::prelude::*`.
///
/// The blanket impl is over `M: Module + ?Sized`, so the methods are equally
/// available on an erased model — `&mut dyn Module` is the type the optimizer
/// `step` takes, and a `Box<dyn Module>` is how a heterogeneous zoo is stored.
/// A `Sized` bound here would have left those holders with no spelling at all.
///
/// The `typed` half of the crate keeps free functions
/// (`typed::nn::state_dict`) rather than a parallel trait. That is deliberate,
/// and the reasons are recorded where a typed user reads them, in the
/// `typed::nn` module docs.
///
/// See the [`nn`](crate::nn) module docs for the replication recipe and the
/// all-or-nothing loading/conversion guarantee every mutating method here
/// follows.
pub trait ModuleExt: Module {
    /// Total number of scalar elements across all **trainable parameters**
    /// (buffers are excluded).
    fn num_params(&self) -> usize {
        num_params(self)
    }

    /// Collect this module's parameters and buffers into a validated,
    /// deterministic [`StateDict`].
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArg`](crate::Error::InvalidArg) if a hand-written
    /// [`Module`] emits duplicate paths, duplicate leaf identities, or an
    /// invalid dotted path.
    fn state_dict(&self) -> Result<StateDict> {
        state_dict(self)
    }

    /// Load values from `state` by matching dotted paths — both parameters
    /// and buffers. The match is exact and loud: nothing is swapped unless
    /// `state` carries exactly the paths this module's walk emits, each with
    /// the same leaf kind, dimensions, dtype, and device as the value it
    /// replaces.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArg`](crate::Error::InvalidArg), naming the offending
    /// path, on a missing path, an unexpected path, a leaf-kind/shape/dtype/
    /// device mismatch, or a malformed walk.
    fn load_state_dict(&mut self, state: &StateDict) -> Result<()> {
        load_state_dict(self, state)
    }

    /// Move every parameter and buffer to `device`, all-or-nothing: a
    /// failure never leaves parameters split across two devices.
    ///
    /// # Errors
    /// Whatever the per-leaf device conversion reports, or
    /// [`Error::InvalidArg`](crate::Error::InvalidArg) if the module's two
    /// walks disagree.
    fn to_device(&mut self, device: &Device) -> Result<()> {
        to_device(self, device)
    }

    /// Cast every **floating-point** parameter and buffer to `dtype`
    /// (integer/boolean leaves are left untouched), all-or-nothing.
    ///
    /// # Errors
    /// [`Error::InvalidArg`](crate::Error::InvalidArg) if `dtype` is not a
    /// float dtype or the module's two walks disagree, or whatever the
    /// per-leaf dtype conversion reports.
    fn to_dtype(&mut self, dtype: DType) -> Result<()> {
        to_dtype(self, dtype)
    }
}

impl<M: Module + ?Sized> ModuleExt for M {}

/// Validated model parameters and persistent buffers keyed by dotted path.
///
/// Values are detached tensor handles. The map is intentionally opaque:
/// model-produced entries retain their leaf kind, while caller-inserted
/// entries are accepted only after path validation and are treated as
/// kind-agnostic during loading.
#[derive(Clone)]
pub struct StateDict {
    entries: BTreeMap<String, StateEntry>,
}

impl StateDict {
    /// Create an empty state dictionary.
    pub fn new() -> StateDict {
        StateDict {
            entries: BTreeMap::new(),
        }
    }
    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether this state dictionary has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether `path` is present.
    pub fn contains_key(&self, path: &str) -> bool {
        self.entries.contains_key(path)
    }

    /// Borrow the tensor at `path`.
    pub fn get(&self, path: &str) -> Option<&Tensor> {
        self.entries.get(path).map(|entry| &entry.value)
    }

    /// Borrow the state paths in deterministic order.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.entries.keys()
    }

    /// Borrow the state paths as string slices in deterministic order.
    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    /// Borrow `(path, tensor)` entries in deterministic order.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Tensor)> {
        self.entries
            .iter()
            .map(|(path, entry)| (path, &entry.value))
    }

    /// Borrow tensor values in deterministic path order.
    pub fn values(&self) -> impl Iterator<Item = &Tensor> {
        self.entries.values().map(|entry| &entry.value)
    }

    /// Consume the dictionary into tensor values in deterministic path order.
    pub fn into_values(self) -> impl Iterator<Item = Tensor> {
        self.entries.into_values().map(|entry| entry.value)
    }

    /// Consume the dictionary into its paths.
    pub fn into_keys(self) -> impl Iterator<Item = String> {
        self.entries.into_keys()
    }

    /// Insert a detached tensor under a validated path.
    ///
    /// Replacing an existing model-produced entry preserves its leaf-kind
    /// metadata. A new path is kind-agnostic and can therefore be loaded only
    /// after the target module validates its ordinary tensor contract.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArg`](crate::Error::InvalidArg) if `path` is empty or
    /// contains an empty dotted segment.
    pub fn insert(&mut self, path: impl Into<String>, value: Tensor) -> Result<Option<Tensor>> {
        let path = path.into();
        validate_path(&path, "StateDict::insert")?;
        let kind = self.entries.get(&path).and_then(|entry| entry.kind);
        Ok(self
            .entries
            .insert(
                path,
                StateEntry {
                    value: value.detach(),
                    kind,
                },
            )
            .map(|entry| entry.value))
    }

    /// Remove the tensor at `path`, if present.
    pub fn remove(&mut self, path: &str) -> Option<Tensor> {
        self.entries.remove(path).map(|entry| entry.value)
    }

    /// Build a state dictionary from tensors whose leaf kind is not known.
    #[cfg(test)]
    pub(crate) fn from_tensors(tensors: BTreeMap<String, Tensor>) -> Result<StateDict> {
        let mut state = StateDict::new();
        for (path, tensor) in tensors {
            state.insert(path, tensor)?;
        }
        Ok(state)
    }
}

impl Default for StateDict {
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Index<&str> for StateDict {
    type Output = Tensor;

    fn index(&self, path: &str) -> &Self::Output {
        self.get(path)
            .unwrap_or_else(|| panic!("state dict has no entry `{path}`"))
    }
}

/// Owning iterator over a [`StateDict`]'s `(path, tensor)` pairs.
pub struct StateDictIntoIter {
    inner: std::collections::btree_map::IntoIter<String, StateEntry>,
}

impl Iterator for StateDictIntoIter {
    type Item = (String, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|(path, entry)| (path, entry.value))
    }
}

impl IntoIterator for StateDict {
    type Item = (String, Tensor);
    type IntoIter = StateDictIntoIter;

    fn into_iter(self) -> Self::IntoIter {
        StateDictIntoIter {
            inner: self.entries.into_iter(),
        }
    }
}

// ---- internals -----------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum LeafKind {
    Param,
    Buffer,
}

#[derive(Clone)]
struct StateEntry {
    value: Tensor,
    kind: Option<LeafKind>,
}

fn validate_path(path: &str, op: &'static str) -> Result<()> {
    if path.is_empty() || path.split('.').any(str::is_empty) {
        return Err(Error::invalid_arg(
            op,
            format!("state path `{path}` must contain non-empty dotted segments"),
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct LeafIdentity {
    kind: LeafKind,
    address: usize,
}

struct Mapped {
    value: Tensor,
    identity: LeafIdentity,
}

fn identity(leaf: &Leaf<'_>) -> LeafIdentity {
    match leaf {
        Leaf::Param(p) => LeafIdentity {
            kind: LeafKind::Param,
            address: (*p as *const crate::nn::Param) as usize,
        },
        Leaf::Buffer(t) => LeafIdentity {
            kind: LeafKind::Buffer,
            address: (*t as *const Tensor) as usize,
        },
    }
}

fn identity_mut(leaf: &LeafMut<'_>) -> LeafIdentity {
    match leaf {
        LeafMut::Param(p) => LeafIdentity {
            kind: LeafKind::Param,
            address: (*p as *const crate::nn::Param) as usize,
        },
        LeafMut::Buffer(t) => LeafIdentity {
            kind: LeafKind::Buffer,
            address: (*t as *const Tensor) as usize,
        },
    }
}

/// Apply `f` to every leaf value, collecting the results by dotted path — the
/// fallible half of every mutating helper here, run to completion **before**
/// anything is swapped.
///
/// Two failures end the walk:
///
/// - `f` reports one. The error is returned unchanged: the underlying op's
///   error is already loud and names the op that could not convert.
/// - The module emits the same path twice, so two distinct leaves collide.
///   A tied parameter is owned once and visited once, so a
///   collision is a `Module` implementation bug; rejecting it keeps one leaf
///   from silently overwriting the other's value on the way back in.
fn mapped<M: Module + ?Sized>(
    op: &'static str,
    module: &M,
    mut f: impl FnMut(&Tensor) -> Result<Tensor>,
) -> Result<BTreeMap<String, Mapped>> {
    let mut out: BTreeMap<String, Mapped> = BTreeMap::new();
    let mut seen = BTreeMap::new();
    let mut failure: Option<Error> = None;
    visit_all(module, &mut |path, leaf| {
        if failure.is_some() {
            return;
        }
        let leaf_identity = identity(&leaf);
        let current = match &leaf {
            Leaf::Param(p) => p.value(),
            Leaf::Buffer(t) => t,
        };
        match f(current) {
            Ok(value) => {
                let path = path.to_string();
                if seen.insert(leaf_identity, path.clone()).is_some() {
                    failure = Some(Error::invalid_arg(
                        op,
                        format!(
                            "module visits one leaf under multiple paths, including `{path}`; \
                             a state dict would duplicate that leaf"
                        ),
                    ));
                } else if out
                    .insert(
                        path.clone(),
                        Mapped {
                            value,
                            identity: leaf_identity,
                        },
                    )
                    .is_some()
                {
                    failure = Some(Error::invalid_arg(
                        op,
                        format!(
                            "module emits the path `{path}` twice: two distinct leaves \
                             collide, so a state dict could not round-trip them"
                        ),
                    ));
                }
            }
            Err(e) => failure = Some(e),
        }
    });
    match failure {
        Some(e) => Err(e),
        None => Ok(out),
    }
}

/// The commit half: validate and write `values` through one final mutable
/// walk. Every leaf is checked again immediately before its replacement is
/// applied, because a safe hand-written `Module` may make successive walks
/// disagree.
///
/// If a later leaf rejects the walk or its replacement, earlier swaps are
/// restored through a second checked walk. A stateful implementation that also
/// changes its topology during rollback returns an explicit rollback error
/// rather than silently reporting success.
fn commit<M: Module + ?Sized>(
    op: &'static str,
    module: &mut M,
    values: &BTreeMap<String, Mapped>,
) -> Result<()> {
    let mut failure: Option<Error> = None;
    let mut seen = std::collections::BTreeSet::new();
    let mut originals = BTreeMap::new();

    visit_all_mut(module, &mut |path, leaf| {
        if failure.is_some() {
            return;
        }
        let Some(expected) = values.get(path) else {
            failure = Some(Error::invalid_arg(
                op,
                format!(
                    "`{path}` is emitted by visit_mut but not by visit: the module's \
                     two walks disagree"
                ),
            ));
            return;
        };
        let actual = identity_mut(&leaf);
        if actual != expected.identity {
            failure = Some(Error::invalid_arg(
                op,
                format!(
                    "`{path}` is emitted by visit_mut for a different {} leaf than visit; \
                     the module's two walks disagree",
                    match actual.kind {
                        LeafKind::Param => "parameter",
                        LeafKind::Buffer => "buffer",
                    }
                ),
            ));
            return;
        }
        let path = path.to_string();
        if !seen.insert(path.clone()) {
            failure = Some(Error::invalid_arg(
                op,
                format!(
                    "visit_mut emits the path `{path}` twice: two distinct leaves would \
                     receive the same value"
                ),
            ));
            return;
        }

        let current = mutable_value(&leaf);
        if current.dims() != expected.value.dims() {
            failure = Some(Error::invalid_arg(
                op,
                format!(
                    "`{path}` changed shape between validation and commit: current {}, \
                     replacement {}",
                    current.shape(),
                    expected.value.shape()
                ),
            ));
            return;
        }
        originals.insert(
            path,
            Mapped {
                value: current.detach(),
                identity: actual,
            },
        );
        if let Err(error) = set_mutable_value(leaf, expected.value.detach()) {
            failure = Some(error);
        }
    });

    if failure.is_none() && seen.len() != values.len() {
        let missed = values.keys().find(|path| !seen.contains(*path));
        failure = Some(Error::invalid_arg(
            op,
            match missed {
                Some(path) => format!(
                    "`{path}` is emitted by visit but not by visit_mut: the module's \
                     two walks disagree"
                ),
                None => "the module's two walks disagree".to_string(),
            },
        ));
    }

    let Some(error) = failure else {
        return Ok(());
    };
    if originals.is_empty() {
        return Err(error);
    }
    match restore(op, module, &originals) {
        Ok(()) => Err(error),
        Err(rollback) => Err(Error::invalid_arg(
            op,
            format!("{error}; rollback failed: {rollback}"),
        )),
    }
}

fn mutable_value<'a>(leaf: &'a LeafMut<'_>) -> &'a Tensor {
    match leaf {
        LeafMut::Param(param) => param.value(),
        LeafMut::Buffer(tensor) => tensor,
    }
}

fn set_mutable_value(leaf: LeafMut<'_>, value: Tensor) -> Result<()> {
    match leaf {
        LeafMut::Param(param) => param.set(value),
        LeafMut::Buffer(tensor) => {
            *tensor = value;
            Ok(())
        }
    }
}

fn restore<M: Module + ?Sized>(
    op: &'static str,
    module: &mut M,
    originals: &BTreeMap<String, Mapped>,
) -> Result<()> {
    let mut failure: Option<Error> = None;
    let mut seen = std::collections::BTreeSet::new();
    visit_all_mut(module, &mut |path, leaf| {
        if failure.is_some() {
            return;
        }
        let Some(original) = originals.get(path) else {
            return;
        };
        let actual = identity_mut(&leaf);
        if actual != original.identity {
            failure = Some(Error::invalid_arg(
                op,
                format!("rollback found a different leaf at `{path}`"),
            ));
            return;
        }
        let path = path.to_string();
        if !seen.insert(path.clone()) {
            failure = Some(Error::invalid_arg(
                op,
                format!("rollback visited `{path}` more than once"),
            ));
            return;
        }
        if mutable_value(&leaf).dims() != original.value.dims() {
            failure = Some(Error::invalid_arg(
                op,
                format!("rollback shape changed at `{path}`"),
            ));
            return;
        }
        if let Err(error) = set_mutable_value(leaf, original.value.detach()) {
            failure = Some(error);
        }
    });
    if let Some(error) = failure {
        return Err(error);
    }
    if seen.len() != originals.len() {
        return Err(Error::invalid_arg(
            op,
            "rollback could not find every previously updated leaf",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Param;
    use crate::nn::visit::{Visitor, VisitorMut};

    fn dev() -> Device {
        Device::Cpu
    }

    fn t(data: &[f32], dims: &[usize]) -> Tensor {
        Tensor::from_vec(data.to_vec(), dims.to_vec(), &dev()).unwrap()
    }

    /// A leaf module with one parameter and one buffer.
    #[derive(rstorch::Module)]
    struct Cell {
        weight: Param,
        running: Tensor,
    }

    impl Cell {
        fn new(fill: f32) -> Cell {
            Cell {
                weight: Param::new(t(&[fill, fill, fill, fill], &[2, 2])),
                running: t(&[fill, fill], &[2]),
            }
        }
    }

    /// A nest with every derive-supported child shape.
    #[derive(rstorch::Module)]
    struct Net {
        cell: Cell,
        blocks: Vec<Cell>,
        head: Option<Cell>,
        bias: Option<Param>,
        #[module(skip)]
        #[allow(dead_code)]
        label: &'static str,
    }

    impl Net {
        fn new(fill: f32) -> Net {
            Net {
                cell: Cell::new(fill),
                blocks: vec![Cell::new(fill), Cell::new(fill)],
                head: Some(Cell::new(fill)),
                bias: Some(Param::new(t(&[fill], &[1]))),
                label: "net",
            }
        }
    }

    fn values(t: &Tensor) -> Vec<f32> {
        t.to_vec::<f32>().unwrap()
    }

    fn err_msg(e: Error) -> String {
        e.to_string()
    }

    #[test]
    fn num_params_counts_params_not_buffers() {
        // 4 cells × 4 weight elements + 1 bias = 17; the 4 × 2 buffer
        // elements do not count.
        assert_eq!(num_params(&Net::new(1.0)), 17);
    }

    /// The blanket impl is over `Module + ?Sized`, so an erased model keeps
    /// every method. A `Sized` bound would leave `&mut dyn Module` — the type
    /// the optimizer `step` takes — with no spelling at all, because the free
    /// functions behind these methods are `pub(crate)`.
    #[test]
    fn module_ext_reaches_an_erased_module() {
        let src = Net::new(2.0);
        let mut dst = Net::new(0.0);

        let erased_src: &dyn Module = &src;
        let erased_dst: &mut dyn Module = &mut dst;

        assert_eq!(erased_src.num_params(), 17);
        let state = erased_src.state_dict().unwrap();
        erased_dst.load_state_dict(&state).unwrap();
        erased_dst.to_device(&dev()).unwrap();
        erased_dst.to_dtype(DType::F32).unwrap();

        assert_eq!(
            values(dst.cell.weight.value()),
            values(src.cell.weight.value())
        );
    }

    #[test]
    fn state_dict_round_trips_values() {
        let src = Net::new(2.0);
        let mut dst = Net::new(0.0);
        assert_eq!(values(dst.cell.weight.value()), vec![0.0; 4]);

        load_state_dict(&mut dst, &state_dict(&src).unwrap()).unwrap();

        let a = state_dict(&src).unwrap();
        let b = state_dict(&dst).unwrap();
        assert_eq!(a.keys().collect::<Vec<_>>(), b.keys().collect::<Vec<_>>());
        for (path, want) in a.iter() {
            assert_eq!(values(want), values(&b[path]), "at {path}");
        }
    }

    #[test]
    fn load_rejects_missing_key() {
        let mut m = Net::new(0.0);
        let mut state = state_dict(&Net::new(1.0)).unwrap();
        assert!(state.remove("cell.running").is_some());
        let msg = err_msg(load_state_dict(&mut m, &state).unwrap_err());
        assert!(msg.contains("missing key `cell.running`"), "{msg}");
        // Nothing was swapped.
        assert_eq!(values(m.cell.weight.value()), vec![0.0; 4]);
    }

    #[test]
    fn load_rejects_unexpected_key() {
        let mut m = Net::new(0.0);
        let mut state = state_dict(&Net::new(1.0)).unwrap();
        state
            .insert("cell.extra".to_string(), t(&[9.0], &[1]))
            .unwrap();
        let msg = err_msg(load_state_dict(&mut m, &state).unwrap_err());
        assert!(msg.contains("unexpected key `cell.extra`"), "{msg}");
        assert_eq!(values(m.cell.weight.value()), vec![0.0; 4]);
    }

    #[test]
    fn load_rejects_shape_mismatch() {
        let mut m = Net::new(0.0);
        let mut state = state_dict(&Net::new(1.0)).unwrap();
        state
            .insert("cell.weight".to_string(), t(&[1.0, 2.0], &[2]))
            .unwrap();
        let msg = err_msg(load_state_dict(&mut m, &state).unwrap_err());
        assert!(msg.contains("`cell.weight` shape mismatch"), "{msg}");
        assert!(msg.contains("[2]") && msg.contains("[2, 2]"), "{msg}");
        assert_eq!(values(m.cell.weight.value()), vec![0.0; 4]);
    }

    #[test]
    fn load_rejects_dtype_mismatch() {
        let mut m = Net::new(0.0);
        let mut state = state_dict(&Net::new(1.0)).unwrap();
        let i64s = Tensor::from_vec(vec![1i64, 2, 3, 4], [2, 2], &dev()).unwrap();
        state.insert("cell.weight".to_string(), i64s).unwrap();
        let msg = err_msg(load_state_dict(&mut m, &state).unwrap_err());
        assert!(msg.contains("`cell.weight` dtype mismatch"), "{msg}");
        assert_eq!(values(m.cell.weight.value()), vec![0.0; 4]);
    }

    #[test]
    fn load_detaches_incoming_values() {
        let mut m = Net::new(0.0);
        let mut state = state_dict(&Net::new(1.0)).unwrap();
        // A value that carries a live graph must not smuggle it into a Param.
        let traced = state["cell.weight"].traced().unwrap();
        state.insert("cell.weight".to_string(), traced).unwrap();
        load_state_dict(&mut m, &state).unwrap();
        assert!(m.cell.weight.value().backward().is_err());
    }

    #[test]
    fn load_is_all_or_nothing_across_the_walk() {
        // `bias` sorts before `cell.weight`, so a valid `bias` would be
        // swapped first by a naive one-pass implementation.
        let mut m = Net::new(0.0);
        let mut state = state_dict(&Net::new(1.0)).unwrap();
        state
            .insert("cell.weight".to_string(), t(&[1.0], &[1]))
            .unwrap();
        assert!(load_state_dict(&mut m, &state).is_err());
        assert_eq!(values(m.bias.as_ref().unwrap().value()), vec![0.0]);
    }

    #[test]
    fn duplicate_paths_are_rejected() {
        struct Collide {
            a: Param,
            b: Param,
        }
        impl Module for Collide {
            fn visit(&self, v: &mut Visitor) {
                v.param("w", &self.a);
                v.param("w", &self.b);
            }
            fn visit_mut(&mut self, v: &mut VisitorMut) {
                v.param("w", &mut self.a);
                v.param("w", &mut self.b);
            }
        }
        let mut m = Collide {
            a: Param::new(t(&[1.0], &[1])),
            b: Param::new(t(&[2.0], &[1])),
        };
        let state = StateDict::from_tensors(BTreeMap::from([(String::from("w"), t(&[9.0], &[1]))]))
            .unwrap();
        let msg = err_msg(load_state_dict(&mut m, &state).unwrap_err());
        assert!(msg.contains("emits the path `w` twice"), "{msg}");
    }

    #[test]
    fn state_dict_rejects_duplicate_paths_without_panicking() {
        struct Collide {
            a: Param,
            b: Param,
        }
        impl Module for Collide {
            fn visit(&self, v: &mut Visitor) {
                v.param("w", &self.a);
                v.param("w", &self.b);
            }
            fn visit_mut(&mut self, v: &mut VisitorMut) {
                v.param("w", &mut self.a);
                v.param("w", &mut self.b);
            }
        }
        let m = Collide {
            a: Param::new(t(&[1.0], &[1])),
            b: Param::new(t(&[2.0], &[1])),
        };
        assert!(state_dict(&m).is_err());
    }

    #[test]
    fn state_dict_validates_paths_and_leaf_kinds() {
        let mut state = state_dict(&Net::new(1.0)).unwrap();
        assert!(state.insert("", t(&[1.0], &[1])).is_err());
        assert!(state.insert("cell..extra", t(&[1.0], &[1])).is_err());

        #[derive(rstorch::Module)]
        struct ParamLeaf {
            leaf: Param,
        }
        #[derive(rstorch::Module)]
        struct BufferLeaf {
            leaf: Tensor,
        }
        let source = ParamLeaf {
            leaf: Param::new(t(&[1.0], &[1])),
        };
        let mut target = BufferLeaf {
            leaf: t(&[0.0], &[1]),
        };
        let error = target
            .load_state_dict(&state_dict(&source).unwrap())
            .unwrap_err();
        assert!(err_msg(error).contains("leaf kind"));
    }

    #[test]
    fn state_dict_detaches_buffer_values() {
        #[derive(rstorch::Module)]
        struct BufferOnly {
            buffer: Tensor,
        }
        let buffer = t(&[1.0], &[1]).traced().unwrap();
        let model = BufferOnly { buffer };
        let state = state_dict(&model).unwrap();
        assert!(state["buffer"].backward().is_err());
    }

    #[test]
    fn load_rolls_back_when_final_mutable_walk_drifts() {
        struct Drifting {
            a: Param,
            b: Param,
        }
        impl Module for Drifting {
            fn visit(&self, v: &mut Visitor) {
                v.param("a", &self.a);
                v.param("b", &self.b);
            }

            fn visit_mut(&mut self, v: &mut VisitorMut) {
                v.param("a", &mut self.a);
                v.param("unexpected", &mut self.b);
            }
        }

        let source = Drifting {
            a: Param::new(t(&[1.0], &[1])),
            b: Param::new(t(&[2.0], &[1])),
        };
        let mut target = Drifting {
            a: Param::new(t(&[0.0], &[1])),
            b: Param::new(t(&[0.0], &[1])),
        };
        let state = state_dict(&source).unwrap();
        let error = target.load_state_dict(&state).unwrap_err();
        assert!(err_msg(error).contains("not by visit"));
        assert_eq!(values(target.a.value()), vec![0.0]);
        assert_eq!(values(target.b.value()), vec![0.0]);
    }

    #[test]
    fn disagreeing_leaf_identities_are_rejected_before_swapping() {
        struct Swapped {
            a: Param,
            b: Param,
        }
        impl Module for Swapped {
            fn visit(&self, v: &mut Visitor) {
                v.param("w", &self.a);
            }
            fn visit_mut(&mut self, v: &mut VisitorMut) {
                v.param("w", &mut self.b);
            }
        }
        let mut m = Swapped {
            a: Param::new(t(&[0.0], &[1])),
            b: Param::new(t(&[0.0], &[1])),
        };
        let state = StateDict::from_tensors(BTreeMap::from([(String::from("w"), t(&[1.0], &[1]))]))
            .unwrap();
        let msg = err_msg(load_state_dict(&mut m, &state).unwrap_err());
        assert!(msg.contains("different parameter leaf"), "{msg}");
        assert_eq!(values(m.a.value()), vec![0.0]);
        assert_eq!(values(m.b.value()), vec![0.0]);
    }

    #[test]
    fn disagreeing_walks_are_rejected_before_anything_is_swapped() {
        // A hand-written `Module` whose mutable walk forgets a leaf: the load
        // would silently leave `b` untrained. (`#[derive(Module)]` cannot
        // produce this; a hand-written impl can.)
        struct Lopsided {
            a: Param,
            b: Param,
        }
        impl Module for Lopsided {
            fn visit(&self, v: &mut Visitor) {
                v.param("a", &self.a);
                v.param("b", &self.b);
            }
            fn visit_mut(&mut self, v: &mut VisitorMut) {
                v.param("a", &mut self.a);
            }
        }
        let mut m = Lopsided {
            a: Param::new(t(&[0.0], &[1])),
            b: Param::new(t(&[0.0], &[1])),
        };
        let mut state = state_dict(&m).unwrap();
        state.insert("a".to_string(), t(&[1.0], &[1])).unwrap();
        state.insert("b".to_string(), t(&[2.0], &[1])).unwrap();

        let msg = err_msg(load_state_dict(&mut m, &state).unwrap_err());
        assert!(
            msg.contains("`b` is emitted by visit but not by visit_mut"),
            "{msg}"
        );
        // `a` was reachable by both walks, and still was not swapped.
        assert_eq!(values(m.a.value()), vec![0.0]);
    }

    #[test]
    fn to_device_cpu_is_a_faithful_walk() {
        let mut m = Net::new(3.0);
        to_device(&mut m, &Device::Cpu).unwrap();
        for (path, v) in state_dict(&m).unwrap() {
            assert_eq!(v.device(), Device::Cpu, "at {path}");
            assert!(values(&v).iter().all(|&x| x == 3.0), "at {path}");
        }
    }

    #[test]
    fn to_dtype_leaves_non_float_leaves_alone() {
        #[derive(rstorch::Module)]
        struct Mixed {
            weight: Param,
            indices: Tensor,
        }
        let mut m = Mixed {
            weight: Param::new(t(&[1.0, 2.0], &[2])),
            indices: Tensor::from_vec(vec![0i64, 1], [2], &dev()).unwrap(),
        };
        // F32 -> F32 is the identity lane; the I64 buffer must survive it.
        to_dtype(&mut m, DType::F32).unwrap();
        assert_eq!(m.weight.value().dtype(), DType::F32);
        assert_eq!(m.indices.dtype(), DType::I64);
        assert_eq!(m.indices.to_vec::<i64>().unwrap(), vec![0, 1]);
    }

    #[test]
    fn to_dtype_rejects_non_float_targets() {
        let mut m = Net::new(1.0);
        let msg = err_msg(to_dtype(&mut m, DType::I64).unwrap_err());
        assert!(msg.contains("must be a float dtype"), "{msg}");
    }

    #[test]
    fn to_dtype_failure_leaves_the_model_untouched() {
        // F32 -> F64 is a deferred cast lane, so the conversion pass fails
        // before anything is committed.
        let mut m = Net::new(1.0);
        assert!(matches!(
            to_dtype(&mut m, DType::F64),
            Err(Error::Unsupported { .. })
        ));
        assert_eq!(m.cell.weight.value().dtype(), DType::F32);
        assert_eq!(values(m.cell.weight.value()), vec![1.0; 4]);
    }

    #[test]
    fn replication_recipe_shares_no_future_updates() {
        let src = Net::new(5.0);
        let mut replica = Net::new(0.0);
        load_state_dict(&mut replica, &state_dict(&src).unwrap()).unwrap();
        // A later "optimizer step" on the replica writes a new value and does
        // not disturb the original.
        replica.cell.weight.set(t(&[7.0; 4], &[2, 2])).unwrap();
        assert_eq!(values(src.cell.weight.value()), vec![5.0; 4]);
        assert_eq!(values(replica.cell.weight.value()), vec![7.0; 4]);
    }
}
