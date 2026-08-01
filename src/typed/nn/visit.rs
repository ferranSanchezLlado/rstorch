use super::{
    LeafContract, LeafKind, Module, RuntimeModuleAdapter, StateEntry, TypedBuffer, TypedLeaf,
    TypedLeafMut, TypedParam, TypedStateDict, TypedVisitor, TypedVisitorMut,
};
use crate::typed::{FloatElement, TypedTensor};
use crate::{Error, Result, Tensor};
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

impl<'a> TypedVisitor<'a> {
    fn new(sink: &'a mut dyn FnMut(&str, TypedLeaf<'_>)) -> Self {
        Self {
            path: String::new(),
            sink,
            error: None,
        }
    }

    /// Emits a trainable parameter at the current dotted prefix.
    pub fn param<T>(&mut self, name: &str, param: &TypedParam<T>)
    where
        T: TypedTensor,
        T::Elem: FloatElement,
    {
        if self.error.is_some() {
            return;
        }
        if let Err(error) = param.validate("TypedVisitor::param") {
            self.error = Some(error);
            return;
        }
        let path = join(&self.path, name);
        (self.sink)(&path, TypedLeaf::Param(&param.runtime, param.contract()));
    }

    /// Emits a persistent buffer at the current dotted prefix.
    pub fn buffer<T: TypedTensor>(&mut self, name: &str, buffer: &TypedBuffer<T>) {
        if self.error.is_some() {
            return;
        }
        if let Err(error) = buffer.validate("TypedVisitor::buffer") {
            self.error = Some(error);
            return;
        }
        let path = join(&self.path, name);
        (self.sink)(&path, TypedLeaf::Buffer(&buffer.runtime, buffer.contract()));
    }

    /// Descends into a typed child module and restores the previous prefix.
    pub fn module<M: Module + ?Sized>(&mut self, name: &str, child: &M) {
        let saved = self.path.len();
        if !self.path.is_empty() {
            self.path.push('.');
        }
        self.path.push_str(name);
        child.visit(self);
        self.path.truncate(saved);
    }
}

impl<'a> TypedVisitorMut<'a> {
    fn new(sink: &'a mut dyn FnMut(&str, TypedLeafMut<'_>)) -> Self {
        Self {
            path: String::new(),
            sink,
            error: None,
        }
    }

    /// Emits a mutable trainable parameter without exposing it to the caller.
    pub fn param<T>(&mut self, name: &str, param: &mut TypedParam<T>)
    where
        T: TypedTensor,
        T::Elem: FloatElement,
    {
        if self.error.is_some() {
            return;
        }
        if let Err(error) = param.validate("TypedVisitorMut::param") {
            self.error = Some(error);
            return;
        }
        let path = join(&self.path, name);
        let contract = param.contract();
        (self.sink)(&path, TypedLeafMut::Param(&mut param.runtime, contract));
    }

    /// Emits a mutable persistent buffer without exposing it to the caller.
    pub fn buffer<T: TypedTensor>(&mut self, name: &str, buffer: &mut TypedBuffer<T>) {
        if self.error.is_some() {
            return;
        }
        if let Err(error) = buffer.validate("TypedVisitorMut::buffer") {
            self.error = Some(error);
            return;
        }
        let path = join(&self.path, name);
        let contract = buffer.contract();
        (self.sink)(&path, TypedLeafMut::Buffer(&mut buffer.runtime, contract));
    }

    /// Descends into a mutable typed child and restores the previous prefix.
    pub fn module<M: Module + ?Sized>(&mut self, name: &str, child: &mut M) {
        let saved = self.path.len();
        if !self.path.is_empty() {
            self.path.push('.');
        }
        self.path.push_str(name);
        child.visit_mut(self);
        self.path.truncate(saved);
    }
}

#[derive(Clone)]
struct WalkLeaf {
    path: String,
    kind: LeafKind,
    identity: usize,
    dims: Vec<usize>,
    contract: LeafContract,
}

fn malformed(op: &'static str, msg: impl Into<String>) -> Error {
    Error::InvalidArg {
        op,
        msg: msg.into(),
    }
}

fn same_contract(a: &LeafContract, b: &LeafContract) -> bool {
    a.kind == b.kind
        && a.rank == b.rank
        && a.markers == b.markers
        && a.dtype == b.dtype
        && a.placement == b.placement
        && Arc::ptr_eq(&a.binding, &b.binding)
}

fn collect_read<M: Module + ?Sized>(
    module: &M,
    with_values: bool,
    op: &'static str,
) -> Result<(Vec<WalkLeaf>, Vec<Tensor>)> {
    let mut leaves = Vec::new();
    let mut values = Vec::new();
    let mut paths = HashSet::new();
    let mut identities = HashSet::new();
    let mut error = None;
    let visitor_error = {
        let mut sink = |path: &str, leaf: TypedLeaf<'_>| {
            let (kind, identity, value, contract) = match leaf {
                TypedLeaf::Param(param, contract) => (
                    LeafKind::Param,
                    param as *const crate::nn::Param as usize,
                    param.value(),
                    contract,
                ),
                TypedLeaf::Buffer(buffer, contract) => (
                    LeafKind::Buffer,
                    buffer as *const Tensor as usize,
                    buffer,
                    contract,
                ),
            };
            if !paths.insert(path.to_string()) {
                error = Some(malformed(op, format!("duplicate state path {path:?}")));
                return;
            }
            if !identities.insert(identity) {
                error = Some(malformed(
                    op,
                    format!("duplicate leaf identity at {path:?}"),
                ));
                return;
            }
            leaves.push(WalkLeaf {
                path: path.to_string(),
                kind,
                identity,
                dims: value.dims().to_vec(),
                contract,
            });
            if with_values {
                values.push(value.detach());
            }
        };
        let mut visitor = TypedVisitor::new(&mut sink);
        module.visit(&mut visitor);
        visitor.error.take()
    };
    if let Some(error) = visitor_error {
        return Err(error);
    }
    if let Some(error) = error {
        return Err(error);
    }
    Ok((leaves, values))
}

fn collect_mut<M: Module + ?Sized>(
    module: &mut M,
    with_values: bool,
) -> Result<(Vec<WalkLeaf>, Vec<Tensor>)> {
    const OP: &str = "typed::nn::load_state_dict";
    let mut leaves = Vec::new();
    let mut values = Vec::new();
    let mut paths = HashSet::new();
    let mut identities = HashSet::new();
    let mut error = None;
    let visitor_error = {
        let mut sink = |path: &str, leaf: TypedLeafMut<'_>| {
            let (kind, identity, value, contract) = match leaf {
                TypedLeafMut::Param(param, contract) => (
                    LeafKind::Param,
                    param as *mut crate::nn::Param as usize,
                    param.value(),
                    contract,
                ),
                TypedLeafMut::Buffer(buffer, contract) => (
                    LeafKind::Buffer,
                    buffer as *mut Tensor as usize,
                    &*buffer,
                    contract,
                ),
            };
            if !paths.insert(path.to_string()) {
                error = Some(malformed(
                    OP,
                    format!("duplicate mutable state path {path:?}"),
                ));
                return;
            }
            if !identities.insert(identity) {
                error = Some(malformed(
                    OP,
                    format!("duplicate mutable leaf identity at {path:?}"),
                ));
                return;
            }
            leaves.push(WalkLeaf {
                path: path.to_string(),
                kind,
                identity,
                dims: value.dims().to_vec(),
                contract,
            });
            if with_values {
                values.push(value.detach());
            }
        };
        let mut visitor = TypedVisitorMut::new(&mut sink);
        module.visit_mut(&mut visitor);
        visitor.error.take()
    };
    if let Some(error) = visitor_error {
        return Err(error);
    }
    if let Some(error) = error {
        return Err(error);
    }
    Ok((leaves, values))
}

fn walks_agree(read: &[WalkLeaf], mutable: &[WalkLeaf]) -> bool {
    read.len() == mutable.len()
        && read.iter().zip(mutable).all(|(a, b)| {
            a.path == b.path
                && a.kind == b.kind
                && a.identity == b.identity
                && a.dims == b.dims
                && same_contract(&a.contract, &b.contract)
        })
}

struct ApplyFailure {
    error: Error,
    changed: Vec<usize>,
}

fn apply_checked<M: Module + ?Sized>(
    module: &mut M,
    expected: &[WalkLeaf],
    values: &[Tensor],
    selected: Option<&HashSet<usize>>,
) -> std::result::Result<(), ApplyFailure> {
    const OP: &str = "typed::nn::load_state_dict";
    let mut index = 0;
    let mut paths = HashSet::new();
    let mut identities = HashSet::new();
    let mut changed = Vec::new();
    let mut error = None;
    let visitor_error = {
        let mut sink = |path: &str, leaf: TypedLeafMut<'_>| {
            if error.is_some() {
                return;
            }
            let Some(schema) = expected.get(index) else {
                error = Some(malformed(OP, "mutable commit walk emitted an extra leaf"));
                return;
            };
            let (kind, identity, dims, contract) = match &leaf {
                TypedLeafMut::Param(param, contract) => (
                    LeafKind::Param,
                    &**param as *const crate::nn::Param as usize,
                    param.value().dims(),
                    contract,
                ),
                TypedLeafMut::Buffer(buffer, contract) => (
                    LeafKind::Buffer,
                    &**buffer as *const Tensor as usize,
                    buffer.dims(),
                    contract,
                ),
            };
            if !paths.insert(path.to_string()) {
                error = Some(malformed(OP, format!("duplicate commit path {path:?}")));
                return;
            }
            if !identities.insert(identity) {
                error = Some(malformed(
                    OP,
                    format!("duplicate commit leaf identity at {path:?}"),
                ));
                return;
            }
            if path != schema.path
                || kind != schema.kind
                || identity != schema.identity
                || dims != schema.dims
                || !same_contract(contract, &schema.contract)
            {
                error = Some(malformed(
                    OP,
                    format!("mutable commit leaf differs from validated schema at {path:?}"),
                ));
                return;
            }
            if selected.is_none_or(|indices| indices.contains(&index)) {
                let Some(value) = values.get(index).cloned() else {
                    error = Some(malformed(OP, "replacement value is missing"));
                    return;
                };
                match leaf {
                    TypedLeafMut::Param(param, _) => {
                        if let Err(set_error) = param.set(value) {
                            error = Some(set_error);
                            return;
                        }
                    }
                    TypedLeafMut::Buffer(buffer, _) => *buffer = value,
                }
                changed.push(index);
            }
            index += 1;
        };
        let mut visitor = TypedVisitorMut::new(&mut sink);
        module.visit_mut(&mut visitor);
        visitor.error.take()
    };
    if let Some(error) = visitor_error.or(error) {
        return Err(ApplyFailure { error, changed });
    }
    if index != expected.len() {
        return Err(ApplyFailure {
            error: malformed(OP, "mutable commit walk omitted a leaf"),
            changed,
        });
    }
    Ok(())
}

fn rollback<M: Module + ?Sized>(
    module: &mut M,
    expected: &[WalkLeaf],
    originals: &[Tensor],
    changed: &[usize],
) -> Result<()> {
    if changed.is_empty() {
        return Ok(());
    }
    let (rollback_walk, _) = collect_mut(module, false)?;
    if !walks_agree(expected, &rollback_walk) {
        return Err(malformed(
            "typed::nn::load_state_dict",
            "mutable rollback walk differs from validated schema",
        ));
    }
    let selected = changed.iter().copied().collect::<HashSet<_>>();
    apply_checked(module, expected, originals, Some(&selected)).map_err(|failure| failure.error)
}

/// Collects an opaque, detached snapshot after validating the complete walk.
pub fn state_dict<M: Module + ?Sized>(module: &M) -> Result<TypedStateDict> {
    let (walk, values) = collect_read(module, true, "typed::nn::state_dict")?;
    let entries = walk
        .into_iter()
        .zip(values)
        .map(|(leaf, value)| {
            (
                leaf.path,
                StateEntry {
                    value,
                    contract: leaf.contract,
                },
            )
        })
        .collect();
    Ok(TypedStateDict { entries })
}

impl TypedStateDict {
    /// CT47's read-only leaf-kind query; it exposes no runtime leaf or value.
    pub(crate) fn is_param_path(&self, path: &str) -> bool {
        self.entries
            .get(path)
            .is_some_and(|entry| entry.contract.kind == LeafKind::Param)
    }
}

/// Checks and stages the entire state before replacing any target leaf.
pub fn load_state_dict<M: Module + ?Sized>(module: &mut M, state: &TypedStateDict) -> Result<()> {
    const OP: &str = "typed::nn::load_state_dict";
    let (read, _) = collect_read(module, false, OP)?;
    let (mutable, _) = collect_mut(module, false)?;
    if !walks_agree(&read, &mutable) {
        return Err(malformed(OP, "read-only and mutable module walks disagree"));
    }
    let target_paths = read
        .iter()
        .map(|leaf| leaf.path.as_str())
        .collect::<HashSet<_>>();
    if let Some(path) = state
        .entries
        .keys()
        .find(|path| !target_paths.contains(path.as_str()))
    {
        return Err(malformed(OP, format!("unexpected state path {path:?}")));
    }

    let mut staged = BTreeMap::new();
    for leaf in &read {
        let entry = state
            .entries
            .get(&leaf.path)
            .ok_or_else(|| malformed(OP, format!("missing state path {:?}", leaf.path)))?;
        if !same_contract(&leaf.contract, &entry.contract) {
            return Err(malformed(
                OP,
                format!("typed contract mismatch at {:?}", leaf.path),
            ));
        }
        if entry.contract.markers.len() != entry.contract.rank
            || entry.value.rank() != entry.contract.rank
            || entry.value.dims() != leaf.dims
            || entry.value.dtype() != entry.contract.dtype
            || entry.value.device() != entry.contract.binding.device
        {
            return Err(malformed(
                OP,
                format!("invalid staged value at {:?}", leaf.path),
            ));
        }
        for (&marker, &actual) in entry.contract.markers.iter().zip(entry.value.dims()) {
            if marker != crate::typed::DYN && marker != actual {
                return Err(malformed(
                    OP,
                    format!("shape marker mismatch at {:?}", leaf.path),
                ));
            }
        }
        staged.insert(leaf.path.clone(), entry.value.detach());
    }

    let (precommit, originals) = collect_mut(module, true)?;
    if !walks_agree(&mutable, &precommit) {
        return Err(malformed(OP, "pre-commit mutable walk changed"));
    }
    let replacements = mutable
        .iter()
        .map(|leaf| {
            staged
                .remove(&leaf.path)
                .ok_or_else(|| malformed(OP, format!("staged path {:?} disappeared", leaf.path)))
        })
        .collect::<Result<Vec<_>>>()?;
    match apply_checked(module, &mutable, &replacements, None) {
        Ok(()) => Ok(()),
        Err(failure) => match rollback(module, &mutable, &originals, &failure.changed) {
            Ok(()) => Err(failure.error),
            Err(rollback_error) => Err(malformed(
                OP,
                format!(
                    "commit rejected ({}); rollback could not restore every changed leaf ({rollback_error})",
                    failure.error
                ),
            )),
        },
    }
}

impl<M: Module + ?Sized> crate::nn::Module for RuntimeModuleAdapter<'_, M> {
    fn visit(&self, visitor: &mut crate::nn::Visitor<'_>) {
        let mut sink = |path: &str, leaf: TypedLeaf<'_>| match leaf {
            TypedLeaf::Param(param, _) => visitor.param(path, param),
            TypedLeaf::Buffer(buffer, _) => visitor.buffer(path, buffer),
        };
        let mut typed = TypedVisitor::new(&mut sink);
        self.module.visit(&mut typed);
    }

    fn visit_mut(&mut self, visitor: &mut crate::nn::VisitorMut<'_>) {
        let mut sink = |path: &str, leaf: TypedLeafMut<'_>| match leaf {
            TypedLeafMut::Param(param, _) => visitor.param(path, param),
            TypedLeafMut::Buffer(buffer, _) => visitor.buffer(path, buffer),
        };
        let mut typed = TypedVisitorMut::new(&mut sink);
        self.module.visit_mut(&mut typed);
    }
}

impl<'a, M: Module + ?Sized> RuntimeModuleAdapter<'a, M> {
    #[allow(dead_code)]
    pub(crate) fn new(module: &'a mut M) -> Self {
        Self { module }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed::sealed::DeviceBinding;
    use crate::typed::{Cpu, DYN, DeviceCtx, Placement, Tensor1};
    use crate::{DType, Device};
    use std::any::TypeId;

    struct Pair<P: Placement = Cpu> {
        a: TypedParam<Tensor1<1, f32, P>>,
        child: Child<P>,
    }

    struct Child<P: Placement> {
        b: TypedParam<Tensor1<1, f32, P>>,
        running: TypedBuffer<Tensor1<1, f32, P>>,
    }

    impl<P: Placement> Module for Pair<P> {
        fn visit(&self, visitor: &mut TypedVisitor<'_>) {
            visitor.param("a", &self.a);
            visitor.module("child", &self.child);
        }

        fn visit_mut(&mut self, visitor: &mut TypedVisitorMut<'_>) {
            visitor.param("a", &mut self.a);
            visitor.module("child", &mut self.child);
        }
    }

    impl<P: Placement> Module for Child<P> {
        fn visit(&self, visitor: &mut TypedVisitor<'_>) {
            visitor.param("b", &self.b);
            visitor.buffer("running", &self.running);
        }

        fn visit_mut(&mut self, visitor: &mut TypedVisitorMut<'_>) {
            visitor.param("b", &mut self.b);
            visitor.buffer("running", &mut self.running);
        }
    }

    fn tensor<P: Placement>(value: f32, ctx: &DeviceCtx<P>) -> Tensor1<1, f32, P> {
        Tensor1::from_vec(vec![value], [1], ctx).unwrap()
    }

    fn pair<P: Placement>(base: f32, ctx: &DeviceCtx<P>) -> Pair<P> {
        Pair {
            a: TypedParam::new(tensor(base, ctx)).unwrap(),
            child: Child {
                b: TypedParam::new(tensor(base + 1.0, ctx)).unwrap(),
                running: TypedBuffer::new(tensor(base + 2.0, ctx)).unwrap(),
            },
        }
    }

    fn values<P: Placement>(model: &Pair<P>) -> Vec<f32> {
        vec![
            model.a.value().unwrap().to_vec().unwrap()[0],
            model.child.b.value().unwrap().to_vec().unwrap()[0],
            model.child.running.value().unwrap().to_vec().unwrap()[0],
        ]
    }

    #[test]
    fn dotted_paths_restore_prefix_and_match_the_dynamic_adapter() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let mut model = pair(1.0, &ctx);
        let typed = state_dict(&model).unwrap();
        assert_eq!(
            typed.paths().collect::<Vec<_>>(),
            vec!["a", "child.b", "child.running"]
        );
        let adapter = RuntimeModuleAdapter::new(&mut model);
        assert_eq!(
            crate::nn::state_dict(&adapter)
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            typed.paths().collect::<Vec<_>>()
        );
    }

    #[test]
    fn checked_load_is_transactional_on_a_late_value_failure() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let source = pair(10.0, &ctx);
        let mut state = state_dict(&source).unwrap();
        state.entries.get_mut("child.running").unwrap().value =
            Tensor::zeros([2], DType::F32, &Device::Cpu).unwrap();
        let mut target = pair(1.0, &ctx);
        let before = values(&target);
        assert!(load_state_dict(&mut target, &state).is_err());
        assert_eq!(values(&target), before);
    }

    struct DynamicPair {
        first: TypedParam<Tensor1<DYN>>,
        second: TypedParam<Tensor1<DYN>>,
    }

    impl Module for DynamicPair {
        fn visit(&self, visitor: &mut TypedVisitor<'_>) {
            visitor.param("first", &self.first);
            visitor.param("second", &self.second);
        }

        fn visit_mut(&mut self, visitor: &mut TypedVisitorMut<'_>) {
            visitor.param("first", &mut self.first);
            visitor.param("second", &mut self.second);
        }
    }

    #[test]
    fn checked_load_stages_actual_dynamic_dimensions_before_commit() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let source = DynamicPair {
            first: TypedParam::new(Tensor1::from_vec(vec![10.0], [1], &ctx).unwrap()).unwrap(),
            second: TypedParam::new(Tensor1::from_vec(vec![20.0, 21.0], [2], &ctx).unwrap())
                .unwrap(),
        };
        let state = state_dict(&source).unwrap();
        let mut target = DynamicPair {
            first: TypedParam::new(Tensor1::from_vec(vec![1.0], [1], &ctx).unwrap()).unwrap(),
            second: TypedParam::new(Tensor1::from_vec(vec![2.0], [1], &ctx).unwrap()).unwrap(),
        };

        assert!(load_state_dict(&mut target, &state).is_err());
        assert_eq!(target.first.value().unwrap().to_vec().unwrap(), vec![1.0]);
        assert_eq!(target.second.value().unwrap().to_vec().unwrap(), vec![2.0]);
    }

    #[test]
    fn successful_load_detaches_and_replaces_every_leaf() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let source = pair(10.0, &ctx);
        let state = state_dict(&source).unwrap();
        let mut target = pair(1.0, &ctx);
        load_state_dict(&mut target, &state).unwrap();
        assert_eq!(values(&target), vec![10.0, 11.0, 12.0]);
        assert!(target.a.value().unwrap().as_dynamic().backward().is_err());
    }

    #[test]
    fn missing_and_extra_paths_are_rejected_without_mutation() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let source = pair(10.0, &ctx);
        let mut missing = state_dict(&source).unwrap();
        missing.entries.remove("child.running");
        let mut target = pair(1.0, &ctx);
        let before = values(&target);
        assert!(load_state_dict(&mut target, &missing).is_err());
        assert_eq!(values(&target), before);

        let mut extra = state_dict(&source).unwrap();
        let entry = extra.entries.remove("child.running").unwrap();
        extra.entries.insert("unexpected".to_string(), entry);
        assert!(load_state_dict(&mut target, &extra).is_err());
        assert_eq!(values(&target), before);
    }

    #[derive(Clone, Copy)]
    enum CommitMutation {
        SwapBufferType,
        DuplicateIdentity,
        OmitLeaf,
        AddLeaf,
    }

    struct StatefulBuffers {
        calls: usize,
        mutation: CommitMutation,
        a: TypedBuffer<Tensor1<1>>,
        b: TypedBuffer<Tensor1<1>>,
        wide: TypedBuffer<Tensor1<2>>,
    }

    impl Module for StatefulBuffers {
        fn visit(&self, visitor: &mut TypedVisitor<'_>) {
            visitor.buffer("a", &self.a);
            visitor.buffer("b", &self.b);
        }

        fn visit_mut(&mut self, visitor: &mut TypedVisitorMut<'_>) {
            self.calls += 1;
            if self.calls != 3 {
                visitor.buffer("a", &mut self.a);
                visitor.buffer("b", &mut self.b);
                return;
            }
            match self.mutation {
                CommitMutation::SwapBufferType => {
                    visitor.buffer("a", &mut self.wide);
                    visitor.buffer("b", &mut self.b);
                }
                CommitMutation::DuplicateIdentity => {
                    visitor.buffer("a", &mut self.a);
                    visitor.buffer("b", &mut self.a);
                }
                CommitMutation::OmitLeaf => visitor.buffer("a", &mut self.a),
                CommitMutation::AddLeaf => {
                    visitor.buffer("a", &mut self.a);
                    visitor.buffer("b", &mut self.b);
                    visitor.buffer("extra", &mut self.wide);
                }
            }
        }
    }

    fn stateful_buffers(base: f32, mutation: CommitMutation) -> StatefulBuffers {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        StatefulBuffers {
            calls: 0,
            mutation,
            a: TypedBuffer::new(tensor(base, &ctx)).unwrap(),
            b: TypedBuffer::new(tensor(base + 1.0, &ctx)).unwrap(),
            wide: TypedBuffer::new(
                Tensor1::from_vec(vec![base + 8.0, base + 9.0], [2], &ctx).unwrap(),
            )
            .unwrap(),
        }
    }

    fn stateful_buffer_values(model: &StatefulBuffers) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        (
            model.a.value().unwrap().to_vec().unwrap(),
            model.b.value().unwrap().to_vec().unwrap(),
            model.wide.value().unwrap().to_vec().unwrap(),
        )
    }

    #[test]
    fn stateful_commit_cannot_swap_a_same_path_buffer_of_another_type() {
        let source = stateful_buffers(10.0, CommitMutation::SwapBufferType);
        let state = state_dict(&source).unwrap();
        let mut target = stateful_buffers(1.0, CommitMutation::SwapBufferType);
        let before = stateful_buffer_values(&target);

        assert!(load_state_dict(&mut target, &state).is_err());
        assert_eq!(stateful_buffer_values(&target), before);
        assert_eq!(target.a.value().unwrap().dims(), [1]);
        assert_eq!(target.wide.value().unwrap().dims(), [2]);
    }

    #[test]
    fn stateful_commit_duplicate_identity_rolls_back_an_earlier_replacement() {
        let source = stateful_buffers(10.0, CommitMutation::DuplicateIdentity);
        let state = state_dict(&source).unwrap();
        let mut target = stateful_buffers(1.0, CommitMutation::DuplicateIdentity);
        let before = stateful_buffer_values(&target);

        assert!(load_state_dict(&mut target, &state).is_err());
        assert_eq!(stateful_buffer_values(&target), before);
    }

    #[test]
    fn stateful_commit_omitted_and_extra_leaves_roll_back_all_replacements() {
        for mutation in [CommitMutation::OmitLeaf, CommitMutation::AddLeaf] {
            let source = stateful_buffers(10.0, mutation);
            let state = state_dict(&source).unwrap();
            let mut target = stateful_buffers(1.0, mutation);
            let before = stateful_buffer_values(&target);

            assert!(load_state_dict(&mut target, &state).is_err());
            assert_eq!(stateful_buffer_values(&target), before);
        }
    }

    struct StatefulParam {
        calls: usize,
        first: TypedBuffer<Tensor1<1>>,
        second: TypedParam<Tensor1<DYN>>,
        wrong: TypedParam<Tensor1<DYN>>,
    }

    impl Module for StatefulParam {
        fn visit(&self, visitor: &mut TypedVisitor<'_>) {
            visitor.buffer("first", &self.first);
            visitor.param("second", &self.second);
        }

        fn visit_mut(&mut self, visitor: &mut TypedVisitorMut<'_>) {
            self.calls += 1;
            visitor.buffer("first", &mut self.first);
            if self.calls == 3 {
                visitor.param("second", &mut self.wrong);
            } else {
                visitor.param("second", &mut self.second);
            }
        }
    }

    fn stateful_param(base: f32) -> StatefulParam {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        StatefulParam {
            calls: 0,
            first: TypedBuffer::new(tensor(base, &ctx)).unwrap(),
            second: TypedParam::new(Tensor1::from_vec(vec![base + 1.0], [1], &ctx).unwrap())
                .unwrap(),
            wrong: TypedParam::new(
                Tensor1::from_vec(vec![base + 8.0, base + 9.0], [2], &ctx).unwrap(),
            )
            .unwrap(),
        }
    }

    #[test]
    fn late_stateful_param_mismatch_rolls_back_the_earlier_buffer() {
        let source = stateful_param(10.0);
        let state = state_dict(&source).unwrap();
        let mut target = stateful_param(1.0);
        let before = (
            target.first.value().unwrap().to_vec().unwrap(),
            target.second.value().unwrap().to_vec().unwrap(),
            target.wrong.value().unwrap().to_vec().unwrap(),
        );

        assert!(load_state_dict(&mut target, &state).is_err());
        assert_eq!(target.first.value().unwrap().to_vec().unwrap(), before.0);
        assert_eq!(target.second.value().unwrap().to_vec().unwrap(), before.1);
        assert_eq!(target.wrong.value().unwrap().to_vec().unwrap(), before.2);
        assert_eq!(target.second.value().unwrap().dims(), [1]);
        assert_eq!(target.wrong.value().unwrap().dims(), [2]);
    }

    fn assert_rejected_unchanged(state: &TypedStateDict, target: &mut Pair) {
        let before = values(target);
        assert!(load_state_dict(target, state).is_err());
        assert_eq!(values(target), before);
    }

    #[test]
    fn every_erased_contract_field_and_binding_identity_is_checked() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let source = pair(10.0, &ctx);

        let check = |mutate: &dyn Fn(&mut StateEntry)| {
            let mut state = state_dict(&source).unwrap();
            mutate(state.entries.get_mut("child.running").unwrap());
            assert_rejected_unchanged(&state, &mut pair(1.0, &ctx));
        };
        check(&|entry| entry.contract.kind = LeafKind::Param);
        check(&|entry| entry.contract.rank = 2);
        check(&|entry| entry.contract.markers = &[DYN]);
        check(&|entry| entry.contract.dtype = DType::F64);
        check(&|entry| entry.contract.placement = TypeId::of::<Main>());
        check(&|entry| {
            entry.contract.binding = Arc::new(DeviceBinding {
                device: Device::Cpu,
            });
        });
    }

    struct DuplicatePath {
        a: TypedParam<Tensor1<1>>,
        b: TypedParam<Tensor1<1>>,
    }

    impl Module for DuplicatePath {
        fn visit(&self, visitor: &mut TypedVisitor<'_>) {
            visitor.param("same", &self.a);
            visitor.param("same", &self.b);
        }

        fn visit_mut(&mut self, visitor: &mut TypedVisitorMut<'_>) {
            visitor.param("same", &mut self.a);
            visitor.param("same", &mut self.b);
        }
    }

    struct DuplicateIdentity {
        value: TypedParam<Tensor1<1>>,
    }

    impl Module for DuplicateIdentity {
        fn visit(&self, visitor: &mut TypedVisitor<'_>) {
            visitor.param("first", &self.value);
            visitor.param("second", &self.value);
        }

        fn visit_mut(&mut self, visitor: &mut TypedVisitorMut<'_>) {
            visitor.param("first", &mut self.value);
        }
    }

    #[test]
    fn duplicate_paths_and_leaf_identities_are_rejected() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let duplicate_path = DuplicatePath {
            a: TypedParam::new(tensor(1.0, &ctx)).unwrap(),
            b: TypedParam::new(tensor(2.0, &ctx)).unwrap(),
        };
        assert!(state_dict(&duplicate_path).is_err());
        let duplicate_identity = DuplicateIdentity {
            value: TypedParam::new(tensor(1.0, &ctx)).unwrap(),
        };
        assert!(state_dict(&duplicate_identity).is_err());
    }

    struct Disagreeing {
        a: TypedParam<Tensor1<1>>,
        b: TypedParam<Tensor1<1>>,
    }

    impl Module for Disagreeing {
        fn visit(&self, visitor: &mut TypedVisitor<'_>) {
            visitor.param("a", &self.a);
        }

        fn visit_mut(&mut self, visitor: &mut TypedVisitorMut<'_>) {
            visitor.param("b", &mut self.b);
        }
    }

    #[test]
    fn disagreeing_read_and_mutable_walks_reject_before_mutation() {
        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let source = DuplicateIdentity {
            value: TypedParam::new(tensor(9.0, &ctx)).unwrap(),
        };
        struct Single<'a>(&'a DuplicateIdentity);
        impl Module for Single<'_> {
            fn visit(&self, visitor: &mut TypedVisitor<'_>) {
                visitor.param("a", &self.0.value);
            }
            fn visit_mut(&mut self, _visitor: &mut TypedVisitorMut<'_>) {}
        }
        let state = state_dict(&Single(&source)).unwrap();
        let mut target = Disagreeing {
            a: TypedParam::new(tensor(1.0, &ctx)).unwrap(),
            b: TypedParam::new(tensor(2.0, &ctx)).unwrap(),
        };
        assert!(load_state_dict(&mut target, &state).is_err());
        assert_eq!(target.a.value().unwrap().to_vec().unwrap(), vec![1.0]);
        assert_eq!(target.b.value().unwrap().to_vec().unwrap(), vec![2.0]);
    }

    struct Main;
    impl Placement for Main {}
    struct Auxiliary;
    impl Placement for Auxiliary {}

    #[test]
    fn logical_placements_do_not_mix_on_the_same_physical_device() {
        let main = DeviceCtx::<Main>::bind(Device::Cpu).unwrap();
        let auxiliary = DeviceCtx::<Auxiliary>::bind(Device::Cpu).unwrap();
        let source = pair(7.0, &main);
        let state = state_dict(&source).unwrap();
        let mut target = pair(1.0, &auxiliary);
        assert!(load_state_dict(&mut target, &state).is_err());
        assert_eq!(values(&target), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn runtime_adapter_keeps_tied_gradient_identity_optimizer_viable() {
        struct One {
            value: TypedParam<Tensor1<1>>,
        }
        impl Module for One {
            fn visit(&self, visitor: &mut TypedVisitor<'_>) {
                visitor.param("value", &self.value);
            }
            fn visit_mut(&mut self, visitor: &mut TypedVisitorMut<'_>) {
                visitor.param("value", &mut self.value);
            }
        }

        let ctx = DeviceCtx::<Cpu>::cpu().unwrap();
        let mut model = One {
            value: TypedParam::new(tensor(2.0, &ctx)).unwrap(),
        };
        let value = model.value.get(super::super::Mode::TRAIN).unwrap();
        let loss = value
            .as_dynamic()
            .mul(value.as_dynamic())
            .unwrap()
            .sum_all()
            .unwrap();
        let grads = loss.backward().unwrap();
        assert_eq!(state_dict(&model).unwrap().len(), 1);
        assert_eq!(
            model.value.grad_from(&grads).unwrap().to_vec().unwrap(),
            vec![4.0]
        );
        let mut adapter = RuntimeModuleAdapter::new(&mut model);
        crate::optim::Sgd::new(0.1)
            .step(&mut adapter, grads)
            .unwrap();
        assert_eq!(model.value.value().unwrap().to_vec().unwrap(), vec![1.6]);
    }
}
