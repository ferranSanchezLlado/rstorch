//! CPU executor for straight-line element-wise chains.
//!
//! The executor handles dense chains over every CPU dtype whose eager
//! element-wise contract is closed under the chain (`F16`, `BF16`, `F32`,
//! `F64`, and the supported `I64` arithmetic/unary subset). Strided bases,
//! pending secondary operands, unsupported integer operations, and
//! non-elementwise expressions fall back to the ordinary backend realizer.
//! The fallback is deliberate: fusion is an optimization, never a second
//! implementation of unsupported semantics.

use super::expr::{CpuStage, Expr, Operand};
use crate::Result;
use crate::backend::cpu::elementwise;
use crate::backend::{BinaryOp, UnaryOp};
use crate::dtype::DType;
use crate::storage::{CpuStorage, PendingStorage, Storage};
use std::sync::Arc;

const TILE: usize = 1024;

/// CPU element type supported by the straight-line executor.
trait FusedElement: Copy + Send + Sync + 'static {
    fn slice(storage: &Storage) -> Option<&[Self]>;
    fn storage(values: Vec<Self>) -> Storage;
    fn supports_unary(kind: UnaryOp) -> bool;
    fn supports_scalar(kind: BinaryOp) -> bool;
    fn supports_binary(kind: BinaryOp) -> bool;
    fn unary(kind: UnaryOp, value: Self) -> Self;
    fn scalar(kind: BinaryOp, value: Self, scalar: f64) -> Self;
    fn binary(kind: BinaryOp, lhs: Self, rhs: Self) -> Self;
}

impl FusedElement for f32 {
    fn slice(storage: &Storage) -> Option<&[Self]> {
        let Storage::Cpu(CpuStorage::F32(values)) = storage else {
            return None;
        };
        Some(values)
    }

    fn storage(values: Vec<Self>) -> Storage {
        Storage::Cpu(CpuStorage::F32(Arc::new(values)))
    }

    fn supports_unary(_: UnaryOp) -> bool {
        true
    }

    fn supports_scalar(_: BinaryOp) -> bool {
        true
    }

    fn supports_binary(_: BinaryOp) -> bool {
        true
    }

    fn unary(kind: UnaryOp, value: Self) -> Self {
        elementwise::fused_unary_f32(kind, value)
    }

    fn scalar(kind: BinaryOp, value: Self, scalar: f64) -> Self {
        elementwise::fused_binary_scalar_f32(kind, value, scalar)
    }

    fn binary(kind: BinaryOp, lhs: Self, rhs: Self) -> Self {
        elementwise::fused_binary_f32(kind, lhs, rhs)
    }
}

impl FusedElement for f64 {
    fn slice(storage: &Storage) -> Option<&[Self]> {
        let Storage::Cpu(CpuStorage::F64(values)) = storage else {
            return None;
        };
        Some(values)
    }

    fn storage(values: Vec<Self>) -> Storage {
        Storage::Cpu(CpuStorage::F64(Arc::new(values)))
    }

    fn supports_unary(_: UnaryOp) -> bool {
        true
    }

    fn supports_scalar(_: BinaryOp) -> bool {
        true
    }

    fn supports_binary(_: BinaryOp) -> bool {
        true
    }

    fn unary(kind: UnaryOp, value: Self) -> Self {
        elementwise::fused_unary_f64(kind, value)
    }

    fn scalar(kind: BinaryOp, value: Self, scalar: f64) -> Self {
        elementwise::fused_binary_scalar_f64(kind, value, scalar)
    }

    fn binary(kind: BinaryOp, lhs: Self, rhs: Self) -> Self {
        elementwise::fused_binary_f64(kind, lhs, rhs)
    }
}

impl FusedElement for half::f16 {
    fn slice(storage: &Storage) -> Option<&[Self]> {
        let Storage::Cpu(CpuStorage::F16(values)) = storage else {
            return None;
        };
        Some(values)
    }

    fn storage(values: Vec<Self>) -> Storage {
        Storage::Cpu(CpuStorage::F16(Arc::new(values)))
    }

    fn supports_unary(_: UnaryOp) -> bool {
        true
    }

    fn supports_scalar(_: BinaryOp) -> bool {
        true
    }

    fn supports_binary(_: BinaryOp) -> bool {
        true
    }

    fn unary(kind: UnaryOp, value: Self) -> Self {
        Self::from_f64(elementwise::fused_unary_f64(kind, value.to_f64()))
    }

    fn scalar(kind: BinaryOp, value: Self, scalar: f64) -> Self {
        Self::from_f32(elementwise::fused_binary_f32(
            kind,
            value.to_f32(),
            scalar as f32,
        ))
    }

    fn binary(kind: BinaryOp, lhs: Self, rhs: Self) -> Self {
        Self::from_f32(elementwise::fused_binary_f32(
            kind,
            lhs.to_f32(),
            rhs.to_f32(),
        ))
    }
}

impl FusedElement for half::bf16 {
    fn slice(storage: &Storage) -> Option<&[Self]> {
        let Storage::Cpu(CpuStorage::BF16(values)) = storage else {
            return None;
        };
        Some(values)
    }

    fn storage(values: Vec<Self>) -> Storage {
        Storage::Cpu(CpuStorage::BF16(Arc::new(values)))
    }

    fn supports_unary(_: UnaryOp) -> bool {
        true
    }

    fn supports_scalar(_: BinaryOp) -> bool {
        true
    }

    fn supports_binary(_: BinaryOp) -> bool {
        true
    }

    fn unary(kind: UnaryOp, value: Self) -> Self {
        Self::from_f64(elementwise::fused_unary_f64(kind, value.to_f64()))
    }

    fn scalar(kind: BinaryOp, value: Self, scalar: f64) -> Self {
        Self::from_f32(elementwise::fused_binary_f32(
            kind,
            value.to_f32(),
            scalar as f32,
        ))
    }

    fn binary(kind: BinaryOp, lhs: Self, rhs: Self) -> Self {
        Self::from_f32(elementwise::fused_binary_f32(
            kind,
            lhs.to_f32(),
            rhs.to_f32(),
        ))
    }
}

impl FusedElement for i64 {
    fn slice(storage: &Storage) -> Option<&[Self]> {
        let Storage::Cpu(CpuStorage::I64(values)) = storage else {
            return None;
        };
        Some(values)
    }

    fn storage(values: Vec<Self>) -> Storage {
        Storage::Cpu(CpuStorage::I64(Arc::new(values)))
    }

    fn supports_unary(kind: UnaryOp) -> bool {
        matches!(kind, UnaryOp::Neg | UnaryOp::Abs)
    }

    fn supports_scalar(kind: BinaryOp) -> bool {
        !matches!(kind, BinaryOp::Pow)
    }

    fn supports_binary(kind: BinaryOp) -> bool {
        !matches!(kind, BinaryOp::Pow)
    }

    fn unary(kind: UnaryOp, value: Self) -> Self {
        elementwise::fused_unary_i64(kind, value)
    }

    fn scalar(kind: BinaryOp, value: Self, scalar: f64) -> Self {
        elementwise::fused_binary_scalar_i64(kind, value, scalar)
    }

    fn binary(kind: BinaryOp, lhs: Self, rhs: Self) -> Self {
        elementwise::fused_binary_i64(kind, lhs, rhs)
    }
}

/// Try to execute a pending root as one dense CPU chain.
pub(crate) fn try_fuse(root: &PendingStorage) -> Result<Option<Storage>> {
    if root.device != crate::Device::Cpu {
        return Ok(None);
    }
    match root.dtype {
        DType::F16 => try_fuse_typed::<half::f16>(root),
        DType::BF16 => try_fuse_typed::<half::bf16>(root),
        DType::F32 => try_fuse_typed::<f32>(root),
        DType::F64 => try_fuse_typed::<f64>(root),
        DType::I64 => try_fuse_typed::<i64>(root),
        DType::Bool => Ok(None),
    }
}

fn try_fuse_typed<E: FusedElement>(root: &PendingStorage) -> Result<Option<Storage>> {
    let mut stages = Stages::new();
    let mut current: &Expr = &root.expr;
    let base: &Operand;
    loop {
        let Some(stage) = current.cpu_stage() else {
            return Ok(None);
        };
        match stage {
            CpuStage::Unary { kind, x } => {
                if !E::supports_unary(kind) {
                    return Ok(None);
                }
                match &x.storage {
                    Storage::Pending(child) if child.cache.get().is_none() => {
                        // A pending storage may also be exposed through a
                        // zero-copy shape view. Only follow the chain when
                        // the consumer sees exactly the layout the child
                        // produced; otherwise materialization must preserve
                        // the view's logical order.
                        if child.layout != x.layout {
                            return Ok(None);
                        }
                        stages.push_op(StageOp::Unary(kind));
                        if Arc::strong_count(child) > 1 {
                            stages.push_spill(child.clone());
                        }
                        current = &child.expr;
                    }
                    _ => {
                        stages.push_op(StageOp::Unary(kind));
                        base = x;
                        break;
                    }
                }
            }
            CpuStage::Scalar { kind, x, k } => {
                if !E::supports_scalar(kind) {
                    return Ok(None);
                }
                match &x.storage {
                    Storage::Pending(child) if child.cache.get().is_none() => {
                        if child.layout != x.layout {
                            return Ok(None);
                        }
                        stages.push_op(StageOp::Scalar(kind, k));
                        if Arc::strong_count(child) > 1 {
                            stages.push_spill(child.clone());
                        }
                        current = &child.expr;
                    }
                    _ => {
                        stages.push_op(StageOp::Scalar(kind, k));
                        base = x;
                        break;
                    }
                }
            }
            CpuStage::Binary { kind, a, b } => {
                if !E::supports_binary(kind) || b.storage.is_pending() {
                    // A pending secondary operand makes this a DAG rather
                    // than a cheap straight-line chain.
                    return Ok(None);
                }
                let Some(rhs) = dense_slice::<E>(b)? else {
                    return Ok(None);
                };
                if rhs.len() != root.len {
                    return Ok(None);
                }
                match &a.storage {
                    Storage::Pending(child) if child.cache.get().is_none() => {
                        if child.layout != a.layout {
                            return Ok(None);
                        }
                        stages.push_op(StageOp::Binary(kind, rhs));
                        if Arc::strong_count(child) > 1 {
                            stages.push_spill(child.clone());
                        }
                        current = &child.expr;
                    }
                    _ => {
                        stages.push_op(StageOp::Binary(kind, rhs));
                        base = a;
                        break;
                    }
                }
            }
        }
    }

    let Some(mut values) = dense_values::<E>(base)? else {
        return Ok(None);
    };
    debug_assert_eq!(values.len(), root.len);
    if let Some(stages) = stages.spilled() {
        // A spill must publish the complete child buffer while forwarding the
        // same value stream to the parent. The sequential path keeps the
        // child cache and parent result ordered without a second backend read.
        for stage in stages.iter().rev() {
            match stage {
                Stage::Op(StageOp::Unary(kind)) => apply_unary_tile(&mut values, *kind),
                Stage::Op(StageOp::Scalar(kind, scalar)) => {
                    apply_scalar_tile(&mut values, *kind, *scalar);
                }
                Stage::Op(StageOp::Binary(kind, rhs)) => {
                    apply_binary_tile(&mut values, *kind, rhs);
                }
                Stage::Spill(node) => {
                    let _ = node.cache.set(E::storage(values.clone()));
                }
            }
        }
    } else {
        let cost = stages
            .len()
            .saturating_mul(crate::backend::parallel::STREAMING_COST)
            .max(1);
        crate::backend::parallel::for_each_window_mut(&mut values, TILE, cost, |start, tile| {
            for stage in stages.iter_ops_rev() {
                match stage {
                    StageOp::Unary(kind) => apply_unary_tile(tile, kind),
                    StageOp::Scalar(kind, scalar) => apply_scalar_tile(tile, kind, scalar),
                    StageOp::Binary(kind, rhs) => {
                        apply_binary_tile(tile, kind, &rhs[start..start + tile.len()]);
                    }
                }
            }
        });
    }
    debug_assert_eq!(values.len(), root.len);
    Ok(Some(E::storage(values)))
}

const INLINE_STAGES: usize = 16;

/// Keep the common short chain on the stack. A `Vec` here used to allocate on
/// every realization, even though the cap makes most fused chains much shorter
/// than the maximum. Longer chains spill only their uncommon tail to the heap.
struct Stages<'a, E: Copy> {
    inline: [Option<StageOp<'a, E>>; INLINE_STAGES],
    len: usize,
    overflow: Vec<StageOp<'a, E>>,
    spilled: Option<Vec<Stage<'a, E>>>,
}

impl<'a, E: Copy> Stages<'a, E> {
    fn new() -> Self {
        Self {
            inline: [const { None }; INLINE_STAGES],
            len: 0,
            overflow: Vec::new(),
            spilled: None,
        }
    }

    fn push_op(&mut self, stage: StageOp<'a, E>) {
        if let Some(stages) = self.spilled.as_mut() {
            stages.push(Stage::Op(stage));
        } else if self.len < INLINE_STAGES {
            self.inline[self.len] = Some(stage);
        } else {
            self.overflow.push(stage);
        }
        self.len += 1;
    }

    fn push_spill(&mut self, node: Arc<PendingStorage>) {
        if self.spilled.is_none() {
            let mut stages = Vec::with_capacity(self.len + 1);
            stages.extend(
                self.inline[..self.len.min(INLINE_STAGES)]
                    .iter()
                    .filter_map(|stage| stage.map(Stage::Op)),
            );
            stages.extend(self.overflow.drain(..).map(Stage::Op));
            self.spilled = Some(stages);
        }
        self.spilled
            .as_mut()
            .expect("spill storage initialized")
            .push(Stage::Spill(node));
    }

    fn len(&self) -> usize {
        self.len
    }

    fn iter_ops_rev(&self) -> impl Iterator<Item = StageOp<'a, E>> + '_ {
        self.overflow.iter().rev().copied().chain(
            self.inline[..self.len.min(INLINE_STAGES)]
                .iter()
                .rev()
                .filter_map(|stage| *stage),
        )
    }

    fn spilled(&self) -> Option<&[Stage<'a, E>]> {
        self.spilled.as_deref()
    }
}

#[derive(Clone, Copy)]
enum StageOp<'a, E: Copy> {
    Unary(UnaryOp),
    Scalar(BinaryOp, f64),
    Binary(BinaryOp, &'a [E]),
}

enum Stage<'a, E: Copy> {
    Op(StageOp<'a, E>),
    Spill(Arc<PendingStorage>),
}
fn dense_values<E: FusedElement>(operand: &Operand) -> Result<Option<Vec<E>>> {
    Ok(dense_slice::<E>(operand)?.map(<[E]>::to_vec))
}

fn dense_slice<E: FusedElement>(operand: &Operand) -> Result<Option<&[E]>> {
    // Do not recursively realize an unresolved child here. The iterative
    // worklist visits children before retrying the parent; returning `None`
    // lets the parent stay on that worklist instead of reintroducing the
    // stack overflow this module is meant to avoid.
    let storage = match &operand.storage {
        Storage::Pending(node) => match node.cache.get() {
            Some(storage) => storage,
            None => return Ok(None),
        },
        storage => storage,
    };
    let Some(values) = E::slice(storage) else {
        return Ok(None);
    };
    let layout = &operand.layout;
    let Some((start, end)) = dense_range(layout) else {
        return Ok(None);
    };
    let Some(slice) = values.get(start..end) else {
        return Ok(None);
    };
    Ok(Some(slice))
}

/// Return the physical range of a layout whose logical walk is one dense run.
/// Unlike `Layout::is_contiguous`, this accepts a non-zero offset, which is
/// exactly the representation produced by narrowing an outer dense axis.
fn dense_range(layout: &crate::layout::Layout) -> Option<(usize, usize)> {
    let mut expected_stride = 1usize;
    for (&dim, &stride) in layout.dims().iter().zip(layout.strides()).rev() {
        if dim == 1 {
            continue;
        }
        if stride != expected_stride {
            return None;
        }
        expected_stride = expected_stride.checked_mul(dim)?;
    }
    let end = layout.offset().checked_add(layout.num_elements())?;
    Some((layout.offset(), end))
}

fn apply_unary_tile<E: FusedElement>(values: &mut [E], kind: UnaryOp) {
    macro_rules! apply_unary_const {
        ($op:expr) => {
            for value in values.iter_mut() {
                *value = E::unary($op, *value);
            }
        };
    }
    match kind {
        UnaryOp::Relu => apply_unary_const!(UnaryOp::Relu),
        UnaryOp::Gelu => apply_unary_const!(UnaryOp::Gelu),
        UnaryOp::Exp => apply_unary_const!(UnaryOp::Exp),
        UnaryOp::Ln => apply_unary_const!(UnaryOp::Ln),
        UnaryOp::Sqrt => apply_unary_const!(UnaryOp::Sqrt),
        UnaryOp::Tanh => apply_unary_const!(UnaryOp::Tanh),
        UnaryOp::Sigmoid => apply_unary_const!(UnaryOp::Sigmoid),
        UnaryOp::Neg => apply_unary_const!(UnaryOp::Neg),
        UnaryOp::Abs => apply_unary_const!(UnaryOp::Abs),
        UnaryOp::Sign => apply_unary_const!(UnaryOp::Sign),
        UnaryOp::Recip => apply_unary_const!(UnaryOp::Recip),
        UnaryOp::Floor => apply_unary_const!(UnaryOp::Floor),
        UnaryOp::Ceil => apply_unary_const!(UnaryOp::Ceil),
        UnaryOp::Round => apply_unary_const!(UnaryOp::Round),
        UnaryOp::Erf => apply_unary_const!(UnaryOp::Erf),
    }
}

fn apply_scalar_tile<E: FusedElement>(values: &mut [E], kind: BinaryOp, scalar: f64) {
    macro_rules! apply_scalar_const {
        ($op:expr) => {
            for value in values.iter_mut() {
                *value = E::scalar($op, *value, scalar);
            }
        };
    }
    match kind {
        BinaryOp::Add => apply_scalar_const!(BinaryOp::Add),
        BinaryOp::Sub => apply_scalar_const!(BinaryOp::Sub),
        BinaryOp::Mul => apply_scalar_const!(BinaryOp::Mul),
        BinaryOp::Div => apply_scalar_const!(BinaryOp::Div),
        BinaryOp::Maximum => apply_scalar_const!(BinaryOp::Maximum),
        BinaryOp::Minimum => apply_scalar_const!(BinaryOp::Minimum),
        BinaryOp::Pow => apply_scalar_const!(BinaryOp::Pow),
    }
}

fn apply_binary_tile<E: FusedElement>(values: &mut [E], kind: BinaryOp, rhs: &[E]) {
    macro_rules! apply_binary_const {
        ($op:expr) => {
            for (value, &other) in values.iter_mut().zip(rhs) {
                *value = E::binary($op, *value, other);
            }
        };
    }
    match kind {
        BinaryOp::Add => apply_binary_const!(BinaryOp::Add),
        BinaryOp::Sub => apply_binary_const!(BinaryOp::Sub),
        BinaryOp::Mul => apply_binary_const!(BinaryOp::Mul),
        BinaryOp::Div => apply_binary_const!(BinaryOp::Div),
        BinaryOp::Maximum => apply_binary_const!(BinaryOp::Maximum),
        BinaryOp::Minimum => apply_binary_const!(BinaryOp::Minimum),
        BinaryOp::Pow => apply_binary_const!(BinaryOp::Pow),
    }
}
