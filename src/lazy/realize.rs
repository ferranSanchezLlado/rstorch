//! Iterative realization of pending expression DAGs.

use super::expr::{Expr, Operand};
use crate::backend::{BinaryOp, CmpOp, UnaryOp, View, dispatch};
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::storage::{PendingStorage, Storage};
use std::collections::HashSet;
use std::sync::Arc;

/// Realize a pending root and every pending operand in post-order.
///
/// The explicit worklist is required: a long scalar chain is a normal shape
/// for recurrent models and must not consume call-stack frames.
pub(crate) fn ensure_ready(root: Arc<PendingStorage>) -> Result<()> {
    let relabel = match &*root.expr {
        Expr::Copy {
            op, relabel: true, ..
        } => Some(*op),
        _ => None,
    };
    let result: Result<()> = (|| {
        if root.cache.get().is_some() {
            return Ok(());
        }

        enum Work {
            Enter(Arc<PendingStorage>),
            Exit(Arc<PendingStorage>),
        }

        let mut stack = vec![Work::Enter(root)];
        let mut visited: HashSet<usize> = HashSet::new();
        while let Some(work) = stack.pop() {
            match work {
                Work::Enter(node) => {
                    if node.cache.get().is_some() {
                        continue;
                    }
                    // `dense_f32` refuses unresolved pending leaves, so this
                    // fast path cannot recurse. It can therefore fuse a
                    // private straight-line chain before its children are
                    // visited, while the unresolved/shared case falls through
                    // to the iterative post-order walk below.
                    if let Some(result) = super::fuse::try_fuse(&node)? {
                        let _ = node.cache.set(result);
                        continue;
                    }
                    let id = Arc::as_ptr(&node) as usize;
                    if !visited.insert(id) {
                        continue;
                    }
                    stack.push(Work::Exit(Arc::clone(&node)));
                    let mut children = Vec::new();
                    node.expr.for_each_operand(|storage| {
                        if let Storage::Pending(child) = storage
                            && child.cache.get().is_none()
                        {
                            children.push(child.clone());
                        }
                    });
                    for child in children.into_iter().rev() {
                        stack.push(Work::Enter(child));
                    }
                }
                Work::Exit(node) => {
                    if node.cache.get().is_some() {
                        continue;
                    }
                    let result = match super::fuse::try_fuse(&node)? {
                        Some(result) => result,
                        None => evaluate(&node.expr)?,
                    };
                    // A concurrent realization may have won the race. Its value
                    // is equivalent; dropping this duplicate is safe because the
                    // expression is pure and failed computations are not cached.
                    let _ = node.cache.set(result);
                }
            }
        }
        Ok(())
    })();
    result.map_err(|error| match relabel {
        Some(op) => error.with_op(op),
        None => error,
    })
}

/// Evaluate one expression whose pending children have already been cached.
pub(crate) fn evaluate(expr: &Expr) -> Result<Storage> {
    let op = expr.op();
    match expr {
        Expr::Unary { kind, x, .. } => {
            let view = ready_view(x)?;
            map_backend(op, dispatch::backend(view.device()).unary(*kind, view))
        }
        Expr::Binary { kind, a, b, .. } => {
            let lhs = ready_view(a)?;
            let rhs = ready_view(b)?;
            map_backend(op, dispatch::backend(lhs.device()).binary(*kind, lhs, rhs))
        }
        Expr::Scalar { kind, x, k, .. } => {
            let view = ready_view(x)?;
            map_backend(
                op,
                dispatch::backend(view.device()).binary_scalar(*kind, view, *k),
            )
        }
        Expr::Compare { kind, a, b, .. } => {
            let lhs = ready_view(a)?;
            let rhs = ready_view(b)?;
            map_backend(op, dispatch::backend(lhs.device()).compare(*kind, lhs, rhs))
        }
        Expr::Where { c, t, f, .. } => {
            let cond = ready_view(c)?;
            let on_true = ready_view(t)?;
            let on_false = ready_view(f)?;
            map_backend(
                op,
                dispatch::backend(on_true.device()).where_cond(cond, on_true, on_false),
            )
        }
        Expr::Masked { x, mask, value, .. } => {
            let value_view = ready_view(x)?;
            let mask_view = ready_view(mask)?;
            map_backend(
                op,
                dispatch::backend(value_view.device()).masked_fill(value_view, mask_view, *value),
            )
        }
        Expr::Cast { x, to, .. } => {
            let view = ready_view(x)?;
            map_backend(op, dispatch::backend(view.device()).cast(view, *to))
        }
        Expr::Copy {
            x, relabel: true, ..
        } => {
            // Relabeling exists only to preserve the public scalar-left
            // operation name. It is an identity in value space: keeping the
            // child's storage avoids a needless device copy and lets the
            // wrapper share the child's allocation.
            Ok(x.storage.ready()?.clone())
        }
        Expr::Copy {
            x, relabel: false, ..
        } => {
            let view = ready_view(x)?;
            map_backend(op, dispatch::backend(view.device()).copy_strided(view))
        }
        Expr::Const {
            value,
            dtype,
            device,
            dims,
            ..
        } => {
            let layout = Layout::contiguous(dims.clone()).map_err(|error| error.with_op(op))?;
            map_backend(
                op,
                dispatch::backend(*device).full(layout.num_elements(), *dtype, *value),
            )
        }
    }
}

fn map_backend<T>(op: &'static str, result: Result<T>) -> Result<T> {
    result.map_err(|error| error.with_op(op))
}

fn ready_view(operand: &Operand) -> Result<View<'_>> {
    let storage = operand.storage.ready()?;
    Ok(View::new(storage, &operand.layout))
}

#[allow(dead_code)]
fn _keep_operation_types_linked(_: BinaryOp, _: CmpOp, _: UnaryOp, _: Error) {}
