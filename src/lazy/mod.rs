//! Deferred element-wise execution.
//!
//! This namespace is experimental and outside the dynamic 1.x stability
//! guarantee. The engine is deliberately opt-in at runtime and defaults to
//! eager execution. [`set_fusion`](crate::lazy::set_fusion) changes only the
//! current thread and returns a guard that restores the previous setting.

mod expr;
mod fuse;
#[cfg(test)]
mod harness;
pub(crate) mod realize;
mod switch;

use crate::Result;
use crate::backend::{BinaryOp, CmpOp, UnaryOp};
use crate::device::Device;
use crate::dtype::DType;
use crate::layout::Layout;
use crate::storage::{PendingStorage, Storage};
pub(crate) use expr::Expr;
use std::collections::HashSet;
use std::sync::Arc;
pub use switch::{Fusion, FusionGuard, default_fusion, fusion, set_fusion};

pub(crate) use expr::Expr as LazyExpr;

/// Number of pending nodes allowed to remain live before a new result is
/// materialized immediately. The cap is on DAG nodes, not expression depth.
pub(crate) const MAX_PENDING_NODES: usize = 256;

pub(crate) fn enabled() -> bool {
    switch::enabled()
}

/// Realize pending roots and synchronize only devices touched by them.
///
/// A pending tensor can outlive the guard that created it, so realization does
/// not depend on the current thread-local switch. If every root is already
/// backed by ordinary storage, this is a no-op; enabling fusion must not turn
/// an otherwise asynchronous backend path into an unconditional synchronize.
pub(crate) fn flush_tensors(tensors: &[&crate::tensor::Tensor]) -> Result<()> {
    let mut devices = HashSet::new();
    for tensor in tensors {
        if !tensor.storage().is_pending() {
            continue;
        }
        tensor.storage().ready()?;
        devices.insert(tensor.device());
    }
    for device in devices {
        device.synchronize()?;
    }
    Ok(())
}

fn pending(dtype: DType, device: Device, layout: Layout, expr: LazyExpr) -> Result<Storage> {
    let node_count = count_nodes(&expr);
    let node = Arc::new(PendingStorage::new(dtype, device, layout, node_count, expr));
    if node_count > MAX_PENDING_NODES {
        // The cap is itself a materialization boundary. Use the same
        // iterative realizer as an explicit host boundary rather than
        // evaluating the root directly and recursively resolving a child.
        realize::ensure_ready(Arc::clone(&node))?;
        return node
            .cache
            .get()
            .cloned()
            .ok_or_else(|| crate::error::Error::Backend {
                op: node.expr.op(),
                msg: "realizer completed without publishing storage".to_string(),
            });
    }
    Ok(Storage::Pending(node))
}

fn count_nodes(expr: &LazyExpr) -> usize {
    // This is deliberately an upper bound. The node cap only decides when to
    // materialize, so counting a shared child twice is safe and avoids a
    // HashSet/Vec walk for every multi-operand expression.
    let mut total: usize = 1;
    expr.for_each_operand(|storage| {
        if let Storage::Pending(node) = storage {
            total = total.saturating_add(node.pending_nodes());
        }
    });
    total
}

pub(crate) fn unary(
    op: &'static str,
    kind: UnaryOp,
    x: &Storage,
    layout: &Layout,
    dtype: DType,
    device: Device,
) -> Result<Storage> {
    pending(
        dtype,
        device,
        layout.clone(),
        LazyExpr::Unary {
            op,
            kind,
            x: expr::Operand::new(x, layout),
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn binary(
    op: &'static str,
    kind: BinaryOp,
    a: &Storage,
    al: &Layout,
    b: &Storage,
    bl: &Layout,
    dtype: DType,
    device: Device,
) -> Result<Storage> {
    pending(
        dtype,
        device,
        al.clone(),
        LazyExpr::Binary {
            op,
            kind,
            a: expr::Operand::new(a, al),
            b: expr::Operand::new(b, bl),
        },
    )
}

pub(crate) fn scalar(
    op: &'static str,
    kind: BinaryOp,
    x: &Storage,
    layout: &Layout,
    k: f64,
    dtype: DType,
    device: Device,
) -> Result<Storage> {
    pending(
        dtype,
        device,
        layout.clone(),
        LazyExpr::Scalar {
            op,
            kind,
            x: expr::Operand::new(x, layout),
            k,
        },
    )
}

pub(crate) fn compare(
    op: &'static str,
    kind: CmpOp,
    a: &Storage,
    al: &Layout,
    b: &Storage,
    bl: &Layout,
    device: Device,
) -> Result<Storage> {
    pending(
        DType::Bool,
        device,
        al.clone(),
        LazyExpr::Compare {
            op,
            kind,
            a: expr::Operand::new(a, al),
            b: expr::Operand::new(b, bl),
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn where_cond(
    op: &'static str,
    c: &Storage,
    cl: &Layout,
    t: &Storage,
    tl: &Layout,
    f: &Storage,
    fl: &Layout,
    dtype: DType,
    device: Device,
) -> Result<Storage> {
    pending(
        dtype,
        device,
        tl.clone(),
        LazyExpr::Where {
            op,
            c: expr::Operand::new(c, cl),
            t: expr::Operand::new(t, tl),
            f: expr::Operand::new(f, fl),
        },
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn masked(
    op: &'static str,
    x: &Storage,
    xl: &Layout,
    mask: &Storage,
    ml: &Layout,
    value: f64,
    dtype: DType,
    device: Device,
) -> Result<Storage> {
    pending(
        dtype,
        device,
        xl.clone(),
        LazyExpr::Masked {
            op,
            x: expr::Operand::new(x, xl),
            mask: expr::Operand::new(mask, ml),
            value,
        },
    )
}

pub(crate) fn cast(
    op: &'static str,
    x: &Storage,
    layout: &Layout,
    to: DType,
    device: Device,
) -> Result<Storage> {
    pending(
        to,
        device,
        layout.clone(),
        LazyExpr::Cast {
            op,
            x: expr::Operand::new(x, layout),
            to,
        },
    )
}

pub(crate) fn copy(
    op: &'static str,
    x: &Storage,
    layout: &Layout,
    dtype: DType,
    device: Device,
) -> Result<Storage> {
    pending(
        dtype,
        device,
        layout.clone(),
        LazyExpr::Copy {
            op,
            x: expr::Operand::new(x, layout),
            relabel: false,
        },
    )
}

pub(crate) fn constant(
    op: &'static str,
    value: f64,
    dtype: DType,
    device: Device,
    dims: Vec<usize>,
) -> Result<Storage> {
    let layout = Layout::contiguous(dims.clone())?;
    pending(
        dtype,
        device,
        layout,
        LazyExpr::Const {
            op,
            value,
            dtype,
            device,
            dims,
        },
    )
}

/// Wrap a deferred result so a composed scalar-left operation retains its
/// public operation name when a child fails during realization.
pub(crate) fn relabel(op: &'static str, value: &Storage, layout: &Layout) -> Result<Storage> {
    if !value.is_pending() {
        return Ok(value.clone());
    }
    pending(
        value.dtype(),
        value.device(),
        layout.clone(),
        LazyExpr::Copy {
            op,
            x: expr::Operand::new(value, layout),
            relabel: true,
        },
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::Layout;

    #[test]
    fn pending_storage_clones_share_cache() {
        let layout = Layout::contiguous([4]).unwrap();
        let storage = Storage::Cpu(crate::storage::CpuStorage::F32(Arc::new(vec![1.0; 4])));

        let pending = unary(
            "neg",
            UnaryOp::Neg,
            &storage,
            &layout,
            DType::F32,
            Device::Cpu,
        )
        .unwrap();
        let Storage::Pending(first) = pending.clone() else {
            panic!("expected pending storage")
        };
        let Storage::Pending(second) = pending else {
            panic!("expected pending storage")
        };
        assert!(Arc::ptr_eq(&first, &second));
        assert!(first.cache.get().is_none());
    }

    #[test]
    fn cpu_chain_defers_until_a_host_boundary() {
        let _guard = set_fusion(Fusion::On);
        let x = crate::Tensor::from_vec(vec![1.0f32, -2.0, 3.0, 4.0], [4], &Device::Cpu).unwrap();
        let y = x
            .add_scalar(2.0)
            .unwrap()
            .mul_scalar(3.0)
            .unwrap()
            .neg()
            .unwrap();
        assert!(y.storage().is_pending());
        assert_eq!(y.dims(), &[4]);
        assert_eq!(y.dtype(), DType::F32);
        assert_eq!(y.device(), Device::Cpu);
        assert_eq!(y.to_vec::<f32>().unwrap(), vec![-9.0, 0.0, -15.0, -18.0]);
        assert!(!y.storage().is_pending() || y.storage().ready().is_ok());
    }

    #[test]
    fn traced_backward_flushes_forward_and_backward_values() {
        let _guard = set_fusion(Fusion::On);
        let x = crate::Tensor::from_vec(vec![2.0f32, 3.0], [2], &Device::Cpu)
            .unwrap()
            .traced()
            .unwrap();
        let y = x.mul_scalar(4.0).unwrap().add_scalar(1.0).unwrap();
        let grads = y.backward().unwrap();
        let dx = grads.wrt_input(&x).unwrap();
        assert_eq!(dx.to_vec::<f32>().unwrap(), vec![4.0, 4.0]);
    }

    #[test]
    fn pending_root_can_be_consumed_after_switch_turns_off() {
        let x = crate::Tensor::from_vec(vec![2.0f32, 3.0], [2], &Device::Cpu)
            .unwrap()
            .traced()
            .unwrap();
        let y = {
            let _guard = set_fusion(Fusion::On);
            x.mul_scalar(4.0).unwrap().add_scalar(1.0).unwrap()
        };
        let grads = y.backward().unwrap();
        let dx = grads.wrt_input(&x).unwrap();
        assert_eq!(dx.to_vec::<f32>().unwrap(), vec![4.0, 4.0]);
    }

    #[test]
    fn deferred_backend_error_keeps_public_operation_name() {
        let _guard = set_fusion(Fusion::On);
        let x = crate::Tensor::from_vec(vec![true], [1], &Device::Cpu).unwrap();
        let y = x.add_scalar(1.0).unwrap();
        let error = y.to_vec::<bool>().unwrap_err();
        assert!(matches!(
            error,
            crate::Error::Unsupported {
                op: "add_scalar",
                ..
            }
        ));
    }

    #[test]
    fn nested_deferred_error_names_the_failing_child() {
        let _guard = set_fusion(Fusion::On);
        let x = crate::Tensor::from_vec(vec![true], [1], &Device::Cpu).unwrap();
        let y = x.add_scalar(1.0).unwrap().mul_scalar(2.0).unwrap();
        let error = y.to_vec::<bool>().unwrap_err();
        assert!(matches!(
            error,
            crate::Error::Unsupported {
                op: "add_scalar",
                ..
            }
        ));
    }

    #[test]
    fn fused_unary_matches_eager_width_and_signed_zero() {
        let x = crate::Tensor::from_vec(vec![-0.0f32, -1.0, 0.25, f32::NAN], [4], &Device::Cpu)
            .unwrap();
        let eager = {
            let _guard = set_fusion(Fusion::Off);
            x.relu().unwrap()
        };
        let fused = {
            let _guard = set_fusion(Fusion::On);
            x.relu().unwrap()
        };
        let eager_bits: Vec<u32> = eager
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect();
        let fused_bits: Vec<u32> = fused
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect();
        assert_eq!(eager_bits, fused_bits);
    }
    #[test]
    fn traced_extra_consumer_spills_intermediate() {
        let _guard = set_fusion(Fusion::On);
        let x = crate::Tensor::from_vec(vec![-1.0f32, 2.0], [2], &Device::Cpu)
            .unwrap()
            .traced()
            .unwrap();
        let add = x.add_scalar(1.0).unwrap();
        let relu = add.relu().unwrap();
        let Storage::Pending(root) = relu.storage() else {
            panic!("expected pending relu")
        };
        let Expr::Unary { x: operand, .. } = &*root.expr else {
            panic!("expected unary root")
        };
        let Storage::Pending(intermediate) = &operand.storage else {
            panic!("expected pending intermediate")
        };
        assert!(Arc::strong_count(intermediate) > 1);
        relu.realize().unwrap();
        assert!(intermediate.cache.get().is_some());
        assert_eq!(relu.to_vec::<f32>().unwrap(), vec![0.0, 3.0]);
    }

    #[test]
    fn pending_node_cap_materializes_the_overflow_tail() {
        let _guard = set_fusion(Fusion::On);
        let x = crate::Tensor::from_vec(vec![1.0f32], [1], &Device::Cpu).unwrap();
        let mut value = x;
        for _ in 0..MAX_PENDING_NODES {
            value = value.add_scalar(1.0).unwrap();
        }
        assert!(value.storage().is_pending());
        value = value.add_scalar(1.0).unwrap();
        assert!(!value.storage().is_pending());
        assert_eq!(
            value.to_vec::<f32>().unwrap(),
            vec![(MAX_PENDING_NODES + 2) as f32]
        );
    }
    #[test]
    fn dense_untraced_chain_fuses_without_child_materialization() {
        let _guard = set_fusion(Fusion::On);
        let x = crate::Tensor::from_vec(vec![1.0f32, 2.0], [2], &Device::Cpu).unwrap();
        let y = x
            .add_scalar(1.0)
            .unwrap()
            .mul_scalar(2.0)
            .unwrap()
            .neg()
            .unwrap();
        let Storage::Pending(root) = y.storage() else {
            panic!("expected pending root")
        };
        let Expr::Unary { x: unary_input, .. } = &*root.expr else {
            panic!("expected unary root")
        };
        let Storage::Pending(child) = &unary_input.storage else {
            panic!("expected pending child")
        };
        y.to_vec::<f32>().unwrap();
        assert!(
            child.cache.get().is_none(),
            "fused root should not materialize its private child"
        );
    }

    #[test]
    fn scalar_left_deferred_error_uses_public_name() {
        let _guard = set_fusion(Fusion::On);
        let x = crate::Tensor::from_vec(vec![true], [1], &Device::Cpu).unwrap();
        let y = 2.0 - &x;
        let error = y.to_vec::<bool>().unwrap_err();
        assert!(matches!(
            error,
            crate::Error::Unsupported {
                op: "sub_scalar",
                ..
            }
        ));
    }
    #[test]
    fn scalar_left_relabel_shares_child_storage() {
        let _guard = set_fusion(Fusion::On);
        let x = crate::Tensor::from_vec(vec![1.0f32, 3.0], [2], &Device::Cpu).unwrap();
        let y = 2.0 - &x;

        let Storage::Pending(root) = y.storage() else {
            panic!("expected a deferred relabel")
        };
        let Expr::Copy {
            x: operand,
            relabel: true,
            ..
        } = &*root.expr
        else {
            panic!("expected a relabel expression")
        };
        let Storage::Pending(child) = &operand.storage else {
            panic!("expected a deferred child")
        };

        y.to_vec::<f32>().unwrap();
        let Storage::Cpu(crate::storage::CpuStorage::F32(result)) =
            root.cache.get().expect("relabel result")
        else {
            panic!("expected CPU relabel result")
        };
        let Storage::Cpu(crate::storage::CpuStorage::F32(child_result)) =
            child.cache.get().expect("child result")
        else {
            panic!("expected CPU child result")
        };
        assert!(Arc::ptr_eq(result, child_result));
        assert_eq!(y.to_vec::<f32>().unwrap(), vec![1.0, -1.0]);
    }
}
