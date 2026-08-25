//! Deferred element-wise expressions and their owned operands.
//!
//! Expressions own storage/layout pairs rather than [`Tensor`](crate::Tensor)
//! handles. This keeps an autograd graph out of a pending value's lifetime.

use crate::backend::{BinaryOp, CmpOp, UnaryOp};
use crate::device::Device;
use crate::dtype::DType;
use crate::layout::Layout;
use crate::storage::Storage;

/// A resolved operand: a storage handle and the layout the operation consumes.
///
/// The storage may itself be pending. The realizer resolves it before handing
/// a view to a backend kernel.
pub(crate) struct Operand {
    pub(crate) storage: Storage,
    pub(crate) layout: Layout,
}

impl Operand {
    pub(crate) fn new(storage: &Storage, layout: &Layout) -> Operand {
        Operand {
            storage: storage.clone(),
            layout: layout.clone(),
        }
    }
}

/// One deferred operation in a straight-line element-wise expression.
///
/// `op` is deliberately stored alongside the enum discriminant. Backends name
/// errors after their primitive family, while deferred errors must name the
/// public operation that created the expression (for example `sub_scalar`).
pub(crate) enum Expr {
    Unary {
        op: &'static str,
        kind: UnaryOp,
        x: Operand,
    },
    Binary {
        op: &'static str,
        kind: BinaryOp,
        a: Operand,
        b: Operand,
    },
    Scalar {
        op: &'static str,
        kind: BinaryOp,
        x: Operand,
        k: f64,
    },
    Compare {
        op: &'static str,
        kind: CmpOp,
        a: Operand,
        b: Operand,
    },
    Where {
        op: &'static str,
        c: Operand,
        t: Operand,
        f: Operand,
    },
    Masked {
        op: &'static str,
        x: Operand,
        mask: Operand,
        value: f64,
    },
    Cast {
        op: &'static str,
        x: Operand,
        to: DType,
    },
    Copy {
        op: &'static str,
        x: Operand,
        /// Whether this wrapper only relabels a composed operation. Relabel
        /// wrappers alias the realized child storage; ordinary copies use the
        /// backend's strided-copy kernel.
        relabel: bool,
    },
    Const {
        op: &'static str,
        value: f64,
        dtype: DType,
        device: Device,
        dims: Vec<usize>,
    },
}

/// A CPU-fusable stage view. Other expression variants terminate the current
/// straight-line executor and are evaluated by the ordinary backend path.
pub(crate) enum CpuStage<'a> {
    Unary {
        kind: UnaryOp,
        x: &'a Operand,
    },
    Scalar {
        kind: BinaryOp,
        x: &'a Operand,
        k: f64,
    },
    Binary {
        kind: BinaryOp,
        a: &'a Operand,
        b: &'a Operand,
    },
}

impl Expr {
    pub(crate) fn cpu_stage(&self) -> Option<CpuStage<'_>> {
        match self {
            Expr::Unary { kind, x, .. } => Some(CpuStage::Unary { kind: *kind, x }),
            Expr::Scalar { kind, x, k, .. } => Some(CpuStage::Scalar {
                kind: *kind,
                x,
                k: *k,
            }),
            Expr::Binary { kind, a, b, .. } => Some(CpuStage::Binary { kind: *kind, a, b }),
            _ => None,
        }
    }
    pub(crate) fn op(&self) -> &'static str {
        match self {
            Expr::Unary { op, .. }
            | Expr::Binary { op, .. }
            | Expr::Scalar { op, .. }
            | Expr::Compare { op, .. }
            | Expr::Where { op, .. }
            | Expr::Masked { op, .. }
            | Expr::Cast { op, .. }
            | Expr::Copy { op, .. }
            | Expr::Const { op, .. } => op,
        }
    }

    /// Visit every owned storage operand without allocating a temporary list.
    pub(crate) fn for_each_operand(&self, mut f: impl FnMut(&Storage)) {
        match self {
            Expr::Unary { x, .. } | Expr::Cast { x, .. } | Expr::Copy { x, .. } => f(&x.storage),
            Expr::Binary { a, b, .. } | Expr::Compare { a, b, .. } => {
                f(&a.storage);
                f(&b.storage);
            }
            Expr::Scalar { x, .. } => f(&x.storage),
            Expr::Where {
                c, t, f: otherwise, ..
            } => {
                f(&c.storage);
                f(&t.storage);
                f(&otherwise.storage);
            }
            Expr::Masked { x, mask, .. } => {
                f(&x.storage);
                f(&mask.storage);
            }
            Expr::Const { .. } => {}
        }
    }

    /// Consume the expression, returning its owned storage operands.
    ///
    /// Used by the iterative pending-storage drop path. Non-storage fields
    /// are dropped normally; pending child arcs are handled by that path's
    /// explicit worklist rather than by recursive Rust drop glue.
    pub(crate) fn into_storages(self, out: &mut Vec<Storage>) {
        match self {
            Expr::Unary { x, .. } | Expr::Cast { x, .. } | Expr::Copy { x, .. } => {
                out.push(x.storage);
            }
            Expr::Binary { a, b, .. } | Expr::Compare { a, b, .. } => {
                out.push(a.storage);
                out.push(b.storage);
            }
            Expr::Scalar { x, .. } => out.push(x.storage),
            Expr::Where { c, t, f, .. } => {
                out.push(c.storage);
                out.push(t.storage);
                out.push(f.storage);
            }
            Expr::Masked { x, mask, .. } => {
                out.push(x.storage);
                out.push(mask.storage);
            }
            Expr::Const { .. } => {}
        }
    }
}
