//! Element-wise CPU kernels: the strided + broadcast-aware iteration
//! engine (the largest chunk of new tensor-core work, exploration §3.2)
//! and the binary/unary/compare/where/masked-fill families built on it.
//!
//! Signatures frozen by T01; **T10b** fills the bodies. The design:
//!
//! - **One iteration engine.** Every kernel walks its inputs through
//!   [`Cursor`], which maps a row-major logical index to a storage index for
//!   an arbitrary [`Layout`](crate::layout::Layout) — contiguous, permuted,
//!   narrowed, or broadcast (stride-0 axes repeat elements). The op layer
//!   pre-broadcasts multi-input kernels to shape-identical views, so the
//!   output is dense/contiguous and one logical index addresses every input.
//! - **A contiguous fast path.** When *every* input view is
//!   [`is_contiguous`](crate::layout::Layout::is_contiguous), the cursor is
//!   the identity map, so the kernel drops it entirely and zips flat slices
//!   instead (adapted from the v2 `backend/cpu.rs` contiguous loops). The
//!   layout decision is made **once per call**, outside the loop: `dense`
//!   turns a view into a `&[E]`, and the `map1_dense`/`map2_dense`/
//!   `map3_dense` drivers walk it with equal-length iterators, so the
//!   per-element work is the arithmetic alone — no layout match and no bounds
//!   check, which lets the loop vectorize. Mixed contiguous/strided inputs (in
//!   particular a broadcast operand) keep the general [`Cursor`] path, which
//!   is allowed to stay slower.
//! - **Two families of dispatch macro, both hoisted out of the loop.**
//!   `dispatch_typed!` routes a runtime [`DType`](crate::dtype::DType) to a
//!   monomorphized body over the concrete element type, so each multi-dtype op
//!   is written once. `dispatch_binary_op!`/`dispatch_unary_op!`/
//!   `dispatch_cmp_op!` do the same for the *op discriminant*: they rebind it
//!   as a `const`, so the kernel closure captures nothing and the arithmetic
//!   `match` inside `binary_f32` and friends folds at compile time instead of
//!   re-running for every element.
//! - **The parallelism switch.** Output buffers are filled through `fill`,
//!   which becomes a rayon parallel loop under the `rayon` feature
//!   (`backend::parallel`) and a sequential loop otherwise.

use crate::backend::{BinaryOp, CmpOp, UnaryOp, View};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::storage::{CpuStorage, Storage};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Iteration engine
// ---------------------------------------------------------------------------

/// Maps a row-major logical element index (`0..num_elements`) to a storage
/// index for one [`Layout`](crate::layout::Layout).
///
/// Two shapes, so the common contiguous case pays nothing:
/// - `Contiguous` — the layout is
///   [`is_contiguous`](crate::layout::Layout::is_contiguous) (offset 0,
///   row-major), so the storage index equals the logical index.
/// - `Strided` — a per-axis `(divisor, dim, stride)` table plus the base
///   offset. The storage index of logical `i` is
///   `offset + Σ ((i / divisor_a) % dim_a) * stride_a`, which handles
///   permuted/narrowed offsets and stride-0 broadcast axes uniformly.
enum Cursor {
    Contiguous,
    Strided {
        offset: usize,
        /// One `(divisor, dim, stride)` per axis. `divisor` is the product of
        /// the dims to the axis's right (its row-major place value).
        axes: Vec<(usize, usize, usize)>,
    },
}

impl Cursor {
    /// Build the cursor for `layout`.
    fn new(layout: &Layout) -> Cursor {
        if layout.is_contiguous() {
            return Cursor::Contiguous;
        }
        let dims = layout.dims();
        let strides = layout.strides();
        let mut axes = Vec::with_capacity(dims.len());
        let mut divisor = 1usize;
        // Walk right-to-left so `divisor` accumulates the place value.
        for (&dim, &stride) in dims.iter().zip(strides).rev() {
            axes.push((divisor, dim, stride));
            divisor *= dim;
        }
        Cursor::Strided {
            offset: layout.offset(),
            axes,
        }
    }

    /// The storage index of the logical element at row-major position `i`.
    #[inline]
    fn index(&self, i: usize) -> usize {
        match self {
            Cursor::Contiguous => i,
            Cursor::Strided { offset, axes } => {
                let mut idx = *offset;
                for &(divisor, dim, stride) in axes {
                    // Broadcast axes carry stride 0, so their contribution is
                    // multiplied away — no special case needed.
                    idx += ((i / divisor) % dim) * stride;
                }
                idx
            }
        }
    }
}

/// Fill `out` by writing `f(i)` at every position, in parallel under the
/// `rayon` feature and sequentially otherwise.
///
/// This is the **general** driver: it pays one [`Cursor`] map per input per
/// element. Kernels whose inputs are all contiguous use the `map*_dense`
/// drivers below instead.
#[inline]
fn fill<T, F>(out: &mut [T], f: F)
where
    T: Send,
    F: Fn(usize) -> T + Send + Sync,
{
    #[cfg(feature = "rayon")]
    {
        crate::backend::parallel::for_each_mut(out, |idx, slot| *slot = f(idx));
    }
    #[cfg(not(feature = "rayon"))]
    {
        for (idx, slot) in out.iter_mut().enumerate() {
            *slot = f(idx);
        }
    }
}

// ---------------------------------------------------------------------------
// Contiguous fast path
// ---------------------------------------------------------------------------

/// Elements per rayon task in the contiguous drivers. Large enough that the
/// per-task overhead is amortized and the inner loop stays vectorized, small
/// enough to keep every core fed on the shapes these kernels see.
#[cfg(feature = "rayon")]
const DENSE_CHUNK: usize = 16 * 1024;

/// The dense `n`-element prefix of `data`, or `None` when `layout` is not the
/// identity map into it.
///
/// `Layout::is_contiguous` means offset 0 and canonical row-major strides, so
/// logical element `i` lives at `data[i]` and the first `n` elements of the
/// storage *are* the view. Non-contiguous layouts (permuted, narrowed,
/// broadcast) return `None` and their kernel takes the `Cursor` path.
///
/// Returning a slice of length exactly `n` is what makes the drivers below
/// fast: every iterator in the zip then has the same length, so the bounds
/// checks fold away. The `get` is also belt-and-braces against a short
/// storage — `View` construction already validates the layout against it, and
/// this keeps the helper panic-free regardless.
#[inline]
fn dense<'a, E>(layout: &Layout, data: &'a [E], n: usize) -> Option<&'a [E]> {
    if layout.is_contiguous() {
        data.get(..n)
    } else {
        None
    }
}

/// Hand `out` to `body` as `(base_index, window)` pairs: the whole slice in one
/// call without `rayon`, and [`DENSE_CHUNK`]-sized parallel windows with it.
///
/// The windows are a pure partition of `out` — every slot is written by exactly
/// one call, from the same input positions — so the result does not depend on
/// the feature flag or on the thread count.
#[cfg(feature = "rayon")]
#[inline]
fn fill_dense_chunks<T, F>(out: &mut [T], body: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Send + Sync,
{
    if out.is_empty() {
        return;
    }
    let chunk = DENSE_CHUNK.min(out.len());
    crate::backend::parallel::for_each_chunk_mut(out, chunk, |idx, window| {
        body(idx * chunk, window)
    });
}

/// Build the dense output `[f(a[0]), f(a[1]), …]`.
///
/// Without `rayon` the buffer is `collect`ed straight from the input iterator:
/// its length is exact (`TrustedLen`), so the allocation happens once and every
/// output byte is written exactly once — where pre-sizing with `vec![ZERO; n]`
/// and then overwriting costs a second pass over the whole output, which on a
/// bandwidth-bound kernel is real traffic. With `rayon` the buffer is pre-sized
/// and filled in windows, because the parallel façade partitions an existing
/// slice. Both branches write the same value to every slot.
#[inline]
fn map1_dense<A, O, F>(a: &[A], f: F) -> Vec<O>
where
    A: Copy + Sync,
    O: TypedSlice,
    F: Fn(A) -> O + Send + Sync,
{
    #[cfg(not(feature = "rayon"))]
    {
        a.iter().map(|&x| f(x)).collect()
    }
    #[cfg(feature = "rayon")]
    {
        let mut out = vec![O::ZERO; a.len()];
        fill_dense_chunks(&mut out, |base, window| {
            // Re-slicing the input to the window length is what lets the loop
            // drop its bounds checks: both iterators then have equal length.
            let a = &a[base..base + window.len()];
            for (slot, &x) in window.iter_mut().zip(a) {
                *slot = f(x);
            }
        });
        out
    }
}

/// Build the dense output `[f(a[0], b[0]), …]`. `a` and `b` must have equal
/// length. See [`map1_dense`] for why the sequential branch collects.
#[inline]
fn map2_dense<A, B, O, F>(a: &[A], b: &[B], f: F) -> Vec<O>
where
    A: Copy + Sync,
    B: Copy + Sync,
    O: TypedSlice,
    F: Fn(A, B) -> O + Send + Sync,
{
    debug_assert_eq!(a.len(), b.len());
    #[cfg(not(feature = "rayon"))]
    {
        a.iter().zip(b).map(|(&x, &y)| f(x, y)).collect()
    }
    #[cfg(feature = "rayon")]
    {
        let mut out = vec![O::ZERO; a.len()];
        fill_dense_chunks(&mut out, |base, window| {
            let end = base + window.len();
            let a = &a[base..end];
            let b = &b[base..end];
            for ((slot, &x), &y) in window.iter_mut().zip(a).zip(b) {
                *slot = f(x, y);
            }
        });
        out
    }
}

/// Build the dense output `[f(a[0], b[0], c[0]), …]`. All three inputs must
/// have equal length. See [`map1_dense`] for why the sequential branch
/// collects.
#[inline]
fn map3_dense<A, B, C, O, F>(a: &[A], b: &[B], c: &[C], f: F) -> Vec<O>
where
    A: Copy + Sync,
    B: Copy + Sync,
    C: Copy + Sync,
    O: TypedSlice,
    F: Fn(A, B, C) -> O + Send + Sync,
{
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());
    #[cfg(not(feature = "rayon"))]
    {
        a.iter()
            .zip(b)
            .zip(c)
            .map(|((&x, &y), &z)| f(x, y, z))
            .collect()
    }
    #[cfg(feature = "rayon")]
    {
        let mut out = vec![O::ZERO; a.len()];
        fill_dense_chunks(&mut out, |base, window| {
            let end = base + window.len();
            let a = &a[base..end];
            let b = &b[base..end];
            let c = &c[base..end];
            for (((slot, &x), &y), &z) in window.iter_mut().zip(a).zip(b).zip(c) {
                *slot = f(x, y, z);
            }
        });
        out
    }
}

// ---------------------------------------------------------------------------
// Typed storage access
// ---------------------------------------------------------------------------

/// Extract a `&[Self]` from the matching [`CpuStorage`] variant and wrap an
/// owned output buffer back into a [`Storage`]. Implemented for the six
/// element types; `Zero` gives a cheap placeholder for pre-sizing buffers.
trait TypedSlice: Sized + Copy + Send + Sync {
    /// The dtype tag for this element type.
    const DTYPE: DType;
    /// The additive identity, used only to pre-size output buffers (every
    /// slot is overwritten by `fill`).
    const ZERO: Self;
    /// Borrow the element slice, erroring if the storage variant differs.
    fn slice<'a>(cpu: &'a CpuStorage, op: &'static str) -> Result<&'a [Self]>;
    /// Wrap an owned output buffer as a CPU [`Storage`].
    fn into_storage(v: Vec<Self>) -> Storage;
}

macro_rules! impl_typed_slice {
    ($ty:ty, $variant:ident, $dtype:ident, $zero:expr) => {
        impl TypedSlice for $ty {
            const DTYPE: DType = DType::$dtype;
            const ZERO: Self = $zero;
            fn slice<'a>(cpu: &'a CpuStorage, op: &'static str) -> Result<&'a [Self]> {
                match cpu {
                    CpuStorage::$variant(v) => Ok(v.as_slice()),
                    other => Err(Error::Backend {
                        op,
                        msg: format!(
                            "cpu elementwise kernel expected {} storage, got {}",
                            DType::$dtype,
                            other.dtype()
                        ),
                    }),
                }
            }
            fn into_storage(v: Vec<Self>) -> Storage {
                Storage::Cpu(CpuStorage::$variant(Arc::new(v)))
            }
        }
    };
}

impl_typed_slice!(half::f16, F16, F16, half::f16::ZERO);
impl_typed_slice!(half::bf16, BF16, BF16, half::bf16::ZERO);
impl_typed_slice!(f32, F32, F32, 0.0);
impl_typed_slice!(f64, F64, F64, 0.0);
impl_typed_slice!(i64, I64, I64, 0);
impl_typed_slice!(bool, Bool, Bool, false);

/// Borrow the typed element slice for `E` from a CPU `Storage`. The caller
/// has dispatched on the view's runtime dtype, so a mismatch or a non-CPU
/// storage is an internal invariant break reported as [`Error::Backend`].
fn cpu_slice<'a, E>(storage: &'a Storage, op: &'static str) -> Result<&'a [E]>
where
    E: TypedSlice,
{
    match storage {
        Storage::Cpu(cpu) => E::slice(cpu, op),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        _ => Err(Error::Backend {
            op,
            msg: "cpu elementwise kernel received non-cpu storage".into(),
        }),
    }
}

/// Route a runtime [`DType`] to a monomorphized block bound to the concrete
/// element type named `$elem`. Dtypes not listed fall through to an
/// [`Error::Unsupported`] for `$op` on `$device`.
macro_rules! dispatch_typed {
    ($dtype:expr, $op:expr, $device:expr, $elem:ident => $body:block, [$($ty:ty),+ $(,)?]) => {{
        match $dtype {
            $(
                <$ty as TypedSlice>::DTYPE => {
                    type $elem = $ty;
                    $body
                }
            )+
            #[allow(unreachable_patterns)]
            other => Err(Error::Unsupported {
                op: $op,
                device: $device,
                dtype: other,
            }),
        }
    }};
}

/// Re-dispatch a runtime op enum into one `const` per variant, binding it as
/// `$konst` inside a copy of `$body` for each arm.
///
/// This is the loop-hoisting half of the fast path. A kernel closure written
/// against the `const` captures *nothing*, and the arithmetic `match` inside
/// `binary_f32` and friends folds to the single reachable arm at compile time —
/// where a closure over a runtime `op` re-ran that match for every element.
///
/// The cost is one monomorphization of the kernel per (op, dtype) pair, which
/// is the price of the specialization.
macro_rules! dispatch_op_const {
    ($ty:ty, $op:expr, $konst:ident => $body:block, [$($variant:ident),+ $(,)?]) => {
        match $op {
            $(
                <$ty>::$variant => {
                    const $konst: $ty = <$ty>::$variant;
                    $body
                }
            )+
        }
    };
}

/// `dispatch_op_const` over every `BinaryOp` variant.
macro_rules! dispatch_binary_op {
    ($op:expr, $konst:ident => $body:block) => {
        dispatch_op_const!(BinaryOp, $op, $konst => $body,
            [Add, Sub, Mul, Div, Maximum, Minimum])
    };
}

/// `dispatch_op_const` over every `UnaryOp` variant.
macro_rules! dispatch_unary_op {
    ($op:expr, $konst:ident => $body:block) => {
        dispatch_op_const!(UnaryOp, $op, $konst => $body,
            [Relu, Gelu, Exp, Ln, Sqrt, Tanh, Sigmoid, Neg, Abs])
    };
}

/// `dispatch_op_const` over every `CmpOp` variant.
macro_rules! dispatch_cmp_op {
    ($op:expr, $konst:ident => $body:block) => {
        dispatch_op_const!(CmpOp, $op, $konst => $body, [Eq, Ne, Lt, Le, Gt, Ge])
    };
}

// ---------------------------------------------------------------------------
// Binary
// ---------------------------------------------------------------------------

/// Apply `f` element-wise over two pre-broadcast, shape-identical views,
/// producing a fresh contiguous `Vec<E>` output.
///
/// Both views contiguous is the fast path: a flat slice zip with no cursor.
/// Otherwise the general `Cursor` path runs. Either way `f` sees the same
/// operand pair for the same output slot, so the two paths agree bit for bit.
fn zip_map<E, F>(op: &'static str, lhs: View<'_>, rhs: View<'_>, f: F) -> Result<Storage>
where
    E: TypedSlice,
    F: Fn(E, E) -> E + Send + Sync,
{
    let n = lhs.layout().num_elements();
    let lhs_data = cpu_slice::<E>(lhs.storage(), op)?;
    let rhs_data = cpu_slice::<E>(rhs.storage(), op)?;
    let out = match (
        dense(lhs.layout(), lhs_data, n),
        dense(rhs.layout(), rhs_data, n),
    ) {
        (Some(a), Some(b)) => map2_dense(a, b, f),
        _ => {
            let lc = Cursor::new(lhs.layout());
            let rc = Cursor::new(rhs.layout());
            let mut out = vec![E::ZERO; n];
            fill(&mut out, |i| {
                f(lhs_data[lc.index(i)], rhs_data[rc.index(i)])
            });
            out
        }
    };
    Ok(E::into_storage(out))
}

/// See [`BackendOps::binary`](crate::backend::BackendOps::binary).
pub(crate) fn binary(op: BinaryOp, lhs: View<'_>, rhs: View<'_>) -> Result<Storage> {
    let name = binary_op_name(op);
    let device = lhs.device();
    // Floats support every op; i64 supports arithmetic (wrapping / guarded);
    // bool has no arithmetic (comparisons live in `compare`).
    dispatch_binary_op!(op, OP => {
        match lhs.dtype() {
            DType::F32 => zip_map::<f32, _>(name, lhs, rhs, |a, b| binary_f32(OP, a, b)),
            DType::F64 => zip_map::<f64, _>(name, lhs, rhs, |a, b| binary_f64(OP, a, b)),
            DType::F16 => zip_map::<half::f16, _>(name, lhs, rhs, |a, b| {
                half::f16::from_f32(binary_f32(OP, a.to_f32(), b.to_f32()))
            }),
            DType::BF16 => zip_map::<half::bf16, _>(name, lhs, rhs, |a, b| {
                half::bf16::from_f32(binary_f32(OP, a.to_f32(), b.to_f32()))
            }),
            DType::I64 => zip_map::<i64, _>(name, lhs, rhs, |a, b| binary_i64(OP, a, b)),
            other => Err(Error::Unsupported {
                op: name,
                device,
                dtype: other,
            }),
        }
    })
}

/// See [`BackendOps::binary_scalar`](crate::backend::BackendOps::binary_scalar).
pub(crate) fn binary_scalar(op: BinaryOp, x: View<'_>, scalar: f64) -> Result<Storage> {
    let name = binary_op_name(op);
    let device = x.device();
    dispatch_binary_op!(op, OP => {
        match x.dtype() {
            DType::F32 => {
                let s = scalar as f32;
                unary_map::<f32, _>(name, x, move |a| binary_f32(OP, a, s))
            }
            DType::F64 => unary_map::<f64, _>(name, x, move |a| binary_f64(OP, a, scalar)),
            DType::F16 => {
                let s = scalar as f32;
                unary_map::<half::f16, _>(name, x, move |a| {
                    half::f16::from_f32(binary_f32(OP, a.to_f32(), s))
                })
            }
            DType::BF16 => {
                let s = scalar as f32;
                unary_map::<half::bf16, _>(name, x, move |a| {
                    half::bf16::from_f32(binary_f32(OP, a.to_f32(), s))
                })
            }
            DType::I64 => {
                let s = scalar as i64;
                unary_map::<i64, _>(name, x, move |a| binary_i64(OP, a, s))
            }
            other => Err(Error::Unsupported {
                op: name,
                device,
                dtype: other,
            }),
        }
    })
}

fn binary_op_name(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "add",
        BinaryOp::Sub => "sub",
        BinaryOp::Mul => "mul",
        BinaryOp::Div => "div",
        BinaryOp::Maximum => "maximum",
        BinaryOp::Minimum => "minimum",
    }
}

// The three scalar bodies below are the single source of truth for the binary
// arithmetic, so every dtype and both layout paths agree by construction.
// `inline(always)` is what makes that free rather than costly: callers pass
// `op` as a `const` (see `dispatch_binary_op!`), so after inlining the match
// folds to the one reachable arm and the enclosing loop vectorizes.

#[inline(always)]
fn binary_f32(op: BinaryOp, a: f32, b: f32) -> f32 {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Maximum => a.max(b),
        BinaryOp::Minimum => a.min(b),
    }
}

#[inline(always)]
fn binary_f64(op: BinaryOp, a: f64, b: f64) -> f64 {
    match op {
        BinaryOp::Add => a + b,
        BinaryOp::Sub => a - b,
        BinaryOp::Mul => a * b,
        BinaryOp::Div => a / b,
        BinaryOp::Maximum => a.max(b),
        BinaryOp::Minimum => a.min(b),
    }
}

#[inline(always)]
fn binary_i64(op: BinaryOp, a: i64, b: i64) -> i64 {
    // Wrapping arithmetic (PyTorch integer-overflow semantics) so kernels are
    // panic-free and identical in debug and release; division guards a zero
    // divisor by yielding 0 rather than panicking.
    match op {
        BinaryOp::Add => a.wrapping_add(b),
        BinaryOp::Sub => a.wrapping_sub(b),
        BinaryOp::Mul => a.wrapping_mul(b),
        BinaryOp::Div => a.checked_div(b).unwrap_or(0),
        BinaryOp::Maximum => a.max(b),
        BinaryOp::Minimum => a.min(b),
    }
}

// ---------------------------------------------------------------------------
// Unary
// ---------------------------------------------------------------------------

/// Apply `f` element-wise over one view, producing a fresh contiguous output.
///
/// A contiguous input takes the flat slice walk; anything else takes the
/// general `Cursor` path. `f` sees the same element for the same output slot
/// either way, so the two paths agree bit for bit.
fn unary_map<E, F>(op: &'static str, x: View<'_>, f: F) -> Result<Storage>
where
    E: TypedSlice,
    F: Fn(E) -> E + Send + Sync,
{
    let n = x.layout().num_elements();
    let data = cpu_slice::<E>(x.storage(), op)?;
    let out = match dense(x.layout(), data, n) {
        Some(src) => map1_dense(src, f),
        None => {
            let cursor = Cursor::new(x.layout());
            let mut out = vec![E::ZERO; n];
            fill(&mut out, |i| f(data[cursor.index(i)]));
            out
        }
    };
    Ok(E::into_storage(out))
}

/// See [`BackendOps::unary`](crate::backend::BackendOps::unary). Note the
/// exact-GELU contract on [`UnaryOp::Gelu`](crate::backend::UnaryOp).
pub(crate) fn unary(op: UnaryOp, x: View<'_>) -> Result<Storage> {
    let name = unary_op_name(op);
    let device = x.device();
    dispatch_unary_op!(op, OP => {
        match x.dtype() {
            // The `f64` round trip is the contract, not an accident: every
            // float dtype gets the same `unary_f64` definition, and the same
            // rounding, so f16/bf16/f32/f64 agree. Unchanged here.
            DType::F32 => unary_map::<f32, _>(name, x, |a| unary_f64(OP, a as f64) as f32),
            DType::F64 => unary_map::<f64, _>(name, x, |a| unary_f64(OP, a)),
            DType::F16 => unary_map::<half::f16, _>(name, x, |a| {
                half::f16::from_f64(unary_f64(OP, a.to_f64()))
            }),
            DType::BF16 => unary_map::<half::bf16, _>(name, x, |a| {
                half::bf16::from_f64(unary_f64(OP, a.to_f64()))
            }),
            DType::I64 => match OP {
                // Only the sign-preserving integer unaries are defined; the
                // rest are float-only per the `UnaryOp` contract.
                UnaryOp::Neg => unary_map::<i64, _>(name, x, |a| a.wrapping_neg()),
                UnaryOp::Abs => unary_map::<i64, _>(name, x, |a| a.wrapping_abs()),
                _ => Err(Error::Unsupported {
                    op: name,
                    device,
                    dtype: DType::I64,
                }),
            },
            other => Err(Error::Unsupported {
                op: name,
                device,
                dtype: other,
            }),
        }
    })
}

fn unary_op_name(op: UnaryOp) -> &'static str {
    match op {
        UnaryOp::Relu => "relu",
        UnaryOp::Gelu => "gelu",
        UnaryOp::Exp => "exp",
        UnaryOp::Ln => "ln",
        UnaryOp::Sqrt => "sqrt",
        UnaryOp::Tanh => "tanh",
        UnaryOp::Sigmoid => "sigmoid",
        UnaryOp::Neg => "neg",
        UnaryOp::Abs => "abs",
    }
}

/// The unary math in `f64`; callers cast to/from the element type. One
/// implementation guarantees f16/bf16/f32/f64 agree on the definition (in
/// particular the **exact** GELU with `erf`, not the tanh approximation).
///
/// `inline(always)` for the same reason as `binary_f32`: `op` arrives as a
/// `const`, so only the selected arm survives in the caller's loop.
#[inline(always)]
fn unary_f64(op: UnaryOp, x: f64) -> f64 {
    match op {
        UnaryOp::Relu => x.max(0.0),
        UnaryOp::Gelu => 0.5 * x * (1.0 + erf(x * std::f64::consts::FRAC_1_SQRT_2)),
        UnaryOp::Exp => x.exp(),
        UnaryOp::Ln => x.ln(),
        UnaryOp::Sqrt => x.sqrt(),
        UnaryOp::Tanh => x.tanh(),
        UnaryOp::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        UnaryOp::Neg => -x,
        UnaryOp::Abs => x.abs(),
    }
}

// ---------------------------------------------------------------------------
// Compare -> Bool
// ---------------------------------------------------------------------------

/// See [`BackendOps::compare`](crate::backend::BackendOps::compare).
pub(crate) fn compare(op: CmpOp, lhs: View<'_>, rhs: View<'_>) -> Result<Storage> {
    let name = cmp_op_name(op);
    match lhs.dtype() {
        DType::F32 => compare_typed::<f32>(op, name, lhs, rhs),
        DType::F64 => compare_typed::<f64>(op, name, lhs, rhs),
        DType::F16 => compare_typed::<half::f16>(op, name, lhs, rhs),
        DType::BF16 => compare_typed::<half::bf16>(op, name, lhs, rhs),
        DType::I64 => compare_typed::<i64>(op, name, lhs, rhs),
        DType::Bool => compare_typed::<bool>(op, name, lhs, rhs),
    }
}

fn compare_typed<E>(op: CmpOp, name: &'static str, lhs: View<'_>, rhs: View<'_>) -> Result<Storage>
where
    E: TypedSlice + PartialOrd,
{
    let n = lhs.layout().num_elements();
    let lhs_data = cpu_slice::<E>(lhs.storage(), name)?;
    let rhs_data = cpu_slice::<E>(rhs.storage(), name)?;
    let out = dispatch_cmp_op!(op, OP => {
        let f = |a: E, b: E| compare_scalar(OP, &a, &b);
        match (
            dense(lhs.layout(), lhs_data, n),
            dense(rhs.layout(), rhs_data, n),
        ) {
            (Some(a), Some(b)) => map2_dense(a, b, f),
            _ => {
                let lc = Cursor::new(lhs.layout());
                let rc = Cursor::new(rhs.layout());
                let mut out = vec![false; n];
                fill(&mut out, |i| f(lhs_data[lc.index(i)], rhs_data[rc.index(i)]));
                out
            }
        }
    });
    Ok(<bool as TypedSlice>::into_storage(out))
}

/// The comparison itself: one definition shared by every dtype and both layout
/// paths.
///
/// `inline(always)` so that a `const` `op` (see `dispatch_cmp_op!`) folds the
/// match away in the caller's loop.
#[inline(always)]
fn compare_scalar<E: PartialOrd>(op: CmpOp, a: &E, b: &E) -> bool {
    match op {
        CmpOp::Eq => a == b,
        CmpOp::Ne => a != b,
        CmpOp::Lt => a < b,
        CmpOp::Le => a <= b,
        CmpOp::Gt => a > b,
        CmpOp::Ge => a >= b,
    }
}

fn cmp_op_name(op: CmpOp) -> &'static str {
    match op {
        CmpOp::Eq => "eq",
        CmpOp::Ne => "ne",
        CmpOp::Lt => "lt",
        CmpOp::Le => "le",
        CmpOp::Gt => "gt",
        CmpOp::Ge => "ge",
    }
}

// ---------------------------------------------------------------------------
// where / masked_fill
// ---------------------------------------------------------------------------

/// See [`BackendOps::where_cond`](crate::backend::BackendOps::where_cond).
pub(crate) fn where_cond(cond: View<'_>, on_true: View<'_>, on_false: View<'_>) -> Result<Storage> {
    const OP: &str = "where";
    dispatch_typed!(
        on_true.dtype(),
        OP,
        on_true.device(),
        Elem => { where_typed::<Elem>(cond, on_true, on_false) },
        [half::f16, half::bf16, f32, f64, i64, bool]
    )
}

fn where_typed<E>(cond: View<'_>, on_true: View<'_>, on_false: View<'_>) -> Result<Storage>
where
    E: TypedSlice,
{
    const OP: &str = "where";
    let n = cond.layout().num_elements();
    let cond_data = cpu_slice::<bool>(cond.storage(), OP)?;
    let t_data = cpu_slice::<E>(on_true.storage(), OP)?;
    let f_data = cpu_slice::<E>(on_false.storage(), OP)?;
    let out = match (
        dense(cond.layout(), cond_data, n),
        dense(on_true.layout(), t_data, n),
        dense(on_false.layout(), f_data, n),
    ) {
        // The dense path loads both operands and selects; the general path
        // below only walks the selected operand's cursor. Same value either
        // way — an unselected load is always in bounds, never observable.
        (Some(c), Some(t), Some(f)) => map3_dense(c, t, f, |c, t, f| if c { t } else { f }),
        _ => {
            let cc = Cursor::new(cond.layout());
            let tc = Cursor::new(on_true.layout());
            let fc = Cursor::new(on_false.layout());
            let mut out = vec![E::ZERO; n];
            fill(&mut out, |i| {
                if cond_data[cc.index(i)] {
                    t_data[tc.index(i)]
                } else {
                    f_data[fc.index(i)]
                }
            });
            out
        }
    };
    Ok(E::into_storage(out))
}

/// See [`BackendOps::masked_fill`](crate::backend::BackendOps::masked_fill).
pub(crate) fn masked_fill(x: View<'_>, mask: View<'_>, value: f64) -> Result<Storage> {
    match x.dtype() {
        DType::F32 => masked_fill_typed::<f32>(x, mask, value as f32),
        DType::F64 => masked_fill_typed::<f64>(x, mask, value),
        DType::F16 => masked_fill_typed::<half::f16>(x, mask, half::f16::from_f64(value)),
        DType::BF16 => masked_fill_typed::<half::bf16>(x, mask, half::bf16::from_f64(value)),
        DType::I64 => masked_fill_typed::<i64>(x, mask, value as i64),
        DType::Bool => masked_fill_typed::<bool>(x, mask, value != 0.0),
    }
}

fn masked_fill_typed<E>(x: View<'_>, mask: View<'_>, value: E) -> Result<Storage>
where
    E: TypedSlice,
{
    const OP: &str = "masked_fill";
    let n = x.layout().num_elements();
    let x_data = cpu_slice::<E>(x.storage(), OP)?;
    let mask_data = cpu_slice::<bool>(mask.storage(), OP)?;
    let out = match (
        dense(x.layout(), x_data, n),
        dense(mask.layout(), mask_data, n),
    ) {
        (Some(src), Some(m)) => map2_dense(src, m, |a, m| if m { value } else { a }),
        _ => {
            let xc = Cursor::new(x.layout());
            let mc = Cursor::new(mask.layout());
            let mut out = vec![E::ZERO; n];
            fill(&mut out, |i| {
                if mask_data[mc.index(i)] {
                    value
                } else {
                    x_data[xc.index(i)]
                }
            });
            out
        }
    };
    Ok(E::into_storage(out))
}

// ---------------------------------------------------------------------------
// erf (for exact GELU)
// ---------------------------------------------------------------------------

/// Error function `erf(x)`, evaluated in `f64` for the exact-GELU kernel.
///
/// `erf` is odd, so the sign is handled up front and the body works on
/// `|x|`. Two convergent, self-contained methods (no external dependency,
/// no magic-constant tables to transcribe) cover the whole range to full f64
/// accuracy:
/// - `|x| < 2`: the Maclaurin series
///   `erf(x) = 2/√π · Σ (−1)ⁿ x^(2n+1) / (n!(2n+1))`, summed until the term
///   is below the running total's ulp.
/// - `2 ≤ |x| < 6`: `erf(x) = 1 − erfc(x)`, with `erfc` from the Lentz
///   evaluation of its standard continued fraction.
/// - `|x| ≥ 6`: `erf` has saturated to `±1` to well within f64.
fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let sign = x.is_sign_negative();
    let ax = x.abs();
    let magnitude = if ax < 2.0 {
        erf_series(ax)
    } else if ax >= 6.0 {
        1.0
    } else {
        1.0 - erfc_cont_frac(ax)
    };
    if sign { -magnitude } else { magnitude }
}

/// `erf(x)` for `x ≥ 0` via its Maclaurin series (converges quickly for the
/// small arguments GELU sees).
fn erf_series(x: f64) -> f64 {
    // `term` is (-1)^n x^(2n+1) / n!; the summand is `term / (2n+1)`.
    let mut total = 0.0f64;
    let mut term = x;
    for n in 0..200usize {
        let add = term / (2.0 * n as f64 + 1.0);
        total += add;
        if n > 2 && add.abs() < total.abs() * f64::EPSILON {
            break;
        }
        term *= -x * x / (n as f64 + 1.0);
    }
    std::f64::consts::FRAC_2_SQRT_PI * total
}

/// `erfc(x)` for `x > 0` via the Lentz evaluation of the continued fraction
/// `erfc(x) = exp(−x²)/√π · 1/(x + ½/(x + 1/(x + 3⁄2/(x + …))))`.
fn erfc_cont_frac(x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let mut b = x;
    let mut c = 1e300;
    let mut d = if b == 0.0 { 1e300 } else { 1.0 / b };
    let mut h = d;
    for i in 1..300usize {
        let a = i as f64 / 2.0;
        b = x;
        d = b + a * d;
        if d == 0.0 {
            d = TINY;
        }
        c = b + a / c;
        if c == 0.0 {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < f64::EPSILON {
            break;
        }
    }
    // 1/√π = ½ · (2/√π); the stable constant is FRAC_2_SQRT_PI.
    (-x * x).exp() * (0.5 * std::f64::consts::FRAC_2_SQRT_PI) * h
}

#[cfg(test)]
mod tests;
