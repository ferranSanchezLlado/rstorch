//! Matmul CPU kernel.
//!
//! Accumulation follows the `Element::Acc` contract, batched over leading
//! dims. Semantics on [`BackendOps`](crate::backend::BackendOps).
//!
//! Both operands are rank ≥ 2. The trailing two axes are the matrix axes
//! (`[..., m, k]` × `[..., k, n]` → `[..., m, n]`); the leading axes are
//! batch dims that broadcast against each other under NumPy/PyTorch rules
//! (right-aligned; a size-1 batch axis repeats). Inner products accumulate in
//! the wide [`Acc`](crate::dtype::Element::Acc) type and cast back to the
//! element type exactly once per output element. The walk is stride-aware, so
//! transposed / narrowed / broadcast operand views are handled directly with
//! no pre-materialization.
//!
//! # Loop order
//!
//! A naive `i` → `j` → `p` nest makes the innermost step
//! `rhs[rhs_col + p * rhs_k_stride]`, striding `rhs` by `n` per multiply. That
//! the dominant cost at 1024², so the order is instead picked from the `rhs`
//! strides:
//!
//! - **`rhs_n_stride == 1`** (row-major `rhs`, the plain `a @ b` case) —
//!   `i` → `p` → `j`, accumulating output rows against consecutive `rhs`
//!   rows. F32 handles four rows together so each `rhs` value is loaded once;
//!   the generic path uses one bounds-check-free, vectorizable row `axpy`.
//! - **otherwise** — `i` → `j` → `p` dot products, `COL_BLOCK` output columns
//!   at a time. The case that matters is a transposed `rhs` view
//!   (`rhs_k_stride == 1`), which is what
//!   [`Linear`](crate::nn::Linear)'s `x @ w.T` produces: `p` is already the
//!   contiguous axis there, so the naive nest was cache-friendly but
//!   latency-bound on one dependent add chain. Blocking over `j` gives
//!   `COL_BLOCK` independent chains and reuses each `lhs` load across them.
//!   F32 with a contiguous matrix axis on both inputs instead uses a 4×4
//!   output tile, reusing both activation and weight loads.
//!
//! Both orders accumulate every output element's `k` terms in ascending `p`
//! into one `Acc` slot and narrow once, so each is bitwise-identical to the
//! other and to the naive nest, for every dtype. Nothing here reassociates a
//! sum — blocking is only ever over independent output elements, never within
//! one element's inner product. That is the property the file is written
//! around, and `both_loop_orders_are_bitwise_identical_to_the_naive_nest`
//! pins it.

use super::cpu_storage;
use crate::backend::View;
use crate::backend::cpu::acc::NumAcc;
use crate::backend::cpu::dispatch::{CpuElement, dispatch_numeric};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::storage::Storage;

/// The resolved matmul geometry: batch shape (already broadcast), matrix
/// dims, and per-operand leading-batch strides padded to the batch rank.
struct Plan {
    /// Broadcast batch dims (may be empty for the pure 2-D case).
    batch: Vec<usize>,
    m: usize,
    k: usize,
    n: usize,
    /// lhs strides for the batch axes (0 where broadcast), length == batch rank.
    lhs_batch_strides: Vec<usize>,
    /// rhs strides for the batch axes (0 where broadcast), length == batch rank.
    rhs_batch_strides: Vec<usize>,
    lhs_offset: usize,
    rhs_offset: usize,
    // Trailing-axis strides.
    lhs_m_stride: usize,
    lhs_k_stride: usize,
    rhs_k_stride: usize,
    rhs_n_stride: usize,
}

/// Broadcast two batch-shape prefixes right-aligned, returning the output
/// batch dims and, for each operand, the stride to advance along each output
/// batch axis (0 where that axis is broadcast or padded).
fn plan(lhs: &Layout, rhs: &Layout) -> Result<Plan> {
    let lr = lhs.rank();
    let rr = rhs.rank();
    if lr < 2 || rr < 2 {
        return Err(Error::InvalidArg {
            op: "matmul",
            msg: format!("operands must be rank >= 2, got ranks {lr} and {rr}"),
        });
    }
    let ld = lhs.dims();
    let rd = rhs.dims();
    let ls = lhs.strides();
    let rs = rhs.strides();

    let m = ld[lr - 2];
    let k = ld[lr - 1];
    let rk = rd[rr - 2];
    let n = rd[rr - 1];
    if k != rk {
        // The op layer normally raises this; the kernel double-checks so a
        // mis-wired call is a loud error, not silent garbage.
        return Err(Error::ShapeMismatch {
            op: "matmul",
            lhs: lhs.shape().clone(),
            rhs: rhs.shape().clone(),
        });
    }

    let lb = &ld[..lr - 2];
    let rb = &rd[..rr - 2];
    let batch_rank = lb.len().max(rb.len());
    let mut batch = vec![0usize; batch_rank];
    let mut lhs_batch_strides = vec![0usize; batch_rank];
    let mut rhs_batch_strides = vec![0usize; batch_rank];
    for i in 0..batch_rank {
        // Right-aligned index into each operand's batch dims.
        let li = (i + lb.len()).checked_sub(batch_rank);
        let ri = (i + rb.len()).checked_sub(batch_rank);
        let ldim = li.map_or(1, |j| lb[j]);
        let rdim = ri.map_or(1, |j| rb[j]);
        let out = if ldim == rdim {
            ldim
        } else if ldim == 1 {
            rdim
        } else if rdim == 1 {
            ldim
        } else {
            return Err(Error::ShapeMismatch {
                op: "matmul",
                lhs: lhs.shape().clone(),
                rhs: rhs.shape().clone(),
            });
        };
        batch[i] = out;
        // A broadcast (size-1 or padded) batch axis contributes stride 0.
        lhs_batch_strides[i] = match li {
            Some(j) if lb[j] != 1 => ls[j],
            _ => 0,
        };
        rhs_batch_strides[i] = match ri {
            Some(j) if rb[j] != 1 => rs[j],
            _ => 0,
        };
    }

    Ok(Plan {
        batch,
        m,
        k,
        n,
        lhs_batch_strides,
        rhs_batch_strides,
        lhs_offset: lhs.offset(),
        rhs_offset: rhs.offset(),
        lhs_m_stride: ls[lr - 2],
        lhs_k_stride: ls[lr - 1],
        rhs_k_stride: rs[rr - 2],
        rhs_n_stride: rs[rr - 1],
    })
}

/// Number of output columns accumulated together on the strided-RHS path.
/// Each output still consumes its `k` terms in ascending order; this only
/// exposes independent multiply-add chains.
const COL_BLOCK: usize = 8;

/// Adjacent logical rows processed together on the F32 row-major-rhs path.
/// This reuses each rhs load and also turns the strided lhs walk used by
/// weight-gradient products (`x.T @ grad`) into adjacent loads, without
/// changing any output element's ascending-`k` accumulation order.
const ROW_BLOCK: usize = 4;

/// Split the output across threads by whole rows and hand each contiguous run
/// to `fill`.
///
/// Every matmul kernel below shares this driver, so the blocking, the batch
/// decode, and the threading rule are stated once. `fill(rows, batch, first)`
/// receives a slice holding `rows.len() / n` complete output rows, the batch
/// index they belong to, and the logical row `first` at which they start; a run
/// never straddles a batch boundary.
///
/// `unit_rows` is the row blocking the kernel wants preserved: every run except
/// the last in a window starts at a multiple of it, so the blocked inner loops
/// stay on their fast path instead of degrading to the scalar tail. Work is
/// costed at `k` per output element — the length of the inner product — which
/// is what keeps a small matmul off the thread pool entirely.
fn for_each_row_run<F>(out: &mut [f32], plan: &Plan, unit_rows: usize, fill: F)
where
    F: Fn(&mut [f32], usize, usize) + Send + Sync,
{
    for_each_row_run_generic(out, plan, unit_rows, fill);
}

/// [`for_each_row_run`] over any element type (the generic kernel's output is
/// `E`, not `f32`).
fn for_each_row_run_generic<E, F>(out: &mut [E], plan: &Plan, unit_rows: usize, fill: F)
where
    E: Send,
    F: Fn(&mut [E], usize, usize) + Send + Sync,
{
    let (m, n) = (plan.m, plan.n);
    crate::backend::parallel::for_each_window_mut(out, unit_rows * n, plan.k, |base, window| {
        // `base` and every window length are multiples of `n`: the window unit
        // is `unit_rows * n` and the total is `batch_count * m * n`, so no
        // window can split a row.
        let mut row = base / n;
        let mut pos = 0;
        while pos < window.len() {
            let batch = row / m;
            let first = row % m;
            // Stop at the batch boundary: `fill` resolves one pair of operand
            // bases, which are only valid within a single batch.
            let take = (m - first).min((window.len() - pos) / n);
            fill(&mut window[pos..pos + take * n], batch, first);
            pos += take * n;
            row += take;
        }
    });
}

/// F32's common row-major-rhs path. Keeping the output in its final allocation
/// avoids the generic accumulator row's refill/copy and reuses rhs values
/// across four independent output rows.
fn matmul_f32_row_major_rhs(lhs: &[f32], rhs: &[f32], plan: &Plan) -> Vec<f32> {
    let (m, k, n) = (plan.m, plan.k, plan.n);
    let batch_count: usize = plan.batch.iter().product();
    if batch_count == 0 || m == 0 || n == 0 {
        return Vec::new();
    }

    let mut out = vec![0.0; batch_count * m * n];
    for_each_row_run(&mut out, plan, ROW_BLOCK, |rows, batch, first| {
        let (lhs_base, rhs_base) = batch_bases(plan, batch);
        let count = rows.len() / n;
        let mut i = 0;
        while i + ROW_BLOCK <= count {
            let block = &mut rows[i * n..(i + ROW_BLOCK) * n];
            let (out0, rest) = block.split_at_mut(n);
            let (out1, rest) = rest.split_at_mut(n);
            let (out2, out3) = rest.split_at_mut(n);
            let row = first + i;
            for p in 0..k {
                let rhs_row = &rhs[rhs_base + p * plan.rhs_k_stride..][..n];
                let lhs_col = lhs_base + p * plan.lhs_k_stride;
                let a0 = lhs[lhs_col + row * plan.lhs_m_stride];
                let a1 = lhs[lhs_col + (row + 1) * plan.lhs_m_stride];
                let a2 = lhs[lhs_col + (row + 2) * plan.lhs_m_stride];
                let a3 = lhs[lhs_col + (row + 3) * plan.lhs_m_stride];
                for j in 0..n {
                    let value = rhs_row[j];
                    out0[j] += a0 * value;
                    out1[j] += a1 * value;
                    out2[j] += a2 * value;
                    out3[j] += a3 * value;
                }
            }
            i += ROW_BLOCK;
        }
        while i < count {
            let lhs_row = lhs_base + (first + i) * plan.lhs_m_stride;
            let out_row = &mut rows[i * n..(i + 1) * n];
            for p in 0..k {
                let a = lhs[lhs_row + p * plan.lhs_k_stride];
                let rhs_row = &rhs[rhs_base + p * plan.rhs_k_stride..][..n];
                for (slot, &value) in out_row.iter_mut().zip(rhs_row) {
                    *slot += a * value;
                }
            }
            i += 1;
        }
    });
    out
}

/// F32 kernel for the `Linear` forward layout: a row-major lhs multiplied by
/// a transposed row-major weight. A 4x4 output tile reuses each activation and
/// weight load while every one of its 16 accumulators still visits `k` in
/// ascending order.
fn matmul_f32_transposed_rhs(lhs: &[f32], rhs: &[f32], plan: &Plan) -> Vec<f32> {
    const TILE: usize = 4;
    let (m, k, n) = (plan.m, plan.k, plan.n);
    let batch_count: usize = plan.batch.iter().product();
    if batch_count == 0 || m == 0 || n == 0 {
        return Vec::new();
    }

    let mut out = vec![0.0; batch_count * m * n];
    for_each_row_run(&mut out, plan, TILE, |rows, batch, first| {
        let (lhs_base, rhs_base) = batch_bases(plan, batch);
        let count = rows.len() / n;
        let mut i = 0;
        while i + TILE <= count {
            let row = first + i;
            let mut j = 0;
            while j + TILE <= n {
                let mut acc = [[0.0f32; TILE]; TILE];
                for p in 0..k {
                    let lhs_col = lhs_base + p * plan.lhs_k_stride;
                    let rhs_row = rhs_base + p * plan.rhs_k_stride;
                    let av = [
                        lhs[lhs_col + row * plan.lhs_m_stride],
                        lhs[lhs_col + (row + 1) * plan.lhs_m_stride],
                        lhs[lhs_col + (row + 2) * plan.lhs_m_stride],
                        lhs[lhs_col + (row + 3) * plan.lhs_m_stride],
                    ];
                    let bv = [
                        rhs[rhs_row + j * plan.rhs_n_stride],
                        rhs[rhs_row + (j + 1) * plan.rhs_n_stride],
                        rhs[rhs_row + (j + 2) * plan.rhs_n_stride],
                        rhs[rhs_row + (j + 3) * plan.rhs_n_stride],
                    ];
                    for (acc_row, a) in acc.iter_mut().zip(av) {
                        for (cell, b) in acc_row.iter_mut().zip(bv) {
                            *cell += a * b;
                        }
                    }
                }
                for (ii, acc_row) in acc.iter().enumerate() {
                    let start = (i + ii) * n + j;
                    rows[start..start + TILE].copy_from_slice(acc_row);
                }
                j += TILE;
            }
            while j < n {
                for ii in 0..TILE {
                    let mut acc = 0.0;
                    for p in 0..k {
                        acc += lhs
                            [lhs_base + (row + ii) * plan.lhs_m_stride + p * plan.lhs_k_stride]
                            * rhs[rhs_base + p * plan.rhs_k_stride + j * plan.rhs_n_stride];
                    }
                    rows[(i + ii) * n + j] = acc;
                }
                j += 1;
            }
            i += TILE;
        }
        while i < count {
            let lhs_row = lhs_base + (first + i) * plan.lhs_m_stride;
            for j in 0..n {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += lhs[lhs_row + p * plan.lhs_k_stride]
                        * rhs[rhs_base + p * plan.rhs_k_stride + j * plan.rhs_n_stride];
                }
                rows[i * n + j] = acc;
            }
            i += 1;
        }
    });
    out
}

/// Generic batched matmul accumulating each inner product in `E::Acc` and
/// casting once at output. `lhs`/`rhs` are the whole backing buffers. See the
/// module docs for why the loop order is chosen from the `rhs` strides.
fn matmul_generic<E>(lhs: &[E], rhs: &[E], plan: &Plan) -> Vec<E>
where
    E: Element,
    E::Acc: NumAcc,
{
    let (m, k, n) = (plan.m, plan.k, plan.n);
    // A rank-2 operand pair has no batch axes, so the product is 1 and the
    // single "batch" is the matrix itself.
    let batch_count: usize = plan.batch.iter().product();
    // An empty output has nothing to accumulate. Returning here keeps the
    // row-slice below from having to reason about a zero-length `rhs` range
    // whose base may sit past the end of a zero-sized buffer, and keeps
    // `batch_bases` off a zero-extent batch axis (which it would divide by).
    // `k == 0` deliberately does *not* return early: that still produces `ZERO`
    // for every output element, as it always has.
    if batch_count == 0 || m == 0 || n == 0 {
        return Vec::new();
    }
    let zero = E::from_acc(<E::Acc as NumAcc>::ZERO);
    let mut out = vec![zero; batch_count * m * n];
    // One output row at a time either way, so the run driver preserves no
    // blocking here and asks only that a run not split a row.
    for_each_row_run_generic(&mut out, plan, 1, |rows, batch, first| {
        let (lhs_base, rhs_base) = batch_bases(plan, batch);
        let count = rows.len() / n;
        if plan.rhs_n_stride == 1 {
            // One wide accumulator slot per output column, allocated once per
            // task and refilled per output row so the hot loops never allocate.
            let mut acc_row = vec![<E::Acc as NumAcc>::ZERO; n];
            for i in 0..count {
                let lhs_row = lhs_base + (first + i) * plan.lhs_m_stride;
                acc_row.fill(<E::Acc as NumAcc>::ZERO);
                for p in 0..k {
                    let a = lhs[lhs_row + p * plan.lhs_k_stride].to_acc();
                    // In bounds for any valid view: with an `n` stride of 1 the
                    // last element of this row is the view's own maximal index
                    // `offset + p*k_stride + (n-1)*1`. Slicing once per `p`
                    // hoists the bounds check out of the `j` loop, which is
                    // what lets it vectorize.
                    let rhs_row = &rhs[rhs_base + p * plan.rhs_k_stride..][..n];
                    for (slot, value) in acc_row.iter_mut().zip(rhs_row) {
                        *slot = slot.mul_add(a, value.to_acc());
                    }
                }
                let out_row = &mut rows[i * n..(i + 1) * n];
                for (slot, &acc) in out_row.iter_mut().zip(acc_row.iter()) {
                    *slot = E::from_acc(acc);
                }
            }
        } else {
            for i in 0..count {
                let lhs_row = lhs_base + (first + i) * plan.lhs_m_stride;
                let out_row = &mut rows[i * n..(i + 1) * n];
                let mut j = 0;
                while j + COL_BLOCK <= n {
                    let mut accs = [<E::Acc as NumAcc>::ZERO; COL_BLOCK];
                    let col_base = rhs_base + j * plan.rhs_n_stride;
                    for p in 0..k {
                        let a = lhs[lhs_row + p * plan.lhs_k_stride].to_acc();
                        let row = col_base + p * plan.rhs_k_stride;
                        for (u, acc) in accs.iter_mut().enumerate() {
                            *acc = acc.mul_add(a, rhs[row + u * plan.rhs_n_stride].to_acc());
                        }
                    }
                    for (slot, acc) in out_row[j..j + COL_BLOCK].iter_mut().zip(accs) {
                        *slot = E::from_acc(acc);
                    }
                    j += COL_BLOCK;
                }
                // Tail columns, and every column when `n < COL_BLOCK`.
                while j < n {
                    let rhs_col = rhs_base + j * plan.rhs_n_stride;
                    let mut acc = <E::Acc as NumAcc>::ZERO;
                    for p in 0..k {
                        let a = lhs[lhs_row + p * plan.lhs_k_stride].to_acc();
                        let bx = rhs[rhs_col + p * plan.rhs_k_stride].to_acc();
                        acc = acc.mul_add(a, bx);
                    }
                    out_row[j] = E::from_acc(acc);
                    j += 1;
                }
            }
        }
    });
    out
}

/// Decode a linear batch index into per-axis coords (row-major over the
/// broadcast batch dims) and return the two operand base offsets it selects.
/// Only called with `b` below the batch count, so every extent is non-zero and
/// the modulo is well defined.
fn batch_bases(plan: &Plan, b: usize) -> (usize, usize) {
    let mut lhs_base = plan.lhs_offset;
    let mut rhs_base = plan.rhs_offset;
    let mut rem = b;
    for ax in (0..plan.batch.len()).rev() {
        let size = plan.batch[ax];
        let coord = rem % size;
        rem /= size;
        lhs_base += coord * plan.lhs_batch_strides[ax];
        rhs_base += coord * plan.rhs_batch_strides[ax];
    }
    (lhs_base, rhs_base)
}

/// See [`BackendOps::matmul`](crate::backend::BackendOps::matmul).
pub(crate) fn matmul(lhs: View<'_>, rhs: View<'_>) -> Result<Storage> {
    if lhs.dtype() != rhs.dtype() {
        return Err(Error::DTypeMismatch {
            op: "matmul",
            expected: lhs.dtype(),
            got: rhs.dtype(),
        });
    }
    let plan = plan(lhs.layout(), rhs.layout())?;
    let lhs_cpu = cpu_storage(lhs);
    let rhs_cpu = cpu_storage(rhs);
    // F32 is the one dtype with dedicated stride-specialized kernels, so it is
    // taken before the dispatch rather than inside it: which of the three runs
    // is a property of the plan, not of the element type.
    if lhs.dtype() == DType::F32 {
        let (a, b) = (f32::slice(lhs_cpu), f32::slice(rhs_cpu));
        let out = if plan.rhs_n_stride == 1 {
            matmul_f32_row_major_rhs(a, b, &plan)
        } else if plan.lhs_k_stride == 1 && plan.rhs_k_stride == 1 {
            matmul_f32_transposed_rhs(a, b, &plan)
        } else {
            matmul_generic(a, b, &plan)
        };
        return Ok(f32::storage(out));
    }
    dispatch_numeric!(lhs.dtype(), "matmul", lhs.device(), E => {
        Ok(E::storage(matmul_generic(E::slice(lhs_cpu), E::slice(rhs_cpu), &plan)))
    })
}

#[cfg(test)]
mod tests;
