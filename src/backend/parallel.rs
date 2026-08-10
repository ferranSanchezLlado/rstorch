//! The parallelism switch: a thin façade over slice iteration that becomes a
//! rayon parallel iterator.
//!
//! # Why this module is always compiled
//!
//! The original façade was itself behind `#[cfg(feature = "rayon")]`, so every
//! kernel that wanted it had to carry two copies of its loop body — one under
//! `#[cfg(feature = "rayon")]` calling the façade, one under
//! `#[cfg(not(...))]` repeating the same loop sequentially. That cost is why
//! adoption stalled at two call sites in `cpu::elementwise` while matmul,
//! reduce, conv, index, and the fused optimizer steps stayed single-threaded.
//!
//! This module is now compiled unconditionally and the feature switch lives
//! *inside* each helper. A kernel adopts parallelism by writing its loop body
//! **once**, and the same source runs sequentially in a default build. The
//! `Send`/`Sync` bounds are also unconditional, so a body that compiles without
//! the feature is guaranteed to compile with it.
//!
//! # The work model
//!
//! Splitting work across threads costs on the order of a microsecond per task,
//! which a small tensor never earns back. Callers therefore describe the size
//! of the job in **work units** as `output_len * cost_per_element`.
//!
//! One work unit is calibrated to **one multiply-add step of a blocked,
//! vectorized F32 matmul** — the cheapest inner step in the crate, which on an
//! M4 Pro core retires at roughly 15 per nanosecond. A step that is dearer than
//! that costs proportionally more units, so a caller whose inner loop is
//! memory-bound rather than arithmetic-bound must say so:
//!
//! | Kernel | `cost_per_element` |
//! |---|---|
//! | matmul | the contracted dimension `k` (one FMA each) |
//! | conv2d / pooling | window size × input channels |
//! | softmax / layernorm | ~32 per element; `exp` and the repeated row sweeps |
//! | dense element-wise | [`STREAMING_COST`] — bandwidth-bound, not FMA-bound |
//! | strided element-wise | [`STREAMING_COST`] plus the index walk per input |
//!
//! Getting this wrong by a small factor only shifts where the thresholds bite;
//! getting it wrong by an order of magnitude is what makes a kernel either
//! thrash the pool on tiny inputs or leave cores idle on large ones. Both
//! failure modes were measured while calibrating the constants below.
//!
//! The estimate never affects *what* a kernel computes — only whether and how
//! finely it is split.
//!
//! # Determinism
//!
//! Every helper partitions an output slice: each slot is written by exactly one
//! task, from the same inputs, in the same order within a task. Results do not
//! depend on the feature flag, the thread count, or the scheduling order. This
//! is the whole reason the façade hands out disjoint `&mut` windows rather than
//! a reduction combinator — a floating-point reduction whose association order
//! followed the thread count would make `rayon` a correctness switch instead of
//! a speed one.
//!
//! # What is deliberately *not* driven from here
//!
//! - **Axis reductions** (`cpu::reduce`). Restructuring the `Vec::push` loop
//!   into a window body put the hot `for_each_on_axis` walk behind two layers
//!   of closure and cost 12–60% on `max_last`/`sum_all`, while the shapes that
//!   occur in training sit below the parallel threshold and so gained nothing
//!   back. Reverting to the original loop restored baseline exactly. Making
//!   large reductions parallel needs a body that survives the nesting, not a
//!   different threshold.
//! - **Scatter-shaped backward kernels** — conv input/weight gradients and both
//!   pool backwards (`cpu::conv`), and `index_add`/`scatter_add` (`cpu::index`).
//!   These accumulate into a shared input-shaped buffer where two output
//!   positions can target the same slot; splitting them needs either a lock or
//!   a per-thread accumulator whose merge would reassociate a gradient sum.
//! - **`copy_view` / `cast` (`cpu::host`)**. Pure memory movement driven by a
//!   stateful `Walk` cursor that cannot be seeded at an arbitrary offset
//!   without being rewritten.

/// The smallest amount of work a single task is given.
///
/// Calibrated against the cheapest work unit in the crate, a matmul inner-
/// product step: the blocked F32 kernels retire on the order of 15 of those per
/// nanosecond, so this floor is ~17 µs of arithmetic there and proportionally
/// more in kernels whose units are dearer. Spawning and joining a `rayon` task
/// costs on the order of a microsecond, so this keeps scheduling under roughly
/// a tenth of a task's runtime even in the worst case.
///
/// Measured on `matmul/square_f32` (M4 Pro, 12 threads). A floor of 16 Ki left
/// the 128³ case handing out 4-row tasks of ~4 µs each and scaling only 1.5×;
/// at this floor the same case takes ~16-row tasks and scales far better.
#[cfg(any(feature = "rayon", test))]
const MIN_TASK_WORK: usize = 1 << 18;

/// Total work below which a kernel runs sequentially.
///
/// Two whole tasks' worth: below this there is not enough work to keep even a
/// second thread busy past its own spawn cost. `matmul/square_f32/64` sits just
/// under it at 2^18 units and was 2.3× *slower* threaded, which is what fixed
/// this constant.
#[cfg(any(feature = "rayon", test))]
const PARALLEL_MIN_WORK: usize = 2 * MIN_TASK_WORK;

/// The cost of one element of a bandwidth-bound streaming kernel — a load, an
/// arithmetic op or two, and a store — in the FMA-calibrated units above.
///
/// Element-wise kernels move several bytes per element and cannot hide that
/// behind arithmetic, so they retire elements a few times slower than a blocked
/// matmul retires multiply-adds. Weighting them at 1 made a 1 Mi-element `add`
/// take four coarse tasks and lose ~18% to the sequential form; weighting them
/// here restores the ~16-task split that measured best.
pub(crate) const STREAMING_COST: usize = 4;

/// How many tasks to aim for per thread. More than one so that an uneven
/// window — a ragged tail, a core stolen by another process — does not leave
/// the whole join waiting on a single straggler.
#[cfg(any(feature = "rayon", test))]
const TASKS_PER_THREAD: usize = 4;

/// The number of threads the pool will actually use (1 without `rayon`).
///
/// Cached. Every kernel call asks this question at least twice, and
/// `rayon::current_num_threads` reaches through to the global registry —
/// cheap in isolation, but it is on the entry path of even the reductions that
/// go on to run sequentially, where it measured ~12% of a small `max` and
/// ~48% of a `sum_all`. The global pool's size is fixed once it is built, so
/// there is nothing to invalidate.
#[cfg(any(feature = "rayon", test))]
#[inline]
fn thread_count() -> usize {
    #[cfg(feature = "rayon")]
    {
        static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *THREADS.get_or_init(|| rayon::current_num_threads().max(1))
    }
    #[cfg(not(feature = "rayon"))]
    {
        1
    }
}

/// Whether a job of `work` units earns the cost of going parallel.
///
/// Deliberately private: the decision is this module's alone, so kernels state
/// only what their work *costs* and never branch on whether it is threaded.
#[cfg(any(feature = "rayon", test))]
#[inline]
fn should_parallelize(work: usize) -> bool {
    thread_count() > 1 && work >= PARALLEL_MIN_WORK
}

/// The window length, in elements, to hand each task: a multiple of `unit`
/// chosen to give every thread several tasks without letting any task fall
/// below [`MIN_TASK_WORK`].
#[inline]
#[cfg(any(feature = "rayon", test))]
fn window_len(total: usize, unit: usize, cost_per_element: usize) -> usize {
    let unit = unit.max(1);
    let units_total = total.div_ceil(unit);
    // Load balance: aim for `TASKS_PER_THREAD` tasks per thread.
    let balanced = units_total
        .div_ceil(thread_count() * TASKS_PER_THREAD)
        .max(1);
    // Granularity floor: never hand out a task smaller than `MIN_TASK_WORK`.
    let work_per_unit = unit.saturating_mul(cost_per_element).max(1);
    let floor = MIN_TASK_WORK.div_ceil(work_per_unit).max(1);
    // `units_total` caps both so a small output still yields one whole window.
    balanced
        .max(floor)
        .min(units_total.max(1))
        .saturating_mul(unit)
}

/// Apply `f(index, &mut element)` to every element of `values`, in parallel
/// when the job is large enough to pay for it.
///
/// The index is the element's position in `values`. `cost_per_element` is the
/// caller's estimate of the work each element costs (see the module docs);
/// pass 1 for a plain element-wise map.
pub(crate) fn for_each_mut<T, F>(values: &mut [T], cost_per_element: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut T) + Send + Sync,
{
    #[cfg(feature = "rayon")]
    if should_parallelize(values.len().saturating_mul(cost_per_element.max(1))) {
        use rayon::prelude::*;
        // Chunked rather than `par_iter_mut().enumerate()`: rayon splits a
        // chunk iterator by whole chunks, so the per-task floor above is
        // actually honored instead of being re-split down to single elements.
        let chunk = window_len(values.len(), 1, cost_per_element);
        values
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(task, window)| {
                let base = task * chunk;
                for (offset, slot) in window.iter_mut().enumerate() {
                    f(base + offset, slot);
                }
            });
        return;
    }
    let _ = cost_per_element;
    for (idx, slot) in values.iter_mut().enumerate() {
        f(idx, slot);
    }
}

/// Apply `body(start_index, window)` to contiguous windows of `values`, in
/// parallel when the job is large enough to pay for it.
///
/// This is the driver kernels with structured output should reach for: it lets
/// the body keep its blocking, its per-row setup, and its scratch buffers
/// instead of paying that cost per element.
///
/// - `unit` is the indivisible run of elements one call must receive — 1 for an
///   element-wise kernel, the row length `n` for a matmul, `ROW_BLOCK * n` for
///   one that blocks four rows at a time. Every window is a multiple of `unit`
///   **except possibly the last**, which holds whatever remains.
/// - `start_index` is the window's offset in `values`, always a multiple of the
///   chosen window length.
/// - `cost_per_element` is as in the module docs.
///
/// The whole slice arrives in a single call when the job runs sequentially, so
/// a body must not assume it is given exactly `unit` elements.
pub(crate) fn for_each_window_mut<T, F>(
    values: &mut [T],
    unit: usize,
    cost_per_element: usize,
    body: F,
) where
    T: Send,
    F: Fn(usize, &mut [T]) + Send + Sync,
{
    if values.is_empty() {
        return;
    }
    #[cfg(feature = "rayon")]
    if should_parallelize(values.len().saturating_mul(cost_per_element.max(1))) {
        use rayon::prelude::*;
        let chunk = window_len(values.len(), unit, cost_per_element);
        if chunk < values.len() {
            values
                .par_chunks_mut(chunk)
                .enumerate()
                .for_each(|(task, window)| body(task * chunk, window));
            return;
        }
    }
    let _ = (unit, cost_per_element);
    body(0, values);
}

/// One output buffer of a row-structured kernel: the buffer and how many of its
/// elements belong to each row.
///
/// A width of 1 makes "row" mean "element", which is how the fused optimizer
/// steps use the row drivers.
pub(crate) type RowOutput<'a, T> = (&'a mut [T], usize);

/// How many rows to give each task, and a debug check that every output is
/// exactly `rows * width` long.
#[cfg(feature = "rayon")]
#[inline]
fn rows_per_task(rows: usize, cost_per_row: usize, widths: &[(usize, usize)]) -> usize {
    for &(len, width) in widths {
        debug_assert_eq!(
            len,
            rows * width,
            "row-structured output length must be rows * width"
        );
    }
    window_len(rows, 1, cost_per_row)
}

/// Split `rows` rows of work across threads, handing each task the matching
/// window of **two** outputs.
///
/// Each output is chunked by its own per-row width, so buffers carrying a
/// different number of elements per row still line up row for row. `body`
/// receives the index of the first row in the window.
#[allow(unused_variables)]
pub(crate) fn for_each_row_mut2<A, B, F>(
    rows: usize,
    a: RowOutput<'_, A>,
    b: RowOutput<'_, B>,
    cost_per_row: usize,
    body: F,
) where
    A: Send,
    B: Send,
    F: Fn(usize, &mut [A], &mut [B]) + Send + Sync,
{
    let (a, aw) = a;
    let (b, bw) = b;
    if rows == 0 {
        return;
    }
    #[cfg(feature = "rayon")]
    if should_parallelize(rows.saturating_mul(cost_per_row.max(1))) {
        use rayon::prelude::*;
        let step = rows_per_task(rows, cost_per_row, &[(a.len(), aw), (b.len(), bw)]);
        if step < rows {
            // Equal chunk counts on both sides, so `zip` pairs window `i` of
            // one output with window `i` of the other.
            a.par_chunks_mut(step * aw)
                .zip(b.par_chunks_mut(step * bw))
                .enumerate()
                .for_each(|(task, (wa, wb))| body(task * step, wa, wb));
            return;
        }
    }
    let _ = cost_per_row;
    body(0, a, b);
}

/// [`for_each_row_mut2`] over three outputs — layer norm, which writes the
/// normalized activation, the saved `xhat`, and one inverse standard deviation
/// per row in a single pass.
#[allow(unused_variables)]
pub(crate) fn for_each_row_mut3<A, B, C, F>(
    rows: usize,
    a: RowOutput<'_, A>,
    b: RowOutput<'_, B>,
    c: RowOutput<'_, C>,
    cost_per_row: usize,
    body: F,
) where
    A: Send,
    B: Send,
    C: Send,
    F: Fn(usize, &mut [A], &mut [B], &mut [C]) + Send + Sync,
{
    let (a, aw) = a;
    let (b, bw) = b;
    let (c, cw) = c;
    if rows == 0 {
        return;
    }
    #[cfg(feature = "rayon")]
    if should_parallelize(rows.saturating_mul(cost_per_row.max(1))) {
        use rayon::prelude::*;
        let step = rows_per_task(
            rows,
            cost_per_row,
            &[(a.len(), aw), (b.len(), bw), (c.len(), cw)],
        );
        if step < rows {
            a.par_chunks_mut(step * aw)
                .zip(b.par_chunks_mut(step * bw))
                .zip(c.par_chunks_mut(step * cw))
                .enumerate()
                .for_each(|(task, ((wa, wb), wc))| body(task * step, wa, wb, wc));
            return;
        }
    }
    let _ = cost_per_row;
    body(0, a, b, c);
}

/// Build a `len`-element output by filling windows of it with `body`.
///
/// The buffer is pre-filled with `zero` and then overwritten: the drivers
/// partition an existing slice, so unlike a sequential `collect` they cannot
/// write into uninitialized memory. For the zeroable element types this
/// pre-fill is `alloc_zeroed` over fresh pages rather than a second traversal.
///
/// `zero` is a value, not a `Default` bound, so element types that carry their
/// identity as an associated constant can pass it directly.
pub(crate) fn build<T, F>(
    len: usize,
    zero: T,
    unit: usize,
    cost_per_element: usize,
    body: F,
) -> Vec<T>
where
    T: Clone + Send,
    F: Fn(usize, &mut [T]) + Send + Sync,
{
    let mut out = vec![zero; len];
    for_each_window_mut(&mut out, unit, cost_per_element, body);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cost that forces the parallel branch on any machine with a pool.
    const HOT: usize = PARALLEL_MIN_WORK;

    #[test]
    fn for_each_mut_writes_index_mapped_values() {
        let mut out = vec![0i64; 64];
        for_each_mut(&mut out, HOT, |idx, value| *value = (idx as i64) * 2);
        let expected: Vec<i64> = (0..64).map(|i| i * 2).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn for_each_mut_empty_is_noop() {
        let mut out: Vec<f32> = Vec::new();
        for_each_mut(&mut out, HOT, |_, v| *v = 1.0);
        assert!(out.is_empty());
    }

    #[test]
    fn for_each_mut_agrees_across_the_threshold() {
        // The same body, once below the parallel threshold and once above it.
        let fill = |cost: usize| {
            let mut out = vec![0usize; 4096];
            for_each_mut(&mut out, cost, |idx, slot| *slot = idx * 3);
            out
        };
        assert_eq!(fill(1), fill(HOT));
    }

    #[test]
    fn windows_partition_the_output_exactly_once() {
        // Every slot written exactly once, with its own global index, for a
        // length that is not a multiple of the unit.
        for len in [1usize, 7, 64, 1000, 4096, 10_007] {
            let mut out = vec![usize::MAX; len];
            for_each_window_mut(&mut out, 8, HOT, |base, window| {
                for (offset, slot) in window.iter_mut().enumerate() {
                    *slot = base + offset;
                }
            });
            let expected: Vec<usize> = (0..len).collect();
            assert_eq!(out, expected, "len {len}");
        }
    }

    #[test]
    fn windows_are_unit_aligned_except_the_last() {
        // A body that blocks by `unit` may only rely on full windows being
        // multiples of it; the tail is whatever remains.
        let unit = 16;
        let len = 10_000;
        let mut out = vec![0usize; len];
        let bad = std::sync::Mutex::new(Vec::new());
        for_each_window_mut(&mut out, unit, HOT, |base, window| {
            if base % unit != 0 || (base + window.len() < len && window.len() % unit != 0) {
                bad.lock().unwrap().push((base, window.len()));
            }
        });
        assert!(bad.lock().unwrap().is_empty(), "{:?}", bad.lock().unwrap());
    }

    #[test]
    fn sequential_jobs_arrive_as_one_window() {
        // Below the threshold the body must still see the whole slice, so a
        // kernel may size its scratch from the window it is handed.
        use std::sync::atomic::{AtomicUsize, Ordering};
        let mut out = vec![0u8; 32];
        let calls = AtomicUsize::new(0);
        for_each_window_mut(&mut out, 4, 1, |base, window| {
            calls.fetch_add(1, Ordering::Relaxed);
            assert_eq!(base, 0);
            assert_eq!(window.len(), 32);
        });
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn window_len_never_exceeds_the_output() {
        // A unit larger than the whole output must still produce one window
        // that covers it, not a zero-length chunk (`par_chunks_mut` panics).
        assert_eq!(window_len(10, 64, 1), 64);
        assert!(window_len(10, 64, 1) >= 10);
        assert!(window_len(0, 8, 1) > 0);
    }

    #[test]
    fn window_len_is_a_multiple_of_unit() {
        for &(total, unit, cost) in &[
            (1_000_000usize, 512usize, 512usize),
            (4096, 64, 64),
            (10_007, 8, 1),
            (1 << 20, 1, 1),
        ] {
            assert_eq!(window_len(total, unit, cost) % unit, 0);
        }
    }

    #[test]
    fn window_len_respects_the_task_work_floor() {
        // Cheap per element: windows must be coarse enough to clear the floor.
        let unit = 1;
        let cost = 1;
        let len = window_len(1 << 24, unit, cost);
        assert!(len >= MIN_TASK_WORK, "{len}");
    }

    #[test]
    fn build_fills_every_slot() {
        let out: Vec<u32> = build(1000, 0, 4, HOT, |base, window| {
            for (offset, slot) in window.iter_mut().enumerate() {
                *slot = (base + offset) as u32;
            }
        });
        assert_eq!(out, (0..1000u32).collect::<Vec<_>>());
    }

    #[test]
    fn should_parallelize_is_false_for_small_jobs() {
        assert!(!should_parallelize(0));
        assert!(!should_parallelize(PARALLEL_MIN_WORK - 1));
    }
}
