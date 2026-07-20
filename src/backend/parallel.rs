//! The parallelism switch: a thin façade over slice iteration that becomes a
//! rayon parallel iterator (ported from the v2 `backend/parallel.rs`,
//! implementation-plan §4 T10b).
//!
//! This module is compiled **only under the `rayon` feature** (its `mod` line
//! in `backend/mod.rs` is `#[cfg(feature = "rayon")]`). Kernels call these
//! helpers from a `#[cfg(feature = "rayon")]` arm and fall back to a plain
//! sequential loop otherwise, so the same loop body runs sequentially or in
//! parallel with no dependency on rayon in the default build. The closures are
//! `Fn + Send + Sync` to satisfy rayon's bounds.

use rayon::prelude::*;

/// Apply `f(index, &mut element)` to every element of `values` in parallel.
///
/// The index is the element's position in `values`.
pub(crate) fn for_each_mut<T, F>(values: &mut [T], f: F)
where
    T: Send,
    F: Fn(usize, &mut T) + Send + Sync,
{
    values
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, value)| f(idx, value));
}

/// Apply `f(chunk_index, &mut chunk)` to every `chunk_len`-sized chunk of
/// `values` in parallel (the final chunk may be shorter).
///
/// The index counts chunks, not elements.
pub(crate) fn for_each_chunk_mut<T, F>(values: &mut [T], chunk_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Send + Sync,
{
    values
        .par_chunks_mut(chunk_len)
        .enumerate()
        .for_each(|(idx, chunk)| f(idx, chunk));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn for_each_mut_writes_index_mapped_values() {
        let mut out = vec![0i64; 64];
        for_each_mut(&mut out, |idx, value| *value = (idx as i64) * 2);
        let expected: Vec<i64> = (0..64).map(|i| i * 2).collect();
        assert_eq!(out, expected);
    }

    #[test]
    fn for_each_mut_empty_is_noop() {
        let mut out: Vec<f32> = Vec::new();
        for_each_mut(&mut out, |_, v| *v = 1.0);
        assert!(out.is_empty());
    }

    #[test]
    fn for_each_chunk_mut_covers_ragged_tail() {
        // 10 elements in chunks of 4 -> chunks of len 4, 4, 2.
        let mut out = vec![0usize; 10];
        for_each_chunk_mut(&mut out, 4, |chunk_idx, chunk| {
            for (i, slot) in chunk.iter_mut().enumerate() {
                *slot = chunk_idx * 100 + i;
            }
        });
        assert_eq!(out, vec![0, 1, 2, 3, 100, 101, 102, 103, 200, 201]);
    }

    #[test]
    fn for_each_chunk_mut_chunk_len_larger_than_slice() {
        let mut out = vec![0i32; 3];
        for_each_chunk_mut(&mut out, 8, |chunk_idx, chunk| {
            assert_eq!(chunk_idx, 0);
            assert_eq!(chunk.len(), 3);
            for (i, slot) in chunk.iter_mut().enumerate() {
                *slot = i as i32;
            }
        });
        assert_eq!(out, vec![0, 1, 2]);
    }
}
