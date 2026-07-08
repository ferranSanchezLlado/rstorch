#[cfg(feature = "rayon")]
use rayon::prelude::*;

pub(crate) fn for_each_mut<T, F>(values: &mut [T], f: F)
where
    T: Send,
    F: Fn(usize, &mut T) + Send + Sync,
{
    #[cfg(feature = "rayon")]
    values
        .par_iter_mut()
        .enumerate()
        .for_each(|(idx, value)| f(idx, value));

    #[cfg(not(feature = "rayon"))]
    values
        .iter_mut()
        .enumerate()
        .for_each(|(idx, value)| f(idx, value));
}

pub(crate) fn for_each_chunk_mut<T, F>(values: &mut [T], chunk_len: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Send + Sync,
{
    #[cfg(feature = "rayon")]
    values
        .par_chunks_mut(chunk_len)
        .enumerate()
        .for_each(|(idx, chunk)| f(idx, chunk));

    #[cfg(not(feature = "rayon"))]
    values
        .chunks_mut(chunk_len)
        .enumerate()
        .for_each(|(idx, chunk)| f(idx, chunk));
}
