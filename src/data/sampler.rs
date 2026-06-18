use crate::rng::SmallRng;
use std::ops::Range;
use std::slice;

/// Sequential or randomized source of dataset indices.
pub trait Sampler {
    type Iter<'a>: Iterator<Item = usize> + ExactSizeIterator
    where
        Self: 'a;

    fn iter(&self) -> Self::Iter<'_>;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Samples indices in ascending order without allocating index storage.
#[derive(Debug, Clone, Copy)]
pub struct SequentialSampler {
    len: usize,
}

/// Samples a precomputed random index order.
#[derive(Debug, Clone)]
pub struct RandomSampler {
    indices: Vec<usize>,
}

/// Groups a sampler into fixed-size compile-time batches.
#[derive(Debug, Clone)]
pub struct BatchSampler<S, const BATCH: usize> {
    pub(crate) sampler: S,
}

/// Groups a sampler into runtime-sized batches, including a trailing partial batch.
#[derive(Debug, Clone)]
pub struct PartialBatchSampler<S> {
    sampler: S,
    batch_size: usize,
}

pub struct BatchSamplerIter<I, const BATCH: usize> {
    indices: I,
}

pub struct PartialBatchSamplerIter<I> {
    pub(crate) indices: I,
    pub(crate) batch_size: usize,
}

impl SequentialSampler {
    pub fn new(len: usize) -> Self {
        Self { len }
    }
}

impl Sampler for SequentialSampler {
    type Iter<'a> = Range<usize>;

    fn iter(&self) -> Self::Iter<'_> {
        0..self.len
    }

    fn len(&self) -> usize {
        self.len
    }
}

impl RandomSampler {
    pub fn new(len: usize, rng: &mut SmallRng) -> Self {
        Self::without_replacement(len, rng)
    }

    pub fn without_replacement(len: usize, rng: &mut SmallRng) -> Self {
        let mut indices: Vec<_> = (0..len).collect();
        shuffle_indices(&mut indices, rng);
        Self { indices }
    }

    pub fn with_replacement(len: usize, num_samples: usize, rng: &mut SmallRng) -> Self {
        assert!(len > 0, "cannot sample from an empty dataset");
        let indices = (0..num_samples)
            .map(|_| (rng.next_u64() as usize) % len)
            .collect();
        Self { indices }
    }
}

impl Sampler for RandomSampler {
    type Iter<'a> = std::iter::Copied<slice::Iter<'a, usize>>;

    fn iter(&self) -> Self::Iter<'_> {
        self.indices.iter().copied()
    }

    fn len(&self) -> usize {
        self.indices.len()
    }
}

impl<T> Sampler for &T
where
    T: Sampler + ?Sized,
{
    type Iter<'a>
        = T::Iter<'a>
    where
        Self: 'a;

    fn iter(&self) -> Self::Iter<'_> {
        (**self).iter()
    }

    fn len(&self) -> usize {
        (**self).len()
    }
}

impl<S, const BATCH: usize> BatchSampler<S, BATCH>
where
    S: Sampler,
{
    pub fn new(sampler: S) -> Self {
        assert!(BATCH > 0, "batch size must be greater than zero");
        Self { sampler }
    }

    pub fn iter(&self) -> BatchSamplerIter<S::Iter<'_>, BATCH> {
        BatchSamplerIter {
            indices: self.sampler.iter(),
        }
    }

    pub fn len(&self) -> usize {
        self.sampler.len() / BATCH
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<S> PartialBatchSampler<S>
where
    S: Sampler,
{
    pub fn new(sampler: S, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch size must be greater than zero");
        Self {
            sampler,
            batch_size,
        }
    }

    pub fn iter(&self) -> PartialBatchSamplerIter<S::Iter<'_>> {
        PartialBatchSamplerIter {
            indices: self.sampler.iter(),
            batch_size: self.batch_size,
        }
    }

    pub fn len(&self) -> usize {
        let len = self.sampler.len();
        (len + self.batch_size - 1) / self.batch_size
    }
}

impl<I, const BATCH: usize> Iterator for BatchSamplerIter<I, BATCH>
where
    I: Iterator<Item = usize> + ExactSizeIterator,
{
    type Item = [usize; BATCH];

    fn next(&mut self) -> Option<Self::Item> {
        if self.indices.len() < BATCH {
            return None;
        }

        Some(std::array::from_fn(|_| {
            self.indices
                .next()
                .expect("batch sampler checked enough indices are available")
        }))
    }
}

impl<I> Iterator for PartialBatchSamplerIter<I>
where
    I: Iterator<Item = usize> + ExactSizeIterator,
{
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.indices.len() == 0 {
            return None;
        }

        let batch_size = self.batch_size.min(self.indices.len());
        Some(self.indices.by_ref().take(batch_size).collect())
    }
}

pub(crate) fn shuffle_indices(indices: &mut [usize], rng: &mut SmallRng) {
    for index in (1..indices.len()).rev() {
        let swap_with = (rng.next_u64() as usize) % (index + 1);
        indices.swap(index, swap_with);
    }
}

#[cfg(test)]
mod tests {
    use super::{BatchSampler, RandomSampler, Sampler, SequentialSampler};
    use crate::rng::SmallRng;

    #[test]
    fn sequential_sampler_does_not_allocate_index_order() {
        let sampler = SequentialSampler::new(5);

        assert_eq!(sampler.len(), 5);
        assert_eq!(sampler.iter().collect::<Vec<_>>(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn random_sampler_is_deterministic_for_seeded_rng() {
        let mut rng_a = SmallRng::seed_from_u64(99);
        let mut rng_b = SmallRng::seed_from_u64(99);
        let sampler_a = RandomSampler::without_replacement(5, &mut rng_a);
        let sampler_b = RandomSampler::without_replacement(5, &mut rng_b);

        assert_eq!(
            sampler_a.iter().collect::<Vec<_>>(),
            sampler_b.iter().collect::<Vec<_>>()
        );
    }

    #[test]
    fn batch_sampler_drops_trailing_partial_batch() {
        let sampler = BatchSampler::<_, 2>::new(SequentialSampler::new(5));

        assert_eq!(sampler.len(), 2);
        assert_eq!(sampler.iter().collect::<Vec<_>>(), vec![[0, 1], [2, 3]]);
    }
}
