use rand::distributions::Uniform;
use rand::prelude::*;
use std::iter::FusedIterator;
use std::ops::Range;

pub trait Sampler {
    type Iter: Iterator<Item = usize>;

    fn iter(&mut self) -> Self::Iter;

    fn len(&self) -> usize;

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct SequentialSampler {
    size: usize,
}

impl SequentialSampler {
    #[inline]
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl Sampler for SequentialSampler {
    type Iter = Range<usize>;

    #[inline]
    fn iter(&mut self) -> Self::Iter {
        0..self.size
    }

    #[inline]
    fn len(&self) -> usize {
        self.size
    }
}

pub struct RandomSampler<R> {
    size: usize,
    num_samples: usize,
    replacement: bool,
    rng: R,
}

impl<R: Rng> RandomSampler<R> {
    #[inline]
    #[must_use]
    pub fn new(size: usize, replacement: bool, num_samples: usize, rng: R) -> Self {
        Self {
            size,
            num_samples,
            replacement,
            rng,
        }
    }

    #[inline]
    #[must_use]
    pub fn new_random(
        size: usize,
        replacement: bool,
        num_samples: usize,
    ) -> RandomSampler<ThreadRng> {
        RandomSampler {
            size,
            num_samples,
            replacement,
            rng: ThreadRng::default(),
        }
    }
}

pub struct RandomSamplerIter {
    indices: Vec<usize>,
}

impl RandomSamplerIter {
    #[inline]
    #[must_use]
    fn new<R: Rng>(size: usize, replacement: bool, num_samples: usize, rng: &mut R) -> Self {
        let indices = match replacement {
            true => Uniform::new(0, num_samples)
                .sample_iter(rng)
                .take(size)
                .collect(),
            false => {
                let mut indices = Vec::with_capacity(size);
                loop {
                    let mut shuffled: Vec<_> = (0..num_samples).collect();
                    shuffled.shuffle(rng);

                    let remaining = size - indices.len();
                    match remaining <= num_samples {
                        true => {
                            shuffled.truncate(remaining);
                            indices.append(&mut shuffled);
                            break;
                        }
                        false => indices.append(&mut shuffled),
                    }
                }
                indices
            }
        };
        Self { indices }
    }
}

impl Iterator for RandomSamplerIter {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.indices.pop()
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.indices.len(), Some(self.indices.len()))
    }
}

impl ExactSizeIterator for RandomSamplerIter {}
impl FusedIterator for RandomSamplerIter {}

impl<R: Rng> Sampler for RandomSampler<R> {
    type Iter = RandomSamplerIter;

    #[inline]
    fn iter(&mut self) -> Self::Iter {
        RandomSamplerIter::new(self.size, self.replacement, self.num_samples, &mut self.rng)
    }

    #[inline]
    fn len(&self) -> usize {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    const SEED: u64 = 256;

    #[test]
    fn sequential() {
        let mut sampler = SequentialSampler::new(100);

        assert_eq!(100, sampler.len());
        assert_eq!(
            (0..100).collect::<Vec<_>>(),
            sampler.iter().collect::<Vec<_>>()
        );
    }

    fn count_elements(iter: impl IntoIterator<Item = usize> + Clone) -> Vec<usize> {
        let max = iter.clone().into_iter().max().unwrap();
        let min = iter.clone().into_iter().min().unwrap();
        assert_eq!(0, min);

        let mut count = (0..=max).map(|_| 0).collect::<Vec<_>>();
        iter.into_iter().for_each(|el| count[el] += 1);
        count
    }

    #[test]
    fn random_sampler_no_replacement_same_lenght_and_samples() {
        let mut sampler = RandomSampler::new(100, true, 100, SmallRng::seed_from_u64(SEED));

        assert_eq!(100, sampler.len());
        let iter: Vec<_> = sampler.iter().collect();
        assert_eq!(100, iter.len());

        // Checks that there is replacement
        let count = count_elements(iter.iter().cloned());
        assert_ne!(1, *count.iter().max().unwrap());
        assert_ne!(1, *count.iter().min().unwrap());
    }

    #[test]
    fn random_sampler_no_replacement_different_lenght_and_samples() {
        let mut sampler = RandomSampler::new(100, true, 10, SmallRng::seed_from_u64(SEED));

        assert_eq!(100, sampler.len());
        let iter: Vec<_> = sampler.iter().collect();
        assert_eq!(100, iter.len());

        // Checks that there is replacement
        let count = count_elements(iter.iter().cloned());
        assert_ne!(10, *count.iter().max().unwrap());
        assert_ne!(10, *count.iter().min().unwrap());

        // Checks indices are between 0 and num_samples
        assert_eq!(
            (0..10).collect::<HashSet<_>>(),
            iter.into_iter().collect::<HashSet<_>>()
        );
    }

    #[test]
    fn random_sampler_with_replacement_same_lenght_and_samples() {
        let mut sampler = RandomSampler::new(100, false, 100, SmallRng::seed_from_u64(SEED));

        assert_eq!(100, sampler.len());
        let iter: Vec<_> = sampler.iter().collect();
        assert_eq!(100, iter.len());

        // Checks that there is no replacement
        let count = count_elements(iter.iter().cloned());
        assert_eq!(1, *count.iter().max().unwrap());
        assert_eq!(1, *count.iter().min().unwrap());
    }

    #[test]
    fn random_sampler_with_replacement_different_lenght_and_samples() {
        let mut sampler = RandomSampler::new(100, false, 10, SmallRng::seed_from_u64(SEED));

        assert_eq!(100, sampler.len());
        let iter: Vec<_> = sampler.iter().collect();
        assert_eq!(100, iter.len());

        // Checks that there is no replacement
        let count = count_elements(iter.iter().cloned());
        assert_eq!(10, *count.iter().max().unwrap());
        assert_eq!(10, *count.iter().min().unwrap());

        // Checks indices are between 0 and num_samples
        assert_eq!(
            (0..10).collect::<HashSet<_>>(),
            iter.into_iter().collect::<HashSet<_>>()
        );
    }
}
