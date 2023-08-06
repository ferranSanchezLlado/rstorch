use rand::distributions::{DistIter, Uniform};
use rand::prelude::*;
use std::iter::{FusedIterator, Take};
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

pub struct NoReplacement<R> {
    data: Vec<usize>,
    index: usize,
    size: usize, // Number elements to generate
    rng: R,
}

impl<R: Rng> NoReplacement<R> {
    #[inline]
    #[must_use]
    fn new(size: usize, num_samples: usize, rng: R) -> Self {
        Self {
            data: (0..num_samples).collect(),
            index: 0,
            size,
            rng,
        }
    }
}

impl<R: Rng> Iterator for NoReplacement<R> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.size {
            return None;
        }

        let index = self.index % self.data.len();
        // Iterated over all elements or is start
        if index == 0 {
            self.data.shuffle(&mut self.rng);
        }

        let el = self.data.get(index);
        self.index += 1;
        el.copied()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size, Some(self.size))
    }

    #[inline]
    fn count(self) -> usize {
        self.size
    }
}

impl<R: Rng> ExactSizeIterator for NoReplacement<R> {}
impl<R: Rng> FusedIterator for NoReplacement<R> {}

pub enum RandomSamplerIter<R> {
    Replacement(Take<DistIter<Uniform<usize>, R, usize>>),
    NoReplacement(NoReplacement<R>),
}

impl<R: Rng> RandomSamplerIter<R> {
    #[inline]
    #[must_use]
    fn new(size: usize, replacement: bool, num_samples: usize, rng: R) -> Self {
        match replacement {
            true => RandomSamplerIter::Replacement(
                Uniform::new(0, num_samples).sample_iter(rng).take(size),
            ),
            false => RandomSamplerIter::NoReplacement(NoReplacement::new(size, num_samples, rng)),
        }
    }

    #[inline]
    fn move_rng(size: usize, replacement: bool, num_samples: usize, mut rng: R) {
        // TODO: Improve rng movement
        match replacement {
            true => (0..size).for_each(|_| {
                rng.gen::<usize>();
            }),
            false => (0..(size / num_samples + 1)).for_each(|_| {
                rng.gen_range(0..num_samples);
            }),
        }
    }
}

impl<R: Rng> Iterator for RandomSamplerIter<R> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            RandomSamplerIter::Replacement(iter) => iter.next(),
            RandomSamplerIter::NoReplacement(iter) => iter.next(),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            RandomSamplerIter::Replacement(iter) => iter.size_hint(),
            RandomSamplerIter::NoReplacement(iter) => iter.size_hint(),
        }
    }

    #[inline]
    fn count(self) -> usize {
        match self {
            RandomSamplerIter::Replacement(iter) => iter.count(),
            RandomSamplerIter::NoReplacement(iter) => iter.count(),
        }
    }
}

impl<R: Rng> ExactSizeIterator for RandomSamplerIter<R> {}
impl<R: Rng> FusedIterator for RandomSamplerIter<R> {}

impl<R: Rng + Clone> Sampler for RandomSampler<R> {
    type Iter = RandomSamplerIter<R>;

    #[inline]
    fn iter(&mut self) -> Self::Iter {
        // Clones rng to avoid the unecessery need to link lifetimes between iterator and random
        // sampler. However, becouse of this the rng doesn't move. Therefore, I opted to move
        // proactively the rng.
        let rng = self.rng.clone();
        RandomSamplerIter::move_rng(self.size, self.replacement, self.num_samples, &mut self.rng);
        RandomSamplerIter::new(self.size, self.replacement, self.num_samples, rng)
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
