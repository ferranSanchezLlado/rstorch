use crate::random::SmallRng;

/// Produces dataset indices for a loader.
///
/// The returned iterator must be an honest [`ExactSizeIterator`]: `len()` must
/// equal the number of indices it will actually yield. `DataLoader` relies on
/// this contract when grouping indices into batches.
pub trait Sampler {
    type Iter: ExactSizeIterator<Item = usize>;

    fn indices(&self, len: usize, epoch: u64) -> Self::Iter;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SequentialSampler;

impl Sampler for SequentialSampler {
    type Iter = std::ops::Range<usize>;

    fn indices(&self, len: usize, _epoch: u64) -> Self::Iter {
        0..len
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ShuffleSampler {
    seed: u64,
}

impl ShuffleSampler {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }
}

impl Sampler for ShuffleSampler {
    type Iter = std::vec::IntoIter<usize>;

    fn indices(&self, len: usize, epoch: u64) -> Self::Iter {
        let mut indices: Vec<_> = (0..len).collect();
        let mut rng = SmallRng::seed_from_u64(epoch_seed(self.seed, epoch));
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(i + 1);
            indices.swap(i, j);
        }
        indices.into_iter()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RandomSampler {
    seed: u64,
    replacement: bool,
    num_samples: Option<usize>,
}

impl RandomSampler {
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            replacement: false,
            num_samples: None,
        }
    }

    pub fn without_replacement(seed: u64, num_samples: usize) -> Self {
        Self {
            seed,
            replacement: false,
            num_samples: Some(num_samples),
        }
    }

    pub fn with_replacement(seed: u64, num_samples: usize) -> Self {
        Self {
            seed,
            replacement: true,
            num_samples: Some(num_samples),
        }
    }

    pub fn replacement(mut self, replacement: bool) -> Self {
        self.replacement = replacement;
        self
    }

    pub fn num_samples(mut self, num_samples: usize) -> Self {
        self.num_samples = Some(num_samples);
        self
    }
}

impl Sampler for RandomSampler {
    type Iter = std::vec::IntoIter<usize>;

    fn indices(&self, len: usize, epoch: u64) -> Self::Iter {
        let requested = self.num_samples.unwrap_or(len);
        if len == 0 || requested == 0 {
            return Vec::new().into_iter();
        }

        let mut rng = SmallRng::seed_from_u64(epoch_seed(self.seed, epoch));
        let indices = if self.replacement {
            (0..requested).map(|_| rng.gen_range(len)).collect()
        } else {
            let mut indices: Vec<_> = (0..len).collect();
            for i in (1..indices.len()).rev() {
                let j = rng.gen_range(i + 1);
                indices.swap(i, j);
            }
            indices.truncate(requested.min(len));
            indices
        };

        indices.into_iter()
    }
}

fn epoch_seed(seed: u64, epoch: u64) -> u64 {
    seed ^ epoch.wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_sampler_is_ordered() {
        assert_eq!(
            SequentialSampler.indices(4, 0).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            SequentialSampler.indices(4, 99).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn shuffle_sampler_is_deterministic_for_seed_and_epoch() {
        let first = ShuffleSampler::new(42).indices(8, 3).collect::<Vec<_>>();
        let second = ShuffleSampler::new(42).indices(8, 3).collect::<Vec<_>>();

        assert_eq!(first, second);
        assert_ne!(first, SequentialSampler.indices(8, 3).collect::<Vec<_>>());
        assert_ne!(
            first,
            ShuffleSampler::new(42).indices(8, 4).collect::<Vec<_>>()
        );
    }

    #[test]
    fn random_sampler_without_replacement_yields_exact_permutation() {
        let sampler = RandomSampler::without_replacement(42, 5);
        let mut indices = sampler.indices(8, 3);

        assert_eq!(indices.len(), 5);
        let mut values = indices.by_ref().collect::<Vec<_>>();
        assert_eq!(indices.len(), 0);
        assert_eq!(values.len(), 5);
        values.sort_unstable();
        values.dedup();
        assert_eq!(values.len(), 5);
        assert!(values.iter().all(|&index| index < 8));
    }

    #[test]
    fn random_sampler_with_replacement_is_deterministic_and_exact() {
        let sampler = RandomSampler::with_replacement(7, 12);
        let first = sampler.indices(3, 2).collect::<Vec<_>>();
        let second = sampler.indices(3, 2).collect::<Vec<_>>();

        assert_eq!(first, second);
        assert_eq!(first.len(), 12);
        assert!(first.iter().all(|&index| index < 3));
        assert_ne!(first, sampler.indices(3, 3).collect::<Vec<_>>());

        let repeated = RandomSampler::with_replacement(7, 4)
            .indices(1, 2)
            .collect::<Vec<_>>();
        assert_eq!(repeated, vec![0, 0, 0, 0]);
    }
}
