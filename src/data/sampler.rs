use rand::prelude::*;
use std::ops::Range;

pub trait Sampler {
    type Iter: Iterator<Item = usize>;

    fn iter(&self) -> Self::Iter;
    fn len(&self) -> usize;
}

pub struct SequentialSampler {
    size: usize,
}

impl SequentialSampler {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl Sampler for SequentialSampler {
    type Iter = Range<usize>;

    fn iter(&self) -> Self::Iter {
        0..self.size
    }
    fn len(&self) -> usize {
        self.size
    }
}

pub struct RandomSampler {
    size: usize,
    num_samples: usize,
    replacement: bool,
}

impl RandomSampler {
    pub fn new(size: usize, replacement: bool, num_samples: usize) -> Self {
        Self {
            size,
            num_samples,
            replacement,
        }
    }
}

pub struct RandomSamplerIter {
    indices: Vec<usize>,
}
impl RandomSamplerIter {
    fn new(size: usize, replacement: bool, num_samples: usize) -> Self {
        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..size).collect();
        indices.shuffle(&mut rng);

        indices = match replacement {
            true => (0..num_samples)
                .map(|_| indices[rng.gen_range(0..size)])
                .collect(),
            false => {
                indices.truncate(num_samples);
                indices
            }
        };
        Self { indices }
    }
}

impl Iterator for RandomSamplerIter {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.indices.pop()
    }
}

impl ExactSizeIterator for RandomSamplerIter {
    fn len(&self) -> usize {
        self.indices.len()
    }
}

impl Sampler for RandomSampler {
    type Iter = RandomSamplerIter;

    fn iter(&self) -> Self::Iter {
        RandomSamplerIter::new(self.size, self.replacement, self.num_samples)
    }
    fn len(&self) -> usize {
        self.size
    }
}
