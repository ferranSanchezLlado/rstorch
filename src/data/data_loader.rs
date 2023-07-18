use super::{sampler::Sampler, Dataset};

pub struct DataLoader<D, S> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    sampler: S,
}

impl<D: Dataset, S: Sampler> DataLoader<D, S> {
    pub fn new(dataset: D, batch_size: usize, shuffle: bool, sampler: S) -> Self {
        Self {
            dataset,
            batch_size,
            shuffle,
            sampler,
        }
    }

    pub fn iter(&self) {
        todo!()
    }
}
