use std::iter::FusedIterator;

use super::dataset::Shuffler;
use super::{dataset::Dataset, sampler::Sampler};
use ndarray::prelude::*;

#[derive(Debug)]
enum Shuffle<D> {
    Yes(Shuffler<D>),
    No(D),
}

impl<D: Dataset> Dataset for Shuffle<D> {
    type Item = D::Item;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        match self {
            Shuffle::Yes(data) => data.get(index),
            Shuffle::No(data) => data.get(index),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        match self {
            Shuffle::Yes(data) => data.len(),
            Shuffle::No(data) => data.len(),
        }
    }
}

pub struct DataLoader<D, S> {
    dataset: Shuffle<D>,
    batch_size: usize,
    sampler: S,
}

impl<D: Dataset, S: Sampler> DataLoader<D, S> {
    #[inline]
    #[must_use]
    pub fn new(dataset: D, batch_size: usize, shuffle: bool, sampler: S) -> Self {
        let dataset = match shuffle {
            true => Shuffle::Yes(Shuffler::new(dataset)),
            false => Shuffle::No(dataset),
        };

        Self {
            dataset,
            batch_size,
            sampler,
        }
    }

    #[inline]
    pub fn shuffle(&mut self) {
        if let Shuffle::Yes(data) = &mut self.dataset {
            data.shuffle();
        }
    }

    #[inline]
    pub fn iter(&mut self) -> DataLoaderIter<'_, D, S> {
        self.shuffle();
        DataLoaderIter::new(self)
    }
}

// -- GENERIC ITER --
pub struct DataLoaderIter<'a, D, S: Sampler> {
    data_loader: &'a DataLoader<D, S>,
    iter: S::Iter,
}

impl<'a, D, S: Sampler> DataLoaderIter<'a, D, S> {
    #[inline]
    #[must_use]
    pub(crate) fn new(data_loader: &'a DataLoader<D, S>) -> Self {
        let iter = data_loader.sampler.iter();
        Self { data_loader, iter }
    }
}

impl<'a, D: Dataset, S: Sampler> Iterator for DataLoaderIter<'a, D, S> {
    type Item = Vec<D::Item>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        (0..self.data_loader.batch_size)
            .map(|_| self.data_loader.dataset.get(self.iter.next().unwrap()))
            .collect()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();

        (
            lower / self.data_loader.batch_size,
            upper.map(|n| n / self.data_loader.batch_size),
        )
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.count() / self.data_loader.batch_size
    }
}

impl<'a, D, S> DoubleEndedIterator for DataLoaderIter<'a, D, S>
where
    D: Dataset,
    S: Sampler,
    S::Iter: DoubleEndedIterator,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        (0..self.data_loader.batch_size)
            .map(|_| self.data_loader.dataset.get(self.iter.next_back().unwrap()))
            .collect()
    }
}

impl<'a, D, S> ExactSizeIterator for DataLoaderIter<'a, D, S>
where
    D: Dataset,
    S: Sampler,
    S::Iter: ExactSizeIterator,
{
}
impl<'a, D, S> FusedIterator for DataLoaderIter<'a, D, S>
where
    D: Dataset,
    S: Sampler,
    S::Iter: FusedIterator,
{
}

// -- ARRAY ITER --
pub struct ArrayDataLoaderIter<'a, D, S: Sampler> {
    data_loader: &'a DataLoader<D, S>,
    iter: S::Iter,
}

impl<'a, D, S: Sampler> ArrayDataLoaderIter<'a, D, S> {
    #[inline]
    pub(crate) fn new(data_loader: &'a DataLoader<D, S>) -> Self {
        let iter = data_loader.sampler.iter();
        Self { data_loader, iter }
    }
}

impl<'a, D, T1, D1, T2, D2, S> Iterator for ArrayDataLoaderIter<'a, D, S>
where
    D: Dataset<Item = (Array<T1, D1>, Array<T2, D2>)>,
    S: Sampler,
    D1: Dimension,
    D2: Dimension,
    T1: Clone,
    T2: Clone,
{
    type Item = (Array<T1, D1::Larger>, Array<T2, D2::Larger>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (sample, target) = self.data_loader.dataset.get(self.iter.next()?)?;
        let mut samples = sample.insert_axis(Axis(0));
        let mut targets = target.insert_axis(Axis(0));
        for _ in 1..self.data_loader.batch_size {
            let (sample, target) = self.data_loader.dataset.get(self.iter.next()?)?;

            samples
                .append(Axis(0), sample.insert_axis(Axis(0)).view())
                .ok()?;
            targets
                .append(Axis(0), target.insert_axis(Axis(0)).view())
                .ok()?;
        }
        Some((samples, targets))
    }
}

impl<'a, D, T1, D1, T2, D2, S> DoubleEndedIterator for ArrayDataLoaderIter<'a, D, S>
where
    D: Dataset<Item = (Array<T1, D1>, Array<T2, D2>)>,
    S: Sampler,
    S::Iter: DoubleEndedIterator,
    D1: Dimension,
    D2: Dimension,
    T1: Clone,
    T2: Clone,
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let (sample, target) = self.data_loader.dataset.get(self.iter.next_back()?)?;
        let mut samples = sample.insert_axis(Axis(0));
        let mut targets = target.insert_axis(Axis(0));
        for _ in 1..self.data_loader.batch_size {
            let (sample, target) = self.data_loader.dataset.get(self.iter.next_back()?)?;

            samples
                .append(Axis(0), sample.insert_axis(Axis(0)).view())
                .ok()?;
            targets
                .append(Axis(0), target.insert_axis(Axis(0)).view())
                .ok()?;
        }
        Some((samples, targets))
    }
}

impl<'a, D, T1, D1, T2, D2, S> ExactSizeIterator for ArrayDataLoaderIter<'a, D, S>
where
    D: Dataset<Item = (Array<T1, D1>, Array<T2, D2>)>,
    S: Sampler,
    S::Iter: ExactSizeIterator,
    D1: Dimension,
    D2: Dimension,
    T1: Clone,
    T2: Clone,
{
}
impl<'a, D, T1, D1, T2, D2, S> FusedIterator for ArrayDataLoaderIter<'a, D, S>
where
    D: Dataset<Item = (Array<T1, D1>, Array<T2, D2>)>,
    S: Sampler,
    S::Iter: FusedIterator,
    D1: Dimension,
    D2: Dimension,
    T1: Clone,
    T2: Clone,
{
}

impl<D, T1, D1, T2, D2, S> DataLoader<D, S>
where
    D: Dataset<Item = (Array<T1, D1>, Array<T2, D2>)>,
    S: Sampler,
    D1: Dimension,
    D2: Dimension,
    T1: Clone,
    T2: Clone,
{
    #[inline]
    pub fn iter_array(&mut self) -> ArrayDataLoaderIter<'_, D, S> {
        self.shuffle();
        ArrayDataLoaderIter::new(self)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::IntoDimension;

    use crate::data::dataset::test::TestDataset;
    use crate::data::SequentialSampler;

    use super::*;

    struct ArrayTestDataset {
        samples: Vec<Array2<f64>>,
        targets: Vec<Array1<f64>>,
    }

    impl Dataset for ArrayTestDataset {
        type Item = (Array2<f64>, Array1<f64>);

        fn get(&self, index: usize) -> Option<Self::Item> {
            Some((
                self.samples.get(index)?.clone(),
                self.targets.get(index)?.clone(),
            ))
        }

        fn len(&self) -> usize {
            self.samples.len()
        }
    }

    #[test]
    fn basic_iter() {
        let data = TestDataset::new(5..500);
        let sampler = SequentialSampler::new(data.len());

        let mut data = DataLoader::new(data, 32, false, sampler);
        assert_eq!(Some((5..37).collect::<Vec<_>>()), data.iter().next());
    }

    fn fill_dataset<const N_DIM: usize>(n: usize) -> Vec<Array<f64, Dim<[usize; N_DIM]>>>
    where
        [usize; N_DIM]: IntoDimension<Dim = Dim<[usize; N_DIM]>>,
        Dim<[usize; N_DIM]>: Dimension,
    {
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            let mut a = Array::zeros([4; N_DIM]);
            a.fill(i as f64);
            data.push(a);
        }
        data
    }

    #[test]
    fn iter_array() {
        let samples = fill_dataset(100);
        let targets = fill_dataset(100);
        let data = ArrayTestDataset { samples, targets };
        let sampler = SequentialSampler::new(data.len());

        let mut data = DataLoader::new(data, 32, false, sampler);

        let (sample, label) = data.iter_array().next().unwrap();
        assert!(fill_dataset(32).into_iter().eq(sample.outer_iter()));
        assert!(fill_dataset(32).into_iter().eq(label.outer_iter()));
    }
}
