use super::{Dataset, IterableDataset};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::{iter::FusedIterator, slice::Iter};

#[derive(Debug)]
pub struct Shuffler<D> {
    data: D,
    indices: Vec<usize>,
}

impl<D: Dataset> Shuffler<D> {
    #[inline]
    #[must_use]
    pub(crate) fn new(data: D) -> Self {
        let mut indices: Vec<_> = (0..data.len()).collect();
        indices.shuffle(&mut thread_rng());

        Shuffler { data, indices }
    }

    #[inline]
    pub(crate) fn shuffle(&mut self) {
        self.indices.shuffle(&mut thread_rng());
    }
}

impl<D: Dataset> Dataset for Shuffler<D> {
    type Item = D::Item;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        self.data.get(*self.indices.get(index)?)
    }

    #[inline]
    fn len(&self) -> usize {
        self.indices.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

pub struct ShufflerIter<'a, D: 'a> {
    data: &'a D,
    index: Iter<'a, usize>,
}

impl<'a, D: 'a + Dataset> Iterator for ShufflerIter<'a, D> {
    type Item = D::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.index.next().map(|i| self.data.get(*i))?
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.index.size_hint()
    }
}

impl<'a, D: 'a + Dataset> DoubleEndedIterator for ShufflerIter<'a, D> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.index.next_back().map(|i| self.data.get(*i))?
    }
}

impl<D: Dataset> ExactSizeIterator for ShufflerIter<'_, D> {}

impl<D: Dataset> FusedIterator for ShufflerIter<'_, D> {}

impl<'a, D: 'a + Dataset> IterableDataset<'a> for Shuffler<D> {
    type Iterator = ShufflerIter<'a, D> where Self::Item: 'a;

    #[inline]
    fn iter(&'a self) -> Self::Iterator {
        ShufflerIter {
            data: &self.data,
            index: self.indices.iter(),
        }
    }
}
