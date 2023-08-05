use super::{Dataset, IterableDataset};
use std::{iter::FusedIterator, slice::Iter};

pub struct Subset<D> {
    data: D,
    indices: Vec<usize>,
}

impl<D: Dataset> Subset<D> {
    #[inline]
    #[must_use]
    pub(in crate::data) fn new(data: D, indices: Vec<usize>) -> Self {
        if indices.iter().any(|i| i >= &data.len()) {
            panic!("One of the indices is outside bound");
        }
        Subset { data, indices }
    }
}

impl<D: Dataset> Dataset for Subset<D> {
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

pub struct SubsetIter<'a, D: 'a> {
    data: &'a D,
    index: Iter<'a, usize>,
}

impl<'a, D: 'a + Dataset> Iterator for SubsetIter<'a, D> {
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

impl<'a, D: 'a + Dataset> DoubleEndedIterator for SubsetIter<'a, D> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.index.next_back().map(|i| self.data.get(*i))?
    }
}

impl<D: Dataset> ExactSizeIterator for SubsetIter<'_, D> {}

impl<D: Dataset> FusedIterator for SubsetIter<'_, D> {}

impl<'a, D: 'a + Dataset> IterableDataset<'a> for Subset<D> {
    type Iterator = SubsetIter<'a, D> where Self::Item: 'a;

    #[inline]
    fn iter(&'a self) -> Self::Iterator {
        SubsetIter {
            data: &self.data,
            index: self.indices.iter(),
        }
    }
}
