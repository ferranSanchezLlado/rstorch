use std::slice::Iter;

use super::{Dataset, IterableDataset};

pub struct Subset<D> {
    dataset: D,
    indices: Vec<usize>,
}

impl<D: Dataset> Subset<D> {
    pub fn new(dataset: D, indices: Vec<usize>) -> Self {
        if indices.iter().any(|i| i >= &dataset.len()) {
            panic!("One of the indices is outside bound");
        }
        Subset { dataset, indices }
    }
}

impl<D: Dataset> Dataset for Subset<D> {
    type Item = D::Item;

    fn get(&self, index: usize) -> Option<&Self::Item> {
        self.dataset.get(*self.indices.get(index)?)
    }
    fn len(&self) -> usize {
        self.indices.len()
    }
}

pub struct SubsetIter<'a, D: 'a + Dataset> {
    dataset: &'a D,
    index: Iter<'a, usize>,
}

impl<'a, D: 'a + Dataset> Iterator for SubsetIter<'a, D> {
    type Item = &'a D::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.index.next().map(|i| self.dataset.get(*i))?
    }
}

impl<'a, D: 'a + Dataset> IterableDataset<'a> for Subset<D> {
    type Iterator = SubsetIter<'a, D> where Self::Item: 'a;

    fn iter(&'a self) -> Self::Iterator {
        SubsetIter {
            dataset: &self.dataset,
            index: self.indices.iter(),
        }
    }
}
