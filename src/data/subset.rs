use std::slice::Iter;

use super::{Dataset, IterableDataset};

pub struct Subset<D> {
    dataset: D,
    indices: Vec<usize>,
}

impl<D: Dataset> Subset<D> {
    #[inline]
    #[must_use]
    pub fn new(dataset: D, indices: Vec<usize>) -> Self {
        if indices.iter().any(|i| i >= &dataset.len()) {
            panic!("One of the indices is outside bound");
        }
        Subset { dataset, indices }
    }
}

impl<D: Dataset> Dataset for Subset<D> {
    type Item = D::Item;

    fn get(&self, index: usize) -> Option<Self::Item> {
        self.dataset.get(*self.indices.get(index)?)
    }
    fn len(&self) -> usize {
        self.indices.len()
    }
    fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

pub struct SubsetIter<'a, D: 'a + Dataset> {
    dataset: &'a D,
    index: Iter<'a, usize>,
}

impl<'a, D: 'a + Dataset> Iterator for SubsetIter<'a, D> {
    type Item = D::Item;

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

#[cfg(test)]
mod test {
    use std::vec;

    use super::*;

    struct TestDataset {
        data: Vec<i32>,
    }

    impl Dataset for TestDataset {
        type Item = i32;

        fn get(&self, index: usize) -> Option<Self::Item> {
            self.data.get(index).copied()
        }
        fn len(&self) -> usize {
            self.data.len()
        }
    }

    #[test]
    fn test_subset() {
        let data = TestDataset {
            data: (-50..50).collect(),
        };
        let indices = vec![1, 3, 4, 10];

        let data = Subset::new(data, indices);
        assert_eq!(Some(-49), data.get(0));
        assert_eq!(vec![-49, -47, -46, -40], data.iter().collect::<Vec<_>>());
    }
}
