use super::{Dataset, IterableDataset};
use std::{iter::Cloned, slice::Iter};

pub struct Basic<T> {
    pub data: Vec<T>,
}

impl<T> Basic<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }
}

impl<T> FromIterator<T> for Basic<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::new(Vec::from_iter(iter))
    }
}

impl<T: Clone> Dataset for Basic<T> {
    type Item = T;

    fn get(&self, index: usize) -> Option<Self::Item> {
        self.data.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<'a, T: 'a + Clone> IterableDataset<'a> for Basic<T> {
    type Iterator = Cloned<Iter<'a, T>>;

    fn iter(&'a self) -> Self::Iterator {
        self.data.iter().cloned()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_from() {
        let data = vec![1, 2, 3, 4, 5];
        let dataset = Basic::new(data.clone());

        assert_eq!(Some(1), dataset.get(0));
        assert_eq!(data, dataset.iter().collect::<Vec<_>>());
    }

    #[test]
    fn test_collect() {
        let dataset: Basic<_> = (-50..50).collect();

        assert_eq!(Some(-50), dataset.get(0));
        let expected = (-50..50).collect::<Vec<_>>();
        assert_eq!(expected, dataset.iter().collect::<Vec<_>>());
    }

    #[test]
    fn test_from_iter() {
        let dataset = Basic::from_iter(-50..50);

        assert_eq!(Some(-50), dataset.get(0));
        let expected = (-50..50).collect::<Vec<_>>();
        assert_eq!(expected, dataset.iter().collect::<Vec<_>>());
    }
}
