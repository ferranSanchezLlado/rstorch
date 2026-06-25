use super::Dataset;
use crate::error::DataError;

#[derive(Debug, Clone)]
pub struct VecDataset<T> {
    items: Vec<T>,
}

impl<T> VecDataset<T> {
    pub fn new(items: Vec<T>) -> Self {
        Self { items }
    }

    pub fn into_inner(self) -> Vec<T> {
        self.items
    }
}

impl<T> Dataset for VecDataset<T>
where
    T: Clone,
{
    type Item = T;
    type Error = DataError;

    fn len(&self) -> usize {
        self.items.len()
    }

    fn get(&self, index: usize) -> std::result::Result<Self::Item, Self::Error> {
        self.items
            .get(index)
            .cloned()
            .ok_or(DataError::IndexOutOfBounds {
                index,
                len: self.items.len(),
            })
    }
}

impl Dataset for std::ops::Range<usize> {
    type Item = usize;
    type Error = DataError;

    fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    fn get(&self, index: usize) -> std::result::Result<Self::Item, Self::Error> {
        let len = Dataset::len(self);
        if index >= len {
            return Err(DataError::IndexOutOfBounds { index, len });
        }

        Ok(self.start + index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_dataset_len_and_get() {
        let dataset = VecDataset::new(vec![10, 20, 30]);

        assert_eq!(dataset.len(), 3);
        assert!(!dataset.is_empty());
        assert_eq!(dataset.get(1).unwrap(), 20);
    }

    #[test]
    fn vec_dataset_out_of_bounds_is_structured() {
        let dataset = VecDataset::new(vec![1, 2]);

        assert!(matches!(
            dataset.get(2),
            Err(DataError::IndexOutOfBounds { index: 2, len: 2 })
        ));
    }
}
