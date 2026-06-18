use super::Dataset;

/// Dataset view over selected indices.
#[derive(Debug, Clone)]
pub struct Subset<D> {
    dataset: D,
    indices: Vec<usize>,
}

impl<D> Subset<D>
where
    D: Dataset,
{
    pub fn new(dataset: D, indices: Vec<usize>) -> Self {
        assert!(
            indices.iter().all(|&index| index < dataset.len()),
            "subset index out of bounds"
        );
        Self { dataset, indices }
    }
}

impl<D> Dataset for Subset<D>
where
    D: Dataset,
{
    type Item = D::Item;

    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        self.dataset.get(*self.indices.get(index)?)
    }
}
