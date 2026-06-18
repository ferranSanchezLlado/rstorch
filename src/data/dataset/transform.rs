use super::Dataset;

/// Dataset view that transforms samples on access.
#[derive(Debug, Clone)]
pub struct Transform<D, F> {
    dataset: D,
    transform: F,
}

impl<D, F> Transform<D, F> {
    pub fn new(dataset: D, transform: F) -> Self {
        Self { dataset, transform }
    }
}

impl<D, F, Output> Dataset for Transform<D, F>
where
    D: Dataset,
    F: Fn(D::Item) -> Output,
{
    type Item = Output;

    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        self.dataset.get(index).map(&self.transform)
    }
}
