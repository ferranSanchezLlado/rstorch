use super::Dataset;
use crate::error::{DataError, Result};
use std::error;
use std::fmt;

#[derive(Debug, Clone)]
pub struct Transform<D, F> {
    dataset: D,
    transform: F,
}

impl<D, F> Transform<D, F> {
    pub fn new(dataset: D, transform: F) -> Self {
        Self { dataset, transform }
    }

    pub fn into_inner(self) -> D {
        self.dataset
    }
}

impl<D, F, B> Dataset for Transform<D, F>
where
    D: Dataset,
    F: Fn(D::Item) -> B,
{
    type Item = B;
    type Error = D::Error;

    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn get(&self, index: usize) -> std::result::Result<Self::Item, Self::Error> {
        self.dataset.get(index).map(&self.transform)
    }
}

#[derive(Debug, Clone)]
pub struct Subset<D> {
    dataset: D,
    indices: Vec<usize>,
}

impl<D> Subset<D>
where
    D: Dataset,
{
    pub fn new(dataset: D, indices: Vec<usize>) -> Result<Self> {
        let len = dataset.len();
        for &index in &indices {
            if index >= len {
                return Err(DataError::IndexOutOfBounds { index, len }.into());
            }
        }

        Ok(Self { dataset, indices })
    }

    pub fn into_inner(self) -> D {
        self.dataset
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }
}

impl<D> Dataset for Subset<D>
where
    D: Dataset,
{
    type Item = D::Item;
    type Error = SubsetError<D::Error>;

    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, index: usize) -> std::result::Result<Self::Item, Self::Error> {
        let mapped = self.indices.get(index).copied().ok_or(SubsetError::Data(
            DataError::IndexOutOfBounds {
                index,
                len: self.indices.len(),
            },
        ))?;
        self.dataset.get(mapped).map_err(SubsetError::Source)
    }
}

#[derive(Debug)]
pub enum SubsetError<E> {
    Data(DataError),
    Source(E),
}

impl<E> fmt::Display for SubsetError<E>
where
    E: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Data(err) => write!(f, "{err}"),
            Self::Source(err) => write!(f, "source dataset error: {err}"),
        }
    }
}

impl<E> error::Error for SubsetError<E>
where
    E: error::Error + Send + Sync + 'static,
{
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Data(err) => Some(err),
            Self::Source(err) => Some(err),
        }
    }
}

impl<E> From<SubsetError<E>> for crate::Error
where
    E: Into<crate::Error>,
{
    fn from(err: SubsetError<E>) -> Self {
        match err {
            SubsetError::Data(err) => err.into(),
            SubsetError::Source(err) => err.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Chain<A, B> {
    first: A,
    second: B,
}

impl<A, B> Chain<A, B> {
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }

    pub fn into_inner(self) -> (A, B) {
        (self.first, self.second)
    }
}

impl<A, B> Dataset for Chain<A, B>
where
    A: Dataset,
    B: Dataset<Item = A::Item>,
{
    type Item = A::Item;
    type Error = ChainError<A::Error, B::Error>;

    fn len(&self) -> usize {
        self.first.len().saturating_add(self.second.len())
    }

    fn get(&self, index: usize) -> std::result::Result<Self::Item, Self::Error> {
        let first_len = self.first.len();
        if index < first_len {
            return self.first.get(index).map_err(ChainError::First);
        }

        let second_index = index - first_len;
        if second_index >= self.second.len() {
            return Err(ChainError::Data(DataError::IndexOutOfBounds {
                index,
                len: self.len(),
            }));
        }

        self.second.get(second_index).map_err(ChainError::Second)
    }
}

#[derive(Debug)]
pub enum ChainError<A, B> {
    Data(DataError),
    First(A),
    Second(B),
}

impl<A, B> fmt::Display for ChainError<A, B>
where
    A: fmt::Display,
    B: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Data(err) => write!(f, "{err}"),
            Self::First(err) => write!(f, "first dataset error: {err}"),
            Self::Second(err) => write!(f, "second dataset error: {err}"),
        }
    }
}

impl<A, B> error::Error for ChainError<A, B>
where
    A: error::Error + Send + Sync + 'static,
    B: error::Error + Send + Sync + 'static,
{
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Data(err) => Some(err),
            Self::First(err) => Some(err),
            Self::Second(err) => Some(err),
        }
    }
}

impl<A, B> From<ChainError<A, B>> for crate::Error
where
    A: Into<crate::Error>,
    B: Into<crate::Error>,
{
    fn from(err: ChainError<A, B>) -> Self {
        match err {
            ChainError::Data(err) => err.into(),
            ChainError::First(err) => err.into(),
            ChainError::Second(err) => err.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Batch, DataLoader, SequentialSampler, StackVecCollate, VecDataset};
    use crate::error::Error;
    use crate::shape::{C, D2, Sym};
    use crate::tensor::Tensor;

    #[test]
    fn transform_maps_samples_and_preserves_len() {
        let dataset = VecDataset::new(vec![1, 2, 3]).transform(|value| value * 2);

        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.get(1).unwrap(), 4);
        assert!(matches!(
            dataset.get(3),
            Err(DataError::IndexOutOfBounds { index: 3, len: 3 })
        ));
    }

    #[test]
    fn subset_remaps_indices_and_validates_construction() {
        let dataset = VecDataset::new(vec![10, 20, 30, 40]);
        let subset = Subset::new(dataset, vec![2, 0]).unwrap();

        assert_eq!(subset.len(), 2);
        assert_eq!(subset.get(0).unwrap(), 30);
        assert_eq!(subset.get(1).unwrap(), 10);
        assert!(matches!(
            subset.get(2),
            Err(SubsetError::Data(DataError::IndexOutOfBounds {
                index: 2,
                len: 2
            }))
        ));

        let err = Subset::new(VecDataset::new(vec![1, 2]), vec![0, 2]).unwrap_err();
        assert!(matches!(
            err,
            Error::Data(DataError::IndexOutOfBounds { index: 2, len: 2 })
        ));
    }

    #[test]
    fn chain_concatenates_datasets() {
        let dataset = VecDataset::new(vec![1, 2]).chain(VecDataset::new(vec![3, 4, 5]));

        assert_eq!(dataset.len(), 5);
        assert_eq!(dataset.get(0).unwrap(), 1);
        assert_eq!(dataset.get(2).unwrap(), 3);
        assert_eq!(dataset.get(4).unwrap(), 5);
        assert!(matches!(
            dataset.get(5),
            Err(ChainError::Data(DataError::IndexOutOfBounds {
                index: 5,
                len: 5
            }))
        ));
    }

    #[test]
    fn adapters_drive_dataloader_end_to_end() {
        let dataset = VecDataset::new(vec![vec![1.0], vec![2.0], vec![3.0]])
            .chain(VecDataset::new(vec![vec![4.0]]))
            .subset(vec![3, 1, 0])
            .unwrap()
            .transform(|mut sample| {
                sample[0] *= 10.0;
                sample
            });
        let loader = DataLoader::new(
            dataset,
            SequentialSampler,
            StackVecCollate::<1>::new(),
            2,
            false,
        )
        .unwrap();

        let batches: Vec<Tensor<D2<Sym<Batch>, C<1>>>> =
            loader.iter().map(|batch| batch.unwrap()).collect();

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].to_vec().unwrap(), vec![40.0, 20.0]);
        assert_eq!(batches[1].to_vec().unwrap(), vec![10.0]);
    }
}
