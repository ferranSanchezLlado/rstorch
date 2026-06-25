mod adapters;
mod tensor;
mod vec;

use crate::error::Result;

pub use adapters::{Chain, ChainError, Subset, SubsetError, Transform};
pub use tensor::{IntoTensorDataset, TensorDataset};
pub use vec::VecDataset;

pub trait Dataset {
    type Item;
    type Error: std::error::Error + Send + Sync + 'static;

    fn len(&self) -> usize;
    fn get(&self, index: usize) -> std::result::Result<Self::Item, Self::Error>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn transform<F, B>(self, transform: F) -> Transform<Self, F>
    where
        Self: Sized,
        F: Fn(Self::Item) -> B,
    {
        Transform::new(self, transform)
    }

    fn subset(self, indices: Vec<usize>) -> Result<Subset<Self>>
    where
        Self: Sized,
    {
        Subset::new(self, indices)
    }

    fn chain<D>(self, other: D) -> Chain<Self, D>
    where
        Self: Sized,
        D: Dataset<Item = Self::Item>,
    {
        Chain::new(self, other)
    }
}
