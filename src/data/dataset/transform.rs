use super::{Dataset, IterableDataset};
use std::iter::Map;

pub struct Transform<D, F> {
    data: D,
    f: F,
}

impl<D, F> Transform<D, F> {
    #[inline]
    #[must_use]
    pub(in crate::data) fn new(data: D, f: F) -> Self {
        Self { data, f }
    }
}

impl<D: Dataset, B, F> Dataset for Transform<D, F>
where
    F: FnMut(D::Item) -> B + Copy,
{
    type Item = B;

    #[inline]
    fn get(&self, index: usize) -> Option<Self::Item> {
        self.data.get(index).map(self.f)
    }

    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<'a, D: IterableDataset<'a>, B, F> IterableDataset<'a> for Transform<D, F>
where
    F: FnMut(D::Item) -> B + Copy,
{
    type Iterator = Map<D::Iterator, F>;

    #[inline]
    fn iter(&'a self) -> Self::Iterator {
        self.data.iter().map(self.f)
    }
}
