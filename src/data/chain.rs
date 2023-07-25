use super::{Dataset, IterableDataset};

pub struct Chain<A, B> {
    a: A,
    b: B,
}

impl<A, B> Chain<A, B> {
    pub(in crate::data) fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A, B> Dataset for Chain<A, B>
where
    A: Dataset,
    B: Dataset<Item = A::Item>,
{
    type Item = A::Item;

    fn get(&self, index: usize) -> Option<Self::Item> {
        match index < self.a.len() {
            true => self.a.get(index),
            false => self.b.get(index - self.a.len()),
        }
    }

    fn len(&self) -> usize {
        self.a.len() + self.b.len()
    }

    fn is_empty(&self) -> bool {
        self.a.is_empty() && self.b.is_empty()
    }
}

impl<'a, A, B> IterableDataset<'a> for Chain<A, B>
where
    A: IterableDataset<'a>,
    B: IterableDataset<'a, Item = A::Item>,
{
    type Iterator = std::iter::Chain<A::Iterator, B::Iterator>;

    fn iter(&'a self) -> Self::Iterator {
        self.a.iter().chain(self.b.iter())
    }
}
