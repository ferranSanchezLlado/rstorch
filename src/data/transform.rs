use super::{Dataset, IterableDataset};
use std::{iter::Map, marker::PhantomData};

pub struct Transform<D, F, T> {
    dataset: D,
    function: F,
    new_type: PhantomData<T>,
}

impl<D, T, T2, F> Transform<D, F, T2>
where
    D: Dataset<Item = T>,
    F: FnMut(T) -> T2 + Copy,
{
    pub fn new(dataset: D, function: F) -> Self {
        Self {
            dataset,
            function,
            new_type: PhantomData,
        }
    }
}

impl<D, T, T2, F> Dataset for Transform<D, F, T2>
where
    D: Dataset<Item = T>,
    F: FnMut(T) -> T2 + Copy,
{
    type Item = T2;

    fn get(&self, index: usize) -> Option<Self::Item> {
        self.dataset.get(index).map(self.function)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }

    fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }
}

impl<'a, D, T, T2, F> IterableDataset<'a> for Transform<D, F, T2>
where
    D: IterableDataset<'a, Item = T>,
    F: FnMut(T) -> T2 + Copy,
{
    type Iterator = Map<D::Iterator, F>;

    fn iter(&'a self) -> Self::Iterator {
        self.dataset.iter().map(self.function)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::{iter::Copied, slice::Iter};

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

    impl<'a> IterableDataset<'a> for TestDataset {
        type Iterator = Copied<Iter<'a, i32>>;

        fn iter(&'a self) -> Self::Iterator {
            self.data.iter().copied()
        }
    }

    fn transform(input: i32) -> u32 {
        input.abs() as u32
    }

    #[test]
    fn test_transform() {
        let data = TestDataset {
            data: (-50..50).collect(),
        };

        let data = Transform::new(data, transform);
        assert_eq!(Some(50), data.get(0));
        let expected = (1..=50).rev().chain(0..50).collect::<Vec<u32>>();
        assert_eq!(expected, data.iter().collect::<Vec<_>>());
    }
}
