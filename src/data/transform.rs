use super::{Dataset, IterableDataset};
use std::iter::Map;

pub struct Transform<D, F> {
    data: D,
    f: F,
}

impl<D, F> Transform<D, F> {
    pub(in crate::data) fn new(data: D, f: F) -> Self {
        Self { data, f }
    }
}

impl<D: Dataset, B, F> Dataset for Transform<D, F>
where
    F: FnMut(D::Item) -> B + Copy,
{
    type Item = B;

    fn get(&self, index: usize) -> Option<Self::Item> {
        self.data.get(index).map(self.f)
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<'a, D: IterableDataset<'a>, B, F> IterableDataset<'a> for Transform<D, F>
where
    F: FnMut(D::Item) -> B + Copy,
{
    type Iterator = Map<D::Iterator, F>;

    fn iter(&'a self) -> Self::Iterator {
        self.data.iter().map(self.f)
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
