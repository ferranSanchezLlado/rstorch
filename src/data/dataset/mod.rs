mod basic;
mod chain;
#[cfg(feature = "dataset_hub")]
pub mod hub;
mod shuffler;
mod subset;
mod transform;

pub use basic::Basic;
pub use chain::Chain;
pub use shuffler::Shuffler;
pub use subset::Subset;
pub use transform::Transform;

pub trait Dataset {
    type Item;

    fn get(&self, index: usize) -> Option<Self::Item>;
    fn len(&self) -> usize;

    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn chain<D>(self, other: D) -> Chain<Self, D>
    where
        Self: Sized,
        D: Dataset<Item = Self::Item>,
    {
        Chain::new(self, other)
    }

    fn subset(self, indices: Vec<usize>) -> Subset<Self>
    where
        Self: Sized,
    {
        Subset::new(self, indices)
    }

    fn transform<B, F>(self, f: F) -> Transform<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Item) -> B,
    {
        Transform::new(self, f)
    }
}

pub trait IterableDataset<'a>: Dataset {
    type Iterator: Iterator<Item = Self::Item>;

    fn iter(&'a self) -> Self::Iterator;
}

#[cfg(test)]
pub mod test {
    use super::*;
    use std::{iter::Copied, slice::Iter};

    pub struct TestDataset {
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

    impl TestDataset {
        pub fn new(data: impl Iterator<Item = i32>) -> Self {
            Self {
                data: data.collect(),
            }
        }
    }

    #[test]
    fn test_dataset() {
        let data = TestDataset::new(-50..50);
        assert_eq!(Some(-10), data.get(40));
    }

    #[test]
    fn test_iterable_dataset() {
        let data = TestDataset::new(-50..50);
        let expected = (-50..50).collect::<Vec<_>>();
        assert_eq!(expected, data.iter().collect::<Vec<_>>());
    }

    fn abs(input: i32) -> u32 {
        input.unsigned_abs()
    }

    #[test]
    fn test_transform() {
        let data = TestDataset::new(-50..50);

        let data = data.transform(abs);
        assert_eq!(Some(50), data.get(0));
        let expected = (1..=50).rev().chain(0..50).collect::<Vec<u32>>();
        assert_eq!(expected, data.iter().collect::<Vec<_>>());
    }

    #[test]
    fn test_subset() {
        let data = TestDataset::new(-50..50);
        let indices = vec![1, 3, 4, 10];

        let data = data.subset(indices);
        assert_eq!(Some(-49), data.get(0));
        assert_eq!(vec![-49, -47, -46, -40], data.iter().collect::<Vec<_>>());
    }

    #[test]
    fn test_chain() {
        let data1 = TestDataset::new(-50..0);
        let data2 = TestDataset::new(0..50);

        let data = data1.chain(data2);
        assert_eq!(Some(-50), data.get(0));
        assert_eq!(Some(0), data.get(50));
        let expected = (-50..50).collect::<Vec<_>>();
        assert_eq!(expected, data.iter().collect::<Vec<_>>());
    }
}
