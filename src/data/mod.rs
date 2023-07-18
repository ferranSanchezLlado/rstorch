mod subset;

pub trait Dataset {
    type Item;

    fn get(&self, index: usize) -> Option<&Self::Item>;
    fn len(&self) -> usize;
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait IterableDataset<'a>: Dataset {
    type Iterator: Iterator<Item = &'a Self::Item>
    where
        Self::Item: 'a;

    fn iter(&'a self) -> Self::Iterator;
}

#[cfg(test)]
mod test {
    use super::*;

    struct TestDataset {
        data: Vec<i32>,
    }

    impl Dataset for TestDataset {
        type Item = i32;

        fn get(&self, index: usize) -> Option<&Self::Item> {
            self.data.get(index)
        }
        fn len(&self) -> usize {
            self.data.len()
        }
    }

    #[test]
    fn test_dataset() {
        let data = TestDataset {
            data: (-50..50).collect(),
        };
        assert_eq!(Some(&-10), data.get(40));
    }
}
