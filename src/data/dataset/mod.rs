mod basic;
mod chain;
#[cfg(feature = "datasets")]
pub mod hub;
mod subset;
mod transform;

pub use basic::Basic;
pub use chain::Chain;
pub use subset::Subset;
pub use transform::Transform;

/// Random-access dataset.
pub trait Dataset {
    type Item;

    fn len(&self) -> usize;

    fn get(&self, index: usize) -> Option<Self::Item>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn subset(self, indices: Vec<usize>) -> Subset<Self>
    where
        Self: Sized,
    {
        Subset::new(self, indices)
    }

    fn transform<F, Output>(self, transform: F) -> Transform<Self, F>
    where
        Self: Sized,
        F: Fn(Self::Item) -> Output,
    {
        Transform::new(self, transform)
    }

    fn chain<Other>(self, other: Other) -> Chain<Self, Other>
    where
        Self: Sized,
        Other: Dataset<Item = Self::Item>,
    {
        Chain::new(self, other)
    }
}

#[cfg(test)]
mod tests {
    use super::{Basic, Dataset};

    #[test]
    fn basic_dataset_reads_cloned_items() {
        let dataset = Basic::new(vec![1, 2, 3]);

        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.get(0), Some(1));
        assert_eq!(dataset.get(3), None);
    }

    #[test]
    fn basic_dataset_zips_inputs_and_targets() {
        let dataset = Basic::new_with_targets(vec![1, 2], vec![3, 4]);

        assert_eq!(dataset.get(0), Some((1, 3)));
        assert_eq!(dataset.get(1), Some((2, 4)));
    }

    #[test]
    fn dataset_adapters_subset_transform_and_chain() {
        let dataset = Basic::new(vec![1, 2, 3, 4])
            .subset(vec![3, 1])
            .transform(|value| value * 10)
            .chain(Basic::new(vec![50, 60]));

        assert_eq!(dataset.len(), 4);
        assert_eq!(dataset.get(0), Some(40));
        assert_eq!(dataset.get(1), Some(20));
        assert_eq!(dataset.get(2), Some(50));
        assert_eq!(dataset.get(3), Some(60));
    }
}
