use super::Dataset;

/// Dataset formed by concatenating two datasets with the same item type.
#[derive(Debug, Clone)]
pub struct Chain<First, Second> {
    first: First,
    second: Second,
}

impl<First, Second> Chain<First, Second> {
    pub fn new(first: First, second: Second) -> Self {
        Self { first, second }
    }
}

impl<First, Second> Dataset for Chain<First, Second>
where
    First: Dataset,
    Second: Dataset<Item = First::Item>,
{
    type Item = First::Item;

    fn len(&self) -> usize {
        self.first.len() + self.second.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        if index < self.first.len() {
            self.first.get(index)
        } else {
            self.second.get(index - self.first.len())
        }
    }
}
