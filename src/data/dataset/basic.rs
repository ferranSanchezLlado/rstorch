use super::Dataset;

/// Dataset backed by a `Vec`.
#[derive(Debug, Clone)]
pub struct Basic<T> {
    data: Vec<T>,
}

impl<T> Basic<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }
}

impl<T, U> Basic<(T, U)> {
    pub fn new_with_targets(input: Vec<T>, targets: Vec<U>) -> Self {
        assert_eq!(
            input.len(),
            targets.len(),
            "input and target lengths must match"
        );
        Self::new(input.into_iter().zip(targets).collect())
    }
}

impl<T> FromIterator<T> for Basic<T> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
    {
        Self::new(iter.into_iter().collect())
    }
}

impl<T> Dataset for Basic<T>
where
    T: Clone,
{
    type Item = T;

    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        self.data.get(index).cloned()
    }

    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
