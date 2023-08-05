use ndarray::prelude::*;
use std::vec::IntoIter;

pub struct ParameterIterator<'a> {
    data: Vec<(&'a mut Array2<f64>, &'a Array2<f64>)>,
}

impl<'a> ParameterIterator<'a> {
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    #[inline]
    #[must_use]
    pub fn add(mut self, weight: &'a mut Array2<f64>, gradient: &'a Array2<f64>) -> Self {
        self.data.push((weight, gradient));
        self
    }
}

impl<'a> Default for ParameterIterator<'a> {
    #[inline]
    #[must_use]
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> IntoIterator for ParameterIterator<'a> {
    type Item = (&'a mut Array2<f64>, &'a Array2<f64>);
    type IntoIter = IntoIter<Self::Item>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a> FromIterator<(&'a mut Array2<f64>, &'a Array2<f64>)> for ParameterIterator<'a> {
    #[inline]
    #[must_use]
    fn from_iter<T: IntoIterator<Item = (&'a mut Array2<f64>, &'a Array2<f64>)>>(iter: T) -> Self {
        Self {
            data: Vec::from_iter(iter),
        }
    }
}
