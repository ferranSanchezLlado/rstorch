use crate::module::{Module, Parameters};
use ndarray::prelude::*;
use std::cmp::Ordering;

#[derive(Debug)]
pub struct Softmax {
    output: Option<Array2<f64>>,
    axis: Axis,
}

impl Default for Softmax {
    #[inline]
    #[must_use]
    fn default() -> Self {
        Self::new()
    }
}

impl Softmax {
    #[inline]
    #[must_use]
    pub fn with_axis(axis: usize) -> Self {
        Softmax {
            output: None,
            axis: Axis(axis),
        }
    }

    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self::with_axis(1)
    }
}

#[inline]
fn max(a: f64, b: &f64) -> f64 {
    match a.partial_cmp(b).unwrap() {
        Ordering::Less => *b,
        Ordering::Equal => a,
        Ordering::Greater => a,
    }
}

impl Module for Softmax {
    #[inline]
    fn forward(&mut self, input: Array2<f64>) -> Array2<f64> {
        // Broadcasting fails, but inserting axis makes it work properly
        let max_axis = input
            .map_axis(self.axis, |axis| axis.iter().fold(f64::MIN, max))
            .insert_axis(self.axis);
        let exp = (input - max_axis).mapv(f64::exp);
        let sum_axis = exp.sum_axis(self.axis).insert_axis(self.axis);

        self.output = Some(exp / sum_axis);
        self.output.clone().unwrap()
    }

    #[inline]
    fn backward(&mut self, gradient: Array2<f64>) -> Array2<f64> {
        let output = self.output.take().unwrap();

        let mut jacobian = -output.t().dot(&output);
        jacobian.diag_mut().into_iter().for_each(|el| {
            *el = -*el;
            *el = *el * (1.0 - *el);
        });

        gradient.dot(&jacobian)
    }

    fn parameters(&mut self) -> Parameters<'_> {
        Parameters::new(0)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::assert_array_eq;

    #[test]
    fn forward() {
        let input = array![[1.0, 10.0], [-3.0, 4.0], [5.0, 6.0]];
        let mut module = Softmax::new();

        let output = module.forward(input);
        assert_array_eq!(array![1.0, 1.0, 1.0], output.sum_axis(Axis(1)));
        assert!(output.sum().eq(&3.0), "{} != {}", 3.0, output.sum());
    }

    #[test]
    fn forward_other_axis() {
        let input = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mut module = Softmax::with_axis(0);

        let output = module.forward(input);
        assert_array_eq!(array![1.0, 1.0], output.sum_axis(Axis(0)));
        assert!(output.sum().eq(&2.0), "{} != {}", 2.0, output.sum());
    }

    #[test]
    fn backward() {
        let input = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mut module = Softmax::new();
        module.forward(input);
        let result = module.backward(Array::ones((3, 2)));
        let expected = array![[-0.41, -1.55], [-0.41, -1.55], [-0.41, -1.55]];
        assert_array_eq!(result, expected, 0.01);
    }
}
