use crate::module::Module;
use ndarray::prelude::*;

#[derive(Debug, Default)]
pub struct Identity;

impl Identity {
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Module for Identity {
    #[inline]
    fn forward(&mut self, input: Array2<f64>) -> Array2<f64> {
        input
    }

    #[inline]
    fn backward(&mut self, gradient: Array2<f64>) -> Array2<f64> {
        gradient
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward() {
        let mut module = Identity::new();
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let result = module.forward(data.clone());

        crate::assert_array_eq!(result, data);
    }

    #[test]
    fn backward() {
        let mut module = Identity::new();
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let result = module.backward(data.clone());

        crate::assert_array_eq!(result, data);
    }
}
