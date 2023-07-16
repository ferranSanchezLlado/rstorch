use crate::module::Module;
use ndarray::prelude::*;

#[derive(Debug, Default)]
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Sequential {
        Sequential { layers }
    }

    pub fn push<M: Module + 'static>(&mut self, layer: M) {
        self.layers.push(Box::new(layer))
    }

    pub fn insert<M: Module + 'static>(&mut self, index: usize, layer: M) {
        self.layers.insert(index, Box::new(layer))
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn remove(&mut self, index: usize) -> Option<Box<dyn Module>> {
        match index >= self.len() {
            true => None,
            false => Some(self.layers.remove(index)),
        }
    }

    pub fn push_box(&mut self, layer: Box<dyn Module>) {
        self.layers.push(layer)
    }

    pub fn insert_box(&mut self, index: usize, layer: Box<dyn Module>) {
        self.layers.insert(index, layer)
    }
}

#[macro_export]
macro_rules! sequential {
    ($ ($layer:ident ($($arg:expr),*) ),* $(,)?) => (
        $crate::__rust_force_expr!(Sequential::new(
            vec![$(Box::new($layer::new($($arg,)*)),)*]
        ))
    );
    ($($layer:expr),* $(,)?) => (
        $crate::__rust_force_expr!(Sequential::new(
            vec![$(Box::new($layer),)*]
        ))
    );
}

impl Module for Sequential {
    fn forward(&mut self, input: Array2<f64>) -> Array2<f64> {
        self.layers
            .iter_mut()
            .fold(input, |input, layer| layer.forward(input))
    }
    fn backward(&mut self, gradient: Array2<f64>) -> Array2<f64> {
        self.layers
            .iter_mut()
            .rev()
            .fold(gradient, |gradient, layer| layer.backward(gradient))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::activation::ReLU;
    use crate::module::linear::Linear;

    #[test]
    fn test_macro() {
        let _module = sequential!(
            Linear(3, 10),
            ReLU(),
            Linear(10, 100),
            ReLU(),
            Linear(100, 2)
        );

        let _module_2 = sequential!(
            Linear::new(3, 10),
            ReLU::new(),
            Linear::new(10, 100),
            ReLU::new(),
            Linear::new(100, 2)
        );
    }
}
