use std::fmt::Debug;

use crate::module::Module;
use ndarray::prelude::*;

use super::Parameters;

pub trait ModuleDebug: Module + Debug {}
impl<T: Module + Debug> ModuleDebug for T {}

#[derive(Debug, Default)]
pub struct Sequential {
    layers: Vec<Box<dyn ModuleDebug>>,
}

impl Sequential {
    #[inline]
    #[must_use]
    pub fn new(layers: Vec<Box<dyn ModuleDebug>>) -> Sequential {
        Sequential { layers }
    }

    #[inline]
    pub fn push<M: ModuleDebug + 'static>(&mut self, layer: M) {
        self.layers.push(Box::new(layer))
    }

    #[inline]
    pub fn insert<M: ModuleDebug + 'static>(&mut self, index: usize, layer: M) {
        self.layers.insert(index, Box::new(layer))
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    #[inline]
    pub fn remove(&mut self, index: usize) -> Option<Box<dyn ModuleDebug>> {
        match index >= self.len() {
            true => None,
            false => Some(self.layers.remove(index)),
        }
    }

    #[inline]
    pub fn push_box(&mut self, layer: Box<dyn ModuleDebug>) {
        self.layers.push(layer)
    }

    #[inline]
    pub fn insert_box(&mut self, index: usize, layer: Box<dyn ModuleDebug>) {
        self.layers.insert(index, layer)
    }
}

#[macro_export]
macro_rules! sequential {
    ($ ($layer:ident ($($arg:expr),* $(,)?) ),* $(,)?) => (
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
    #[inline]
    fn forward(&mut self, input: Array2<f64>) -> Array2<f64> {
        self.layers
            .iter_mut()
            .fold(input, |input, layer| layer.forward(input))
    }

    #[inline]
    fn backward(&mut self, gradient: Array2<f64>) -> Array2<f64> {
        self.layers
            .iter_mut()
            .rev()
            .fold(gradient, |gradient, layer| layer.backward(gradient))
    }

    #[inline]
    fn train(&mut self) {
        self.layers.iter_mut().for_each(|layer| layer.train())
    }

    #[inline]
    fn eval(&mut self) {
        self.layers.iter_mut().for_each(|layer| layer.eval())
    }

    fn parameters(&mut self) -> Parameters<'_> {
        let parms = self
            .layers
            .iter_mut()
            .flat_map(|l| l.parameters().iter())
            .collect::<Vec<_>>();
        Parameters { parms }
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
            Linear(3, 10,),
            ReLU(),
            Linear(10, 100),
            ReLU(),
            Linear(100, 2)
        );

        let _module_2 = sequential!(
            Linear::new(3, 10,),
            ReLU::new(),
            Linear::new(10, 100),
            ReLU::new(),
            Linear::new(100, 2)
        );
    }
}
