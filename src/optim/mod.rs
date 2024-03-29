use crate::prelude::Module;

mod sgd;
pub use sgd::SGD;

pub trait Optimizer {
    fn step<M: Module>(&mut self, module: &mut M);
}
