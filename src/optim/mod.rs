use crate::prelude::Module;

mod sgd;

pub trait Optimizer {
    fn step<M: Module>(&mut self, module: &mut M);
    fn zero_grad<M: Module>(&mut self, module: &mut M);
}
