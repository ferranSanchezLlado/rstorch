use super::Optimizer;
use crate::module::Parameter;
use crate::prelude::Module;

#[allow(clippy::upper_case_acronyms)]
pub struct SGD {
    lr: f64,
}

impl SGD {
    #[inline]
    #[must_use]
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }
}

impl Optimizer for SGD {
    fn step<M: Module>(&mut self, module: &mut M) {
        module
            .parameters()
            .iter()
            .for_each(|Parameter { parm, grad }| *parm = parm.clone() - (self.lr * grad.clone()))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array;

    use super::*;
    use crate::Linear;

    #[test]
    fn optimize() {
        let mut optim = SGD::new(0.1);
        let mut linear = Linear::new(2, 2);

        linear.forward(Array::zeros((2, 2)));
        linear.backward(Array::zeros((2, 2)));

        optim.step(&mut linear);
    }
}
