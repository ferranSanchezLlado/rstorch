use super::Optimizer;
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
        todo!()
    }
    fn zero_grad<M: Module>(&mut self, module: &mut M) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimize() {
        let _optim = SGD::new(0.1);
        unimplemented!()
    }
}
