use crate::prelude::*;
use ndarray::prelude::*;

pub struct BasicModel<M, L, O> {
    module: M,
    loss: L,
    optim: O,
}

impl<M: Module, L: Loss, O: Optimizer> BasicModel<M, L, O> {
    fn new(module: M, loss: L, optim: O) -> Self {
        Self {
            module,
            loss,
            optim,
        }
    }

    fn step(&mut self, input: Array2<f64>, truth: Array2<f64>) {
        let input = self.module.forward(input);
        self.loss.forward(input, truth);

        let gradient = self.loss.backward();
        self.module.backward(gradient);

        self.optim.step(&mut self.module);
        self.optim.zero_grad(&mut self.module)
    }

    fn predict(&mut self, input: Array2<f64>) -> Array2<f64> {
        self.module.forward(input)
    }
}
