use super::Loss;
use ndarray::prelude::*;

pub struct CrossEntropyLoss {
    pred: Option<Array2<f64>>,
    truth: Option<Array2<f64>>,
}

impl Loss for CrossEntropyLoss {
    fn forward(&mut self, input: Array2<f64>, truth: Array2<f64>) -> f64 {
        let loss = -(&input * &truth).sum() / (input.nrows() as f64);
        self.pred = Some(input);
        self.truth = Some(truth);
        loss
    }

    fn backward(&mut self) -> Array2<f64> {
        -self.truth.take().unwrap() / self.pred.take().unwrap()
    }
}
