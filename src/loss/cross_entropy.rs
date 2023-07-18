use super::Loss;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::num_traits::Zero;

const EPSILON: f64 = f64::EPSILON;

pub struct CrossEntropyLoss {
    input: Option<Array2<f64>>,
    truth: Option<Array2<f64>>,
}

impl Loss for CrossEntropyLoss {
    fn forward(&mut self, input: Array2<f64>, truth: Array2<f64>) -> f64 {
        let batch_size = input.nrows() as f64;
        let loss = -(&truth * input.mapv(f64::ln)).sum() / batch_size;
        self.input = Some(input);
        self.truth = Some(truth);
        loss
    }

    fn backward(&mut self) -> Array2<f64> {
        let batch_size = self.input.as_ref().unwrap().nrows() as f64;
        let input = self
            .input
            .take()
            .unwrap()
            .mapv(|el| if el.is_zero() { EPSILON } else { el });
        -self.truth.take().unwrap() / input / batch_size
    }
}
