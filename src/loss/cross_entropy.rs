use super::Loss;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::num_traits::Zero;

const EPSILON: f64 = f64::EPSILON;

pub struct CrossEntropyLoss {
    input: Option<Array2<f64>>,
    truth: Option<Array2<f64>>,
}

impl CrossEntropyLoss {
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            input: None,
            truth: None,
        }
    }
}

impl Default for CrossEntropyLoss {
    #[inline]
    #[must_use]
    fn default() -> Self {
        Self::new()
    }
}

impl Loss for CrossEntropyLoss {
    #[inline]
    fn forward(&mut self, input: Array2<f64>, truth: Array2<f64>) -> f64 {
        let batch_size = input.nrows() as f64;
        let loss = -(&truth * input.mapv(f64::ln)).sum() / batch_size;
        self.input = Some(input);
        self.truth = Some(truth);
        loss
    }

    #[inline]
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
