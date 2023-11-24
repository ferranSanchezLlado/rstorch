use crate::module::activation::softmax::softmax;

use super::Loss;
use ndarray::prelude::*;

const EPSILON: f64 = f64::EPSILON;

pub struct CrossEntropyLoss {
    pred: Option<Array2<f64>>,
    truth: Option<Array2<f64>>,
}

impl CrossEntropyLoss {
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self {
            pred: None,
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
    fn forward(&mut self, pred: Array2<f64>, truth: Array2<f64>) -> f64 {
        let batch_size = pred.nrows() as f64;

        let pred = softmax(pred, Axis(1)); // Default axis = 1
        let loss = -(&truth * pred.mapv(f64::ln)).sum() / batch_size;
        self.pred = Some(pred);
        self.truth = Some(truth);
        loss
    }

    #[inline]
    fn backward(&mut self) -> Array2<f64> {
        -(self.truth.take().unwrap() - self.pred.take().unwrap())
    }
}
