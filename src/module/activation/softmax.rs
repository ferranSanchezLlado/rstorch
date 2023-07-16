use crate::module::Module;
use ndarray::prelude::*;

#[derive(Debug, Default)]
pub struct Softmax {
    output: Option<Array2<f64>>,
}

impl Softmax {
    pub fn new() -> Self {
        Softmax { output: None }
    }
}

impl Module for Softmax {
    fn forward(&mut self, input: Array2<f64>) -> Array2<f64> {
        let exp = (&input - input.mapv(|x| f64::from(x > 0.0))).mapv(f64::exp);
        self.output = Some(&exp / exp.sum_axis(Axis(0)));
        self.output.clone().unwrap()
    }
    fn backward(&mut self, gradient: Array2<f64>) -> Array2<f64> {
        let output = self.output.take().unwrap();
        &output * (&gradient - (&output * &gradient).sum_axis(Axis(0)))
    }
}
