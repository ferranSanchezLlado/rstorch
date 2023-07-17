use crate::module::Module;
use ndarray::prelude::*;

#[derive(Debug, Default)]
pub struct ReLU {
    prev_input: Option<Array2<f64>>,
}

impl ReLU {
    pub fn new() -> Self {
        ReLU { prev_input: None }
    }
}

impl Module for ReLU {
    fn forward(&mut self, input: Array2<f64>) -> Array2<f64> {
        self.prev_input = Some(input.clone());
        input.mapv(|x| x.max(0.0))
    }

    fn backward(&mut self, gradient: Array2<f64>) -> Array2<f64> {
        gradient * self.prev_input.take().unwrap().mapv(|x| f64::from(x > 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward() {
        let mut module = ReLU::new();
        let data = array![[1.0, 2.0], [3.0, -4.0], [-5.0, -6.0]];
        let result = module.forward(data);
        let expected = array![[1.0, 2.0], [3.0, 0.0], [0.0, 0.0]];

        crate::assert_array_eq!(result, expected);
    }

    #[test]
    fn backward() {
        let mut module = ReLU::new();
        let data = array![[1.0, 2.0], [3.0, -4.0], [-5.0, -6.0]];
        module.forward(data);
        let result = module.backward(Array::ones((3, 2)));
        let expected = array![[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

        crate::assert_array_eq!(result, expected);
    }
}
