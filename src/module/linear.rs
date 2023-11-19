use crate::module::init::{InitParameters, KaimingNormal};
use crate::module::Module;
use ndarray::prelude::*;

use super::Parameters;

#[derive(Debug)]
pub struct Linear {
    weight: Array2<f64>,
    bias: Option<Array2<f64>>,

    prev_input: Option<Array2<f64>>,
    grad_weight: Option<Array2<f64>>,
    grad_bias: Option<Array2<f64>>,
}

impl Linear {
    #[inline]
    #[must_use]
    pub fn new_with_kernel<I: InitParameters>(
        input_size: usize,
        output_size: usize,
        init: I,
    ) -> Linear {
        Linear {
            weight: init.weight(input_size, output_size),
            bias: Some(init.bias(input_size, output_size)),
            prev_input: None,
            grad_weight: None,
            grad_bias: None,
        }
    }

    #[inline]
    #[must_use]
    pub fn new(input_size: usize, output_size: usize) -> Linear {
        Linear::new_with_kernel(input_size, output_size, KaimingNormal)
    }
}

impl Module for Linear {
    #[inline]
    fn forward(&mut self, input: Array2<f64>) -> Array2<f64> {
        let mut x = input.dot(&self.weight.t());

        if let Some(bias) = &self.bias {
            x += bias;
        }

        self.prev_input = Some(input);
        x
    }

    #[inline]
    fn backward(&mut self, gradient: Array2<f64>) -> Array2<f64> {
        let prev_input = self.prev_input.take().unwrap();
        let n = prev_input.nrows() as f64;
        self.grad_weight = Some(gradient.t().dot(&prev_input) / n);

        if self.bias.is_some() {
            self.grad_bias = Some((gradient.sum_axis(Axis(0)) / n).insert_axis(Axis(1)));
        }
        gradient.dot(&self.weight)
    }

    fn parameters(&mut self) -> Parameters<'_> {
        let params = Parameters::new(2).add(&mut self.weight, self.grad_weight.as_mut().unwrap());

        match self.bias.as_mut() {
            Some(bias) => params.add(bias, self.grad_bias.as_mut().unwrap()),
            None => params,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct FixedInit;

    impl InitParameters for FixedInit {
        fn weight(&self, _input_size: usize, _output_size: usize) -> Array2<f64> {
            array![[1.0, 0.5]]
        }
        fn bias(&self, _input_size: usize, _output_size: usize) -> Array2<f64> {
            array![[10.0]]
        }
    }

    #[test]
    fn forward() {
        let mut module = Linear::new_with_kernel(2, 1, FixedInit);
        let data = array![[1.0, 2.0], [3.0, -4.0], [-5.0, -6.0]];
        let result = module.forward(data.clone());
        let expected = array![[12.0], [11.0], [2.0]];

        crate::assert_array_eq!(result, expected);
    }

    #[test]
    fn backward() {
        let mut module = Linear::new_with_kernel(2, 1, FixedInit);
        let data = array![[1.0, 2.0], [3.0, -4.0], [-5.0, -6.0], [-7.0, 8.0]];
        module.forward(data.clone());
        let result = module.backward(Array::ones((4, 1)));

        assert_eq!(
            module.grad_weight.as_ref().unwrap().shape(),
            module.weight.shape()
        );

        assert_eq!(
            module.grad_bias.as_ref().unwrap().shape(),
            module.bias.unwrap().shape()
        );

        let expected_grad_w = array![[-2.0, 0.0]];
        let expected_grad_b = array![[1.0]];
        let expected_grad = array![[1.0, 0.5], [1.0, 0.5], [1.0, 0.5], [1.0, 0.5]];

        crate::assert_array_eq!(module.grad_weight.clone().unwrap(), expected_grad_w);
        crate::assert_array_eq!(module.grad_bias.clone().unwrap(), expected_grad_b);
        crate::assert_array_eq!(result, expected_grad);
    }
}
