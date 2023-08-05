use crate::module::init::InitParameters;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

#[derive(Debug, Default)]
pub struct XavierNormal;

impl InitParameters for XavierNormal {
    #[inline]
    fn weight(&self, input_size: usize, output_size: usize) -> Array2<f64> {
        let sigma = f64::sqrt(2.0 / (input_size as f64 + output_size as f64));
        Array::random((output_size, input_size), Normal::new(0.0, sigma).unwrap())
    }
}
