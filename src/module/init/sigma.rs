use crate::module::init::InitParameters;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Normal as NormalDist;
use ndarray_rand::RandomExt;

#[derive(Debug, Default)]
pub struct Normal {
    mean: f64,
    std: f64,
}

impl Normal {
    pub fn new(mean: f64, std: f64) -> Option<Self> {
        match std.is_finite() {
            true => Some(Self { mean, std }),
            false => None,
        }
    }

    pub fn new_std(std: f64) -> Option<Self> {
        Self::new(0.0, std)
    }
}

impl InitParameters for Normal {
    fn weight(&self, input_size: usize, output_size: usize) -> Array2<f64> {
        Array::random(
            (output_size, input_size),
            NormalDist::new(self.mean, self.std).unwrap(),
        )
    }
}
