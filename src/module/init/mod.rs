use ndarray::prelude::*;

mod he;
mod sigma;
mod xavier;

pub use he::KaimingNormal;
pub use sigma::Normal;
pub use xavier::XavierNormal;

pub trait InitParameters {
    fn weight(&self, input_size: usize, output_size: usize) -> Array2<f64>;
    fn bias(&self, _input_size: usize, output_size: usize) -> Array2<f64> {
        Array2::zeros((output_size, 1))
    }
}
