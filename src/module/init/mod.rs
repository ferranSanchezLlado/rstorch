use ndarray::prelude::*;

mod he;
mod xavier;

pub use he::KaimingNormal;
pub use xavier::XavierNormal;

pub trait InitParameters: std::fmt::Debug {
    fn weight(input_size: usize, output_size: usize) -> Array2<f64>;
    fn bias(_input_size: usize, output_size: usize) -> Array1<f64> {
        Array1::zeros(output_size)
    }
}
