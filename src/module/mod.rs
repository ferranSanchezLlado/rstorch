use ndarray::prelude::*;

pub mod activation;
pub mod init;
mod linear;
mod sequential;

pub use activation::Identity;
pub use activation::ReLU;
pub use activation::Softmax;
pub use linear::Linear;
pub use sequential::Sequential;

pub trait Module: std::fmt::Debug {
    /// (batch_size, input_size) -> (batch_size, output_size)
    fn forward(&mut self, input: Array2<f64>) -> Array2<f64>;
    /// (batch_size, output_size) -> (batch_size, input_size)
    fn backward(&mut self, gradient: Array2<f64>) -> Array2<f64>;
}
