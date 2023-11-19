use std::fmt::Debug;

use ndarray::prelude::*;

pub mod activation;
pub mod init;
mod linear;
mod safe_module;
mod sequential;

pub use activation::Identity;
pub use activation::ReLU;
pub use activation::Softmax;
pub use linear::Linear;
pub use safe_module::SafeModule;
pub use sequential::Sequential;

pub trait Module: Debug {
    /// (batch_size, input_size) -> (batch_size, output_size)
    fn forward(&mut self, input: Array2<f64>) -> Array2<f64>;

    /// (batch_size, output_size) -> (batch_size, input_size)
    fn backward(&mut self, gradient: Array2<f64>) -> Array2<f64>;

    #[inline]
    fn train(&mut self) {}

    #[inline]
    fn eval(&mut self) {}
}
