use ndarray::prelude::*;

mod cross_entropy;
pub use cross_entropy::CrossEntropyLoss;

pub trait Loss {
    /// (batch_size, input_size)
    fn forward(&mut self, input: Array2<f64>, truth: Array2<f64>) -> f64;
    /// (batch_size, input_size)
    fn backward(&mut self) -> Array2<f64>;
}
