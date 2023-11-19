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

pub struct Parameter<'a> {
    pub parm: &'a mut Array2<f64>,
    pub grad: &'a mut Array2<f64>,
}

pub struct Parameters<'a> {
    parms: Vec<Parameter<'a>>,
}

impl<'a> Parameters<'a> {
    fn new(size: usize) -> Self {
        Self {
            parms: Vec::with_capacity(size),
        }
    }

    pub fn add(mut self, parm: &'a mut Array2<f64>, grad: &'a mut Array2<f64>) -> Self {
        self.parms.push(Parameter { parm, grad });
        self
    }

    pub fn iter(self) -> impl Iterator<Item = Parameter<'a>> {
        self.parms.into_iter()
    }
}

pub trait Module {
    /// (batch_size, input_size) -> (batch_size, output_size)
    fn forward(&mut self, input: Array2<f64>) -> Array2<f64>;

    /// (batch_size, output_size) -> (batch_size, input_size)
    fn backward(&mut self, gradient: Array2<f64>) -> Array2<f64>;

    fn parameters(&mut self) -> Parameters<'_>;

    #[inline]
    fn train(&mut self) {}

    #[inline]
    fn eval(&mut self) {}
}
