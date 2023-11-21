#![allow(dead_code)]
pub mod data;
mod iterator;
pub mod loss;
mod model;
pub mod module;
pub mod optim;

#[cfg(feature = "dataset_hub")]
pub use data::dataset::hub;
pub use module::{Identity, Linear, ReLU, SafeModule, Sequential, Softmax};
pub use optim::SGD;

mod macros {
    #[doc(hidden)]
    #[macro_export]
    macro_rules! __rust_force_expr {
        ($e:expr) => {
            $e
        };
    }

    #[cfg(test)]
    #[macro_export]
    macro_rules! assert_array_eq {
        ($lhs:expr, $rhs:expr) => {
            $crate::assert_array_eq!($lhs, $rhs, 1e-6)
        };
        ($lhs:expr, $rhs:expr, $tol:literal) => {
            if $lhs.shape() != $rhs.shape() {
                panic!(
                    "Incompatible shape \n- a={:?} \n\n- b={:?}",
                    $lhs.shape(),
                    $rhs.shape()
                );
            }

            for (a, b) in $lhs.iter().zip(&$rhs) {
                let diff = if a < b { b - a } else { a - b };
                if (diff > $tol) {
                    panic!(
                        "Not equal with tolerance={}\n- a={} \n\n- b={}",
                        $tol, &$lhs, &$rhs
                    );
                }
            }
        };
    }
}

pub mod prelude {
    // traits
    pub use crate::module::init::InitParameters;
    pub use crate::module::Module;

    pub use crate::data::dataset::{Dataset, IterableDataset};
    pub use crate::data::sampler::Sampler;

    pub use crate::loss::Loss;

    pub use crate::optim::Optimizer;

    // macros
    pub use crate::{safe, sequential};

    pub use ndarray::prelude::*;
}
