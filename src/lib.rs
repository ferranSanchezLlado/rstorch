pub mod module;

pub use module::{Identity, Linear, ReLU, Sequential, Softmax};

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
    pub use crate::module::init::InitParameters;
    pub use crate::module::Module;
}
