#![allow(dead_code)]
use crate::module::Module;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;

use ndarray::prelude::*;

// Training states
pub struct Forward;
pub struct Backward;

// Evaluations modes
pub struct Train;
pub struct Evaluation;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct SafeModule<M, S, E> {
    module: ManuallyDrop<M>,
    state: PhantomData<S>,
    mode: PhantomData<E>,
}

impl<M, S, E> SafeModule<M, S, E> {
    #[inline]
    pub fn into_inner(mut self) -> M {
        unsafe { ManuallyDrop::take(&mut self.module) }
    }

    #[inline]
    fn new_state<N>(self) -> SafeModule<M, N, E> {
        SafeModule {
            module: ManuallyDrop::new(self.into_inner()),
            state: PhantomData,
            mode: PhantomData,
        }
    }

    #[inline]
    fn new_mode<N>(self) -> SafeModule<M, S, N> {
        SafeModule {
            module: ManuallyDrop::new(self.into_inner()),
            state: PhantomData,
            mode: PhantomData,
        }
    }
}

impl<M: Module> SafeModule<M, Forward, Train> {
    pub fn new(module: M) -> Self {
        Self {
            module: ManuallyDrop::new(module),
            state: PhantomData,
            mode: PhantomData,
        }
    }
}

impl<M: Module, E> SafeModule<M, Forward, E> {
    /// (batch_size, input_size) -> (batch_size, output_size)
    pub fn forward(mut self, input: Array2<f64>) -> (SafeModule<M, Backward, E>, Array2<f64>) {
        let pred = self.module.forward(input);
        let new_state = self.new_state();
        (new_state, pred)
    }
}

impl<M: Module, E> SafeModule<M, Backward, E> {
    /// (batch_size, output_size) -> (batch_size, input_size)
    pub fn backward(mut self, gradient: Array2<f64>) -> (SafeModule<M, Forward, E>, Array2<f64>) {
        let grad = self.module.backward(gradient);
        let new_state = self.new_state();
        (new_state, grad)
    }
}

impl<M: Module, S> SafeModule<M, S, Train> {
    pub fn eval(mut self) -> SafeModule<M, S, Evaluation> {
        self.module.eval();
        self.new_mode()
    }
}

impl<M: Module, S> SafeModule<M, S, Evaluation> {
    pub fn train(mut self) -> SafeModule<M, S, Train> {
        self.module.train();
        self.new_mode()
    }
}

impl<M: Module> From<M> for SafeModule<M, Forward, Train> {
    fn from(module: M) -> Self {
        Self::new(module)
    }
}

#[macro_export]
macro_rules! safe {
    ($layer:ident ($($arg:expr),* $(,)?) ) => (
        $crate::__rust_force_expr!(
            SafeModule::new($layer::new($($arg,)*))
        )
    );
    ($layer:expr) => (
        $crate::__rust_force_expr!(
            SafeModule::new($layer)
        )
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Linear, ReLU};

    #[test]
    fn test_macro() {
        let _relu = safe!(ReLU());
        let _linear = safe!(Linear(10, 20,));

        let _relu = safe!(ReLU::new());
        let _linear = safe!(Linear::new(10, 20));
    }

    #[test]
    fn test_state() {
        let relu = safe!(ReLU());
        let (relu, _) = relu.forward(Array::ones((2, 3)));
        let (_relu, _) = relu.backward(Array::ones((2, 3)));
    }

    #[test]
    fn test_mode() {
        let linear = safe!(Linear(10, 20,));
        let linear = linear.eval();
        let _linear = linear.train();
    }
}
