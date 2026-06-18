#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub mod backend;
pub mod data;
pub mod dtype;
pub mod nn;
pub mod optim;
pub mod rng;
pub mod shape;
pub mod tensor;

pub mod prelude {
    pub use crate::backend::Cpu;
    #[cfg(feature = "cuda")]
    pub use crate::backend::Cuda;
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub use crate::backend::Metal;
    #[cfg(feature = "datasets")]
    pub use crate::data::dataset::hub::{Mnist, MnistError, MnistSplit};
    pub use crate::data::{
        Basic, BatchSampler, BatchedDataLoader, Chain, Collate, DataLoader, Dataset,
        IdentityCollate, ImageOneHotClassification, OneHotClassification, PartialBatchSampler,
        RandomSampler, Sampler, SequentialSampler, Subset, Transform, one_hot_label,
        one_hot_labels,
    };
    pub use crate::dtype::FloatElement;
    pub use crate::nn::loss::{binary_cross_entropy, cross_entropy_one_hot, mse_loss};
    pub use crate::nn::{Layer, Linear, Module, Parameter, ReLU, Sequential, Sigmoid, Tanh};
    pub use crate::optim::{Adam, OptimParameter, SGD, SGDMomentum};
    pub use crate::rng::SmallRng;
    pub use crate::shape::{D0, D1, D2, D3, Shape};
    pub use crate::tensor::autograd::{NoGradGuard, is_grad_enabled, no_grad, with_no_grad};
    pub use crate::tensor::{Scalar, Tensor, Tensor1D, Tensor2D, Tensor3D, TensorError};
}
