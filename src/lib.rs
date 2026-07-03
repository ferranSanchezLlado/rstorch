//! RsTorch typed tensor facade and CPU-first neural-network utilities.
//!
//! Tensor conversions are explicit and intentionally detach from autograd:
//! [`Tensor::cast`], [`Tensor::to_backend`], and [`Tensor::to_backend_on`]
//! return new leaf tensors because autograd graphs are single-dtype and
//! single-backend.
//!
//! Optimizer parameter data access through [`nn::ParameterRefMut`] currently
//! uses host `Vec<E>` round trips. That surface is intentionally unstable for
//! external optimizer implementors until the device-resident optimizer path is
//! settled.

pub mod backend;
pub mod data;
pub mod dtype;
pub mod error;
pub mod nn;
pub mod optim;
pub mod random;
pub mod shape;
pub mod tensor;
pub mod transformer;

pub use backend::{Backend, Cpu, CpuDevice, CpuError};
#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
pub use backend::{Cuda, CudaDevice, CudaError};
#[cfg(all(feature = "metal", target_os = "macos"))]
pub use backend::{Metal, MetalDevice, MetalError};
#[cfg(feature = "wgpu")]
pub use backend::{Wgpu, WgpuDevice, WgpuError};
pub use data::{
    Batch, Chain, Collate, DataLoader, Dataset, DynamicFeatures, Features, ImageBatch,
    IntoTensorDataset, RandomSampler, SequentialSampler, ShuffleSampler, StackDynVecCollate,
    StackImageCollate, StackVecCollate, StaticDataLoader, StaticFeatures, StaticImageBatch,
    StaticStackImageCollate, StaticStackVecCollate, Subset, TensorDataset, Transform, VecDataset,
    dynamic_features, features, images, loader, static_features, static_images, static_loader,
};
#[cfg(feature = "hub")]
pub use data::{
    DatasetHub, DatasetResource, Mnist, MnistCollate, MnistImageBatch, MnistImageCollate,
    MnistSample, MnistSplit, TINY_SHAKESPEARE, TinyShakespeare,
};
pub use dtype::{DType, DTypeId, FloatDType, bf16, f16};
pub use error::{DTypeError, DataError, DeviceError, Error, Result, ShapeError};
pub use nn::{
    CrossEntropyOpts, Dropout, Embedding, Gelu, HasParameters, Layer, LayerNorm, Linear, Module,
    MultiHeadAttention, Parameter, ParameterId, PositionalEmbedding, Reduction, Relu, RngSource,
    Sequential, Sigmoid, Tanh, TrainContext, TrainingMode, causal_attention_mask,
    causal_attention_mask_for_backend, mse_loss, scaled_dot_product_attention,
};
pub use optim::{
    Adam, AdamW, ConstantLr, CosineLr, LrSchedule, Optimizer, Sgd, StepLr, WarmupLr, clip_grad_norm,
};
pub use random::SmallRng;
pub use shape::{AnyDim, C, D0, D1, D2, D3, D4, DimSpec, Shape, ShapeSpec, StaticShape, Sym};
pub use tensor::{
    Mask, NoGradGuard, Scalar, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D, is_grad_enabled,
    no_grad,
};
pub use transformer::{
    BpeTokenizer, CausalLmBatch, CausalLmSample, CharTokenizer, DecoderOnlyTransformer,
    PaddedCausalLmBatch, PaddedCausalLmCollator, TextSequenceDataset, Tokenizer, TransformerBlock,
    TransformerConfig, text_sequence_dataset,
};

pub mod prelude {
    pub use crate::backend::{Backend, Cpu};
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    pub use crate::backend::{Cuda, CudaDevice, CudaError};
    #[cfg(all(feature = "metal", target_os = "macos"))]
    pub use crate::backend::{Metal, MetalDevice, MetalError};
    #[cfg(feature = "wgpu")]
    pub use crate::backend::{Wgpu, WgpuDevice, WgpuError};
    pub use crate::data::{
        Batch, Chain, Collate, DataLoader, Dataset, DynamicFeatures, Features, ImageBatch,
        IntoTensorDataset, RandomSampler, SequentialSampler, ShuffleSampler, StackDynVecCollate,
        StackImageCollate, StackVecCollate, StaticDataLoader, StaticFeatures, StaticImageBatch,
        StaticStackImageCollate, StaticStackVecCollate, Subset, TensorDataset, Transform,
        VecDataset, dynamic_features, features, images, loader, static_features, static_images,
        static_loader,
    };
    #[cfg(feature = "hub")]
    pub use crate::data::{
        DatasetHub, DatasetResource, Mnist, MnistCollate, MnistImageBatch, MnistImageCollate,
        MnistSample, MnistSplit, TINY_SHAKESPEARE, TinyShakespeare,
    };
    pub use crate::dtype::{DType, FloatDType, bf16, f16};
    pub use crate::error::Result;
    pub use crate::nn::{
        CrossEntropyOpts, Dropout, Embedding, Gelu, HasParameters, Layer, LayerNorm, Linear,
        Module, MultiHeadAttention, Parameter, PositionalEmbedding, Reduction, Relu, RngSource,
        Sequential, Sigmoid, Tanh, TrainContext, TrainingMode, causal_attention_mask,
        causal_attention_mask_for_backend, mse_loss, scaled_dot_product_attention,
    };
    pub use crate::optim::{
        Adam, AdamW, ConstantLr, CosineLr, LrSchedule, Optimizer, Sgd, StepLr, WarmupLr,
        clip_grad_norm,
    };
    pub use crate::random::SmallRng;
    pub use crate::seq;
    pub use crate::shape::{AnyDim, C, D0, D1, D2, D3, D4, DimSpec, ShapeSpec, StaticShape, Sym};
    pub use crate::tensor::{
        Mask, Scalar, Tensor, Tensor1D, Tensor2D, Tensor3D, Tensor4D, is_grad_enabled, no_grad,
    };
    pub use crate::transformer::{
        BpeTokenizer, CausalLmBatch, CausalLmSample, CharTokenizer, DecoderOnlyTransformer,
        PaddedCausalLmBatch, PaddedCausalLmCollator, TextSequenceDataset, Tokenizer,
        TransformerBlock, TransformerConfig, text_sequence_dataset,
    };
}
