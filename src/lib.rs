//! RsTorch typed tensor facade and CPU-first neural-network utilities.
//!
//! # Backends and feature flags
//!
//! The default build is CPU-only, and CPU is the only supported training
//! path. The optional GPU backends behind the `metal`, `wgpu`, and `cuda`
//! features are **experimental**: each owns real device storage and kernels
//! for constructors, same-shape and scalar arithmetic, `matmul`, and `sum`;
//! Metal additionally has native forward hooks for the Epoch 16.6 training
//! set, including unaries, row softmax/log-softmax, `sum_last`, `bmm`,
//! broadcasts, masks, row gather, cross-entropy forward, and fused
//! LayerNorm/RMSNorm. Metal also has fused device-resident SGD/Adam update
//! kernels used by the built-in optimizers. Remaining tensor ops and some
//! backward formulas still execute as reference code through documented host
//! round trips, so none of the GPU backends is a supported training path.
//! Metal and WGPU are
//! parity-tested against CPU on real hardware; CUDA compiles and has typed
//! device errors but has never been verified on a real device — treat it as
//! untested. The audited support matrix, the 1.0 backend claims
//! decision, and the promotion path for experimental backends are recorded in
//! `docs/backend-dtype-support.md`.
//!
//! Tensor conversions are explicit and intentionally detach from autograd:
//! [`Tensor::cast`], [`Tensor::to_backend`], and [`Tensor::to_backend_on`]
//! return new leaf tensors because autograd graphs are single-dtype and
//! single-backend.
//!
//! `i64` is supported as a data dtype for ids, labels, and indices on CPU. It
//! intentionally does not support autograd, typed tensor arithmetic, matmul, or
//! GPU storage before 1.0; the existing `usize` id APIs remain the universal
//! every-backend paths.
//!
//! Static-shape tensor operators (`+`, `-`, `*`, `/`, unary `-`) are ergonomic
//! wrappers around the fallible method forms. They return [`Result`] instead of
//! panicking because backend kernels can still fail even when the type system has
//! removed the shape mismatch case.
//!
//! Autograd grad mode is thread-local: [`no_grad`] only affects the thread that
//! creates its non-`Send` guard, and [`is_grad_enabled`] reports the current
//! thread's mode. Tensor gradient state is internally mutex-guarded, so tensors
//! can be shared across threads when their backend storage is `Send + Sync`, but
//! concurrent gradient accumulation order is not specified.
//!
//! Public APIs return structured [`Error`] values for expected failures. Public
//! panics are limited to documented precondition violations such as
//! [`SmallRng::gen_range`] with a zero upper bound, internal invariant failures,
//! and poisoned synchronization primitives.
//!
//! Built-in optimizers use crate-internal backend update hooks. Direct optimizer
//! parameter data access through [`nn::ParameterRefMut`] remains a host `Vec<E>`
//! fallback for checkpointing and external optimizer implementations.

pub mod backend;
pub mod data;
pub mod dtype;
pub mod error;
pub mod nn;
pub mod optim;
pub mod persistence;
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
    Batch, Chain, Collate, DataLoader, Dataset, DynamicFeatures, FeatureBatch, Features,
    ImageBatch, ImageBatchWith, IntoTensorDataset, RandomSampler, SequentialSampler,
    ShuffleSampler, StackDynVecCollate, StackFeatures, StackImageCollate, StackImages,
    StackVecCollate, StaticDataLoader, StaticFeatures, StaticImageBatch, StaticStackImageCollate,
    StaticStackVecCollate, Subset, TensorDataset, Transform, VecDataset, dynamic_features,
    features, images, loader, normalize_image_sample, static_features, static_images,
    static_loader,
};
#[cfg(feature = "hub")]
pub use data::{
    DatasetHub, DatasetResource, Mnist, MnistCollate, MnistImageBatch, MnistImageCollate,
    MnistSample, MnistSplit, TINY_SHAKESPEARE, TinyShakespeare,
};
pub use dtype::{DType, DTypeId, FloatDType, bf16, f16};
pub use error::{DTypeError, DataError, DeviceError, Error, PersistenceError, Result, ShapeError};
pub use nn::{
    AvgPool2d, BatchNorm2d, Buffer, BufferRef, Conv2d, CrossEntropyOpts, Dropout, Embedding,
    Flatten, Gelu, HasParameters, Layer, LayerNorm, Linear, MaxPool2d, Module, MultiHeadAttention,
    Parameter, ParameterId, ParameterRef, ParameterRefMut, PositionalEmbedding, RMSNorm, Reduction,
    Relu, RngSource, Sequential, Sigmoid, Tanh, TrainContext, TrainingMode, bce_with_logits_loss,
    causal_attention_mask, causal_attention_mask_for_backend, huber_loss, l1_loss, mse_loss,
    scaled_dot_product_attention,
};
pub use optim::{
    Adam, AdamW, ConstantLr, CosineLr, LrSchedule, Optimizer, Sgd, StepLr, WarmupLr, clip_grad_norm,
};
pub use persistence::{
    Checkpoint, OptimizerKind, OptimizerParameterState, OptimizerState, OptimizerStateDict,
    ScalarRecord, StateDict, TensorRecord, load_checkpoint, load_tensor, save_checkpoint,
    save_tensor,
};
pub use random::SmallRng;
pub use shape::{
    AnyDim, C, D0, D1, D2, D3, D4, DimSpec, LastAxis, LeadingAxis, Shape, ShapeSpec, StaticShape,
    Sym,
};
pub use tensor::{
    Conv2dOptions, Mask, NoGradGuard, Padding2d, Pool2dOptions, Scalar, Tensor, Tensor1D, Tensor2D,
    Tensor3D, Tensor4D, is_grad_enabled, no_grad,
};
pub use transformer::{
    BpeTokenizer, CausalLmBatch, CausalLmSample, CharTokenizer, DecoderOnlyTransformer,
    GenerateOpts, PaddedCausalLmBatch, PaddedCausalLmCollator, TextSequenceDataset, Tokenizer,
    TransformerBlock, TransformerConfig, text_sequence_dataset,
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
        Batch, Chain, Collate, DataLoader, Dataset, DynamicFeatures, FeatureBatch, Features,
        ImageBatch, ImageBatchWith, IntoTensorDataset, RandomSampler, SequentialSampler,
        ShuffleSampler, StackDynVecCollate, StackFeatures, StackImageCollate, StackImages,
        StackVecCollate, StaticDataLoader, StaticFeatures, StaticImageBatch,
        StaticStackImageCollate, StaticStackVecCollate, Subset, TensorDataset, Transform,
        VecDataset, dynamic_features, features, images, loader, normalize_image_sample,
        static_features, static_images, static_loader,
    };
    #[cfg(feature = "hub")]
    pub use crate::data::{
        DatasetHub, DatasetResource, Mnist, MnistCollate, MnistImageBatch, MnistImageCollate,
        MnistSample, MnistSplit, TINY_SHAKESPEARE, TinyShakespeare,
    };
    pub use crate::dtype::{DType, FloatDType, bf16, f16};
    pub use crate::error::Result;
    pub use crate::nn::{
        AvgPool2d, BatchNorm2d, Buffer, BufferRef, Conv2d, CrossEntropyOpts, Dropout, Embedding,
        Flatten, Gelu, HasParameters, Layer, LayerNorm, Linear, MaxPool2d, Module,
        MultiHeadAttention, Parameter, ParameterRef, ParameterRefMut, PositionalEmbedding, RMSNorm,
        Reduction, Relu, RngSource, Sequential, Sigmoid, Tanh, TrainContext, TrainingMode,
        bce_with_logits_loss, causal_attention_mask, causal_attention_mask_for_backend, huber_loss,
        l1_loss, mse_loss, scaled_dot_product_attention,
    };
    pub use crate::optim::{
        Adam, AdamW, ConstantLr, CosineLr, LrSchedule, Optimizer, Sgd, StepLr, WarmupLr,
        clip_grad_norm,
    };
    pub use crate::persistence::{
        Checkpoint, OptimizerKind, OptimizerState, OptimizerStateDict, StateDict, TensorRecord,
        load_checkpoint, load_tensor, save_checkpoint, save_tensor,
    };
    pub use crate::random::SmallRng;
    pub use crate::seq;
    pub use crate::shape::{
        AnyDim, C, D0, D1, D2, D3, D4, DimSpec, LastAxis, LeadingAxis, ShapeSpec, StaticShape, Sym,
    };
    pub use crate::tensor::{
        Conv2dOptions, Mask, Padding2d, Pool2dOptions, Scalar, Tensor, Tensor1D, Tensor2D,
        Tensor3D, Tensor4D, is_grad_enabled, no_grad,
    };
    pub use crate::transformer::{
        BpeTokenizer, CausalLmBatch, CausalLmSample, CharTokenizer, DecoderOnlyTransformer,
        GenerateOpts, PaddedCausalLmBatch, PaddedCausalLmCollator, TextSequenceDataset, Tokenizer,
        TransformerBlock, TransformerConfig, text_sequence_dataset,
    };
}
