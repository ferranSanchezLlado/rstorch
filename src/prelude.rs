//! Common tensor, neural-network, optimizer, data, text, and model APIs.
pub use crate::{DType, Device, Error, Grads, Result, Rng, Shape, Tensor};

pub use crate::nn::{Forward, Mode, Module, ModuleExt, Param, Sequential, StateDict};

// Common activation and layer types.
pub use crate::nn::{Dropout, Gelu, Linear, Relu};

// Normalization layers.
pub use crate::nn::{BatchNorm2d, LayerNorm, RMSNorm};

// Optimizers commonly used in the examples.
pub use crate::optim::{Adam, AdamW, Sgd};

pub use crate::text::{BpeTokenizer, CharTokenizer, Tokenizer};

// In-memory datasets and the data loader.
pub use crate::data::{DataLoader, Dataset, TensorDataset, VecDataset};

// Attention and embedding layers.
pub use crate::nn::{AttentionInput, Embedding, MultiHeadAttention};

// The trait and derive macro intentionally share the name `Module`.
pub use crate::Module;

// Transformer model and generation cache.
pub use crate::models::{DecoderTransformer, KvCache, TransformerConfig};

// Convolution and pooling layers.
pub use crate::nn::{AvgPool2d, Conv2d, Flatten, Identity, MaxPool2d};
