//! Ready-to-train model assemblies built from the neural-network layers.

mod transformer;

pub use transformer::{DecoderTransformer, KvCache, TransformerConfig};
