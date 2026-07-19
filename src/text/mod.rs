//! Text utilities: fallible character and BPE tokenizers.

pub mod tokenizer;

pub use tokenizer::{BpeTokenizer, CharTokenizer, Tokenizer};
