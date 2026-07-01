use super::{DatasetHub, DatasetResource};
use crate::data::Dataset;
use crate::error::{DataError, Result};
use crate::transformer::{CausalLmSample, TextSequenceDataset, Tokenizer};
use std::fs;

const TINY_SHAKESPEARE_DATASET: &str = "tiny_shakespeare";

pub const TINY_SHAKESPEARE: DatasetResource = DatasetResource {
    name: "TinyShakespeare input.txt",
    url: "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    file_name: "input.txt",
};

#[derive(Debug, Clone)]
pub struct TinyShakespeare<const SEQ: usize> {
    windows: TextSequenceDataset<SEQ>,
}

impl<const SEQ: usize> TinyShakespeare<SEQ> {
    pub fn download(hub: &DatasetHub) -> Result<()> {
        hub.ensure_resource(TINY_SHAKESPEARE_DATASET, &TINY_SHAKESPEARE)?;
        Ok(())
    }

    pub fn load_text(hub: &DatasetHub) -> Result<String> {
        let path = hub.ensure_resource(TINY_SHAKESPEARE_DATASET, &TINY_SHAKESPEARE)?;
        fs::read_to_string(path).map_err(|source| DataError::Io { source }.into())
    }

    pub fn from_text<T>(text: &str, tokenizer: &T, add_special_tokens: bool) -> Self
    where
        T: Tokenizer,
    {
        Self {
            windows: TextSequenceDataset::from_text(text, tokenizer, add_special_tokens),
        }
    }

    pub fn load<T>(hub: &DatasetHub, tokenizer: &T, add_special_tokens: bool) -> Result<Self>
    where
        T: Tokenizer,
    {
        let text = Self::load_text(hub)?;
        Ok(Self::from_text(&text, tokenizer, add_special_tokens))
    }
}

impl<const SEQ: usize> Dataset for TinyShakespeare<SEQ> {
    type Item = CausalLmSample<SEQ>;
    type Error = crate::error::Error;

    fn len(&self) -> usize {
        self.windows.len()
    }

    fn get(&self, index: usize) -> Result<Self::Item> {
        self.windows.get(index)
    }
}
