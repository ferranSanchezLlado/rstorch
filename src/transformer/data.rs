use crate::data::Dataset;
use crate::error::{DataError, Result};
use crate::shape::{C, D2};
use crate::tensor::Mask;
use crate::transformer::Tokenizer;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalLmSample<const SEQ: usize> {
    pub input: [usize; SEQ],
    pub target: [usize; SEQ],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CausalLmBatch<const BATCH: usize, const SEQ: usize> {
    pub input: [[usize; SEQ]; BATCH],
    pub target: [[usize; SEQ]; BATCH],
}

#[derive(Debug, Clone)]
pub struct TextSequenceDataset<const SEQ: usize> {
    ids: Vec<usize>,
}

impl<const SEQ: usize> TextSequenceDataset<SEQ> {
    pub fn new(ids: Vec<usize>) -> Self {
        Self { ids }
    }

    pub fn from_text<T>(text: &str, tokenizer: &T, add_special_tokens: bool) -> Self
    where
        T: Tokenizer,
    {
        Self::new(tokenizer.encode(text, add_special_tokens))
    }
}

pub fn text_sequence_dataset<const SEQ: usize, T>(
    text: &str,
    tokenizer: &T,
    add_special_tokens: bool,
) -> TextSequenceDataset<SEQ>
where
    T: Tokenizer,
{
    TextSequenceDataset::from_text(text, tokenizer, add_special_tokens)
}

impl<const SEQ: usize> Dataset for TextSequenceDataset<SEQ> {
    type Item = CausalLmSample<SEQ>;
    type Error = crate::error::Error;

    fn len(&self) -> usize {
        self.ids.len().saturating_sub(SEQ)
    }

    fn get(&self, index: usize) -> Result<Self::Item> {
        if index >= self.len() {
            return Err(DataError::IndexOutOfBounds {
                index,
                len: self.len(),
            }
            .into());
        }
        Ok(CausalLmSample {
            input: self.ids[index..index + SEQ]
                .try_into()
                .expect("slice length is exactly SEQ"),
            target: self.ids[index + 1..index + SEQ + 1]
                .try_into()
                .expect("slice length is exactly SEQ"),
        })
    }
}

#[derive(Debug, Clone)]
pub struct PaddedCausalLmBatch<const BATCH: usize, const SEQ: usize> {
    pub input: [[usize; SEQ]; BATCH],
    pub target: [[usize; SEQ]; BATCH],
    pub padding_mask: Mask<D2<C<BATCH>, C<SEQ>>>,
}

#[derive(Debug, Clone, Copy)]
pub struct PaddedCausalLmCollator<const BATCH: usize, const SEQ: usize> {
    pad_id: usize,
}

impl<const BATCH: usize, const SEQ: usize> PaddedCausalLmCollator<BATCH, SEQ> {
    pub fn new(pad_id: usize) -> Self {
        Self { pad_id }
    }

    pub fn collate(&self, sequences: Vec<Vec<usize>>) -> Result<PaddedCausalLmBatch<BATCH, SEQ>> {
        if sequences.len() != BATCH {
            return Err(DataError::WrongBatchSize {
                expected: BATCH,
                found: sequences.len(),
            }
            .into());
        }

        let mut input = [[self.pad_id; SEQ]; BATCH];
        let mut target = [[self.pad_id; SEQ]; BATCH];
        let mut mask = Vec::with_capacity(BATCH * SEQ);
        for (batch, sequence) in sequences.iter().enumerate() {
            for step in 0..SEQ {
                if step < sequence.len() {
                    input[batch][step] = sequence[step];
                }
                if step + 1 < sequence.len() {
                    target[batch][step] = sequence[step + 1];
                }
                mask.push(step >= sequence.len());
            }
        }

        Ok(PaddedCausalLmBatch {
            input,
            target,
            padding_mask: Mask::from_vec(mask)?,
        })
    }
}
