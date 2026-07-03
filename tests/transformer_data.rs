use rstorch::prelude::*;
use rstorch::{DataError, Error};

#[test]
fn text_sequence_dataset_produces_shifted_lm_windows() {
    let tokenizer = CharTokenizer::from_text("hello");
    let ids = tokenizer.encode("hello", true);
    let dataset = TextSequenceDataset::<3>::new(ids.clone());

    assert_eq!(dataset.len(), ids.len() - 3);
    let sample = dataset.get(0).unwrap();
    assert_eq!(sample.input, [tokenizer.bos_id().unwrap(), ids[1], ids[2]]);
    assert_eq!(sample.target, [ids[1], ids[2], ids[3]]);
}

#[test]
fn padded_causal_lm_collator_builds_padding_mask() {
    let tokenizer = CharTokenizer::from_text("abcd");
    let batch = PaddedCausalLmCollator::<2, 4>::new(&tokenizer)
        .unwrap()
        .collate(vec![vec![1, 2, 3], vec![4]])
        .unwrap();

    assert_eq!(batch.input, [[1, 2, 3, 0], [4, 0, 0, 0]]);
    assert_eq!(batch.target, [[2, 3, 0, 0], [0, 0, 0, 0]]);
    assert_eq!(
        batch.padding_mask.to_vec().unwrap(),
        vec![false, false, false, true, false, true, true, true]
    );
}

#[test]
fn padded_causal_lm_collator_requires_pad_token() {
    struct NoPadTokenizer;

    impl Tokenizer for NoPadTokenizer {
        fn encode(&self, _text: &str, _add_special_tokens: bool) -> Vec<usize> {
            Vec::new()
        }

        fn decode(&self, _ids: &[usize]) -> String {
            String::new()
        }

        fn vocab_size(&self) -> usize {
            0
        }
    }

    let err = PaddedCausalLmCollator::<2, 4>::new(&NoPadTokenizer).unwrap_err();
    assert!(matches!(
        err,
        Error::Data(DataError::MissingSpecialToken { token: "pad" })
    ));
}

#[cfg(feature = "hub")]
#[test]
fn tiny_shakespeare_wraps_text_as_dataset() {
    let tokenizer = CharTokenizer::from_text("hello");
    let dataset = TinyShakespeare::<3>::from_text("hello", &tokenizer, false);

    assert_eq!(dataset.len(), 2);
    assert_eq!(dataset.get(0).unwrap().input.len(), 3);
}
