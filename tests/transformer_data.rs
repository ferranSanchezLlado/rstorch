use rstorch::prelude::*;

#[test]
fn text_sequence_dataset_produces_shifted_lm_windows() {
    let tokenizer = CharTokenizer::from_text("hello");
    let ids = tokenizer.encode("hello", true);
    let dataset = TextSequenceDataset::<3>::new(ids.clone());

    assert_eq!(dataset.len(), ids.len() - 3);
    let sample = dataset.get(0).unwrap();
    assert_eq!(sample.input, [tokenizer.bos_id(), ids[1], ids[2]]);
    assert_eq!(sample.target, [ids[1], ids[2], ids[3]]);
}

#[test]
fn padded_causal_lm_collator_builds_padding_mask() {
    let batch = PaddedCausalLmCollator::<2, 4>::new(0)
        .collate(vec![vec![1, 2, 3], vec![4]])
        .unwrap();

    assert_eq!(batch.input, [[1, 2, 3, 0], [4, 0, 0, 0]]);
    assert_eq!(batch.target, [[2, 3, 0, 0], [0, 0, 0, 0]]);
    assert_eq!(
        batch.padding_mask.values(),
        &[false, false, false, true, false, true, true, true]
    );
}

#[cfg(feature = "hub")]
#[test]
fn tiny_shakespeare_wraps_text_as_dataset() {
    let tokenizer = CharTokenizer::from_text("hello");
    let dataset = TinyShakespeare::<3>::from_text("hello", &tokenizer, false);

    assert_eq!(dataset.len(), 2);
    assert_eq!(dataset.get(0).unwrap().input.len(), 3);
}
