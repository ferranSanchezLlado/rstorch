use rstorch::prelude::*;

#[test]
fn char_tokenizer_round_trips_with_special_tokens() {
    let tokenizer = CharTokenizer::from_text("hello");
    let ids = tokenizer.encode("hello", true);

    assert_eq!(tokenizer.vocab_size(), 8);
    assert_eq!(tokenizer.pad_id(), Some(0));
    assert_eq!(tokenizer.unk_id(), Some(1));
    assert_eq!(tokenizer.bos_id(), Some(2));
    assert_eq!(tokenizer.eos_id(), Some(3));
    assert_eq!(tokenizer.decode(&ids), "hello");
}

#[test]
fn bpe_tokenizer_is_deterministic_and_round_trips_bytes() {
    let tokenizer = BpeTokenizer::train("abababab", 256 + 4 + 2);

    assert_eq!(tokenizer.vocab_size(), 262);
    assert_eq!(tokenizer.decode(&tokenizer.encode("abab", true)), "abab");

    let again = BpeTokenizer::train("abababab", 256 + 4 + 2);
    assert_eq!(tokenizer.merges(), again.merges());
}

#[test]
fn tokenizer_special_tokens_default_to_none() {
    struct NoSpecials;

    impl Tokenizer for NoSpecials {
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

    let tokenizer = NoSpecials;
    assert_eq!(tokenizer.pad_id(), None);
    assert_eq!(tokenizer.unk_id(), None);
    assert_eq!(tokenizer.bos_id(), None);
    assert_eq!(tokenizer.eos_id(), None);
}
