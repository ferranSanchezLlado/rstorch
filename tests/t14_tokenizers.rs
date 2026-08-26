//! Public API tests for the character and byte-pair tokenizers.

use rstorch::prelude::*;

#[test]
fn char_tokenizer_round_trips_with_special_tokens() {
    let tokenizer = CharTokenizer::from_text("hello");
    let ids = tokenizer.encode("hello", true).unwrap();

    assert_eq!(tokenizer.vocab_size(), 8);
    assert_eq!(tokenizer.pad_id(), Some(0));
    assert_eq!(tokenizer.unk_id(), Some(1));
    assert_eq!(tokenizer.bos_id(), Some(2));
    assert_eq!(tokenizer.eos_id(), Some(3));
    assert_eq!(tokenizer.decode(&ids).unwrap(), "hello");
}

#[test]
fn bpe_tokenizer_is_deterministic_and_round_trips_bytes() {
    let tokenizer = BpeTokenizer::train("abababab", 256 + 4 + 2).unwrap();

    assert_eq!(tokenizer.vocab_size(), 262);
    let ids = tokenizer.encode("abab", true).unwrap();
    assert_eq!(tokenizer.decode(&ids).unwrap(), "abab");

    let again = BpeTokenizer::train("abababab", 256 + 4 + 2).unwrap();
    assert_eq!(tokenizer.merges(), again.merges());
}

#[test]
fn tokenizer_trait_is_object_safe_and_dispatches() {
    let boxed: Box<dyn Tokenizer> = Box::new(CharTokenizer::from_text("abc"));
    let ids = boxed.encode("abc", false).unwrap();
    assert_eq!(boxed.decode(&ids).unwrap(), "abc");
    assert_eq!(boxed.vocab_size(), 7);
}

#[test]
fn tokenizer_errors_are_structured() {
    let chars = CharTokenizer::from_text("abc");
    let bpe = BpeTokenizer::train("", 260).unwrap();
    let errors = [
        chars.decode(&[100]).unwrap_err(),
        bpe.decode(&[4 + 0xFF]).unwrap_err(),
        BpeTokenizer::train("abababab", 10).unwrap_err(),
    ];

    for error in errors {
        assert!(matches!(error, Error::Tokenizer { .. }));
    }
}
