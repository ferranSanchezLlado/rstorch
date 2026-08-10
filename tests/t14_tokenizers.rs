//! Public-surface tests for the ported tokenizers.
//!
//! Mirrors the v2 `tests/tokenizers.rs` acceptance tests adapted to the
//! fallible encode/decode surface, plus the new 16.13 fallibility cases
//! (invalid UTF-8, unknown token id, invalid vocab size) exercised through
//! the public prelude.

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
fn char_decode_rejects_unknown_token_id() {
    let tokenizer = CharTokenizer::from_text("abc");
    let err = tokenizer.decode(&[100]).unwrap_err();
    assert!(matches!(err, Error::Tokenizer { .. }));
}

#[test]
fn bpe_decode_rejects_invalid_utf8() {
    let tokenizer = BpeTokenizer::train("", 260).unwrap();
    // Id for the raw byte 0xFF, which is never valid standalone UTF-8.
    let bad_byte_id = 4 + 0xFF;
    let err = tokenizer.decode(&[bad_byte_id]).unwrap_err();
    assert!(matches!(err, Error::Tokenizer { .. }));
    assert!(err.to_string().contains("not valid UTF-8"));
}

#[test]
fn bpe_train_rejects_target_vocab_below_floor() {
    let err = BpeTokenizer::train("abababab", 10).unwrap_err();
    assert!(matches!(err, Error::Tokenizer { .. }));
    assert!(err.to_string().contains("below the minimum of 260"));
}
