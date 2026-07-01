use rstorch::prelude::*;

#[test]
fn char_tokenizer_round_trips_with_special_tokens() {
    let tokenizer = CharTokenizer::from_text("hello");
    let ids = tokenizer.encode("hello", true);

    assert_eq!(tokenizer.vocab_size(), 8);
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
