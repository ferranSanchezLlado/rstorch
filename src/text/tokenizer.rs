//! Character-level and byte-pair-encoding tokenizers.
//!
//! Tokenization is strict by default, with one explicit exception: the
//! character tokenizer maps out-of-vocabulary characters to its `unk` token.
//! BPE encoding/decoding and invalid token ids remain **loud rather than
//! lossy** — unsupported input is an [`Error::Tokenizer`] instead of an
//! accidental substitution:
//!
//! - [`encode`](Tokenizer::encode) and [`decode`](Tokenizer::decode) return
//!   [`Result`]. An id that maps to nothing is an error; see
//!   [`CharTokenizer::encode`] for its documented `unk` exception.
//! - Decoding is **strict UTF-8**: a byte sequence that is not valid UTF-8 is
//!   an error, never a `U+FFFD` replacement.
//! - [`BpeTokenizer::train`] **validates** the requested vocabulary size
//!   rather than silently clamping it up to the byte-plus-specials floor.
//!
//! Token ids are plain `usize` values.

use std::collections::{BTreeSet, HashMap};

use crate::{Error, Result};

/// Number of reserved special-token ids (`pad`, `unk`, `bos`, `eos`).
const SPECIALS: usize = 4;
/// Number of single-byte base tokens the BPE vocabulary always contains.
const BYTE_BASE: usize = 256;

/// A reversible mapping between text and integer token ids.
///
/// [`encode`](Self::encode) and [`decode`](Self::decode) are fallible: an id
/// that maps to nothing, or a byte sequence that is not valid UTF-8, is
/// reported through the crate [`Error`] rather than being silently skipped.
/// Implementations may define an explicit unknown-token policy; for example,
/// [`CharTokenizer`] maps out-of-vocabulary characters to `unk` and decodes
/// that token as `?`.
///
/// # Examples
///
/// ```
/// use rstorch::text::{CharTokenizer, Tokenizer};
///
/// let tok = CharTokenizer::from_text("abc");
/// let ids = tok.encode("cab", false)?;
/// assert_eq!(tok.decode(&ids)?, "cab");
/// # Ok::<(), rstorch::Error>(())
/// ```
pub trait Tokenizer {
    /// Encodes text into token ids.
    ///
    /// When `add_special_tokens` is set the implementation may wrap the
    /// output in its boundary tokens (typically `bos`/`eos`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Tokenizer`] when the input
    /// cannot be represented in this tokenizer's vocabulary.
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<usize>>;

    /// Decodes token ids back into text.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Tokenizer`] when an id does
    /// not map to any piece, or when the decoded bytes are not valid UTF-8.
    fn decode(&self, ids: &[usize]) -> Result<String>;

    /// The number of distinct token ids this tokenizer can produce.
    fn vocab_size(&self) -> usize;

    /// The padding token id, if this tokenizer defines one.
    fn pad_id(&self) -> Option<usize> {
        None
    }

    /// The unknown-token id, if this tokenizer defines one.
    fn unk_id(&self) -> Option<usize> {
        None
    }

    /// The beginning-of-sequence token id, if this tokenizer defines one.
    fn bos_id(&self) -> Option<usize> {
        None
    }

    /// The end-of-sequence token id, if this tokenizer defines one.
    fn eos_id(&self) -> Option<usize> {
        None
    }
}

/// A character-level tokenizer built from the distinct characters of a corpus.
///
/// Ids `0..4` are reserved for the `pad`, `unk`, `bos`, and `eos` special
/// tokens; corpus characters occupy ids `4..4 + n` in sorted order.
/// Characters not present in the corpus are encoded as `unk` and therefore do
/// not round-trip losslessly: decoding them yields `?`.
///
/// # Examples
///
/// ```
/// use rstorch::text::{CharTokenizer, Tokenizer};
///
/// let tok = CharTokenizer::from_text("hello");
/// let ids = tok.encode("hello", true)?;
/// assert_eq!(ids.first(), tok.bos_id().as_ref());
/// assert_eq!(tok.decode(&ids)?, "hello"); // bos/eos are dropped on decode
/// # Ok::<(), rstorch::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct CharTokenizer {
    /// Distinct corpus characters, sorted; index `i` maps to id `i + SPECIALS`.
    chars: Vec<char>,
    /// Reverse map from character to token id.
    ids: HashMap<char, usize>,
    pad_id: usize,
    unk_id: usize,
    bos_id: usize,
    eos_id: usize,
}

impl CharTokenizer {
    /// Builds a tokenizer from the distinct characters appearing in `text`.
    ///
    /// Characters are assigned ids in sorted order starting after the four
    /// reserved special-token ids (`pad`, `unk`, `bos`, `eos`).
    #[must_use]
    pub fn from_text(text: &str) -> Self {
        let mut set = BTreeSet::new();
        for ch in text.chars() {
            set.insert(ch);
        }
        let chars: Vec<_> = set.into_iter().collect();
        let ids = chars
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, ch)| (ch, idx + SPECIALS))
            .collect();
        Self {
            chars,
            ids,
            pad_id: 0,
            unk_id: 1,
            bos_id: 2,
            eos_id: 3,
        }
    }

    /// The number of distinct token ids, including the four special tokens.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.chars.len() + SPECIALS
    }

    /// The padding token id.
    #[must_use]
    pub fn pad_id(&self) -> Option<usize> {
        Some(self.pad_id)
    }

    /// The unknown-token id.
    #[must_use]
    pub fn unk_id(&self) -> Option<usize> {
        Some(self.unk_id)
    }

    /// The beginning-of-sequence token id.
    #[must_use]
    pub fn bos_id(&self) -> Option<usize> {
        Some(self.bos_id)
    }

    /// The end-of-sequence token id.
    #[must_use]
    pub fn eos_id(&self) -> Option<usize> {
        Some(self.eos_id)
    }

    /// Encodes `text`, mapping unknown characters to the `unk` token.
    ///
    /// This encoder is total over Unicode text: characters absent from the
    /// vocabulary become [`unk_id`](Self::unk_id) rather than an error, so a
    /// [`Result`] is returned only for signature parity with the fallible
    /// [`Tokenizer`] trait.
    ///
    /// # Errors
    ///
    /// This inherent method never fails; it returns [`Result`] to match the
    /// [`Tokenizer`] trait surface.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<usize>> {
        let mut out =
            Vec::with_capacity(text.chars().count() + usize::from(add_special_tokens) * 2);
        if add_special_tokens {
            out.push(self.bos_id);
        }
        out.extend(
            text.chars()
                .map(|ch| self.ids.get(&ch).copied().unwrap_or(self.unk_id)),
        );
        if add_special_tokens {
            out.push(self.eos_id);
        }
        Ok(out)
    }

    /// Decodes token ids back into text.
    ///
    /// The `pad`, `bos`, and `eos` special tokens are dropped; the `unk`
    /// token decodes to `?`. Any other id must correspond to a known
    /// character.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Tokenizer`] if an id is not a
    /// special token and does not map to a character in the vocabulary. This
    /// is an error rather than a silently skipped token.
    pub fn decode(&self, ids: &[usize]) -> Result<String> {
        let mut out = String::new();
        for &id in ids {
            if id == self.pad_id || id == self.bos_id || id == self.eos_id {
                continue;
            }
            if id == self.unk_id {
                out.push('?');
            } else if let Some(&ch) = self.chars.get(id.saturating_sub(SPECIALS)) {
                out.push(ch);
            } else {
                return Err(Error::tokenizer(format!(
                    "unknown token id {id} (vocab size {})",
                    self.vocab_size()
                )));
            }
        }
        Ok(out)
    }
}

impl Tokenizer for CharTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<usize>> {
        CharTokenizer::encode(self, text, add_special_tokens)
    }

    fn decode(&self, ids: &[usize]) -> Result<String> {
        CharTokenizer::decode(self, ids)
    }

    fn vocab_size(&self) -> usize {
        CharTokenizer::vocab_size(self)
    }

    fn pad_id(&self) -> Option<usize> {
        CharTokenizer::pad_id(self)
    }

    fn unk_id(&self) -> Option<usize> {
        CharTokenizer::unk_id(self)
    }

    fn bos_id(&self) -> Option<usize> {
        CharTokenizer::bos_id(self)
    }

    fn eos_id(&self) -> Option<usize> {
        CharTokenizer::eos_id(self)
    }
}

/// A byte-level byte-pair-encoding tokenizer.
///
/// The vocabulary always contains the four special tokens and the 256
/// single-byte tokens; learned merges extend it up to the requested target
/// size.
///
/// # Examples
///
/// ```
/// use rstorch::text::{BpeTokenizer, Tokenizer};
///
/// let tok = BpeTokenizer::train("hello hello hello", 260)?;
/// let ids = tok.encode("hello", false)?;
/// assert_eq!(tok.decode(&ids)?, "hello");
/// # Ok::<(), rstorch::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// Learned merges in rank order; entry `r` merges its pair into id
    /// `SPECIALS + BYTE_BASE + r`.
    merges: Vec<(usize, usize)>,
    /// Byte content of every token, indexed by id.
    token_bytes: Vec<Vec<u8>>,
    /// Map from a merge pair to the id of its merged token.
    pair_to_id: HashMap<(usize, usize), usize>,
}

impl BpeTokenizer {
    /// Trains a BPE tokenizer over `corpus`, learning merges until the
    /// vocabulary reaches `target_vocab` or no frequent pair remains.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Tokenizer`] if `target_vocab` is smaller than the
    /// mandatory floor of 260 tokens (4 special tokens + 256 byte tokens) —
    /// a request the vocabulary cannot honor is rejected, not clamped.
    pub fn train(corpus: &str, target_vocab: usize) -> Result<Self> {
        let floor = SPECIALS + BYTE_BASE;
        if target_vocab < floor {
            return Err(Error::tokenizer(format!(
                "target vocabulary size {target_vocab} is below the minimum of {floor} \
                 ({SPECIALS} special tokens + {BYTE_BASE} byte tokens)"
            )));
        }

        let mut token_bytes = initial_token_bytes();
        let mut symbols: Vec<usize> = corpus.bytes().map(byte_id).collect();
        let mut merges = Vec::new();

        while token_bytes.len() < target_vocab {
            let Some(pair) = most_frequent_pair(&symbols) else {
                break;
            };
            let merged_id = token_bytes.len();
            let mut bytes = token_bytes[pair.0].clone();
            bytes.extend_from_slice(&token_bytes[pair.1]);
            token_bytes.push(bytes);
            merges.push(pair);
            symbols = apply_pair_merge(&symbols, pair, merged_id);
        }

        let pair_to_id = merges
            .iter()
            .copied()
            .enumerate()
            .map(|(rank, pair)| (pair, SPECIALS + BYTE_BASE + rank))
            .collect();
        Ok(Self {
            merges,
            token_bytes,
            pair_to_id,
        })
    }

    /// The learned merges in rank order.
    #[must_use]
    pub fn merges(&self) -> &[(usize, usize)] {
        &self.merges
    }
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<usize>> {
        let mut out = Vec::new();
        if add_special_tokens && let Some(bos_id) = self.bos_id() {
            out.push(bos_id);
        }
        let mut symbols: Vec<usize> = text.bytes().map(byte_id).collect();
        for &pair in &self.merges {
            let merged_id = self.pair_to_id[&pair];
            symbols = apply_pair_merge(&symbols, pair, merged_id);
        }
        out.extend(symbols);
        if add_special_tokens && let Some(eos_id) = self.eos_id() {
            out.push(eos_id);
        }
        Ok(out)
    }

    fn decode(&self, ids: &[usize]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            if Some(id) == self.pad_id() || Some(id) == self.bos_id() || Some(id) == self.eos_id() {
                continue;
            }
            let Some(piece) = self.token_bytes.get(id) else {
                return Err(Error::tokenizer(format!(
                    "unknown token id {id} (vocab size {})",
                    self.vocab_size()
                )));
            };
            bytes.extend_from_slice(piece);
        }
        // Strict UTF-8: an invalid byte sequence is an error, never a
        // `U+FFFD` replacement.
        String::from_utf8(bytes).map_err(|err| {
            Error::tokenizer_with(
                format!(
                    "decoded bytes are not valid UTF-8 at offset {}",
                    err.utf8_error().valid_up_to()
                ),
                err,
            )
        })
    }

    fn vocab_size(&self) -> usize {
        self.token_bytes.len()
    }

    fn pad_id(&self) -> Option<usize> {
        Some(0)
    }

    fn unk_id(&self) -> Option<usize> {
        Some(1)
    }

    fn bos_id(&self) -> Option<usize> {
        Some(2)
    }

    fn eos_id(&self) -> Option<usize> {
        Some(3)
    }
}

/// The initial per-id byte content: four specials followed by all 256 bytes.
fn initial_token_bytes() -> Vec<Vec<u8>> {
    let mut out = vec![Vec::new(), b"?".to_vec(), Vec::new(), Vec::new()];
    out.extend((0u8..=255).map(|byte| vec![byte]));
    out
}

/// Maps a raw byte to its base token id.
fn byte_id(byte: u8) -> usize {
    SPECIALS + byte as usize
}

/// Returns the most frequent adjacent symbol pair (count `>= 2`), breaking
/// ties deterministically by preferring the lexicographically smaller pair.
fn most_frequent_pair(symbols: &[usize]) -> Option<(usize, usize)> {
    let mut counts = HashMap::new();
    for pair in symbols.windows(2) {
        *counts.entry((pair[0], pair[1])).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .filter(|(_, count)| *count >= 2)
        .max_by(|(left_pair, left_count), (right_pair, right_count)| {
            left_count
                .cmp(right_count)
                .then_with(|| right_pair.cmp(left_pair))
        })
        .map(|(pair, _)| pair)
}

/// Replaces every non-overlapping occurrence of `pair` in `symbols` with
/// `merged_id`.
fn apply_pair_merge(symbols: &[usize], pair: (usize, usize), merged_id: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(symbols.len());
    let mut idx = 0;
    while idx < symbols.len() {
        if idx + 1 < symbols.len() && (symbols[idx], symbols[idx + 1]) == pair {
            out.push(merged_id);
            idx += 2;
        } else {
            out.push(symbols[idx]);
            idx += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(
            tokenizer
                .decode(&tokenizer.encode("abab", true).unwrap())
                .unwrap(),
            "abab"
        );

        let again = BpeTokenizer::train("abababab", 256 + 4 + 2).unwrap();
        assert_eq!(tokenizer.merges(), again.merges());
    }

    #[test]
    fn tokenizer_special_tokens_default_to_none() {
        struct NoSpecials;

        impl Tokenizer for NoSpecials {
            fn encode(&self, _text: &str, _add_special_tokens: bool) -> Result<Vec<usize>> {
                Ok(Vec::new())
            }

            fn decode(&self, _ids: &[usize]) -> Result<String> {
                Ok(String::new())
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

    #[test]
    fn char_unknown_char_encodes_to_unk_and_decodes_to_question_mark() {
        // Vocabulary built without 'z'; encoding 'z' yields the unk token.
        let tokenizer = CharTokenizer::from_text("abc");
        let ids = tokenizer.encode("z", false).unwrap();
        assert_eq!(ids, vec![tokenizer.unk_id]);
        assert_eq!(tokenizer.decode(&ids).unwrap(), "?");
    }

    #[test]
    fn char_decode_rejects_unknown_token_id() {
        let tokenizer = CharTokenizer::from_text("abc");
        // vocab_size is 7 (3 chars + 4 specials); id 100 is out of range.
        let err = tokenizer.decode(&[100]).unwrap_err();
        assert!(matches!(err, Error::Tokenizer { .. }));
        assert!(err.to_string().contains("unknown token id 100"));
    }

    #[test]
    fn bpe_decode_rejects_unknown_token_id() {
        let tokenizer = BpeTokenizer::train("hello world", 260).unwrap();
        let bad = tokenizer.vocab_size() + 5;
        let err = tokenizer.decode(&[bad]).unwrap_err();
        assert!(matches!(err, Error::Tokenizer { .. }));
        assert!(err.to_string().contains("unknown token id"));
    }

    #[test]
    fn bpe_decode_rejects_invalid_utf8() {
        // Train with no merges, so single-byte tokens map 1:1 to their bytes.
        let tokenizer = BpeTokenizer::train("", 260).unwrap();
        // 0xFF is never a valid standalone UTF-8 byte.
        let lone_continuation = byte_id(0xFF);
        let err = tokenizer.decode(&[lone_continuation]).unwrap_err();
        assert!(matches!(err, Error::Tokenizer { .. }));
        assert!(err.to_string().contains("not valid UTF-8"));
    }

    #[test]
    fn bpe_decode_accepts_valid_multibyte_utf8() {
        // A round trip through the byte tokenizer must preserve non-ASCII text.
        let tokenizer = BpeTokenizer::train("", 260).unwrap();
        let ids = tokenizer.encode("héllo — 世界", false).unwrap();
        assert_eq!(tokenizer.decode(&ids).unwrap(), "héllo — 世界");
    }

    #[test]
    fn bpe_train_rejects_target_vocab_below_floor() {
        let err = BpeTokenizer::train("abababab", 100).unwrap_err();
        assert!(matches!(err, Error::Tokenizer { .. }));
        let msg = err.to_string();
        assert!(msg.contains("below the minimum of 260"));
    }

    #[test]
    fn bpe_train_accepts_exact_floor() {
        // The floor itself is valid and learns no merges.
        let tokenizer = BpeTokenizer::train("abababab", 260).unwrap();
        assert_eq!(tokenizer.vocab_size(), 260);
        assert!(tokenizer.merges().is_empty());
    }
}
