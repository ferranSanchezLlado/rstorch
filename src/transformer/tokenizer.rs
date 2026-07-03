use std::collections::{BTreeSet, HashMap};

const SPECIALS: usize = 4;
const BYTE_BASE: usize = 256;

pub trait Tokenizer {
    /// Encodes text into token ids.
    ///
    /// Implementations keep this infallible. Unknown input should be substituted
    /// with an implementation-defined unknown token when one exists, or encoded
    /// by the tokenizer's native fallback strategy.
    fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<usize>;

    /// Decodes token ids into text.
    ///
    /// Implementations keep this infallible and may be lossy for invalid byte
    /// sequences or ids that do not map to text.
    fn decode(&self, ids: &[usize]) -> String;
    fn vocab_size(&self) -> usize;
    fn pad_id(&self) -> Option<usize> {
        None
    }
    fn unk_id(&self) -> Option<usize> {
        None
    }
    fn bos_id(&self) -> Option<usize> {
        None
    }
    fn eos_id(&self) -> Option<usize> {
        None
    }
}

#[derive(Debug, Clone)]
pub struct CharTokenizer {
    chars: Vec<char>,
    ids: HashMap<char, usize>,
    pad_id: usize,
    unk_id: usize,
    bos_id: usize,
    eos_id: usize,
}

impl CharTokenizer {
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
            .map(|(idx, ch)| (ch, idx + 4))
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

    pub fn vocab_size(&self) -> usize {
        self.chars.len() + 4
    }

    pub fn pad_id(&self) -> Option<usize> {
        Some(self.pad_id)
    }

    pub fn unk_id(&self) -> Option<usize> {
        Some(self.unk_id)
    }

    pub fn bos_id(&self) -> Option<usize> {
        Some(self.bos_id)
    }

    pub fn eos_id(&self) -> Option<usize> {
        Some(self.eos_id)
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<usize> {
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
        out
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let mut out = String::new();
        for &id in ids {
            if id == self.pad_id || id == self.bos_id || id == self.eos_id {
                continue;
            }
            if id == self.unk_id {
                out.push('?');
            } else if let Some(&ch) = self.chars.get(id.saturating_sub(4)) {
                out.push(ch);
            }
        }
        out
    }
}

impl Tokenizer for CharTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<usize> {
        CharTokenizer::encode(self, text, add_special_tokens)
    }

    fn decode(&self, ids: &[usize]) -> String {
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

#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    merges: Vec<(usize, usize)>,
    token_bytes: Vec<Vec<u8>>,
    pair_to_id: HashMap<(usize, usize), usize>,
}

impl BpeTokenizer {
    pub fn train(corpus: &str, target_vocab: usize) -> Self {
        let target_vocab = target_vocab.max(SPECIALS + BYTE_BASE);
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
        Self {
            merges,
            token_bytes,
            pair_to_id,
        }
    }

    pub fn merges(&self) -> &[(usize, usize)] {
        &self.merges
    }
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<usize> {
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
        out
    }

    fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if Some(id) == self.pad_id() || Some(id) == self.bos_id() || Some(id) == self.eos_id() {
                continue;
            }
            if let Some(piece) = self.token_bytes.get(id) {
                bytes.extend_from_slice(piece);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
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

fn initial_token_bytes() -> Vec<Vec<u8>> {
    let mut out = vec![Vec::new(), b"?".to_vec(), Vec::new(), Vec::new()];
    out.extend((0u8..=255).map(|byte| vec![byte]));
    out
}

fn byte_id(byte: u8) -> usize {
    SPECIALS + byte as usize
}

fn most_frequent_pair(symbols: &[usize]) -> Option<(usize, usize)> {
    let mut counts = HashMap::<(usize, usize), usize>::new();
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
