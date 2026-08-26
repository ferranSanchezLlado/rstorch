//! `TinyShakespeare` as a batch-level [`Dataset`] of causal-LM windows: the
//! tensor-side wrapper over the raw [`TinyShakespeare`] corpus.
//!
//! The hub downloads and verifies one text file; this module tokenizes it once with
//! a [`CharTokenizer`], uploads the ids as a single `[tokens]` `I64` tensor, and
//! turns a batch of positions into the `(inputs, targets)` pair a decoder-only
//! LM trains on: `targets` is `inputs` shifted one token left, so predicting
//! position `t` from `0..=t` is the whole task.
//!
//! A "window" is a *view* into the one token buffer, not a copy of the corpus:
//! item `i` is `tokens[i .. i + window]` with target
//! `tokens[i + 1 ..= i + window]`, and consecutive items overlap. That is why
//! [`len`](Dataset::len) is `tokens − window` rather than a division: every
//! offset is a training example.

use super::DatasetHub;
use super::tiny_shakespeare::TinyShakespeare;
use crate::data::Dataset;
use crate::device::Device;
use crate::error::{Error, Result};
use crate::tensor::Tensor;
use crate::text::CharTokenizer;

/// Character-level causal-LM windows over a text corpus.
///
/// Built from the `TinyShakespeare` cache ([`from_cache`](Self::from_cache), or
/// `load`, which the `hub` feature adds) or from any text at all
/// ([`from_text`](Self::from_text) — what the tests and small examples use).
/// The tokenizer is built from the corpus it is given and kept, so
/// [`vocab_size`](Self::vocab_size) is the embedding size a model needs and
/// [`tokenizer`](Self::tokenizer) can decode a sample back to text.
///
/// The corpus is encoded **without** special tokens: it is one continuous
/// stream, and a `bos`/`eos` pair around the whole works of Shakespeare would
/// only teach the model about two positions.
///
/// # Example
///
/// ```
/// use rstorch::data::Dataset;
/// use rstorch::data::hub::TinyShakespeareDataset;
/// use rstorch::{DType, Device};
///
/// # fn main() -> rstorch::Result<()> {
/// let ds = TinyShakespeareDataset::from_text("abcabc", 3, &Device::Cpu)?;
/// assert_eq!(ds.len(), 3); // starts 0, 1, 2 — one window per offset
/// assert_eq!(ds.window(), 3);
///
/// let (x, y) = ds.batch(&[0, 2])?;
/// assert_eq!(x.dims(), &[2, 3]);
/// assert_eq!(x.dtype(), DType::I64); // token ids index an embedding
///
/// // The target is the input shifted one step left.
/// let ids: Vec<usize> = x.to_vec::<i64>()?.iter().map(|&i| i as usize).collect();
/// assert_eq!(ds.tokenizer().decode(&ids[..3])?, "abc");
/// let next: Vec<usize> = y.to_vec::<i64>()?.iter().map(|&i| i as usize).collect();
/// assert_eq!(ds.tokenizer().decode(&next[..3])?, "bca");
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct TinyShakespeareDataset {
    /// The whole corpus as one `[tokens]` `I64` tensor on the target device.
    tokens: Tensor,
    window: usize,
    tokenizer: CharTokenizer,
}

impl TinyShakespeareDataset {
    /// Tokenizes `text` with a [`CharTokenizer`] built from it and uploads the
    /// ids to `device`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArg`] (`TinyShakespeareDataset::from_text`) for
    /// `window == 0` (a window of no tokens predicts nothing), and
    /// [`Error::Data`] if the corpus holds `window` tokens or fewer: the last
    /// window needs one further token to be its target, so a corpus that short
    /// has no examples at all and is a mistake rather than an empty dataset.
    pub fn from_text(text: &str, window: usize, device: &Device) -> Result<TinyShakespeareDataset> {
        const OP: &str = "TinyShakespeareDataset::from_text";
        if window == 0 {
            return Err(Error::InvalidArg {
                op: OP,
                msg: "a causal-LM window must hold at least one token".to_string(),
            });
        }
        let tokenizer = CharTokenizer::from_text(text);
        let ids = tokenizer.encode(text, false)?;
        if ids.len() <= window {
            return Err(Error::data(format!(
                "corpus of {} tokens is too short for a window of {window}: \
                     a window needs one more token as its target",
                ids.len()
            )));
        }
        let values: Vec<i64> = ids
            .iter()
            .map(|&id| {
                i64::try_from(id).map_err(|_| Error::InvalidArg {
                    op: OP,
                    msg: format!("token id {id} does not fit in i64"),
                })
            })
            .collect::<Result<_>>()?;
        let tokens = Tensor::from_vec(values, [ids.len()], device)?;
        Ok(TinyShakespeareDataset {
            tokens,
            window,
            tokenizer,
        })
    }

    /// Builds the dataset from the corpus **already** in `hub`'s cache, with no
    /// network access at all.
    ///
    /// This is the offline entry point: pair it with `TinyShakespeare::download`
    /// (the `hub` feature) or a hand-populated cache when the download and the
    /// training run are separate steps.
    ///
    /// # Errors
    ///
    /// [`Error::Io`] if the corpus has not been cached, [`Error::Data`] if the
    /// cached resource fails its size or checksum verification, plus anything
    /// [`from_text`](Self::from_text) reports.
    pub fn from_cache(
        hub: &DatasetHub,
        window: usize,
        device: &Device,
    ) -> Result<TinyShakespeareDataset> {
        let text = TinyShakespeare::read_cached_text(hub)?;
        TinyShakespeareDataset::from_text(&text, window, device)
    }

    /// Builds the dataset, downloading and verifying the corpus into `hub`'s
    /// cache first if it is missing.
    ///
    /// Requires the `hub` feature (network access).
    ///
    /// # Errors
    ///
    /// Anything [`TinyShakespeare::load_text`] reports (download, size cap,
    /// checksum) or [`from_text`](Self::from_text) reports.
    #[cfg(feature = "hub")]
    pub fn load(
        hub: &DatasetHub,
        window: usize,
        device: &Device,
    ) -> Result<TinyShakespeareDataset> {
        let text = TinyShakespeare::load_text(hub)?;
        TinyShakespeareDataset::from_text(&text, window, device)
    }

    /// The whole corpus as one `[tokens]` `I64` tensor.
    pub fn tokens(&self) -> &Tensor {
        &self.tokens
    }

    /// The number of tokens in the corpus (**not** the number of windows; that
    /// is [`len`](Dataset::len)).
    pub fn num_tokens(&self) -> usize {
        self.tokens.dims()[0]
    }

    /// The context length of one window.
    pub fn window(&self) -> usize {
        self.window
    }

    /// The tokenizer built from this corpus — the one that can decode a batch
    /// or a sample back to text.
    pub fn tokenizer(&self) -> &CharTokenizer {
        &self.tokenizer
    }

    /// The number of distinct token ids, including the four special tokens: the
    /// vocabulary an [`Embedding`](crate::nn::Embedding) and the output
    /// projection must both be sized for.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }
}

impl Dataset for TinyShakespeareDataset {
    type Batch = (Tensor, Tensor);

    /// One window per start offset. The constructors reject a corpus of
    /// `window` tokens or fewer, so this subtraction cannot underflow.
    fn len(&self) -> usize {
        self.num_tokens() - self.window
    }

    /// Collates one window per position: each index `i` narrows
    /// `tokens[i ..= i + window]` out of the corpus tensor, the rows are
    /// stacked into `[batch, window + 1]`, and the pair is that block's first
    /// and last `window` columns — so the shift is two views, not a second
    /// gather.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArg`] for an empty index slice and
    /// [`Error::IndexOutOfBounds`] for a start position at or past
    /// [`len`](Dataset::len) (both `op:
    /// "TinyShakespeareDataset::batch"`), plus anything the narrows or the
    /// stack report.
    fn batch(&self, indices: &[usize]) -> Result<(Tensor, Tensor)> {
        const OP: &str = "TinyShakespeareDataset::batch";
        let len = self.len();
        if indices.is_empty() {
            return Err(Error::InvalidArg {
                op: OP,
                msg: "an empty index slice has no batch; ask for at least one window".to_string(),
            });
        }
        for &index in indices {
            if index >= len {
                return Err(Error::IndexOutOfBounds {
                    op: OP,
                    index: i64::try_from(index).unwrap_or(i64::MAX),
                    axis: 0,
                    size: len,
                });
            }
        }

        let windows: Vec<Tensor> = indices
            .iter()
            .map(|&start| self.tokens.narrow(0, start, self.window + 1))
            .collect::<Result<_>>()?;
        let refs: Vec<&Tensor> = windows.iter().collect();
        let rows = Tensor::stack(&refs, 0)?;
        Ok((
            rows.narrow(1, 0, self.window)?,
            rows.narrow(1, 1, self.window)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataLoader;
    use crate::dtype::DType;
    use crate::testing::check_grad;
    use std::path::PathBuf;

    const CPU: Device = Device::Cpu;

    fn scratch(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "rstorch-ts-dataset-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    /// The ids of `text` under the dataset's own tokenizer, as `i64`.
    fn expected_ids(ds: &TinyShakespeareDataset, text: &str) -> Vec<i64> {
        ds.tokenizer()
            .encode(text, false)
            .unwrap()
            .iter()
            .map(|&id| id as i64)
            .collect()
    }

    #[test]
    fn windows_are_every_offset_and_targets_are_shifted_by_one() {
        let ds = TinyShakespeareDataset::from_text("abcdef", 4, &CPU).unwrap();

        assert_eq!(ds.num_tokens(), 6);
        assert_eq!(ds.window(), 4);
        assert_eq!(ds.len(), 2); // starts 0 and 1
        assert!(!ds.is_empty());
        assert_eq!(ds.tokens().dtype(), DType::I64);
        assert_eq!(ds.tokens().dims(), &[6]);

        let (x, y) = ds.batch(&[0, 1]).unwrap();
        assert_eq!(x.dims(), &[2, 4]);
        assert_eq!(y.dims(), &[2, 4]);
        let mut want_x = expected_ids(&ds, "abcd");
        want_x.extend(expected_ids(&ds, "bcde"));
        let mut want_y = expected_ids(&ds, "bcde");
        want_y.extend(expected_ids(&ds, "cdef"));
        assert_eq!(x.to_vec::<i64>().unwrap(), want_x);
        assert_eq!(y.to_vec::<i64>().unwrap(), want_y);
    }

    #[test]
    fn a_batch_follows_the_index_slice_including_repeats() {
        let ds = TinyShakespeareDataset::from_text("abcabcabc", 2, &CPU).unwrap();
        let (x, y) = ds.batch(&[6, 0, 6]).unwrap();
        assert_eq!(x.dims(), &[3, 2]);

        let row = expected_ids(&ds, "ab");
        let want_x: Vec<i64> = [&row[..], &row[..], &row[..]].concat();
        assert_eq!(x.to_vec::<i64>().unwrap(), want_x);
        let shifted = expected_ids(&ds, "bc");
        let want_y: Vec<i64> = [&shifted[..], &shifted[..], &shifted[..]].concat();
        assert_eq!(y.to_vec::<i64>().unwrap(), want_y);
    }

    #[test]
    fn the_tokenizer_is_built_from_the_corpus() {
        let ds = TinyShakespeareDataset::from_text("aab", 1, &CPU).unwrap();
        // Two distinct characters plus the four reserved special ids.
        assert_eq!(ds.vocab_size(), 6);
        assert_eq!(ds.tokenizer().vocab_size(), ds.vocab_size());
        // Round-tripping a whole window is what a sampler does after decoding.
        let ids: Vec<usize> = ds
            .batch(&[0])
            .unwrap()
            .0
            .to_vec::<i64>()
            .unwrap()
            .iter()
            .map(|&id| id as usize)
            .collect();
        assert_eq!(ds.tokenizer().decode(&ids).unwrap(), "a");
    }

    #[test]
    fn a_zero_window_is_rejected() {
        assert!(matches!(
            TinyShakespeareDataset::from_text("abc", 0, &CPU),
            Err(Error::InvalidArg {
                op: "TinyShakespeareDataset::from_text",
                ..
            })
        ));
    }

    #[test]
    fn a_corpus_too_short_for_one_window_is_rejected() {
        // Exactly `window` tokens: the last window has no target token.
        assert!(matches!(
            TinyShakespeareDataset::from_text("abc", 3, &CPU),
            Err(Error::Data { .. })
        ));
        assert!(matches!(
            TinyShakespeareDataset::from_text("", 1, &CPU),
            Err(Error::Data { .. })
        ));
        // One more token and it is a dataset of exactly one window.
        assert_eq!(
            TinyShakespeareDataset::from_text("abcd", 3, &CPU)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn empty_and_out_of_range_index_slices_are_rejected() {
        let ds = TinyShakespeareDataset::from_text("abcd", 3, &CPU).unwrap();
        assert_eq!(ds.len(), 1);
        assert!(matches!(
            ds.batch(&[]),
            Err(Error::InvalidArg {
                op: "TinyShakespeareDataset::batch",
                ..
            })
        ));
        // Position 1 would need a target token past the end of the corpus.
        assert!(matches!(
            ds.batch(&[1]),
            Err(Error::IndexOutOfBounds {
                op: "TinyShakespeareDataset::batch",
                index: 1,
                axis: 0,
                size: 1,
            })
        ));
    }

    #[test]
    fn from_cache_rejects_unverified_bytes_without_network_access() {
        let root = scratch("cache");
        let _ = std::fs::remove_dir_all(&root);
        let hub = DatasetHub::new(&root);
        let path = TinyShakespeare::cache_path(&hub).unwrap();
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, "To be, or not to be").unwrap();

        assert!(matches!(
            TinyShakespeareDataset::from_cache(&hub, 4, &CPU),
            Err(Error::Data { msg, .. }) if msg.contains("checksum mismatch")
        ));

        let _ = std::fs::remove_dir_all(root);
    }

    /// A corpus drops straight into the loader: every window shape is
    /// `[batch, window]` on both halves, and one epoch covers every offset.
    #[test]
    fn a_corpus_drives_a_data_loader() {
        let ds = TinyShakespeareDataset::from_text("abcabcabca", 3, &CPU).unwrap();
        assert_eq!(ds.len(), 7);
        let loader = DataLoader::new(&ds, 3).shuffle(11);

        assert_eq!(loader.num_batches(), 3); // 3 + 3 + 1
        let mut seen = 0;
        for batch in loader.batches() {
            let (x, y) = batch.unwrap();
            assert_eq!(x.dims()[1], 3);
            assert_eq!(y.dims(), x.dims());
            assert_eq!(x.dtype(), DType::I64);
            assert_eq!(y.dtype(), DType::I64);
            seen += x.dims()[0];
        }
        assert_eq!(seen, 7);
    }

    #[test]
    fn from_cache_without_a_cached_corpus_is_an_io_error() {
        let hub = DatasetHub::new(scratch("missing"));
        assert!(matches!(
            TinyShakespeareDataset::from_cache(&hub, 2, &CPU),
            Err(Error::Io(_))
        ));
    }

    /// The batch's token ids are usable as real indices into a traced
    /// embedding table: the gradient of the gathered rows is checked
    /// numerically, which is the property an LM training step depends on (and
    /// would fail if the ids were `F32`, or off by the special-token offset).
    #[test]
    fn a_batch_indexes_a_traced_embedding_table() {
        let ds = TinyShakespeareDataset::from_text("abab", 2, &CPU).unwrap();
        let (x, _) = ds.batch(&[0, 1]).unwrap();
        let flat = x.reshape([4]).unwrap();
        let table: Vec<f64> = (0..ds.vocab_size() * 2).map(|v| v as f64 * 0.5).collect();
        let table = Tensor::from_vec(table, [ds.vocab_size(), 2], &CPU).unwrap();

        check_grad(
            |inputs| inputs[0].index_select(0, &flat)?.sum_all(),
            &[table],
            1e-4,
            1e-6,
        )
        .unwrap();
    }
}
