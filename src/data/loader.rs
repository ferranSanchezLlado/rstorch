//! [`DataLoader`]: the one loader — order and batching, and
//! nothing else.
//!
//! v2 shipped `DataLoader` *and* `StaticDataLoader`, three collators in two
//! batch-axis flavors, eight batch aliases and a `Sampler` trait, because a
//! batch's shape was part of its type. Here the batch is whatever the
//! [`Dataset`] says it is, so the loader's entire job is choosing item
//! positions and grouping them: `new(ds, 64)`, optionally `.shuffle(seed)`,
//! then [`batches`](DataLoader::batches).

use crate::data::dataset::Dataset;
use crate::error::Result;
use crate::rng::Rng;

/// Batches a [`Dataset`] in order, or in a seeded shuffled order.
///
/// The crate's one loader. It owns three decisions — batch
/// size, order, and what to do with a short final batch — and delegates
/// everything about *what a batch is* to the dataset.
///
/// ```
/// use rstorch::data::{DataLoader, Dataset, VecDataset};
/// # use rstorch::{DType, Device, Result, Tensor};
/// # fn main() -> Result<()> {
/// # let dev = Device::Cpu;
/// # let items = (0..10)
/// #     .map(|i| Ok((Tensor::full([3], f64::from(i), DType::F32, &dev)?,
/// #                  Tensor::full([], f64::from(i % 2), DType::I64, &dev)?)))
/// #     .collect::<Result<Vec<_>>>()?;
/// let ds = VecDataset::new(items)?; // 10 items
///
/// for batch in DataLoader::new(&ds, 4).shuffle(7).batches() {
///     let (x, y) = batch?;
///     // 4, 4, then the 2-item tail: the last batch is kept, not padded.
///     assert_eq!(x.dims()[0], y.dims()[0]);
///     assert_eq!(x.dims()[1], 3);
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Determinism
///
/// The order is a pure function of `(seed, epoch, dataset length)`: no global
/// RNG, no interior mutability, nothing consumed by iterating. Two loaders
/// built with the same seed over datasets of the same length visit positions
/// in the same order, and [`batches`](DataLoader::batches) can be called
/// repeatedly for the same order every time. A fresh permutation per epoch is
/// [`batches_for_epoch`](DataLoader::batches_for_epoch), which is also what
/// makes an interrupted run resumable: epoch 7's order does not depend on
/// epochs 0–6 having been walked.
///
/// # Multiple loaders over one dataset
///
/// `&D` is a [`Dataset`] when `D` is, so `DataLoader::new(&ds, 64)` borrows
/// rather than moves and a train/eval pair of loaders can share one split.
#[derive(Debug, Clone)]
pub struct DataLoader<D> {
    dataset: D,
    batch_size: usize,
    /// `None` = sequential order; `Some(seed)` = shuffled from that seed.
    seed: Option<u64>,
    drop_last: bool,
}

impl<D: Dataset> DataLoader<D> {
    /// A loader over `dataset` yielding batches of `batch_size` items in
    /// dataset order.
    ///
    /// The final batch is **short, not dropped**, when the length is not a
    /// multiple of `batch_size` (PyTorch's default; see
    /// [`drop_last`](DataLoader::drop_last) to change it).
    ///
    /// # Panics
    ///
    /// If `batch_size` is `0`. A zero-item batch has no shape and would make
    /// the loader an infinite iterator over empty index slices; like
    /// [`slice::chunks`], this is treated as a programmer error rather than a
    /// runtime one, so the happy path stays free of a `Result`.
    pub fn new(dataset: D, batch_size: usize) -> DataLoader<D> {
        assert!(
            batch_size > 0,
            "DataLoader::new: batch_size must be nonzero"
        );
        DataLoader {
            dataset,
            batch_size,
            seed: None,
            drop_last: false,
        }
    }

    /// Visit positions in a shuffled order derived from `seed` (a consuming
    /// builder, like [`Sequential::push`](crate::nn::Sequential::push)).
    ///
    /// The permutation is a full Fisher–Yates shuffle of `0..len` driven by
    /// [`Rng`](crate::Rng) — every item appears exactly once per epoch, which
    /// is what training expects and what sampling *with* replacement would
    /// not give. Calling it twice keeps the last seed.
    ///
    /// ```
    /// # use rstorch::data::{DataLoader, TensorDataset};
    /// # use rstorch::{Device, Result, Tensor};
    /// # fn main() -> Result<()> {
    /// # let dev = Device::Cpu;
    /// # let x = Tensor::from_vec((0..6).map(|v| v as f32).collect::<Vec<_>>(), [6, 1], &dev)?;
    /// # let y = Tensor::from_vec(vec![0i64; 6], [6], &dev)?;
    /// let ds = TensorDataset::new(x, y)?;
    /// let ordered = DataLoader::new(&ds, 6).batches().next().unwrap()?.0;
    /// let shuffled = DataLoader::new(&ds, 6).shuffle(3).batches().next().unwrap()?.0;
    ///
    /// let mut sorted = shuffled.to_vec::<f32>()?;
    /// sorted.sort_by(f32::total_cmp);
    /// assert_eq!(sorted, ordered.to_vec::<f32>()?); // a permutation, not a sample
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn shuffle(mut self, seed: u64) -> DataLoader<D> {
        self.seed = Some(seed);
        self
    }

    /// Choose what happens to a final batch with fewer than `batch_size`
    /// items: keep it (`false`, the default) or drop it (`true`).
    ///
    /// Dropping matters when a step's cost or correctness assumes a fixed
    /// batch (a fused kernel benchmark, `BatchNorm` on a batch of one). Note
    /// that with `drop_last(true)` a dataset shorter than one batch yields
    /// **no batches at all**.
    #[must_use]
    pub fn drop_last(mut self, drop_last: bool) -> DataLoader<D> {
        self.drop_last = drop_last;
        self
    }

    /// The dataset being loaded.
    pub fn dataset(&self) -> &D {
        &self.dataset
    }

    /// The configured batch size (the size of every batch but possibly the
    /// last).
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// How many batches one epoch yields: `ceil(len / batch_size)` normally,
    /// `len / batch_size` under [`drop_last(true)`](DataLoader::drop_last).
    ///
    /// This is exactly the length reported by the
    /// [`Batches`] iterator, so it can size a progress bar or a loss buffer
    /// without walking the epoch.
    pub fn num_batches(&self) -> usize {
        let len = self.dataset.len();
        if self.drop_last {
            len / self.batch_size
        } else {
            len.div_ceil(self.batch_size)
        }
    }

    /// The batches of one epoch — the loader's main entry point.
    ///
    /// Equivalent to `batches_for_epoch(0)`. Iterating borrows the loader and
    /// consumes nothing: call it again and the same batches come back in the
    /// same order.
    pub fn batches(&self) -> Batches<'_, D> {
        self.batches_for_epoch(0)
    }

    /// The batches of epoch `epoch`, reshuffled for each epoch when a seed is
    /// set (and identical for every epoch when one is not).
    ///
    /// This is the multi-epoch loop, and it is stateless: epoch `n`'s order
    /// depends only on `(seed, n, len)`, so a run resumed from a checkpoint at
    /// epoch `n` sees exactly the order it would have seen without the
    /// interruption.
    ///
    /// ```
    /// # use rstorch::data::{DataLoader, TensorDataset};
    /// # use rstorch::{Device, Result, Tensor};
    /// # fn main() -> Result<()> {
    /// # let dev = Device::Cpu;
    /// # let x = Tensor::from_vec((0..8).map(|v| v as f32).collect::<Vec<_>>(), [8, 1], &dev)?;
    /// # let y = Tensor::from_vec(vec![0i64; 8], [8], &dev)?;
    /// let loader = DataLoader::new(TensorDataset::new(x, y)?, 8).shuffle(0);
    /// let first = loader.batches_for_epoch(0).next().unwrap()?.0.to_vec::<f32>()?;
    /// let second = loader.batches_for_epoch(1).next().unwrap()?.0.to_vec::<f32>()?;
    ///
    /// assert_ne!(first, second);                                  // reshuffled
    /// assert_eq!(first, loader.batches().next().unwrap()?.0.to_vec::<f32>()?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn batches_for_epoch(&self, epoch: u64) -> Batches<'_, D> {
        let order = self.order(epoch);
        // With `drop_last`, stop at the last whole batch instead of trimming
        // the order itself, so the dropped positions stay inspectable.
        let end = if self.drop_last {
            self.num_batches() * self.batch_size
        } else {
            order.len()
        };
        Batches {
            dataset: &self.dataset,
            order,
            batch_size: self.batch_size,
            next: 0,
            end,
        }
    }

    /// The item positions of one epoch, in visiting order: `0..len` when
    /// unshuffled, a Fisher–Yates permutation of it when a seed is set.
    fn order(&self, epoch: u64) -> Vec<usize> {
        let len = self.dataset.len();
        let mut order: Vec<usize> = (0..len).collect();
        if let Some(seed) = self.seed {
            let mut rng = Rng::seed(epoch_seed(seed, epoch));
            // Fisher-Yates, top down: swap each position with a uniform pick
            // from the not-yet-placed prefix.
            for i in (1..len).rev() {
                let j = below(&mut rng, i as u64 + 1) as usize;
                order.swap(i, j);
            }
        }
        order
    }
}

/// Derive epoch `epoch`'s generator seed from the loader's seed.
///
/// The golden-ratio multiple decorrelates neighboring epochs before
/// [`Rng::seed`] diffuses the result through splitmix64, so epoch 0 and epoch
/// 1 are unrelated streams rather than one stream a step apart. Ported from
/// v2's sampler, which needed the same property.
fn epoch_seed(seed: u64, epoch: u64) -> u64 {
    seed ^ epoch.wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// A uniform integer in `0..bound` (`bound > 0`), without the modulo bias of
/// `next_u64() % bound`.
///
/// Lemire's multiply-shift: take the high 64 bits of `word * bound` as the
/// result and reject the leftover low window that would otherwise be counted
/// twice. Rejection is per-draw and vanishingly rare for the small bounds a
/// shuffle uses.
fn below(rng: &mut Rng, bound: u64) -> u64 {
    debug_assert!(bound > 0, "below: bound must be nonzero");
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let product = u128::from(rng.next_u64()) * u128::from(bound);
        if (product as u64) >= threshold {
            return (product >> 64) as u64;
        }
    }
}

/// The batches of one epoch: an [`ExactSizeIterator`] over
/// `Result<D::Batch>`, produced by [`DataLoader::batches`].
///
/// Each `next` hands the dataset one slice of item positions, so a failing
/// [`Dataset::batch`] surfaces as an `Err` **item** — the loop can log and
/// continue, or `?` out of the epoch, and iteration is not poisoned either
/// way.
#[derive(Debug)]
pub struct Batches<'a, D> {
    dataset: &'a D,
    /// This epoch's full visiting order (all `len` positions).
    order: Vec<usize>,
    batch_size: usize,
    /// Cursor into `order`.
    next: usize,
    /// One past the last position to be yielded: `order.len()`, or the last
    /// whole-batch boundary under `drop_last`.
    end: usize,
}

impl<D: Dataset> Iterator for Batches<'_, D> {
    type Item = Result<D::Batch>;

    fn next(&mut self) -> Option<Result<D::Batch>> {
        if self.next >= self.end {
            return None;
        }
        let stop = (self.next + self.batch_size).min(self.end);
        let indices = &self.order[self.next..stop];
        self.next = stop;
        Some(self.dataset.batch(indices))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.end - self.next).div_ceil(self.batch_size);
        (remaining, Some(remaining))
    }
}

impl<D: Dataset> ExactSizeIterator for Batches<'_, D> {}

impl<D: Dataset> std::iter::FusedIterator for Batches<'_, D> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::dataset::{TensorDataset, VecDataset};
    use crate::device::Device;
    use crate::dtype::DType;
    use crate::error::Error;
    use crate::tensor::Tensor;

    const CPU: Device = Device::Cpu;

    /// `[n, 1]` inputs whose single feature is the item index, `[n]` targets.
    fn tensor_dataset(n: usize) -> TensorDataset {
        let x: Vec<f32> = (0..n).map(|v| v as f32).collect();
        let y: Vec<i64> = (0..n).map(|v| v as i64).collect();
        TensorDataset::new(
            Tensor::from_vec(x, [n, 1], &CPU).unwrap(),
            Tensor::from_vec(y, [n], &CPU).unwrap(),
        )
        .unwrap()
    }

    /// The item indices a loader actually delivers, read out of the target
    /// column of every batch (so this observes the *batches*, not the private
    /// order vector).
    fn delivered<D>(batches: Batches<'_, D>) -> Vec<Vec<i64>>
    where
        D: Dataset<Batch = (Tensor, Tensor)>,
    {
        batches
            .map(|batch| batch.unwrap().1.to_vec::<i64>().unwrap())
            .collect()
    }

    /// Flattened [`delivered`].
    fn flat<D>(batches: Batches<'_, D>) -> Vec<i64>
    where
        D: Dataset<Batch = (Tensor, Tensor)>,
    {
        delivered(batches).concat()
    }

    // ---- batching --------------------------------------------------------

    #[test]
    fn batches_are_batch_size_wide_in_dataset_order() {
        let loader = DataLoader::new(tensor_dataset(6), 2);
        assert_eq!(loader.batch_size(), 2);
        assert_eq!(loader.num_batches(), 3);
        assert_eq!(loader.dataset().len(), 6);

        let shapes: Vec<Vec<usize>> = loader
            .batches()
            .map(|b| b.unwrap().0.dims().to_vec())
            .collect();
        assert_eq!(shapes, vec![vec![2, 1], vec![2, 1], vec![2, 1]]);
        assert_eq!(
            delivered(loader.batches()),
            vec![vec![0, 1], vec![2, 3], vec![4, 5]]
        );
    }

    /// The documented last-partial-batch policy: kept by default.
    #[test]
    fn a_short_final_batch_is_kept_by_default() {
        let loader = DataLoader::new(tensor_dataset(7), 3);
        assert_eq!(loader.num_batches(), 3);
        assert_eq!(
            delivered(loader.batches()),
            vec![vec![0, 1, 2], vec![3, 4, 5], vec![6]]
        );
        // Shapes are exact, never padded to batch_size.
        let last = loader.batches().last().unwrap().unwrap();
        assert_eq!(last.0.dims(), &[1, 1]);
        assert_eq!(last.1.dims(), &[1]);
    }

    #[test]
    fn drop_last_drops_the_short_final_batch() {
        let loader = DataLoader::new(tensor_dataset(7), 3).drop_last(true);
        assert_eq!(loader.num_batches(), 2);
        assert_eq!(
            delivered(loader.batches()),
            vec![vec![0, 1, 2], vec![3, 4, 5]]
        );
    }

    #[test]
    fn an_exact_multiple_is_unaffected_by_drop_last() {
        let ds = tensor_dataset(6);
        let kept = DataLoader::new(&ds, 3);
        let dropped = DataLoader::new(&ds, 3).drop_last(true);
        assert_eq!(kept.num_batches(), 2);
        assert_eq!(dropped.num_batches(), 2);
        assert_eq!(delivered(kept.batches()), delivered(dropped.batches()));
    }

    #[test]
    fn a_batch_larger_than_the_dataset_yields_one_short_batch() {
        let loader = DataLoader::new(tensor_dataset(3), 8);
        assert_eq!(loader.num_batches(), 1);
        assert_eq!(delivered(loader.batches()), vec![vec![0, 1, 2]]);

        // ... and none at all when short batches are dropped.
        let dropping = DataLoader::new(tensor_dataset(3), 8).drop_last(true);
        assert_eq!(dropping.num_batches(), 0);
        assert_eq!(dropping.batches().count(), 0);
    }

    #[test]
    fn an_empty_dataset_yields_no_batches() {
        let loader = DataLoader::new(VecDataset::new(Vec::new()).unwrap(), 4).shuffle(0);
        assert_eq!(loader.num_batches(), 0);
        assert_eq!(loader.batches().len(), 0);
        assert!(loader.batches().next().is_none());
    }

    #[test]
    fn num_batches_matches_what_the_iterator_reports_and_yields() {
        for len in 0..10usize {
            for batch_size in 1..5usize {
                for drop_last in [false, true] {
                    let loader = DataLoader::new(tensor_dataset(len), batch_size)
                        .drop_last(drop_last)
                        .shuffle(1);
                    let case = format!("len {len}, batch {batch_size}, drop {drop_last}");
                    let expected = loader.num_batches();
                    assert_eq!(loader.batches().len(), expected, "{case}");

                    let sizes: Vec<usize> =
                        delivered(loader.batches()).iter().map(Vec::len).collect();
                    assert_eq!(sizes.len(), expected, "{case}");
                    // Every batch but the last is exactly `batch_size`; the
                    // last is nonempty, and full unless it is a kept tail.
                    let (last, rest) = match sizes.split_last() {
                        Some(split) => split,
                        None => continue,
                    };
                    assert!(rest.iter().all(|size| *size == batch_size), "{case}");
                    assert!(*last > 0 && *last <= batch_size, "{case}");
                    if drop_last {
                        assert_eq!(*last, batch_size, "{case}");
                    }
                    // Nothing is visited twice and nothing but a dropped tail
                    // is skipped.
                    let visited: usize = sizes.iter().sum();
                    assert_eq!(
                        visited,
                        if drop_last {
                            expected * batch_size
                        } else {
                            len
                        },
                        "{case}"
                    );
                }
            }
        }
    }

    #[test]
    fn exact_size_len_counts_down_as_batches_are_taken() {
        let loader = DataLoader::new(tensor_dataset(7), 3);
        let mut batches = loader.batches();
        assert_eq!(batches.len(), 3);
        batches.next().unwrap().unwrap();
        assert_eq!(batches.len(), 2);
        batches.next().unwrap().unwrap();
        assert_eq!(batches.len(), 1);
        batches.next().unwrap().unwrap();
        assert_eq!(batches.len(), 0);
        assert!(batches.next().is_none());
        assert!(batches.next().is_none()); // fused
    }

    // ---- determinism -----------------------------------------------------

    #[test]
    fn the_same_seed_gives_the_same_batch_order() {
        let ds = tensor_dataset(16);
        let first = flat(DataLoader::new(&ds, 3).shuffle(42).batches());
        let second = flat(DataLoader::new(&ds, 3).shuffle(42).batches());
        assert_eq!(first, second);

        // ... and re-iterating one loader is also stable (nothing consumed).
        let loader = DataLoader::new(&ds, 3).shuffle(42);
        assert_eq!(flat(loader.batches()), first);
        assert_eq!(flat(loader.batches()), first);
    }

    #[test]
    fn a_different_seed_gives_a_different_batch_order() {
        let ds = tensor_dataset(16);
        let a = flat(DataLoader::new(&ds, 4).shuffle(0).batches());
        let b = flat(DataLoader::new(&ds, 4).shuffle(1).batches());
        let c = flat(DataLoader::new(&ds, 4).shuffle(2).batches());
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);

        // Nearby seeds are the interesting case: `Rng::seed` diffuses them.
        assert_ne!(a[0], b[0]);
    }

    #[test]
    fn a_shuffled_epoch_visits_every_item_exactly_once() {
        let ds = tensor_dataset(16);
        for seed in 0..8u64 {
            let mut seen = flat(DataLoader::new(&ds, 5).shuffle(seed).batches());
            assert_eq!(seen.len(), 16);
            seen.sort_unstable();
            assert_eq!(seen, (0..16i64).collect::<Vec<_>>());
        }
    }

    #[test]
    fn shuffling_actually_reorders() {
        let ds = tensor_dataset(32);
        let ordered: Vec<i64> = (0..32).collect();
        assert_eq!(flat(DataLoader::new(&ds, 8).batches()), ordered);
        assert_ne!(flat(DataLoader::new(&ds, 8).shuffle(0).batches()), ordered);
    }

    #[test]
    fn each_epoch_reshuffles_and_each_epoch_is_reproducible() {
        let ds = tensor_dataset(12);
        let loader = DataLoader::new(&ds, 4).shuffle(9);
        let epochs: Vec<Vec<i64>> = (0..4).map(|e| flat(loader.batches_for_epoch(e))).collect();

        // Distinct orders...
        for (i, a) in epochs.iter().enumerate() {
            for b in &epochs[i + 1..] {
                assert_ne!(a, b);
            }
        }
        // ... each one a permutation, each one reproducible out of order.
        for (e, expected) in epochs.iter().enumerate() {
            let mut sorted = expected.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..12i64).collect::<Vec<_>>());
            assert_eq!(&flat(loader.batches_for_epoch(e as u64)), expected);
        }
        // `batches()` is epoch 0.
        assert_eq!(flat(loader.batches()), epochs[0]);
    }

    #[test]
    fn an_unshuffled_loader_ignores_the_epoch() {
        let loader = DataLoader::new(tensor_dataset(5), 2);
        let expected: Vec<i64> = (0..5).collect();
        for epoch in [0, 1, 7, u64::MAX] {
            assert_eq!(flat(loader.batches_for_epoch(epoch)), expected);
        }
    }

    /// Pins the permutation, not just its properties: the shuffle is part of
    /// the reproducibility contract (a resumed run must replay the same
    /// order), so changing the algorithm has to be a deliberate edit here.
    #[test]
    fn the_permutation_is_pinned_to_the_algorithm() {
        let loader = DataLoader::new(tensor_dataset(8), 8).shuffle(0);
        assert_eq!(
            flat(loader.batches()),
            vec![5, 7, 3, 6, 2, 1, 4, 0],
            "the Fisher-Yates/Rng pairing changed; update deliberately"
        );
        assert_eq!(
            flat(loader.batches_for_epoch(1)),
            vec![2, 4, 3, 7, 0, 1, 5, 6]
        );
    }

    #[test]
    fn below_is_in_range_and_covers_it() {
        let mut rng = Rng::seed(5);
        let mut seen = [0usize; 3];
        for _ in 0..3000 {
            let value = below(&mut rng, 3);
            assert!(value < 3);
            seen[value as usize] += 1;
        }
        // A bound of 1 is degenerate but legal.
        assert_eq!(below(&mut rng, 1), 0);
        assert!(seen.iter().all(|count| *count > 800), "{seen:?}");
    }

    // ---- error propagation and composition -------------------------------

    /// A dataset that fails on a chosen batch, to check the loader surfaces
    /// the error as an item rather than panicking or stopping early.
    struct Failing {
        len: usize,
        fail_from: usize,
    }

    impl Dataset for Failing {
        type Batch = usize;

        fn len(&self) -> usize {
            self.len
        }

        fn batch(&self, indices: &[usize]) -> Result<usize> {
            if indices[0] >= self.fail_from {
                return Err(Error::InvalidArg {
                    op: "Failing::batch",
                    msg: "as requested".to_string(),
                });
            }
            Ok(indices.len())
        }
    }

    #[test]
    fn a_dataset_error_becomes_an_err_item_without_ending_the_epoch() {
        let loader = DataLoader::new(
            Failing {
                len: 6,
                fail_from: 2,
            },
            2,
        );
        let outcomes: Vec<Result<usize>> = loader.batches().collect();
        assert_eq!(outcomes.len(), 3);
        assert_eq!(*outcomes[0].as_ref().unwrap(), 2);
        assert!(matches!(
            outcomes[1],
            Err(Error::InvalidArg {
                op: "Failing::batch",
                ..
            })
        ));
        assert!(outcomes[2].is_err()); // iteration continued past the failure
    }

    #[test]
    fn a_batch_is_never_empty_so_datasets_need_not_handle_that() {
        // The `Dataset` contract lets implementations reject an empty slice;
        // the loader must therefore never produce one, at any length.
        for len in 0..7usize {
            for batch_size in 1..4usize {
                let loader = DataLoader::new(tensor_dataset(len), batch_size).shuffle(3);
                for batch in loader.batches() {
                    assert!(batch.unwrap().1.dims()[0] > 0);
                }
            }
        }
    }

    #[test]
    fn two_loaders_can_share_one_borrowed_dataset() {
        let ds = tensor_dataset(8);
        let train = DataLoader::new(&ds, 4).shuffle(0);
        let eval = DataLoader::new(&ds, 8);
        assert_eq!(train.num_batches(), 2);
        assert_eq!(eval.num_batches(), 1);
        assert_eq!(flat(eval.batches()), (0..8i64).collect::<Vec<_>>());
    }

    #[test]
    #[should_panic(expected = "batch_size must be nonzero")]
    fn a_zero_batch_size_is_a_programmer_error() {
        let _ = DataLoader::new(tensor_dataset(4), 0);
    }

    /// The loader must not sever the graph either: a batch drawn through a
    /// `DataLoader` over a traced `TensorDataset` still differentiates back to
    /// the whole split, which is only possible if batching stayed in tensor
    /// ops (no host round-trip).
    #[test]
    fn batches_of_a_traced_dataset_stay_differentiable() {
        let x = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], [4, 1], &CPU)
            .unwrap()
            .traced()
            .unwrap();
        let y = Tensor::from_vec(vec![0i64; 4], [4], &CPU).unwrap();
        let ds = TensorDataset::new(x.clone(), y).unwrap();
        let loader = DataLoader::new(&ds, 2).shuffle(0);

        let mut total = 0.0;
        for batch in loader.batches() {
            let (inputs, _) = batch.unwrap();
            let grads = inputs.sum_all().unwrap().backward().unwrap();
            let grad = grads.wrt_input(&x).unwrap();
            assert_eq!(grad.dims(), &[4, 1]);
            // Each batch's gradient is 1 at the two rows it selected.
            let values = grad.to_vec::<f64>().unwrap();
            assert_eq!(values.iter().sum::<f64>(), 2.0);
            total += values.iter().sum::<f64>();
        }
        assert_eq!(total, 4.0); // every row selected exactly once per epoch
        assert_eq!(x.dtype(), DType::F64);
    }
}
