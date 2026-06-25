use super::sampler::{SequentialSampler, ShuffleSampler};
use super::{Collate, Dataset, Sampler};
use crate::error::{DataError, Error, Result};
use std::sync::atomic::{AtomicU64, Ordering};

pub fn loader<D>(dataset: D) -> DataLoaderBuilder<D, SequentialSampler, ()>
where
    D: Dataset,
{
    DataLoaderBuilder {
        dataset,
        sampler: SequentialSampler,
        collate: (),
        batch_size: 1,
        drop_last: false,
    }
}

pub fn static_loader<const BATCH: usize, D>(
    dataset: D,
) -> StaticDataLoaderBuilder<D, SequentialSampler, (), BATCH>
where
    D: Dataset,
{
    StaticDataLoaderBuilder {
        dataset,
        sampler: SequentialSampler,
        collate: (),
    }
}

#[derive(Debug)]
pub struct DataLoaderBuilder<D, S, C> {
    dataset: D,
    sampler: S,
    collate: C,
    batch_size: usize,
    drop_last: bool,
}

impl<D, S, C> DataLoaderBuilder<D, S, C> {
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    pub fn sampler<S2>(self, sampler: S2) -> DataLoaderBuilder<D, S2, C>
    where
        S2: Sampler,
    {
        DataLoaderBuilder {
            dataset: self.dataset,
            sampler,
            collate: self.collate,
            batch_size: self.batch_size,
            drop_last: self.drop_last,
        }
    }

    pub fn shuffle(self, seed: u64) -> DataLoaderBuilder<D, ShuffleSampler, C> {
        self.sampler(ShuffleSampler::new(seed))
    }

    pub fn collate<C2>(self, collate: C2) -> DataLoaderBuilder<D, S, C2>
    where
        D: Dataset,
        C2: Collate<D::Item>,
        C2::Error: Into<Error>,
    {
        DataLoaderBuilder {
            dataset: self.dataset,
            sampler: self.sampler,
            collate,
            batch_size: self.batch_size,
            drop_last: self.drop_last,
        }
    }

    pub fn build(self) -> Result<DataLoader<D, S, C>>
    where
        D: Dataset,
        D::Error: Into<Error>,
        S: Sampler,
        C: Collate<D::Item>,
        C::Error: Into<Error>,
    {
        DataLoader::new(
            self.dataset,
            self.sampler,
            self.collate,
            self.batch_size,
            self.drop_last,
        )
    }
}

#[derive(Debug)]
pub struct DataLoader<D, S, C> {
    dataset: D,
    sampler: S,
    collate: C,
    batch_size: usize,
    drop_last: bool,
    epoch: AtomicU64,
}

impl<D, S, C> DataLoader<D, S, C> {
    pub fn new(
        dataset: D,
        sampler: S,
        collate: C,
        batch_size: usize,
        drop_last: bool,
    ) -> Result<Self>
    where
        D: Dataset,
        D::Error: Into<Error>,
        S: Sampler,
        C: Collate<D::Item>,
        C::Error: Into<Error>,
    {
        if batch_size == 0 {
            return Err(DataError::InvalidBatchSize { batch_size }.into());
        }

        Ok(Self {
            dataset,
            sampler,
            collate,
            batch_size,
            drop_last,
            epoch: AtomicU64::new(0),
        })
    }

    pub fn iter(&self) -> DataLoaderIter<'_, D, S, C>
    where
        D: Dataset,
        D::Error: Into<Error>,
        S: Sampler,
        C: Collate<D::Item>,
        C::Error: Into<Error>,
    {
        let epoch = self.epoch.fetch_add(1, Ordering::Relaxed);
        self.iter_epoch(epoch)
    }

    pub fn iter_epoch(&self, epoch: u64) -> DataLoaderIter<'_, D, S, C>
    where
        D: Dataset,
        D::Error: Into<Error>,
        S: Sampler,
        C: Collate<D::Item>,
        C::Error: Into<Error>,
    {
        DataLoaderIter {
            loader: self,
            indices: self.sampler.indices(self.dataset.len(), epoch),
        }
    }
}

pub struct DataLoaderIter<'a, D, S, C>
where
    D: Dataset,
    D::Error: Into<Error>,
    S: Sampler,
    C: Collate<D::Item>,
    C::Error: Into<Error>,
{
    loader: &'a DataLoader<D, S, C>,
    indices: S::Iter,
}

#[derive(Debug)]
pub struct StaticDataLoader<D, S, C, const BATCH: usize> {
    dataset: D,
    sampler: S,
    collate: C,
    epoch: AtomicU64,
}

impl<D, S, C, const BATCH: usize> StaticDataLoader<D, S, C, BATCH> {
    pub fn new(dataset: D, sampler: S, collate: C) -> Result<Self>
    where
        D: Dataset,
        D::Error: Into<Error>,
        S: Sampler,
        C: Collate<D::Item>,
        C::Error: Into<Error>,
    {
        if BATCH == 0 {
            return Err(DataError::InvalidBatchSize { batch_size: BATCH }.into());
        }

        Ok(Self {
            dataset,
            sampler,
            collate,
            epoch: AtomicU64::new(0),
        })
    }

    pub fn iter(&self) -> StaticDataLoaderIter<'_, D, S, C, BATCH>
    where
        D: Dataset,
        D::Error: Into<Error>,
        S: Sampler,
        C: Collate<D::Item>,
        C::Error: Into<Error>,
    {
        let epoch = self.epoch.fetch_add(1, Ordering::Relaxed);
        self.iter_epoch(epoch)
    }

    pub fn iter_epoch(&self, epoch: u64) -> StaticDataLoaderIter<'_, D, S, C, BATCH>
    where
        D: Dataset,
        D::Error: Into<Error>,
        S: Sampler,
        C: Collate<D::Item>,
        C::Error: Into<Error>,
    {
        StaticDataLoaderIter {
            loader: self,
            indices: self.sampler.indices(self.dataset.len(), epoch),
        }
    }
}

#[derive(Debug)]
pub struct StaticDataLoaderBuilder<D, S, C, const BATCH: usize> {
    dataset: D,
    sampler: S,
    collate: C,
}

impl<D, S, C, const BATCH: usize> StaticDataLoaderBuilder<D, S, C, BATCH> {
    pub fn sampler<S2>(self, sampler: S2) -> StaticDataLoaderBuilder<D, S2, C, BATCH>
    where
        S2: Sampler,
    {
        StaticDataLoaderBuilder {
            dataset: self.dataset,
            sampler,
            collate: self.collate,
        }
    }

    pub fn shuffle(self, seed: u64) -> StaticDataLoaderBuilder<D, ShuffleSampler, C, BATCH> {
        self.sampler(ShuffleSampler::new(seed))
    }

    pub fn collate<C2>(self, collate: C2) -> StaticDataLoaderBuilder<D, S, C2, BATCH>
    where
        D: Dataset,
        C2: Collate<D::Item>,
        C2::Error: Into<Error>,
    {
        StaticDataLoaderBuilder {
            dataset: self.dataset,
            sampler: self.sampler,
            collate,
        }
    }

    pub fn build(self) -> Result<StaticDataLoader<D, S, C, BATCH>>
    where
        D: Dataset,
        D::Error: Into<Error>,
        S: Sampler,
        C: Collate<D::Item>,
        C::Error: Into<Error>,
    {
        StaticDataLoader::new(self.dataset, self.sampler, self.collate)
    }
}

pub struct StaticDataLoaderIter<'a, D, S, C, const BATCH: usize>
where
    D: Dataset,
    D::Error: Into<Error>,
    S: Sampler,
    C: Collate<D::Item>,
    C::Error: Into<Error>,
{
    loader: &'a StaticDataLoader<D, S, C, BATCH>,
    indices: S::Iter,
}

impl<D, S, C, Item> Iterator for DataLoaderIter<'_, D, S, C>
where
    D: Dataset<Item = Item>,
    D::Error: Into<Error>,
    S: Sampler,
    C: Collate<Item>,
    C::Error: Into<Error>,
{
    type Item = Result<C::Batch>;

    fn next(&mut self) -> Option<Self::Item> {
        let remaining = self.indices.len();
        if remaining == 0 {
            return None;
        }

        if remaining < self.loader.batch_size && self.loader.drop_last {
            return None;
        }

        let batch_size = self.loader.batch_size.min(remaining);
        let items = match collect_batch(&self.loader.dataset, &mut self.indices, batch_size) {
            Ok(items) => items,
            Err(err) => return Some(Err(err)),
        };

        Some(self.loader.collate.collate(items).map_err(Into::into))
    }
}

impl<D, S, C, Item, const BATCH: usize> Iterator for StaticDataLoaderIter<'_, D, S, C, BATCH>
where
    D: Dataset<Item = Item>,
    D::Error: Into<Error>,
    S: Sampler,
    C: Collate<Item>,
    C::Error: Into<Error>,
{
    type Item = Result<C::Batch>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.indices.len() < BATCH {
            return None;
        }

        let items = match collect_batch(&self.loader.dataset, &mut self.indices, BATCH) {
            Ok(items) => items,
            Err(err) => return Some(Err(err)),
        };

        Some(self.loader.collate.collate(items).map_err(Into::into))
    }
}

impl<'a, D, S, C, Item> IntoIterator for &'a DataLoader<D, S, C>
where
    D: Dataset<Item = Item>,
    D::Error: Into<Error>,
    S: Sampler,
    C: Collate<Item>,
    C::Error: Into<Error>,
{
    type Item = Result<C::Batch>;
    type IntoIter = DataLoaderIter<'a, D, S, C>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, D, S, C, Item, const BATCH: usize> IntoIterator for &'a StaticDataLoader<D, S, C, BATCH>
where
    D: Dataset<Item = Item>,
    D::Error: Into<Error>,
    S: Sampler,
    C: Collate<Item>,
    C::Error: Into<Error>,
{
    type Item = Result<C::Batch>;
    type IntoIter = StaticDataLoaderIter<'a, D, S, C, BATCH>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

fn collect_batch<D, I, Item>(dataset: &D, indices: &mut I, batch_size: usize) -> Result<Vec<Item>>
where
    D: Dataset<Item = Item>,
    D::Error: Into<Error>,
    I: Iterator<Item = usize>,
{
    let mut items = Vec::with_capacity(batch_size);
    for offset in 0..batch_size {
        let index = indices
            .next()
            .expect("sampler iterator ended before its exact size hint");
        match dataset.get(index) {
            Ok(item) => items.push(item),
            Err(err) => {
                for _ in offset + 1..batch_size {
                    let _ = indices.next();
                }
                return Err(err.into());
            }
        }
    }
    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{
        Batch, SequentialSampler, ShuffleSampler, StackVecCollate, VecDataset, features,
        static_features,
    };
    use crate::error::Error;
    use crate::nn::{HasParameters, Linear, Module, mse_loss};
    use crate::optim::{Optimizer, Sgd};
    use crate::shape::{C, D2, Sym};
    use crate::tensor::Tensor;

    #[test]
    fn dataloader_batches_with_and_without_drop_last() {
        let dataset = VecDataset::new(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]]);
        let loader = DataLoader::new(
            dataset,
            SequentialSampler,
            StackVecCollate::<1>::new(),
            2,
            false,
        )
        .unwrap();
        let shapes: Vec<_> = loader
            .iter()
            .map(|batch| batch.unwrap().shape().dims().to_vec())
            .collect();
        assert_eq!(shapes, vec![vec![2, 1], vec![2, 1], vec![1, 1]]);

        let dataset = VecDataset::new(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]]);
        let loader = DataLoader::new(
            dataset,
            SequentialSampler,
            StackVecCollate::<1>::new(),
            2,
            true,
        )
        .unwrap();
        assert_eq!(loader.iter().count(), 2);
    }

    #[test]
    fn dataloader_rejects_zero_batch_size() {
        let err = DataLoader::new(
            VecDataset::new(vec![vec![1.0]]),
            SequentialSampler,
            StackVecCollate::<1>::new(),
            0,
            false,
        )
        .unwrap_err();

        assert!(matches!(
            err,
            Error::Data(DataError::InvalidBatchSize { batch_size: 0 })
        ));
    }

    #[test]
    fn dataloader_advances_shuffle_epoch_between_passes() {
        let dataset = VecDataset::new((0..8).map(|value| vec![value as f32]).collect());
        let sampler = ShuffleSampler::new(7);
        let loader =
            DataLoader::new(dataset, sampler, StackVecCollate::<1>::new(), 8, false).unwrap();

        let first = loader.iter().next().unwrap().unwrap().to_vec().unwrap();
        let second = loader.iter().next().unwrap().unwrap().to_vec().unwrap();
        let expected_first: Vec<_> = sampler.indices(8, 0).map(|v| v as f32).collect();
        let expected_second: Vec<_> = sampler.indices(8, 1).map(|v| v as f32).collect();

        assert_eq!(first, expected_first);
        assert_eq!(second, expected_second);
        assert_ne!(first, second);
    }

    #[test]
    fn dataloader_iter_epoch_uses_explicit_epoch_without_advancing() {
        let dataset = VecDataset::new((0..8).map(|value| vec![value as f32]).collect());
        let sampler = ShuffleSampler::new(7);
        let loader =
            DataLoader::new(dataset, sampler, StackVecCollate::<1>::new(), 8, false).unwrap();

        let first = loader
            .iter_epoch(3)
            .next()
            .unwrap()
            .unwrap()
            .to_vec()
            .unwrap();
        let second = loader
            .iter_epoch(3)
            .next()
            .unwrap()
            .unwrap()
            .to_vec()
            .unwrap();
        let expected: Vec<_> = sampler.indices(8, 3).map(|v| v as f32).collect();

        assert_eq!(first, expected);
        assert_eq!(second, expected);
    }

    #[test]
    fn tiny_training_loop_consumes_loader_batches() {
        let samples = vec![
            (vec![0.0, 0.0], vec![0.0]),
            (vec![1.0, 0.0], vec![2.0]),
            (vec![0.0, 1.0], vec![3.0]),
            (vec![1.0, 1.0], vec![5.0]),
        ];
        let dataset = VecDataset::new(samples);
        let loader = super::loader(dataset)
            .batch_size(4)
            .collate((features::<2>(), features::<1>()))
            .build()
            .unwrap();
        let mut model = Linear::<2, 1>::zeros().unwrap();
        let mut opt = Sgd::new(0.1);

        let first_loss = epoch_loss(&loader, &model);
        for _ in 0..40 {
            let mut refs = Vec::new();
            model.parameters(&mut refs);
            opt.zero_grad(&refs);
            drop(refs);

            for batch in &loader {
                let (x, y) = batch.unwrap();
                let pred = model.forward(&x).unwrap();
                mse_loss(&pred, &y).unwrap().backward().unwrap();
            }

            let mut refs = Vec::new();
            model.parameters_mut(&mut refs);
            opt.step(&mut refs).unwrap();
        }
        let final_loss = epoch_loss(&loader, &model);

        assert!(final_loss < first_loss, "{final_loss} >= {first_loss}");
    }

    fn epoch_loss<D, S, Col>(loader: &DataLoader<D, S, Col>, model: &Linear<2, 1>) -> f32
    where
        D: Dataset<Item = (Vec<f32>, Vec<f32>)>,
        D::Error: Into<Error>,
        S: Sampler,
        Col: Collate<
                (Vec<f32>, Vec<f32>),
                Batch = (Tensor<D2<Sym<Batch>, C<2>>>, Tensor<D2<Sym<Batch>, C<1>>>),
            >,
        Col::Error: Into<Error>,
    {
        let mut total = 0.0;
        for batch in loader {
            let (x, y) = batch.unwrap();
            total += mse_loss(&model.forward(&x).unwrap(), &y)
                .unwrap()
                .to_vec()
                .unwrap()[0];
        }
        total
    }

    #[test]
    fn builder_uses_sequential_sampler_by_default() {
        let loader = super::loader(VecDataset::new(vec![vec![1.0], vec![2.0], vec![3.0]]))
            .batch_size(2)
            .collate(features::<1>())
            .build()
            .unwrap();

        let batches: Vec<_> = loader
            .iter()
            .map(|batch| batch.unwrap().to_vec().unwrap())
            .collect();

        assert_eq!(batches, vec![vec![1.0, 2.0], vec![3.0]]);
    }

    #[test]
    fn static_loader_yields_only_full_static_batches() {
        let loader =
            super::static_loader::<2, _>(VecDataset::new(vec![vec![1.0], vec![2.0], vec![3.0]]))
                .collate(static_features::<2, 1>())
                .build()
                .unwrap();

        let batches: Vec<Tensor<D2<C<2>, C<1>>>> =
            loader.iter().map(|batch| batch.unwrap()).collect();

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].to_vec().unwrap(), vec![1.0, 2.0]);
    }

    #[test]
    fn static_loader_rejects_zero_batch_size() {
        let err = super::static_loader::<0, _>(VecDataset::new(vec![vec![1.0]]))
            .collate(static_features::<0, 1>())
            .build()
            .unwrap_err();

        assert!(matches!(
            err,
            Error::Data(DataError::InvalidBatchSize { batch_size: 0 })
        ));
    }
}
