use super::collate::{Collate, IdentityCollate};
use super::dataset::Dataset;
use super::sampler::{
    BatchSampler, BatchSamplerIter, PartialBatchSamplerIter, RandomSampler, Sampler,
    SequentialSampler,
};
use crate::backend::{Backend, Cpu};
use crate::const_check::nonzero;
use crate::dtype::FloatElement;
use crate::rng::SmallRng;
use std::marker::PhantomData;

/// Builder for fixed-size dataset batches.
pub struct DataLoader<D, C = IdentityCollate, S = SequentialSampler, E = f32, BK = Cpu>
where
    E: FloatElement,
    BK: Backend<E>,
{
    dataset: D,
    sampler: S,
    collator: C,
    marker: PhantomData<(E, BK)>,
}

/// A data loader with the batch size encoded in its type.
pub struct BatchedDataLoader<D, C, S, const BATCH: usize, E = f32, BK = Cpu>
where
    E: FloatElement,
    BK: Backend<E>,
{
    dataset: D,
    batch_sampler: BatchSampler<S, BATCH>,
    collator: C,
    marker: PhantomData<(E, BK)>,
}

/// Iterator over full batches. Trailing partial batches are dropped.
pub struct DataLoaderIter<'a, D, C, S, const BATCH: usize, E = f32, BK = Cpu>
where
    E: FloatElement,
    BK: Backend<E>,
    S: Sampler + 'a,
{
    dataset: &'a D,
    collator: &'a C,
    batches: BatchSamplerIter<S::Iter<'a>, BATCH>,
    marker: PhantomData<(E, BK)>,
}

/// Iterator over batches of dataset items, including a trailing partial batch.
pub struct PartialDataLoaderIter<'a, D, S>
where
    S: Sampler + 'a,
{
    dataset: &'a D,
    batches: PartialBatchSamplerIter<S::Iter<'a>>,
}

impl<D> DataLoader<D, IdentityCollate, SequentialSampler, f32, Cpu>
where
    D: Dataset,
{
    pub fn new(dataset: D) -> Self {
        Self::new_typed(dataset)
    }
}

impl<D, E, BK> DataLoader<D, IdentityCollate, SequentialSampler, E, BK>
where
    D: Dataset,
    E: FloatElement,
    BK: Backend<E>,
{
    pub fn new_typed(dataset: D) -> Self {
        let len = dataset.len();
        Self {
            dataset,
            sampler: SequentialSampler::new(len),
            collator: IdentityCollate,
            marker: PhantomData,
        }
    }
}

impl<D, C, S, E, BK> DataLoader<D, C, S, E, BK>
where
    D: Dataset,
    S: Sampler,
    E: FloatElement,
    BK: Backend<E>,
{
    pub fn collate<Next>(self) -> DataLoader<D, Next, S, E, BK>
    where
        Next: Default,
    {
        self.with_collate(Next::default())
    }

    pub fn with_collate<Next>(self, collator: Next) -> DataLoader<D, Next, S, E, BK> {
        DataLoader {
            dataset: self.dataset,
            sampler: self.sampler,
            collator,
            marker: PhantomData,
        }
    }

    pub fn sampler<Next>(self, sampler: Next) -> DataLoader<D, C, Next, E, BK>
    where
        Next: Sampler,
    {
        DataLoader {
            dataset: self.dataset,
            sampler,
            collator: self.collator,
            marker: PhantomData,
        }
    }

    pub fn shuffle(self, rng: &mut SmallRng) -> DataLoader<D, C, RandomSampler, E, BK> {
        let sampler = RandomSampler::without_replacement(self.dataset.len(), rng);
        self.sampler(sampler)
    }

    pub fn batch_size<const BATCH: usize>(self) -> BatchedDataLoader<D, C, S, BATCH, E, BK>
    where
        [(); nonzero(BATCH, "BatchSampler", "BATCH")]:,
    {
        BatchedDataLoader {
            dataset: self.dataset,
            batch_sampler: BatchSampler::new(self.sampler),
            collator: self.collator,
            marker: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.sampler.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sampler.is_empty()
    }

    pub fn dataset(&self) -> &D {
        &self.dataset
    }
}

impl<D, C, S, const BATCH: usize, E, BK> BatchedDataLoader<D, C, S, BATCH, E, BK>
where
    D: Dataset,
    S: Sampler,
    E: FloatElement,
    BK: Backend<E>,
    [(); nonzero(BATCH, "BatchSampler", "BATCH")]:,
{
    pub fn sampler<Next>(self, sampler: Next) -> BatchedDataLoader<D, C, Next, BATCH, E, BK>
    where
        Next: Sampler,
    {
        BatchedDataLoader {
            dataset: self.dataset,
            batch_sampler: BatchSampler::new(sampler),
            collator: self.collator,
            marker: PhantomData,
        }
    }

    pub fn shuffle(
        self,
        rng: &mut SmallRng,
    ) -> BatchedDataLoader<D, C, RandomSampler, BATCH, E, BK> {
        let sampler = RandomSampler::without_replacement(self.dataset.len(), rng);
        self.sampler(sampler)
    }

    pub fn iter(&self) -> DataLoaderIter<'_, D, C, S, BATCH, E, BK>
    where
        C: Collate<D::Item, BATCH, E, BK>,
    {
        DataLoaderIter {
            dataset: &self.dataset,
            collator: &self.collator,
            batches: self.batch_sampler.iter(),
            marker: PhantomData,
        }
    }

    pub fn iter_partial(&self) -> PartialDataLoaderIter<'_, D, S> {
        PartialDataLoaderIter {
            dataset: &self.dataset,
            batches: PartialBatchSamplerIter {
                indices: self.batch_sampler.sampler.iter(),
                batch_size: BATCH,
            },
        }
    }

    pub fn len(&self) -> usize {
        self.batch_sampler.len()
    }

    pub fn len_with_partial(&self) -> usize {
        let len = self.batch_sampler.sampler.len();
        (len + BATCH - 1) / BATCH
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn batch_size(&self) -> usize {
        BATCH
    }

    pub fn dataset(&self) -> &D {
        &self.dataset
    }

    pub fn sampler_ref(&self) -> &S {
        &self.batch_sampler.sampler
    }
}

impl<'a, D, C, S, const BATCH: usize, E, BK> Iterator for DataLoaderIter<'a, D, C, S, BATCH, E, BK>
where
    D: Dataset,
    S: Sampler + 'a,
    E: FloatElement,
    BK: Backend<E>,
    C: Collate<D::Item, BATCH, E, BK>,
{
    type Item = C::Batch;

    fn next(&mut self) -> Option<Self::Item> {
        let batch_indices = self.batches.next()?;
        let items = batch_indices.into_iter().map(|index| {
            self.dataset
                .get(index)
                .expect("dataset returned no sample for an index inside its length")
        });
        Some(self.collator.collate(items))
    }
}

impl<'a, D, S> Iterator for PartialDataLoaderIter<'a, D, S>
where
    D: Dataset,
    S: Sampler + 'a,
{
    type Item = Vec<D::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        self.batches.next().map(|indices| {
            indices
                .into_iter()
                .map(|index| {
                    self.dataset
                        .get(index)
                        .expect("dataset returned no sample for an index inside its length")
                })
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::DataLoader;
    use crate::data::{
        Collate, Dataset, ImageOneHotClassification, OneHotClassification, RandomSampler, Sampler,
    };
    use crate::rng::SmallRng;
    use crate::tensor::{Tensor2D, Tensor3D};

    #[derive(Clone)]
    struct TinyFlatDataset {
        samples: Vec<([f32; 3], u8)>,
    }

    impl Dataset for TinyFlatDataset {
        type Item = ([f32; 3], u8);

        fn len(&self) -> usize {
            self.samples.len()
        }

        fn get(&self, index: usize) -> Option<Self::Item> {
            self.samples.get(index).copied()
        }
    }

    #[derive(Clone)]
    struct TinyImageDataset {
        samples: Vec<([[f32; 2]; 2], u8)>,
    }

    impl Dataset for TinyImageDataset {
        type Item = ([[f32; 2]; 2], u8);

        fn len(&self) -> usize {
            self.samples.len()
        }

        fn get(&self, index: usize) -> Option<Self::Item> {
            self.samples.get(index).copied()
        }
    }

    fn flat_dataset() -> TinyFlatDataset {
        TinyFlatDataset {
            samples: vec![
                ([1.0, 2.0, 3.0], 0),
                ([4.0, 5.0, 6.0], 1),
                ([7.0, 8.0, 9.0], 2),
                ([10.0, 11.0, 12.0], 1),
                ([13.0, 14.0, 15.0], 0),
            ],
        }
    }

    fn image_dataset() -> TinyImageDataset {
        TinyImageDataset {
            samples: vec![([[1.0, 2.0], [3.0, 4.0]], 0), ([[5.0, 6.0], [7.0, 8.0]], 1)],
        }
    }

    #[derive(Clone, Copy)]
    struct ScaleFeatures {
        scale: f32,
    }

    impl<const BATCH: usize> Collate<([f32; 3], u8), BATCH> for ScaleFeatures {
        type Batch = Tensor2D<BATCH, 3>;

        fn collate<I>(&self, items: I) -> Self::Batch
        where
            I: IntoIterator<Item = ([f32; 3], u8)>,
        {
            let mut values = Vec::with_capacity(BATCH * 3);
            let mut count = 0;
            for (index, (features, _)) in items.into_iter().enumerate() {
                assert!(index < BATCH, "collator received more than BATCH items");
                values.extend(features.map(|value| value * self.scale));
                count = index + 1;
            }
            assert_eq!(count, BATCH, "collator received fewer than BATCH items");
            Tensor2D::<BATCH, 3>::from_vec(values).unwrap()
        }
    }

    #[test]
    fn identity_collate_yields_arrays_of_items() {
        let loader = DataLoader::new(flat_dataset()).batch_size::<2>();
        let batches: Vec<[([f32; 3], u8); 2]> = loader.iter().collect();

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0], [([1.0, 2.0, 3.0], 0), ([4.0, 5.0, 6.0], 1)]);
    }

    #[test]
    fn partial_iterator_keeps_trailing_items() {
        let loader = DataLoader::new(flat_dataset()).batch_size::<2>();
        let batches: Vec<Vec<([f32; 3], u8)>> = loader.iter_partial().collect();

        assert_eq!(loader.len(), 2);
        assert_eq!(loader.len_with_partial(), 3);
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[2], vec![([13.0, 14.0, 15.0], 0)]);
    }

    #[test]
    fn collator_can_hold_runtime_configuration() {
        let loader = DataLoader::new(flat_dataset())
            .with_collate(ScaleFeatures { scale: 0.5 })
            .batch_size::<2>();
        let batches: Vec<Tensor2D<2, 3>> = loader.iter().collect();

        assert_eq!(batches[0].to_vec(), vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0]);
    }

    #[test]
    fn explicit_random_sampler_controls_loader_order() {
        let mut rng_a = SmallRng::seed_from_u64(99);
        let mut rng_b = SmallRng::seed_from_u64(99);
        let sampler_a = RandomSampler::without_replacement(5, &mut rng_a);
        let sampler_b = RandomSampler::without_replacement(5, &mut rng_b);

        assert_eq!(
            sampler_a.iter().collect::<Vec<_>>(),
            sampler_b.iter().collect::<Vec<_>>()
        );

        let loader = DataLoader::new(flat_dataset())
            .sampler(sampler_a)
            .batch_size::<2>();
        let batches: Vec<[([f32; 3], u8); 2]> = loader.iter().collect();

        assert_eq!(batches.len(), 2);
    }

    #[test]
    fn flat_classification_collate_yields_typed_batches_and_drops_partial_batch() {
        let loader = DataLoader::new(flat_dataset())
            .collate::<OneHotClassification<3, 3>>()
            .batch_size::<2>();
        let batches: Vec<(Tensor2D<2, 3>, Tensor2D<2, 3>)> = loader.iter().collect();

        assert_eq!(loader.len(), 2);
        assert_eq!(loader.batch_size(), 2);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].0.shape(), &[2, 3]);
        assert_eq!(batches[0].1.shape(), &[2, 3]);
        assert_eq!(batches[0].0.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(batches[0].1.to_vec(), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        assert_eq!(batches[1].0.to_vec(), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn image_classification_collate_preserves_image_shape() {
        let loader = DataLoader::new(image_dataset())
            .collate::<ImageOneHotClassification<2, 2, 2>>()
            .batch_size::<2>();
        let batches: Vec<(Tensor3D<2, 2, 2>, Tensor2D<2, 2>)> = loader.iter().collect();

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].0.shape(), &[2, 2, 2]);
        assert_eq!(
            batches[0].0.to_vec(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
        assert_eq!(batches[0].1.to_vec(), vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn data_loader_shuffle_is_deterministic_for_seeded_rng() {
        let mut rng_a = SmallRng::seed_from_u64(42);
        let mut rng_b = SmallRng::seed_from_u64(42);

        let loader_a = DataLoader::new(flat_dataset())
            .shuffle(&mut rng_a)
            .collate::<OneHotClassification<3, 3>>()
            .batch_size::<2>();
        let loader_b = DataLoader::new(flat_dataset())
            .shuffle(&mut rng_b)
            .collate::<OneHotClassification<3, 3>>()
            .batch_size::<2>();

        let values_a: Vec<_> = loader_a
            .iter()
            .flat_map(|(features, _)| features.to_vec())
            .collect();
        let values_b: Vec<_> = loader_b
            .iter()
            .flat_map(|(features, _)| features.to_vec())
            .collect();

        assert_eq!(values_a, values_b);
        assert_ne!(
            values_a,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0
            ]
        );
    }
}
