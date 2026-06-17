//! Dataset and fixed-shape batching utilities.

use crate::backend::{Backend, Cpu};
use crate::dtype::FloatElement;
use crate::rng::SmallRng;
use crate::tensor::{Tensor2D, Tensor3D};
use std::marker::PhantomData;

/// Random-access dataset.
pub trait Dataset {
    type Item;

    fn len(&self) -> usize;

    fn get(&self, index: usize) -> Option<Self::Item>;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Converts a fixed number of dataset items into one typed batch.
pub trait Collate<Item, const BATCH: usize, E = f32, BK = Cpu>
where
    E: FloatElement,
    BK: Backend<E>,
{
    type Batch;

    fn collate(items: [Item; BATCH]) -> Self::Batch;
}

/// Collator that leaves a batch as an array of items.
pub struct IdentityCollate;

/// Collator for flat feature vectors and one-hot classification labels.
pub struct OneHotClassification<const FEATURES: usize, const CLASSES: usize>;

/// Collator for image-shaped samples and one-hot classification labels.
pub struct ImageOneHotClassification<const HEIGHT: usize, const WIDTH: usize, const CLASSES: usize>;

/// Builder for fixed-size dataset batches.
pub struct DataLoader<D, C = IdentityCollate, E = f32, BK = Cpu>
where
    E: FloatElement,
    BK: Backend<E>,
{
    dataset: D,
    indices: Vec<usize>,
    marker: PhantomData<(C, E, BK)>,
}

/// A data loader with the batch size encoded in its type.
pub struct BatchedDataLoader<D, C, const BATCH: usize, E = f32, BK = Cpu>
where
    E: FloatElement,
    BK: Backend<E>,
{
    dataset: D,
    indices: Vec<usize>,
    marker: PhantomData<(C, E, BK)>,
}

/// Iterator over full batches. Trailing partial batches are dropped.
pub struct DataLoaderIter<'a, D, C, const BATCH: usize, E = f32, BK = Cpu>
where
    E: FloatElement,
    BK: Backend<E>,
{
    dataset: &'a D,
    indices: &'a [usize],
    position: usize,
    marker: PhantomData<(C, E, BK)>,
}

impl<D> DataLoader<D, IdentityCollate, f32, Cpu>
where
    D: Dataset,
{
    pub fn new(dataset: D) -> Self {
        Self::new_typed(dataset)
    }
}

impl<D, E, BK> DataLoader<D, IdentityCollate, E, BK>
where
    D: Dataset,
    E: FloatElement,
    BK: Backend<E>,
{
    pub fn new_typed(dataset: D) -> Self {
        let indices = (0..dataset.len()).collect();
        Self {
            dataset,
            indices,
            marker: PhantomData,
        }
    }
}

impl<D, C, E, BK> DataLoader<D, C, E, BK>
where
    D: Dataset,
    E: FloatElement,
    BK: Backend<E>,
{
    pub fn collate<Next>(self) -> DataLoader<D, Next, E, BK> {
        DataLoader {
            dataset: self.dataset,
            indices: self.indices,
            marker: PhantomData,
        }
    }

    pub fn batch_size<const BATCH: usize>(self) -> BatchedDataLoader<D, C, BATCH, E, BK> {
        assert!(BATCH > 0, "batch size must be greater than zero");

        BatchedDataLoader {
            dataset: self.dataset,
            indices: self.indices,
            marker: PhantomData,
        }
    }

    pub fn shuffle(mut self, rng: &mut SmallRng) -> Self {
        shuffle_indices(&mut self.indices, rng);
        self
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn dataset(&self) -> &D {
        &self.dataset
    }
}

impl<D, C, const BATCH: usize, E, BK> BatchedDataLoader<D, C, BATCH, E, BK>
where
    D: Dataset,
    E: FloatElement,
    BK: Backend<E>,
{
    pub fn shuffle(mut self, rng: &mut SmallRng) -> Self {
        shuffle_indices(&mut self.indices, rng);
        self
    }

    pub fn iter(&self) -> DataLoaderIter<'_, D, C, BATCH, E, BK>
    where
        C: Collate<D::Item, BATCH, E, BK>,
    {
        DataLoaderIter {
            dataset: &self.dataset,
            indices: &self.indices,
            position: 0,
            marker: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.indices.len() / BATCH
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
}

impl<'a, D, C, const BATCH: usize, E, BK> Iterator for DataLoaderIter<'a, D, C, BATCH, E, BK>
where
    D: Dataset,
    E: FloatElement,
    BK: Backend<E>,
    C: Collate<D::Item, BATCH, E, BK>,
{
    type Item = C::Batch;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position + BATCH > self.indices.len() {
            return None;
        }

        let items: Vec<_> = self.indices[self.position..self.position + BATCH]
            .iter()
            .copied()
            .map(|index| {
                self.dataset
                    .get(index)
                    .expect("dataset returned no sample for an index inside its length")
            })
            .collect();

        self.position += BATCH;

        let items = items
            .try_into()
            .unwrap_or_else(|_| unreachable!("collected exactly BATCH items"));
        Some(C::collate(items))
    }
}

impl<Item, const BATCH: usize, E, BK> Collate<Item, BATCH, E, BK> for IdentityCollate
where
    E: FloatElement,
    BK: Backend<E>,
{
    type Batch = [Item; BATCH];

    fn collate(items: [Item; BATCH]) -> Self::Batch {
        items
    }
}

impl<const FEATURES: usize, const CLASSES: usize, const BATCH: usize, E, BK>
    Collate<([E; FEATURES], u8), BATCH, E, BK> for OneHotClassification<FEATURES, CLASSES>
where
    E: FloatElement,
    BK: Backend<E>,
{
    type Batch = (
        Tensor2D<BATCH, FEATURES, E, BK>,
        Tensor2D<BATCH, CLASSES, E, BK>,
    );

    fn collate(items: [([E; FEATURES], u8); BATCH]) -> Self::Batch {
        let mut features = Vec::with_capacity(BATCH * FEATURES);
        let mut labels = [0_u8; BATCH];

        for (index, (sample_features, label)) in items.into_iter().enumerate() {
            features.extend(sample_features);
            labels[index] = label;
        }

        (
            Tensor2D::<BATCH, FEATURES, E, BK>::from_vec(features).unwrap(),
            one_hot_labels::<BATCH, CLASSES, E, BK>(labels),
        )
    }
}

impl<const HEIGHT: usize, const WIDTH: usize, const CLASSES: usize, const BATCH: usize, E, BK>
    Collate<([[E; WIDTH]; HEIGHT], u8), BATCH, E, BK>
    for ImageOneHotClassification<HEIGHT, WIDTH, CLASSES>
where
    E: FloatElement,
    BK: Backend<E>,
{
    type Batch = (
        Tensor3D<BATCH, HEIGHT, WIDTH, E, BK>,
        Tensor2D<BATCH, CLASSES, E, BK>,
    );

    fn collate(items: [([[E; WIDTH]; HEIGHT], u8); BATCH]) -> Self::Batch {
        let mut images = Vec::with_capacity(BATCH * HEIGHT * WIDTH);
        let mut labels = [0_u8; BATCH];

        for (index, (image, label)) in items.into_iter().enumerate() {
            images.extend(image.into_iter().flatten());
            labels[index] = label;
        }

        (
            Tensor3D::<BATCH, HEIGHT, WIDTH, E, BK>::from_vec(images).unwrap(),
            one_hot_labels::<BATCH, CLASSES, E, BK>(labels),
        )
    }
}

/// Builds a one-hot class vector.
pub fn one_hot_label<const CLASSES: usize, E>(label: u8) -> [E; CLASSES]
where
    E: FloatElement,
{
    assert!(CLASSES > 0, "one-hot labels require at least one class");
    let class = usize::from(label);
    assert!(
        class < CLASSES,
        "label index {class} is out of range for {CLASSES} classes"
    );

    let mut values = [E::zero(); CLASSES];
    values[class] = E::one();
    values
}

/// Builds a batch of one-hot class vectors.
pub fn one_hot_labels<const BATCH: usize, const CLASSES: usize, E, BK>(
    labels: [u8; BATCH],
) -> Tensor2D<BATCH, CLASSES, E, BK>
where
    E: FloatElement,
    BK: Backend<E>,
{
    let mut values = Vec::with_capacity(BATCH * CLASSES);
    for label in labels {
        values.extend(one_hot_label::<CLASSES, E>(label));
    }

    Tensor2D::<BATCH, CLASSES, E, BK>::from_vec(values).unwrap()
}

fn shuffle_indices(indices: &mut [usize], rng: &mut SmallRng) {
    for index in (1..indices.len()).rev() {
        let swap_with = (rng.next_u64() as usize) % (index + 1);
        indices.swap(index, swap_with);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DataLoader, Dataset, ImageOneHotClassification, OneHotClassification, one_hot_label,
        one_hot_labels,
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

    #[test]
    fn one_hot_helpers_build_expected_targets() {
        assert_eq!(one_hot_label::<4, f32>(2), [0.0, 0.0, 1.0, 0.0]);

        let labels = one_hot_labels::<2, 3, f32, crate::backend::Cpu>([2, 0]);

        assert_eq!(labels.shape(), &[2, 3]);
        assert_eq!(labels.to_vec(), vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn identity_collate_yields_arrays_of_items() {
        let loader = DataLoader::new(flat_dataset()).batch_size::<2>();
        let batches: Vec<[([f32; 3], u8); 2]> = loader.iter().collect();

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0], [([1.0, 2.0, 3.0], 0), ([4.0, 5.0, 6.0], 1)]);
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
