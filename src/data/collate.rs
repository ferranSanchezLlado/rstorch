use crate::backend::{Backend, Cpu};
use crate::const_check::nonzero;
use crate::dtype::FloatElement;
use crate::tensor::{Tensor2D, Tensor3D};
use std::fmt::Debug;

/// Converts one fixed-size stream of dataset items into one typed batch.
pub trait Collate<Item, const BATCH: usize, E = f32, BK = Cpu>
where
    E: FloatElement,
    BK: Backend<E>,
{
    type Batch;

    fn collate<I>(&self, items: I) -> Self::Batch
    where
        I: IntoIterator<Item = Item>;
}

/// Collator that leaves a batch as an array of items.
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityCollate;

/// Collator for flat feature vectors and one-hot classification labels.
#[derive(Debug, Clone, Copy, Default)]
pub struct OneHotClassification<const FEATURES: usize, const CLASSES: usize>;

/// Collator for image-shaped samples and one-hot classification labels.
#[derive(Debug, Clone, Copy, Default)]
pub struct ImageOneHotClassification<const HEIGHT: usize, const WIDTH: usize, const CLASSES: usize>;

impl<Item, const BATCH: usize, E, BK> Collate<Item, BATCH, E, BK> for IdentityCollate
where
    E: FloatElement,
    BK: Backend<E>,
{
    type Batch = [Item; BATCH];

    fn collate<I>(&self, items: I) -> Self::Batch
    where
        I: IntoIterator<Item = Item>,
    {
        let mut items = items.into_iter();
        let batch = std::array::from_fn(|_| {
            items
                .next()
                .expect("data loader passes exactly BATCH items")
        });
        assert!(
            items.next().is_none(),
            "collator received more than BATCH items"
        );
        batch
    }
}

impl<const FEATURES: usize, const CLASSES: usize, const BATCH: usize, E, BK, L>
    Collate<([E; FEATURES], L), BATCH, E, BK> for OneHotClassification<FEATURES, CLASSES>
where
    E: FloatElement,
    BK: Backend<E>,
    L: TryInto<usize>,
    L::Error: Debug,
    [(); nonzero(CLASSES, "one_hot_label", "CLASSES")]:,
{
    type Batch = (
        Tensor2D<BATCH, FEATURES, E, BK>,
        Tensor2D<BATCH, CLASSES, E, BK>,
    );

    fn collate<I>(&self, items: I) -> Self::Batch
    where
        I: IntoIterator<Item = ([E; FEATURES], L)>,
    {
        let mut features = Vec::with_capacity(BATCH * FEATURES);
        let mut labels = [0_usize; BATCH];
        let mut count = 0;

        for (index, (sample_features, label)) in items.into_iter().enumerate() {
            assert!(index < BATCH, "collator received more than BATCH items");
            features.extend(sample_features);
            labels[index] = label_to_usize(label);
            count = index + 1;
        }
        assert_eq!(count, BATCH, "collator received fewer than BATCH items");

        (
            Tensor2D::<BATCH, FEATURES, E, BK>::from_vec(features).unwrap(),
            one_hot_label_indices::<BATCH, CLASSES, E, BK>(labels),
        )
    }
}

impl<const HEIGHT: usize, const WIDTH: usize, const CLASSES: usize, const BATCH: usize, E, BK, L>
    Collate<([[E; WIDTH]; HEIGHT], L), BATCH, E, BK>
    for ImageOneHotClassification<HEIGHT, WIDTH, CLASSES>
where
    E: FloatElement,
    BK: Backend<E>,
    L: TryInto<usize>,
    L::Error: Debug,
    [(); nonzero(CLASSES, "one_hot_label", "CLASSES")]:,
{
    type Batch = (
        Tensor3D<BATCH, HEIGHT, WIDTH, E, BK>,
        Tensor2D<BATCH, CLASSES, E, BK>,
    );

    fn collate<I>(&self, items: I) -> Self::Batch
    where
        I: IntoIterator<Item = ([[E; WIDTH]; HEIGHT], L)>,
    {
        let mut images = Vec::with_capacity(BATCH * HEIGHT * WIDTH);
        let mut labels = [0_usize; BATCH];
        let mut count = 0;

        for (index, (image, label)) in items.into_iter().enumerate() {
            assert!(index < BATCH, "collator received more than BATCH items");
            images.extend(image.into_iter().flatten());
            labels[index] = label_to_usize(label);
            count = index + 1;
        }
        assert_eq!(count, BATCH, "collator received fewer than BATCH items");

        (
            Tensor3D::<BATCH, HEIGHT, WIDTH, E, BK>::from_vec(images).unwrap(),
            one_hot_label_indices::<BATCH, CLASSES, E, BK>(labels),
        )
    }
}

/// Builds a one-hot class vector.
pub fn one_hot_label<const CLASSES: usize, E>(label: u8) -> [E; CLASSES]
where
    E: FloatElement,
    [(); nonzero(CLASSES, "one_hot_label", "CLASSES")]:,
{
    one_hot_label_index(usize::from(label))
}

fn one_hot_label_index<const CLASSES: usize, E>(class: usize) -> [E; CLASSES]
where
    E: FloatElement,
    [(); nonzero(CLASSES, "one_hot_label", "CLASSES")]:,
{
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
    [(); nonzero(CLASSES, "one_hot_label", "CLASSES")]:,
{
    one_hot_label_indices(labels.map(usize::from))
}

fn one_hot_label_indices<const BATCH: usize, const CLASSES: usize, E, BK>(
    labels: [usize; BATCH],
) -> Tensor2D<BATCH, CLASSES, E, BK>
where
    E: FloatElement,
    BK: Backend<E>,
    [(); nonzero(CLASSES, "one_hot_label", "CLASSES")]:,
{
    let mut values = Vec::with_capacity(BATCH * CLASSES);
    for label in labels {
        values.extend(one_hot_label_index::<CLASSES, E>(label));
    }

    Tensor2D::<BATCH, CLASSES, E, BK>::from_vec(values).unwrap()
}

fn label_to_usize<L>(label: L) -> usize
where
    L: TryInto<usize>,
    L::Error: Debug,
{
    label
        .try_into()
        .expect("label value cannot be represented as usize")
}

#[cfg(test)]
mod tests {
    use super::{one_hot_label, one_hot_labels};

    #[test]
    fn one_hot_helpers_build_expected_targets() {
        assert_eq!(one_hot_label::<4, f32>(2), [0.0, 0.0, 1.0, 0.0]);

        let labels = one_hot_labels::<2, 3, f32, crate::backend::Cpu>([2, 0]);

        assert_eq!(labels.shape(), &[2, 3]);
        assert_eq!(labels.to_vec(), vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
    }
}
