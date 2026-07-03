use crate::backend::{Backend, Cpu};
use crate::dtype::FloatDType;
use crate::error::{DataError, Error, Result};
use crate::shape::{AnyDim, C, D2, D4, Sym};
use crate::tensor::Tensor;
use std::marker::PhantomData;

#[derive(Debug)]
pub struct Batch;

pub type Features<const N: usize, E = f32, B = Cpu> = Tensor<D2<Sym<Batch>, C<N>>, E, B>;
pub type DynamicFeatures<E = f32, B = Cpu> = Tensor<D2<Sym<Batch>, AnyDim>, E, B>;
pub type ImageBatch<const CH: usize, const H: usize, const W: usize, E = f32, B = Cpu> =
    Tensor<D4<Sym<Batch>, C<CH>, C<H>, C<W>>, E, B>;
pub type StaticFeatures<const BATCH: usize, const N: usize, E = f32, B = Cpu> =
    Tensor<D2<C<BATCH>, C<N>>, E, B>;
pub type StaticImageBatch<
    const BATCH: usize,
    const CH: usize,
    const H: usize,
    const W: usize,
    E = f32,
    B = Cpu,
> = Tensor<D4<C<BATCH>, C<CH>, C<H>, C<W>>, E, B>;

pub fn features<const N: usize>() -> StackVecCollate<N> {
    StackVecCollate::new()
}

pub fn dynamic_features() -> StackDynVecCollate {
    StackDynVecCollate::new()
}

pub fn images<const CH: usize, const H: usize, const W: usize>() -> StackImageCollate<CH, H, W> {
    StackImageCollate::new()
}

pub fn static_features<const BATCH: usize, const N: usize>() -> StaticStackVecCollate<BATCH, N> {
    StaticStackVecCollate::new()
}

pub fn static_images<const BATCH: usize, const CH: usize, const H: usize, const W: usize>()
-> StaticStackImageCollate<BATCH, CH, H, W> {
    StaticStackImageCollate::new()
}

/// Normalizes a flattened CHW image sample with per-channel mean and stddev.
pub fn normalize_image_sample<const CH: usize, const H: usize, const W: usize, E>(
    sample: Vec<E>,
    mean: [E; CH],
    std: [E; CH],
) -> Result<Vec<E>>
where
    E: FloatDType,
{
    let expected = CH * H * W;
    if sample.len() != expected {
        return Err(DataError::InconsistentSampleShape {
            index: 0,
            expected: vec![CH, H, W],
            found: vec![sample.len()],
        }
        .into());
    }

    let mut out = Vec::with_capacity(expected);
    for ch in 0..CH {
        for offset in 0..H * W {
            let value = sample[ch * H * W + offset];
            out.push((value - mean[ch]) / std[ch]);
        }
    }
    Ok(out)
}

/// Converts a list of samples into a batch.
///
/// Built-in collators only produce dynamic runtime batch axes (`Sym<Batch>`),
/// including the final partial batch when the loader does not drop it.
///
/// [`DataLoader`]: crate::data::DataLoader
pub trait Collate<Item> {
    type Batch;
    type Error: std::error::Error + Send + Sync + 'static;

    fn collate(&self, items: Vec<Item>) -> std::result::Result<Self::Batch, Self::Error>;
}

/// Validate that every sample has length `N` and flatten them row-major.
fn flatten_vec_samples<const N: usize, E>(items: Vec<Vec<E>>) -> Result<Vec<E>>
where
    E: FloatDType,
{
    let mut data = Vec::with_capacity(items.len() * N);
    for (index, item) in items.into_iter().enumerate() {
        if item.len() != N {
            return Err(DataError::InconsistentSampleShape {
                index,
                expected: vec![N],
                found: vec![item.len()],
            }
            .into());
        }
        data.extend(item);
    }
    Ok(data)
}

fn flatten_dynamic_vec_samples<E>(items: Vec<Vec<E>>) -> Result<(Vec<E>, usize)>
where
    E: FloatDType,
{
    let width = items.first().ok_or(DataError::EmptyBatch)?.len();
    let mut data = Vec::with_capacity(items.len() * width);
    for (index, item) in items.into_iter().enumerate() {
        if item.len() != width {
            return Err(DataError::InconsistentSampleShape {
                index,
                expected: vec![width],
                found: vec![item.len()],
            }
            .into());
        }
        data.extend(item);
    }
    Ok((data, width))
}

/// Validate that every sample has `CH * H * W` elements and flatten them.
fn flatten_image_samples<E>(items: Vec<Vec<E>>, ch: usize, h: usize, w: usize) -> Result<Vec<E>>
where
    E: FloatDType,
{
    let expected = ch * h * w;
    let mut data = Vec::with_capacity(items.len() * expected);
    for (index, item) in items.into_iter().enumerate() {
        if item.len() != expected {
            return Err(DataError::InconsistentSampleShape {
                index,
                expected: vec![ch, h, w],
                found: vec![item.len()],
            }
            .into());
        }
        data.extend(item);
    }
    Ok(data)
}

/// Stacks `Vec<E>` samples into a `Tensor<D2<Sym<Batch>, C<N>>>` with a runtime
/// batch dimension.
#[derive(Debug, Clone, Copy, Default)]
pub struct StackVecCollate<const N: usize, E = f32, B = Cpu>(PhantomData<(E, B)>);

impl<const N: usize, E, B> StackVecCollate<N, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<const N: usize, E, B> Collate<Vec<E>> for StackVecCollate<N, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    type Batch = Features<N, E, B>;
    type Error = Error;

    fn collate(&self, items: Vec<Vec<E>>) -> Result<Self::Batch> {
        if items.is_empty() {
            return Err(DataError::EmptyBatch.into());
        }

        let batch = items.len();
        let data = flatten_vec_samples::<N, E>(items)?;
        Tensor::from_vec_with_shape(data, [batch, N])
    }
}

impl<const N: usize, E, B, L> Collate<(Vec<E>, L)> for StackVecCollate<N, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    type Batch = (Features<N, E, B>, Vec<L>);
    type Error = Error;

    fn collate(&self, items: Vec<(Vec<E>, L)>) -> Result<Self::Batch> {
        let (features, labels): (Vec<_>, Vec<_>) = items.into_iter().unzip();
        Ok((self.collate(features)?, labels))
    }
}

impl<I0, I1, C0, C1> Collate<(I0, I1)> for (C0, C1)
where
    C0: Collate<I0>,
    C0::Error: Into<Error>,
    C1: Collate<I1>,
    C1::Error: Into<Error>,
{
    type Batch = (C0::Batch, C1::Batch);
    type Error = Error;

    fn collate(&self, items: Vec<(I0, I1)>) -> Result<Self::Batch> {
        let (left, right): (Vec<_>, Vec<_>) = items.into_iter().unzip();
        Ok((
            self.0.collate(left).map_err(Into::into)?,
            self.1.collate(right).map_err(Into::into)?,
        ))
    }
}

/// Stacks `Vec<E>` samples into a `Tensor<D2<Sym<Batch>, AnyDim>>`, validating
/// that all samples share the same runtime feature width.
#[derive(Debug, Clone, Copy, Default)]
pub struct StackDynVecCollate<E = f32, B = Cpu>(PhantomData<(E, B)>);

impl<E, B> StackDynVecCollate<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<E, B> Collate<Vec<E>> for StackDynVecCollate<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    type Batch = DynamicFeatures<E, B>;
    type Error = Error;

    fn collate(&self, items: Vec<Vec<E>>) -> Result<Self::Batch> {
        let batch = items.len();
        let (data, width) = flatten_dynamic_vec_samples(items)?;
        Tensor::from_vec_with_shape(data, [batch, width])
    }
}

impl<E, B, L> Collate<(Vec<E>, L)> for StackDynVecCollate<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    type Batch = (DynamicFeatures<E, B>, Vec<L>);
    type Error = Error;

    fn collate(&self, items: Vec<(Vec<E>, L)>) -> Result<Self::Batch> {
        let (features, labels): (Vec<_>, Vec<_>) = items.into_iter().unzip();
        Ok((self.collate(features)?, labels))
    }
}

/// Stacks `Vec<E>` image samples into a `Tensor<D4<Sym<Batch>, C<CH>, C<H>,
/// C<W>>>` with a runtime batch dimension.
#[derive(Debug, Clone, Copy, Default)]
pub struct StackImageCollate<const CH: usize, const H: usize, const W: usize, E = f32, B = Cpu>(
    PhantomData<(E, B)>,
);

impl<const CH: usize, const H: usize, const W: usize, E, B> StackImageCollate<CH, H, W, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<const CH: usize, const H: usize, const W: usize, E, B> Collate<Vec<E>>
    for StackImageCollate<CH, H, W, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    type Batch = ImageBatch<CH, H, W, E, B>;
    type Error = Error;

    fn collate(&self, items: Vec<Vec<E>>) -> Result<Self::Batch> {
        if items.is_empty() {
            return Err(DataError::EmptyBatch.into());
        }

        let batch = items.len();
        let data = flatten_image_samples(items, CH, H, W)?;
        Tensor::from_vec_with_shape(data, [batch, CH, H, W])
    }
}

impl<const CH: usize, const H: usize, const W: usize, E, B, L> Collate<(Vec<E>, L)>
    for StackImageCollate<CH, H, W, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    type Batch = (ImageBatch<CH, H, W, E, B>, Vec<L>);
    type Error = Error;

    fn collate(&self, items: Vec<(Vec<E>, L)>) -> Result<Self::Batch> {
        let (images, labels): (Vec<_>, Vec<_>) = items.into_iter().unzip();
        Ok((self.collate(images)?, labels))
    }
}

/// Stacks exactly `BATCH` `Vec<E>` samples into a fully static 2D tensor.
#[derive(Debug, Clone, Copy, Default)]
pub struct StaticStackVecCollate<const BATCH: usize, const N: usize, E = f32, B = Cpu>(
    PhantomData<(E, B)>,
);

impl<const BATCH: usize, const N: usize, E, B> StaticStackVecCollate<BATCH, N, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<const BATCH: usize, const N: usize, E, B> Collate<Vec<E>>
    for StaticStackVecCollate<BATCH, N, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    type Batch = StaticFeatures<BATCH, N, E, B>;
    type Error = Error;

    fn collate(&self, items: Vec<Vec<E>>) -> Result<Self::Batch> {
        if items.len() != BATCH {
            return Err(DataError::WrongBatchSize {
                expected: BATCH,
                found: items.len(),
            }
            .into());
        }

        Tensor::from_vec(flatten_vec_samples::<N, E>(items)?)
    }
}

impl<const BATCH: usize, const N: usize, E, B, L> Collate<(Vec<E>, L)>
    for StaticStackVecCollate<BATCH, N, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    type Batch = (StaticFeatures<BATCH, N, E, B>, Vec<L>);
    type Error = Error;

    fn collate(&self, items: Vec<(Vec<E>, L)>) -> Result<Self::Batch> {
        let (features, labels): (Vec<_>, Vec<_>) = items.into_iter().unzip();
        Ok((self.collate(features)?, labels))
    }
}

/// Stacks exactly `BATCH` image samples into a fully static 4D tensor.
#[derive(Debug, Clone, Copy, Default)]
pub struct StaticStackImageCollate<
    const BATCH: usize,
    const CH: usize,
    const H: usize,
    const W: usize,
    E = f32,
    B = Cpu,
>(PhantomData<(E, B)>);

impl<const BATCH: usize, const CH: usize, const H: usize, const W: usize, E, B>
    StaticStackImageCollate<BATCH, CH, H, W, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn new() -> Self {
        Self(PhantomData)
    }
}

impl<const BATCH: usize, const CH: usize, const H: usize, const W: usize, E, B> Collate<Vec<E>>
    for StaticStackImageCollate<BATCH, CH, H, W, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    type Batch = StaticImageBatch<BATCH, CH, H, W, E, B>;
    type Error = Error;

    fn collate(&self, items: Vec<Vec<E>>) -> Result<Self::Batch> {
        if items.len() != BATCH {
            return Err(DataError::WrongBatchSize {
                expected: BATCH,
                found: items.len(),
            }
            .into());
        }

        Tensor::from_vec(flatten_image_samples(items, CH, H, W)?)
    }
}

impl<const BATCH: usize, const CH: usize, const H: usize, const W: usize, E, B, L>
    Collate<(Vec<E>, L)> for StaticStackImageCollate<BATCH, CH, H, W, E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    type Batch = (StaticImageBatch<BATCH, CH, H, W, E, B>, Vec<L>);
    type Error = Error;

    fn collate(&self, items: Vec<(Vec<E>, L)>) -> Result<Self::Batch> {
        let (images, labels): (Vec<_>, Vec<_>) = items.into_iter().unzip();
        Ok((self.collate(images)?, labels))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    #[test]
    fn collate_creates_symbolic_batch_tensor() {
        let batch: Tensor<D2<Sym<Batch>, C<3>>> = StackVecCollate::<3>::new()
            .collate(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]])
            .unwrap();

        assert_eq!(batch.shape().dims(), &[2, 3]);
        assert_eq!(batch.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn collate_rejects_inconsistent_sample_lengths() {
        let err = StackVecCollate::<2>::new()
            .collate(vec![vec![1.0, 2.0], vec![3.0]])
            .unwrap_err();

        assert!(matches!(
            err,
            Error::Data(DataError::InconsistentSampleShape { index: 1, .. })
        ));
    }

    #[test]
    fn collate_rejects_empty_batch_directly() {
        let err = StackVecCollate::<2>::new()
            .collate(Vec::<Vec<f32>>::new())
            .unwrap_err();

        assert!(matches!(err, Error::Data(DataError::EmptyBatch)));
    }

    #[test]
    fn dynamic_width_collate_creates_any_dim_tensor() {
        let batch: Tensor<D2<Sym<Batch>, AnyDim>> = StackDynVecCollate::new()
            .collate(vec![vec![1.0, 2.0], vec![3.0, 4.0]])
            .unwrap();

        assert_eq!(batch.shape().dims(), &[2, 2]);
        assert_eq!(batch.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn dynamic_width_collate_rejects_mismatched_widths() {
        let err = StackDynVecCollate::<f32>::new()
            .collate(vec![vec![1.0, 2.0], vec![3.0]])
            .unwrap_err();

        assert!(matches!(
            err,
            Error::Data(DataError::InconsistentSampleShape {
                index: 1,
                expected,
                found
            }) if expected == vec![2] && found == vec![1]
        ));
    }

    #[test]
    fn constructor_functions_create_typed_collators() {
        let batch: Features<2> = features::<2>()
            .collate(vec![vec![1.0, 2.0], vec![3.0, 4.0]])
            .unwrap();

        assert_eq!(batch.shape().dims(), &[2, 2]);
        assert_eq!(batch.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

        let batch: DynamicFeatures = dynamic_features()
            .collate(vec![vec![1.0], vec![2.0]])
            .unwrap();
        assert_eq!(batch.shape().dims(), &[2, 1]);

        let batch: ImageBatch<1, 1, 2> = images::<1, 1, 2>()
            .collate(vec![vec![1.0, 2.0], vec![3.0, 4.0]])
            .unwrap();
        assert_eq!(batch.shape().dims(), &[2, 1, 1, 2]);
    }

    #[test]
    fn tuple_collator_batches_both_sides() {
        let batch: (Features<2>, Features<1>) = (features::<2>(), features::<1>())
            .collate(vec![
                (vec![1.0, 2.0], vec![10.0]),
                (vec![3.0, 4.0], vec![20.0]),
            ])
            .unwrap();

        assert_eq!(batch.0.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(batch.1.to_vec().unwrap(), vec![10.0, 20.0]);
    }

    #[test]
    fn static_collator_requires_exact_batch_size() {
        let batch: StaticFeatures<2, 2> = static_features::<2, 2>()
            .collate(vec![vec![1.0, 2.0], vec![3.0, 4.0]])
            .unwrap();
        assert_eq!(batch.shape().dims(), &[2, 2]);

        let err = static_features::<3, 2>()
            .collate(vec![vec![1.0, 2.0], vec![3.0, 4.0]])
            .unwrap_err();
        assert!(matches!(
            err,
            Error::Data(DataError::WrongBatchSize {
                expected: 3,
                found: 2
            })
        ));
    }
}
