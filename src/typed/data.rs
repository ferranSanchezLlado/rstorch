//! Typed boundaries for the runtime data pipeline.
//!
//! The adapters in this module add item shape, element, and logical placement
//! contracts without replacing the runtime data implementation. Both adapters
//! implement [`Dataset`], so the re-exported [`DataLoader`] performs ordering,
//! seeded shuffling, epoch selection, tail handling, and `drop_last` exactly as
//! it does for dynamic datasets.
//!
//! A dataset is parameterized by its *item* tensor types. Its batch tensors
//! retain every item marker and insert one leading [`DYN`]. Batch size is
//! therefore always a runtime value, including for full and dropped-tail
//! batches.
//!
//! ```
//! use rstorch::typed::data::{DataLoader, TensorDataset};
//! use rstorch::typed::{Cpu, DYN, DeviceCtx, Tensor0, Tensor1, Tensor2};
//! use rstorch::Result;
//!
//! # fn main() -> Result<()> {
//! let cpu = DeviceCtx::<Cpu>::cpu()?;
//! let features = Tensor2::<DYN, 2>::from_vec(
//!     vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0],
//!     [3, 2],
//!     &cpu,
//! )?;
//! let labels = Tensor1::<DYN, i64>::from_vec(vec![0, 1, 0], [3], &cpu)?;
//! let dataset = TensorDataset::<Tensor1<2>, Tensor0<i64>>::new(features, labels)?;
//!
//! let batches: Vec<_> = DataLoader::new(dataset, 2).batches().collect();
//! let (full_x, _) = batches[0].as_ref().unwrap();
//! let (tail_x, _) = batches[1].as_ref().unwrap();
//! assert_eq!(full_x.dims(), [2, 2]);
//! assert_eq!(tail_x.dims(), [1, 2]);
//! # Ok(())
//! # }
//! ```
//!
//! Item rank 7 maps to batch rank 8, the typed ceiling. Item rank 8 does not
//! implement [`BatchItem`], so it has no typed dataset or loader boundary:
//!
//! ```compile_fail
//! use rstorch::typed::data::VecDataset;
//! use rstorch::typed::{Cpu, Tensor0, Tensor8};
//!
//! type Rank8 = Tensor8<1, 1, 1, 1, 1, 1, 1, 1, f32, Cpu>;
//! let _ = VecDataset::<Rank8, Tensor0<i64, Cpu>>::new(Vec::new());
//! ```

use super::device::validate_binding;
use super::tensor::checked_wrap;
use super::{
    DYN, DeviceBinding, DeviceCtx, Placement, Tensor0, Tensor1, Tensor2, Tensor3, Tensor4, Tensor5,
    Tensor6, Tensor7, Tensor8, TypedTensor,
};
use crate::{Element, Error, Result, Tensor};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

pub use crate::data::{DataLoader, Dataset};

type ErasedStage = Arc<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>;
type TransformPipeline = Arc<Mutex<Vec<ErasedStage>>>;

mod sealed {
    pub trait BatchItem {}
    pub trait DatasetColumn {}
}

/// A sealed item tensor that can gain a leading runtime batch axis.
///
/// Implementations exist for [`Tensor0`] through [`Tensor7`]. The associated
/// [`Batch`](BatchItem::Batch) preserves the item's dimensions, element type,
/// and logical placement exactly while inserting [`DYN`] at axis zero.
pub trait BatchItem: sealed::BatchItem + TypedTensor {
    /// The precise tensor produced by collating this item type.
    type Batch: TypedTensor<Elem = Self::Elem, Placement = Self::Placement>;
}

/// A sealed whole-dataset tensor whose leading axis is the item count.
///
/// Implementations exist for [`Tensor1`] through [`Tensor8`]. The leading
/// marker may be static or [`DYN`], but [`Item`](DatasetColumn::Item) removes it
/// and batching always rebuilds it as [`DYN`] through [`BatchItem::Batch`].
pub trait DatasetColumn: sealed::DatasetColumn + TypedTensor {
    /// The exact unbatched item contract carried after the leading item axis.
    type Item: BatchItem<Elem = Self::Elem, Placement = Self::Placement>;
}

macro_rules! impl_batch_items {
    ($(($item:ident, [$($dim:ident),*] => $batch:ty)),+ $(,)?) => {
        $(
            impl<$(const $dim: usize,)* E: Element, P: Placement> sealed::BatchItem
                for $item<$($dim,)* E, P>
            {}

            impl<$(const $dim: usize,)* E: Element, P: Placement> BatchItem
                for $item<$($dim,)* E, P>
            {
                type Batch = $batch;
            }
        )+
    };
}

impl_batch_items! {
    (Tensor0, [] => Tensor1<DYN, E, P>),
    (Tensor1, [D0] => Tensor2<DYN, D0, E, P>),
    (Tensor2, [D0, D1] => Tensor3<DYN, D0, D1, E, P>),
    (Tensor3, [D0, D1, D2] => Tensor4<DYN, D0, D1, D2, E, P>),
    (Tensor4, [D0, D1, D2, D3] => Tensor5<DYN, D0, D1, D2, D3, E, P>),
    (Tensor5, [D0, D1, D2, D3, D4] => Tensor6<DYN, D0, D1, D2, D3, D4, E, P>),
    (Tensor6, [D0, D1, D2, D3, D4, D5] => Tensor7<DYN, D0, D1, D2, D3, D4, D5, E, P>),
    (Tensor7, [D0, D1, D2, D3, D4, D5, D6] => Tensor8<DYN, D0, D1, D2, D3, D4, D5, D6, E, P>),
}

macro_rules! impl_dataset_columns {
    ($(($column:ident, [$count:ident $(, $dim:ident)*] => $item:ty)),+ $(,)?) => {
        $(
            impl<const $count: usize, $(const $dim: usize,)* E: Element, P: Placement>
                sealed::DatasetColumn for $column<$count, $($dim,)* E, P>
            {}

            impl<const $count: usize, $(const $dim: usize,)* E: Element, P: Placement>
                DatasetColumn for $column<$count, $($dim,)* E, P>
            {
                type Item = $item;
            }
        )+
    };
}

impl_dataset_columns! {
    (Tensor1, [N] => Tensor0<E, P>),
    (Tensor2, [N, D0] => Tensor1<D0, E, P>),
    (Tensor3, [N, D0, D1] => Tensor2<D0, D1, E, P>),
    (Tensor4, [N, D0, D1, D2] => Tensor3<D0, D1, D2, E, P>),
    (Tensor5, [N, D0, D1, D2, D3] => Tensor4<D0, D1, D2, D3, E, P>),
    (Tensor6, [N, D0, D1, D2, D3, D4] => Tensor5<D0, D1, D2, D3, D4, E, P>),
    (Tensor7, [N, D0, D1, D2, D3, D4, D5] => Tensor6<D0, D1, D2, D3, D4, D5, E, P>),
    (Tensor8, [N, D0, D1, D2, D3, D4, D5, D6] => Tensor7<D0, D1, D2, D3, D4, D5, D6, E, P>),
}

fn validate_pair<F, L>(
    feature_binding: &Arc<DeviceBinding>,
    label_binding: &Arc<DeviceBinding>,
    op: &'static str,
) -> Result<()>
where
    F: BatchItem,
    L: BatchItem + TypedTensor<Placement = F::Placement>,
{
    validate_binding::<F::Placement>(feature_binding, op)?;
    validate_binding::<L::Placement>(label_binding, op)?;
    if !Arc::ptr_eq(feature_binding, label_binding) {
        return Err(Error::InvalidArg {
            op,
            msg: "features and labels must share one canonical placement binding".into(),
        });
    }
    Ok(())
}

fn checked_batch<F, L>(
    batch: (Tensor, Tensor),
    feature_binding: &Arc<DeviceBinding>,
    label_binding: &Arc<DeviceBinding>,
    op: &'static str,
) -> Result<(F::Batch, L::Batch)>
where
    F: BatchItem,
    L: BatchItem + TypedTensor<Placement = F::Placement>,
{
    validate_pair::<F, L>(feature_binding, label_binding, op)?;
    Ok((
        checked_wrap::<F::Batch>(batch.0, Arc::clone(feature_binding), op)?,
        checked_wrap::<L::Batch>(batch.1, Arc::clone(label_binding), op)?,
    ))
}

/// Typed adapter over a runtime [`crate::data::TensorDataset`].
///
/// `F` and `L` describe one feature and label item respectively. Construction
/// accepts whole-dataset tensors with any leading item-count marker. The
/// runtime dataset retains them without copying and performs each batch as
/// device-side index selection, preserving the autograd graph. Regardless of
/// the source item-count marker, output batches always use leading [`DYN`].
pub struct TensorDataset<F, L>
where
    F: BatchItem,
    L: BatchItem + TypedTensor<Placement = F::Placement>,
{
    inner: crate::data::TensorDataset,
    feature_binding: Arc<DeviceBinding>,
    label_binding: Arc<DeviceBinding>,
    marker: PhantomData<(F, L)>,
}

impl<F, L> TensorDataset<F, L>
where
    F: BatchItem,
    L: BatchItem + TypedTensor<Placement = F::Placement>,
{
    /// Creates a typed tensor dataset without copying or moving tensor data.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArg`] if features and labels do not carry the
    /// same canonical placement identity. Runtime item-count and device checks
    /// are delegated to [`crate::data::TensorDataset::new`].
    pub fn new<X, Y>(features: X, labels: Y) -> Result<Self>
    where
        X: DatasetColumn<Item = F>,
        Y: DatasetColumn<Item = L> + TypedTensor<Placement = X::Placement>,
    {
        const OP: &str = "typed::data::TensorDataset::new";
        let feature_binding = Arc::clone(features.binding());
        let label_binding = Arc::clone(labels.binding());
        validate_pair::<F, L>(&feature_binding, &label_binding, OP)?;
        let inner =
            crate::data::TensorDataset::new(features.dynamic().clone(), labels.dynamic().clone())?;
        Ok(Self {
            inner,
            feature_binding,
            label_binding,
            marker: PhantomData,
        })
    }

    /// Checks and wraps a runtime tensor dataset without copying its tensors.
    ///
    /// The caller names the feature and label item contracts in `F` and `L`.
    /// Their ranks, static dimensions, element types, runtime device, and the
    /// context's canonical placement identity are checked immediately.
    ///
    /// # Errors
    ///
    /// Returns the same structured rank, shape, dtype, device, or canonical
    /// binding errors as typed tensor conversion.
    pub fn try_from_dynamic(
        inner: crate::data::TensorDataset,
        ctx: &DeviceCtx<F::Placement>,
    ) -> Result<Self> {
        const OP: &str = "typed::data::TensorDataset::try_from_dynamic";
        checked_wrap::<F::Batch>(inner.inputs().clone(), Arc::clone(ctx.binding()), OP)?;
        checked_wrap::<L::Batch>(inner.targets().clone(), Arc::clone(ctx.binding()), OP)?;
        Ok(Self {
            inner,
            feature_binding: Arc::clone(ctx.binding()),
            label_binding: Arc::clone(ctx.binding()),
            marker: PhantomData,
        })
    }

    /// Borrows the unchanged runtime dataset.
    pub fn as_dynamic(&self) -> &crate::data::TensorDataset {
        &self.inner
    }

    /// Removes compile-time metadata without copying the stored tensors.
    pub fn into_dynamic(self) -> crate::data::TensorDataset {
        self.inner
    }
}

impl<F, L> std::fmt::Debug for TensorDataset<F, L>
where
    F: BatchItem,
    L: BatchItem + TypedTensor<Placement = F::Placement>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorDataset")
            .field("inner", &self.inner)
            .field("feature_item", &std::any::type_name::<F>())
            .field("label_item", &std::any::type_name::<L>())
            .finish()
    }
}

impl<F, L> Dataset for TensorDataset<F, L>
where
    F: BatchItem,
    L: BatchItem + TypedTensor<Placement = F::Placement>,
{
    type Batch = (F::Batch, L::Batch);

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn batch(&self, indices: &[usize]) -> Result<Self::Batch> {
        let batch = self.inner.batch(indices)?;
        checked_batch::<F, L>(
            batch,
            &self.feature_binding,
            &self.label_binding,
            "typed::data::TensorDataset::batch",
        )
    }
}

/// Typed adapter over a runtime [`crate::data::VecDataset`].
///
/// Each `(F, L)` pair is an unbatched item. The runtime implementation checks
/// uniformity and collates selected items with `Tensor::stack`; this adapter
/// checks the resulting tensors and restores precise leading-`DYN` batch types.
pub struct VecDataset<F, L>
where
    F: BatchItem,
    L: BatchItem + TypedTensor<Placement = F::Placement>,
{
    inner: crate::data::VecDataset,
    bindings: Option<(Arc<DeviceBinding>, Arc<DeviceBinding>)>,
    transform: Option<TransformPipeline>,
    marker: PhantomData<(F, L)>,
}

impl<F, L> VecDataset<F, L>
where
    F: BatchItem,
    L: BatchItem + TypedTensor<Placement = F::Placement>,
{
    /// Creates a typed vector dataset from unbatched item tensors.
    ///
    /// Empty datasets are accepted. For nonempty datasets every feature and
    /// label must carry the same canonical placement identity. Shape, dtype,
    /// and runtime-device uniformity are delegated to the runtime constructor.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArg`] for a noncanonical or differing logical
    /// placement binding, plus the runtime vector dataset's uniformity errors.
    pub fn new(items: Vec<(F, L)>) -> Result<Self> {
        const OP: &str = "typed::data::VecDataset::new";
        let bindings = items
            .first()
            .map(|(feature, label)| {
                let feature_binding = Arc::clone(feature.binding());
                let label_binding = Arc::clone(label.binding());
                validate_pair::<F, L>(&feature_binding, &label_binding, OP)?;
                Ok::<_, Error>((feature_binding, label_binding))
            })
            .transpose()?;

        for (feature, label) in &items {
            validate_pair::<F, L>(feature.binding(), label.binding(), OP)?;
            if let Some((expected_feature, expected_label)) = &bindings
                && (!Arc::ptr_eq(expected_feature, feature.binding())
                    || !Arc::ptr_eq(expected_label, label.binding()))
            {
                return Err(Error::InvalidArg {
                    op: OP,
                    msg: "all items must share one canonical placement binding".into(),
                });
            }
        }

        let dynamic = items
            .iter()
            .map(|(feature, label)| (feature.dynamic().clone(), label.dynamic().clone()))
            .collect();
        Ok(Self {
            inner: crate::data::VecDataset::new(dynamic)?,
            bindings,
            transform: None,
            marker: PhantomData,
        })
    }

    /// Checks and wraps a runtime vector dataset without copying its items.
    ///
    /// # Errors
    ///
    /// Every stored item is checked against `F`, `L`, and the canonical
    /// context, so wrong rank, static dimensions, dtype, device, or placement
    /// identity is rejected before loading starts.
    pub fn try_from_dynamic(
        inner: crate::data::VecDataset,
        ctx: &DeviceCtx<F::Placement>,
    ) -> Result<Self> {
        const OP: &str = "typed::data::VecDataset::try_from_dynamic";
        for (feature, label) in inner.items() {
            checked_wrap::<F>(feature.clone(), Arc::clone(ctx.binding()), OP)?;
            checked_wrap::<L>(label.clone(), Arc::clone(ctx.binding()), OP)?;
        }
        let bindings = (!inner.items().is_empty())
            .then(|| (Arc::clone(ctx.binding()), Arc::clone(ctx.binding())));
        Ok(Self {
            inner,
            bindings,
            transform: None,
            marker: PhantomData,
        })
    }

    /// Appends a typed per-item feature transform.
    ///
    /// The runtime [`crate::data::VecDataset::transform`] hook still controls
    /// when and how often the transform runs. At batch time its runtime input
    /// is checked and wrapped as `F`, `transform` may produce any batchable `G`
    /// on the same logical placement, and the result is erased without a host
    /// read or device transfer. Consequently a shape- or element-changing
    /// transform changes this dataset's feature batch type from `F::Batch` to
    /// `G::Batch`, retaining the leading [`DYN`] marker.
    ///
    /// Typed transform calls compose in call order, including when `F` and `G`
    /// are the same type. This differs deliberately from calling the runtime
    /// transform builder repeatedly, where the latest closure replaces the
    /// previous closure: generic typed chaining requires each new closure to
    /// receive the preceding closure's output. The composed erased pipeline is
    /// reinstalled as one runtime transform over the unchanged original items.
    /// Appending a stage is O(1); transforming an item is O(number of stages)
    /// and briefly locks the private pipeline to clone its flat vector of stage
    /// `Arc`s. User stages run after that lock is released, so concurrent and
    /// reentrant batches do not serialize on it. Pipeline destruction remains
    /// iterative.
    ///
    /// Transform and boundary errors remain errors from [`Dataset::batch`]. An
    /// empty dataset has no binding and can never invoke an item transform; it
    /// remains empty and the closure is never invoked.
    #[must_use]
    pub fn transform<G>(
        self,
        transform: impl Fn(&F) -> Result<G> + Send + Sync + 'static,
    ) -> VecDataset<G, L>
    where
        G: BatchItem + TypedTensor<Placement = F::Placement>,
    {
        const OP: &str = "typed::data::VecDataset::transform";
        let Self {
            inner,
            bindings,
            transform: previous,
            marker: _,
        } = self;

        let feature_binding = bindings
            .as_ref()
            .map(|(feature_binding, _)| Arc::clone(feature_binding));
        let stage: ErasedStage = Arc::new(move |feature| {
            let feature_binding = feature_binding.as_ref().ok_or_else(|| Error::InvalidArg {
                op: OP,
                msg: "a transform input requires a nonempty dataset binding".into(),
            })?;
            let feature = checked_wrap::<F>(feature.clone(), Arc::clone(feature_binding), OP)?;
            let output = transform(&feature)?;
            validate_binding::<G::Placement>(output.binding(), OP)?;
            if !Arc::ptr_eq(output.binding(), feature_binding) {
                return Err(Error::InvalidArg {
                    op: OP,
                    msg: "transform output must retain the dataset's canonical placement binding"
                        .into(),
                });
            }
            Ok(output.dynamic().clone())
        });
        let pipeline = previous.unwrap_or_else(|| Arc::new(Mutex::new(Vec::new())));
        pipeline
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(stage);

        let runtime_pipeline = Arc::clone(&pipeline);
        let inner = inner.transform(move |original| {
            let stages = {
                let stages = runtime_pipeline
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                stages.clone()
            };
            let mut current = original.clone();
            for stage in &stages {
                current = stage(&current)?;
            }
            Ok(current)
        });

        VecDataset {
            inner,
            bindings,
            transform: Some(pipeline),
            marker: PhantomData,
        }
    }

    /// Borrows the unchanged runtime dataset.
    pub fn as_dynamic(&self) -> &crate::data::VecDataset {
        &self.inner
    }

    /// Removes compile-time metadata without copying stored tensors.
    pub fn into_dynamic(self) -> crate::data::VecDataset {
        self.inner
    }
}

impl<F, L> std::fmt::Debug for VecDataset<F, L>
where
    F: BatchItem,
    L: BatchItem + TypedTensor<Placement = F::Placement>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VecDataset")
            .field("inner", &self.inner)
            .field("typed_transform", &self.transform.is_some())
            .field("feature_item", &std::any::type_name::<F>())
            .field("label_item", &std::any::type_name::<L>())
            .finish()
    }
}

impl<F, L> Dataset for VecDataset<F, L>
where
    F: BatchItem,
    L: BatchItem + TypedTensor<Placement = F::Placement>,
{
    type Batch = (F::Batch, L::Batch);

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn batch(&self, indices: &[usize]) -> Result<Self::Batch> {
        let batch = self.inner.batch(indices)?;
        let (feature_binding, label_binding) = self
            .bindings
            .as_ref()
            .expect("a runtime batch implies a nonempty typed VecDataset");
        checked_batch::<F, L>(
            batch,
            feature_binding,
            label_binding,
            "typed::data::VecDataset::batch",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed::Cpu;
    use crate::typed::sealed::TypedTensor as SealedTypedTensor;
    use crate::{DType, Device, Grads};

    fn cpu() -> DeviceCtx<Cpu> {
        DeviceCtx::cpu().unwrap()
    }

    fn runtime_tensor_dataset(n: usize) -> crate::data::TensorDataset {
        let features = Tensor::from_vec(
            (0..n * 2).map(|value| value as f32).collect::<Vec<_>>(),
            [n, 2],
            &Device::Cpu,
        )
        .unwrap();
        let labels = Tensor::from_vec(
            (0..n).map(|value| value as i64).collect::<Vec<_>>(),
            [n],
            &Device::Cpu,
        )
        .unwrap();
        crate::data::TensorDataset::new(features, labels).unwrap()
    }

    fn typed_tensor_dataset(n: usize) -> TensorDataset<Tensor1<2>, Tensor0<i64>> {
        TensorDataset::try_from_dynamic(runtime_tensor_dataset(n), &cpu()).unwrap()
    }

    fn labels<I>(batches: I) -> Vec<Vec<i64>>
    where
        I: IntoIterator<Item = Result<(Tensor2<DYN, 2>, Tensor1<DYN, i64>)>>,
    {
        batches
            .into_iter()
            .map(|batch| batch.unwrap().1.to_vec().unwrap())
            .collect()
    }

    #[test]
    fn tensor_dataset_matches_runtime_for_selected_and_repeated_items() {
        let runtime = runtime_tensor_dataset(4);
        let typed =
            TensorDataset::<Tensor1<2>, Tensor0<i64>>::try_from_dynamic(runtime.clone(), &cpu())
                .unwrap();
        let indices = [3, 1, 1];
        let dynamic_batch = runtime.batch(&indices).unwrap();
        let typed_batch = typed.batch(&indices).unwrap();

        assert_eq!(typed_batch.0.dims(), [3, 2]);
        assert_eq!(typed_batch.1.dims(), [3]);
        assert_eq!(
            typed_batch.0.to_vec().unwrap(),
            dynamic_batch.0.to_vec::<f32>().unwrap()
        );
        assert_eq!(
            typed_batch.1.to_vec().unwrap(),
            dynamic_batch.1.to_vec::<i64>().unwrap()
        );
    }

    #[test]
    fn vec_dataset_matches_runtime_collation() {
        let ctx = cpu();
        let typed_items = (0..4)
            .map(|value| {
                (
                    Tensor1::<2>::from_vec(vec![value as f32; 2], [2], &ctx).unwrap(),
                    Tensor0::<i64>::from_vec(vec![value], [], &ctx).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let runtime_items = typed_items
            .iter()
            .map(|(x, y)| (x.as_dynamic().clone(), y.as_dynamic().clone()))
            .collect();
        let runtime = crate::data::VecDataset::new(runtime_items).unwrap();
        let typed = VecDataset::new(typed_items).unwrap();

        let expected = runtime.batch(&[2, 0, 2]).unwrap();
        let actual = typed.batch(&[2, 0, 2]).unwrap();
        assert_eq!(actual.0.dims(), [3, 2]);
        assert_eq!(actual.1.dims(), [3]);
        assert_eq!(
            actual.0.to_vec().unwrap(),
            expected.0.to_vec::<f32>().unwrap()
        );
        assert_eq!(
            actual.1.to_vec().unwrap(),
            expected.1.to_vec::<i64>().unwrap()
        );
    }

    #[test]
    fn vec_dataset_transform_changes_typed_shape_and_element() {
        let ctx = cpu();
        let items = (0..3)
            .map(|value| {
                (
                    Tensor1::<2>::from_vec(vec![value as f32; 2], [2], &ctx).unwrap(),
                    Tensor0::<i64>::from_vec(vec![value], [], &ctx).unwrap(),
                )
            })
            .collect();
        let dataset = VecDataset::new(items)
            .unwrap()
            .transform::<Tensor2<1, 2, bool>>(|feature| {
                let reshaped = feature.reshape::<Tensor2<1, 2>>([1, 2])?;
                reshaped.eq(&reshaped)
            });

        let (features, labels): (Tensor3<DYN, 1, 2, bool>, Tensor1<DYN, i64>) =
            dataset.batch(&[2, 0]).unwrap();
        assert_eq!(features.dims(), [2, 1, 2]);
        assert_eq!(features.to_vec().unwrap(), vec![true; 4]);
        assert_eq!(labels.to_vec().unwrap(), vec![2, 0]);
    }

    #[test]
    fn vec_dataset_chains_rank_shape_and_element_changes() {
        let ctx = cpu();
        let dataset = VecDataset::new(vec![(
            Tensor1::<2>::from_vec(vec![1.0, 2.0], [2], &ctx).unwrap(),
            Tensor0::<i64>::from_vec(vec![7], [], &ctx).unwrap(),
        )])
        .unwrap()
        .transform::<Tensor2<1, 2>>(|feature| feature.reshape([1, 2]))
        .transform::<Tensor1<2, bool>>(|feature| {
            let flattened = feature.reshape::<Tensor1<2>>([2])?;
            flattened.eq(&flattened)
        });

        let (features, labels): (Tensor2<DYN, 2, bool>, Tensor1<DYN, i64>) =
            dataset.batch(&[0]).unwrap();
        assert_eq!(features.dims(), [1, 2]);
        assert_eq!(features.to_vec().unwrap(), vec![true, true]);
        assert_eq!(labels.to_vec().unwrap(), vec![7]);
    }

    #[test]
    fn same_type_typed_transforms_compose_in_call_order() {
        let ctx = cpu();
        let dataset = VecDataset::new(vec![(
            Tensor1::<2>::from_vec(vec![1.0, 2.0], [2], &ctx).unwrap(),
            Tensor0::<i64>::from_vec(vec![0], [], &ctx).unwrap(),
        )])
        .unwrap()
        .transform::<Tensor1<2>>(|feature| feature.add_scalar(1.0))
        .transform::<Tensor1<2>>(|feature| feature.mul_scalar(3.0));

        assert_eq!(
            dataset.batch(&[0]).unwrap().0.to_vec().unwrap(),
            vec![6.0, 9.0]
        );
    }

    #[test]
    fn chained_transform_errors_stop_at_the_first_failing_stage() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let ctx = cpu();
        let later_calls = Arc::new(AtomicUsize::new(0));
        let calls = Arc::clone(&later_calls);
        let dataset = VecDataset::new(vec![(
            Tensor1::<2>::from_vec(vec![1.0, 2.0], [2], &ctx).unwrap(),
            Tensor0::<i64>::from_vec(vec![0], [], &ctx).unwrap(),
        )])
        .unwrap()
        .transform::<Tensor1<2>>(|_| {
            Err(Error::InvalidArg {
                op: "first_transform",
                msg: "first stage failed".into(),
            })
        })
        .transform::<Tensor1<2>>(move |feature| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(feature.clone())
        });

        assert!(matches!(
            dataset.batch(&[0]),
            Err(Error::InvalidArg {
                op: "first_transform",
                ..
            })
        ));
        assert_eq!(later_calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn into_dynamic_retains_the_complete_chained_transform() {
        let ctx = cpu();
        let dataset = VecDataset::new(vec![(
            Tensor1::<2>::from_vec(vec![1.0, 2.0], [2], &ctx).unwrap(),
            Tensor0::<i64>::from_vec(vec![3], [], &ctx).unwrap(),
        )])
        .unwrap()
        .transform::<Tensor2<1, 2>>(|feature| feature.reshape([1, 2]))
        .transform::<Tensor2<1, 2>>(|feature| feature.mul_scalar(4.0));

        let typed = dataset.batch(&[0]).unwrap();
        let dynamic = dataset.into_dynamic().batch(&[0]).unwrap();
        assert_eq!(dynamic.0.dims(), typed.0.dims());
        assert_eq!(
            dynamic.0.to_vec::<f32>().unwrap(),
            typed.0.to_vec().unwrap()
        );
        assert_eq!(
            dynamic.1.to_vec::<i64>().unwrap(),
            typed.1.to_vec().unwrap()
        );
    }

    #[test]
    fn long_transform_pipeline_runs_and_drops_on_a_small_stack() {
        const STAGES: usize = 5_000;
        const STACK_BYTES: usize = 256 * 1024;

        std::thread::Builder::new()
            .name("typed-transform-small-stack".into())
            .stack_size(STACK_BYTES)
            .spawn(|| {
                let ctx = cpu();
                let mut dataset = VecDataset::new(vec![(
                    Tensor1::<1>::from_vec(vec![3.0], [1], &ctx).unwrap(),
                    Tensor0::<i64>::from_vec(vec![4], [], &ctx).unwrap(),
                )])
                .unwrap();

                // Identity stages isolate pipeline traversal and destruction
                // from tensor-kernel costs while retaining all typed checks.
                for _ in 0..STAGES {
                    dataset = dataset.transform::<Tensor1<1>>(|feature| Ok(feature.clone()));
                }

                let batch = dataset.batch(&[0]).unwrap();
                assert_eq!(batch.0.to_vec().unwrap(), vec![3.0]);
                assert_eq!(batch.1.to_vec().unwrap(), vec![4]);
                drop(dataset);
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn concurrent_batches_can_overlap_inside_a_user_transform() {
        use std::sync::{Barrier, Condvar};
        use std::time::Duration;

        let ctx = cpu();
        let overlap = Arc::new((Mutex::new(0usize), Condvar::new()));
        let transform_overlap = Arc::clone(&overlap);
        let dataset = Arc::new(
            VecDataset::new(vec![(
                Tensor1::<1>::from_vec(vec![2.0], [1], &ctx).unwrap(),
                Tensor0::<i64>::from_vec(vec![3], [], &ctx).unwrap(),
            )])
            .unwrap()
            .transform::<Tensor1<1>>(move |feature| {
                let (entered, ready) = &*transform_overlap;
                let mut entered = entered.lock().unwrap();
                *entered += 1;
                ready.notify_all();
                let (entered, timeout) = ready
                    .wait_timeout_while(entered, Duration::from_secs(2), |count| *count < 2)
                    .unwrap();
                if timeout.timed_out() {
                    return Err(Error::InvalidArg {
                        op: "concurrent_transform_test",
                        msg: "user transforms did not overlap".into(),
                    });
                }
                drop(entered);
                Ok(feature.clone())
            }),
        );
        let start = Arc::new(Barrier::new(3));
        let threads = (0..2)
            .map(|_| {
                let dataset = Arc::clone(&dataset);
                let start = Arc::clone(&start);
                std::thread::spawn(move || {
                    start.wait();
                    dataset.batch(&[0])
                })
            })
            .collect::<Vec<_>>();
        start.wait();

        for thread in threads {
            let batch = thread.join().unwrap().unwrap();
            assert_eq!(batch.0.to_vec().unwrap(), vec![2.0]);
        }
        assert_eq!(*overlap.0.lock().unwrap(), 2);
    }

    #[test]
    fn a_user_transform_can_reenter_the_same_dataset_once() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{OnceLock, Weak, mpsc};
        use std::time::Duration;

        type ReentrantDataset = VecDataset<Tensor1<1>, Tensor0<i64>>;

        let ctx = cpu();
        let self_handle = Arc::new(OnceLock::<Weak<ReentrantDataset>>::new());
        let closure_handle = Arc::clone(&self_handle);
        let reentered = Arc::new(AtomicBool::new(false));
        let closure_reentered = Arc::clone(&reentered);
        let dataset = Arc::new(
            VecDataset::new(vec![(
                Tensor1::<1>::from_vec(vec![5.0], [1], &ctx).unwrap(),
                Tensor0::<i64>::from_vec(vec![6], [], &ctx).unwrap(),
            )])
            .unwrap()
            .transform::<Tensor1<1>>(move |feature| {
                if !closure_reentered.swap(true, Ordering::SeqCst) {
                    let dataset =
                        closure_handle
                            .get()
                            .and_then(Weak::upgrade)
                            .ok_or_else(|| Error::InvalidArg {
                                op: "reentrant_transform_test",
                                msg: "dataset weak handle was unavailable".into(),
                            })?;
                    let nested = dataset.batch(&[0])?;
                    assert_eq!(nested.0.to_vec().unwrap(), vec![5.0]);
                }
                Ok(feature.clone())
            }),
        );
        self_handle.set(Arc::downgrade(&dataset)).unwrap();

        let (complete, completed) = mpsc::sync_channel(1);
        let run = Arc::clone(&dataset);
        let thread = std::thread::spawn(move || {
            complete.send(run.batch(&[0])).unwrap();
        });
        let batch = completed
            .recv_timeout(Duration::from_secs(2))
            .expect("reentrant transform deadlocked")
            .unwrap();
        thread.join().unwrap();

        assert!(reentered.load(Ordering::SeqCst));
        assert_eq!(batch.0.to_vec().unwrap(), vec![5.0]);
        assert_eq!(batch.1.to_vec().unwrap(), vec![6]);
    }

    #[test]
    fn vec_dataset_transform_errors_surface_at_batch_time() {
        let ctx = cpu();
        let dataset = VecDataset::new(vec![(
            Tensor1::<2>::from_vec(vec![1.0, 2.0], [2], &ctx).unwrap(),
            Tensor0::<i64>::from_vec(vec![0], [], &ctx).unwrap(),
        )])
        .unwrap()
        .transform::<Tensor1<2>>(|_| {
            Err(Error::InvalidArg {
                op: "intentional_transform",
                msg: "failed in typed transform".into(),
            })
        });

        assert!(matches!(
            dataset.batch(&[0]),
            Err(Error::InvalidArg {
                op: "intentional_transform",
                ..
            })
        ));
    }

    #[test]
    fn vec_dataset_transform_rejects_noncanonical_output_binding() {
        let ctx = cpu();
        let dataset = VecDataset::new(vec![(
            Tensor1::<2>::from_vec(vec![1.0, 2.0], [2], &ctx).unwrap(),
            Tensor0::<i64>::from_vec(vec![0], [], &ctx).unwrap(),
        )])
        .unwrap()
        .transform::<Tensor1<2>>(|feature| {
            Ok(<Tensor1<2> as SealedTypedTensor>::trusted_from_validated(
                feature.as_dynamic().clone(),
                Arc::new(DeviceBinding {
                    device: Device::Cpu,
                }),
            ))
        });

        assert!(matches!(
            dataset.batch(&[0]),
            Err(Error::InvalidArg {
                op: "typed::data::VecDataset::transform",
                ..
            })
        ));
    }

    #[test]
    fn transformed_loader_matches_runtime_order_values_and_shapes() {
        let ctx = cpu();
        let typed_items = (0..7)
            .map(|value| {
                (
                    Tensor1::<2>::from_vec(vec![value as f32; 2], [2], &ctx).unwrap(),
                    Tensor0::<i64>::from_vec(vec![value], [], &ctx).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let runtime_items = typed_items
            .iter()
            .map(|(feature, label)| (feature.as_dynamic().clone(), label.as_dynamic().clone()))
            .collect();
        let runtime = crate::data::VecDataset::new(runtime_items)
            .unwrap()
            .transform(|feature| {
                let reshaped = feature.reshape([1, 2])?;
                reshaped.eq(&reshaped)
            });
        let typed = VecDataset::new(typed_items)
            .unwrap()
            .transform::<Tensor2<1, 2, bool>>(|feature| {
                let reshaped = feature.reshape::<Tensor2<1, 2>>([1, 2])?;
                reshaped.eq(&reshaped)
            });
        let runtime_loader = crate::data::DataLoader::new(runtime, 3).shuffle(17);
        let typed_loader = DataLoader::new(typed, 3).shuffle(17);

        for epoch in [0, 2] {
            let expected = runtime_loader
                .batches_for_epoch(epoch)
                .map(|batch| {
                    let (features, labels) = batch.unwrap();
                    (
                        features.dims().to_vec(),
                        features.to_vec::<bool>().unwrap(),
                        labels.to_vec::<i64>().unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            let actual = typed_loader
                .batches_for_epoch(epoch)
                .map(|batch| {
                    let (features, labels) = batch.unwrap();
                    (
                        features.dims().to_vec(),
                        features.to_vec().unwrap(),
                        labels.to_vec().unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn transforming_an_empty_vec_dataset_is_empty_without_invoking_the_closure() {
        let dataset = VecDataset::<Tensor1<2>, Tensor0<i64>>::new(Vec::new())
            .unwrap()
            .transform::<Tensor2<1, 2>>(|_| -> Result<_> {
                panic!("an empty dataset must not invoke its transform")
            });
        let loader = DataLoader::new(&dataset, 2);

        assert_eq!(dataset.len(), 0);
        assert_eq!(loader.num_batches(), 0);
        assert!(loader.batches().next().is_none());
        assert!(matches!(
            dataset.batch(&[0]),
            Err(Error::IndexOutOfBounds { size: 0, .. })
        ));
    }

    #[test]
    fn loader_keeps_dynamic_tail_and_delegates_drop_last() {
        let kept = DataLoader::new(typed_tensor_dataset(7), 3);
        let shapes = kept
            .batches()
            .map(|batch| batch.unwrap().0.dims())
            .collect::<Vec<_>>();
        assert_eq!(shapes, vec![[3, 2], [3, 2], [1, 2]]);

        let dropped = DataLoader::new(typed_tensor_dataset(7), 3).drop_last(true);
        assert_eq!(dropped.num_batches(), 2);
        assert_eq!(
            labels(dropped.batches()),
            vec![vec![0, 1, 2], vec![3, 4, 5]]
        );
    }

    #[test]
    fn static_dataset_count_still_produces_a_dynamic_batch_marker() {
        let ctx = cpu();
        let features = Tensor2::<4, 2>::from_vec(vec![0.0f32; 8], [4, 2], &ctx).unwrap();
        let labels = Tensor1::<4, i64>::from_vec(vec![0; 4], [4], &ctx).unwrap();
        let dataset = TensorDataset::<Tensor1<2>, Tensor0<i64>>::new(features, labels).unwrap();

        let (features, labels): (Tensor2<DYN, 2>, Tensor1<DYN, i64>) = DataLoader::new(dataset, 3)
            .batches()
            .next()
            .unwrap()
            .unwrap();
        assert_eq!(features.dims(), [3, 2]);
        assert_eq!(labels.dims(), [3]);
    }

    #[test]
    fn seeded_epoch_order_is_identical_to_runtime_loader() {
        let runtime = runtime_tensor_dataset(12);
        let typed =
            TensorDataset::<Tensor1<2>, Tensor0<i64>>::try_from_dynamic(runtime.clone(), &cpu())
                .unwrap();
        let runtime_loader = crate::data::DataLoader::new(runtime, 5).shuffle(9);
        let typed_loader = DataLoader::new(typed, 5).shuffle(9);

        for epoch in [0, 1, 7] {
            let dynamic = runtime_loader
                .batches_for_epoch(epoch)
                .map(|batch| batch.unwrap().1.to_vec::<i64>().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(labels(typed_loader.batches_for_epoch(epoch)), dynamic);
        }
    }

    #[test]
    fn rank_zero_and_rank_seven_items_batch_at_the_supported_edges() {
        let ctx = cpu();
        let scalar = VecDataset::new(vec![
            (
                Tensor0::from_vec(vec![1.0f32], [], &ctx).unwrap(),
                Tensor0::<i64>::from_vec(vec![2], [], &ctx).unwrap(),
            ),
            (
                Tensor0::from_vec(vec![3.0f32], [], &ctx).unwrap(),
                Tensor0::<i64>::from_vec(vec![4], [], &ctx).unwrap(),
            ),
        ])
        .unwrap();
        let scalar_batch = scalar.batch(&[1, 0]).unwrap();
        assert_eq!(scalar_batch.0.dims(), [2]);
        assert_eq!(scalar_batch.1.dims(), [2]);

        type Item7 = Tensor7<1, 1, 1, 1, 1, 1, 1>;
        let rank_seven = VecDataset::new(vec![(
            Item7::from_vec(vec![5.0], [1; 7], &ctx).unwrap(),
            Tensor0::<i64>::from_vec(vec![6], [], &ctx).unwrap(),
        )])
        .unwrap();
        let batch = rank_seven.batch(&[0]).unwrap();
        let _: Tensor8<DYN, 1, 1, 1, 1, 1, 1, 1> = batch.0;
        assert_eq!(batch.1.dims(), [1]);
    }

    #[test]
    fn dynamic_conversion_rejects_wrong_feature_and_label_dimensions() {
        let ctx = cpu();
        let runtime = runtime_tensor_dataset(3);
        assert!(matches!(
            TensorDataset::<Tensor1<3>, Tensor0<i64>>::try_from_dynamic(runtime, &ctx),
            Err(Error::ShapeMismatch {
                op: "typed::data::TensorDataset::try_from_dynamic",
                ..
            })
        ));

        let features = Tensor::zeros([3, 2], DType::F32, &Device::Cpu).unwrap();
        let labels = Tensor::zeros([3, 1], DType::I64, &Device::Cpu).unwrap();
        let runtime = crate::data::TensorDataset::new(features, labels).unwrap();
        assert!(matches!(
            TensorDataset::<Tensor1<2>, Tensor0<i64>>::try_from_dynamic(runtime, &ctx),
            Err(Error::RankMismatch {
                op: "typed::data::TensorDataset::try_from_dynamic",
                expected: 1,
                got: 2,
            })
        ));

        let features = Tensor::zeros([3, 2], DType::F32, &Device::Cpu).unwrap();
        let labels = Tensor::zeros([3, 1], DType::I64, &Device::Cpu).unwrap();
        let runtime = crate::data::TensorDataset::new(features, labels).unwrap();
        assert!(matches!(
            TensorDataset::<Tensor1<2>, Tensor1<2, i64>>::try_from_dynamic(runtime, &ctx),
            Err(Error::ShapeMismatch {
                op: "typed::data::TensorDataset::try_from_dynamic",
                ..
            })
        ));
    }

    /// The [`VecDataset`] half of the check above. Its per-item wrap and the
    /// input wrap inside [`VecDataset::transform`] are *jointly* the only
    /// barrier between a stored runtime item and the `&F` a user closure
    /// receives: with both removed, a `VecDataset<Tensor1<3>, _>` built over
    /// `[2]`-shaped items constructs successfully and hands the closure a
    /// `Tensor1<3>` whose runtime dims are `[2]`. This pins the reachable half
    /// by rendered message, so neither guard can be dropped silently.
    #[test]
    fn vec_dataset_conversion_rejects_wrong_item_shape_element_and_rank() {
        let ctx = cpu();
        let runtime = || {
            crate::data::VecDataset::new(vec![(
                Tensor::from_vec(vec![1.0f32, 2.0], [2], &Device::Cpu).unwrap(),
                Tensor::from_vec(vec![7i64], [], &Device::Cpu).unwrap(),
            )])
            .unwrap()
        };

        assert_eq!(
            VecDataset::<Tensor1<3>, Tensor0<i64>>::try_from_dynamic(runtime(), &ctx)
                .unwrap_err()
                .to_string(),
            "typed::data::VecDataset::try_from_dynamic: shape mismatch: lhs [2] vs rhs [3]"
        );
        assert_eq!(
            VecDataset::<Tensor1<2, f64>, Tensor0<i64>>::try_from_dynamic(runtime(), &ctx)
                .unwrap_err()
                .to_string(),
            "typed::data::VecDataset::try_from_dynamic: dtype mismatch: expected f64, got f32 \
             (no implicit promotion; cast explicitly with to_dtype)"
        );
        assert_eq!(
            VecDataset::<Tensor1<2>, Tensor1<1, i64>>::try_from_dynamic(runtime(), &ctx)
                .unwrap_err()
                .to_string(),
            "typed::data::VecDataset::try_from_dynamic: rank mismatch: expected rank 1, got rank 0"
        );

        // The matching contract is accepted and batches to leading `DYN`.
        let accepted =
            VecDataset::<Tensor1<2>, Tensor0<i64>>::try_from_dynamic(runtime(), &ctx).unwrap();
        assert_eq!(accepted.batch(&[0]).unwrap().0.dims(), [1, 2]);
    }

    /// The positive half of the barrier described above: a stage observes an
    /// `&F` whose runtime metadata matches `F` exactly, and a chained stage
    /// observes its predecessor's declared output type.
    ///
    /// The stage's own `checked_wrap::<F>` has no *reachable* rejection while
    /// the construction-time item wrap stands — every route to a `VecDataset`
    /// (`new` takes already-typed items, `try_from_dynamic` checks each one)
    /// guarantees the stored item matches `F`. So this test pins what the stage
    /// is contracted to hand the user rather than pretending to exercise a
    /// rejection that cannot be reached from safe code.
    #[test]
    fn a_transform_stage_observes_exactly_its_declared_item_contract() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let ctx = cpu();
        let first = Arc::new(AtomicUsize::new(0));
        let second = Arc::new(AtomicUsize::new(0));
        let first_calls = Arc::clone(&first);
        let second_calls = Arc::clone(&second);

        let dataset = VecDataset::<Tensor1<2>, Tensor0<i64>>::try_from_dynamic(
            crate::data::VecDataset::new(vec![(
                Tensor::from_vec(vec![1.0f32, 2.0], [2], &Device::Cpu).unwrap(),
                Tensor::from_vec(vec![7i64], [], &Device::Cpu).unwrap(),
            )])
            .unwrap(),
            &ctx,
        )
        .unwrap()
        .transform::<Tensor2<1, 2>>(move |feature| {
            first_calls.fetch_add(1, Ordering::SeqCst);
            assert_eq!(feature.dims(), [2]);
            assert_eq!(feature.as_dynamic().dtype(), DType::F32);
            feature.reshape([1, 2])
        })
        .transform::<Tensor2<1, 2, bool>>(move |feature| {
            second_calls.fetch_add(1, Ordering::SeqCst);
            assert_eq!(feature.dims(), [1, 2]);
            feature.eq(feature)
        });

        let (features, labels) = dataset.batch(&[0]).unwrap();
        assert_eq!(features.dims(), [1, 1, 2]);
        assert_eq!(features.to_vec().unwrap(), vec![true, true]);
        assert_eq!(labels.to_vec().unwrap(), vec![7]);
        assert_eq!(first.load(Ordering::SeqCst), 1);
        assert_eq!(second.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn construction_rejects_noncanonical_binding_identity() {
        let ctx = cpu();
        let forged = Arc::new(DeviceBinding {
            device: Device::Cpu,
        });
        let features = <Tensor2<DYN, 2> as SealedTypedTensor>::trusted_from_validated(
            Tensor::zeros([2, 2], DType::F32, &Device::Cpu).unwrap(),
            Arc::clone(&forged),
        );
        let labels = <Tensor1<DYN, i64> as SealedTypedTensor>::trusted_from_validated(
            Tensor::zeros([2], DType::I64, &Device::Cpu).unwrap(),
            forged,
        );

        assert!(matches!(
            TensorDataset::<Tensor1<2>, Tensor0<i64>>::new(features, labels),
            Err(Error::InvalidArg {
                op: "typed::data::TensorDataset::new",
                ..
            })
        ));
        drop(ctx);
    }

    #[test]
    fn tensor_dataset_boundary_is_no_copy_and_batches_keep_the_graph() {
        let ctx = cpu();
        let dynamic = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0], [4, 1], &Device::Cpu)
            .unwrap()
            .traced()
            .unwrap();
        let original = dynamic.clone();
        let features = Tensor2::<DYN, 1, f64>::try_from_dynamic(dynamic, &ctx).unwrap();
        let labels = Tensor1::<DYN, i64>::from_vec(vec![0; 4], [4], &ctx).unwrap();
        let dataset =
            TensorDataset::<Tensor1<1, f64>, Tensor0<i64>>::new(features, labels).unwrap();
        assert!(original.ptr_eq(dataset.as_dynamic().inputs()));

        let batch = dataset.batch(&[3, 1, 1]).unwrap().0;
        let grads: Grads = batch.sum_all().unwrap().backward().unwrap();
        assert_eq!(
            grads.wrt_input(&original).unwrap().to_vec::<f64>().unwrap(),
            vec![0.0, 2.0, 0.0, 1.0]
        );
        assert_eq!(batch.as_dynamic().device(), Device::Cpu);
    }
}
