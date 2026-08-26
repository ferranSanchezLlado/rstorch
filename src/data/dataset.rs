//! The batch-level [`Dataset`] trait and the [`VecDataset`] and
//! [`TensorDataset`] implementations.
//!
//! Datasets yield batches rather than individual items. This lets an existing
//! device tensor use one indexed batch and lets a language-model dataset expose
//! a window into one token buffer. [`VecDataset`] provides the common
//! in-memory collation.

use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// A source of **batches**, indexed by position.
///
/// Implementors expose a fixed [`len`](Dataset::len) and a
/// [`batch`](Dataset::batch) that maps a slice of item positions to one
/// collated [`Batch`](Dataset::Batch). [`DataLoader`](crate::data::DataLoader)
/// is the only consumer: it decides *which* positions (in order, or shuffled
/// from a seed) and hands them over in `batch_size`-sized slices.
///
/// `Batch` is deliberately unconstrained. The two provided datasets both use
/// `(Tensor, Tensor)` — the supervised `(inputs, targets)` pair the training
/// loop destructures — but a batch may be a single tensor, a struct with an
/// attention mask, or anything else a model's `forward` wants.
///
/// # Contract
///
/// - `batch` must accept any slice of positions in `0..len()`, in any order,
///   **with repeats** (bootstrap sampling and small-dataset shuffling both
///   produce them), and must return a batch whose leading axis is
///   `indices.len()`.
/// - `batch` is `&self`: batching is pure. Nothing about the dataset changes
///   as it is read, so one dataset can back several loaders.
/// - Both provided implementations reject an **empty** index slice with
///   [`Error::InvalidArg`] (a stack of nothing has no shape).
///   [`DataLoader`](crate::data::DataLoader) never emits one.
///
/// # Implementing it
///
/// ```
/// use rstorch::data::Dataset;
/// use rstorch::{DType, Device, Result, Tensor};
///
/// /// Fixed-length windows over one long token buffer (the LM case): the
/// /// batch is built with tensor ops, never by reading tokens back to the host.
/// struct Windows { tokens: Tensor, window: usize }
///
/// impl Dataset for Windows {
///     type Batch = (Tensor, Tensor);
///
///     fn len(&self) -> usize {
///         self.tokens.dims()[0].saturating_sub(self.window)
///     }
///
///     fn batch(&self, indices: &[usize]) -> Result<(Tensor, Tensor)> {
///         let starts: Vec<Tensor> = indices
///             .iter()
///             .map(|&s| self.tokens.narrow(0, s, self.window + 1))
///             .collect::<Result<_>>()?;
///         let refs: Vec<&Tensor> = starts.iter().collect();
///         let rows = Tensor::stack(&refs, 0)?;
///         Ok((rows.narrow(1, 0, self.window)?, rows.narrow(1, 1, self.window)?))
///     }
/// }
///
/// # fn main() -> Result<()> {
/// let tokens = Tensor::arange(0.0, 8.0, 1.0, DType::I64, &Device::Cpu)?;
/// let ds = Windows { tokens, window: 3 };
/// assert_eq!(ds.len(), 5);
/// let (x, y) = ds.batch(&[0, 4])?;
/// assert_eq!(x.dims(), &[2, 3]);
/// assert_eq!(x.to_vec::<i64>()?, vec![0, 1, 2, 4, 5, 6]);
/// assert_eq!(y.to_vec::<i64>()?, vec![1, 2, 3, 5, 6, 7]);
/// # Ok(())
/// # }
/// ```
pub trait Dataset {
    /// What one batch is — `(inputs, targets)` for both provided datasets.
    type Batch;

    /// The number of items (**not** batches; that is
    /// [`DataLoader::num_batches`](crate::data::DataLoader::num_batches)).
    fn len(&self) -> usize;

    /// Whether the dataset has no items.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Collate the items at `indices` into one batch.
    ///
    /// # Errors
    ///
    /// Implementation-defined; both provided datasets report
    /// [`Error::IndexOutOfBounds`] for a position at or past
    /// [`len`](Dataset::len) and [`Error::InvalidArg`] for an empty slice.
    fn batch(&self, indices: &[usize]) -> Result<Self::Batch>;
}

/// A borrowed dataset is a dataset, so one split can back several loaders
/// (`DataLoader::new(&ds, 64)` for training, another for evaluation) without
/// moving or duplicating it.
impl<D: Dataset + ?Sized> Dataset for &D {
    type Batch = D::Batch;

    fn len(&self) -> usize {
        (**self).len()
    }

    fn batch(&self, indices: &[usize]) -> Result<Self::Batch> {
        (**self).batch(indices)
    }
}

/// Validate a batch index slice against a dataset length: non-empty, every
/// position in range. Shared by both provided datasets so their diagnostics
/// name the dataset method rather than the tensor op underneath.
fn check_indices(op: &'static str, indices: &[usize], len: usize) -> Result<()> {
    if indices.is_empty() {
        return Err(Error::InvalidArg {
            op,
            msg: "an empty index slice has no batch; ask for at least one item".to_string(),
        });
    }
    for &index in indices {
        if index >= len {
            return Err(Error::IndexOutOfBounds {
                op,
                index: i64::try_from(index).unwrap_or(i64::MAX),
                axis: 0,
                size: len,
            });
        }
    }
    Ok(())
}

/// The optional per-item transform of a [`VecDataset`].
type Transform = Box<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>;

/// An in-memory dataset of `(input, target)` item pairs, collated by stacking
/// along a new leading axis.
///
/// This is the three-line case: hand it a `Vec` of per-item tensors and it is a
/// [`Dataset`]. Items must agree on shape, dtype and device (separately for
/// inputs and for targets) — the default collation is
/// [`Tensor::stack`], which requires it — and
/// [`new`](VecDataset::new) checks that up front rather than at batch 47. A
/// ragged corpus (variable-length sequences needing a padding collate) is
/// precisely the case that should implement [`Dataset`] directly.
///
/// ```
/// use rstorch::data::{Dataset, VecDataset};
/// # use rstorch::{DType, Device, Result, Tensor};
/// # fn main() -> Result<()> {
/// let dev = Device::Cpu;
/// let items = (0..4)
///     .map(|i| {
///         Ok((
///             Tensor::full([2], f64::from(i), DType::F32, &dev)?,
///             Tensor::full([], f64::from(i % 2), DType::I64, &dev)?,
///         ))
///     })
///     .collect::<Result<Vec<_>>>()?;
///
/// let ds = VecDataset::new(items)?;
/// let (x, y) = ds.batch(&[3, 1])?;
/// assert_eq!(x.dims(), &[2, 2]);          // [batch, item shape...]
/// assert_eq!(y.dims(), &[2]);             // scalar targets stack to [batch]
/// assert_eq!(x.to_vec::<f32>()?, vec![3.0, 3.0, 1.0, 1.0]);
/// assert_eq!(y.to_vec::<i64>()?, vec![1, 1]);
/// # Ok(())
/// # }
/// ```
pub struct VecDataset {
    items: Vec<(Tensor, Tensor)>,
    transform: Option<Transform>,
}

impl VecDataset {
    /// Build a dataset from `(input, target)` pairs.
    ///
    /// # Errors
    ///
    /// [`Error::ShapeMismatch`], [`Error::DTypeMismatch`] or
    /// [`Error::DeviceMismatch`] (`op: "VecDataset::new"`) if an item's input
    /// disagrees with the first item's input, or its target with the first
    /// item's target. An empty `Vec` is legal: it is a dataset of length 0,
    /// and a loader over it yields no batches.
    pub fn new(items: Vec<(Tensor, Tensor)>) -> Result<VecDataset> {
        const OP: &str = "VecDataset::new";
        if let Some((first_input, first_target)) = items.first() {
            for (input, target) in &items {
                check_uniform(OP, first_input, input)?;
                check_uniform(OP, first_target, target)?;
            }
        }
        Ok(VecDataset {
            items,
            transform: None,
        })
    }

    /// Attach a per-item transform applied to each **input** as the batch is
    /// built (the target is passed through unchanged).
    ///
    /// This is the augmentation and normalization hook — `|x| x.div_scalar(255.0)`,
    /// a crop, a flip. It runs once per item per batch, so a value cached in the
    /// dataset is cheaper than a transform that recomputes it every epoch.
    ///
    /// The closure is `Fn`, not `FnMut`: batching is pure and shareable across
    /// threads. A *stochastic* transform therefore has to carry its own
    /// synchronized [`Rng`](crate::Rng) (a `Mutex<Rng>` capture), which is the
    /// honest cost of randomness in a `&self` method — the alternative is to
    /// implement [`Dataset`] yourself and own the stream.
    ///
    /// Calling it twice replaces the previous transform.
    ///
    /// ```
    /// use rstorch::data::{Dataset, VecDataset};
    /// # use rstorch::{DType, Device, Result, Tensor};
    /// # fn main() -> Result<()> {
    /// # let dev = Device::Cpu;
    /// # let items = vec![(
    /// #     Tensor::full([1], 255.0, DType::F32, &dev)?,
    /// #     Tensor::full([], 0.0, DType::I64, &dev)?,
    /// # )];
    /// let ds = VecDataset::new(items)?.transform(|x| x.div_scalar(255.0));
    /// assert_eq!(ds.batch(&[0])?.0.to_vec::<f32>()?, vec![1.0]);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn transform(
        mut self,
        transform: impl Fn(&Tensor) -> Result<Tensor> + Send + Sync + 'static,
    ) -> VecDataset {
        self.transform = Some(Box::new(transform));
        self
    }

    /// The stored items, untransformed.
    pub fn items(&self) -> &[(Tensor, Tensor)] {
        &self.items
    }
}

impl std::fmt::Debug for VecDataset {
    /// The transform is a closure with no `Debug` bound, so it is reported by
    /// presence only.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VecDataset")
            .field("items", &self.items.len())
            .field("transform", &self.transform.is_some())
            .finish()
    }
}

impl Dataset for VecDataset {
    type Batch = (Tensor, Tensor);

    fn len(&self) -> usize {
        self.items.len()
    }

    /// Stack the selected inputs (after the transform, if any) and targets
    /// along a new leading axis, so a batch of `n` items whose inputs are
    /// `[28, 28]` is `[n, 28, 28]`.
    ///
    /// # Errors
    ///
    /// [`Error::IndexOutOfBounds`] / [`Error::InvalidArg`]
    /// (`op: "VecDataset::batch"`) per the [`Dataset`] contract, plus anything
    /// the transform or [`Tensor::stack`] reports — a transform that returns
    /// differently-shaped tensors for different items fails the stack.
    fn batch(&self, indices: &[usize]) -> Result<(Tensor, Tensor)> {
        check_indices("VecDataset::batch", indices, self.items.len())?;

        let mut inputs = Vec::with_capacity(indices.len());
        let mut targets = Vec::with_capacity(indices.len());
        for &index in indices {
            let (input, target) = &self.items[index];
            inputs.push(match &self.transform {
                Some(transform) => transform(input)?,
                None => input.clone(),
            });
            targets.push(target.clone());
        }

        let input_refs: Vec<&Tensor> = inputs.iter().collect();
        let target_refs: Vec<&Tensor> = targets.iter().collect();
        Ok((
            Tensor::stack(&input_refs, 0)?,
            Tensor::stack(&target_refs, 0)?,
        ))
    }
}

/// Two whole tensors — all inputs and all targets, stacked along axis 0 —
/// batched **on the device** by [`Tensor::index_select`].
///
/// This is the dataset the batch-level trait exists for. Nothing is read back
/// to the host per batch and nothing is copied at construction: the split stays
/// wherever it already lives (`to_device` it once, up front) and a batch is one
/// index vector plus one gather per tensor. Autograd sees an ordinary
/// `index_select`, so a batch of a traced input is itself traced — which is how
/// the "stays on device" property is *tested* rather than asserted.
///
/// Inputs and targets must share their leading dimension (the item count) and
/// their device; their **dtypes are independent**, so `F32` features with `I64`
/// class labels — the MNIST case — is the expected shape, not an exception.
///
/// ```
/// use rstorch::data::{Dataset, TensorDataset};
/// # use rstorch::{DType, Device, Result, Tensor};
/// # fn main() -> Result<()> {
/// let dev = Device::Cpu;
/// let x = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0], [3, 2], &dev)?;
/// let y = Tensor::from_vec(vec![7i64, 8, 9], [3], &dev)?;
///
/// let ds = TensorDataset::new(x, y)?;
/// assert_eq!(ds.len(), 3);
/// let (bx, by) = ds.batch(&[2, 0])?;
/// assert_eq!(bx.to_vec::<f32>()?, vec![4.0, 5.0, 0.0, 1.0]);
/// assert_eq!(by.to_vec::<i64>()?, vec![9, 7]);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct TensorDataset {
    inputs: Tensor,
    targets: Tensor,
}

impl TensorDataset {
    /// Build a dataset over two device-resident tensors whose leading axis is
    /// the item axis.
    ///
    /// Both tensors are taken by value and stored as they are — an `Arc` bump,
    /// no element copy and no layout change.
    ///
    /// # Errors
    ///
    /// [`Error::RankMismatch`] (`op: "TensorDataset::new"`) if either tensor is
    /// a scalar (there is no item axis to index), [`Error::ShapeMismatch`] if
    /// their leading dimensions differ, [`Error::DeviceMismatch`] if they live
    /// on different devices.
    pub fn new(inputs: Tensor, targets: Tensor) -> Result<TensorDataset> {
        const OP: &str = "TensorDataset::new";
        for tensor in [&inputs, &targets] {
            if tensor.rank() == 0 {
                return Err(Error::RankMismatch {
                    op: OP,
                    expected: 1,
                    got: 0,
                });
            }
        }
        if inputs.dims()[0] != targets.dims()[0] {
            return Err(Error::ShapeMismatch {
                op: OP,
                lhs: inputs.shape().clone(),
                rhs: targets.shape().clone(),
            });
        }
        if inputs.device() != targets.device() {
            return Err(Error::DeviceMismatch {
                op: OP,
                expected: inputs.device(),
                got: targets.device(),
            });
        }
        Ok(TensorDataset { inputs, targets })
    }

    /// The whole input tensor, `[items, ...]`.
    pub fn inputs(&self) -> &Tensor {
        &self.inputs
    }

    /// The whole target tensor, `[items, ...]`.
    pub fn targets(&self) -> &Tensor {
        &self.targets
    }
}

impl Dataset for TensorDataset {
    type Batch = (Tensor, Tensor);

    fn len(&self) -> usize {
        self.inputs.dims()[0]
    }

    /// One [`Tensor::index_vec`] on the dataset's device, then one
    /// [`Tensor::index_select`] per tensor. The host never sees an element.
    ///
    /// # Errors
    ///
    /// [`Error::IndexOutOfBounds`] / [`Error::InvalidArg`]
    /// (`op: "TensorDataset::batch"`) per the [`Dataset`] contract, plus
    /// anything the index build or the gather reports.
    fn batch(&self, indices: &[usize]) -> Result<(Tensor, Tensor)> {
        check_indices("TensorDataset::batch", indices, self.len())?;
        let positions = Tensor::index_vec(indices, &self.inputs.device())?;
        Ok((
            self.inputs.index_select(0, &positions)?,
            self.targets.index_select(0, &positions)?,
        ))
    }
}

/// Reject a tensor that disagrees with `first` on shape, dtype or device — the
/// three things [`Tensor::stack`] requires of a [`VecDataset`]'s items.
fn check_uniform(op: &'static str, first: &Tensor, other: &Tensor) -> Result<()> {
    if first.dtype() != other.dtype() {
        return Err(Error::DTypeMismatch {
            op,
            expected: first.dtype(),
            got: other.dtype(),
        });
    }
    if first.device() != other.device() {
        return Err(Error::DeviceMismatch {
            op,
            expected: first.device(),
            got: other.device(),
        });
    }
    if first.dims() != other.dims() {
        return Err(Error::ShapeMismatch {
            op,
            lhs: first.shape().clone(),
            rhs: other.shape().clone(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Device;
    use crate::dtype::DType;
    use crate::storage::{CpuStorage, Storage};
    use crate::testing::check_grad;

    const CPU: Device = Device::Cpu;

    /// `n` items whose input is `[2]` filled with the item index and whose
    /// target is the scalar index.
    fn pairs(n: usize) -> Vec<(Tensor, Tensor)> {
        (0..n)
            .map(|i| {
                (
                    Tensor::full([2], i as f64, DType::F32, &CPU).unwrap(),
                    Tensor::full([], i as f64, DType::I64, &CPU).unwrap(),
                )
            })
            .collect()
    }

    /// `[n, 2]` inputs counting up from zero, `[n]` I64 targets.
    fn tensor_dataset(n: usize) -> TensorDataset {
        let x: Vec<f32> = (0..2 * n).map(|v| v as f32).collect();
        let y: Vec<i64> = (0..n).map(|v| v as i64).collect();
        TensorDataset::new(
            Tensor::from_vec(x, [n, 2], &CPU).unwrap(),
            Tensor::from_vec(y, [n], &CPU).unwrap(),
        )
        .unwrap()
    }

    /// The address of the f32 buffer behind `t` ("view or copy?" checks).
    fn f32_buf_ptr(t: &Tensor) -> *const f32 {
        match t.storage() {
            Storage::Cpu(CpuStorage::F32(v)) => v.as_ptr(),
            _ => panic!("expected an f32 CPU tensor"),
        }
    }

    // ---- VecDataset ------------------------------------------------------

    #[test]
    fn vec_dataset_stacks_selected_items() {
        let ds = VecDataset::new(pairs(4)).unwrap();
        assert_eq!(ds.len(), 4);
        assert!(!ds.is_empty());

        // Order follows the index slice, and repeats are legal.
        let (x, y) = ds.batch(&[3, 0, 3]).unwrap();
        assert_eq!(x.dims(), &[3, 2]);
        assert_eq!(y.dims(), &[3]);
        assert_eq!(
            x.to_vec::<f32>().unwrap(),
            vec![3.0, 3.0, 0.0, 0.0, 3.0, 3.0]
        );
        assert_eq!(y.to_vec::<i64>().unwrap(), vec![3, 0, 3]);
        // Dtypes are per-column and preserved.
        assert_eq!(x.dtype(), DType::F32);
        assert_eq!(y.dtype(), DType::I64);
    }

    #[test]
    fn vec_dataset_is_empty_when_it_has_no_items() {
        let ds = VecDataset::new(Vec::new()).unwrap();
        assert_eq!(ds.len(), 0);
        assert!(ds.is_empty());
        assert!(matches!(
            ds.batch(&[0]),
            Err(Error::IndexOutOfBounds { size: 0, .. })
        ));
    }

    #[test]
    fn vec_dataset_transform_applies_to_inputs_only() {
        let ds = VecDataset::new(pairs(3))
            .unwrap()
            .transform(|x| x.add_scalar(10.0));
        let (x, y) = ds.batch(&[1, 2]).unwrap();
        assert_eq!(x.to_vec::<f32>().unwrap(), vec![11.0, 11.0, 12.0, 12.0]);
        assert_eq!(y.to_vec::<i64>().unwrap(), vec![1, 2]);
        // The stored items are untouched: the transform runs per batch.
        assert_eq!(ds.items()[1].0.to_vec::<f32>().unwrap(), vec![1.0, 1.0]);
    }

    #[test]
    fn vec_dataset_transform_can_change_the_item_shape() {
        // Uniformly, that is: reshaping every [2] input to [1, 2] is fine.
        let ds = VecDataset::new(pairs(2))
            .unwrap()
            .transform(|x| x.reshape([1, 2]));
        assert_eq!(ds.batch(&[0, 1]).unwrap().0.dims(), &[2, 1, 2]);
    }

    #[test]
    fn vec_dataset_rejects_ragged_items() {
        let mut items = pairs(2);
        items[1].0 = Tensor::full([3], 1.0, DType::F32, &CPU).unwrap();
        assert!(matches!(
            VecDataset::new(items),
            Err(Error::ShapeMismatch {
                op: "VecDataset::new",
                ..
            })
        ));

        let mut items = pairs(2);
        items[1].0 = Tensor::full([2], 1.0, DType::F64, &CPU).unwrap();
        assert!(matches!(
            VecDataset::new(items),
            Err(Error::DTypeMismatch {
                op: "VecDataset::new",
                ..
            })
        ));

        // A disagreeing *target* is caught too.
        let mut items = pairs(2);
        items[1].1 = Tensor::full([1], 1.0, DType::I64, &CPU).unwrap();
        assert!(matches!(
            VecDataset::new(items),
            Err(Error::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn provided_datasets_reject_an_empty_index_slice() {
        assert!(matches!(
            VecDataset::new(pairs(2)).unwrap().batch(&[]),
            Err(Error::InvalidArg {
                op: "VecDataset::batch",
                ..
            })
        ));
        assert!(matches!(
            tensor_dataset(2).batch(&[]),
            Err(Error::InvalidArg {
                op: "TensorDataset::batch",
                ..
            })
        ));
    }

    #[test]
    fn provided_datasets_reject_an_out_of_range_index() {
        assert!(matches!(
            VecDataset::new(pairs(2)).unwrap().batch(&[0, 2]),
            Err(Error::IndexOutOfBounds {
                op: "VecDataset::batch",
                index: 2,
                axis: 0,
                size: 2,
            })
        ));
        assert!(matches!(
            tensor_dataset(2).batch(&[2]),
            Err(Error::IndexOutOfBounds {
                op: "TensorDataset::batch",
                index: 2,
                size: 2,
                ..
            })
        ));
    }

    // ---- TensorDataset ---------------------------------------------------

    #[test]
    fn tensor_dataset_batches_exact_rows() {
        let ds = tensor_dataset(4);
        assert_eq!(ds.len(), 4);
        let (x, y) = ds.batch(&[3, 1, 1]).unwrap();
        assert_eq!(x.dims(), &[3, 2]);
        assert_eq!(y.dims(), &[3]);
        assert_eq!(
            x.to_vec::<f32>().unwrap(),
            vec![6.0, 7.0, 2.0, 3.0, 2.0, 3.0]
        );
        assert_eq!(y.to_vec::<i64>().unwrap(), vec![3, 1, 1]);
        assert_eq!(x.dtype(), DType::F32);
        assert_eq!(y.dtype(), DType::I64);
        assert_eq!(x.device(), CPU);
    }

    #[test]
    fn tensor_dataset_new_validates_its_operands() {
        let x = Tensor::from_vec(vec![0.0f32; 6], [3, 2], &CPU).unwrap();
        let y = Tensor::from_vec(vec![0i64; 2], [2], &CPU).unwrap();
        assert!(matches!(
            TensorDataset::new(x.clone(), y),
            Err(Error::ShapeMismatch {
                op: "TensorDataset::new",
                ..
            })
        ));

        let scalar = Tensor::full([], 1.0, DType::I64, &CPU).unwrap();
        assert!(matches!(
            TensorDataset::new(x, scalar),
            Err(Error::RankMismatch {
                op: "TensorDataset::new",
                expected: 1,
                got: 0,
            })
        ));
    }

    #[test]
    fn tensor_dataset_construction_does_not_copy() {
        let x = Tensor::from_vec(vec![0.0f32; 6], [3, 2], &CPU).unwrap();
        let before = f32_buf_ptr(&x);
        let ds =
            TensorDataset::new(x, Tensor::from_vec(vec![0i64; 3], [3], &CPU).unwrap()).unwrap();
        assert_eq!(f32_buf_ptr(ds.inputs()), before);
    }

    /// The gate's "indexing stays on-device" test.
    ///
    /// There is no CPU-side hook that can observe a host round-trip directly,
    /// so this asserts the property that *implies* the absence of one: the
    /// batch is the output of a differentiable tensor op over the stored
    /// tensor. `Tensor::to_vec` (a host read) severs the autograd graph and
    /// rebuilds a fresh leaf, so a batch built by reading elements back and
    /// re-uploading them could not be traced back to the source at all — and
    /// the finite-difference check below could not pass.
    #[test]
    fn tensor_dataset_indexing_stays_on_device() {
        let x = Tensor::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2], &CPU).unwrap();
        let y = Tensor::from_vec(vec![0i64, 1, 2], [3], &CPU).unwrap();
        let traced = x.traced().unwrap();
        let ds = TensorDataset::new(traced.clone(), y.clone()).unwrap();

        // The batch carries a graph back to the whole-split tensor.
        let (bx, by) = ds.batch(&[2, 0, 0]).unwrap();
        let grads = bx
            .mul_scalar(2.0)
            .unwrap()
            .sum_all()
            .unwrap()
            .backward()
            .unwrap();
        let grad = grads.wrt_input(&traced).unwrap();
        assert_eq!(grad.dims(), &[3, 2]);
        // Row 0 was selected twice, row 1 never, row 2 once — repeats sum.
        assert_eq!(
            grad.to_vec::<f64>().unwrap(),
            vec![4.0, 4.0, 0.0, 0.0, 2.0, 2.0]
        );
        // An integral target column takes no gradient and stays integral.
        assert_eq!(by.dtype(), DType::I64);

        // The same claim, checked numerically through the dataset API.
        check_grad(
            |inputs| {
                let ds = TensorDataset::new(inputs[0].clone(), y.clone())?;
                ds.batch(&[2, 0, 0])?.0.sum_all()
            },
            &[x],
            1e-4,
            1e-6,
        )
        .unwrap();
    }

    #[test]
    fn a_reference_is_a_dataset() {
        let ds = tensor_dataset(4);
        let borrowed: &TensorDataset = &ds;
        assert_eq!(borrowed.len(), 4);
        assert_eq!(borrowed.batch(&[0]).unwrap().0.dims(), &[1, 2]);
    }
}
