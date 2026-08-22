//! MNIST as a batch-level [`Dataset`]: the tensor-side wrapper over the raw
//! [`Mnist`] parse.
//!
//! The split is uploaded **once** — one `[items, ...]` `F32` pixel tensor and
//! one `[items]` `I64` label tensor — and every batch is then a device-side
//! [`Tensor::index_select`] through [`TensorDataset`]. Nothing is read back to
//! the host per batch, and the per-item normalization to `[0.0, 1.0]` that
//! [`Mnist`] already did during the parse is not repeated per epoch.
//!
//! Which of the two layouts you ask for is what tells an MLP apart from a
//! convolutional net: see [`MnistLayout`].

#[cfg(feature = "hub")]
use super::DatasetHub;
use super::mnist::Mnist;
#[cfg(feature = "hub")]
use super::mnist::MnistSplit;
use crate::data::{Dataset, TensorDataset};
use crate::device::Device;
use crate::error::{Error, Result};
use crate::tensor::Tensor;

/// How an MNIST image is shaped inside a batch.
///
/// The pixels are identical either way — only the input tensor's shape
/// differs, and with it which layers the batch can be fed to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MnistLayout {
    /// One row of `rows * cols` pixels per image: a batch is
    /// `[batch, rows * cols]`, what a [`Linear`](crate::nn::Linear) stack
    /// expects.
    Flat,
    /// One single-channel image per item: a batch is `[batch, 1, rows, cols]`,
    /// the `NCHW` layout the convolution and pooling ops expect.
    Nchw,
}

/// An MNIST split as a [`Dataset`] of `(pixels, labels)` batches.
///
/// Construct it from an already-parsed [`Mnist`] with [`new`](Self::new), or
/// straight from the cache (downloading if needed) with `load`, which the `hub`
/// feature adds. The batch is `(F32 pixels, I64 labels)` — the exact
/// pair [`Tensor::cross_entropy`] takes, so `logits.cross_entropy(&labels)`
/// needs no conversion.
///
/// # Preprocessing
///
/// Pixels arrive already scaled to `[0.0, 1.0]`. Anything further — the
/// conventional `(x − 0.1307) / 0.3081` standardization, say — is one op over
/// the **whole** split rather than a per-batch transform, because both halves
/// are public tensors:
///
/// ```no_run
/// # use rstorch::data::{TensorDataset, hub::MnistDataset};
/// # fn go(ds: MnistDataset) -> rstorch::Result<TensorDataset> {
/// TensorDataset::new(
///     ds.inputs().sub_scalar(0.1307)?.div_scalar(0.3081)?,
///     ds.targets().clone(),
/// )
/// # }
/// ```
///
/// # Example
///
/// Two 2×2 "images" in IDX form (the real files are the same format, 28×28 and
/// gzipped), wrapped in both layouts:
///
/// ```
/// use rstorch::data::hub::{Mnist, MnistDataset, MnistLayout};
/// use rstorch::data::Dataset;
/// use rstorch::{DType, Device};
///
/// # fn main() -> rstorch::Result<()> {
/// let mut images = Vec::new();
/// images.extend(2051u32.to_be_bytes()); // IDX3 magic
/// images.extend([0, 0, 0, 2u8]); // 2 images
/// images.extend([0, 0, 0, 2u8]); // 2 rows
/// images.extend([0, 0, 0, 2u8]); // 2 cols
/// images.extend([0, 255, 0, 255, 51, 51, 51, 51]);
/// let mut labels = Vec::new();
/// labels.extend(2049u32.to_be_bytes()); // IDX1 magic
/// labels.extend([0, 0, 0, 2u8]);
/// labels.extend([7u8, 3]);
///
/// let raw = Mnist::from_idx_bytes(&images, &labels)?;
/// let flat = MnistDataset::new(&raw, MnistLayout::Flat, &Device::Cpu)?;
/// assert_eq!(flat.len(), 2);
/// assert_eq!(flat.image_shape(), (2, 2));
///
/// let (x, y) = flat.batch(&[1, 0])?;
/// assert_eq!(x.dims(), &[2, 4]);
/// assert_eq!(y.to_vec::<i64>()?, vec![3, 7]); // labels widen u8 -> I64
/// assert_eq!(y.dtype(), DType::I64);
///
/// // The same split for a conv net: `[batch, channels, rows, cols]`.
/// let nchw = MnistDataset::new(&raw, MnistLayout::Nchw, &Device::Cpu)?;
/// assert_eq!(nchw.batch(&[1, 0])?.0.dims(), &[2, 1, 2, 2]);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct MnistDataset {
    inner: TensorDataset,
    rows: usize,
    cols: usize,
    layout: MnistLayout,
}

impl MnistDataset {
    /// Uploads a parsed split to `device` under `layout`.
    ///
    /// This is where the host-to-device transfer happens: two allocations, one
    /// per tensor, and then nothing per batch. A split of zero examples is
    /// legal (it is a dataset of length 0, and a loader over it yields no
    /// batches).
    ///
    /// # Errors
    ///
    /// [`Error::Data`] (`MnistDataset::new`) if the parsed images disagree on
    /// their pixel count — the geometry in the IDX header must describe every
    /// image — plus anything the tensor upload reports.
    pub fn new(raw: &Mnist, layout: MnistLayout, device: &Device) -> Result<MnistDataset> {
        let (rows, cols) = raw.image_shape();
        let per_image = rows
            .checked_mul(cols)
            .ok_or_else(|| Error::data(format!("MNIST image geometry {rows}x{cols} overflows")))?;

        let mut pixels = Vec::with_capacity(raw.len().saturating_mul(per_image));
        for image in raw.images() {
            // Unreachable through the validating IDX parser, which produces
            // exact `rows * cols` chunks; kept so a future raw-layer change
            // fails loudly instead of silently mis-shaping the split.
            if image.len() != per_image {
                return Err(Error::data(format!(
                    "MNIST image has {} pixels but the header declares {rows}x{cols}",
                    image.len()
                )));
            }
            pixels.extend_from_slice(image);
        }

        let shape: Vec<usize> = match layout {
            MnistLayout::Flat => vec![raw.len(), per_image],
            MnistLayout::Nchw => vec![raw.len(), 1, rows, cols],
        };
        let inputs = Tensor::from_vec(pixels, shape, device)?;
        let labels: Vec<i64> = raw.labels().iter().map(|&label| i64::from(label)).collect();
        let targets = Tensor::from_vec(labels, [raw.len()], device)?;

        Ok(MnistDataset {
            inner: TensorDataset::new(inputs, targets)?,
            rows,
            cols,
            layout,
        })
    }

    /// Loads a split from `hub`'s cache — downloading and verifying the
    /// archives first if they are missing — and uploads it to `device`.
    ///
    /// Requires the `hub` feature (network access + gzip). Offline, parse the
    /// bytes yourself with [`Mnist::from_idx_bytes`] and hand the result to
    /// [`new`](Self::new).
    ///
    /// # Errors
    ///
    /// Anything [`Mnist::load`] reports (download, checksum, IDX parse) or
    /// [`new`](Self::new) reports.
    #[cfg(feature = "hub")]
    pub fn load(
        hub: &DatasetHub,
        split: MnistSplit,
        layout: MnistLayout,
        device: &Device,
    ) -> Result<MnistDataset> {
        MnistDataset::new(&Mnist::load(hub, split)?, layout, device)
    }

    /// The `(rows, cols)` geometry of one image — `(28, 28)` for the real
    /// dataset, whatever the [`layout`](Self::layout).
    pub fn image_shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// The layout this dataset was built with.
    pub fn layout(&self) -> MnistLayout {
        self.layout
    }

    /// The whole `F32` pixel tensor: `[items, rows * cols]` under
    /// [`MnistLayout::Flat`], `[items, 1, rows, cols]` under
    /// [`MnistLayout::Nchw`].
    pub fn inputs(&self) -> &Tensor {
        self.inner.inputs()
    }

    /// The whole `[items]` `I64` label tensor.
    pub fn targets(&self) -> &Tensor {
        self.inner.targets()
    }

    /// The underlying [`TensorDataset`] the batching is delegated to.
    pub fn tensors(&self) -> &TensorDataset {
        &self.inner
    }
}

impl Dataset for MnistDataset {
    type Batch = (Tensor, Tensor);

    fn len(&self) -> usize {
        self.inner.len()
    }

    /// One gather per tensor on the dataset's device, delegated to
    /// [`TensorDataset::batch`].
    ///
    /// # Errors
    ///
    /// Exactly what the inner [`TensorDataset`] reports, so an empty or
    /// out-of-range index slice is an [`Error::InvalidArg`] /
    /// [`Error::IndexOutOfBounds`] naming `op: "TensorDataset::batch"` — the
    /// wrapper adds no index handling of its own.
    fn batch(&self, indices: &[usize]) -> Result<(Tensor, Tensor)> {
        self.inner.batch(indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataLoader;
    use crate::dtype::DType;
    use crate::testing::check_grad;

    const CPU: Device = Device::Cpu;

    fn image_fixture(count: u32, rows: u32, cols: u32, pixels: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(2051u32.to_be_bytes());
        bytes.extend(count.to_be_bytes());
        bytes.extend(rows.to_be_bytes());
        bytes.extend(cols.to_be_bytes());
        bytes.extend(pixels);
        bytes
    }

    fn label_fixture(labels: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(2049u32.to_be_bytes());
        bytes.extend(u32::try_from(labels.len()).unwrap().to_be_bytes());
        bytes.extend(labels);
        bytes
    }

    /// Two 2x2 images: the first all-zero-to-255 ramp, the second constant.
    /// Labels `[1, 2]` so they are valid classes for a 3-way loss.
    fn raw_split() -> Mnist {
        Mnist::from_idx_bytes(
            &image_fixture(2, 2, 2, &[0, 51, 102, 153, 204, 204, 255, 255]),
            &label_fixture(&[1, 2]),
        )
        .unwrap()
    }

    #[test]
    fn flat_layout_uploads_one_row_per_image() {
        let ds = MnistDataset::new(&raw_split(), MnistLayout::Flat, &CPU).unwrap();

        assert_eq!(ds.len(), 2);
        assert!(!ds.is_empty());
        assert_eq!(ds.image_shape(), (2, 2));
        assert_eq!(ds.layout(), MnistLayout::Flat);
        assert_eq!(ds.inputs().dims(), &[2, 4]);
        assert_eq!(ds.inputs().dtype(), DType::F32);
        assert_eq!(ds.targets().dims(), &[2]);
        assert_eq!(ds.targets().dtype(), DType::I64);
        assert_eq!(ds.targets().to_vec::<i64>().unwrap(), vec![1, 2]);
        // The parse already scaled pixels into [0, 1]; the wrapper keeps them.
        let first = ds.inputs().to_vec::<f32>().unwrap();
        assert_eq!(first[0], 0.0);
        assert_eq!(first[7], 1.0);
    }

    #[test]
    fn nchw_layout_reshapes_without_moving_pixels() {
        let raw = raw_split();
        let flat = MnistDataset::new(&raw, MnistLayout::Flat, &CPU).unwrap();
        let nchw = MnistDataset::new(&raw, MnistLayout::Nchw, &CPU).unwrap();

        assert_eq!(nchw.inputs().dims(), &[2, 1, 2, 2]);
        assert_eq!(nchw.layout(), MnistLayout::Nchw);
        assert_eq!(nchw.image_shape(), (2, 2));
        // Same elements in the same row-major order, only regrouped.
        assert_eq!(
            nchw.inputs().to_vec::<f32>().unwrap(),
            flat.inputs().to_vec::<f32>().unwrap()
        );
        assert_eq!(nchw.batch(&[0, 1]).unwrap().0.dims(), &[2, 1, 2, 2]);
    }

    #[test]
    fn batch_follows_the_index_slice_including_repeats() {
        let ds = MnistDataset::new(&raw_split(), MnistLayout::Flat, &CPU).unwrap();
        let (x, y) = ds.batch(&[1, 0, 1]).unwrap();

        assert_eq!(x.dims(), &[3, 4]);
        assert_eq!(y.to_vec::<i64>().unwrap(), vec![2, 1, 2]);
        let rows = x.to_vec::<f32>().unwrap();
        let whole = ds.inputs().to_vec::<f32>().unwrap();
        assert_eq!(&rows[0..4], &whole[4..8]);
        assert_eq!(&rows[4..8], &whole[0..4]);
        assert_eq!(&rows[8..12], &whole[4..8]);
    }

    #[test]
    fn an_empty_split_is_a_dataset_of_length_zero() {
        let raw = Mnist::from_idx_bytes(&image_fixture(0, 2, 2, &[]), &label_fixture(&[])).unwrap();
        let ds = MnistDataset::new(&raw, MnistLayout::Nchw, &CPU).unwrap();
        assert_eq!(ds.len(), 0);
        assert!(ds.is_empty());
        assert_eq!(ds.inputs().dims(), &[0, 1, 2, 2]);
    }

    #[test]
    fn index_errors_come_from_the_inner_tensor_dataset() {
        let ds = MnistDataset::new(&raw_split(), MnistLayout::Flat, &CPU).unwrap();
        assert!(matches!(
            ds.batch(&[2]),
            Err(Error::IndexOutOfBounds {
                op: "TensorDataset::batch",
                index: 2,
                size: 2,
                ..
            })
        ));
        assert!(matches!(
            ds.batch(&[]),
            Err(Error::InvalidArg {
                op: "TensorDataset::batch",
                ..
            })
        ));
        assert_eq!(ds.tensors().len(), 2);
    }

    /// The point of the wrapper: a parsed split drops straight into the loader,
    /// which is the only thing that decides which positions land in which batch.
    #[test]
    fn a_split_drives_a_data_loader() {
        let raw = Mnist::from_idx_bytes(
            &image_fixture(5, 1, 2, &[0, 10, 20, 30, 40, 50, 60, 70, 80, 90]),
            &label_fixture(&[0, 1, 2, 1, 0]),
        )
        .unwrap();
        let ds = MnistDataset::new(&raw, MnistLayout::Nchw, &CPU).unwrap();
        let loader = DataLoader::new(&ds, 2).shuffle(7);

        assert_eq!(loader.num_batches(), 3); // 2 + 2 + 1: the tail is kept
        let mut seen = 0;
        for batch in loader.batches() {
            let (x, y) = batch.unwrap();
            assert_eq!(x.dims()[1..], [1, 1, 2]);
            assert_eq!(x.dims()[0], y.dims()[0]);
            assert_eq!(y.dtype(), DType::I64);
            seen += y.dims()[0];
        }
        assert_eq!(seen, 5); // every item exactly once per epoch
    }

    /// A batch is an ordinary pair of tensors in the autograd graph: the pixels
    /// multiply a traced weight and the dataset's own `I64` labels index the
    /// cross-entropy, so a wrong label dtype or an out-of-range class would
    /// fail here rather than in a fixture.
    ///
    /// The objective runs in the split's own `F32` (MNIST pixels are `F32`, and
    /// the `F32`↔`F64` cast lane is not implemented yet), so the step is
    /// widened to `1e-2`: below that the central difference is dominated by
    /// single-precision roundoff rather than by the backward formula.
    #[test]
    fn a_batch_trains_a_traced_weight() {
        let ds = MnistDataset::new(&raw_split(), MnistLayout::Flat, &CPU).unwrap();
        let (x, y) = ds.batch(&[0, 1, 0]).unwrap();
        assert_eq!(x.dtype(), DType::F32);
        let weight = Tensor::from_vec(
            vec![
                0.5f32, -0.25, 1.0, 0.75, -0.5, 0.25, 0.1, -0.1, 0.3, 0.2, -0.7, 0.6,
            ],
            [4, 3],
            &CPU,
        )
        .unwrap();

        check_grad(
            |inputs| x.matmul(&inputs[0])?.cross_entropy(&y),
            &[weight],
            1e-2,
            1e-3,
        )
        .unwrap();
    }
}
