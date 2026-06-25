use super::{DatasetHub, DatasetResource};
use crate::backend::{Backend, Cpu};
use crate::data::{Batch, Collate, Dataset, StackImageCollate, StackVecCollate};
use crate::dtype::FloatDType;
use crate::error::{DataError, Error, Result};
use crate::shape::{C, D2, D4, Sym};
use crate::tensor::Tensor;
use flate2::read::GzDecoder;
use std::fs;
use std::io::Read;
use std::marker::PhantomData;
use std::path::Path;

const MNIST_DATASET: &str = "mnist";
pub const TRAIN_IMAGES: DatasetResource = DatasetResource {
    name: "train images",
    url: "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    file_name: "train-images-idx3-ubyte.gz",
};
pub const TRAIN_LABELS: DatasetResource = DatasetResource {
    name: "train labels",
    url: "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    file_name: "train-labels-idx1-ubyte.gz",
};
pub const TEST_IMAGES: DatasetResource = DatasetResource {
    name: "test images",
    url: "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    file_name: "t10k-images-idx3-ubyte.gz",
};
pub const TEST_LABELS: DatasetResource = DatasetResource {
    name: "test labels",
    url: "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
    file_name: "t10k-labels-idx1-ubyte.gz",
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MnistSplit {
    Train,
    Test,
}

impl MnistSplit {
    fn resources(self) -> [DatasetResource; 2] {
        match self {
            Self::Train => [TRAIN_IMAGES, TRAIN_LABELS],
            Self::Test => [TEST_IMAGES, TEST_LABELS],
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MnistSample {
    pub image: Vec<f32>,
    pub label: u8,
}

#[derive(Debug, Clone)]
pub struct Mnist {
    images: Vec<Vec<f32>>,
    labels: Vec<u8>,
    rows: usize,
    cols: usize,
}

impl Mnist {
    pub fn download(hub: &DatasetHub) -> Result<()> {
        hub.ensure_cached(
            MNIST_DATASET,
            &[TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS],
        )?;
        Ok(())
    }

    pub fn load(hub: &DatasetHub, split: MnistSplit) -> Result<Self> {
        let [images_resource, labels_resource] = split.resources();
        let image_path = hub.ensure_resource(MNIST_DATASET, &images_resource)?;
        let label_path = hub.ensure_resource(MNIST_DATASET, &labels_resource)?;
        Self::from_gzip_files(image_path, label_path)
    }

    pub fn from_gzip_files(
        images_path: impl AsRef<Path>,
        labels_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let images = read_gzip(images_path)?;
        let labels = read_gzip(labels_path)?;
        Self::from_idx_bytes(&images, &labels)
    }

    pub fn from_idx_bytes(images: &[u8], labels: &[u8]) -> Result<Self> {
        let parsed_images = parse_idx_images(images)?;
        let labels = parse_idx_labels(labels)?;
        if parsed_images.images.len() != labels.len() {
            return Err(DataError::InconsistentSampleShape {
                index: 0,
                expected: vec![parsed_images.images.len()],
                found: vec![labels.len()],
            }
            .into());
        }
        Ok(Self {
            images: parsed_images.images,
            labels,
            rows: parsed_images.rows,
            cols: parsed_images.cols,
        })
    }

    pub fn image_shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

impl Dataset for Mnist {
    type Item = MnistSample;
    type Error = DataError;

    fn len(&self) -> usize {
        self.labels.len()
    }

    fn get(&self, index: usize) -> std::result::Result<Self::Item, Self::Error> {
        if index >= self.len() {
            return Err(DataError::IndexOutOfBounds {
                index,
                len: self.len(),
            });
        }
        Ok(MnistSample {
            image: self.images[index].clone(),
            label: self.labels[index],
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MnistCollate<E = f32, B = Cpu> {
    _marker: PhantomData<(E, B)>,
}

impl<E, B> MnistCollate<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<E, B> Default for MnistCollate<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<E, B> Collate<MnistSample> for MnistCollate<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    type Batch = (Tensor<D2<Sym<Batch>, C<784>>, E, B>, Vec<u8>);
    type Error = Error;

    fn collate(&self, items: Vec<MnistSample>) -> Result<Self::Batch> {
        let items = convert_mnist_samples::<E>(items);
        StackVecCollate::<784, E, B>::new().collate(items)
    }
}

pub type MnistImageBatch<E = f32, B = Cpu> = Tensor<D4<Sym<Batch>, C<1>, C<28>, C<28>>, E, B>;

#[derive(Debug, Clone, Copy)]
pub struct MnistImageCollate<E = f32, B = Cpu> {
    _marker: PhantomData<(E, B)>,
}

impl<E, B> MnistImageCollate<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<E, B> Default for MnistImageCollate<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<E, B> Collate<MnistSample> for MnistImageCollate<E, B>
where
    E: FloatDType,
    B: Backend<E>,
{
    type Batch = (MnistImageBatch<E, B>, Vec<u8>);
    type Error = Error;

    fn collate(&self, items: Vec<MnistSample>) -> Result<Self::Batch> {
        let items = convert_mnist_samples::<E>(items);
        StackImageCollate::<1, 28, 28, E, B>::new().collate(items)
    }
}

fn convert_mnist_samples<E>(items: Vec<MnistSample>) -> Vec<(Vec<E>, u8)>
where
    E: FloatDType,
{
    items
        .into_iter()
        .map(|sample| {
            (
                sample
                    .image
                    .into_iter()
                    .map(|pixel| E::from_f64(pixel as f64))
                    .collect(),
                sample.label,
            )
        })
        .collect()
}

struct ParsedImages {
    images: Vec<Vec<f32>>,
    rows: usize,
    cols: usize,
}

fn read_gzip(path: impl AsRef<Path>) -> Result<Vec<u8>> {
    let file = fs::File::open(path).map_err(|source| DataError::Io { source })?;
    let mut decoder = GzDecoder::new(file);
    let mut bytes = Vec::new();
    decoder
        .read_to_end(&mut bytes)
        .map_err(|source| DataError::Io { source })?;
    Ok(bytes)
}

fn parse_idx_images(bytes: &[u8]) -> Result<ParsedImages> {
    if bytes.len() < 16 {
        return parse_error("truncated IDX image header");
    }
    let magic = read_u32(bytes, 0);
    if magic != 2051 {
        return parse_error("invalid IDX image magic");
    }
    let count = read_u32(bytes, 4) as usize;
    let rows = read_u32(bytes, 8) as usize;
    let cols = read_u32(bytes, 12) as usize;
    let image_len = rows.checked_mul(cols).ok_or_else(|| DataError::Parse {
        source: "IDX image dimensions overflow".into(),
    })?;
    let expected = 16usize
        .checked_add(
            count
                .checked_mul(image_len)
                .ok_or_else(|| DataError::Parse {
                    source: "IDX image byte count overflow".into(),
                })?,
        )
        .ok_or_else(|| DataError::Parse {
            source: "IDX image byte count overflow".into(),
        })?;
    if bytes.len() != expected {
        return parse_error("truncated IDX image data");
    }

    let images = bytes[16..]
        .chunks_exact(image_len)
        .map(|image| image.iter().map(|&pixel| pixel as f32 / 255.0).collect())
        .collect();
    Ok(ParsedImages { images, rows, cols })
}

fn parse_idx_labels(bytes: &[u8]) -> Result<Vec<u8>> {
    if bytes.len() < 8 {
        return parse_error("truncated IDX label header");
    }
    let magic = read_u32(bytes, 0);
    if magic != 2049 {
        return parse_error("invalid IDX label magic");
    }
    let count = read_u32(bytes, 4) as usize;
    let expected = 8usize.checked_add(count).ok_or_else(|| DataError::Parse {
        source: "IDX label byte count overflow".into(),
    })?;
    if bytes.len() != expected {
        return parse_error("truncated IDX label data");
    }
    Ok(bytes[8..].to_vec())
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn parse_error<T>(message: &'static str) -> Result<T> {
    Err(DataError::Parse {
        source: message.into(),
    }
    .into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{DataLoader, SequentialSampler};
    use crate::error::Error;

    #[test]
    fn synthetic_idx_image_parser_works() {
        let images = image_fixture(2, 2, 2, &[0, 127, 255, 10, 20, 30, 40, 50]);
        let parsed = parse_idx_images(&images).unwrap();

        assert_eq!(parsed.rows, 2);
        assert_eq!(parsed.cols, 2);
        assert_eq!(parsed.images.len(), 2);
        assert_eq!(parsed.images[0][0], 0.0);
        assert_eq!(parsed.images[0][2], 1.0);
    }

    #[test]
    fn synthetic_idx_label_parser_works() {
        let labels = label_fixture(&[3, 7]);

        assert_eq!(parse_idx_labels(&labels).unwrap(), vec![3, 7]);
    }

    #[test]
    fn invalid_magic_and_truncated_files_are_parse_errors() {
        let mut images = image_fixture(1, 1, 1, &[42]);
        images[3] = 0;
        assert!(matches!(
            parse_idx_images(&images),
            Err(Error::Data(DataError::Parse { .. }))
        ));
        assert!(matches!(
            parse_idx_images(&images[..8]),
            Err(Error::Data(DataError::Parse { .. }))
        ));

        let mut labels = label_fixture(&[1]);
        labels[3] = 0;
        assert!(matches!(
            parse_idx_labels(&labels),
            Err(Error::Data(DataError::Parse { .. }))
        ));
        assert!(matches!(
            parse_idx_labels(&labels[..4]),
            Err(Error::Data(DataError::Parse { .. }))
        ));
    }

    #[test]
    fn mnist_dataset_and_collate_produce_flat_batches() {
        let pixels: Vec<u8> = (0..(2 * 28 * 28))
            .map(|value| (value % 255) as u8)
            .collect();
        let dataset =
            Mnist::from_idx_bytes(&image_fixture(2, 28, 28, &pixels), &label_fixture(&[4, 9]))
                .unwrap();
        let loader = DataLoader::new(
            dataset,
            SequentialSampler,
            MnistCollate::<f32>::new(),
            2,
            false,
        )
        .unwrap();
        let (images, labels) = loader.iter().next().unwrap().unwrap();

        assert_eq!(images.shape().dims(), &[2, 784]);
        assert_eq!(labels, vec![4, 9]);
    }

    #[test]
    fn mnist_collate_can_produce_image_batches() {
        let pixels: Vec<u8> = (0..(2 * 28 * 28))
            .map(|value| (value % 255) as u8)
            .collect();
        let dataset =
            Mnist::from_idx_bytes(&image_fixture(2, 28, 28, &pixels), &label_fixture(&[1, 2]))
                .unwrap();
        let loader = DataLoader::new(
            dataset,
            SequentialSampler,
            MnistImageCollate::<f32>::new(),
            2,
            false,
        )
        .unwrap();
        let (images, labels) = loader.iter().next().unwrap().unwrap();

        assert_eq!(images.shape().dims(), &[2, 1, 28, 28]);
        assert_eq!(labels, vec![1, 2]);
    }

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
        bytes.extend((labels.len() as u32).to_be_bytes());
        bytes.extend(labels);
        bytes
    }
}
