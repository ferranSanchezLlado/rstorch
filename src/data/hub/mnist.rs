//! Raw MNIST loading: download, cache, and IDX parsing into plain `Vec`s.
//!
//! This module is `Tensor`-free. It yields
//! normalized `f32` pixel vectors and `u8` labels; turning those into batched
//! tensors is the job of the `Dataset` wrappers.

#[cfg(feature = "hub")]
use super::DatasetHub;
use super::DatasetResource;
use crate::error::{Error, Result};

#[cfg(feature = "hub")]
const MNIST_DATASET: &str = "mnist";

/// The training-set image archive.
pub const TRAIN_IMAGES: DatasetResource = DatasetResource {
    name: "train images",
    url: "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    file_name: "train-images-idx3-ubyte.gz",
    sha256: Some("440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609"),
    max_bytes: Some(12_000_000),
};
/// The training-set label archive.
pub const TRAIN_LABELS: DatasetResource = DatasetResource {
    name: "train labels",
    url: "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    file_name: "train-labels-idx1-ubyte.gz",
    sha256: Some("3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c"),
    max_bytes: Some(50_000),
};
/// The test-set image archive.
pub const TEST_IMAGES: DatasetResource = DatasetResource {
    name: "test images",
    url: "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    file_name: "t10k-images-idx3-ubyte.gz",
    sha256: Some("8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6"),
    max_bytes: Some(2_000_000),
};
/// The test-set label archive.
pub const TEST_LABELS: DatasetResource = DatasetResource {
    name: "test labels",
    url: "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
    file_name: "t10k-labels-idx1-ubyte.gz",
    sha256: Some("f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6"),
    max_bytes: Some(10_000),
};

/// Which half of MNIST to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MnistSplit {
    /// The 60 000-image training split.
    Train,
    /// The 10 000-image test split.
    Test,
}

impl MnistSplit {
    /// The `[images, labels]` resources backing this split.
    #[cfg(feature = "hub")]
    fn resources(self) -> [DatasetResource; 2] {
        match self {
            Self::Train => [TRAIN_IMAGES, TRAIN_LABELS],
            Self::Test => [TEST_IMAGES, TEST_LABELS],
        }
    }
}

/// A single MNIST example: one flattened image and its class label.
#[derive(Debug, Clone, PartialEq)]
pub struct MnistSample {
    /// Row-major pixels normalized to `[0.0, 1.0]`; length `rows * cols`.
    pub image: Vec<f32>,
    /// The digit class in `0..=9`.
    pub label: u8,
}

/// The result of parsing an IDX image file: the raw pixel data plus its
/// geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct RawImages {
    /// One `f32` pixel vector per image, each normalized to `[0.0, 1.0]`.
    pub images: Vec<Vec<f32>>,
    /// Image height in pixels.
    pub rows: usize,
    /// Image width in pixels.
    pub cols: usize,
}

/// A parsed MNIST split held entirely in memory as plain `Vec`s.
#[derive(Debug, Clone)]
pub struct Mnist {
    images: Vec<Vec<f32>>,
    labels: Vec<u8>,
    rows: usize,
    cols: usize,
}

impl Mnist {
    /// Downloads all four MNIST archives into `hub`'s cache if missing.
    ///
    /// Requires the `hub` feature (network access).
    ///
    /// # Errors
    ///
    /// As [`DatasetHub::ensure_cached`].
    #[cfg(feature = "hub")]
    pub fn download(hub: &DatasetHub) -> Result<()> {
        hub.ensure_cached(
            MNIST_DATASET,
            &[TRAIN_IMAGES, TRAIN_LABELS, TEST_IMAGES, TEST_LABELS],
        )?;
        Ok(())
    }

    /// Loads a split, downloading and caching the archives if necessary.
    ///
    /// Requires the `hub` feature (network access + gzip).
    ///
    /// # Errors
    ///
    /// As [`DatasetHub::ensure_resource`] and
    /// [`from_gzip_files`](Self::from_gzip_files).
    #[cfg(feature = "hub")]
    pub fn load(hub: &DatasetHub, split: MnistSplit) -> Result<Self> {
        let [images_resource, labels_resource] = split.resources();
        let image_path = hub.ensure_resource(MNIST_DATASET, &images_resource)?;
        let label_path = hub.ensure_resource(MNIST_DATASET, &labels_resource)?;
        Self::from_gzip_files(image_path, label_path)
    }

    /// Parses a split from two gzip-compressed IDX files on disk.
    ///
    /// Requires the `hub` feature (gzip decompression).
    ///
    /// # Errors
    ///
    /// Propagates any file I/O error opening `images_path`/`labels_path`.
    /// Returns [`Error::Data`] if decompression exceeds the crate's fixed
    /// 64&nbsp;MiB cap, or as [`from_idx_bytes`](Self::from_idx_bytes) for
    /// the decompressed contents.
    #[cfg(feature = "hub")]
    pub fn from_gzip_files(
        images_path: impl AsRef<std::path::Path>,
        labels_path: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        let images = read_gzip(images_path)?;
        let labels = read_gzip(labels_path)?;
        Self::from_idx_bytes(&images, &labels)
    }

    /// Parses a split directly from decompressed IDX image and label bytes.
    ///
    /// This is feature-free: give it the raw IDX bytes (already decompressed)
    /// and it validates the headers, normalizes pixels, and cross-checks that
    /// the image and label counts agree.
    ///
    /// # Errors
    ///
    /// As [`parse_idx_images`] and [`parse_idx_labels`]. Returns
    /// [`Error::Data`] if the parsed image and label counts disagree.
    pub fn from_idx_bytes(images: &[u8], labels: &[u8]) -> Result<Self> {
        let parsed = parse_idx_images(images)?;
        let labels = parse_idx_labels(labels)?;
        if parsed.images.len() != labels.len() {
            return Err(Error::Data {
                msg: format!(
                    "MNIST image/label count mismatch: {} images vs {} labels",
                    parsed.images.len(),
                    labels.len()
                ),
            });
        }
        Ok(Self {
            images: parsed.images,
            labels,
            rows: parsed.rows,
            cols: parsed.cols,
        })
    }

    /// The `(rows, cols)` geometry of every image (always `(28, 28)` for the
    /// real dataset).
    pub fn image_shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// The number of examples in this split.
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Whether this split has no examples.
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// All flattened, normalized image vectors, in order.
    pub fn images(&self) -> &[Vec<f32>] {
        &self.images
    }

    /// All class labels, in order.
    pub fn labels(&self) -> &[u8] {
        &self.labels
    }

    /// Borrows the flattened pixels and label for one example, or `None` if
    /// `index` is out of range.
    pub fn get(&self, index: usize) -> Option<(&[f32], u8)> {
        Some((self.images.get(index)?.as_slice(), self.labels[index]))
    }

    /// Clones one example into an owned [`MnistSample`], or `None` if `index`
    /// is out of range.
    pub fn sample(&self, index: usize) -> Option<MnistSample> {
        Some(MnistSample {
            image: self.images.get(index)?.clone(),
            label: self.labels[index],
        })
    }
}

/// The largest decompressed IDX payload this module will hold in memory.
///
/// The biggest real MNIST member is the training image set at
/// `60_000 * 28 * 28 + 16` = 47 040 016 bytes, so 64 MiB clears it with room to
/// spare. The bound matters because the size caps on the resources above apply
/// to the *compressed* archive: DEFLATE reaches ratios above 1000:1, so a
/// 12 MB archive that passed its cap can otherwise expand to gigabytes. Small
/// enough to keep a malicious archive from exhausting memory, large enough that
/// no legitimate MNIST file comes close.
#[cfg(feature = "hub")]
const MAX_DECOMPRESSED_BYTES: u64 = 64 * 1024 * 1024;

/// Decompresses a gzip file into raw bytes. Requires the `hub` feature.
///
/// The output is bounded by [`MAX_DECOMPRESSED_BYTES`]; a larger archive is
/// rejected rather than expanded, so a gzip bomb cannot exhaust memory. The
/// reader takes one byte past the cap so an over-cap archive is *detected*
/// instead of being silently truncated to a valid-looking prefix.
#[cfg(feature = "hub")]
fn read_gzip(path: impl AsRef<std::path::Path>) -> Result<Vec<u8>> {
    use std::io::Read as _;
    let file = std::fs::File::open(path)?;
    let decoder = flate2::read::GzDecoder::new(file);
    let mut bytes = Vec::new();
    decoder
        .take(MAX_DECOMPRESSED_BYTES + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > MAX_DECOMPRESSED_BYTES {
        return Err(data_error(&format!(
            "decompressed archive exceeds the {MAX_DECOMPRESSED_BYTES} byte cap"
        )));
    }
    Ok(bytes)
}

/// Parses a decompressed IDX3 (image) file into normalized pixel vectors.
///
/// Pixels are scaled from `0..=255` to `[0.0, 1.0]`. The header magic, the
/// declared geometry, and the exact byte length are all validated.
///
/// # Errors
///
/// Returns [`Error::Data`] if `bytes` is shorter than the IDX3 header, has
/// the wrong magic, declares a zero-sized geometry, the declared element
/// count overflows, or the byte length does not match the declared
/// geometry.
pub fn parse_idx_images(bytes: &[u8]) -> Result<RawImages> {
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
    // A zero-sized geometry makes the payload length zero regardless of
    // `count`, so the exact-length check below would accept a 16-byte header
    // declaring billions of (empty) images and `count` alone would drive the
    // allocation. Rejecting it here keeps `count` bounded by the input length,
    // which is what makes the rest of this function allocation-safe.
    if rows == 0 || cols == 0 {
        return parse_error("IDX image geometry must be non-zero");
    }
    let image_len = rows
        .checked_mul(cols)
        .ok_or_else(|| data_error("IDX image dimensions overflow"))?;
    let payload = count
        .checked_mul(image_len)
        .ok_or_else(|| data_error("IDX image byte count overflow"))?;
    let expected = 16usize
        .checked_add(payload)
        .ok_or_else(|| data_error("IDX image byte count overflow"))?;
    if bytes.len() != expected {
        return parse_error("truncated IDX image data");
    }

    // `image_len >= 1` (the geometry check above), so `chunks_exact` yields
    // exactly `count` chunks and needs no empty-image special case.
    let images = bytes[16..]
        .chunks_exact(image_len)
        .map(|image| image.iter().map(|&pixel| pixel as f32 / 255.0).collect())
        .collect();
    Ok(RawImages { images, rows, cols })
}

/// Parses a decompressed IDX1 (label) file into a vector of `u8` labels.
///
/// The header magic and the exact byte length are validated.
///
/// # Errors
///
/// Returns [`Error::Data`] if `bytes` is shorter than the IDX1 header, has
/// the wrong magic, the declared count overflows, or the byte length does
/// not match the declared count.
pub fn parse_idx_labels(bytes: &[u8]) -> Result<Vec<u8>> {
    if bytes.len() < 8 {
        return parse_error("truncated IDX label header");
    }
    let magic = read_u32(bytes, 0);
    if magic != 2049 {
        return parse_error("invalid IDX label magic");
    }
    let count = read_u32(bytes, 4) as usize;
    let expected = 8usize
        .checked_add(count)
        .ok_or_else(|| data_error("IDX label byte count overflow"))?;
    if bytes.len() != expected {
        return parse_error("truncated IDX label data");
    }
    Ok(bytes[8..].to_vec())
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn data_error(message: &str) -> Error {
    Error::Data {
        msg: message.to_owned(),
    }
}

fn parse_error<T>(message: &str) -> Result<T> {
    Err(data_error(message))
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn invalid_magic_and_truncated_files_are_data_errors() {
        let mut images = image_fixture(1, 1, 1, &[42]);
        images[3] = 0; // corrupt the magic number
        assert!(matches!(parse_idx_images(&images), Err(Error::Data { .. })));
        assert!(matches!(
            parse_idx_images(&images[..8]),
            Err(Error::Data { .. })
        ));

        let mut labels = label_fixture(&[1]);
        labels[3] = 0;
        assert!(matches!(parse_idx_labels(&labels), Err(Error::Data { .. })));
        assert!(matches!(
            parse_idx_labels(&labels[..4]),
            Err(Error::Data { .. })
        ));
    }

    #[test]
    fn wrong_length_is_rejected() {
        // Header declares two 1x1 images but only one pixel of data follows.
        let mut images = image_fixture(2, 1, 1, &[42]);
        // truncated: only one pixel present for two declared images
        assert!(matches!(parse_idx_images(&images), Err(Error::Data { .. })));
        // extra trailing byte also rejected
        images.push(0);
        images.push(0);
        assert!(matches!(parse_idx_images(&images), Err(Error::Data { .. })));
    }

    /// A zero-sized geometry zeroes the declared payload, so the exact-length
    /// check alone would accept a 16-byte header claiming `u32::MAX` images and
    /// then allocate ~96 GiB for the outer `Vec` — an abort, not a `Result`.
    /// The geometry must be rejected before anything is allocated.
    #[test]
    fn zero_geometry_is_rejected_before_allocating() {
        for (rows, cols) in [(0u32, 28u32), (28, 0), (0, 0)] {
            let header = image_fixture(u32::MAX, rows, cols, &[]);
            assert_eq!(header.len(), 16, "fixture must be header-only");
            assert!(
                matches!(parse_idx_images(&header), Err(Error::Data { .. })),
                "rows={rows} cols={cols} must be a data error, not an allocation"
            );
        }
        // A legitimately empty split (no images, real geometry) still parses.
        let empty = parse_idx_images(&image_fixture(0, 28, 28, &[])).unwrap();
        assert!(empty.images.is_empty());
        assert_eq!((empty.rows, empty.cols), (28, 28));
    }

    /// The decompression cap must reject a bomb rather than expand it. Built
    /// here with `flate2`'s encoder so no fixture file is needed: a highly
    /// compressible run far larger than the cap.
    #[cfg(feature = "hub")]
    #[test]
    fn gzip_bomb_is_rejected_by_the_decompressed_cap() {
        use std::io::Write as _;

        let dir = std::env::temp_dir().join(format!("rstorch-gzip-cap-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bomb.gz");

        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::best());
        // One byte past the cap is enough to trip it; zeros compress ~1000:1.
        let chunk = vec![0u8; 1 << 20];
        let mut written = 0u64;
        while written <= MAX_DECOMPRESSED_BYTES {
            encoder.write_all(&chunk).unwrap();
            written += chunk.len() as u64;
        }
        std::fs::write(&path, encoder.finish().unwrap()).unwrap();

        let err = read_gzip(&path).expect_err("over-cap archive must be rejected");
        assert!(
            matches!(&err, Error::Data { msg } if msg.contains("exceeds")),
            "expected a cap error, got {err:?}"
        );

        // A small archive still round-trips through the same path.
        let mut ok = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::best());
        ok.write_all(b"hello").unwrap();
        std::fs::write(&path, ok.finish().unwrap()).unwrap();
        assert_eq!(read_gzip(&path).unwrap(), b"hello");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_idx_bytes_builds_dataset() {
        let pixels: Vec<u8> = (0..(2 * 4)).map(|v| (v * 10) as u8).collect();
        let ds = Mnist::from_idx_bytes(&image_fixture(2, 2, 2, &pixels), &label_fixture(&[4, 9]))
            .unwrap();

        assert_eq!(ds.len(), 2);
        assert!(!ds.is_empty());
        assert_eq!(ds.image_shape(), (2, 2));
        assert_eq!(ds.labels(), &[4, 9]);
        assert_eq!(ds.images().len(), 2);
        assert_eq!(ds.images()[0].len(), 4);

        let (pix, label) = ds.get(1).unwrap();
        assert_eq!(pix.len(), 4);
        assert_eq!(label, 9);
        assert!(ds.get(2).is_none());

        let sample = ds.sample(0).unwrap();
        assert_eq!(sample.label, 4);
        assert_eq!(sample.image.len(), 4);
        assert!(ds.sample(2).is_none());
    }

    #[test]
    fn image_label_count_mismatch_is_rejected() {
        let images = image_fixture(2, 1, 1, &[1, 2]);
        let labels = label_fixture(&[0]);
        assert!(matches!(
            Mnist::from_idx_bytes(&images, &labels),
            Err(Error::Data { .. })
        ));
    }

    // Exercises the gzip decode path fully offline: gzip is only compiled
    // with `hub`, so we both compress (encoder) and decompress (decoder)
    // in-process without ever touching the network.
    #[cfg(feature = "hub")]
    #[test]
    fn from_gzip_files_round_trips_synthetic_fixtures() {
        use flate2::{Compression, write::GzEncoder};
        use std::io::Write as _;

        fn gz(bytes: &[u8]) -> Vec<u8> {
            let mut enc = GzEncoder::new(Vec::new(), Compression::fast());
            enc.write_all(bytes).unwrap();
            enc.finish().unwrap()
        }

        let root = std::env::temp_dir().join(format!(
            "rstorch-mnist-gz-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        let img_path = root.join("images.gz");
        let lbl_path = root.join("labels.gz");
        std::fs::write(&img_path, gz(&image_fixture(1, 2, 2, &[0, 255, 128, 64]))).unwrap();
        std::fs::write(&lbl_path, gz(&label_fixture(&[5]))).unwrap();

        let ds = Mnist::from_gzip_files(&img_path, &lbl_path).unwrap();
        assert_eq!(ds.len(), 1);
        assert_eq!(ds.image_shape(), (2, 2));
        assert_eq!(ds.labels(), &[5]);
        assert_eq!(ds.images()[0][0], 0.0);
        assert_eq!(ds.images()[0][1], 1.0);

        let _ = std::fs::remove_dir_all(root);
    }
}
