use crate::data::Dataset;
use flate2::read::GzDecoder;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const IDX_UNSIGNED_BYTE: u8 = 0x08;
const MNIST_ROWS: usize = 28;
const MNIST_COLS: usize = 28;
const MNIST_PIXELS: usize = MNIST_ROWS * MNIST_COLS;
const MNIST_BASE_URL: &str = "https://storage.googleapis.com/cvdf-datasets/mnist/";

const MNIST_FILES: [MnistFile; 4] = [
    MnistFile {
        name: "train-images-idx3-ubyte",
    },
    MnistFile {
        name: "train-labels-idx1-ubyte",
    },
    MnistFile {
        name: "t10k-images-idx3-ubyte",
    },
    MnistFile {
        name: "t10k-labels-idx1-ubyte",
    },
];

#[derive(Clone, Copy)]
struct MnistFile {
    name: &'static str,
}

#[derive(Debug)]
pub enum MnistError {
    Io(std::io::Error),
    InvalidIdx { reason: String },
    InvalidMnist { reason: String },
    Download { url: String, reason: String },
}

impl std::fmt::Display for MnistError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::InvalidIdx { reason } => write!(f, "invalid IDX file: {reason}"),
            Self::InvalidMnist { reason } => write!(f, "invalid MNIST data: {reason}"),
            Self::Download { url, reason } => write!(f, "failed to download {url}: {reason}"),
        }
    }
}

impl std::error::Error for MnistError {}

impl From<std::io::Error> for MnistError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Debug)]
struct IdxData {
    dimensions: Vec<usize>,
    data: Vec<u8>,
}

fn parse_idx(bytes: &[u8]) -> Result<IdxData, MnistError> {
    if bytes.len() < 4 {
        return Err(invalid_idx("truncated IDX header"));
    }

    if bytes[0] != 0 || bytes[1] != 0 {
        return Err(invalid_idx("magic number must start with two zero bytes"));
    }

    if bytes[2] != IDX_UNSIGNED_BYTE {
        return Err(invalid_idx(format!(
            "unsupported data type 0x{:02x}; only unsigned byte is supported",
            bytes[2]
        )));
    }

    let rank = usize::from(bytes[3]);
    if rank == 0 {
        return Err(invalid_idx("rank must be greater than zero"));
    }

    let header_len = 4 + rank * 4;
    if bytes.len() < header_len {
        return Err(invalid_idx("truncated IDX dimensions"));
    }

    let mut dimensions = Vec::with_capacity(rank);
    let mut expected_len = 1_usize;
    for dimension in 0..rank {
        let offset = 4 + dimension * 4;
        let size = u32::from_be_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]) as usize;
        expected_len = expected_len
            .checked_mul(size)
            .ok_or_else(|| invalid_idx("IDX dimensions overflow usize"))?;
        dimensions.push(size);
    }

    let actual_len = bytes.len() - header_len;
    if actual_len != expected_len {
        return Err(invalid_idx(format!(
            "payload length mismatch: expected {expected_len} bytes, found {actual_len}"
        )));
    }

    Ok(IdxData {
        dimensions,
        data: bytes[header_len..].to_vec(),
    })
}

fn invalid_idx(reason: impl Into<String>) -> MnistError {
    MnistError::InvalidIdx {
        reason: reason.into(),
    }
}

fn invalid_mnist(reason: impl Into<String>) -> MnistError {
    MnistError::InvalidMnist {
        reason: reason.into(),
    }
}

fn normalize_pixel(pixel: u8) -> f32 {
    f32::from(pixel) / 255.0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MnistSplit {
    Train,
    Test,
}

impl MnistSplit {
    fn files(self) -> (&'static str, &'static str) {
        match self {
            Self::Train => ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
            Self::Test => ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Mnist {
    images: Vec<[[u8; MNIST_COLS]; MNIST_ROWS]>,
    labels: Vec<u8>,
}

impl Mnist {
    pub fn train() -> Result<Self, MnistError> {
        let cache_dir = ensure_mnist_cache()?;
        Self::from_dir_split(cache_dir, MnistSplit::Train)
    }

    pub fn test() -> Result<Self, MnistError> {
        let cache_dir = ensure_mnist_cache()?;
        Self::from_dir_split(cache_dir, MnistSplit::Test)
    }

    pub fn from_dir(path: impl AsRef<Path>) -> Result<Self, MnistError> {
        Self::train_from_dir(path)
    }

    pub fn train_from_dir(path: impl AsRef<Path>) -> Result<Self, MnistError> {
        Self::from_dir_split(path, MnistSplit::Train)
    }

    pub fn test_from_dir(path: impl AsRef<Path>) -> Result<Self, MnistError> {
        Self::from_dir_split(path, MnistSplit::Test)
    }

    pub fn from_dir_split(path: impl AsRef<Path>, split: MnistSplit) -> Result<Self, MnistError> {
        let (images_file, labels_file) = split.files();
        let images = read_idx_file(path.as_ref(), images_file)?;
        let labels = read_idx_file(path.as_ref(), labels_file)?;
        let images = parse_mnist_images(&images)?;
        let labels = parse_mnist_labels(&labels)?;

        if images.len() != labels.len() {
            return Err(invalid_mnist(format!(
                "image count {} does not match label count {}",
                images.len(),
                labels.len()
            )));
        }

        Ok(Self { images, labels })
    }

    pub fn labels(&self) -> &[u8] {
        &self.labels
    }
}

impl Dataset for Mnist {
    type Item = ([[f32; MNIST_COLS]; MNIST_ROWS], u8);

    fn len(&self) -> usize {
        self.labels.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        let image = self.images.get(index)?;
        let label = *self.labels.get(index)?;
        Some((normalize_image(image), label))
    }
}

fn normalize_image(image: &[[u8; MNIST_COLS]; MNIST_ROWS]) -> [[f32; MNIST_COLS]; MNIST_ROWS] {
    let mut normalized = [[0.0; MNIST_COLS]; MNIST_ROWS];
    for row in 0..MNIST_ROWS {
        for col in 0..MNIST_COLS {
            normalized[row][col] = normalize_pixel(image[row][col]);
        }
    }
    normalized
}

fn parse_mnist_images(bytes: &[u8]) -> Result<Vec<[[u8; MNIST_COLS]; MNIST_ROWS]>, MnistError> {
    let idx = parse_idx(bytes)?;
    if idx.dimensions.len() != 3
        || idx.dimensions[1] != MNIST_ROWS
        || idx.dimensions[2] != MNIST_COLS
    {
        return Err(invalid_mnist(format!(
            "expected image dimensions [N, {MNIST_ROWS}, {MNIST_COLS}], found {:?}",
            idx.dimensions
        )));
    }

    let count = idx.dimensions[0];
    let mut images = Vec::with_capacity(count);
    for chunk in idx.data.chunks_exact(MNIST_PIXELS) {
        let mut image = [[0_u8; MNIST_COLS]; MNIST_ROWS];
        for (row, pixels) in chunk.chunks_exact(MNIST_COLS).enumerate() {
            image[row].copy_from_slice(pixels);
        }
        images.push(image);
    }
    Ok(images)
}

fn parse_mnist_labels(bytes: &[u8]) -> Result<Vec<u8>, MnistError> {
    let idx = parse_idx(bytes)?;
    if idx.dimensions.len() != 1 {
        return Err(invalid_mnist(format!(
            "expected label dimensions [N], found {:?}",
            idx.dimensions
        )));
    }
    if let Some(label) = idx.data.iter().find(|&&label| label > 9) {
        return Err(invalid_mnist(format!(
            "label {label} is out of range for MNIST"
        )));
    }
    Ok(idx.data)
}

fn ensure_mnist_cache() -> Result<PathBuf, MnistError> {
    let cache_dir = default_cache_dir();
    fs::create_dir_all(&cache_dir)?;

    for file in MNIST_FILES {
        download_mnist_file(&cache_dir, file)?;
        if decompress_mnist_file(&cache_dir, file).is_err() {
            let _ = fs::remove_file(cache_dir.join(format!("{}.gz", file.name)));
            let _ = fs::remove_file(cache_dir.join(file.name));
            download_mnist_file(&cache_dir, file)?;
            decompress_mnist_file(&cache_dir, file)?;
        }
    }

    Ok(cache_dir)
}

fn default_cache_dir() -> PathBuf {
    if let Some(path) = std::env::var_os("RSTORCH_DATA") {
        return PathBuf::from(path).join("mnist");
    }
    PathBuf::from("target").join("rstorch-cache").join("mnist")
}

fn download_mnist_file(cache_dir: &Path, file: MnistFile) -> Result<(), MnistError> {
    let gz_name = format!("{}.gz", file.name);
    let gz_path = cache_dir.join(&gz_name);
    if gz_path.exists() {
        return Ok(());
    }

    let url = format!("{MNIST_BASE_URL}{gz_name}");
    let response = ureq::get(&url)
        .call()
        .map_err(|error| MnistError::Download {
            url: url.clone(),
            reason: error.to_string(),
        })?;

    let tmp_path = unique_tmp_path(cache_dir, &gz_name);
    let mut reader = response.into_reader();
    let mut output = File::create(&tmp_path)?;
    std::io::copy(&mut reader, &mut output)?;
    output.flush()?;
    finish_tmp_file(&tmp_path, &gz_path)?;
    Ok(())
}

fn decompress_mnist_file(cache_dir: &Path, file: MnistFile) -> Result<(), MnistError> {
    let output_path = cache_dir.join(file.name);
    if output_path.exists() {
        return Ok(());
    }

    let gz_path = cache_dir.join(format!("{}.gz", file.name));
    let gz_file = File::open(gz_path)?;
    let mut decoder = GzDecoder::new(gz_file);
    let tmp_path = unique_tmp_path(cache_dir, file.name);
    let mut output = File::create(&tmp_path)?;
    std::io::copy(&mut decoder, &mut output)?;
    output.flush()?;
    finish_tmp_file(&tmp_path, &output_path)?;
    Ok(())
}

fn unique_tmp_path(cache_dir: &Path, name: &str) -> PathBuf {
    let thread_id = format!("{:?}", std::thread::current().id());
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    cache_dir
        .join(format!(
            "{name}.{}.{}.tmp",
            std::process::id(),
            thread_id.replace(['(', ')'], "")
        ))
        .with_extension(format!("{timestamp}.tmp"))
}

fn finish_tmp_file(tmp_path: &Path, output_path: &Path) -> Result<(), MnistError> {
    if output_path.exists() {
        fs::remove_file(tmp_path)?;
        return Ok(());
    }

    match fs::rename(tmp_path, output_path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            fs::remove_file(tmp_path)?;
            Ok(())
        }
        Err(error) => Err(error.into()),
    }
}

fn read_idx_file(dir: &Path, name: &str) -> Result<Vec<u8>, MnistError> {
    let plain_path = dir.join(name);
    if plain_path.exists() {
        return Ok(fs::read(plain_path)?);
    }

    let gz_path = dir.join(format!("{name}.gz"));
    if gz_path.exists() {
        let gz_file = File::open(gz_path)?;
        let mut decoder = GzDecoder::new(gz_file);
        let mut bytes = Vec::new();
        decoder.read_to_end(&mut bytes)?;
        return Ok(bytes);
    }

    Err(MnistError::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("missing {name} or {name}.gz in {}", dir.display()),
    )))
}

#[cfg(test)]
mod tests {
    use super::{IdxData, MnistError, normalize_pixel, parse_idx};

    fn idx_bytes(rank: u8, dimensions: &[u32], payload: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0, 0, 0x08, rank];
        for dimension in dimensions {
            bytes.extend(dimension.to_be_bytes());
        }
        bytes.extend(payload);
        bytes
    }

    #[test]
    fn idx_parser_reads_dimensions_and_payload() {
        let bytes = idx_bytes(2, &[2, 3], &[1, 2, 3, 4, 5, 6]);
        let IdxData { dimensions, data } = parse_idx(&bytes).unwrap();

        assert_eq!(dimensions, vec![2, 3]);
        assert_eq!(data, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn idx_parser_rejects_bad_magic_number() {
        let mut bytes = idx_bytes(1, &[1], &[7]);
        bytes[1] = 1;

        let error = parse_idx(&bytes).unwrap_err();
        assert!(matches!(error, MnistError::InvalidIdx { .. }));
    }

    #[test]
    fn idx_parser_rejects_truncated_payload() {
        let bytes = idx_bytes(2, &[2, 3], &[1, 2, 3, 4, 5]);

        let error = parse_idx(&bytes).unwrap_err();
        assert!(matches!(error, MnistError::InvalidIdx { .. }));
    }

    #[test]
    fn pixel_normalization_maps_bytes_to_unit_interval() {
        assert_eq!(normalize_pixel(0), 0.0);
        assert_eq!(normalize_pixel(255), 1.0);
        assert!((normalize_pixel(128) - 128.0 / 255.0).abs() < f32::EPSILON);
    }

    #[test]
    fn mnist_parser_normalizes_samples_from_synthetic_idx_files() {
        use super::{Mnist, MnistSplit};
        use crate::data::Dataset;
        use std::fs;

        let dir = std::env::temp_dir().join(format!("rstorch-mnist-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let mut image_payload = vec![0_u8; 2 * 28 * 28];
        image_payload[0] = 255;
        image_payload[28 * 28] = 128;
        fs::write(
            dir.join("train-images-idx3-ubyte"),
            idx_bytes(3, &[2, 28, 28], &image_payload),
        )
        .unwrap();
        fs::write(
            dir.join("train-labels-idx1-ubyte"),
            idx_bytes(1, &[2], &[3, 7]),
        )
        .unwrap();

        let mnist = Mnist::from_dir_split(&dir, MnistSplit::Train).unwrap();
        assert_eq!(mnist.len(), 2);
        assert_eq!(mnist.labels(), &[3, 7]);

        let (first_image, first_label) = mnist.get(0).unwrap();
        assert_eq!(first_label, 3);
        assert_eq!(first_image[0][0], 1.0);
        assert_eq!(first_image[0][1], 0.0);

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    #[ignore = "downloads the MNIST dataset"]
    fn mnist_train_downloads_and_loads_expected_count() {
        use super::Mnist;
        use crate::data::Dataset;

        let mnist = Mnist::train().unwrap();
        assert_eq!(mnist.len(), 60_000);
        let (image, label) = mnist.get(0).unwrap();
        assert_eq!(image.len(), 28);
        assert_eq!(image[0].len(), 28);
        assert!(label < 10);
    }

    #[test]
    #[ignore = "downloads the MNIST dataset"]
    fn mnist_test_downloads_and_loads_expected_count() {
        use super::Mnist;
        use crate::data::Dataset;

        let mnist = Mnist::test().unwrap();
        assert_eq!(mnist.len(), 10_000);
        let (_, label) = mnist.get(0).unwrap();
        assert!(label < 10);
    }
}
