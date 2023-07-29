use flate2::bufread::GzDecoder;
use ndarray::{iter::AxisIter, prelude::*, Array, IntoDimension, RemoveAxis};
use reqwest::{self, blocking::Response};
use std::fs;
use std::io::{copy, BufReader, Read};
use std::path::{Path, PathBuf};

use crate::iterator::IteratorExt;
use crate::prelude::*;

#[allow(clippy::upper_case_acronyms)]
pub struct MNIST {
    root: PathBuf,
    data: Array3<u8>,
    labels: Array1<u8>,
    train: bool,
}

impl MNIST {
    const MIRRORS: [&str; 2] = [
        "http://yann.lecun.com/exdb/mnist/",
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
    ];

    const RESOURCES: [(&str, &str); 4] = [
        (
            "train-images-idx3-ubyte.gz",
            "f68b3c2dcbeaaa9fbdd348bbdeb94873",
        ),
        (
            "train-labels-idx1-ubyte.gz",
            "d53e105ee54ea40749a09fcbcd1e9432",
        ),
        (
            "t10k-images-idx3-ubyte.gz",
            "9fb629c4189551a2d022fa330f9573f3",
        ),
        (
            "t10k-labels-idx1-ubyte.gz",
            "ec29112dd5afa0611ce80d1b7f02629c",
        ),
    ];

    const TRAINING_FILE: &str = "training.pt";
    const TEST_FILE: &str = "test.pt";
    const CLASSES: [&str; 10] = [
        "0 - zero",
        "1 - one",
        "2 - two",
        "3 - three",
        "4 - four",
        "5 - five",
        "6 - six",
        "7 - seven",
        "8 - eight",
        "9 - nine",
    ];

    pub fn new<P: AsRef<Path>>(root: P, train: bool, download: bool) -> Self {
        if download {
            MNIST::download(&root);
        }

        assert!(
            MNIST::check_exits(&root),
            "Dataset not found. You can set download=true to download it"
        );

        let (data, labels) = MNIST::load_data(&root, train);
        let mut path = PathBuf::new();
        path.push(&root);
        MNIST {
            root: path,
            data,
            labels,
            train,
        }
    }

    fn check_exits<P: AsRef<Path>>(root: P) -> bool {
        match fs::read_dir(root) {
            Err(_) => false,
            Ok(mut folder) => folder.all_default(false, |f| match f {
                Err(_) => false,
                Ok(f) => Self::RESOURCES
                    .iter()
                    .any(|r| match f.file_name().to_str() {
                        None => false,
                        Some(f) => r.0.contains(f),
                    }),
            }),
        }
    }

    fn download<P: AsRef<Path>>(root: P) {
        // TODO: Download multiple files same time
        // TODO: Check md5
        // TODO: Improve exception handelning
        // TODO: FIND WHY FILENAME METHOD FAILS
        // TODO: Add better error handling
        if MNIST::check_exits(&root) {
            return;
        }

        match fs::create_dir_all(&root) {
            Ok(_) => {}
            Err(e) if matches!(e.kind(), std::io::ErrorKind::AlreadyExists) => {}
            Err(e) => panic!("Failed to create root folder for dataset: {}", e),
        }

        for (filename, _md5) in Self::RESOURCES {
            let mut data: Option<Response> = None;
            for mirror in Self::MIRRORS {
                let url = format!("{}{}", mirror, filename);
                println!("Downloading {url}");

                data = reqwest::blocking::get(url).ok();
                data = match data.is_some() && data.as_ref().unwrap().status().is_success() {
                    true => break,
                    false => None,
                }
            }

            let gz: Vec<_> = match data {
                None => panic!("Failed to download the file ({filename}) for any of the mirrors"),
                Some(data) => data.bytes().unwrap().into_iter().collect(),
            };

            // Decode file
            let mut gz = GzDecoder::new(gz.as_slice());

            // Find file_name
            // let filename = match gz
            //     .header()
            //     .and_then(|h| h.filename())
            //     .and_then(|f| String::from_utf8(f.to_owned()).ok())
            // {
            //     Some(f) => f,
            //     _ => filename
            //         .split_once('.')
            //         .map(|f| f.0)
            //         .unwrap_or(filename)
            //         .to_owned(),
            // };
            let filename = filename
                .split_once('.')
                .map(|f| f.0)
                .unwrap_or(filename)
                .to_owned();
            let mut path = PathBuf::new();
            path.push(&root);
            path.push(filename);

            // Store raw file
            let mut file = fs::File::create(path).unwrap();
            copy(&mut gz, &mut file).unwrap();
        }
    }

    fn load_data<P: AsRef<Path>>(root: P, train: bool) -> (Array3<u8>, Array1<u8>) {
        let mut path = PathBuf::new();
        path.push(root);

        let start = if train { "train" } else { "t10k" };
        path.push(format!("{start}-images-idx3-ubyte"));
        let data = read_image_file(&path);

        path.pop();
        path.push(format!("{start}-labels-idx1-ubyte"));
        let labels = read_label_file(&path);

        assert_eq!(data.len_of(Axis(0)), labels.len_of(Axis(0)));
        (data, labels)
    }
}

impl Dataset for MNIST {
    type Item = (Array2<u8>, Array0<u8>);

    fn get(&self, index: usize) -> Option<Self::Item> {
        if index > self.len() {
            return None;
        }

        Some((
            self.data.index_axis(Axis(0), index).to_owned(),
            self.labels.index_axis(Axis(0), index).to_owned(),
        ))
    }

    fn len(&self) -> usize {
        self.labels.len()
    }
}

pub struct CombinedIter<'a, T, D, T2, D2> {
    data: AxisIter<'a, T, D>,
    labels: AxisIter<'a, T2, D2>,
}

impl<'a, T, D: Dimension, T2, D2: Dimension> CombinedIter<'a, T, D, T2, D2> {
    pub(crate) fn new<Di, Di2>(data: &'a Array<T, Di>, labels: &'a Array<T2, Di2>) -> Self
    where
        Di: RemoveAxis<Smaller = D>,
        Di2: RemoveAxis<Smaller = D2>,
    {
        Self {
            data: data.outer_iter(),
            labels: labels.outer_iter(),
        }
    }
}

impl<'a, T: Clone, D: Dimension, T2: Clone, D2: Dimension> Iterator
    for CombinedIter<'a, T, D, T2, D2>
{
    type Item = (Array<T, D>, Array<T2, D2>);

    fn next(&mut self) -> Option<Self::Item> {
        Some((self.data.next()?.to_owned(), self.labels.next()?.to_owned()))
    }
}

impl<'a> IterableDataset<'a> for MNIST {
    type Iterator = CombinedIter<'a, u8, Ix2, u8, Ix0>;

    fn iter(&'a self) -> Self::Iterator {
        CombinedIter::new(&self.data, &self.labels)
    }
}

trait MagicType<const N: usize> {
    const MAGIC: u32;

    fn from_be_bytes(data: [u8; N]) -> Self;
}

macro_rules! magic {
    ($t:ty, $m:literal, $b:literal) => {
        impl MagicType<$b> for $t {
            const MAGIC: u32 = $m;

            fn from_be_bytes(data: [u8; $b]) -> Self {
                <$t>::from_be_bytes(data)
            }
        }
    };
}

magic!(u8, 8, 1);
magic!(i8, 9, 1);
magic!(i16, 11, 2);
magic!(i32, 12, 4);
magic!(f32, 13, 4);
magic!(f64, 14, 8);

fn read_int<R: Read>(mut data: R) -> u32 {
    let mut buff = [0; 4];
    data.read_exact(&mut buff).unwrap();
    u32::from_be_bytes(buff)
}

fn read_sn3<T: MagicType<N_BYTES>, const N_BYTES: usize, const N_DIM: usize>(
    path: &Path,
) -> Array<T, Dim<[usize; N_DIM]>>
where
    [usize; N_DIM]: IntoDimension<Dim = Dim<[usize; N_DIM]>>,
{
    let mut data = BufReader::new(fs::File::open(path).unwrap());

    let magic = read_int(&mut data);
    let nd = magic % 256;
    let ty = magic / 256;
    assert_eq!(nd, N_DIM as u32);
    assert_eq!(ty, T::MAGIC);

    let mut shape = [0; N_DIM];
    shape
        .iter_mut()
        .for_each(|d| *d = read_int(&mut data) as usize);

    let mut byte_array = Vec::new();
    data.read_to_end(&mut byte_array).unwrap();

    let parsed: Array1<T> = byte_array
        .into_iter()
        .array_chunks_costum()
        .map(T::from_be_bytes)
        .collect();

    parsed.into_shape(shape).unwrap()
}

fn read_label_file(path: &Path) -> Array1<u8> {
    read_sn3(path)
}

fn read_image_file(path: &Path) -> Array3<u8> {
    read_sn3(path)
}

#[cfg(test)]
mod test {
    use super::*;
    use fs_extra::dir::{copy, remove, CopyOptions};
    use std::cmp::{max, min};
    use std::sync::Mutex;

    static TESTS_EXECUTED: Mutex<u8> = Mutex::new(0);
    const TOTAL_TESTS: u8 = 4;

    #[inline]
    fn data_path(n: u8) -> PathBuf {
        PathBuf::from(format!(".data_{n}"))
    }
    #[inline]
    fn tmp_path() -> PathBuf {
        PathBuf::from(".data_tmp")
    }

    #[inline]
    fn prepare(loaded_data: bool) -> u8 {
        let mut lock = TESTS_EXECUTED.lock().unwrap();
        let n_test = lock.to_owned();

        if n_test == 0 {
            MNIST::download(tmp_path());
        }
        *lock += 1;
        // Moves data from
        if loaded_data {
            fs::create_dir_all(data_path(n_test)).unwrap();

            let options = CopyOptions::new().content_only(true).overwrite(true);
            copy(tmp_path(), data_path(n_test), &options).unwrap();
        }
        n_test
    }

    #[inline]
    fn finish(n: u8) {
        remove(data_path(n)).unwrap();

        if TESTS_EXECUTED.lock().unwrap().eq(&TOTAL_TESTS) {
            remove(tmp_path()).unwrap();
        }
    }

    #[test]
    fn download() {
        let n = prepare(false);

        let path = data_path(n);
        MNIST::download(&path);

        let iter = fs::read_dir(path).unwrap();
        assert_eq!(4, iter.count());

        finish(n);
    }

    #[test]
    fn check_exists() {
        let n = prepare(true);

        let path = data_path(n);
        assert!(MNIST::check_exits(&path));
        finish(n);

        assert!(!MNIST::check_exits(&path));
    }

    #[test]
    fn load_data() {
        let n = prepare(true);

        let path = data_path(n);
        let (data, labels) = MNIST::load_data(&path, false);
        // Data inputs
        assert_eq!(10_000, data.len_of(Axis(0)));
        assert_eq!(10_000, labels.len());

        // Image dimensions
        assert_eq!(28, data.len_of(Axis(1)));
        assert_eq!(28, data.len_of(Axis(2)));

        let (data, labels) = MNIST::load_data(&path, true);
        // Data inputs
        assert_eq!(60_000, data.len_of(Axis(0)));
        assert_eq!(60_000, labels.len());

        // Image dimensions
        assert_eq!(28, data.len_of(Axis(1)));
        assert_eq!(28, data.len_of(Axis(2)));

        finish(n);
    }

    #[test]
    fn new() {
        let n = prepare(true);

        let path = data_path(n);
        let data = MNIST::new(path, false, false);

        assert_eq!(10_000, data.len());
        let (sample, _) = data.get(5_001).unwrap();
        assert_eq!(28, sample.len_of(Axis(0)));
        assert_eq!(28, sample.len_of(Axis(1)));

        let (max_sample, max_label) =
            data.iter()
                .fold((u8::MIN, u8::MIN), |(max_s, max_l), (s, l)| {
                    (
                        max(max_s, s.into_iter().max().unwrap()),
                        max(max_l, l.into_scalar()),
                    )
                });

        let (min_sample, min_label) =
            data.iter()
                .fold((u8::MAX, u8::MAX), |(max_s, max_l), (s, l)| {
                    (
                        min(max_s, s.into_iter().min().unwrap()),
                        min(max_l, l.into_scalar()),
                    )
                });

        assert_eq!(255, max_sample);
        assert_eq!(0, min_sample);

        assert_eq!(9, max_label);
        assert_eq!(0, min_label);

        finish(n);
    }
}
