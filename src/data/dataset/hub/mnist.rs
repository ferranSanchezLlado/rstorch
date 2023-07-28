use flate2::bufread::GzDecoder;
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::Array;
use ndarray::IntoDimension;
use reqwest;
use reqwest::blocking::Response;
use std::fs;
use std::io::{copy, BufReader, Read};
use std::path::{Path, PathBuf};

use crate::iterator::IteratorExt;

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

    pub fn new(mut root: PathBuf, train: bool, download: bool) -> Self {
        root.push("mnist");
        root.push("raw");

        if download {
            MNIST::download(&root);
        }

        assert!(
            !MNIST::check_exits(&root),
            "Dataset not found. You can set download=true to download it"
        );

        let (data, labels) = MNIST::load_data(&root, train);
        MNIST {
            root,
            data,
            labels,
            train,
        }
    }

    fn download(root: &Path) {
        // TODO: Download multiple files same time
        // TODO: Check md5
        // TODO: Improve exception handelning
        // TODO: FIND WHY FILENAME METHOD FAILS
        // TODO: Add better error handling
        if MNIST::check_exits(root) {
            return;
        }

        match fs::create_dir_all(root) {
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
            let mut path = PathBuf::from(root);
            path.push(filename);

            // Store raw file
            let mut file = fs::File::create(path).unwrap();
            copy(&mut gz, &mut file).unwrap();
        }
    }

    fn check_exits(root: &Path) -> bool {
        match fs::read_dir(root) {
            Err(_) => false,
            Ok(mut folder) => folder.all_default(false, |f| match f {
                Err(_) => false,
                Ok(f) => {
                    Self::RESOURCES
                        .iter()
                        .map(|r| r.1)
                        .any(|r| match f.file_name().to_str() {
                            None => false,
                            Some(f) => r.contains(f),
                        })
                }
            }),
        }
    }

    fn load_data(root: &Path, train: bool) -> (Array3<u8>, Array1<u8>) {
        let start = if train { "train" } else { "t10k" };

        let mut path = PathBuf::from(root);
        path.push(format!("{start}-images-idx3-ubyte"));
        let data = read_image_file(&path);

        path.pop();
        path.push(format!("{start}-labels-idx1-ubyte"));
        let labels = read_label_file(&path);

        (data, labels)
    }
}

trait MagicType {
    const MAGIC: u32;
    const BYTES: usize;

    fn from_magic<I: Iterator<Item = u8>>(iter: I) -> Self;
}

macro_rules! magic {
    ($t:ty, $m:literal, $b:literal) => {
        impl MagicType for $t {
            const MAGIC: u32 = $m;
            const BYTES: usize = $b;

            fn from_magic<I: Iterator<Item = u8>>(iter: I) -> Self {
                let data: Vec<_> = iter.collect();
                <$t>::from_be_bytes(data.try_into().unwrap())
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

fn read_int<R: Read>(data: &mut R) -> u32 {
    let mut buff = [0; 4];
    data.read_exact(&mut buff).unwrap();
    u32::from_be_bytes(buff)
}

fn read_sn3<T: MagicType, const N: usize>(path: &Path) -> Array<T, Dim<[usize; N]>>
where
    [usize; N]: IntoDimension<Dim = Dim<[usize; N]>>,
{
    let mut data = BufReader::new(fs::File::open(path).unwrap());

    let magic = read_int(&mut data);
    let nd = magic % 256;
    let ty = magic / 256;
    assert_eq!(nd, N as u32);
    assert_eq!(ty, T::MAGIC);

    let mut shape = [0; N];
    shape
        .iter_mut()
        .for_each(|d| *d = read_int(&mut data) as usize);

    let mut byte_array = Vec::new();
    data.read_to_end(&mut byte_array).unwrap();

    let parsed: Array1<T> = byte_array
        .into_iter()
        .chunks(T::BYTES)
        .into_iter()
        .map(T::from_magic)
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

    #[test]
    fn download() {
        let folder = Path::new("download_dataset");
        MNIST::download(folder.clone());

        let iter = fs::read_dir(folder.clone()).unwrap();
        assert_eq!(4, iter.count());

        MNIST::load_data(folder, false);

        fs::remove_dir_all(folder).unwrap();
    }
}
