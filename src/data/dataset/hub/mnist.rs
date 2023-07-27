use flate2::read::GzDecoder;
use reqwest;
use reqwest::blocking::Response;
use std::fs;
use std::io::copy;
use std::path::{Path, PathBuf};

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

#[allow(clippy::upper_case_acronyms)]
pub struct MNIST {
    root: PathBuf,
    data: Vec<String>,
    targets: Vec<String>,
    train: bool,
}

impl MNIST {
    pub fn new(mut root: PathBuf, train: bool, download: bool) -> Self {
        root.push("mnist");
        root.push("raw");

        if download {
            MNIST::download(&root);
        }

        if MNIST::check_exits(&root) {
            panic!("Dataset not found. You can use download=True to download it");
        }

        let (data, targets) = MNIST::load_data(&root);
        MNIST {
            root,
            data,
            targets,
            train,
        }
    }

    fn download(root: &Path) {
        // TODO: Download multiple files same time
        // TODO: Check md5
        // TODO: Improve exception handelning
        if MNIST::check_exits(root) {
            return;
        }

        match fs::create_dir_all(root) {
            Ok(_) => {}
            Err(e) if matches!(e.kind(), std::io::ErrorKind::AlreadyExists) => {}
            Err(e) => panic!("Failed to create root folder for dataset: {}", e),
        }

        for (filename, _md5) in RESOURCES {
            let mut data: Option<Response> = None;
            for mirror in MIRRORS {
                let url = format!("{}{}", mirror, filename);
                println!("Downloading {url}");

                data = reqwest::blocking::get(url).ok();
                data = match data.is_some() && data.as_ref().unwrap().status().is_success() {
                    true => break,
                    false => None,
                }
            }

            let zip: Vec<_> = match data {
                None => panic!("Failed to download the file ({filename}) for any of the mirrors"),
                Some(data) => data.bytes().unwrap().into_iter().collect(),
            };
            // Decode file
            let mut data = GzDecoder::new(zip.as_slice());

            // Decide path to stre raw
            let mut path = PathBuf::from(root);
            path.push(filename);
            path.set_extension("");

            // Store raw file
            let mut file = fs::File::create(path).unwrap();
            copy(&mut data, &mut file).unwrap();
        }
    }

    fn check_exits(root: &Path) -> bool {
        match fs::read_dir(root) {
            Ok(folder) => folder.count() == 4,
            Err(_) => false,
        }
    }

    fn load_data(_root: &Path) -> (Vec<String>, Vec<String>) {
        todo!()
    }
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

        fs::remove_dir_all(folder).unwrap();
    }
}
