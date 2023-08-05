#![cfg(feature = "dataset_hub")]
use rstorch::hub::MNIST;
use rstorch::prelude::*;
use rstorch::{Identity, Linear, ReLU, Sequential, Softmax};
use std::path::PathBuf;

#[test]
fn test_macro() {
    let path: PathBuf = [".test_cache", "mnist"].iter().collect();

    let _data = MNIST::new(path, false, true)
        .transform(|d| (d.0.mapv(f64::from) / 255.0, d.1.mapv(f64::from) / 9.0));

    let _module = sequential!(
        Identity(),
        Linear(3, 10),
        ReLU(),
        Linear(10, 100),
        ReLU(),
        Linear(100, 2),
        Softmax()
    );
}
