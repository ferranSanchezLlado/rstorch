#![cfg(feature = "dataset_hub")]
use rand::thread_rng;
use rstorch::data::{DataLoader, RandomSampler};
use rstorch::hub::MNIST;
use rstorch::loss::CrossEntropyLoss;
use rstorch::prelude::*;
use rstorch::{Identity, Linear, ReLU, Sequential, Softmax, SGD};
use std::path::PathBuf;

fn one_hot(x: Array0<u8>, n: usize) -> Array1<f64> {
    let mut encoded = Array1::zeros(n);
    encoded[x.into_scalar() as usize] = 1.0;
    encoded
}

fn argmax(arr: Array2<f64>, axis: Axis) -> Option<Array1<usize>> {
    arr.axis_iter(axis)
        .map(|v| {
            v.into_iter()
                .enumerate()
                .reduce(|acc, e| if e.1 > acc.1 { e } else { acc })
                .map(|v| v.0)
        })
        .collect()
}

fn accuracy(pred: Array2<f64>, truth: Array2<f64>) -> f64 {
    let pred = argmax(pred, Axis(0)).unwrap();
    let truth = argmax(truth, Axis(0)).unwrap();

    let n = pred.len() as f64;
    pred.into_iter()
        .zip(truth)
        .map(|(a, b)| usize::from(a == b))
        .sum::<usize>() as f64
        / n
}

#[test]
fn basic_test() {
    const BATCH_SIZE: usize = 32;
    const EPOCHS: usize = 5;

    let path: PathBuf = [".test_cache", "mnist"].iter().collect();

    let data = MNIST::new(path, false, true)
        .transform(|(x, y)| (x.mapv(f64::from) / 255.0, y))
        .transform(|(x, y)| (Array::from_iter(x), one_hot(y, 10)));
    let sampler = RandomSampler::new(data.len(), false, BATCH_SIZE, thread_rng());
    let mut data_loader = DataLoader::new(data, BATCH_SIZE, true, sampler);

    let mut model = sequential!(
        Identity(),
        Linear(784, 100),
        ReLU(),
        Linear(100, 100),
        ReLU(),
        Linear(100, 10),
        Softmax()
    );
    let mut loss = CrossEntropyLoss::new();

    let mut optim = SGD::new(0.1);

    for _ in 0..EPOCHS {
        // TODO
        for (x, y) in data_loader.iter_array() {
            let pred = model.forward(x);
            let l = loss.forward(pred.clone(), y.clone());
            let acc = accuracy(pred, y);

            println!("Loss {l}, Accuracy {acc}");

            model.backward(loss.backward());
            optim.step(&mut model);
        }
    }
}
