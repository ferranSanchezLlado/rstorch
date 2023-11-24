use rstorch::data::{DataLoader, SequentialSampler};
use rstorch::hub::MNIST;
use rstorch::loss::CrossEntropyLoss;
use rstorch::prelude::*;
use rstorch::{Identity, Linear, ReLU, Sequential, SGD};
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

fn main() {
    const BATCH_SIZE: usize = 32;
    const EPOCHS: usize = 10;

    let path: PathBuf = [".test_cache", "mnist"].iter().collect();

    let data = MNIST::new(path, false, true)
        .transform(|(x, y)| (x.mapv(f64::from) / 255.0, y))
        .transform(|(x, y)| (Array::from_iter(x), one_hot(y, 10)));
    let sampler = SequentialSampler::new(data.len());
    let mut data_loader = DataLoader::new(data, BATCH_SIZE, true, sampler);

    let mut model = sequential!(
        Identity(),
        Linear(784, 100),
        ReLU(),
        Linear(100, 100),
        ReLU(),
        Linear(100, 10),
    );
    let mut loss = CrossEntropyLoss::new();
    let mut optim = SGD::new(0.001);

    for _ in 0..EPOCHS {
        let n = data_loader.len() as f64;
        let mut total_loss = 0.0;
        let mut total_acc = 0.0;
        for (x, y) in data_loader.iter_array() {
            let pred = model.forward(x);
            let l = loss.forward(pred.clone(), y.clone());
            let acc = accuracy(pred, y);

            total_loss += l;
            total_acc += acc;

            model.backward(loss.backward());
            optim.step(&mut model);
        }

        let avg_loss = total_loss / n;
        let avg_acc = total_acc / n;
        println!("Avarage loss {avg_loss} - Avarage accuracy {avg_acc}");
    }
}
