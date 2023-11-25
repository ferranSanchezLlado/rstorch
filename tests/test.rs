#![cfg(feature = "dataset_hub")]
use rstorch::data::{DataLoader, SequentialSampler};
use rstorch::hub::MNIST;
use rstorch::loss::CrossEntropyLoss;
use rstorch::prelude::*;
use rstorch::utils::{accuracy, one_hot};
use rstorch::{Identity, Linear, ReLU, Sequential, SGD};
use std::path::PathBuf;

#[test]
fn training() {
    const BATCH_SIZE: usize = 32;
    const EPOCHS: usize = 5;

    let path: PathBuf = [".test_cache", "training"].iter().collect();

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
    // Set lr to 1.0 to check for insatibilitites on the gradients
    let mut optim = SGD::new(1.0);

    for i in 0..EPOCHS {
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
        println!("EPOCH {i}: Avarage loss {avg_loss} - Avarage accuracy {avg_acc}");
    }
}
