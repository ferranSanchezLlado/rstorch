# RsTorch

Implementation from scratch of a deep learning framework in Rust with a PyTorch-like API. The project is still in its early stages and is not ready for production use. Therefore, the API is not stable and may change at any time.

Currently, the project achieved the Minimum Viable Product allow the user to train a sequential model. Furthermore, it also provides the MNIST dataset that will download automatically from the internet.

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
rstorch = "0.2.0"
```

Or if you want to use the latest version from the master branch:

```toml
[dependencies]
rstorch = { git = "https://github.com/ferranSanchezLlado/rstorch.git" }
```

## Usage

Small example on how to use the library to train a model with the MNIST dataset:

```rust
use rstorch::data::{DataLoader, SequentialSampler};
use rstorch::hub::MNIST;
use rstorch::prelude::*;
use rstorch::utils::{accuracy, flatten, normalize_zero_one, one_hot};
use rstorch::{CrossEntropyLoss, Identity, Linear, ReLU, Sequential, SGD};
use std::fs;
use std::path::PathBuf;

const BATCH_SIZE: usize = 32;
const EPOCHS: usize = 5;

fn main() {
    // Path that gets deleted by tests
    let path: PathBuf = ["data", "mnist"].iter().collect();

    let train_data = MNIST::new(path, true, true)
        .transform(|(x, y)| (flatten(normalize_zero_one(x)), one_hot(y, 10)));
    let sampler = SequentialSampler::new(train_data.len());
    let mut data_loader = DataLoader::new(train_data, BATCH_SIZE, true, sampler);

    let mut model = sequential!(
        Identity(),
        Linear(784, 100),
        ReLU(),
        Linear(100, 100),
        ReLU(),
        Linear(100, 10),
    );
    let mut loss = CrossEntropyLoss::new();
    let mut optim = SGD::new(0.01);

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
```

## License

This project is licensed under the [MIT License](MIT-LICENSE) or [Apache License, Version 2.0](APACHE-LICENSE) at your option.
