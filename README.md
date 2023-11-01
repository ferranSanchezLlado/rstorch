# RsTorch

Implementation from scratch of a deep learning framework in Rust with a PyTorch-like API. The project is still in its early stages and is not ready for production use. Therefore, the API is not stable and may change at any time.

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
rstorch = "0.1.0"
```

Or if you want to use the latest version from the master branch:

```toml
[dependencies]
rstorch = { git = "https://github.com/ferranSanchezLlado/rstorch.git" }
```

## Usage

Still in development.

Small Example on how to create a sequential model and load the MNIST dataset automatically:

```rust
use rstorch::prelude::*;

let path: PathBuf = [".test_cache", "mnist"].iter().collect();

// Download the dataset if it doesn't exist
let data = MNIST::new(path, false, true)
    .transform(|d| (d.0.mapv(f64::from) / 255.0, d.1.mapv(f64::from) / 9.0));
// Transfroms the labels and images to f64 and normalizes the images to [0, 1]

// Create the model
let model = sequential!(
    Identity(),
    Linear(28 * 28, 100), // Input size is 28 * 28 and output size is 100
    ReLU(),
    Linear(100, 100), // Input size is 100 and output size is 100
    ReLU(),
    Linear(100, 10), // Input size is 100 and output size is 10
    Softmax()
);

## License

This project is licensed under the [MIT License](MIT-LICENSE) or [Apache License, Version 2.0](APACHE-LICENSE) at your option.
