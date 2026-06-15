# RsTorch

Implementation from scratch of a deep learning framework in Rust with a PyTorch-like API. The project is in an autograd restart and is not ready for production use. The API is not stable and may change at any time.

## Rust Toolchain

RsTorch currently targets nightly Rust for const-generic shape support:

```rust
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
```

Use `cargo +nightly check` while the autograd restart is in progress.

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

## Status

The current restart foundation defines the module layout, feature flags, shape markers, floating-point trait, backend placeholders, and tensor aliases. Tensor constructors, operations, autograd, neural network layers, and optimizers are planned for later epochs.

## License

This project is licensed under the [MIT License](MIT-LICENSE) or [Apache License, Version 2.0](APACHE-LICENSE) at your option.
