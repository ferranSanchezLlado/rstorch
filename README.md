# RsTorch

Work-in-progress implementation from scratch of a deep learning framework in Rust with a PyTorch-like API. This repository is a place to explore tensor, autograd, neural-network, optimizer, and data-loading ideas. Nothing here should be treated as a final API or production-ready design.

## Rust Toolchain

RsTorch currently uses nightly Rust while the const-generic tensor API is being
explored. Code using those APIs may need the same gates:

```rust
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
```

Use `cargo +nightly test` while the project is in active development.
Compile-fail tests are part of the normal test suite.

## Development

Clone the repository and run the test suite with:

```text
cargo +nightly test
```

## Status

This is a moving snapshot of the current implementation. It currently includes
CPU tensor storage, const-generic shape markers, typed tensor aliases, forward
CPU operations, dynamic autograd, trainable parameters, seeded initialization,
basic neural-network layers, losses, optimizers, and early data-loading support.
Metal and CUDA experiments are feature-gated.

Current design directions being explored:

- Compile-time tensor shapes where they help ergonomics and safety.
- Explicit dtype and backend typing.
- Dynamic autograd with scalar and non-scalar backward paths.
- PyTorch-inspired modules, optimizers, datasets, samplers, and data loaders.
- Safe owned CPU storage as the baseline backend.

Several areas are still being designed, including higher-rank tensor ergonomics,
data APIs, serialization, checkpointing, schedulers, GPU backends, and model
composition.

## Safety

The goal is to keep the safe tensor layer free of `unsafe` code. Backend-specific
or platform-specific implementations may need stronger invariants as they evolve.

## License

This project is licensed under the [MIT License](MIT-LICENSE) or [Apache License, Version 2.0](APACHE-LICENSE) at your option.
