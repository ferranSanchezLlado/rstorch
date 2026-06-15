# RsTorch

Implementation from scratch of a deep learning framework in Rust with a PyTorch-like API. The project is in an autograd restart and is not ready for production use. The API is not stable and may change at any time.

## Rust Toolchain

RsTorch currently targets nightly Rust for const-generic shape support. Crates
that use the const-generic tensor APIs should enable the same gates:

```rust
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
```

Use `cargo +nightly test` while the autograd restart is in progress. Compile-fail
tests are part of the normal test suite.

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

The current restart includes the CPU backend, owned tensor storage, const-generic
shape markers, `f32` as the default dtype, `f64` support, typed tensor aliases,
forward CPU operations, dynamic autograd, trainable parameters, a `Linear` layer,
and SGD.

Supported guarantees before GPU backends:

- Tensor shapes are compile-time types where possible.
- Elementwise operations require matching shape, dtype, and backend.
- Matrix multiplication dimensions are encoded in the method signature.
- `backward()` without an explicit seed is only available on scalar tensors.
- Non-scalar tensors use `backward_with(seed)`.
- CPU storage is safe owned `Vec` data.
- Optimizer steps run with graph construction disabled through `no_grad`.

The stable initial backend is CPU. Metal, CUDA, and WGPU are feature-gated
placeholders planned for later epochs.

## Safety

The safe tensor layer contains no `unsafe` code. GPU backends may require unsafe
or platform-specific code in later epochs; that code should stay isolated inside
backend modules with documented invariants.

## License

This project is licensed under the [MIT License](MIT-LICENSE) or [Apache License, Version 2.0](APACHE-LICENSE) at your option.
