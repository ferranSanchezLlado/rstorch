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
shape markers, `f32` as the default dtype, `f64` support on CPU, typed tensor
aliases, forward CPU operations, dynamic autograd, trainable parameters, seeded
project-owned random initialization, a `Linear` layer, common losses,
activations, SGD, SGD with momentum, and Adam. Metal and CUDA are feature-gated
GPU backends for `f32` tensors.

Supported guarantees before GPU backends:

- Tensor shapes are compile-time types where possible.
- Elementwise operations require matching shape, dtype, and backend.
- Matrix multiplication dimensions are encoded in the method signature.
- `backward()` without an explicit seed is only available on scalar tensors.
- Non-scalar tensors use `backward_with(seed)`.
- CPU storage is safe owned `Vec` data.
- Optimizer steps run with graph construction disabled through `no_grad`.

## Training Example

Small models can be trained today with explicit typed layers and manual parameter
collection. The API is still unstable, but the current workflow supports seeded
initialization, scalar losses, activations, and stateful optimizers:

```rust
use rstorch::prelude::*;

struct Mlp {
    hidden: Linear<2, 8>,
    output: Linear<8, 1>,
}

impl Mlp {
    fn new(rng: &mut SmallRng) -> Self {
        Self {
            hidden: Linear::kaiming_uniform(rng),
            output: Linear::xavier_uniform(rng),
        }
    }

    fn forward<const BATCH: usize>(&self, x: &Tensor2D<BATCH, 2>) -> Tensor2D<BATCH, 1> {
        self.output.forward(&self.hidden.forward(x).tanh()).sigmoid()
    }

    fn parameters_mut(&mut self) -> Vec<&mut dyn OptimParameter<f32, Cpu>> {
        let mut parameters = self.hidden.parameters_mut();
        parameters.extend(self.output.parameters_mut());
        parameters
    }

    fn zero_grad(&mut self) {
        self.hidden.zero_grad();
        self.output.zero_grad();
    }
}

let mut rng = SmallRng::seed_from_u64(42);
let mut model = Mlp::new(&mut rng);
let input = Tensor2D::<4, 2>::from_array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
let target = Tensor2D::<4, 1>::from_array([[0.0], [1.0], [1.0], [0.0]]);
let mut optimizer = Adam::new(0.05);

for _ in 0..1000 {
    model.zero_grad();
    let prediction = model.forward(&input);
    let loss = binary_cross_entropy(&prediction, &target);
    loss.backward();
    optimizer.step(model.parameters_mut());
}
```

Runnable CPU-only programs are available with:

```text
cargo +nightly run --example tiny_regression
cargo +nightly run --example xor_mlp
```

Current model-building limitations are intentional: there is no dataset loader,
serialization, checkpointing, typed `Sequential`, gradient clipping, scheduler,
or higher-rank tensor API yet. Compose small models manually and collect
parameters explicitly.

The stable initial backend is CPU. Metal is available on macOS with
`--features metal`. CUDA is available on Linux and Windows with
`--features cuda` and uses runtime-compiled kernels through `cudarc`; it
requires a CUDA driver/NVRTC at runtime. WGPU is still a feature-gated
placeholder planned for a later epoch.

## Safety

The safe tensor layer contains no `unsafe` code. GPU backends may require unsafe
or platform-specific code in later epochs; that code should stay isolated inside
backend modules with documented invariants.

## License

This project is licensed under the [MIT License](MIT-LICENSE) or [Apache License, Version 2.0](APACHE-LICENSE) at your option.
