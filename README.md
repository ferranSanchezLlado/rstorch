# RsTorch

[![crates.io](https://img.shields.io/crates/v/rstorch.svg)](https://crates.io/crates/rstorch)
[![docs.rs](https://img.shields.io/docsrs/rstorch)](https://docs.rs/rstorch)
[![license](https://img.shields.io/crates/l/rstorch.svg)](#license)

A PyTorch-inspired deep-learning library for Rust with one dynamic `Tensor`
type, reverse-mode autograd, and linear gradients.

`1.0.0` is still being prepared. To try this tree before it is published:

```toml
[dependencies]
rstorch = { git = "https://github.com/ferranSanchezLlado/rstorch.git" }
```

```rust
use rstorch::prelude::*;

#[derive(Module)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Forward for Mlp {
    type Output = Tensor;

    fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        let hidden = self.fc1.forward(x, mode)?.relu()?;
        self.fc2.forward(&hidden, mode)
    }
}

fn train_step(model: &mut Mlp, optimizer: &mut Adam, x: &Tensor, y: &Tensor) -> Result<()> {
    let loss = model.forward(x, Mode::TRAIN)?.cross_entropy(y)?;
    optimizer.step(model, loss.backward()?)?;
    Ok(())
}
```

There is no `zero_grad`: gradients are returned by `backward` instead of being
stored in parameters. The complete loop is in [`examples/mlp.rs`](examples/mlp.rs).

## The short version

- `Tensor` has no type parameters. Shape, dtype, and device are runtime values.
- `backward` returns a `Grads` value. It is not `Clone`, and the built-in
  optimizer `step` methods consume it, so one gradient result cannot be
  accidentally stepped twice.
- Fallible tensor operations have named `Result`-returning methods. Operator
  forms are available when a panic on invalid input is acceptable.
- Dtypes are explicit. Missing backend kernels return an error instead of
  silently promoting or copying through the host.
- A layer that needs extra per-call data can implement `Forward<Input>` with a
  small input struct. `MultiHeadAttention` uses this for its optional mask.

## What is included

| Area | Contents |
|---|---|
| Tensors | Elementwise operations, broadcasting, matmul, reductions, indexing, convolution, pooling, softmax, and losses |
| Autograd | Reverse-mode gradients, input gradients, gradient norms, clipping, scaling, and merging |
| `nn` | Linear, convolution/pooling, activations, dropout, embedding, attention, normalization, sequential modules, and `#[derive(Module)]` |
| `optim` | `Sgd`, `Adam`, `AdamW`, parameter groups, and learning-rate schedules |
| `data` | `Dataset`, `DataLoader`, in-memory datasets, MNIST, and Tiny Shakespeare |
| `text` | `CharTokenizer` and `BpeTokenizer` |
| `models` | `DecoderTransformer` and its generation cache |
| `persist` | Safetensors state and versioned checkpoint envelopes |

Model checkpoints contain model state. The transformer convenience checkpoint
contains its config and model tensors, not optimizer, RNG, cache, tokenizer, or
application state. Model and optimizer loads can be restored independently.
Checkpoint files have no checksum or authentication; add an integrity check
when loading files from an untrusted or transported source.

## Examples

```sh
cargo run --example tensors
cargo run --example mlp
cargo run --example custom_optimizer
cargo run --example typed --features typed
cargo run --release --example mnist --features hub
cargo run --release --example tiny_shakespeare --features hub
cargo run --release --example typed_mnist --features typed,hub
```

The examples that download data need network access. The hub cache uses
`RSTORCH_DATA`, then `$HOME/.cache/rstorch`, then `./data`.

## Feature flags

| Feature | Default | What it does |
|---|---|---|
| `typed` | off | Experimental compile-time checked rank, dimensions, dtype, and placement over the same `Tensor` |
| `rayon` | off | Parallel CPU kernels |
| `hub` | off | Dataset downloads |
| `metal` | on | Metal on macOS |
| `cuda` | off | Native NVIDIA CUDA on Linux and Windows, including WSL; F16/F32 compute and lossless I64/Bool storage on supported GPUs |
| `wgpu` | off | Portable WebGPU; F32 compute, optional native F16, and lossless I64/Bool storage |
| `testing` | off | The finite-difference gradient helper used by the test suite |
| `bench-resnet` | off | Enables the long `resnet_mnist` benchmark |

Dense elementwise chains can also use the experimental deferred executor:

```rust
let _restore = rstorch::lazy::set_fusion(true);
let y = x.add_scalar(1.0)?.mul_scalar(2.0)?;
y.realize()?;
```

Fusion is off by default and may change independently of the dynamic API.

## Devices and dtypes

CPU is the reference backend. Metal supports F16/F32 arithmetic and
F16/F32/I64/Bool storage; BF16 and F64 are unsupported. CUDA supports F16/F32
compute and lossless I64/Bool storage on compute-capability 6.0 or newer. WGPU
supports F32 compute, optional native F16 when the adapter advertises
`SHADER_F16`, and lossless I64/Bool storage. Individual operations can support
fewer dtype combinations and return `Error::Unsupported` for the rest.

`Device::best_available()` tries Metal, CUDA, the best eligible WGPU adapter,
and CPU, in that order. It skips a backend that cannot initialize. WGPU
ordinals are process-local adapter indices, not stable device identities; do
not persist them. CUDA is built in CI but its kernels are validated locally
because CI has no NVIDIA hardware.

On WSL, the CUDA feature needs a compatible NVIDIA Windows driver. The bundled
PTX means the CUDA toolkit is not needed at runtime, and a Linux display driver
should not be installed inside WSL.

Feature-enabled accelerator tests require the corresponding hardware. On a
machine that intentionally has no accelerator, skip those tests explicitly:

```sh
RSTORCH_SKIP_METAL_TESTS=1 RSTORCH_SKIP_CUDA_TESTS=1 \
RSTORCH_SKIP_WGPU_TESTS=1 cargo test --all-features
```

Do not set these variables on a hardware test lane.

## Compile-time shapes

The optional `typed` API wraps `Tensor` and adds compile-time rank,
dimension, dtype, and placement checks. Use `DYN` for dimensions known only at
runtime:

```rust,ignore
let ctx = DeviceCtx::cpu()?;
let x = Tensor2::<2, 3>::from_vec(vec![1.0f32; 6], [2, 3], &ctx)?;
let w = Tensor2::<3, 4>::from_vec(vec![0.1f32; 12], [3, 4], &ctx)?;
let y: Tensor2<2, 4> = x.matmul(&w)?;
```

## Status and license

The current release target is `1.0.0`; the minimum supported Rust version is
**1.88** and the crate uses edition 2024. See [`CHANGELOG.md`](CHANGELOG.md)
for the history of the rewrite from 0.x.

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or the
[MIT license](LICENSE-MIT), at your option.
