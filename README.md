# RsTorch

[![crates.io](https://img.shields.io/crates/v/rstorch.svg)](https://crates.io/crates/rstorch)
[![docs.rs](https://img.shields.io/docsrs/rstorch)](https://docs.rs/rstorch)
[![license](https://img.shields.io/crates/l/rstorch.svg)](#license)

A safer PyTorch-inspired deep learning library for Rust: **one concrete tensor
type with zero generic parameters, and linear gradients.**

It spends Rust's type system on *state* — ownership and linearity — rather than
on shapes, so the everyday API stays as short as PyTorch's while the mistakes
that cost you a training run become compile errors or loud failures.

```toml
[dependencies]
rstorch = "1.0"
```

```rust
use rstorch::prelude::*;

#[derive(Module)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
}

impl Forward for Mlp {
    fn forward(&mut self, x: &Tensor, mode: Mode) -> Result<Tensor> {
        let hidden = self.fc1.forward(x, mode)?.relu()?;
        self.fc2.forward(&hidden, mode)
    }
}

// ...and that is the whole of a training step:
fn train_step(model: &mut Mlp, optimizer: &mut Adam, x: &Tensor, y: &Tensor) -> Result<()> {
    let loss = model.forward(x, Mode::TRAIN)?.cross_entropy(y)?;
    optimizer.step(model, loss.backward()?)?;
    Ok(())
}
```

There is no `zero_grad`. Gradients never live in the parameters, so there is
nothing to clear. The full loop is [`examples/mlp.rs`](examples/mlp.rs).

## What "safer" means here

**One tensor type.** `Tensor` has no generic parameters. Rank, dtype and device
are values it carries, so a 2×2 `f32` and a 3-D `f64` are the same Rust type
and no signature in your code has to be generic over them.

**Linear gradients.** `backward` returns one `Grads`. It is not `Clone`, it is
`#[must_use]`, and an optimizer step consumes it. Stepping twice on the same
gradients is a compile error. Ignoring them emits the normal `#[must_use]`
warning, while the optimizer's missing-gradient checks catch parameters that
were never reached. There is no `zero_grad` because gradients never live in
the parameters.

**Loud, never silent.** Every failure names the operation and the values that
were wrong:

```text
matmul: shape mismatch: lhs [2, 2] vs rhs [3, 3]
add: dtype mismatch: expected f32, got i64 (no implicit promotion; cast explicitly with to_dtype)
```

There is no implicit dtype promotion, no silent host round-trip when a kernel
is missing, and a parameter that never got a gradient fails at the next step
instead of quietly never training.

**Two tiers of fallibility.** Every named method returns `Result`. Operator
sugar (`+`, `-`, `*`, `/`) panics with the identical message under
`#[track_caller]`, so exploratory code stays terse and library code stays
total.

## What is in the box

| | |
|---|---|
| **Tensors** | elementwise, broadcasting, matmul, reductions, indexing/gather, `conv2d`, `max_pool2d`, softmax and losses over six runtime dtypes, with operation-specific support and loud `Unsupported` errors |
| **Autograd** | reverse-mode over the whole op set, `Grads::clip_norm`/`scale`/`merge`, input gradients for saliency |
| **`nn`** | `Linear`, `Dropout`, `Relu`, `Gelu`, `Embedding`, `MultiHeadAttention`, `LayerNorm`, `RMSNorm`, `BatchNorm2d`, `Sequential`, `#[derive(Module)]` |
| **`optim`** | `Sgd`, `Adam`, `AdamW`, parameter groups, learning-rate schedules |
| **`data`** | `Dataset`, `DataLoader` with seeded shuffling, `TensorDataset`, `VecDataset`, MNIST and Tiny Shakespeare loaders |
| **`text`** | `CharTokenizer`, `BpeTokenizer` |
| **`models`** | `DecoderTransformer` with a config and a `KvCache` for generation |
| **`persist`** | safetensors state dicts and training checkpoints that reload optimizer state; files are transactional but not checksummed or authenticated |

## Examples

```sh
cargo run --example tensors                  # tensors, errors, autograd
cargo run --example mlp                      # a full training loop on two spirals
cargo run --example typed --features typed   # compile-time shapes
cargo run --release --example mnist --features hub
                                              # real MNIST, best device, cosine LR
cargo run --release --example tiny_shakespeare --features hub
                                              # AdamW, clipping, cached generation
cargo run --release --example typed_mnist --features typed,hub
                                              # typed CNN and resumable checkpoints
```

## Feature flags

| Feature | Default | What it does |
|---|---|---|
| `typed` | off | Compile-time checked rank, dimensions, dtype and device placement, as a wrapper over the same `Tensor`. Mismatched shapes become type errors. |
| `rayon` | off | Multi-threaded CPU kernels. Results stay bit-identical: kernels partition by output element, so no float is accumulated across threads in a racing order. |
| `hub` | off | Downloads for the bundled MNIST and Tiny Shakespeare datasets. |
| `metal` | on | GPU backend on macOS. `Device::best_available` selects the first Metal device when present, then considers WGPU and CPU. |
| `cuda` | off | Native NVIDIA CUDA backend on Linux and Windows, including Linux under WSL. Supports F16/F32 compute and lossless I64/Bool storage on compute capability 6.0 or newer. Bundled PTX requires a compatible NVIDIA driver but not the CUDA toolkit. |
| `wgpu` | off | Portable native WebGPU backend. F32 is always supported; native F16 is enabled only when the adapter advertises `SHADER_F16`. I64 index storage remains lossless. |
| `testing` | off | The finite-difference gradient harness the crate tests itself with. The one public module outside the stability guarantee. |

## Backends and dtypes

CPU is the reference backend and supports the six runtime dtypes subject to
each operation's contract. On macOS, Metal supports F16/F32 arithmetic and
F16/F32/I64/Bool storage; BF16 and F64 are rejected. On Linux and Windows,
native CUDA is opt-in and supports F16/F32 compute plus lossless I64 and Bool
storage on GPUs with compute capability 6.0 or newer. Its PTX kernels are
bundled with the crate, so a compatible NVIDIA driver is required at runtime
but a CUDA toolkit installation is not. WGPU is also opt-in and supports F32
compute plus native F16 when `SHADER_F16` is available, with
lossless I64 and Bool storage. Unsupported combinations return an error rather
than silently promoting or copying through the host.

`Device::best_available` chooses Metal first on macOS, then CUDA on Linux or
Windows, then the first hardware WGPU adapter, then CPU. WGPU uses native F16
when the adapter exposes `SHADER_F16`; otherwise F16 operations fail loudly
like other unsupported combinations. The feature and hardware must be
available for a device to be selected; the selection order does not promise a
performance win. The complete stability and capability policy is in
[STABILITY.md](STABILITY.md).

### NVIDIA from WSL

For native Linux CUDA in WSL, enable the opt-in `cuda` feature:

```sh
cargo run --release --features cuda,hub --example mnist
```

The crate includes its CUDA kernels as bundled PTX, so no CUDA toolkit is
needed at runtime. CUDA does require a compatible NVIDIA Windows driver with
CUDA support for WSL; do not install a Linux display driver inside WSL.

WGPU over DX12 remains an alternative. WSL does not expose the Windows NVIDIA
Vulkan driver to Linux processes, so build the Windows target from WSL and let
WGPU use the native DX12 driver. On Ubuntu, install the cross-linker and Rust
target once:

```sh
sudo apt-get install gcc-mingw-w64-x86-64
rustup target add x86_64-pc-windows-gnu
```

Then pass the Windows target and `wgpu` feature to the usual Cargo command:

```sh
cargo run --release --target x86_64-pc-windows-gnu --features wgpu,hub --example mnist
```

WSL interoperability runs the resulting `.exe` directly.
`Device::best_available` ignores software-only WGPU adapters such as Vulkan
`llvmpipe`, so their presence does not displace the NVIDIA adapter or prevent
the CPU fallback.

### GPU tests

Enabling `cuda` or `wgpu` makes that backend's hardware tests required: an
initialization failure fails the test run instead of being treated as a skip.
Environments that intentionally validate feature composition without GPU
hardware must opt out explicitly:

```sh
RSTORCH_SKIP_CUDA_TESTS=1 RSTORCH_SKIP_WGPU_TESTS=1 cargo test --all-features
```

Do not set these variables on a hardware test lane; they are intended for
generic CI, documentation builds, and CPU-only development machines.

## Compile-time shapes, if you want them

`typed` is opt-in and additive — it wraps `Tensor` rather than replacing it,
owns no kernels of its own, and computes identical values:

```rust,ignore
let ctx = DeviceCtx::cpu()?;
let x = Tensor2::<2, 3>::from_vec(vec![1.0f32; 6], [2, 3], &ctx)?;
let w = Tensor2::<3, 4>::from_vec(vec![0.1f32; 12], [3, 4], &ctx)?;
let y: Tensor2<2, 4> = x.matmul(&w)?;   // the output type is computed, not asserted
```

A `DYN` axis opts out of static checking where a dimension genuinely is not
known until runtime — a batch size, usually — and is validated at runtime by
the same code the dynamic API uses.

## Stability and MSRV

1.0 is a semver commitment: see [STABILITY.md](STABILITY.md) for exactly what
is covered and what an MSRV bump means. The minimum supported Rust version is
**1.88**, and the crate is edition 2024.

## Version history

This is a complete rewrite. The 0.x line was a different library with a
different API; see [CHANGELOG.md](CHANGELOG.md) for what changed and why there
is no migration path from it.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
