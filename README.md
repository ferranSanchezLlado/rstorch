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
gradients, or dropping them on the floor, is a compile error — not a flat loss
curve you debug for an afternoon.

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
| **Tensors** | elementwise, broadcasting, matmul, reductions, indexing/gather, `conv2d`, `max_pool2d`, softmax and losses over `f16`/`bf16`/`f32`/`f64`/`i64`/`bool` |
| **Autograd** | reverse-mode over the whole op set, `Grads::clip_norm`/`scale`/`merge`, input gradients for saliency |
| **`nn`** | `Linear`, `Dropout`, `Relu`, `Gelu`, `Embedding`, `MultiHeadAttention`, `LayerNorm`, `RMSNorm`, `BatchNorm2d`, `Sequential`, `#[derive(Module)]` |
| **`optim`** | `Sgd`, `Adam`, `AdamW`, parameter groups, learning-rate schedules |
| **`data`** | `Dataset`, `DataLoader` with seeded shuffling, `TensorDataset`, `VecDataset`, MNIST and Tiny Shakespeare loaders |
| **`text`** | `CharTokenizer`, `BpeTokenizer` |
| **`models`** | `DecoderTransformer` with a config and a `KvCache` for generation |
| **`persist`** | safetensors state dicts, and training checkpoints that reload optimizer state |

## Examples

```sh
cargo run --example tensors                  # tensors, errors, autograd
cargo run --example mlp                      # a full training loop on two spirals
cargo run --example typed --features typed   # compile-time shapes
```

## Feature flags

| Feature | Default | What it does |
|---|---|---|
| `typed` | off | Compile-time checked rank, dimensions, dtype and device placement, as a wrapper over the same `Tensor`. Mismatched shapes become type errors. |
| `rayon` | off | Multi-threaded CPU kernels. Results stay bit-identical: kernels partition by output element, so no float is accumulated across threads in a racing order. |
| `hub` | off | Downloads for the bundled MNIST and Tiny Shakespeare datasets. |
| `metal` | off | GPU backend on macOS. Correct and conformance-tested, but **slower than the CPU backend** on the recorded training workloads, which is why `Device::best_available` still returns CPU. See [STABILITY.md](STABILITY.md). |
| `testing` | off | The finite-difference gradient harness the crate tests itself with. The one public module outside the stability guarantee. |

## Compile-time shapes, if you want them

`typed` is opt-in and additive — it wraps `Tensor` rather than replacing it,
owns no kernels of its own, and computes identical values:

```rust,ignore
let ctx = DeviceCtx::<Cpu>::cpu()?;
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
