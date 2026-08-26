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
    type Output = Tensor;

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

`Forward` is `Forward<Input = Tensor>`, so a layer that needs more than one
tensor — an attention mask, a conditioning embedding — declares a struct and
implements `Forward<ThatStruct>` instead of smuggling the extra state through
`&mut self` in call order. `MultiHeadAttention` is the crate's own case:
`Forward<AttentionInput>`, where the mask is an explicit `Option` field that
has to be written out rather than hidden state a layer picks for you. `Mode`
stays crate-owned and closed; the *input* is the user's channel.

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
sugar (`+`, `-`, `*`, `/`, unary `-`, and the scalar forms on either side)
panics with the identical message under `#[track_caller]`, so exploratory code
stays terse and library code stays total.

**Errors land where you can act on them.** Every argument error — mismatched
shapes, ranks, dtypes or devices, an out-of-range axis, a rejected argument —
is decided from tensor metadata before any kernel runs, so it is reported at
the call site, which is what makes the `#[track_caller]` panic location useful.
`Unsupported` and `Backend` come from execution, and backends batch work: on
Metal, dispatches are encoded into a command buffer and flushed on a threshold,
so a kernel failure can surface at the next transfer rather than at the
operation that queued it. Both name the operation that failed, so the message
identifies what broke even when the location belongs to the transfer. An
out-of-range value *inside an index tensor* is the one case that belongs to
both groups: the CPU backend reads it on the host and reports immediately, the
GPU backends validate on device and report at the next transfer — never later
than the moment a wrong value would have become visible. `STABILITY.md` states
the split precisely.

## Deferred element-wise execution (experimental)

Dense element-wise chains have an opt-in deferred executor. It is an
experimental execution policy and is **off by default**. Enable it per thread
for an experiment or benchmark:

```rust
let _restore = rstorch::lazy::set_fusion(true);
let y = x.add_scalar(1.0)?.mul_scalar(2.0)?;
y.realize()?; // force y's value, then flush its device
```

The `rstorch::lazy` namespace is outside the dynamic 1.x stability guarantee.
Its thread-local controls, supported operation set, and realization schedule
may change independently. The `RSTORCH_FUSION=on` environment variable selects
the process default for threads that have not overridden it. `set_fusion`
returns a restore guard, so tests and concurrent callers do not race a
process-global switch. Deferred errors are reported at the next materialization
and retain the failing operation name; shape, dtype, device and axis validation
remains at the call site.

## What is in the box

| | |
|---|---|
| **Tensors** | elementwise, broadcasting, matmul, reductions, indexing/gather, `conv2d`, `max_pool2d`, softmax and losses over six runtime dtypes, with operation-specific support and loud `Unsupported` errors |
| **Autograd** | reverse-mode over the whole op set, `Grads::norm`/`clip_norm`/`scale`/`merge`, input gradients for saliency |
| **`nn`** | `Linear`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `Flatten`, `Identity`, `Dropout`, `Relu`, `Gelu`, `Embedding`, `MultiHeadAttention`, `LayerNorm`, `RMSNorm`, `BatchNorm2d`, `Sequential`, `nn::init`, `ModuleExt`, `#[derive(Module)]` |
| **`optim`** | `Sgd`, `Adam`, `AdamW`, parameter groups, learning-rate schedules |
| **`data`** | `Dataset`, `DataLoader` with seeded shuffling, `TensorDataset`, `VecDataset`, MNIST and Tiny Shakespeare loaders |
| **`models`** | `DecoderTransformer` with a config and a `KvCache` for generation |
| **`persist`** | safetensors model state and composable envelope sections; model and optimizer loads are independently transactional, but files are not checksummed or authenticated |

### Checkpoints and state

Generic model checkpoints contain model state. `DecoderTransformer` convenience
checkpoints contain config plus model tensors and exclude optimizer, RNG,
application, cache, and tokenizer state. The `Envelope` is versioned as a
container, while semantic sections own independent schemas; the transformer
config is strict `version=1`, and future readers must explicitly support prior
versions. There is no combined transaction spanning model, optimizer, RNG,
caches, and application state.

Randomness is caller-owned: capture an `Rng` with `state` and resume it with
`from_state` when needed. `Dropout`'s private child stream is not in model
state, so exact training resume with dropout requires reconstructing that model
stream.

## Examples

```sh
cargo run --example tensors                  # tensors, errors, autograd
cargo run --example mlp                      # a full training loop on two spirals
cargo run --example custom_optimizer          # RMSprop from public items only
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
| `typed` | off | Experimental compile-time checked rank, dimensions, dtype and device placement, as a wrapper over the same `Tensor`; outside the dynamic 1.x stability guarantee. |
| `rayon` | off | Multi-threaded CPU kernels. Results stay bit-identical to the single-threaded kernels of the same build: kernels partition by output element, so no float is accumulated across threads in a racing order. |
| `hub` | off | Downloads for the bundled MNIST and Tiny Shakespeare datasets. `DatasetHub::default_cache` uses `RSTORCH_DATA`, then `$HOME/.cache/rstorch`, then CWD-relative `data/`. |
| `metal` | on | GPU backend on macOS. `Device::best_available` probes Metal ordinal 0 with complete context initialization, then considers CUDA, WGPU and CPU. |
| `cuda` | off | Native NVIDIA CUDA backend on Linux and Windows, including Linux under WSL. Supports F16/F32 compute and lossless I64/Bool storage on compute capability 6.0 or newer. Bundled PTX requires a compatible NVIDIA driver but not the CUDA toolkit. **Validated locally, not CI-gated** — see [STABILITY.md](STABILITY.md). |
| `wgpu` | off | Portable native WebGPU backend. F32 is always supported; native F16 is enabled only when the adapter advertises `SHADER_F16`. I64 index storage remains lossless. |
| `testing` | off | The finite-difference gradient harness; outside the stability guarantee. |
| `bench-resnet` | off | Benchmark-only feature enabling the `resnet_mnist` benchmark target; it adds no library runtime surface. |

### Dataset hub cache and paths

`DatasetHub::dataset_dir`, `resource_path`, and related public helpers validate
that each dataset/resource name is one ordinary path component and return an
error otherwise. Cache hits are checked against the declared size and checksum
before use; Tiny Shakespeare verifies both before UTF-8 decoding. The checksum
is an integrity check, not a cryptographic authenticity guarantee.

## Backends and dtypes

CPU is the reference backend and supports the six runtime dtypes subject to
each operation's contract. On macOS, Metal supports F16/F32 arithmetic and
F16/F32/I64/Bool storage; BF16 and F64 are rejected. On Linux and Windows,
native CUDA is opt-in and supports F16/F32 compute plus lossless I64 and Bool
storage on GPUs with compute capability 6.0 or newer. Its PTX kernels are
bundled with the crate, so a compatible NVIDIA driver is required at runtime
but a CUDA toolkit installation is not. CUDA is validated locally against the
CPU reference rather than in CI (no CI runner has CUDA hardware), so its
*behaviour* carries a weaker claim than the other backends' — its API is
covered like everything else; see [STABILITY.md](STABILITY.md). WGPU is also
opt-in and supports F32 compute plus native F16 when `SHADER_F16` is available,
with lossless I64 and Bool storage. Unsupported combinations return an error
rather than silently promoting or copying through the host.

`Device::best_available` first accepts Metal on macOS only when its complete
context initialization succeeds, then considers CUDA, the best eligible WGPU
adapter, and CPU. WGPU adapter type and backend are driver-reported
classifications; automatic selection excludes adapters reported as `Cpu`, while
other software classification is backend-dependent. WGPU ordinals are indices
in the current process's adapter set, not stable device identities, and must
not be persisted. WGPU uses native F16 when
the adapter exposes `SHADER_F16`; otherwise F16 operations fail loudly like
other unsupported combinations. The selection order does not promise a
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
`Device::best_available` excludes adapters the driver reports as `Cpu`; other
software classifications are backend-dependent. Process-local WGPU ordinals
are not stable identities and must not be persisted, so callers should not
serialize them as device selections.

### GPU tests

On macOS, Metal is enabled by default; when a Metal device is present its
hardware tests are required. Enabling `cuda` or `wgpu` likewise makes that
backend's hardware tests required: an initialization failure fails the test
run instead of being treated as a skip. Environments that intentionally
validate feature composition without accelerator hardware must opt out
explicitly:

```sh
RSTORCH_SKIP_METAL_TESTS=1 RSTORCH_SKIP_CUDA_TESTS=1 \
RSTORCH_SKIP_WGPU_TESTS=1 cargo test --all-features
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

Public record-shaped types use constructors/builders or non-exhaustive
boundaries, so downstream code should not depend on exhaustive struct
literals. Traits are intentional extension points: 1.x additions use defaults
or extension traits rather than new required methods. The derive macros resolve
renamed runtime dependencies; `rstorch` and `rstorch-derive` release in exact
lockstep.

## Version history

This is a complete rewrite. The 0.x line was a different library with a
different API; see [CHANGELOG.md](CHANGELOG.md) for what changed and why there
is no migration path from it.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
