# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and from 1.0.0 the
project follows [semantic versioning](https://semver.org) under the scope
recorded in [STABILITY.md](STABILITY.md).

## [1.0.0] - unreleased

The first stable API, and a complete rewrite. Nothing from the 0.x line
carries over: this shares a name and a purpose with it, and no code.

### The design

- **One concrete `Tensor` with zero generic parameters.** Rank, dtype and
  device are values a tensor carries, not type parameters, so no user
  signature has to be generic over them.
- **Linear gradients.** `Tensor::backward` returns one `Grads`, which is not
  `Clone`, is `#[must_use]`, and is consumed by `Optimizer::step`. Stepping
  twice on the same gradients is a compile error. Accumulation and clipping are
  explicit (`acc.merge(step)?`, `grads.clip_norm(1.0)?`), and there is no
  `zero_grad` because gradients never live in the parameters.
- **One error type.** A single `#[non_exhaustive]` enum whose variants carry
  the operation name and the offending values, so a failure is self-describing
  without a backtrace.
- **Two tiers of fallibility.** Named methods return `Result`; operator sugar
  panics with the identical message under `#[track_caller]`.
- **No silence.** No implicit dtype promotion, no host round-trip when a device
  lacks a kernel, and a parameter that never received a gradient fails at the
  next optimizer step rather than never training.

### Added

- Tensors over `f16`, `bf16`, `f32`, `f64`, `i64` and `bool`: elementwise and
  broadcasting ops, matmul, reductions, indexing and `gather`, `conv2d`,
  `max_pool2d`, softmax and the loss functions.
- Reverse-mode autograd over the whole operation set, including input
  gradients for saliency.
- `nn`: `Linear`, `Dropout`, `Relu`, `Gelu`, `Embedding`,
  `MultiHeadAttention`, `LayerNorm`, `RMSNorm`, `BatchNorm2d`, `Sequential`,
  and `#[derive(Module)]` with loud-by-default field classification.
- `optim`: `Sgd`, `Adam`, `AdamW`, parameter groups and learning-rate
  schedules, with optimizer state that survives a checkpoint round trip.
- `data`: `Dataset`, `DataLoader` with seeded shuffling, `TensorDataset`,
  `VecDataset`, and MNIST and Tiny Shakespeare loaders behind `hub`.
- `text`: `CharTokenizer` and `BpeTokenizer`.
- `models`: a config-driven `DecoderTransformer` with `KvCache` generation.
- `persist`: safetensors state dicts and training checkpoints.
- `typed` (optional): compile-time checked rank, dimensions, dtype and device
  placement, as a wrapper that owns no kernels and computes identical values.
  A `DYN` axis opts out where a dimension is only known at runtime.
- `rayon` (optional): multi-threaded CPU kernels that leave results
  bit-identical.
- `metal` (optional): a macOS GPU backend, conformance-tested against the CPU
  backend. It is currently slower than CPU on the recorded training workloads,
  so `Device::best_available` still returns CPU.

### Changed since 0.2.0

The 0.x API — `SafeModule`, `CrossEntropyLoss`, `SGD`, the `module`/`loss`
namespaces and the `ndarray`-shaped tensor behind them — no longer exists.
There is no migration path and no deprecation cycle; 0.2.0 remains on
crates.io for anyone who needs it.

### Infrastructure

- The public surface is recorded in `api/` for six feature combinations and
  diffed by CI on every run, so a break in the 1.0 guarantee fails the build
  rather than reaching a release.
- CI enforces the 1.88 MSRV, `clippy -D warnings`, warning-free docs, the
  compile-fail diagnostic suites on a pinned toolchain, and Metal execution on
  a macOS runner.

## [0.2.0] and earlier

See the git history on `master` at `ce27c8c` and before. Those releases remain
available on [crates.io](https://crates.io/crates/rstorch).
