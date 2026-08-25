# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and from 1.0.0 the
project follows [semantic versioning](https://semver.org) under the scope
recorded in [STABILITY.md](STABILITY.md).

## [Unreleased]

### Release-boundary hardening

- Dynamic `ModuleExt::state_dict` is now fallible and returns an opaque,
  validated `StateDict`; generic dynamic model checkpoint helpers are available
  in `rstorch::persist`.
- Parameters detach incoming tensors before retaining values or cached leaves,
  and non-floating parameters never create autograd leaves.
- Typed and lazy namespaces are explicitly experimental and outside the
  dynamic 1.x stability guarantee.
- Optimizer scalar validation is backend-neutral, optimizer gradient
  validation avoids reduced-precision observation conversions, and the global
  step clock advances only after the final lazy realization boundary succeeds.

### Lazy element-wise execution

- Added an opt-in thread-local deferred executor for dense CPU element-wise
  chains. The default remains eager (`RSTORCH_FUSION=off`); use
  `rstorch::lazy::set_fusion(true)` or `RSTORCH_FUSION=on` for measurement.
- `Tensor::realize()` now realizes the receiver's own value before flushing
  its device. Deferred execution errors retain the operation name and are not
  cached.

## [1.0.0] - unreleased

The first stable API, and a complete rewrite. Nothing from the 0.x line
carries over: this shares a name and a purpose with it, and no code.

### The design

- **One concrete `Tensor` with zero generic parameters.** Rank, dtype and
  device are values a tensor carries, not type parameters, so no user
  signature has to be generic over them.
- **Linear gradients.** `Tensor::backward` returns one `Grads`, which is not
  `Clone`, is `#[must_use]`, and is consumed by the optimizer's `step`
  (`Sgd`/`Adam`/`AdamW` — there is deliberately no `Optimizer` trait in
  1.0.0; see "Deferred to 1.x" below). Reusing a moved value is a compile
  error; ignoring it emits the normal `#[must_use]` warning. Accumulation,
  measurement and clipping are explicit (`acc.merge(step)?`, `grads.norm()?`,
  `grads.clip_norm(1.0)?`), and there is no `zero_grad` because gradients
  never live in the parameters.
- **One error type.** A single `#[non_exhaustive]` enum whose variants carry
  the operation name and the offending values, so a failure is self-describing
  without a backtrace.
- **Two tiers of fallibility.** Named methods return `Result`; operator sugar
  panics with the identical message under `#[track_caller]`.
- **No silence.** No implicit dtype promotion, no host round-trip when a device
  lacks a kernel, and a parameter that never received a gradient fails at the
  next optimizer step rather than never training.

### Added

- Tensors over `f16`, `bf16`, `f32`, `f64`, `i64` and `bool`, with explicit
  operation-specific dtype contracts and loud unsupported-operation errors.
- Reverse-mode autograd over the whole operation set, including input
  gradients for saliency.
- `nn`: `Linear`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `Flatten`, `Identity`,
  `Dropout`, `Relu`, `Gelu`, `Embedding`, `MultiHeadAttention`, `LayerNorm`,
  `RMSNorm`, `BatchNorm2d`, `Sequential`, `nn::init` (`kaiming_uniform`/
  `kaiming_normal`/`xavier_uniform`/`xavier_normal`/`zeros`/`ones`, plus a
  tree-wide `apply`), `ModuleExt` (`num_params`/`state_dict`/
  `load_state_dict`/`to_device`/`to_dtype` as discoverable methods), and
  `#[derive(Module)]` with loud-by-default field classification.
- `Forward<Input = Tensor>`: the dynamic forward trait now carries an
  associated `Output` type and a defaulted `Input` type parameter, so a
  layer needing more than one tensor (a mask, a conditioning value)
  implements `Forward<ThatStruct>` instead of smuggling state through
  `&mut self`. `MultiHeadAttention` is reachable this way via
  `Forward<AttentionInput>`; `Sequential<Input = Tensor>` follows the same
  parameter.
- `Tensor`: `zeros_like`/`ones_like`/`full_like`; `Neg` and scalar-LHS
  arithmetic operators; `clamp`, `pow`, `sign`, `recip`, `floor`, `ceil`,
  `round`, `erf`, `prod`, `cumsum`, `norm`, `silu`, `leaky_relu`,
  `softplus`, `elu`; `tril`, `triu`, `one_hot`, `repeat`, `split`, `chunk`,
  `flip`, `take_along_dim`, `topk`, `sort`; `Device::synchronize` and
  `Tensor::realize` for the Metal command-buffer batching boundary.
- `optim`: `Sgd`, `Adam`, `AdamW`, parameter groups and learning-rate
  schedules, with optimizer state that survives a checkpoint round trip. A
  hand-written third-party optimizer is supported without a trait —
  `examples/custom_optimizer.rs` builds RMSprop from `Param::get`/`set`/
  `is_frozen`, `Grads::wrt`, and `Envelope::set_section`/`insert_tensor`.
- `data`: `Dataset`, `DataLoader` with seeded shuffling, `TensorDataset`,
  `VecDataset`, and MNIST and Tiny Shakespeare loaders behind `hub`.
- `text`: `CharTokenizer` and `BpeTokenizer`.
- `models`: a config-driven `DecoderTransformer` with `KvCache` generation.
- `models::TransformerConfig`, `persist::MissingPolicy`, and
  `persist::UnexpectedPolicy` are `#[non_exhaustive]`, so a knob or policy
  variant can be added in a later 1.x without a breaking change;
  `TransformerConfig::new`/`with_*` is the constructor path.
- `Error` is `#[non_exhaustive]` and so is every one of its struct variants,
  so a later 1.x can add a variant — or a field to an existing variant —
  without a breaking change. Fifteen public constructors
  (`Error::invalid_arg`, `Error::shape_mismatch`, `Error::data_with`, …) are
  the construction path, since a struct literal is not available downstream.
  `Data`, `Tokenizer` and `Persistence` carry the underlying I/O, JSON or
  safetensors failure as a cause, reachable through
  `std::error::Error::source`, so wrapping no longer discards it.
- `persist`: safetensors state dicts and training checkpoints.
- `typed` (optional): compile-time checked rank, dimensions, dtype and device
  placement, as a wrapper that owns no kernels and computes identical values.
  A `DYN` axis opts out where a dimension is only known at runtime.
- `rayon` (optional): multi-threaded CPU kernels whose results stay
  bit-identical to the single-threaded kernels of the same build.
- `metal` (enabled by default): a macOS GPU backend, conformance-tested against
  the CPU backend. `Device::best_available` selects Metal when available, then
  considers CUDA, WGPU and CPU.
- Metal uses tiled matrix multiplication, SIMD-group reductions, contiguous
  addressing and deterministic indexing fast paths, with synchronized
  CPU-vs-Metal kernel benchmarks covering its main operation families.
- `cuda` (optional): a native NVIDIA backend on Linux and Windows, including
  Linux under WSL, with F16/F32 compute and lossless I64/Bool storage on
  compute capability 6.0 or newer. The bundled PTX needs a compatible driver
  but not the CUDA toolkit (`cudarc` is configured for dynamic loading). It is
  selected after Metal and before WGPU and CPU. No CI runner has NVIDIA
  hardware, so it is validated locally against the CPU reference rather than
  gated — see "Infrastructure" below.
- `wgpu` (optional): a portable native backend with F32 compute and lossless
  I64/Bool storage for supported operations. It is selected after Metal and
  CUDA and before CPU when an adapter is available.

### Changed since 0.2.0

The 0.x API — `SafeModule`, `CrossEntropyLoss`, `SGD`, the `module`/`loss`
namespaces and the `ndarray`-shaped tensor behind them — no longer exists.
There is no migration path and no deprecation cycle; 0.2.0 remains on
crates.io for anyone who needs it.

### Infrastructure

- CI enforces the 1.88 MSRV, `clippy -D warnings`, warning-free docs, the
  compile-fail diagnostic suites on a pinned toolchain, Metal execution and
  conformance on a macOS runner, WGPU execution against a software Vulkan
  adapter (`wgpu-software-adapter`), a Windows build/CPU-test lane, a
  `cargo-deny` dependency gate, and a docs.rs-matching docs build on both
  target platforms. `Cargo.lock` is committed and the diagnostic-pinned
  lanes run `--locked`.
- CUDA has no execution lane in CI (no CI runner has NVIDIA hardware) and is
  therefore validated locally against the CPU reference rather than gated;
  `STABILITY.md` states this explicitly rather than claiming coverage CI
  does not provide.

### Known limitations

- Four CPU kernel-quality targets set during development remain unmet at
  release: `matmul/square_f32/256` measures 1.2157 ms against a ≤ 1.05 ms
  target (open all along; the portable pure-Rust kernel is a deliberate
  choice); `elementwise/add_f32/1048576` measures ~155–160 µs against
  ≤ 104 µs, but its previously-passing 85.844 µs baseline did not reproduce
  on the measuring machine even for code predating this release, so that row
  needs re-measuring on an idle box rather than treating as a regression;
  softmax and autograd forward/backward improved substantially
  (509.65 µs → 134.36 µs, 2.2083 ms → 1.8257 ms) but still narrowly miss
  their 131 µs and 1.77 ms targets. Measured on an Apple M4 Pro with
  Criterion's bench profile. None of this is a semver commitment —
  `STABILITY.md`'s "What Is Not Covered" already says backend performance is
  not a release guarantee — but the targets are recorded here rather than
  quietly dropped.

## [0.2.0] and earlier

See the git history on `master` at `ce27c8c` and before. Those releases remain
available on [crates.io](https://crates.io/crates/rstorch).
