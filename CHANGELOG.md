# Changelog

This project is a rewrite of the 0.x crate. The 0.x releases remain available
on [crates.io](https://crates.io/crates/rstorch), but they do not share this API.

## [Unreleased]

The next release is intended to be `1.0.0`.

### Added

- A dynamic `Tensor` API with reverse-mode autograd and one linear `Grads`
  value per backward pass. Gradients are consumed by optimizer steps; there is
  no parameter-side `zero_grad` state.
- Six runtime dtypes (`f16`, `bf16`, `f32`, `f64`, `i64`, and `bool`) with
  operation-specific dtype checks and structured errors.
- Tensor operations including broadcasting, matmul, reductions, indexing,
  convolution, pooling, softmax, losses, and input gradients.
- Neural-network layers, initialization helpers, `Sequential`, parameter
  visitors, and `#[derive(Module)]`.
- `Sgd`, `Adam`, and `AdamW`, parameter groups, schedules, safetensors state,
  and versioned checkpoint envelopes.
- Data loaders and in-memory datasets, optional MNIST and Tiny Shakespeare
  downloads, `CharTokenizer`, `BpeTokenizer`, and a configurable decoder
  transformer with a generation cache.
- Optional compile-time checked tensor wrappers in `typed`, parallel CPU
  kernels in `rayon`, and native Metal, CUDA, and WGPU backends.
- An opt-in deferred executor for dense CPU elementwise chains. It is
  experimental and off by default.

### Changed

- `Forward` accepts an explicit input type, so layers that need more than one
  tensor can carry that data in a small input struct.
- The `typed` and `lazy` namespaces are experimental and are not part of the
  dynamic API compatibility promise.
- Model and optimizer state loading is validated and transactional per state
  type. Transformer configuration sections use a strict versioned schema.
- Reduced-precision optimizer state uses a wider accumulation dtype. Optimizer
  validation is independent of the backend and lazy execution advances its
  step clock only after realization succeeds.
- Hub cache paths and downloaded resources are validated before use. Metadata
  errors retain their operation names, invalid MNIST labels are rejected, and
  automatic Metal selection checks that the device can initialize fully.
- The derive macros work when the runtime dependency is renamed, and the root
  crate requires the matching derive version exactly.
- The public persistence and hub records use constructors, builders, or
  non-exhaustive boundaries so they can gain fields in a minor release.

### Compatibility notes

- This is a complete rewrite from 0.x; there is no migration layer.
- CPU is the reference backend. Metal and WGPU run conformance tests in CI;
  CUDA is compiled and checked locally because CI has no NVIDIA hardware.
- Backend performance and benchmark results are not release guarantees.

## 0.2.0 and earlier

See the git history before the 1.0 rewrite. Those releases remain available on
[crates.io](https://crates.io/crates/rstorch).
