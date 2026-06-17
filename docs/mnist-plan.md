# RsTorch MNIST And Standard Constructs Plan

This document is the implementation guide for taking RsTorch from "trains tiny
hand-written models" to "trains a real sequential model on the MNIST dataset
through standardized, reusable constructs."

The earlier autograd restart built a safe, const-generic, backend-generic tensor
and autograd core. The training-usability plan added initialization, losses,
activations, and optimizers. This plan adds the higher-level ergonomics a user
expects from a PyTorch-like framework: a rank-3 tensor with reshape, a
`Sequential` model container, a `Dataset`/`DataLoader` abstraction, a built-in
MNIST dataset, and a capstone integration test that proves the whole stack works
end to end.

## Project Goals

- Add `Tensor3D` and a typed `reshape` so batched, multi-axis data has a home
  without abandoning compile-time shape safety.
- Introduce a standardized layer/activation trait and a `Sequential` container
  so models are composed instead of hand-written every time.
- Introduce a `Dataset` and `DataLoader` abstraction for batching, shuffling,
  and iteration, with fixed compile-time batch shapes at the boundary.
- Ship a built-in MNIST dataset that downloads, caches, decompresses, and parses
  the IDX files behind a feature flag.
- Add a capstone integration test that builds, trains, and evaluates a
  sequential network on real MNIST.
- Preserve every existing safety invariant: typed shapes, dtypes, backends, safe
  storage, optimizer-only mutation, and scalar-only `backward()`.
- Keep the default CPU build dependency-free; isolate new dependencies behind a
  dataset feature flag.

## Non-Goals

- No higher-rank tensors beyond `Tensor3D` in this plan (`Tensor4D` and a
  generic rank-N redesign are deferred).
- No convolutional layers, pooling, or image-specific kernels.
- No serialization or checkpointing of models or optimizer state.
- No automatic device placement or multi-GPU.
- No streaming or memory-mapped datasets; the MNIST subset is loaded into memory.
- No derive macros for module composition.

## Rust Requirements

The project continues to require nightly Rust for type-level shape arithmetic:

```rust
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
```

The capstone integration test must run on the default toolchain command
`cargo +nightly test` without network access; the network-dependent path is
opt-in (feature-gated and/or `#[ignore]`).

## Public API Direction

The target user experience is a compact, familiar training script:

```rust
use rstorch::prelude::*;

let mut rng = SmallRng::seed_from_u64(42);

let mut model = Sequential::new()
    .add(Linear::<784, 128>::kaiming_uniform(&mut rng))
    .add(ReLU)
    .add(Linear::<128, 10>::xavier_uniform(&mut rng));

let dataset = Mnist::train()?;                 // feature = "datasets"
let loader = DataLoader::new(dataset)
    .shuffle(&mut rng)
    .collate::<ImageOneHotClassification<28, 28, 10>>()
    .batch_size::<64>();

let mut optimizer = Adam::new(0.001);

for (images, labels) in loader.iter() {        // images: Tensor3D<64, 28, 28>
    model.zero_grad();
    let images = images.flatten_2d();
    let logits = model.forward(&images);       // logits: Tensor2D<64, 10>
    let loss = cross_entropy_one_hot(&logits, &labels);
    loss.backward();
    optimizer.step(model.parameters_mut());
}
```

The exact signatures may change during implementation, but the result should be
this level of ergonomics: composed models, named layers and activations, and an
iterable batched dataset, all while keeping batch and feature dimensions in the
type system.

## Core Type Additions

### Rank-3 Tensor

Add a third shape marker and tensor alias alongside the existing ones:

```rust
pub struct D3<const A: usize, const B: usize, const C: usize>;

pub type Tensor3D<const A: usize, const B: usize, const C: usize, E = f32, BK = Cpu>
    = Tensor<D3<A, B, C>, E, BK>;
```

`Tensor3D` exists to represent batched, multi-axis inputs (for example MNIST
images as `Tensor3D<BATCH, 28, 28>`) before they are flattened into the matrix
form `Linear` consumes. Keep its op surface small: construction, `to_vec`,
`shape`, `flatten_2d`, and `reshape_2d`.

### Typed Reshape

Add reshape between ranks with a compile-time element-count guard, following the
existing `flatten` precedent that uses `generic_const_exprs`:

```rust
impl<const A: usize, const B: usize, const C: usize, E, BK> Tensor3D<A, B, C, E, BK> {
    pub fn reshape_2d<const M: usize, const N: usize>(&self) -> Tensor2D<M, N, E, BK>
    where
        [(); A * B * C]:,
        [(); M * N]:;
}
```

Reshape must preserve total element count, carry gradients (identity reshape in
the backward pass), and never copy-reinterpret across dtype or backend.

## Module And Composition Model

The existing `Linear` already implements `Module`. This plan standardizes a
single trait that both layers and activations implement so they can be composed,
then provides a `Sequential` container.

Design direction:

- A `Layer` trait with a `forward` that ties input and output shapes through the
  type system where possible.
- Activations (`ReLU`, `Tanh`, `Sigmoid`) become zero-field unit structs that
  implement the layer trait by calling the existing tensor ops.
- `Sequential` chains layers and forwards `parameters_mut`/`zero_grad` to its
  contents, satisfying the existing `Module` trait.

Rust makes fully generic heterogeneous sequential composition hard. Prefer a
pragmatic builder (`.add(...)`) over solving the most general abstraction first,
consistent with the autograd plan's guidance to not solve the hardest
abstraction prematurely. If a fully type-checked shape chain proves too costly,
a documented runtime shape assertion at `add`/first-`forward` time is acceptable
as a fallback, but compile-time chaining is preferred.

## Data Abstraction Model

Introduce two traits and one concrete loader:

- `Dataset`: random-access collection exposing `len()` and an associated `Item`
  type. It must not assume classification, flat tensors, or labels.
- `Collate`: converts `[Dataset::Item; BATCH]` into one typed batch. This is
  where task-specific tensor construction lives, including one-hot labels,
  padding, masks, image stacking, or regression targets.
- `DataLoader`: wraps a `Dataset`, owns batch size as a const generic after
  `.batch_size::<BATCH>()`, supports optional seeded shuffling via `SmallRng`,
  and yields whatever fixed-shape batch the selected collator produces.

Constraints:

- The batch dimension is a compile-time const generic at the loader boundary so
  downstream tensor shapes stay typed.
- A trailing partial batch is dropped by default (documented), because batch
  size is encoded in the type and cannot vary per iteration.
- Shuffling uses the project-owned `SmallRng`; it must be deterministic given a
  seed.
- One-hot label construction for classification lives in small helpers and
  classification collators, not in the core `Dataset` trait or loader.
- MNIST batching must preserve the image shape as `Tensor3D<BATCH, 28, 28>`;
  MLP training code calls `flatten_2d()` before the first `Linear`.

## MNIST Dataset Model

Ship a built-in MNIST dataset behind a feature flag so the default build stays
dependency-free.

- Feature: `datasets` (enables a download client and gzip decompression).
- `Mnist::train()` / `Mnist::test()` return a `Dataset` whose item is a normalized
  `28 x 28` image and a `u8` label. Batched MNIST images should be collated as
  `Tensor3D<BATCH, 28, 28>` rather than flattened at dataset construction time.
- On first use the dataset downloads the four IDX gzip files from a stable
  mirror into a cache directory (default under the OS cache dir or
  `target/`), skips files already present, decompresses, and parses IDX.
- Pixels normalize to `f32` in `[0, 1]`.
- Provide `Mnist::from_dir(path)` for offline/pre-downloaded use so tests and
  air-gapped environments can avoid the network.

## Dependency Policy

The default CPU build must remain dependency-free. New dependencies are allowed
only under the `datasets` feature:

```toml
[features]
datasets = ["dep:ureq", "dep:flate2"]
```

- HTTP download: a small, well-maintained blocking client (for example `ureq`).
- Gzip decompression: `flate2`.

Exact crate choices may be adjusted during implementation, but they must be
optional, behind `datasets`, and must not leak into the default feature set.
Pin reasonable versions and prefer minimal feature sets.

## Design Constraints

- Tensor shapes remain compile-time const generics. The MNIST input is built as
  `Tensor3D<BATCH, 28, 28>` then flattened with `flatten_2d()` to
  `Tensor2D<BATCH, 784>` before the first `Linear` in MLP examples.
- Gradients accumulate on leaves; training loops must call `zero_grad()` each
  iteration.
- Optimizer updates run under `no_grad`; no new mutation path is introduced.
- Keep type erasure for graph traversal internal; all new public APIs stay
  strongly typed.
- Backend kernels receive runtime dimensions; do not leak `generic_const_exprs`
  bounds through the data or module APIs more than necessary.
- The capstone integration test must be reproducible with a fixed seed and must
  not flake on the default offline test run.

## Safety Invariants

The existing non-negotiable invariants from the autograd plan still hold. This
plan adds:

- `reshape` must preserve element count at compile time and must not
  reinterpret across dtype or backend.
- The dataset download path must validate IDX magic numbers and dimensions
  before constructing tensors, and must fail loudly on corrupt or truncated
  files rather than producing silently wrong shapes.
- The `datasets` feature must not change behavior or dependencies of the default
  CPU build.

## Testing Strategy

Unit tests:

- `Tensor3D` construction, shape metadata, `to_vec`.
- `reshape` value correctness and element-count guard.
- activation unit-struct forward values match the existing tensor ops.
- `Sequential` forward output shape and parameter collection.
- `Dataset`/`DataLoader`/`Collate` batch shapes, batch count, drop-last behavior,
  deterministic shuffle given a seed, identity collation, flat classification
  collation, and image-shaped classification collation.
- IDX parser on small synthetic IDX byte buffers (no network).

Compile-fail tests with `trybuild`:

- reshape with mismatched element counts.
- `Sequential` with incompatible adjacent layer shapes (if compile-time
  chaining is implemented).

Integration tests:

- offline synthetic classification through `Sequential` + `DataLoader` +
  `cross_entropy_one_hot` + `Adam` (loss decreases).
- capstone real-MNIST training and evaluation, feature-gated and/or `#[ignore]`.

## Epochs

The work is split into focused epochs. Each epoch must compile and pass tests
before the next begins.

- [Epoch 01: Rank-3 Tensor And Reshape](mnist/epoch-01-tensor3d-and-reshape.md)
- [Epoch 02: Layer Trait, Activations, And Sequential](mnist/epoch-02-sequential-and-activations.md)
- [Epoch 03: Dataset And DataLoader](mnist/epoch-03-dataset-and-dataloader.md)
- [Epoch 04: MNIST Dataset](mnist/epoch-04-mnist-dataset.md)
- [Epoch 05: MNIST Training Integration Test](mnist/epoch-05-mnist-integration-test.md)

## Implementation Rule

At the start of each epoch:

1. Read this top-level plan.
2. Read the epoch document.
3. Keep changes limited to that epoch's scope.
4. Make the crate compile.
5. Add or update tests appropriate to that epoch.
6. Do not begin the next epoch with failing tests unless the failure is
   documented and intentional.
