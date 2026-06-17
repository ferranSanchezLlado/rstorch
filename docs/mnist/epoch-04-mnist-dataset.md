# Epoch 04: MNIST Dataset

## Goal

Ship a built-in MNIST dataset that downloads, caches, decompresses, and parses
the IDX files, behind a feature flag, so the default build stays
dependency-free.

## Context

Epoch 03 introduced generic `Dataset`/`DataLoader` machinery. This epoch adds
the first concrete dataset. MNIST is distributed as four gzip-compressed IDX
files (train/test images and labels). Acquiring it requires HTTP download and
gzip decompression, which means new dependencies; these must be isolated behind
a feature so the default CPU build remains zero-dependency.

## Scope

In scope:

- A `datasets` feature gating new dependencies.
- An IDX parser (magic number, dimensions, raw bytes).
- An MNIST loader that downloads, caches, decompresses, and parses the four
  files.
- `Mnist::train()`, `Mnist::test()`, and `Mnist::from_dir(path)` returning a
  `Dataset` whose item is a normalized `28 x 28` image and a `u8` label.
- Pixel normalization to `f32` in `[0, 1]`.
- Prelude exports under the `datasets` feature.

Out of scope:

- Other datasets (CIFAR, Fashion-MNIST, etc.).
- Augmentation or transforms beyond normalization.
- Streaming or memory-mapped access.
- Bundling MNIST binary assets in the repository.

## Dependencies

Add an optional feature with the minimal crates needed:

```toml
[features]
datasets = ["dep:ureq", "dep:flate2"]
```

- `ureq` (or a comparably small blocking HTTP client) for downloads.
- `flate2` for gzip decompression.

Both must be `optional = true` and excluded from `default`. Pin reasonable
versions and minimal feature sets. The default CPU build must not pull these in.

## IDX Parser

Parse the IDX format defensively:

- read and validate the magic number (data type and rank),
- read dimension sizes,
- read the raw payload and verify its length matches the product of dimensions,
- fail with a clear error on truncated, corrupt, or unexpected files rather than
  constructing wrong-shaped tensors.

The parser should be unit-testable on small in-memory byte buffers without
network access.

## MNIST Loader

- Resolve a cache directory (OS cache dir or a `target/`-based path).
- Download each of the four files from a stable mirror if not already cached;
  skip files already present.
- Decompress gzip into the cache.
- Parse images to `28 x 28` `u8` pixels and labels to class indices.
- Normalize pixels to `f32` in `[0, 1]` and expose each sample as image-shaped
  data, not a flattened `784`-element vector.
- `Mnist::from_dir(path)` reads pre-downloaded (and optionally pre-decompressed)
  files for offline use, sharing the parser with the download path.

Expose the result as a `Dataset` (Epoch 03) so it plugs into `DataLoader` with
`ImageOneHotClassification<28, 28, 10>`, producing `Tensor3D<BATCH, 28, 28>`
images and one-hot `Tensor2D<BATCH, 10>` targets.

## Tests

Unit tests (no network, always run):

- IDX parser on synthetic byte buffers: correct dims and values.
- IDX parser rejects bad magic numbers and truncated payloads.
- pixel normalization maps `0..=255` to `[0, 1]`.

Integration tests (feature-gated and/or `#[ignore]`):

- `Mnist::train()` / `Mnist::test()` download (or load from cache) and report
  expected example counts (`60000` train, `10000` test) and shapes.
- `Mnist::from_dir` loads a pre-downloaded copy offline.

Network tests must not run on the default `cargo +nightly test` invocation.

## Verification

Default offline suite:

```text
cargo +nightly test
```

Dataset integration (network):

```text
cargo +nightly test --features datasets -- --ignored
```

## Done Criteria

- A `datasets` feature exists and gates all new dependencies.
- The default CPU build remains dependency-free.
- The IDX parser is validated offline on synthetic buffers.
- `Mnist::train`/`test`/`from_dir` return a `Dataset` of normalized `28 x 28`
  images and labels; batching preserves images as `Tensor3D<BATCH, 28, 28>`.
- Download path caches files and skips re-downloading.
- Network-dependent tests are opt-in and do not affect the default suite.
