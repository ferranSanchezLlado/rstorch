# Epoch 03: Dataset And DataLoader

## Goal

Introduce reusable dataset and batching abstractions without baking in MNIST,
classification, flat feature vectors, or one-hot labels.

## Context

The later MNIST dataset needs a loader that can produce fixed-shape batches for
typed training code, but the core data API should also support future datasets
such as text, regression rows, image samples, unsupervised inputs, and custom
targets. The dataset should describe individual examples; collation should
describe how examples become a typed batch.

## Scope

In scope:

- A generic `Dataset` trait with an associated `Item` type.
- A `Collate` trait that converts `[Dataset::Item; BATCH]` into a typed batch.
- A `DataLoader` that owns a dataset, supports deterministic shuffling, encodes
  batch size as a const generic, and drops trailing partial batches.
- An identity collator for non-tensor or custom item batches.
- A flat one-hot classification collator for tabular/simple MLP examples.
- An image one-hot classification collator for MNIST-style `Tensor3D` batches.
- One-hot helper functions for classification collators.

Out of scope:

- Downloading or parsing MNIST files.
- Padding or tokenization for text datasets.
- Variable-size batches.
- Multi-worker or asynchronous loading.
- Streaming or memory-mapped datasets.

## API Shape

Datasets expose individual items only:

```rust
pub trait Dataset {
    type Item;

    fn len(&self) -> usize;
    fn get(&self, index: usize) -> Option<Self::Item>;
}
```

Collators own task-specific batching:

```rust
pub trait Collate<Item, const BATCH: usize, E = f32, BK = Cpu> {
    type Batch;

    fn collate(items: [Item; BATCH]) -> Self::Batch;
}
```

For MNIST, the dataset item should be `([[f32; 28]; 28], u8)`, and batching via
`ImageOneHotClassification<28, 28, 10>` should yield:

```rust
(Tensor3D<BATCH, 28, 28>, Tensor2D<BATCH, 10>)
```

An MLP training loop then explicitly reshapes the images:

```rust
let loader = DataLoader::new(dataset)
    .shuffle(&mut rng)
    .collate::<ImageOneHotClassification<28, 28, 10>>()
    .batch_size::<64>();

for (images, labels) in loader.iter() {
    let images = images.flatten_2d();
    let logits = model.forward(&images);
    let loss = cross_entropy_one_hot(&logits, &labels);
}
```

## Constraints

- The core `Dataset` trait must not mention labels, classes, tensors, or feature
  dimensions.
- The batch dimension is a compile-time const generic at the data-loader
  boundary.
- A trailing partial batch is dropped by default because the batch size is part
  of the output type.
- Shuffling uses the project-owned `SmallRng` and must be deterministic given a
  seed.
- Classification one-hot encoding belongs in helpers/collators, not in
  `Dataset` or `DataLoader`.
- MNIST image batches preserve image shape as `Tensor3D<BATCH, 28, 28>` and MLP
  examples call `flatten_2d()` before the first `Linear`.

## Tests

Unit tests:

- identity collation returns `[Item; BATCH]`.
- flat classification collation returns `Tensor2D<BATCH, FEATURES>` inputs and
  one-hot `Tensor2D<BATCH, CLASSES>` targets.
- image classification collation returns `Tensor3D<BATCH, HEIGHT, WIDTH>` inputs
  and one-hot `Tensor2D<BATCH, CLASSES>` targets.
- `DataLoader` reports full-batch count and drops trailing partial batches.
- seeded shuffling is deterministic.
- one-hot helpers produce expected targets and reject out-of-range labels.

## Verification

```text
cargo +nightly test
```

## Done Criteria

- `Dataset`, `Collate`, and `DataLoader` are public and exported through the
  prelude.
- The loader is reusable beyond MNIST and flat classification.
- MNIST has a clear path to image-shaped `Tensor3D` batches.
- The crate compiles and the default test suite passes.
