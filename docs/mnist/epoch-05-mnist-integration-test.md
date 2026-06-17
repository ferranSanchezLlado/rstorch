# Epoch 05: MNIST Training Integration Test

## Goal

Add the capstone integration test that builds, trains, and evaluates a
sequential network on MNIST, proving the whole stack works together:
`Tensor3D`/reshape, `Sequential`/activations, `Dataset`/`DataLoader`, the MNIST
dataset, cross entropy, autograd, and an optimizer.

## Context

Epochs 01-04 standardized the constructs. This epoch wires them into a single
end-to-end training run. It is both a demonstration of the intended user
experience and a regression guard across the tensor, autograd, nn, optim, data,
and dataset modules.

Two test paths are needed: an offline path that always runs on
`cargo +nightly test`, and a real-MNIST path that is feature-gated and/or
`#[ignore]` so the default suite stays fast and network-free.

## Scope

In scope:

- One integration test file (for example `tests/mnist_training.rs`).
- An offline synthetic test exercising the full
  `Sequential` + `DataLoader` + `cross_entropy_one_hot` + `Adam` pipeline.
- A real-MNIST test using `Mnist` + `DataLoader`, feature-gated and/or
  `#[ignore]`.
- A reusable accuracy helper.
- Optionally, a runnable `examples/mnist.rs` mirroring the README workflow.

Out of scope:

- Hyperparameter tuning beyond reaching a reasonable accuracy floor.
- Model serialization.
- Benchmarking or performance assertions.

## Model

Compose the model with `Sequential` (Epoch 02), seeded for reproducibility:

```rust
let mut rng = SmallRng::seed_from_u64(42);
let mut model = Sequential::new()
    .add(Linear::<784, 128>::kaiming_uniform(&mut rng))
    .add(ReLU)
    .add(Linear::<128, 10>::xavier_uniform(&mut rng));
```

`forward` returns raw logits; `cross_entropy_one_hot` applies log-softmax
internally.

## Data Path

- Real path: `Mnist::train()` / `Mnist::test()` (Epoch 04) wrapped in a
  `DataLoader` with `ImageOneHotClassification<28, 28, 10>`, a fixed
  const-generic batch size (for example `64`), and seeded shuffle. Images arrive
  as `Tensor3D<BATCH, 28, 28>` and are flattened with `flatten_2d()` to
  `Tensor2D<BATCH, 784>` immediately before the first `Linear`.
- Offline path: a seeded synthetic image dataset of separable `28 x 28` samples
  across `10` classes, wrapped in the same image collator, so the identical
  training code runs without network.

## Training Loop

```rust
let mut optimizer = Adam::new(0.001);
for (images, labels) in loader.iter() {
    model.zero_grad();
    let images = images.flatten_2d();
    let logits = model.forward(&images);
    let loss = cross_entropy_one_hot(&logits, &labels);
    loss.backward();
    optimizer.step(model.parameters_mut());
}
```

Record the initial loss before training for the convergence assertion. Call
`zero_grad()` every iteration because gradients accumulate.

## Evaluation

Add an accuracy helper that takes argmax over logit rows and compares to the
label, evaluated on a held-out batch (test set for the real path, held-out
synthetic batch for the offline path):

```rust
fn accuracy<const N: usize>(logits: &Tensor2D<N, 10>, labels: &[u8]) -> f32;
```

## Tests

Offline (always runs on `cargo +nightly test`):

- synthetic classification: final loss strictly below initial loss.
- synthetic classification: held-out accuracy above a conservative floor.

Real MNIST (feature-gated and/or `#[ignore]`):

- final loss strictly below initial loss.
- test accuracy above a conservative threshold (for example `> 0.90` for an MLP,
  chosen to avoid flakiness).

Use a fixed seed throughout. Avoid brittle exact-value assertions; assert robust
loss decrease and a reasonable accuracy floor, consistent with the existing
convergence-test policy.

## Verification

Default offline suite:

```text
cargo +nightly test
```

Real-MNIST capstone (network):

```text
cargo +nightly test --features datasets --test mnist_training -- --ignored
```

If an example is added:

```text
cargo +nightly run --features datasets --example mnist
```

## Done Criteria

- A `Sequential` MLP trains on MNIST through the standardized constructs.
- The real-data test asserts loss decrease and a reasonable accuracy floor.
- The offline synthetic test exercises the same pipeline and always passes on
  the default suite.
- The real-data path is opt-in (feature-gated and/or `#[ignore]`) and does not
  affect default `cargo +nightly test`.
- The end-to-end workflow matches the public API direction in the top-level
  plan.
