# Epoch 04: NN And Optim

## Goal

Add trainable parameters, a minimal module pattern, `Linear`, and SGD.

At the end of this epoch, a tiny linear regression example should train and show decreasing loss.

## Context

This epoch validates that tensors and autograd are usable for real learning loops.

Keep module abstractions simple. Do not build a complex `Sequential` before manual model structs are comfortable.

## Scope

In scope:

- `Parameter<S, E, B>`.
- Parameter gradient access and zeroing.
- A lightweight way to collect parameters from modules.
- `Linear<IN, OUT, E, B>`.
- `SGD` optimizer.
- Tiny training example/test.

Out of scope:

- Adam.
- Momentum.
- Complex `Sequential` typing.
- Serialization.
- Dataset/data-loader API.
- GPU optimizers.

## Parameter

Parameters wrap tensors and default to `requires_grad = true`.

```rust
pub struct Parameter<S, E = f32, B = Cpu>
where
    S: Shape,
    E: FloatElement,
    B: Backend<E>,
{
    tensor: Tensor<S, E, B>,
}
```

Suggested API:

```rust
impl<S, E, B> Parameter<S, E, B>
where
    S: Shape,
    E: FloatElement,
    B: Backend<E>,
{
    pub fn new(tensor: Tensor<S, E, B>) -> Self;
    pub fn tensor(&self) -> &Tensor<S, E, B>;
    pub fn tensor_mut(&mut self) -> &mut Tensor<S, E, B>;
    pub fn zero_grad(&self);
}
```

If `tensor_mut` conflicts with the internal `Arc` storage design, expose narrower mutation APIs instead. Avoid broad uncontrolled tensor mutation.

## Parameter Collection

Start with a simple trait:

```rust
pub trait Module<E: FloatElement, B: Backend<E>> {
    fn parameters_mut(&mut self) -> Vec<&mut dyn OptimParameter<E, B>>;
}
```

Keep optimizer parameter erasure data-only so adding optimizers does not require changing parameter implementors:

```rust
pub trait OptimParameter<E: FloatElement, B: Backend<E>> {
    fn param_id(&self) -> usize;
    fn values(&self) -> Vec<E>;
    fn grad_values(&self) -> Option<Vec<E>>;
    fn set_values(&mut self, values: Vec<E>);
    fn zero_grad(&self);
}
```

Optimizer-specific update rules should live in optimizer implementations, not on `OptimParameter`.
This allows SGD, momentum, Adam, and later optimizers to share the same parameter interface.

## Linear Module

Implement:

```rust
pub struct Linear<const IN: usize, const OUT: usize, E = f32, B = Cpu>
where
    E: FloatElement,
    B: Backend<E>,
{
    weight: Parameter<D2<IN, OUT>, E, B>,
    bias: Parameter<D1<OUT>, E, B>,
}
```

Forward:

```rust
impl<const IN: usize, const OUT: usize, E, B> Linear<IN, OUT, E, B>
where
    E: FloatElement,
    B: Backend<E>,
{
    pub fn forward<const BATCH: usize>(
        &self,
        x: &Tensor2D<BATCH, IN, E, B>,
    ) -> Tensor2D<BATCH, OUT, E, B>
    where
        [(); BATCH * IN]:,
        [(); IN * OUT]:,
        [(); BATCH * OUT]:,
    {
        x.matmul(self.weight.tensor()).add_row(self.bias.tensor())
    }
}
```

Initialization can be simple at first:

- weights from a small deterministic range or zeros for initial tests.
- bias zeros.

Add random initialization later.

## SGD

Minimal optimizer:

```rust
pub struct SGD<E = f32> {
    lr: E,
}
```

Step rule:

```text
param = param - lr * grad
```

Optimizer updates must happen under `no_grad`:

```rust
pub fn step<P>(&mut self, params: P)
where
    P: IntoIterator<Item = &mut dyn OptimParameter<E, B>>
{
    let _guard = no_grad();
    // update params
}
```

For this CPU-first epoch, optimizers may operate on flat `Vec<E>` values through `OptimParameter`.
Backend-storage-based optimizer updates can replace this later when GPU backends need accelerated optimizer kernels.

## Mutation Rule

Do not add general-purpose in-place tensor ops for users in this epoch.

Only optimizer internals should mutate parameter storage.

If mutation requires `Arc::make_mut`, verify that graph sharing semantics remain correct. It may be cleaner to put mutable parameter data behind a controlled lock or interior mutability primitive.

## Training Test

Add a simple regression test:

```text
y = x * w + b
```

Use a tiny fixed batch and known target.

Check that loss decreases after several steps.

Do not require exact convergence in the first test. Check a robust decrease.

## Tests

Add tests for:

- `Parameter::new` sets `requires_grad = true`.
- `Linear` output shape is `Tensor2D<BATCH, OUT>`.
- `SGD` updates parameter values.
- `SGD` does not create graph nodes.
- `zero_grad` works through parameters/modules.
- tiny regression loss decreases.

## Verification

Run:

```text
cargo +nightly test
```

## Done Criteria

- Parameters exist and default to requiring gradients.
- `Linear` works with const-generic input/output dimensions.
- SGD can update parameters safely.
- Optimizer updates run without graph construction.
- A tiny training loop decreases loss.
