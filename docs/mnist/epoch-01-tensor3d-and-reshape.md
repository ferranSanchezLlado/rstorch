# Epoch 01: Rank-3 Tensor And Reshape

## Goal

Add a `Tensor3D` type and a typed `reshape` so batched, multi-axis data can be
represented and converted into the matrix form `Linear` consumes, without
abandoning compile-time shape safety.

## Context

The tensor core currently has `Scalar`, `Tensor1D`, and `Tensor2D` with shape
markers `D0`, `D1`, `D2` (`src/shape.rs`). `Tensor2D` already exposes `flatten`
using `generic_const_exprs` (`src/tensor/ops.rs`), which is the precedent for
shape arithmetic at the API boundary.

MNIST images are naturally `28 x 28`. Representing a batch as
`Tensor3D<BATCH, 28, 28>` and flattening to `Tensor2D<BATCH, 784>` is cleaner
and more PyTorch-like than building the flat matrix by hand, and it gives later
work (datasets, examples) a typed multi-axis input to target.

## Scope

In scope:

- A `D3<const A, const B, const C>` shape marker implementing `Shape`.
- A `Tensor3D` alias in the tensor module and prelude.
- Construction (`zeros`, `ones`, `from_vec`), `to_vec`, and `shape` for
  `Tensor3D`.
- A typed `reshape`/`reshape_2d` with a compile-time element-count guard.
- A `flatten_2d()` convenience for the common batched-input path,
  `Tensor3D<BATCH, H, W>` to `Tensor2D<BATCH, H * W>`.
- Autograd support for reshape and flattening (identity gradient, preserves
  element order).

Out of scope:

- `Tensor4D` or a generic rank-N tensor.
- Permutation, transpose, or axis-aware reductions on rank-3 tensors.
- Broadcasting.
- Backend kernels beyond what reshape needs (reshape is a metadata/layout
  operation on CPU storage).

## Shape Marker

Add to `src/shape.rs`, mirroring `D2`:

```rust
pub struct D3<const A: usize, const B: usize, const C: usize>(/* PhantomData */);

impl<const A: usize, const B: usize, const C: usize> Shape for D3<A, B, C> {
    const RANK: usize = 3;
    const NUMEL: usize = A * B * C;
    fn dims() -> &'static [usize] { /* &[A, B, C] */ }
}
```

## Tensor Alias

Add to `src/tensor/mod.rs` and export through the prelude:

```rust
pub type Tensor3D<const A: usize, const B: usize, const C: usize, E = f32, B2 = Cpu>
    = Tensor<D3<A, B, C>, E, B2>;
```

Provide `from_vec` returning `Result<_, TensorError>` consistent with the other
constructors, validating length `A * B * C`.

## Reshape

Add reshape from rank-3 to rank-2 with element-count guards:

```rust
pub fn reshape_2d<const M: usize, const N: usize>(&self) -> Tensor2D<M, N, E, B>
where
    [(); A * B * C]:,
    [(); M * N]:;
```

Rules:

- Total element count must match; encode this in the bound or assert it with a
  clear message backed by the const generics.
- Element order is preserved (row-major flatten then regroup).
- Backward pass is the identity reshape of the incoming gradient back to the
  source shape, so gradients flow through unchanged.
- No dtype or backend reinterpretation.

A general rank-to-rank reshape is out of scope; `reshape_2d` plus
`flatten_2d()` is sufficient for the MNIST path.

## Flatten Convenience

Add a convenience method for batched rank-3 inputs:

```rust
pub fn flatten_2d(&self) -> Tensor2D<A, { B * C }, E, BK>
where
    [(); B * C]:;
```

This preserves the first dimension as the batch dimension, avoids repeating the
batch size at call sites, and shares the same value-ordering and autograd rules
as `reshape_2d`.

## Tests

Unit tests:

- `D3` metadata: `RANK`, `NUMEL`, `dims`.
- `Tensor3D` construction and `to_vec` round-trip.
- `from_vec` rejects wrong lengths.
- `reshape_2d` preserves values and order for a known small tensor.
- `flatten_2d` preserves the batch dimension, values, and order for a known
  small tensor.
- reshape/flatten gradient flows: building a `Tensor3D` leaf, reshaping or
  flattening, summing, and calling `backward()` yields the expected gradient
  shape and values.

Compile-fail tests with `trybuild`:

- `reshape_2d` to a shape whose element count differs from the source.

## Verification

```text
cargo +nightly test
```

## Done Criteria

- `Tensor3D` exists, is constructible, and is exported through the prelude.
- `reshape_2d` converts `Tensor3D<A, B, C>` to `Tensor2D<M, N>` only when
  `A * B * C == M * N`.
- `flatten_2d` converts `Tensor3D<BATCH, H, W>` to `Tensor2D<BATCH, H * W>`
  without requiring the caller to repeat `BATCH`.
- Reshape and flattening participate correctly in autograd.
- Element-count violations fail to compile.
- The default test suite passes with no new dependencies.
