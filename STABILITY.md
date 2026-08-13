# Stability Guarantee

`rstorch` 1.0 follows [semantic versioning](https://semver.org). This file
states what that promise covers. The supported feature and target combinations
are the ones the manifests and CI can build; a feature-gated item is covered
when its feature is enabled on a supported target.

## Covered Surface

Everything reachable from `rstorch::` is covered, except the
`rstorch::testing` module described below. This includes the dynamic API, the
optional `typed`, `hub`, `rayon`, `metal`, and `wgpu` features, and the public
derive macros. Removing an item, narrowing a signature, or moving a covered
item to another module is a breaking change.

This is enforced by review and semantic versioning discipline rather than a
mechanical baseline diff; an intentional public change is called out in the
same change that makes it.

The following are also covered contracts:

- `Error` is `#[non_exhaustive]`, so adding a variant is a minor-release
  change. Removing a variant or changing the documented error category for an
  existing invalid input is breaking.
- `Grads` is not `Clone`, and `Tensor::backward` returns a single linear value.
  Reusing a moved value is a compiler error; ignoring it emits the normal
  `#[must_use]` warning, which CI promotes to an error in the relevant suites.
- `Device` has no public backend trait. CPU is the reference implementation;
  Metal and WGPU are validated against it where their capability contracts
  apply. Unsupported operations return `Error::Unsupported` rather than
  silently copying data to the host.

## Backend And Dtype Scope

The six dtypes are part of the CPU tensor and persistence surface:
`F16`, `BF16`, `F32`, `F64`, `I64`, and `Bool`. Operation support is
intentional rather than universal: for example, losses and normalization are
float-only, while indices are `I64`. A backend may reject a dtype/op pair
with `Error::Unsupported`; it must not silently promote or reinterpret it.

| Backend | Current scope |
|---|---|
| CPU | Reference kernels over all six dtypes, subject to each operation's documented contract. |
| Metal | F16, F32, I64, and Bool storage with operation-specific kernel support. BF16 and F64 storage are unsupported. |
| WGPU | F32 compute plus lossless I64 and Bool storage for the operations that support them. Native F16 is available when the adapter exposes `SHADER_F16`; BF16 and F64 storage are unsupported. |

`Device::best_available` selects the first Metal device on macOS when the
`metal` feature is enabled, then the first native WGPU adapter when `wgpu` is
enabled, and otherwise CPU. Hardware availability is not guaranteed by the
library. `to_device` is the explicit way to move a tensor between devices;
missing operation kernels do not invoke it implicitly.

## Persistence

Persistence provides versioned envelopes, reader limits, atomic replacement,
schema validation, and transactional model/optimizer loading. The tensor data
is not authenticated and carries no checksum: bit-level corruption can produce
a valid file with different values. Applications that load untrusted or
transported checkpoints should add an external integrity or authenticity check.

## What Is Not Covered

- `rstorch::testing`, behind the `testing` feature, is the finite-difference
  harness used by the crate's own tests. Its signature follows test needs and
  may change without a semver event.
- Floating-point results are not guaranteed to match bit-for-bit across
  backends. The contract is deterministic execution and documented numerical
  tolerances, not a particular instruction order.
- Backend performance, benchmark numbers, device availability, and the order
  in which hardware vendors report devices are not release guarantees.
- Private backends, layouts, storage, and dispatch internals may change freely.

Features are additive: enabling a feature does not remove or change an item
that exists without it. The `typed` API is a wrapper over the dynamic `Tensor`
and owns no separate kernels.

## Derive Macros

The derive macros are loud by default: unrecognized fields are treated as
child modules and fail to compile when they do not implement the appropriate
`Module` trait. The dynamic derive's `#[module(skip)]` is an explicit escape
hatch and can intentionally omit any field, including a `Param`; an omitted
parameter is not visited by the optimizer or state-dict utilities. The typed
derive rejects `skip` on typed leaves and on `Option`/`Vec` containers.

## Metal And WGPU Performance

The Metal and WGPU APIs are covered when their features are enabled, but their
speed is not. CPU remains the reference for correctness, and backend
optimizations may change performance in any release. The automatic device
selection policy above is separate from any benchmark result.

## Minimum Supported Rust Version

The MSRV is **1.88**, and CI enforces it. Raising the MSRV is a minor-version
change and is recorded in the changelog.

## Reporting A Break

If a minor or patch release breaks a covered build, please open an issue. The
recorded API surfaces make the answer checkable rather than a matter of memory.
