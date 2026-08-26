# Stability Guarantee

`rstorch` 1.0 follows [semantic versioning](https://semver.org). This file
states what that promise covers. The supported feature and target combinations
are the ones the manifests and CI can build; a feature-gated item is covered
when its feature is enabled on a supported target.

## Covered Surface

Everything reachable from `rstorch::` is covered, except the
`rstorch::testing` module and the explicitly experimental `rstorch::typed` and
`rstorch::lazy` namespaces described below. This includes the dynamic API, the
optional `hub`, `rayon`, `metal`, `cuda`, and `wgpu` features, and the public
derive macros. Removing an item, narrowing a signature, or moving a covered
item to another module is a breaking change.

That is a statement about *signatures*. CUDA's **behaviour** carries a weaker
claim than the other backends' because no CI runner executes its kernels; see
"Backend And Dtype Scope" below.

This is enforced by review and semantic versioning discipline rather than a
mechanical baseline diff; an intentional public change is called out in the
same change that makes it.

The following are also covered contracts:

- `Error` is `#[non_exhaustive]`, so adding a variant is a minor-release
  change. Every *struct variant* is `#[non_exhaustive]` as well, so adding a
  field to one is a minor-release change too. Downstream code consequently
  cannot name a variant with a struct literal, and a match on one needs a `..`
  rest pattern; the fifteen public constructors (`Error::invalid_arg`,
  `Error::shape_mismatch`, `Error::data_with`, …) are the supported spelling
  for building an `Error`. Removing a variant or a field, or changing the
  documented error category for an existing invalid input, is breaking.
- `Grads` is not `Clone`, and `Tensor::backward` returns a single linear value.
  Reusing a moved value is a compiler error; ignoring it emits the normal
  `#[must_use]` warning, which CI promotes to an error in the relevant suites.
- `Device` has no public backend trait. CPU is the reference implementation;
  Metal and WGPU are validated against it in CI where their capability
  contracts apply. CUDA is validated locally against the same reference, not
  in CI — see "Backend And Dtype Scope". Unsupported operations return
  `Error::Unsupported` rather than silently copying data to the host. Because
  that trait is private, a backend implemented outside this crate is not a
  supported extension point; see [Execution Model](#execution-model).
- `Mode`'s axes belong to the crate. Its fields are private and there is no
  public constructor — the only ways to name a mode are `Mode::TRAIN`,
  `Mode::EVAL`, `recorded()`, and `frozen()` — so **adding an axis is a
  minor-release change**, not a breaking one. An autocast dtype for mixed
  precision and a determinism flag are the plausible additions; neither is
  promised here. Code that needs to carry its own per-call information extends
  the *input* type (`Forward<Input>`), not `Mode`.

Public record-shaped API types, including persistence and hub records, use a
constructor/builder or a `#[non_exhaustive]` boundary rather than promising
exhaustive struct literals. They may gain fields in a minor release; callers
should use the documented constructors/builders. Public traits remain
intentional extension points: additions in 1.x use defaults or extension traits,
not new required methods.

The derive macros resolve the runtime crate through Cargo, including a renamed
dependency. The root crate pins `rstorch-derive` to the exact matching version;
that lockstep is part of the release contract.

## Execution Model

Operations are eager by default. The opt-in lazy executor for dense CPU
element-wise chains is also an implementation detail: it must produce the
same documented values and errors, while its schedule and intermediate
allocation are not part of the promise. Backends already batch — Metal encodes
dispatches into a command buffer and flushes on a threshold — so one call to a
named method is not guaranteed to be one kernel launch. Deferring or fusing
work *within* a step is deliberately reserved as a purely internal change.

Two larger things are out of scope, stated here rather than left ambiguous:

- **Trace-and-replay compilation** (capture a graph once, compile it, replay
  it) is out of scope for 1.x. Module state is ordinary owned state: `Dropout`
  steps an embedded `Rng` on every call and `BatchNorm2d` updates its running
  statistics as plain fields, so a captured graph is valid for exactly one
  step. Making it valid for more would mean threading module state through
  `forward` explicitly, which costs the property the design is built on — no
  interior mutability and no mutexes on the training path.
- **Third-party backends** are out of scope permanently. `BackendOps` is
  `pub(crate)` and stays that way, so the set of backends is the set tabulated
  below; adding one is a release of this crate, not an implementation of a
  public trait.

## Error Timing

*When* an error surfaces is a covered contract, separately from which variant
it is.

**Always at the call site.** `ShapeMismatch`, `RankMismatch`, `InvalidAxis`,
`DTypeMismatch`, `DeviceMismatch`, `ReshapeMismatch`, and `InvalidArg` are
decidable from tensor metadata — shapes, ranks, dtypes, devices, axes — without
executing anything, so the failing expression is the one that reports them.
This is what makes the `#[track_caller]` location on the operator sugar honest,
and it stays honest if execution is ever deferred. Moving one of these seven
off its call site is a breaking change.

**At or before the next materialization.** `Unsupported` and `Backend` come
from execution, and execution is batched, so a kernel failure may be observed
at a later step that forces completion — a `to_vec`, a device transfer, a save,
or an explicit `Device::synchronize`/`Tensor::realize` — rather than at the
operation that queued the work. Both variants carry the operation name, so the
failing operation is identified even when the reported source location belongs
to the later transfer. Reporting one of these later within the same step is not
a breaking change.

**`IndexOutOfBounds` is in both buckets, and which one depends on where the
index lives.** When the offending value is an argument — `narrow`'s range, an
axis length, a scalar index — it is metadata, and the first bucket's rule
applies. When it is an *element of an index tensor* (`index_select`, `gather`,
`index_add`, `scatter_add`), only executing the lookup can decide it. The CPU
backend reads the index buffer on the host and reports at the call site; Metal,
CUDA, and WGPU run a validation kernel and collect its verdict at the next host
boundary, so the error surfaces from the following `to_vec`/transfer/flush
instead. That is a deliberate trade — reading the verdict eagerly cost a full
GPU round trip per embedding lookup — and no incorrect value is observable in
the meantime, because the host cannot see a result without passing the check.
Which of the two a given backend does is not covered; that the error is raised
before any wrong value reaches the host is.

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
| CUDA (validated locally, not CI-gated) | Opt-in natively on Linux and Windows, including Linux under WSL. Compute capability 6.0 or newer; F16/F32 compute plus lossless I64 and Bool storage for the operations that support them. BF16 and F64 storage are unsupported. Kernels are bundled as PTX, so runtime use requires a compatible NVIDIA driver but not the CUDA toolkit. |
| WGPU | F32 compute plus lossless I64 and Bool storage for the operations that support them. Native F16 is available when the adapter exposes `SHADER_F16`; BF16 and F64 storage are unsupported. |

**CUDA's behavioural claim is weaker than the other backends', and is stated
here rather than left implicit:** CI compiles it (`--features cuda` builds on
every lane that enables it) but does not execute a kernel — there is no
CUDA-capable CI runner. CUDA is therefore validated locally against the CPU
reference, not gated, unlike Metal and WGPU, which run their kernels in CI.

The consequence is precise, and narrower than "CUDA is uncovered". Its **API**
is covered exactly like every other feature-gated item: removing
`Device::Cuda`, or narrowing a `cuda`-gated signature, is still a breaking
change. What is *not* promised is that a CUDA numerical or behavioural
regression will have been caught before release. Such a regression is handled
as a bug — fixed in a patch — rather than as the kind of documented-behaviour
break the rest of this document treats as semver-relevant. When an execution
lane exists, this paragraph goes and CUDA joins Metal and WGPU.

`Device::best_available` probes Metal ordinal 0 with complete initialization
when `metal` is enabled, then considers CUDA on Linux or Windows, the best
eligible WGPU adapter when `wgpu` is enabled, and otherwise CPU. Metal's probe
includes runtime shader compilation, command-queue creation, and
validation-buffer allocation, so a Metal ordinal that fails any of those steps
is skipped.
WGPU adapter type and backend are driver-reported classifications; automatic
selection excludes adapters reported as `Cpu`, while other software
classification is backend-dependent. WGPU ordinals are indices in the current
process's adapter set, not stable device identities, and must not be persisted.
Hardware and driver availability are not guaranteed by the library. `to_device`
is the explicit way to move a tensor between devices; missing operation
kernels do not invoke it implicitly.

## Persistence

Persistence provides a versioned `Envelope` container, reader limits, atomic
replacement, and schema validation. The envelope version covers the container;
semantic sections (such as `config`, `optimizer`, and `rng`) own independent
schemas and compatibility rules. The DecoderTransformer config section is
strict `version=1`: unknown, missing, duplicate, or unsupported fields/versions
are rejected, and a future reader must explicitly support any prior version it
accepts.

Generic model checkpoints contain model state. The DecoderTransformer convenience
checkpoint contains its config and model tensors only; optimizer, RNG,
application, cache, and tokenizer state are excluded. Dynamic model-state and
optimizer-state loads are independently transactional, but there is no combined
transaction spanning model, optimizer, RNG, caches, or application state.
Tensor data is not authenticated and carries no checksum: bit-level corruption
can produce a valid file with different values. Applications that load
untrusted or transported checkpoints should add an external integrity or
authenticity check.

## Dataset Hub

With `hub`, `DatasetHub::default_cache` resolves `RSTORCH_DATA`, then
`$HOME/.cache/rstorch`, then the project-local `data/` fallback. Public dataset
and resource path helpers accept only one ordinary path component and return an
error otherwise. Cache hits are re-verified against their declared size and
checksum before use; Tiny Shakespeare performs that verification before UTF-8
decoding. These checks are integrity checks, not cryptographic authenticity.

## Randomness

Randomness is explicit: caller-owned sampling constructors take `&mut Rng`,
and modules such as `Dropout` receive their own stream explicitly at
construction. There is no ambient, global, or thread-local generator. A
caller can capture a caller-owned generator with `Rng::state` and resume it
with `Rng::from_state`. `Dropout`'s private child stream is not persisted by
model state; exact training resume with dropout requires the caller to
reconstruct that model stream.
A given seed reproduces a given tensor for a fixed shape, dtype, **backend,
feature set, and build** — that tuple is the scope of the promise. Results are
not guaranteed to be reproducible across backends or across versions: whether
samples are drawn on the host and uploaded or generated on the device is an
implementation detail, and so is the order in which a tensor's elements
consume the stream.

## What Is Not Covered

- `rstorch::testing`, behind the `testing` feature, is the finite-difference
  harness used by the crate's own tests. Its signature follows test needs and
  may change without a semver event.
- The `bench-resnet` feature only enables the `resnet_mnist` benchmark target;
  it is benchmark-only and does not add library API or runtime guarantees.
- `rstorch::typed`, behind the `typed` feature, is an experimental second
  frontend over the dynamic tensor runtime. Its modules, traits, wrappers,
  placement bindings, and persistence helpers may change independently of the
  dynamic 1.x API.
- `rstorch::lazy`, including `Fusion`, `FusionGuard`, and its control
  functions, is experimental execution policy. Its thread-local controls,
  supported operation set, and realization schedule may change independently
  of the dynamic 1.x API.
- Floating-point results are not guaranteed to match bit-for-bit across
  backends, and no particular instruction order is promised. What is guaranteed
  is determinism **within a build**: the same input, the same crate version, the
  same backend, and the same feature set produce the same values, run after run.
  Kernel selection and fusion may change results within the documented
  numerical tolerances between minor versions.
- Fusion is an implementation detail, not a semver commitment: it may be
  enabled, disabled or changed in a minor release. The current executor is
  opt-in and defaults off while backend gates remain open.
- A backend that enables it must preserve eager bit patterns for the fused
  chain; the implementation detail remains free to change without a semver
  promise.
- Backend performance, benchmark numbers, device availability, and the order
  in which hardware vendors report devices are not release guarantees.
- Private backends, layouts, storage, and dispatch internals may change freely.

Features are additive for covered surfaces: enabling one does not remove or
change an item that exists without it. Experimental namespaces are exempt from
that guarantee as stated above. The `typed` API is a wrapper over the dynamic
`Tensor` and owns no separate kernels.

## Derive Macros

The derive macros are loud by default: unrecognized fields are treated as
child modules and fail to compile when they do not implement the appropriate
`Module` trait. They resolve the runtime crate, including renamed Cargo
dependencies, rather than assuming the dependency is spelled `rstorch`. The
dynamic derive's `#[module(skip)]` is an explicit escape hatch and can
intentionally omit any field, including a `Param`; an omitted parameter is not
visited by the optimizer or state-dict utilities. The typed derive rejects
`skip` on typed leaves and on `Option`/`Vec` containers.

## Accelerator Performance

The Metal, CUDA, and WGPU APIs are covered when their features are enabled,
but their speed is not. CPU remains the reference for correctness, and backend
optimizations may change performance in any release. The automatic device
selection policy above is separate from any benchmark result.

## Minimum Supported Rust Version

The MSRV is **1.88**, and CI enforces it. Raising the MSRV is a minor-version
change and is recorded in the changelog.

## Reporting A Break

If a minor or patch release breaks a covered build, please open an issue.
