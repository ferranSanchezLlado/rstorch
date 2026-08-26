# Releasing

Two crates publish, and the order matters: `rstorch` depends on
`rstorch-derive` by an **exact matching version** (`=1.0.0` for this release),
so the derive must be available on crates.io before the root package can be
resolved or verified as a publishable package. The derive macro also resolves
renamed runtime dependencies, and the two crates must remain in exact
lockstep.

The tag workflow is authoritative: it reruns the release gates on the exact
commit named by the tag, checks that the tag version matches the workspace,
and publishes `rstorch-derive` before `rstorch`. The local checklist below is a
useful subset, not an exact copy of those tag-only gates.

## Before tagging

This is a local preflight subset of the tag workflow; running it locally first
just makes failures cheaper. Tag-only checks (including the exact tag/version
match, all release-gate jobs, and publication sequencing) remain authoritative.


```sh
cargo fmt --all --check
cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
cargo test --locked --workspace --all-features
cargo test --locked --workspace --no-default-features   # the CPU-only library
RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --no-deps --all-features
cargo +1.88 check --locked --workspace --all-features        # MSRV
cargo +1.88 test --locked --workspace --features testing     # MSRV, executed
cargo +1.88 test --locked --workspace --no-default-features  # MSRV, CPU-only
cargo deny check advisories licenses bans sources   # see deny.toml
```

```sh

# Every trybuild suite early-returns without this variable, so a run that omits
# it compiles the compile-fail fixtures and asserts nothing. The recorded
# diagnostics are pinned to the MSRV compiler, which is also what the nested
# cargo commands in `typed_ui` must use.
RSTORCH_UI=1 RUSTUP_TOOLCHAIN=1.88 cargo test --locked --workspace --all-features

# For local information only. The tag workflow independently checks that this
# manifest version matches the `v*` tag before running its release gates.
cargo metadata --no-deps --format-version 1 \
  | jq -r '.packages[] | select(.name == "rstorch") | .version'
```

Before creating the tag, finalize the changelog entry and confirm the release:

- [ ] `CHANGELOG.md` has an entry for this version, with a finalized heading
      rather than `unreleased`.
- [ ] The workspace version in `Cargo.toml` matches it, and
      `rstorch-derive = "=1.0.0"` (the exact matching derive version) does too.
- [ ] The tag workflow verifies and packages `rstorch-derive` first; only after
      that derive is available does it verify the root `rstorch` package.
- [ ] The macOS lanes are green: they are the only ones that build the Metal
      surface at all.

## Publishing

Pushing the tag *is* the release action. `release.yml` triggers on `v*`, reruns
the full tag-only gate on the exact commit the tag points at, verifies the
derive package first, and only then verifies/publishes the root package:
`rstorch-derive` first, `rstorch` second, because the second cannot resolve
until the first is on the index.

```sh
git tag -a v1.0.0 -m "rstorch 1.0.0"
git push origin v1.0.0
```

The verification jobs prevent publication before the publish job starts. The
publish job is intentionally two-stage: if the derive publish succeeds and the
root publish later fails, the derive version remains on crates.io and the
release must be resumed rather than rolled back.

### Manual fallback

For when the workflow cannot run at all — Actions unavailable, or no registry
token on the runner. The ordering is the same, and it is not advisory: the
second command cannot resolve until the first version is live.

```sh
cargo publish -p rstorch-derive
cargo publish -p rstorch          # only after the derive is live on the index
```

`fixtures` is `publish = false`: it is the downstream acceptance crate, and it
depends on the public API by path precisely so it cannot be published against.

## After publishing

- [ ] docs.rs built both targets. The `[package.metadata.docs.rs]` block asks
      for all features plus `aarch64-apple-darwin`, because `metal` compiles
      only on macOS and would otherwise be missing from the rendered docs.
- [ ] Open the next `unreleased` section in `CHANGELOG.md`.

## Raising the MSRV

An MSRV bump is a minor release and needs the toolchain references updated in
step: the `rust-version` in `[workspace.package]`, the toolchain matrix in
`.github/workflows/ci.yml` (including the pinned lanes the UI suites use), the
pinned lanes in `.github/workflows/release.yml` (`verify`'s MSRV check,
`verify-ui`, `verify-msrv` and `verify-ct20-boundary`), the commands and pinned
toolchain references in this runbook, and the number quoted in `STABILITY.md`
and `README.md`.
