# Releasing

Two crates publish, and the order matters: `rstorch` depends on
`rstorch-derive` **by version**, so `cargo publish -p rstorch` cannot resolve
until the matching derive is already on crates.io. This is also why CI's
packaging job verifies the derive fully but only lists the file set for
`rstorch` — the full verification is not available until the moment of
release.

## Before tagging

Everything here is what CI runs; running it locally first just makes the
failure cheaper.

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
cargo test --workspace --no-default-features        # the CPU-only library
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
cargo +1.88 check --workspace --all-features        # MSRV
cargo +1.88 test --workspace --features testing     # MSRV, executed
cargo deny check advisories licenses bans sources   # see deny.toml

# Every trybuild suite early-returns without this variable, so a run that omits
# it compiles the compile-fail fixtures and asserts nothing. The recorded
# diagnostics are pinned to the MSRV compiler, which is also what the nested
# cargo commands in `typed_ui` must use.
RSTORCH_UI=1 RUSTUP_TOOLCHAIN=1.88 cargo test --locked --workspace --all-features

# The tag is the version being published, so `release.yml` checks this before
# it runs anything else; disagreement means the whole gate verified the wrong
# number. Compare it against the tag you are about to push, minus the `v`.
cargo metadata --no-deps --format-version 1 \
  | jq -r '.packages[] | select(.name == "rstorch") | .version'
```

Then confirm the release itself:

- [ ] `CHANGELOG.md` has an entry for this version, and its heading is no
      longer `unreleased`.
- [ ] The workspace version in `Cargo.toml` matches it, and so does the
      `rstorch-derive` dependency requirement.
- [ ] `cargo package -p rstorch --list` contains no file the build does not
      need, and no directory that is only present in a working tree.
- [ ] The macOS lanes are green: they are the only ones that build the Metal
      surface at all.

## Publishing

Pushing the tag *is* the release action. `release.yml` triggers on `v*`, runs
the gate above again on the exact commit the tag points at, and only then
publishes: `rstorch-derive` first, `rstorch` second, because the second cannot
resolve until the first is on the index.

```sh
git tag -a v1.0.0 -m "rstorch 1.0.0"
git push origin v1.0.0
```

Nothing reaches crates.io if the gate fails, so a red run costs only the tag.

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

An MSRV bump is a minor release and needs four edits in step: the
`rust-version` in `[workspace.package]`, the toolchain matrix in
`.github/workflows/ci.yml` (including the pinned lanes the UI suites use), the
pinned `1.88` lanes in `.github/workflows/release.yml` (`verify`'s MSRV check,
`verify-ui`, `verify-msrv` and `verify-ct20-boundary`), and the number quoted in
`STABILITY.md` and `README.md`.
