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
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features
cargo +1.88 check --workspace --all-features          # MSRV
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

```sh
cargo publish -p rstorch-derive
cargo publish -p rstorch          # only after the derive is live on the index
git tag -a v1.0.0 -m "rstorch 1.0.0"
git push origin v1.0.0
```

`fixtures` is `publish = false`: it is the downstream acceptance crate, and it
depends on the public API by path precisely so it cannot be published against.

## After publishing

- [ ] docs.rs built both targets. The `[package.metadata.docs.rs]` block asks
      for all features plus `aarch64-apple-darwin`, because `metal` compiles
      only on macOS and would otherwise be missing from the rendered docs.
- [ ] Open the next `unreleased` section in `CHANGELOG.md`.

## Raising the MSRV

An MSRV bump is a minor release and needs three edits in step: the
`rust-version` in `[workspace.package]`, the toolchain matrix in
`.github/workflows/ci.yml` (including the pinned lanes the UI suites use), and
the number quoted in `STABILITY.md` and `README.md`.
