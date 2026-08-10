//! Pinned-toolchain UI (compile-fail) suite for `#[derive(Module)]`.
//!
//! Each case in `tests/ui/default/nn/derive/*.rs` is a program that must
//! **fail** to compile, and its diagnostics must match the adjacent
//! `.stderr` fixture. Together they pin the *loud-by-default* rule at the
//! compiler level:
//!
//! - `non_module_field` — an unrecognized, non-`#[module(skip)]` field is a
//!   child module, so a non-`Module` type is a compile error.
//! - `type_alias_param` — a type alias for `Param` defeats the syntactic
//!   token match and falls to that same loud default.
//! - `unknown_module_option` — `#[module(...)]` accepts only `skip`.
//! - `derive_on_enum` — the derive is `struct`-only.
//!
//! # Why this suite is gated
//!
//! `trybuild` compares compiler diagnostics byte-for-byte against the
//! `.stderr` fixtures, and those diagnostics drift across `rustc` versions.
//! The fixtures are therefore **pinned to one toolchain** (regenerate with
//! `cargo +1.88 test -p rstorch-derive --test ui`, per
//! `docs/restart-v3/reference/restart-v2/epoch-10.1-compile-time-test-standardization.md`).
//! To keep `cargo test` green on any other toolchain, the run is opt-in: set
//! `RSTORCH_UI=1` to execute it. CI runs it on the pinned toolchain; the
//! `TRYBUILD=overwrite` convention regenerates fixtures there.

/// Run the compile-fail suite when `RSTORCH_UI=1` (pinned toolchain only).
///
/// Off by default so a mismatch caused by an unpinned `rustc` version does not
/// fail the ordinary `cargo test` gate.
#[test]
fn derive_module_ui() {
    if std::env::var_os("RSTORCH_UI").is_none() {
        eprintln!(
            "skipping derive UI suite; set RSTORCH_UI=1 on the pinned \
             toolchain (see this file's module docs) to run it"
        );
        return;
    }
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/default/nn/derive/*.rs");
}
