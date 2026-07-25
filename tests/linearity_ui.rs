//! Pinned-toolchain UI (compile-fail) suite for the **linearity guarantees**
//! of [`Grads`](rstorch::Grads).
//!
//! Exploration §5 claims three bug classes are unrepresentable rather than
//! merely discouraged, and §6 keeps a small compile-fail suite alive for
//! exactly the claims that would otherwise regress silently. Each case in
//! `tests/ui/default/autograd/linearity/*.rs` is a program that must **fail**
//! to compile, with diagnostics matching its `.stderr` fixture:
//!
//! - `grads_are_not_clone` — "same gradients applied twice" starts with a
//!   duplicate, so `Grads: !Clone`.
//! - `grads_used_after_move` — every consumer (`merge`, and the optimizer's
//!   `step`) takes `Grads` by move, so a second use is a move error.
//! - `grads_must_be_used` — `#[must_use]`, so computing gradients and never
//!   applying them is a diagnostic, not a silent no-op training loop.
//!
//! # Why this suite is gated
//!
//! `trybuild` compares compiler diagnostics byte-for-byte, and those drift
//! across `rustc` versions, so the fixtures are **pinned to one toolchain**
//! (regenerate with `TRYBUILD=overwrite cargo +1.88 test --test linearity_ui`
//! with `RSTORCH_UI=1` set, matching the convention of the
//! `#[derive(Module)]` suite in `rstorch-derive/tests/ui.rs`). To keep
//! `cargo test` green on any other toolchain the run is opt-in: set
//! `RSTORCH_UI=1` to execute it.

/// Run the compile-fail suite when `RSTORCH_UI=1` (pinned toolchain only).
#[test]
fn grads_linearity_ui() {
    if std::env::var_os("RSTORCH_UI").is_none() {
        eprintln!(
            "skipping Grads linearity UI suite; set RSTORCH_UI=1 on the \
             pinned toolchain (see this file's module docs) to run it"
        );
        return;
    }
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/default/autograd/linearity/*.rs");
}
