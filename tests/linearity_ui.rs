//! Compile-fail tests for the linear behavior of [`Grads`](rstorch::Grads).
//!
//! The cases cover `Grads: !Clone`, use-after-move, and the `#[must_use]`
//! warning. Each source file under `tests/ui/default/autograd/linearity/` must
//! fail with the matching `.stderr` fixture.
//!
//! `trybuild` diagnostics vary by compiler version, so this suite runs only
//! with `RSTORCH_UI=1` on the pinned Rust 1.88 toolchain.

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
