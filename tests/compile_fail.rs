#[test]
fn compile_fail_shape_guarantees() {
    let t = trybuild::TestCases::new();

    #[cfg(not(feature = "hub"))]
    t.compile_fail("tests/ui/*_test_features_default.rs");

    #[cfg(feature = "hub")]
    t.compile_fail("tests/ui/*_test_features_*hub*.rs");
}
