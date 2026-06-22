#[test]
fn compile_fail_shape_guarantees() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
