use rstorch::prelude::*;

pub(crate) fn available() -> bool {
    if std::env::var_os("RSTORCH_SKIP_WGPU_TESTS").is_some() {
        eprintln!("skipping WGPU hardware test: RSTORCH_SKIP_WGPU_TESTS is set");
        return false;
    }

    let device = Device::Wgpu(0);
    Tensor::from_vec(vec![1.0f32], [1], &device)
        .and_then(|tensor| tensor.to_vec::<f32>())
        .unwrap_or_else(|error| {
            panic!(
                "WGPU feature enabled but adapter initialization failed: {error}. Set \
                 RSTORCH_SKIP_WGPU_TESTS=1 only when this test environment intentionally has no WGPU adapter"
            )
        });
    true
}
