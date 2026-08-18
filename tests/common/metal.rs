use rstorch::prelude::*;

pub(crate) fn available() -> bool {
    if std::env::var_os("RSTORCH_SKIP_METAL_TESTS").is_some() {
        eprintln!("skipping Metal hardware test: RSTORCH_SKIP_METAL_TESTS is set");
        return false;
    }

    let device = Device::Metal(0);
    Tensor::zeros([1], DType::F32, &device).unwrap_or_else(|error| {
        panic!(
            "Metal is enabled but device initialization failed: {error}. Set \
             RSTORCH_SKIP_METAL_TESTS=1 only when this test environment intentionally has no Metal device"
        )
    });
    true
}
