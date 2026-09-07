use rstorch::prelude::*;

pub(crate) fn available() -> bool {
    if std::env::var_os("RSTORCH_SKIP_CUDA_TESTS").is_some() {
        eprintln!("skipping CUDA hardware test: RSTORCH_SKIP_CUDA_TESTS is set");
        return false;
    }

    let device = Device::Cuda(0);
    Tensor::zeros([0], DType::F32, &device).unwrap_or_else(|error| {
        panic!(
            "CUDA feature enabled but device initialization failed: {error}. Set \
             RSTORCH_SKIP_CUDA_TESTS=1 only when this test environment intentionally has no CUDA device"
        )
    });
    true
}
