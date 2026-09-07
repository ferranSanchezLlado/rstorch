use rstorch::{DType, Device, Tensor};
#[cfg(all(feature = "metal", target_os = "macos"))]
#[path = "common/metal.rs"]
mod metal_gpu;

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[path = "common/wgpu.rs"]
mod wgpu_gpu;

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[path = "common/cuda.rs"]
mod cuda_gpu;

fn bits(tensor: &Tensor) -> Vec<u32> {
    tensor
        .to_vec::<f32>()
        .expect("host read")
        .into_iter()
        .map(f32::to_bits)
        .collect()
}

#[test]
fn cpu_chain_matches_eager_bit_for_bit() {
    let input = Tensor::from_vec(
        (0..4096)
            .map(|index| (index as f32 - 2048.0) / 127.0)
            .collect(),
        [4096],
        &Device::Cpu,
    )
    .expect("input");

    let eager = {
        let _guard = rstorch::lazy::set_fusion(false);
        input
            .mul_scalar(1.5)
            .and_then(|value| value.add_scalar(-2.0))
            .and_then(|value| value.relu())
            .expect("eager chain")
    };
    let fused = {
        let _guard = rstorch::lazy::set_fusion(true);
        input
            .mul_scalar(1.5)
            .and_then(|value| value.add_scalar(-2.0))
            .and_then(|value| value.relu())
            .expect("deferred chain")
    };
    assert_eq!(eager.dtype(), DType::F32);
    assert_eq!(bits(&eager), bits(&fused));
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_deferred_chain_matches_when_hardware_is_available() {
    if !metal_gpu::available() {
        return;
    }
    parity_on_device(Device::Metal(0));
}

#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
#[test]
fn wgpu_deferred_chain_matches_when_hardware_is_available() {
    if !wgpu_gpu::available() {
        return;
    }
    parity_on_device(Device::Wgpu(0));
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
fn cuda_deferred_chain_matches_when_hardware_is_available() {
    if !cuda_gpu::available() {
        return;
    }
    parity_on_device(Device::Cuda(0));
}

#[cfg(any(
    all(feature = "metal", target_os = "macos"),
    all(feature = "wgpu", not(target_arch = "wasm32")),
    all(feature = "cuda", any(target_os = "linux", target_os = "windows"))
))]
fn parity_on_device(device: Device) {
    let input = Tensor::from_vec(
        (0..1024).map(|index| index as f32 / 31.0 - 16.0).collect(),
        [1024],
        &device,
    )
    .expect("backend input");
    let build = || {
        input
            .add_scalar(1.0)
            .and_then(|value| value.mul_scalar(0.5))
            .and_then(|value| value.neg())
            .expect("backend chain")
    };
    let eager = {
        let _guard = rstorch::lazy::set_fusion(false);
        build()
    };
    let deferred = {
        let _guard = rstorch::lazy::set_fusion(true);
        build()
    };
    assert_eq!(bits(&eager), bits(&deferred));
}
