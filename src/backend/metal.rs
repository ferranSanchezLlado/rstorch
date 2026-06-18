//! Metal backend for macOS.
//!
//! This module owns all Metal-specific buffer, command queue, and kernel launch
//! code so tensor and autograd layers stay backend-generic.

use crate::backend::Backend;
use metal as metal_rs;
use metal_rs::{CompileOptions, MTLResourceOptions, MTLSize};
use std::marker::PhantomData;
use std::mem::size_of;
use std::slice;
use std::sync::OnceLock;

const KERNELS: &str = include_str!("metal/kernels.metal");

/// Metal backend marker.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Metal;

/// Single Metal device and command queue used by tensors.
///
/// Multi-device execution is intentionally not implemented in this epoch.
#[derive(Clone)]
pub struct MetalDevice {
    device: metal_rs::Device,
    queue: metal_rs::CommandQueue,
    registry_id: u64,
}

/// Metal buffer-backed tensor storage.
#[derive(Clone, Debug)]
pub struct MetalStorage<E = f32> {
    buffer: metal_rs::Buffer,
    len: usize,
    device_id: u64,
    dtype: PhantomData<E>,
}

impl Metal {
    /// Returns whether the current machine exposes a default Metal device.
    pub fn is_available() -> bool {
        metal_rs::Device::system_default().is_some()
    }
}

impl MetalDevice {
    /// Returns the system default Metal device, or `None` when Metal is unavailable.
    pub fn system_default() -> Option<Self> {
        let device = metal_rs::Device::system_default()?;
        let queue = device.new_command_queue();
        let registry_id = device.registry_id();

        Some(Self {
            device,
            queue,
            registry_id,
        })
    }

    pub fn registry_id(&self) -> u64 {
        self.registry_id
    }
}

impl Backend<f32> for Metal {
    type Device = MetalDevice;
    type Storage = MetalStorage<f32>;

    fn default_device() -> Self::Device {
        static DEFAULT_DEVICE: OnceLock<MetalDevice> = OnceLock::new();

        DEFAULT_DEVICE
            .get_or_init(|| {
                MetalDevice::system_default()
                    .expect("Metal backend requested, but no default Metal device is available")
            })
            .clone()
    }

    fn zeros(device: &Self::Device, len: usize) -> Self::Storage {
        Self::from_vec(device, vec![0.0; len])
    }

    fn ones(device: &Self::Device, len: usize) -> Self::Storage {
        Self::from_vec(device, vec![1.0; len])
    }

    fn from_array<const N: usize>(device: &Self::Device, data: [f32; N]) -> Self::Storage {
        Self::from_vec(device, Vec::from(data))
    }

    fn from_vec(device: &Self::Device, data: Vec<f32>) -> Self::Storage {
        upload(device, &data)
    }

    fn to_vec(storage: &Self::Storage) -> Vec<f32> {
        let byte_len = byte_len(storage.len);
        assert_eq!(
            storage.buffer.length(),
            byte_len,
            "Metal storage byte length mismatch"
        );

        unsafe {
            // SAFETY: MetalStorage is created from f32 data with StorageModeShared.
            // The buffer is at least `len * size_of::<f32>()` bytes long, checked above.
            let ptr = storage.buffer.contents().cast::<f32>();
            slice::from_raw_parts(ptr, storage.len).to_vec()
        }
    }

    fn add(device: &Self::Device, lhs: &Self::Storage, rhs: &Self::Storage) -> Self::Storage {
        binary_kernel(device, lhs, rhs, "add_f32")
    }

    fn sub(device: &Self::Device, lhs: &Self::Storage, rhs: &Self::Storage) -> Self::Storage {
        binary_kernel(device, lhs, rhs, "sub_f32")
    }

    fn mul(device: &Self::Device, lhs: &Self::Storage, rhs: &Self::Storage) -> Self::Storage {
        binary_kernel(device, lhs, rhs, "mul_f32")
    }

    fn div(device: &Self::Device, lhs: &Self::Storage, rhs: &Self::Storage) -> Self::Storage {
        binary_kernel(device, lhs, rhs, "div_f32")
    }

    fn add_scalar(device: &Self::Device, lhs: &Self::Storage, rhs: f32) -> Self::Storage {
        scalar_kernel(device, lhs, rhs, "add_scalar_f32")
    }

    fn sub_scalar(device: &Self::Device, lhs: &Self::Storage, rhs: f32) -> Self::Storage {
        scalar_kernel(device, lhs, rhs, "sub_scalar_f32")
    }

    fn mul_scalar(device: &Self::Device, lhs: &Self::Storage, rhs: f32) -> Self::Storage {
        scalar_kernel(device, lhs, rhs, "mul_scalar_f32")
    }

    fn div_scalar(device: &Self::Device, lhs: &Self::Storage, rhs: f32) -> Self::Storage {
        scalar_kernel(device, lhs, rhs, "div_scalar_f32")
    }

    fn powf(device: &Self::Device, lhs: &Self::Storage, exponent: f32) -> Self::Storage {
        scalar_kernel(device, lhs, exponent, "powf_f32")
    }

    fn relu(device: &Self::Device, input: &Self::Storage) -> Self::Storage {
        unary_kernel(device, input, "relu_f32")
    }

    fn exp(device: &Self::Device, input: &Self::Storage) -> Self::Storage {
        unary_kernel(device, input, "exp_f32")
    }

    fn ln(device: &Self::Device, input: &Self::Storage) -> Self::Storage {
        unary_kernel(device, input, "ln_f32")
    }

    fn sum(device: &Self::Device, input: &Self::Storage) -> Self::Storage {
        reduction_kernel(device, input, "sum_f32")
    }

    fn mean(device: &Self::Device, input: &Self::Storage) -> Self::Storage {
        reduction_kernel(device, input, "mean_f32")
    }

    fn matmul(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        rows: usize,
        inner: usize,
        cols: usize,
    ) -> Self::Storage {
        assert_storage_on_device(device, lhs);
        assert_storage_on_device(device, rhs);
        assert_eq!(lhs.len, rows * inner, "Metal matmul lhs length mismatch");
        assert_eq!(rhs.len, inner * cols, "Metal matmul rhs length mismatch");

        let output_len = rows * cols;
        let output = empty_storage(device, output_len);
        let rows = checked_u32(rows, "matmul rows");
        let inner = checked_u32(inner, "matmul inner");
        let cols = checked_u32(cols, "matmul cols");
        let pipeline = pipeline(device, "matmul_f32");
        let command_buffer = device.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&lhs.buffer), 0);
        encoder.set_buffer(1, Some(&rhs.buffer), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        encoder.set_bytes(3, size_of::<u32>() as u64, (&rows as *const u32).cast());
        encoder.set_bytes(4, size_of::<u32>() as u64, (&inner as *const u32).cast());
        encoder.set_bytes(5, size_of::<u32>() as u64, (&cols as *const u32).cast());
        dispatch(encoder, &pipeline, output_len.max(1));
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        output
    }

    fn transpose(
        device: &Self::Device,
        input: &Self::Storage,
        rows: usize,
        cols: usize,
    ) -> Self::Storage {
        assert_storage_on_device(device, input);
        assert_eq!(input.len, rows * cols, "Metal transpose length mismatch");

        let output = empty_storage(device, input.len);
        let rows = checked_u32(rows, "transpose rows");
        let cols = checked_u32(cols, "transpose cols");
        let pipeline = pipeline(device, "transpose_f32");
        let command_buffer = device.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&output.buffer), 0);
        encoder.set_bytes(2, size_of::<u32>() as u64, (&rows as *const u32).cast());
        encoder.set_bytes(3, size_of::<u32>() as u64, (&cols as *const u32).cast());
        dispatch(encoder, &pipeline, input.len.max(1));
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        output
    }

    fn add_row(
        device: &Self::Device,
        input: &Self::Storage,
        row: &Self::Storage,
        rows: usize,
        cols: usize,
    ) -> Self::Storage {
        add_vector_kernel(device, input, row, rows, cols, "add_row_f32")
    }

    fn add_col(
        device: &Self::Device,
        input: &Self::Storage,
        col: &Self::Storage,
        rows: usize,
        cols: usize,
    ) -> Self::Storage {
        add_vector_kernel(device, input, col, rows, cols, "add_col_f32")
    }
}

fn upload(device: &MetalDevice, data: &[f32]) -> MetalStorage<f32> {
    let buffer = device.device.new_buffer_with_data(
        data.as_ptr().cast(),
        byte_len(data.len()),
        MTLResourceOptions::StorageModeShared,
    );

    MetalStorage {
        buffer,
        len: data.len(),
        device_id: device.registry_id,
        dtype: PhantomData,
    }
}

fn empty_storage(device: &MetalDevice, len: usize) -> MetalStorage<f32> {
    let buffer = device
        .device
        .new_buffer(byte_len(len), MTLResourceOptions::StorageModeShared);

    MetalStorage {
        buffer,
        len,
        device_id: device.registry_id,
        dtype: PhantomData,
    }
}

fn binary_kernel(
    device: &MetalDevice,
    lhs: &MetalStorage<f32>,
    rhs: &MetalStorage<f32>,
    name: &str,
) -> MetalStorage<f32> {
    assert_storage_on_device(device, lhs);
    assert_storage_on_device(device, rhs);
    assert_eq!(lhs.len, rhs.len, "Metal elementwise length mismatch");

    let output = empty_storage(device, lhs.len);
    let len = checked_u32(lhs.len, "elementwise length");
    let pipeline = pipeline(device, name);
    let command_buffer = device.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&lhs.buffer), 0);
    encoder.set_buffer(1, Some(&rhs.buffer), 0);
    encoder.set_buffer(2, Some(&output.buffer), 0);
    encoder.set_bytes(3, size_of::<u32>() as u64, (&len as *const u32).cast());
    dispatch(encoder, &pipeline, lhs.len.max(1));
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    output
}

fn unary_kernel(device: &MetalDevice, input: &MetalStorage<f32>, name: &str) -> MetalStorage<f32> {
    assert_storage_on_device(device, input);

    let output = empty_storage(device, input.len);
    let len = checked_u32(input.len, "unary length");
    let pipeline = pipeline(device, name);
    let command_buffer = device.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(&output.buffer), 0);
    encoder.set_bytes(2, size_of::<u32>() as u64, (&len as *const u32).cast());
    dispatch(encoder, &pipeline, input.len.max(1));
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    output
}

fn scalar_kernel(
    device: &MetalDevice,
    input: &MetalStorage<f32>,
    rhs: f32,
    name: &str,
) -> MetalStorage<f32> {
    assert_storage_on_device(device, input);

    let output = empty_storage(device, input.len);
    let len = checked_u32(input.len, "scalar length");
    let pipeline = pipeline(device, name);
    let command_buffer = device.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(&output.buffer), 0);
    encoder.set_bytes(2, size_of::<f32>() as u64, (&rhs as *const f32).cast());
    encoder.set_bytes(3, size_of::<u32>() as u64, (&len as *const u32).cast());
    dispatch(encoder, &pipeline, input.len.max(1));
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    output
}

fn reduction_kernel(
    device: &MetalDevice,
    input: &MetalStorage<f32>,
    name: &str,
) -> MetalStorage<f32> {
    assert_storage_on_device(device, input);

    let output = empty_storage(device, 1);
    let len = checked_u32(input.len, "reduction length");
    let pipeline = pipeline(device, name);
    let command_buffer = device.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(&output.buffer), 0);
    encoder.set_bytes(2, size_of::<u32>() as u64, (&len as *const u32).cast());
    dispatch(encoder, &pipeline, 1);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    output
}

fn add_vector_kernel(
    device: &MetalDevice,
    input: &MetalStorage<f32>,
    vector: &MetalStorage<f32>,
    rows: usize,
    cols: usize,
    name: &str,
) -> MetalStorage<f32> {
    assert_storage_on_device(device, input);
    assert_storage_on_device(device, vector);
    assert_eq!(
        input.len,
        rows * cols,
        "Metal broadcast input length mismatch"
    );

    let expected_vector_len = if name == "add_row_f32" { cols } else { rows };
    assert_eq!(
        vector.len, expected_vector_len,
        "Metal broadcast vector length mismatch"
    );

    let output = empty_storage(device, input.len);
    let rows = checked_u32(rows, "broadcast rows");
    let cols = checked_u32(cols, "broadcast cols");
    let pipeline = pipeline(device, name);
    let command_buffer = device.queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&input.buffer), 0);
    encoder.set_buffer(1, Some(&vector.buffer), 0);
    encoder.set_buffer(2, Some(&output.buffer), 0);
    encoder.set_bytes(3, size_of::<u32>() as u64, (&rows as *const u32).cast());
    encoder.set_bytes(4, size_of::<u32>() as u64, (&cols as *const u32).cast());
    dispatch(encoder, &pipeline, input.len.max(1));
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    output
}

fn pipeline(device: &MetalDevice, name: &str) -> metal_rs::ComputePipelineState {
    let options = CompileOptions::new();
    let library = device
        .device
        .new_library_with_source(KERNELS, &options)
        .unwrap_or_else(|error| panic!("failed to compile Metal kernels: {error}"));
    let function = library
        .get_function(name, None)
        .unwrap_or_else(|error| panic!("failed to load Metal kernel `{name}`: {error}"));

    device
        .device
        .new_compute_pipeline_state_with_function(&function)
        .unwrap_or_else(|error| panic!("failed to create Metal pipeline `{name}`: {error}"))
}

fn dispatch(
    encoder: &metal_rs::ComputeCommandEncoderRef,
    pipeline: &metal_rs::ComputePipelineStateRef,
    len: usize,
) {
    let len = u64::from(checked_u32(len, "dispatch length"));
    let threads = pipeline
        .thread_execution_width()
        .min(pipeline.max_total_threads_per_threadgroup())
        .max(1);

    encoder.dispatch_threads(MTLSize::new(len, 1, 1), MTLSize::new(threads, 1, 1));
}

fn assert_storage_on_device(device: &MetalDevice, storage: &MetalStorage<f32>) {
    assert_eq!(
        device.registry_id, storage.device_id,
        "Metal tensors from different devices cannot be used together"
    );
}

fn byte_len(len: usize) -> u64 {
    len.checked_mul(size_of::<f32>())
        .and_then(|bytes| bytes.try_into().ok())
        .expect("Metal buffer byte length overflow")
}

fn checked_u32(value: usize, name: &str) -> u32 {
    value
        .try_into()
        .unwrap_or_else(|_| panic!("Metal {name} exceeds u32::MAX"))
}
