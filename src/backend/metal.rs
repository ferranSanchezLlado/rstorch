use super::{Backend, sealed};
use crate::dtype::{DType, f16};
use std::borrow::Cow;
use std::collections::HashMap;
use std::error;
use std::ffi::c_void;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

use ::metal as metal_rs;
use metal_rs::objc::rc::autoreleasepool;

const SHADERS_F32: &str = include_str!("kernels/metal_f32.metal");
const SHADERS_F16: &str = include_str!("kernels/metal_f16.metal");

#[derive(Debug, Clone, Copy, Default)]
pub struct Metal;

#[derive(Clone)]
pub struct MetalDevice {
    raw: Arc<metal_rs::Device>,
    queue: Arc<metal_rs::CommandQueue>,
    pipelines: Arc<Mutex<HashMap<&'static str, Arc<metal_rs::ComputePipelineState>>>>,
    registry_id: u64,
}

#[derive(Clone)]
pub struct MetalStorage {
    buffer: Arc<metal_rs::Buffer>,
    len: usize,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum MetalError {
    NoDevice,
    LibraryCompile(String),
    Pipeline(String),
    Command(String),
    LengthMismatch {
        expected: usize,
        found: usize,
    },
    BadMatmulDims {
        m: usize,
        k: usize,
        n: usize,
        lhs_len: usize,
        rhs_len: usize,
    },
}

unsafe impl Send for MetalDevice {}
unsafe impl Sync for MetalDevice {}
unsafe impl Send for MetalStorage {}
unsafe impl Sync for MetalStorage {}

impl fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalDevice")
            .field("registry_id", &self.registry_id)
            .finish()
    }
}

impl PartialEq for MetalDevice {
    fn eq(&self, other: &Self) -> bool {
        self.registry_id == other.registry_id
    }
}

impl fmt::Debug for MetalStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetalStorage")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for MetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDevice => write!(f, "no Metal device is available"),
            Self::LibraryCompile(message) => write!(f, "Metal library compile error: {message}"),
            Self::Pipeline(message) => write!(f, "Metal pipeline error: {message}"),
            Self::Command(message) => write!(f, "Metal command error: {message}"),
            Self::LengthMismatch { expected, found } => {
                write!(f, "length mismatch: expected {expected}, found {found}")
            }
            Self::BadMatmulDims {
                m,
                k,
                n,
                lhs_len,
                rhs_len,
            } => write!(
                f,
                "bad matmul dims ({m}, {k}, {n}) for operand lengths {lhs_len} and {rhs_len}"
            ),
        }
    }
}

impl error::Error for MetalError {}

impl sealed::SealedBackend for Metal {}

trait MetalDType: DType {
    const SHADERS: &'static str;
    const BINARY_KERNEL: &'static str;
    const SCALAR_KERNEL: &'static str;
    const MATMUL_KERNEL: &'static str;
    const SUM_KERNEL: &'static str;
}

impl MetalDType for f32 {
    const SHADERS: &'static str = SHADERS_F32;
    const BINARY_KERNEL: &'static str = "binary_f32_kernel";
    const SCALAR_KERNEL: &'static str = "scalar_f32_kernel";
    const MATMUL_KERNEL: &'static str = "matmul_f32_kernel";
    const SUM_KERNEL: &'static str = "sum_f32_kernel";
}

impl MetalDType for f16 {
    const SHADERS: &'static str = SHADERS_F16;
    const BINARY_KERNEL: &'static str = "binary_f16_kernel";
    const SCALAR_KERNEL: &'static str = "scalar_f16_kernel";
    const MATMUL_KERNEL: &'static str = "matmul_f16_kernel";
    const SUM_KERNEL: &'static str = "sum_f16_kernel";
}

impl<E> Backend<E> for Metal
where
    E: MetalDType,
{
    type Device = MetalDevice;
    type Storage = MetalStorage;
    type Error = MetalError;

    fn default_device() -> std::result::Result<Self::Device, Self::Error> {
        static DEVICE: OnceLock<std::result::Result<MetalDevice, MetalError>> = OnceLock::new();
        DEVICE.get_or_init(create_default_device).clone()
    }

    fn zeros(device: &Self::Device, len: usize) -> std::result::Result<Self::Storage, Self::Error> {
        Self::from_vec(device, vec![E::ZERO; len])
    }

    fn ones(device: &Self::Device, len: usize) -> std::result::Result<Self::Storage, Self::Error> {
        Self::from_vec(device, vec![E::ONE; len])
    }

    fn from_vec(
        device: &Self::Device,
        data: Vec<E>,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        let len = data.len();
        let byte_len = buffer_byte_len::<E>(len);
        let buffer = if len == 0 {
            device
                .raw
                .new_buffer(byte_len, metal_rs::MTLResourceOptions::StorageModeShared)
        } else {
            device.raw.new_buffer_with_data(
                data.as_ptr().cast::<c_void>(),
                byte_len,
                metal_rs::MTLResourceOptions::StorageModeShared,
            )
        };
        Ok(MetalStorage {
            buffer: Arc::new(buffer),
            len,
        })
    }

    fn to_vec(
        _device: &Self::Device,
        storage: &Self::Storage,
    ) -> std::result::Result<Vec<E>, Self::Error> {
        if storage.len == 0 {
            return Ok(Vec::new());
        }

        unsafe {
            let ptr = storage.buffer.contents().cast::<E>();
            Ok(std::slice::from_raw_parts(ptr, storage.len).to_vec())
        }
    }

    fn host_access<'a>(
        device: &Self::Device,
        storage: &'a Self::Storage,
    ) -> std::result::Result<Cow<'a, [E]>, Self::Error> {
        Self::to_vec(device, storage).map(Cow::Owned)
    }

    fn storage_len(storage: &Self::Storage) -> usize {
        storage.len
    }

    fn matmul(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        m: usize,
        k: usize,
        n: usize,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        if lhs.len != m.saturating_mul(k) || rhs.len != k.saturating_mul(n) {
            return Err(MetalError::BadMatmulDims {
                m,
                k,
                n,
                lhs_len: lhs.len,
                rhs_len: rhs.len,
            });
        }

        let len = m.saturating_mul(n);
        let output = empty_storage::<E>(device, len);
        if len == 0 {
            return Ok(output);
        }

        let pipeline = pipeline::<E>(device, E::MATMUL_KERNEL)?;
        let m = checked_u32(m, "m")?;
        let k = checked_u32(k, "k")?;
        let n = checked_u32(n, "n")?;
        let total = checked_u32(len, "matmul output length")?;
        encode_and_wait(device, &pipeline, total as usize, |encoder| {
            encoder.set_buffer(0, Some(&lhs.buffer), 0);
            encoder.set_buffer(1, Some(&rhs.buffer), 0);
            encoder.set_buffer(2, Some(&output.buffer), 0);
            set_u32(encoder, 3, m);
            set_u32(encoder, 4, k);
            set_u32(encoder, 5, n);
        })?;
        Ok(output)
    }

    fn add(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        binary::<E>(device, lhs, rhs, len, 0)
    }

    fn sub(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        binary::<E>(device, lhs, rhs, len, 1)
    }

    fn mul(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        binary::<E>(device, lhs, rhs, len, 2)
    }

    fn div(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        binary::<E>(device, lhs, rhs, len, 3)
    }

    fn add_scalar(
        device: &Self::Device,
        input: &Self::Storage,
        rhs: E,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        scalar(device, input, rhs, len, 0)
    }

    fn sub_scalar(
        device: &Self::Device,
        input: &Self::Storage,
        rhs: E,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        scalar(device, input, rhs, len, 1)
    }

    fn mul_scalar(
        device: &Self::Device,
        input: &Self::Storage,
        rhs: E,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        scalar(device, input, rhs, len, 2)
    }

    fn div_scalar(
        device: &Self::Device,
        input: &Self::Storage,
        rhs: E,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        scalar(device, input, rhs, len, 3)
    }

    fn sum(
        device: &Self::Device,
        input: &Self::Storage,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        ensure_len(input.len, len)?;
        if len == 0 {
            return Self::from_vec(device, vec![E::ZERO]);
        }

        let output = empty_storage::<E>(device, 1);
        let pipeline = pipeline::<E>(device, E::SUM_KERNEL)?;
        let len_u32 = checked_u32(len, "sum length")?;
        encode_and_wait(device, &pipeline, 1, |encoder| {
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&output.buffer), 0);
            set_u32(encoder, 2, len_u32);
        })?;
        Ok(output)
    }
}

fn binary<E: MetalDType>(
    device: &MetalDevice,
    lhs: &MetalStorage,
    rhs: &MetalStorage,
    len: usize,
    op: u32,
) -> std::result::Result<MetalStorage, MetalError> {
    ensure_len(lhs.len, len)?;
    ensure_len(rhs.len, len)?;
    let output = empty_storage::<E>(device, len);
    if len == 0 {
        return Ok(output);
    }

    let pipeline = pipeline::<E>(device, E::BINARY_KERNEL)?;
    let len_u32 = checked_u32(len, "binary length")?;
    encode_and_wait(device, &pipeline, len, |encoder| {
        encoder.set_buffer(0, Some(&lhs.buffer), 0);
        encoder.set_buffer(1, Some(&rhs.buffer), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_u32(encoder, 3, op);
        set_u32(encoder, 4, len_u32);
    })?;
    Ok(output)
}

fn scalar<E: MetalDType>(
    device: &MetalDevice,
    input: &MetalStorage,
    rhs: E,
    len: usize,
    op: u32,
) -> std::result::Result<MetalStorage, MetalError> {
    ensure_len(input.len, len)?;
    let output = empty_storage::<E>(device, len);
    if len == 0 {
        return Ok(output);
    }

    let pipeline = pipeline::<E>(device, E::SCALAR_KERNEL)?;
    let len_u32 = checked_u32(len, "scalar length")?;
    encode_and_wait(device, &pipeline, len, |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&output.buffer), 0);
        set_value(encoder, 2, rhs);
        set_u32(encoder, 3, op);
        set_u32(encoder, 4, len_u32);
    })?;
    Ok(output)
}

fn create_default_device() -> std::result::Result<MetalDevice, MetalError> {
    let raw = metal_rs::Device::system_default().ok_or(MetalError::NoDevice)?;
    let registry_id = raw.registry_id();
    let queue = raw.new_command_queue();
    Ok(MetalDevice {
        raw: Arc::new(raw),
        queue: Arc::new(queue),
        pipelines: Arc::new(Mutex::new(HashMap::new())),
        registry_id,
    })
}

fn pipeline<E: MetalDType>(
    device: &MetalDevice,
    name: &'static str,
) -> std::result::Result<Arc<metal_rs::ComputePipelineState>, MetalError> {
    let mut cache = device.pipelines.lock().expect("pipeline cache poisoned");
    if let Some(pipeline) = cache.get(name) {
        return Ok(Arc::clone(pipeline));
    }

    let pipeline = autoreleasepool(|| {
        let options = metal_rs::CompileOptions::new();
        let library = device
            .raw
            .new_library_with_source(E::SHADERS, &options)
            .map_err(MetalError::LibraryCompile)?;
        let function = library
            .get_function(name, None)
            .map_err(MetalError::Pipeline)?;
        device
            .raw
            .new_compute_pipeline_state_with_function(&function)
            .map_err(MetalError::Pipeline)
    })?;
    let pipeline = Arc::new(pipeline);
    cache.insert(name, Arc::clone(&pipeline));
    Ok(pipeline)
}

fn encode_and_wait(
    device: &MetalDevice,
    pipeline: &metal_rs::ComputePipelineStateRef,
    len: usize,
    set_args: impl FnOnce(&metal_rs::ComputeCommandEncoderRef),
) -> std::result::Result<(), MetalError> {
    // Command buffers and encoders are autoreleased Objective-C objects; the
    // pool keeps sustained op streams (training loops) from accumulating them
    // until command submission fails.
    autoreleasepool(|| {
        let command_buffer = device.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        set_args(encoder);

        let width = pipeline
            .thread_execution_width()
            .min(pipeline.max_total_threads_per_threadgroup())
            .max(1);
        let groups = len.div_ceil(width as usize) as u64;
        encoder.dispatch_thread_groups(
            metal_rs::MTLSize::new(groups, 1, 1),
            metal_rs::MTLSize::new(width, 1, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        if command_buffer.status() == metal_rs::MTLCommandBufferStatus::Error {
            return Err(MetalError::Command("command buffer failed".to_string()));
        }
        Ok(())
    })
}

fn empty_storage<E>(device: &MetalDevice, len: usize) -> MetalStorage {
    let buffer = device.raw.new_buffer(
        buffer_byte_len::<E>(len),
        metal_rs::MTLResourceOptions::StorageModeShared,
    );
    MetalStorage {
        buffer: Arc::new(buffer),
        len,
    }
}

fn ensure_len(found: usize, expected: usize) -> std::result::Result<(), MetalError> {
    if found != expected {
        return Err(MetalError::LengthMismatch { expected, found });
    }
    Ok(())
}

fn checked_u32(value: usize, name: &'static str) -> std::result::Result<u32, MetalError> {
    u32::try_from(value).map_err(|_| MetalError::Command(format!("{name} exceeds u32::MAX")))
}

fn set_u32(encoder: &metal_rs::ComputeCommandEncoderRef, index: u64, value: u32) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<u32>() as u64,
        (&value as *const u32).cast::<c_void>(),
    );
}

fn set_value<E>(encoder: &metal_rs::ComputeCommandEncoderRef, index: u64, value: E) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<E>() as u64,
        (&value as *const E).cast::<c_void>(),
    );
}

fn buffer_byte_len<E>(len: usize) -> u64 {
    len.max(1).saturating_mul(std::mem::size_of::<E>()) as u64
}

#[cfg(test)]
mod tests {
    use super::{SHADERS_F16, SHADERS_F32};

    #[test]
    fn bundled_kernel_source_contains_required_entrypoints() {
        assert!(SHADERS_F32.contains("float apply_op_f32"));
        assert!(SHADERS_F32.contains("kernel void binary_f32_kernel"));
        assert!(SHADERS_F32.contains("kernel void scalar_f32_kernel"));
        assert!(SHADERS_F32.contains("kernel void matmul_f32_kernel"));
        assert!(SHADERS_F32.contains("kernel void sum_f32_kernel"));
        assert!(SHADERS_F16.contains("half apply_op_f16"));
        assert!(SHADERS_F16.contains("kernel void binary_f16_kernel"));
        assert!(SHADERS_F16.contains("kernel void scalar_f16_kernel"));
        assert!(SHADERS_F16.contains("kernel void matmul_f16_kernel"));
        assert!(SHADERS_F16.contains("kernel void sum_f16_kernel"));
    }
}
