use super::{Backend, sealed};
use crate::dtype::{DType, f16};
use std::borrow::Cow;
use std::collections::HashMap;
use std::error;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock, mpsc};

use wgpu as wgpu_rs;
use wgpu::util::DeviceExt;

const SHADERS_F32: &str = include_str!("kernels/wgpu_f32.wgsl");
const SHADERS_F16: &str = include_str!("kernels/wgpu_f16.wgsl");
const WORKGROUP_SIZE: u32 = 64;

#[derive(Debug, Clone, Copy, Default)]
pub struct Wgpu;

#[derive(Clone)]
pub struct WgpuDevice {
    raw: Arc<wgpu_rs::Device>,
    queue: Arc<wgpu_rs::Queue>,
    name: Arc<str>,
    features: wgpu_rs::Features,
    pipelines: Arc<WgpuPipelineCache>,
}

struct WgpuPipelineCache {
    layout: wgpu_rs::BindGroupLayout,
    pipelines: Mutex<HashMap<&'static str, Arc<wgpu_rs::ComputePipeline>>>,
}

#[derive(Clone)]
pub struct WgpuStorage {
    buffer: Arc<wgpu_rs::Buffer>,
    len: usize,
}

#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum WgpuError {
    NoAdapter,
    MissingFeature(&'static str),
    RequestDevice(String),
    BufferMap(String),
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
    SizeOverflow,
}

impl fmt::Debug for WgpuDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WgpuDevice")
            .field("name", &self.name)
            .finish()
    }
}

impl PartialEq for WgpuDevice {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.raw, &other.raw)
    }
}

impl fmt::Debug for WgpuStorage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WgpuStorage")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for WgpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter => write!(f, "no WGPU adapter is available"),
            Self::MissingFeature(feature) => write!(f, "WGPU adapter is missing {feature}"),
            Self::RequestDevice(message) => write!(f, "WGPU device request error: {message}"),
            Self::BufferMap(message) => write!(f, "WGPU buffer map error: {message}"),
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
            Self::SizeOverflow => write!(f, "WGPU buffer size overflow"),
        }
    }
}

impl error::Error for WgpuError {}

impl sealed::SealedBackend for Wgpu {}

trait WgpuDType: DType {
    const FEATURES: wgpu_rs::Features;
    const SHADERS: &'static str;
    const BINARY_KERNEL: &'static str;
    const SCALAR_KERNEL: &'static str;
    const MATMUL_KERNEL: &'static str;
    const SUM_KERNEL: &'static str;

    fn default_device() -> std::result::Result<WgpuDevice, WgpuError>;
    fn to_param_f32(value: Self) -> f32;
}

impl WgpuDType for f32 {
    const FEATURES: wgpu_rs::Features = wgpu_rs::Features::empty();
    const SHADERS: &'static str = SHADERS_F32;
    const BINARY_KERNEL: &'static str = "binary_f32_kernel";
    const SCALAR_KERNEL: &'static str = "scalar_f32_kernel";
    const MATMUL_KERNEL: &'static str = "matmul_f32_kernel";
    const SUM_KERNEL: &'static str = "sum_f32_kernel";

    fn default_device() -> std::result::Result<WgpuDevice, WgpuError> {
        static DEVICE: OnceLock<std::result::Result<WgpuDevice, WgpuError>> = OnceLock::new();
        DEVICE
            .get_or_init(|| create_default_device(wgpu_rs::Features::empty()))
            .clone()
    }

    fn to_param_f32(value: Self) -> f32 {
        value
    }
}

impl WgpuDType for f16 {
    const FEATURES: wgpu_rs::Features = wgpu_rs::Features::SHADER_F16;
    const SHADERS: &'static str = SHADERS_F16;
    const BINARY_KERNEL: &'static str = "binary_f16_kernel";
    const SCALAR_KERNEL: &'static str = "scalar_f16_kernel";
    const MATMUL_KERNEL: &'static str = "matmul_f16_kernel";
    const SUM_KERNEL: &'static str = "sum_f16_kernel";

    fn default_device() -> std::result::Result<WgpuDevice, WgpuError> {
        static DEVICE: OnceLock<std::result::Result<WgpuDevice, WgpuError>> = OnceLock::new();
        DEVICE
            .get_or_init(|| create_default_device(wgpu_rs::Features::SHADER_F16))
            .clone()
    }

    fn to_param_f32(value: Self) -> f32 {
        value.to_f32()
    }
}

impl<E> Backend<E> for Wgpu
where
    E: WgpuDType,
{
    type Device = WgpuDevice;
    type Storage = WgpuStorage;
    type Error = WgpuError;

    fn default_device() -> std::result::Result<Self::Device, Self::Error> {
        E::default_device()
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
        ensure_features::<E>(device)?;
        let bytes = padded_bytes(slice_as_bytes(&data));
        let buffer = device
            .raw
            .create_buffer_init(&wgpu_rs::util::BufferInitDescriptor {
                label: Some("rstorch-wgpu-storage"),
                contents: &bytes,
                usage: storage_usage(),
            });
        Ok(WgpuStorage {
            buffer: Arc::new(buffer),
            len: data.len(),
        })
    }

    fn to_vec(
        device: &Self::Device,
        storage: &Self::Storage,
    ) -> std::result::Result<Vec<E>, Self::Error> {
        ensure_features::<E>(device)?;
        if storage.len == 0 {
            return Ok(Vec::new());
        }

        let byte_len = buffer_byte_len::<E>(storage.len)?;
        let staging = device.raw.create_buffer(&wgpu_rs::BufferDescriptor {
            label: Some("rstorch-wgpu-readback"),
            size: byte_len,
            usage: wgpu_rs::BufferUsages::COPY_DST | wgpu_rs::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device
            .raw
            .create_command_encoder(&wgpu_rs::CommandEncoderDescriptor {
                label: Some("rstorch-wgpu-readback"),
            });
        encoder.copy_buffer_to_buffer(&storage.buffer, 0, &staging, 0, byte_len);
        device.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        slice.map_async(wgpu_rs::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.raw.poll(wgpu_rs::PollType::Wait).ok();
        receiver
            .recv()
            .map_err(|err| WgpuError::BufferMap(err.to_string()))?
            .map_err(|err| WgpuError::BufferMap(err.to_string()))?;

        let bytes = slice.get_mapped_range();
        let values = bytes
            .chunks_exact(std::mem::size_of::<E>())
            .take(storage.len)
            .map(read_value)
            .collect();
        drop(bytes);
        staging.unmap();
        Ok(values)
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
            return Err(WgpuError::BadMatmulDims {
                m,
                k,
                n,
                lhs_len: lhs.len,
                rhs_len: rhs.len,
            });
        }

        let len = m.saturating_mul(n);
        if len == 0 {
            return empty_storage::<E>(device, 0);
        }
        run_kernel::<E>(
            device,
            E::MATMUL_KERNEL,
            lhs,
            rhs,
            len,
            len,
            params(0.0, 0, len, m, k, n)?,
        )
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
        scalar::<E>(device, input, rhs, len, 0)
    }

    fn sub_scalar(
        device: &Self::Device,
        input: &Self::Storage,
        rhs: E,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        scalar::<E>(device, input, rhs, len, 1)
    }

    fn mul_scalar(
        device: &Self::Device,
        input: &Self::Storage,
        rhs: E,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        scalar::<E>(device, input, rhs, len, 2)
    }

    fn div_scalar(
        device: &Self::Device,
        input: &Self::Storage,
        rhs: E,
        len: usize,
    ) -> std::result::Result<Self::Storage, Self::Error> {
        scalar::<E>(device, input, rhs, len, 3)
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
        run_kernel::<E>(
            device,
            E::SUM_KERNEL,
            input,
            input,
            1,
            1,
            params(0.0, 0, len, 0, 0, 0)?,
        )
    }
}

fn create_default_device(
    features: wgpu_rs::Features,
) -> std::result::Result<WgpuDevice, WgpuError> {
    pollster::block_on(async {
        let instance = wgpu_rs::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu_rs::RequestAdapterOptions::default())
            .await
            .map_err(|_| WgpuError::NoAdapter)?;
        let info = adapter.get_info();
        if !adapter.features().contains(features) {
            return Err(WgpuError::MissingFeature("SHADER_F16"));
        }
        let (device, queue) = adapter
            .request_device(&wgpu_rs::DeviceDescriptor {
                label: Some("rstorch-wgpu-device"),
                required_features: features,
                required_limits: wgpu_rs::Limits::downlevel_defaults(),
                memory_hints: wgpu_rs::MemoryHints::Performance,
                trace: wgpu_rs::Trace::Off,
            })
            .await
            .map_err(|err| WgpuError::RequestDevice(err.to_string()))?;
        let layout = device.create_bind_group_layout(&wgpu_rs::BindGroupLayoutDescriptor {
            label: Some("rstorch-wgpu-bind-group-layout"),
            entries: &[
                storage_layout_entry(0, true),
                storage_layout_entry(1, true),
                storage_layout_entry(2, false),
                storage_layout_entry(3, true),
            ],
        });
        Ok(WgpuDevice {
            raw: Arc::new(device),
            queue: Arc::new(queue),
            name: Arc::from(info.name),
            features,
            pipelines: Arc::new(WgpuPipelineCache {
                layout,
                pipelines: Mutex::new(HashMap::new()),
            }),
        })
    })
}

fn binary<E: WgpuDType>(
    device: &WgpuDevice,
    lhs: &WgpuStorage,
    rhs: &WgpuStorage,
    len: usize,
    op: u32,
) -> std::result::Result<WgpuStorage, WgpuError> {
    ensure_len(lhs.len, len)?;
    ensure_len(rhs.len, len)?;
    if len == 0 {
        return empty_storage::<E>(device, 0);
    }
    run_kernel::<E>(
        device,
        E::BINARY_KERNEL,
        lhs,
        rhs,
        len,
        len,
        params(0.0, op, len, 0, 0, 0)?,
    )
}

fn scalar<E: WgpuDType>(
    device: &WgpuDevice,
    input: &WgpuStorage,
    rhs: E,
    len: usize,
    op: u32,
) -> std::result::Result<WgpuStorage, WgpuError> {
    ensure_len(input.len, len)?;
    if len == 0 {
        return empty_storage::<E>(device, 0);
    }
    run_kernel::<E>(
        device,
        E::SCALAR_KERNEL,
        input,
        input,
        len,
        len,
        params(E::to_param_f32(rhs), op, len, 0, 0, 0)?,
    )
}

fn run_kernel<E: WgpuDType>(
    device: &WgpuDevice,
    entry_point: &'static str,
    lhs: &WgpuStorage,
    rhs: &WgpuStorage,
    output_len: usize,
    dispatch_len: usize,
    params: [u8; 32],
) -> std::result::Result<WgpuStorage, WgpuError> {
    ensure_features::<E>(device)?;
    let output = empty_storage::<E>(device, output_len)?;
    let params_buffer = device
        .raw
        .create_buffer_init(&wgpu_rs::util::BufferInitDescriptor {
            label: Some("rstorch-wgpu-params"),
            contents: &params,
            usage: wgpu_rs::BufferUsages::STORAGE,
        });
    let pipeline = pipeline::<E>(device, entry_point);
    let bind_group = device.raw.create_bind_group(&wgpu_rs::BindGroupDescriptor {
        label: Some("rstorch-wgpu-bind-group"),
        layout: &device.pipelines.layout,
        entries: &[
            wgpu_rs::BindGroupEntry {
                binding: 0,
                resource: lhs.buffer.as_entire_binding(),
            },
            wgpu_rs::BindGroupEntry {
                binding: 1,
                resource: rhs.buffer.as_entire_binding(),
            },
            wgpu_rs::BindGroupEntry {
                binding: 2,
                resource: output.buffer.as_entire_binding(),
            },
            wgpu_rs::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device
        .raw
        .create_command_encoder(&wgpu_rs::CommandEncoderDescriptor {
            label: Some("rstorch-wgpu-compute"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu_rs::ComputePassDescriptor {
            label: Some(entry_point),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((dispatch_len as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    device.queue.submit(Some(encoder.finish()));
    device.raw.poll(wgpu_rs::PollType::Wait).ok();
    Ok(output)
}

fn pipeline<E: WgpuDType>(
    device: &WgpuDevice,
    entry_point: &'static str,
) -> Arc<wgpu_rs::ComputePipeline> {
    let mut cache = device
        .pipelines
        .pipelines
        .lock()
        .expect("pipeline cache poisoned");
    if let Some(pipeline) = cache.get(entry_point) {
        return Arc::clone(pipeline);
    }

    let shader = device
        .raw
        .create_shader_module(wgpu_rs::ShaderModuleDescriptor {
            label: Some("rstorch-wgpu-shader"),
            source: wgpu_rs::ShaderSource::Wgsl(E::SHADERS.into()),
        });
    let pipeline_layout = device
        .raw
        .create_pipeline_layout(&wgpu_rs::PipelineLayoutDescriptor {
            label: Some("rstorch-wgpu-pipeline-layout"),
            bind_group_layouts: &[&device.pipelines.layout],
            push_constant_ranges: &[],
        });
    let pipeline = Arc::new(device.raw.create_compute_pipeline(
        &wgpu_rs::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: wgpu_rs::PipelineCompilationOptions::default(),
            cache: None,
        },
    ));
    cache.insert(entry_point, Arc::clone(&pipeline));
    pipeline
}

fn empty_storage<E>(
    device: &WgpuDevice,
    len: usize,
) -> std::result::Result<WgpuStorage, WgpuError> {
    let buffer = device.raw.create_buffer(&wgpu_rs::BufferDescriptor {
        label: Some("rstorch-wgpu-empty"),
        size: buffer_byte_len::<E>(len)?,
        usage: storage_usage(),
        mapped_at_creation: false,
    });
    Ok(WgpuStorage {
        buffer: Arc::new(buffer),
        len,
    })
}

fn ensure_len(found: usize, expected: usize) -> std::result::Result<(), WgpuError> {
    if found != expected {
        return Err(WgpuError::LengthMismatch { expected, found });
    }
    Ok(())
}

fn ensure_features<E: WgpuDType>(device: &WgpuDevice) -> std::result::Result<(), WgpuError> {
    if !device.features.contains(E::FEATURES) {
        return Err(WgpuError::MissingFeature("SHADER_F16"));
    }
    Ok(())
}

fn params(
    rhs: f32,
    op: u32,
    len: usize,
    m: usize,
    k: usize,
    n: usize,
) -> std::result::Result<[u8; 32], WgpuError> {
    let mut bytes = [0; 32];
    bytes[0..4].copy_from_slice(&rhs.to_ne_bytes());
    bytes[4..8].copy_from_slice(&op.to_ne_bytes());
    bytes[8..12].copy_from_slice(&checked_u32(len)?.to_ne_bytes());
    bytes[12..16].copy_from_slice(&checked_u32(m)?.to_ne_bytes());
    bytes[16..20].copy_from_slice(&checked_u32(k)?.to_ne_bytes());
    bytes[20..24].copy_from_slice(&checked_u32(n)?.to_ne_bytes());
    Ok(bytes)
}

fn checked_u32(value: usize) -> std::result::Result<u32, WgpuError> {
    u32::try_from(value).map_err(|_| WgpuError::SizeOverflow)
}

fn buffer_byte_len<E>(len: usize) -> std::result::Result<u64, WgpuError> {
    let bytes = len
        .max(1)
        .checked_mul(std::mem::size_of::<E>())
        .ok_or(WgpuError::SizeOverflow)?;
    u64::try_from(align_copy_bytes(bytes)).map_err(|_| WgpuError::SizeOverflow)
}

fn align_copy_bytes(bytes: usize) -> usize {
    bytes.next_multiple_of(wgpu_rs::COPY_BUFFER_ALIGNMENT as usize)
}

fn padded_bytes(bytes: &[u8]) -> Vec<u8> {
    let mut padded = bytes.to_vec();
    padded.resize(align_copy_bytes(padded.len().max(1)), 0);
    padded
}

fn storage_usage() -> wgpu_rs::BufferUsages {
    wgpu_rs::BufferUsages::COPY_SRC
        | wgpu_rs::BufferUsages::COPY_DST
        | wgpu_rs::BufferUsages::STORAGE
}

fn storage_layout_entry(binding: u32, read_only: bool) -> wgpu_rs::BindGroupLayoutEntry {
    wgpu_rs::BindGroupLayoutEntry {
        binding,
        visibility: wgpu_rs::ShaderStages::COMPUTE,
        ty: wgpu_rs::BindingType::Buffer {
            ty: wgpu_rs::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn slice_as_bytes<E>(values: &[E]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn read_value<E: DType>(bytes: &[u8]) -> E {
    let mut value = E::ZERO;
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            (&mut value as *mut E).cast::<u8>(),
            std::mem::size_of::<E>(),
        );
    }
    value
}

#[cfg(test)]
mod tests {
    use super::{SHADERS_F16, SHADERS_F32};

    #[test]
    fn bundled_kernel_source_contains_required_entrypoints() {
        assert!(SHADERS_F32.contains("fn apply_op_f32"));
        assert!(SHADERS_F32.contains("fn binary_f32_kernel"));
        assert!(SHADERS_F32.contains("fn scalar_f32_kernel"));
        assert!(SHADERS_F32.contains("fn matmul_f32_kernel"));
        assert!(SHADERS_F32.contains("fn sum_f32_kernel"));
        assert!(SHADERS_F16.contains("enable f16"));
        assert!(SHADERS_F16.contains("fn apply_op_f16"));
        assert!(SHADERS_F16.contains("fn binary_f16_kernel"));
        assert!(SHADERS_F16.contains("fn scalar_f16_kernel"));
        assert!(SHADERS_F16.contains("fn matmul_f16_kernel"));
        assert!(SHADERS_F16.contains("fn sum_f16_kernel"));
    }
}
