//! Portable native WebGPU backend.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, mpsc};

use ::wgpu::util::DeviceExt;

use crate::backend::{
    ArgReduceOp, BackendOps, BinaryOp, CmpOp, Conv2dParams, ConvOp, FusedOp, ReduceOp, UnaryOp,
    View, conv_geometry::Conv2dGeometry,
};
use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::storage::{CpuStorage, Storage};

const SOURCE: &str = include_str!("kernels.wgsl");
const F16_SOURCE: &str = include_str!("f16_kernels.wgsl");
const F16_MIXED_SOURCE: &str = include_str!("f16_mixed.wgsl");
const F16_OUTPUT_SOURCE: &str = include_str!("f16_output.wgsl");
const F16_INPUT_SOURCE: &str = include_str!("f16_input.wgsl");
const MAX_RANK: usize = 8;
const WORKGROUP: u32 = 256;

pub(crate) struct WgpuBackend {
    ordinal: usize,
}

#[derive(Clone)]
pub(crate) struct WgpuStorage {
    buffer: Arc<::wgpu::Buffer>,
    dtype: DType,
    len: usize,
    context: Arc<Context>,
    validations: Arc<Vec<Validation>>,
}

#[derive(Clone)]
struct Validation {
    buffer: Arc<::wgpu::Buffer>,
    op: &'static str,
    axis: usize,
    bound: usize,
}

struct Context {
    ordinal: usize,
    device: ::wgpu::Device,
    queue: ::wgpu::Queue,
    shader: ::wgpu::ShaderModule,
    f16_shaders: Option<F16Shaders>,
    bind_layout: ::wgpu::BindGroupLayout,
    pipeline_layout: ::wgpu::PipelineLayout,
    pipelines: Mutex<HashMap<&'static str, ::wgpu::ComputePipeline>>,
    buffers: Mutex<HashMap<u64, Vec<Arc<::wgpu::Buffer>>>>,
    readbacks: Mutex<HashMap<u64, Vec<Arc<::wgpu::Buffer>>>>,
    dummy: Arc<::wgpu::Buffer>,
    status: Arc<::wgpu::Buffer>,
}

struct F16Shaders {
    pure: ::wgpu::ShaderModule,
    mixed: ::wgpu::ShaderModule,
    output: ::wgpu::ShaderModule,
    input: ::wgpu::ShaderModule,
}

type ContextResult = std::result::Result<Arc<Context>, String>;

pub(crate) fn best_adapter_ordinal() -> Option<usize> {
    adapters()
        .iter()
        .enumerate()
        .filter(|(ordinal, adapter)| {
            adapter.get_info().device_type != ::wgpu::DeviceType::Cpu && context(*ordinal).is_ok()
        })
        .min_by_key(|(ordinal, adapter)| adapter_rank(*ordinal, &adapter.get_info()))
        .map(|(ordinal, _)| ordinal)
}

fn adapter_rank(ordinal: usize, info: &::wgpu::AdapterInfo) -> (u8, u8, usize) {
    let device = match info.device_type {
        ::wgpu::DeviceType::DiscreteGpu => 0,
        ::wgpu::DeviceType::IntegratedGpu => 1,
        ::wgpu::DeviceType::VirtualGpu => 2,
        ::wgpu::DeviceType::Other => 3,
        ::wgpu::DeviceType::Cpu => 4,
    };
    #[cfg(target_os = "windows")]
    let backend = match info.backend {
        ::wgpu::Backend::Dx12 => 0,
        ::wgpu::Backend::Vulkan => 1,
        ::wgpu::Backend::Gl => 2,
        _ => 3,
    };
    #[cfg(target_os = "macos")]
    let backend = match info.backend {
        ::wgpu::Backend::Metal => 0,
        ::wgpu::Backend::Vulkan => 1,
        ::wgpu::Backend::Gl => 2,
        _ => 3,
    };
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let backend = match info.backend {
        ::wgpu::Backend::Vulkan => 0,
        ::wgpu::Backend::Gl => 1,
        _ => 2,
    };
    (device, backend, ordinal)
}

fn required_limits() -> ::wgpu::Limits {
    let mut limits = ::wgpu::Limits::downlevel_defaults();
    limits.max_storage_buffers_per_shader_stage = 6;
    limits
}

pub(crate) fn supports_f16(device: Device) -> bool {
    match device {
        Device::Wgpu(ordinal) => adapters()
            .get(ordinal)
            .is_some_and(|adapter| adapter.features().contains(::wgpu::Features::SHADER_F16)),
        _ => false,
    }
}

fn adapters() -> &'static Vec<::wgpu::Adapter> {
    static ADAPTERS: OnceLock<Vec<::wgpu::Adapter>> = OnceLock::new();
    ADAPTERS.get_or_init(|| {
        let instance = ::wgpu::Instance::new(&::wgpu::InstanceDescriptor::default());
        let mut adapters = instance.enumerate_adapters(::wgpu::Backends::all());
        let required = required_limits();
        adapters.retain(|adapter| required.check_limits(&adapter.limits()));
        adapters.sort_by_key(|adapter| {
            let info = adapter.get_info();
            (
                format!("{:?}", info.backend),
                info.vendor,
                info.device,
                format!("{:?}", info.device_type),
                info.name,
            )
        });
        adapters
    })
}

pub(crate) fn backend(ordinal: usize) -> &'static dyn BackendOps {
    static BACKENDS: OnceLock<Mutex<HashMap<usize, &'static WgpuBackend>>> = OnceLock::new();
    let mut values = BACKENDS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("wgpu backend registry poisoned");
    *values
        .entry(ordinal)
        .or_insert_with(|| Box::leak(Box::new(WgpuBackend { ordinal })))
}

fn context(ordinal: usize) -> Result<Arc<Context>> {
    static CONTEXTS: OnceLock<Mutex<HashMap<usize, ContextResult>>> = OnceLock::new();
    let mut values = CONTEXTS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("wgpu context registry poisoned");
    values
        .entry(ordinal)
        .or_insert_with(|| create_context(ordinal))
        .clone()
        .map_err(|msg| Error::Backend {
            op: "wgpu_device",
            msg,
        })
}

fn create_context(ordinal: usize) -> std::result::Result<Arc<Context>, String> {
    let adapter = adapters()
        .get(ordinal)
        .ok_or_else(|| format!("adapter ordinal {ordinal} is unavailable"))?;
    let limits = required_limits();
    let shader_f16 = adapter.features().contains(::wgpu::Features::SHADER_F16);
    let required_features = if shader_f16 {
        ::wgpu::Features::SHADER_F16
    } else {
        ::wgpu::Features::empty()
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&::wgpu::DeviceDescriptor {
        label: Some("rstorch-wgpu"),
        required_features,
        required_limits: limits,
        memory_hints: ::wgpu::MemoryHints::MemoryUsage,
        trace: ::wgpu::Trace::Off,
    }))
    .map_err(|error| error.to_string())?;
    let shader = device.create_shader_module(::wgpu::ShaderModuleDescriptor {
        label: Some("rstorch kernels"),
        source: ::wgpu::ShaderSource::Wgsl(SOURCE.into()),
    });
    let f16_shaders = shader_f16.then(|| F16Shaders {
        pure: shader_module(&device, "rstorch f16 kernels", F16_SOURCE),
        mixed: shader_module(&device, "rstorch f16 mixed kernels", F16_MIXED_SOURCE),
        output: shader_module(&device, "rstorch f16 output kernels", F16_OUTPUT_SOURCE),
        input: shader_module(&device, "rstorch f16 input kernels", F16_INPUT_SOURCE),
    });
    let bind_layout = device.create_bind_group_layout(&::wgpu::BindGroupLayoutDescriptor {
        label: Some("rstorch storage layout"),
        entries: &[
            binding_layout(0, true),
            binding_layout(1, true),
            binding_layout(2, true),
            binding_layout(3, false),
            binding_layout(4, true),
            binding_layout(5, false),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&::wgpu::PipelineLayoutDescriptor {
        label: Some("rstorch pipeline layout"),
        bind_group_layouts: &[&bind_layout],
        push_constant_ranges: &[],
    });
    let dummy = Arc::new(
        device.create_buffer_init(&::wgpu::util::BufferInitDescriptor {
            label: Some("rstorch dummy"),
            contents: &[0; 16],
            usage: ::wgpu::BufferUsages::STORAGE
                | ::wgpu::BufferUsages::COPY_SRC
                | ::wgpu::BufferUsages::COPY_DST,
        }),
    );
    let status = Arc::new(
        device.create_buffer_init(&::wgpu::util::BufferInitDescriptor {
            label: Some("rstorch dummy status"),
            contents: &[0; 16],
            usage: ::wgpu::BufferUsages::STORAGE | ::wgpu::BufferUsages::COPY_DST,
        }),
    );
    Ok(Arc::new(Context {
        ordinal,
        device,
        queue,
        shader,
        f16_shaders,
        bind_layout,
        pipeline_layout,
        pipelines: Mutex::new(HashMap::new()),
        buffers: Mutex::new(HashMap::new()),
        readbacks: Mutex::new(HashMap::new()),
        dummy,
        status,
    }))
}

fn shader_module(
    device: &::wgpu::Device,
    label: &'static str,
    source: &'static str,
) -> ::wgpu::ShaderModule {
    device.create_shader_module(::wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: ::wgpu::ShaderSource::Wgsl(source.into()),
    })
}

fn binding_layout(binding: u32, read_only: bool) -> ::wgpu::BindGroupLayoutEntry {
    ::wgpu::BindGroupLayoutEntry {
        binding,
        visibility: ::wgpu::ShaderStages::COMPUTE,
        ty: ::wgpu::BindingType::Buffer {
            ty: ::wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl WgpuStorage {
    pub(crate) fn dtype(&self) -> DType {
        self.dtype
    }

    pub(crate) fn device(&self) -> Device {
        Device::Wgpu(self.context.ordinal)
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

fn unsupported(op: &'static str, device: Device, dtype: DType) -> Error {
    Error::Unsupported { op, device, dtype }
}

fn words(dtype: DType) -> Option<usize> {
    match dtype {
        DType::F32 | DType::Bool => Some(1),
        DType::I64 => Some(2),
        DType::F16 | DType::BF16 | DType::F64 => None,
    }
}

fn element_bytes(dtype: DType) -> Option<usize> {
    match dtype {
        DType::F16 => Some(2),
        DType::F32 | DType::Bool => Some(4),
        DType::I64 => Some(8),
        DType::BF16 | DType::F64 => None,
    }
}

fn bytes_of_words(values: &[u32]) -> &[u8] {
    // SAFETY: u32 has no padding, and the byte slice has the same lifetime.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len() * 4) }
}

fn bytes_of_f16(values: &[half::f16]) -> &[u8] {
    // SAFETY: f16 has no padding, and the byte slice has the same lifetime.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), values.len() * 2) }
}

fn u32_checked(value: usize, op: &'static str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::InvalidArg {
        op,
        msg: format!("wgpu address {value} exceeds the portable u32 kernel limit"),
    })
}

fn descriptor(
    params: &mut [u32; 64],
    base: usize,
    layout: &Layout,
    op: &'static str,
) -> Result<()> {
    if layout.rank() > MAX_RANK {
        return Err(Error::InvalidArg {
            op,
            msg: format!("wgpu kernels support rank at most {MAX_RANK}"),
        });
    }
    params[base] = u32_checked(layout.offset(), op)?;
    params[base + 1] = layout.rank() as u32;
    for (i, (&dim, &stride)) in layout.dims().iter().zip(layout.strides()).enumerate() {
        params[base + 2 + i] = u32_checked(dim, op)?;
        params[base + 10 + i] = u32_checked(stride, op)?;
    }
    Ok(())
}

fn virtual_descriptor(
    params: &mut [u32; 64],
    base: usize,
    layout: &Layout,
    dims: &[usize],
    strides: &[usize],
    op: &'static str,
) -> Result<()> {
    params[base] = u32_checked(layout.offset(), op)?;
    params[base + 1] = dims.len() as u32;
    for (i, (&dim, &stride)) in dims.iter().zip(strides).enumerate() {
        params[base + 2 + i] = u32_checked(dim, op)?;
        params[base + 10 + i] = u32_checked(stride, op)?;
    }
    Ok(())
}

fn storage<'a>(view: View<'a>, op: &'static str) -> Result<&'a WgpuStorage> {
    match view.storage() {
        Storage::Wgpu(value) => Ok(value),
        other => Err(Error::DeviceMismatch {
            op,
            expected: view.device(),
            got: other.device(),
        }),
    }
}

impl Context {
    fn pipeline(&self, entry: &'static str) -> ::wgpu::ComputePipeline {
        let mut pipelines = self.pipelines.lock().expect("wgpu pipeline cache poisoned");
        pipelines
            .entry(entry)
            .or_insert_with(|| {
                let module = if entry.starts_with("f16_") {
                    let shaders = self
                        .f16_shaders
                        .as_ref()
                        .expect("F16 pipeline requested without SHADER_F16");
                    if matches!(
                        entry,
                        "f16_validate_indices"
                            | "f16_where"
                            | "f16_masked_fill"
                            | "f16_index_select"
                            | "f16_gather"
                            | "f16_index_add"
                            | "f16_scatter_add"
                    ) {
                        &shaders.mixed
                    } else if matches!(entry, "f16_compare" | "f16_cast_out" | "f16_arg_reduce") {
                        &shaders.output
                    } else if entry == "f16_cast_in" {
                        &shaders.input
                    } else {
                        &shaders.pure
                    }
                } else {
                    &self.shader
                };
                self.device
                    .create_compute_pipeline(&::wgpu::ComputePipelineDescriptor {
                        label: Some(entry),
                        layout: Some(&self.pipeline_layout),
                        module,
                        entry_point: Some(entry),
                        compilation_options: ::wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    })
            })
            .clone()
    }

    fn allocate(&self, bytes: usize, label: &'static str) -> Arc<::wgpu::Buffer> {
        let requested = bytes.max(4) as u64;
        let alignment = if requested <= 1 << 20 { 256 } else { 4096 };
        let size = requested
            .checked_add(alignment - 1)
            .map_or(requested, |rounded| rounded / alignment * alignment);
        // Held across the allocation below on purpose: releasing it early
        // would let two callers race past the reuse check and each allocate
        // a fresh buffer instead of one reusing the other's.
        #[allow(clippy::significant_drop_tightening)]
        let mut buffers = self.buffers.lock().expect("wgpu buffer pool poisoned");
        let entries = buffers.entry(size).or_default();
        if let Some(buffer) = entries.iter().find(|buffer| Arc::strong_count(buffer) == 1) {
            return Arc::clone(buffer);
        }
        let buffer = Arc::new(self.device.create_buffer(&::wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: ::wgpu::BufferUsages::STORAGE
                | ::wgpu::BufferUsages::COPY_SRC
                | ::wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        entries.push(Arc::clone(&buffer));
        buffer
    }

    // Keep this submission boundary flat: each argument maps directly to one
    // WebGPU binding or dispatch property, and bundling them would obscure the
    // low-level call sites.
    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        self: &Arc<Self>,
        entry: &'static str,
        inputs: &[&WgpuStorage],
        output: Arc<::wgpu::Buffer>,
        output_len: usize,
        dispatch_len: usize,
        output_dtype: DType,
        params: &[u32; 64],
        validate_indices: bool,
        workgroups: Option<[u32; 3]>,
    ) -> Result<WgpuStorage> {
        let pipeline = self.pipeline(entry);
        let params_buffer = self
            .device
            .create_buffer_init(&::wgpu::util::BufferInitDescriptor {
                label: Some("rstorch params"),
                contents: bytes_of_words(params),
                usage: ::wgpu::BufferUsages::STORAGE,
            });
        let validation = validate_indices.then(|| {
            Arc::new(
                self.device
                    .create_buffer_init(&::wgpu::util::BufferInitDescriptor {
                        label: Some("rstorch index status"),
                        contents: &[0; 16],
                        usage: ::wgpu::BufferUsages::STORAGE
                            | ::wgpu::BufferUsages::COPY_SRC
                            | ::wgpu::BufferUsages::COPY_DST,
                    }),
            )
        });
        let buffers = [
            inputs
                .first()
                .map_or_else(|| self.dummy.as_ref(), |v| v.buffer.as_ref()),
            inputs
                .get(1)
                .map_or_else(|| self.dummy.as_ref(), |v| v.buffer.as_ref()),
            inputs
                .get(2)
                .map_or_else(|| self.dummy.as_ref(), |v| v.buffer.as_ref()),
        ];
        let bind_group = self.device.create_bind_group(&::wgpu::BindGroupDescriptor {
            label: Some("rstorch bind group"),
            layout: &self.bind_layout,
            entries: &[
                ::wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffers[0].as_entire_binding(),
                },
                ::wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers[1].as_entire_binding(),
                },
                ::wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buffers[2].as_entire_binding(),
                },
                ::wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output.as_entire_binding(),
                },
                ::wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
                ::wgpu::BindGroupEntry {
                    binding: 5,
                    resource: validation
                        .as_deref()
                        .unwrap_or(&self.status)
                        .as_entire_binding(),
                },
            ],
        });
        // Index kernels reserve p[1] for their logical index count. If the
        // output is empty, run only this validation pass so no index is skipped.
        let validate_empty = validate_indices && dispatch_len == 0 && params[1] != 0;
        if validate_empty || dispatch_len != 0 {
            let mut encoder = self
                .device
                .create_command_encoder(&::wgpu::CommandEncoderDescriptor::default());
            if validate_empty {
                let mut pass = encoder.begin_compute_pass(&::wgpu::ComputePassDescriptor {
                    label: Some("validate_indices"),
                    timestamp_writes: None,
                });
                let validation_entry = if entry.starts_with("f16_") {
                    "f16_validate_indices"
                } else {
                    "validate_indices"
                };
                pass.set_pipeline(&self.pipeline(validation_entry));
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(params[1].div_ceil(WORKGROUP), 1, 1);
            }
            if dispatch_len != 0 {
                let groups = workgroups.unwrap_or([
                    u32_checked(dispatch_len, entry)?.div_ceil(WORKGROUP),
                    1,
                    1,
                ]);
                let mut pass = encoder.begin_compute_pass(&::wgpu::ComputePassDescriptor {
                    label: Some(entry),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
            }
            self.queue.submit([encoder.finish()]);
        }
        let mut validations = Vec::new();
        for input in inputs {
            validations.extend(input.validations.iter().cloned());
        }
        if let Some(value) = validation {
            validations.push(Validation {
                buffer: value,
                op: entry,
                axis: params[2] as usize,
                bound: params[3] as usize,
            });
        }
        Ok(WgpuStorage {
            buffer: output,
            dtype: output_dtype,
            len: output_len,
            context: Arc::clone(self),
            validations: Arc::new(validations),
        })
    }

    fn read(&self, source: &::wgpu::Buffer, bytes: usize) -> Result<Vec<u8>> {
        let requested = bytes.max(4).next_multiple_of(4) as u64;
        let size = requested
            .checked_add(255)
            .map_or(requested, |rounded| rounded / 256 * 256);
        let staging = {
            // Held across the allocation below on purpose: releasing it early
            // would let two callers race past the reuse check and each
            // allocate a fresh buffer instead of one reusing the other's.
            #[allow(clippy::significant_drop_tightening)]
            let mut readbacks = self.readbacks.lock().expect("wgpu readback pool poisoned");
            let entries = readbacks.entry(size).or_default();
            if let Some(buffer) = entries.iter().find(|buffer| Arc::strong_count(buffer) == 1) {
                Arc::clone(buffer)
            } else {
                let buffer = Arc::new(self.device.create_buffer(&::wgpu::BufferDescriptor {
                    label: Some("rstorch readback"),
                    size,
                    usage: ::wgpu::BufferUsages::MAP_READ | ::wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                entries.push(Arc::clone(&buffer));
                buffer
            }
        };
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, requested);
        self.queue.submit([encoder.finish()]);
        let (send, recv) = mpsc::sync_channel(1);
        staging
            .slice(..)
            .map_async(::wgpu::MapMode::Read, move |result| {
                let _ = send.send(result);
            });
        self.device
            .poll(::wgpu::PollType::Wait)
            .map_err(|error| Error::Backend {
                op: "transfer_out",
                msg: format!("{error:?}"),
            })?;
        recv.recv()
            .map_err(|error| Error::Backend {
                op: "transfer_out",
                msg: error.to_string(),
            })?
            .map_err(|error| Error::Backend {
                op: "transfer_out",
                msg: error.to_string(),
            })?;
        let result = {
            let mapped = staging.slice(..).get_mapped_range();
            mapped[..bytes].to_vec()
        };
        staging.unmap();
        Ok(result)
    }
}

fn op_code_binary(op: BinaryOp) -> u32 {
    match op {
        BinaryOp::Add => 0,
        BinaryOp::Sub => 1,
        BinaryOp::Mul => 2,
        BinaryOp::Div => 3,
        BinaryOp::Maximum => 4,
        BinaryOp::Minimum => 5,
    }
}
fn op_code_unary(op: UnaryOp) -> u32 {
    match op {
        UnaryOp::Relu => 0,
        UnaryOp::Gelu => 1,
        UnaryOp::Exp => 2,
        UnaryOp::Ln => 3,
        UnaryOp::Sqrt => 4,
        UnaryOp::Tanh => 5,
        UnaryOp::Sigmoid => 6,
        UnaryOp::Neg => 7,
        UnaryOp::Abs => 8,
    }
}
fn op_code_cmp(op: CmpOp) -> u32 {
    match op {
        CmpOp::Eq => 0,
        CmpOp::Ne => 1,
        CmpOp::Lt => 2,
        CmpOp::Le => 3,
        CmpOp::Gt => 4,
        CmpOp::Ge => 5,
    }
}

impl WgpuBackend {
    fn compute(
        &self,
        entry: &'static str,
        inputs: &[View<'_>],
        dtype: DType,
        len: usize,
        mut params: [u32; 64],
        validate: bool,
    ) -> Result<Storage> {
        let context = context(self.ordinal)?;
        if dtype == DType::F16 && context.f16_shaders.is_none() {
            return Err(unsupported(entry, Device::Wgpu(self.ordinal), dtype));
        }
        let values = inputs
            .iter()
            .map(|view| storage(*view, entry))
            .collect::<Result<Vec<_>>>()?;
        if values
            .iter()
            .any(|value| !Arc::ptr_eq(&value.context, &context))
        {
            return Err(Error::DeviceMismatch {
                op: entry,
                expected: Device::Wgpu(self.ordinal),
                got: inputs[0].device(),
            });
        }
        params[0] = u32_checked(len, entry)?;
        for (view, base) in inputs.iter().zip([8usize, 26, 44]) {
            descriptor(&mut params, base, view.layout(), entry)?;
        }
        let output = context.allocate(
            len * element_bytes(dtype)
                .ok_or_else(|| unsupported(entry, Device::Wgpu(self.ordinal), dtype))?,
            entry,
        );
        context
            .dispatch(
                entry, &values, output, len, len, dtype, &params, validate, None,
            )
            .map(Storage::Wgpu)
    }

    fn float_dtype(&self, op: &'static str, views: &[View<'_>]) -> Result<DType> {
        let dtype = views.first().map_or(DType::F32, View::dtype);
        if !matches!(dtype, DType::F16 | DType::F32)
            || views.iter().any(|view| view.dtype() != dtype)
            || (dtype == DType::F16 && !supports_f16(Device::Wgpu(self.ordinal)))
        {
            return Err(unsupported(op, Device::Wgpu(self.ordinal), dtype));
        }
        Ok(dtype)
    }
}

impl BackendOps for WgpuBackend {
    fn transfer_in(&self, host: CpuStorage) -> Result<Storage> {
        let dtype = host.dtype();
        let context = context(self.ordinal)?;
        element_bytes(dtype)
            .ok_or_else(|| unsupported("transfer_in", Device::Wgpu(self.ordinal), dtype))?;
        if dtype == DType::F16 && context.f16_shaders.is_none() {
            return Err(unsupported(
                "transfer_in",
                Device::Wgpu(self.ordinal),
                dtype,
            ));
        }
        let len = host.len();
        let f16_data = match &host {
            CpuStorage::F16(values) => Some(bytes_of_f16(values).to_vec()),
            _ => None,
        };
        let data: Vec<u32> = match host {
            CpuStorage::F32(values) => values.iter().map(|v| v.to_bits()).collect(),
            CpuStorage::I64(values) => values
                .iter()
                .flat_map(|v| {
                    let bits = *v as u64;
                    [bits as u32, (bits >> 32) as u32]
                })
                .collect(),
            CpuStorage::Bool(values) => values.iter().map(|v| u32::from(*v)).collect(),
            CpuStorage::F16(_) => Vec::new(),
            CpuStorage::BF16(_) | CpuStorage::F64(_) => unreachable!(),
        };
        let contents = f16_data.as_deref().unwrap_or_else(|| bytes_of_words(&data));
        let buffer = if contents.is_empty() {
            context.allocate(4, "rstorch upload")
        } else {
            Arc::new(
                context
                    .device
                    .create_buffer_init(&::wgpu::util::BufferInitDescriptor {
                        label: Some("rstorch upload"),
                        contents,
                        usage: ::wgpu::BufferUsages::STORAGE
                            | ::wgpu::BufferUsages::COPY_SRC
                            | ::wgpu::BufferUsages::COPY_DST,
                    }),
            )
        };
        Ok(Storage::Wgpu(WgpuStorage {
            buffer,
            dtype,
            len,
            context,
            validations: Arc::new(Vec::new()),
        }))
    }

    fn transfer_out(&self, x: View<'_>) -> Result<CpuStorage> {
        let source = storage(x, "transfer_out")?;
        let dense = if x.layout().is_contiguous()
            && x.layout().offset() == 0
            && x.layout().num_elements() == source.len
        {
            source.clone()
        } else {
            match self.copy_strided(x)? {
                Storage::Wgpu(v) => v,
                _ => unreachable!(),
            }
        };
        for validation in dense.validations.iter() {
            let bytes = dense.context.read(&validation.buffer, 16)?;
            let fields: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|v| u32::from_ne_bytes(v.try_into().unwrap()))
                .collect();
            if fields[0] != 0 {
                let bits = u64::from(fields[1]) | (u64::from(fields[2]) << 32);
                return Err(Error::IndexOutOfBounds {
                    op: validation.op,
                    index: bits as i64,
                    axis: validation.axis,
                    size: validation.bound,
                });
            }
        }
        let bytes = dense.context.read(
            &dense.buffer,
            dense.len * element_bytes(dense.dtype).unwrap(),
        )?;
        if dense.dtype == DType::F16 {
            return Ok(CpuStorage::F16(Arc::new(
                bytes
                    .chunks_exact(2)
                    .map(|v| half::f16::from_bits(u16::from_ne_bytes(v.try_into().unwrap())))
                    .collect(),
            )));
        }
        let raw: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|v| u32::from_ne_bytes(v.try_into().unwrap()))
            .collect();
        Ok(match dense.dtype {
            DType::F32 => CpuStorage::F32(Arc::new(raw.into_iter().map(f32::from_bits).collect())),
            DType::Bool => CpuStorage::Bool(Arc::new(raw.into_iter().map(|v| v != 0).collect())),
            DType::I64 => CpuStorage::I64(Arc::new(
                raw.chunks_exact(2)
                    .map(|v| (u64::from(v[0]) | (u64::from(v[1]) << 32)) as i64)
                    .collect(),
            )),
            _ => unreachable!(),
        })
    }

    fn copy_strided(&self, x: View<'_>) -> Result<Storage> {
        if x.dtype() == DType::F16 {
            if !supports_f16(x.device()) {
                return Err(unsupported("copy", x.device(), x.dtype()));
            }
            let mut params = [0; 64];
            let output = Layout::contiguous(x.layout().dims().to_vec())?;
            params[3] = u32::from(x.layout().is_contiguous());
            params[4] = 1;
            descriptor(&mut params, 26, &output, "copy")?;
            return self.compute(
                "f16_copy",
                &[x],
                DType::F16,
                x.layout().num_elements(),
                params,
                false,
            );
        }
        words(x.dtype()).ok_or_else(|| unsupported("copy", x.device(), x.dtype()))?;
        let mut params = [0; 64];
        params[2] = words(x.dtype()).unwrap() as u32;
        let output = Layout::contiguous(x.layout().dims().to_vec())?;
        params[3] = u32::from(x.layout().is_contiguous());
        params[4] = 1;
        descriptor(&mut params, 26, &output, "copy")?;
        self.compute(
            "copy_words",
            &[x],
            x.dtype(),
            x.layout().num_elements(),
            params,
            false,
        )
    }

    fn copy_into(&self, src: View<'_>, dst: &mut Storage, dst_layout: &Layout) -> Result<()> {
        let Storage::Wgpu(target) = dst else {
            return Err(Error::DeviceMismatch {
                op: "copy_into",
                expected: src.device(),
                got: dst.device(),
            });
        };
        if src.dtype() != target.dtype {
            return Err(Error::DTypeMismatch {
                op: "copy_into",
                expected: target.dtype,
                got: src.dtype(),
            });
        }
        let source = storage(src, "copy_into")?;
        let context = context(self.ordinal)?;
        if !Arc::ptr_eq(&source.context, &context) || !Arc::ptr_eq(&target.context, &context) {
            return Err(Error::DeviceMismatch {
                op: "copy_into",
                expected: Device::Wgpu(self.ordinal),
                got: if !Arc::ptr_eq(&source.context, &context) {
                    source.device()
                } else {
                    target.device()
                },
            });
        }
        let mut params = [0; 64];
        params[0] = u32_checked(src.layout().num_elements(), "copy_into")?;
        params[2] = if src.dtype() == DType::F16 {
            if !supports_f16(src.device()) {
                return Err(unsupported("copy_into", src.device(), src.dtype()));
            }
            1
        } else {
            words(src.dtype()).ok_or_else(|| unsupported("copy_into", src.device(), src.dtype()))?
                as u32
        };
        params[3] = u32::from(src.layout().is_contiguous());
        params[4] = u32::from(dst_layout.is_contiguous());
        descriptor(&mut params, 8, src.layout(), "copy_into")?;
        descriptor(&mut params, 26, dst_layout, "copy_into")?;
        let updated = context.dispatch(
            if src.dtype() == DType::F16 {
                "f16_copy"
            } else {
                "copy_words"
            },
            &[source],
            Arc::clone(&target.buffer),
            src.layout().num_elements(),
            target.len,
            target.dtype,
            &params,
            false,
            None,
        )?;
        target.validations = updated.validations;
        Ok(())
    }

    fn full(&self, len: usize, dtype: DType, value: f64) -> Result<Storage> {
        if dtype == DType::F16 {
            if !supports_f16(Device::Wgpu(self.ordinal)) {
                return Err(unsupported("full", Device::Wgpu(self.ordinal), dtype));
            }
            let mut params = [0; 64];
            params[3] = (value as f32).to_bits();
            return self.compute("f16_full", &[], dtype, len, params, false);
        }
        let count =
            words(dtype).ok_or_else(|| unsupported("full", Device::Wgpu(self.ordinal), dtype))?;
        let bits = match dtype {
            DType::F32 => (value as f32).to_bits() as u64,
            DType::Bool => u64::from(value != 0.0),
            DType::I64 => (value as i64) as u64,
            _ => unreachable!(),
        };
        let mut params = [0; 64];
        params[2] = count as u32;
        params[3] = bits as u32;
        params[4] = (bits >> 32) as u32;
        self.compute("full", &[], dtype, len, params, false)
    }

    fn cast(&self, x: View<'_>, to: DType) -> Result<Storage> {
        if x.dtype() == to {
            return self.copy_strided(x);
        }
        if x.dtype() == DType::F16 {
            if !supports_f16(x.device()) || !matches!(to, DType::F32 | DType::I64 | DType::Bool) {
                return Err(unsupported("cast", x.device(), x.dtype()));
            }
            let mut params = [0; 64];
            params[2] = match to {
                DType::F32 => 0,
                DType::Bool => 1,
                DType::I64 => 2,
                _ => unreachable!(),
            };
            params[3] = u32::from(x.layout().is_contiguous());
            return self.compute(
                "f16_cast_out",
                &[x],
                to,
                x.layout().num_elements(),
                params,
                false,
            );
        }
        if to == DType::F16 {
            if !supports_f16(x.device())
                || !matches!(x.dtype(), DType::F32 | DType::I64 | DType::Bool)
            {
                return Err(unsupported("cast", x.device(), x.dtype()));
            }
            let mut params = [0; 64];
            params[2] = match x.dtype() {
                DType::F32 => 0,
                DType::Bool => 1,
                DType::I64 => 2,
                _ => unreachable!(),
            };
            params[3] = u32::from(x.layout().is_contiguous());
            return self.compute(
                "f16_cast_in",
                &[x],
                DType::F16,
                x.layout().num_elements(),
                params,
                false,
            );
        }
        if !matches!(
            (x.dtype(), to),
            (DType::F32, DType::Bool) | (DType::Bool, DType::F32)
        ) {
            return Err(unsupported("cast", x.device(), x.dtype()));
        }
        let mut params = [0; 64];
        params[2] = u32::from(to == DType::Bool);
        params[3] = u32::from(x.layout().is_contiguous());
        self.compute(
            "cast_kernel",
            &[x],
            to,
            x.layout().num_elements(),
            params,
            false,
        )
    }

    fn binary(&self, op: BinaryOp, lhs: View<'_>, rhs: View<'_>) -> Result<Storage> {
        let dtype = self.float_dtype("binary", &[lhs, rhs])?;
        let mut p = [0; 64];
        p[2] = op_code_binary(op);
        p[5] = u32::from(lhs.layout().is_contiguous() && rhs.layout().is_contiguous());
        self.compute(
            if dtype == DType::F16 {
                "f16_binary"
            } else {
                "binary"
            },
            &[lhs, rhs],
            dtype,
            lhs.layout().num_elements(),
            p,
            false,
        )
    }
    fn binary_scalar(&self, op: BinaryOp, x: View<'_>, scalar: f64) -> Result<Storage> {
        let dtype = self.float_dtype("binary_scalar", &[x])?;
        let mut p = [0; 64];
        p[2] = op_code_binary(op);
        p[3] = (scalar as f32).to_bits();
        p[5] = u32::from(x.layout().is_contiguous());
        self.compute(
            if dtype == DType::F16 {
                "f16_binary_scalar"
            } else {
                "binary_scalar"
            },
            &[x],
            dtype,
            x.layout().num_elements(),
            p,
            false,
        )
    }
    fn unary(&self, op: UnaryOp, x: View<'_>) -> Result<Storage> {
        let dtype = self.float_dtype("unary", &[x])?;
        let mut p = [0; 64];
        p[2] = op_code_unary(op);
        p[5] = u32::from(x.layout().is_contiguous());
        self.compute(
            if dtype == DType::F16 {
                "f16_unary"
            } else {
                "unary"
            },
            &[x],
            dtype,
            x.layout().num_elements(),
            p,
            false,
        )
    }
    fn compare(&self, op: CmpOp, lhs: View<'_>, rhs: View<'_>) -> Result<Storage> {
        let dtype = self.float_dtype("compare", &[lhs, rhs])?;
        let mut p = [0; 64];
        p[2] = op_code_cmp(op);
        p[5] = u32::from(lhs.layout().is_contiguous() && rhs.layout().is_contiguous());
        self.compute(
            if dtype == DType::F16 {
                "f16_compare"
            } else {
                "compare"
            },
            &[lhs, rhs],
            DType::Bool,
            lhs.layout().num_elements(),
            p,
            false,
        )
    }
    fn where_cond(&self, cond: View<'_>, on_true: View<'_>, on_false: View<'_>) -> Result<Storage> {
        let dtype = self.float_dtype("where", &[on_true, on_false])?;
        if cond.dtype() != DType::Bool {
            return Err(unsupported("where", cond.device(), cond.dtype()));
        }
        let mut p = [0; 64];
        p[5] = u32::from(
            cond.layout().is_contiguous()
                && on_true.layout().is_contiguous()
                && on_false.layout().is_contiguous(),
        );
        let (entry, inputs) = if dtype == DType::F16 {
            ("f16_where", [on_true, cond, on_false])
        } else {
            ("where_cond", [cond, on_true, on_false])
        };
        self.compute(
            entry,
            &inputs,
            dtype,
            cond.layout().num_elements(),
            p,
            false,
        )
    }
    fn masked_fill(&self, x: View<'_>, mask: View<'_>, value: f64) -> Result<Storage> {
        let dtype = self.float_dtype("masked_fill", &[x])?;
        if mask.dtype() != DType::Bool {
            return Err(unsupported("masked_fill", mask.device(), mask.dtype()));
        }
        let mut p = [0; 64];
        p[2] = (value as f32).to_bits();
        p[5] = u32::from(x.layout().is_contiguous() && mask.layout().is_contiguous());
        self.compute(
            if dtype == DType::F16 {
                "f16_masked_fill"
            } else {
                "masked_fill"
            },
            &[x, mask],
            dtype,
            x.layout().num_elements(),
            p,
            false,
        )
    }

    fn reduce(&self, op: ReduceOp, x: View<'_>, axis: usize) -> Result<Storage> {
        let dtype = self.float_dtype("reduce", &[x])?;
        let count = x.layout().dims()[axis];
        let len = x
            .layout()
            .dims()
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis)
            .map(|(_, v)| v)
            .product();
        if count == 0 {
            return self.full(len, dtype, 0.0);
        }
        let mut p = [0; 64];
        p[2] = match op {
            ReduceOp::Sum => 0,
            ReduceOp::Mean => 1,
            ReduceOp::Max => 2,
            ReduceOp::Min => 3,
        };
        p[3] = axis as u32;
        p[4] = u32_checked(count, "reduce")?;
        p[5] = u32::from(axis + 1 == x.layout().rank() && x.layout().is_contiguous());
        p[0] = u32_checked(len, "reduce")?;
        descriptor(&mut p, 8, x.layout(), "reduce")?;
        let context = context(self.ordinal)?;
        let input = storage(x, "reduce")?;
        if !Arc::ptr_eq(&input.context, &context) {
            return Err(Error::DeviceMismatch {
                op: "reduce",
                expected: Device::Wgpu(self.ordinal),
                got: input.device(),
            });
        }
        let output = context.allocate(len * element_bytes(dtype).unwrap(), "reduce");
        context
            .dispatch(
                if dtype == DType::F16 {
                    "f16_reduce"
                } else {
                    "reduce"
                },
                &[input],
                output,
                len,
                len,
                dtype,
                &p,
                false,
                Some([u32_checked(len, "reduce")?, 1, 1]),
            )
            .map(Storage::Wgpu)
    }
    fn arg_reduce(&self, op: ArgReduceOp, x: View<'_>, axis: usize) -> Result<Storage> {
        let dtype = self.float_dtype("arg_reduce", &[x])?;
        let mut p = [0; 64];
        p[2] = u32::from(op == ArgReduceOp::ArgMin);
        p[3] = axis as u32;
        p[4] = u32_checked(x.layout().dims()[axis], "arg_reduce")?;
        self.compute(
            if dtype == DType::F16 {
                "f16_arg_reduce"
            } else {
                "arg_reduce"
            },
            &[x],
            DType::I64,
            x.layout().num_elements() / x.layout().dims()[axis],
            p,
            false,
        )
    }

    fn matmul(&self, lhs: View<'_>, rhs: View<'_>) -> Result<Storage> {
        let dtype = self.float_dtype("matmul", &[lhs, rhs])?;
        let ld = lhs.layout().dims();
        let rd = rhs.layout().dims();
        let (m, k, n) = (ld[ld.len() - 2], ld[ld.len() - 1], rd[rd.len() - 1]);
        let batch_dims = crate::shape::Shape::from(ld[..ld.len() - 2].to_vec()).broadcast_with(
            &crate::shape::Shape::from(rd[..rd.len() - 2].to_vec()),
            "matmul",
        )?;
        let batch = batch_dims.num_elements();
        let rank = batch_dims.rank() + 2;
        if rank > MAX_RANK {
            return Err(unsupported("matmul", lhs.device(), lhs.dtype()));
        }
        let expand = |layout: &Layout, dims: &[usize], batch: &[usize]| {
            let prefix = &dims[..dims.len() - 2];
            let pad = batch.len() - prefix.len();
            let mut vd = batch.to_vec();
            vd.extend_from_slice(&dims[dims.len() - 2..]);
            let mut vs = vec![0; pad];
            for (i, &d) in prefix.iter().enumerate() {
                vs.push(if d == 1 && batch[pad + i] != 1 {
                    0
                } else {
                    layout.strides()[i]
                });
            }
            vs.extend_from_slice(&layout.strides()[dims.len() - 2..]);
            (vd, vs)
        };
        let (av, as_) = expand(lhs.layout(), ld, batch_dims.dims());
        let (bv, bs) = expand(rhs.layout(), rd, batch_dims.dims());
        let mut p = [0; 64];
        p[0] = u32_checked(batch * m * n, "matmul")?;
        p[2] = m as u32;
        p[3] = n as u32;
        p[4] = k as u32;
        virtual_descriptor(&mut p, 8, lhs.layout(), &av, &as_, "matmul")?;
        virtual_descriptor(&mut p, 26, rhs.layout(), &bv, &bs, "matmul")?;
        let context = context(self.ordinal)?;
        let a = storage(lhs, "matmul")?;
        let b = storage(rhs, "matmul")?;
        let output = context.allocate(batch * m * n * element_bytes(dtype).unwrap(), "matmul");
        context
            .dispatch(
                if dtype == DType::F16 {
                    "f16_matmul"
                } else {
                    "matmul"
                },
                &[a, b],
                output,
                batch * m * n,
                batch * m * n,
                dtype,
                &p,
                false,
                None,
            )
            .map(Storage::Wgpu)
    }

    fn index_select(&self, x: View<'_>, axis: usize, indices: View<'_>) -> Result<Storage> {
        let dtype = x.dtype();
        if !matches!(dtype, DType::F16 | DType::F32 | DType::I64 | DType::Bool)
            || (dtype == DType::F16 && !supports_f16(Device::Wgpu(self.ordinal)))
        {
            return Err(unsupported("index_select", x.device(), dtype));
        }
        if indices.dtype() != DType::I64 {
            return Err(unsupported(
                "index_select",
                indices.device(),
                indices.dtype(),
            ));
        }
        let mut dims = x.layout().dims().to_vec();
        dims[axis] = indices.layout().num_elements();
        let out = Layout::contiguous(dims)?;
        let mut p = [0; 64];
        p[2] = axis as u32;
        p[3] = u32_checked(x.layout().dims()[axis], "index_select")?;
        p[1] = u32_checked(indices.layout().num_elements(), "index_select")?;
        descriptor(&mut p, 44, &out, "index_select")?;
        self.compute(
            match dtype {
                DType::F16 => "f16_index_select",
                DType::I64 => "i64_index_select",
                DType::F32 | DType::Bool => "index_select",
                _ => unreachable!(),
            },
            &[x, indices],
            dtype,
            out.num_elements(),
            p,
            true,
        )
    }
    fn index_add(
        &self,
        x: View<'_>,
        axis: usize,
        indices: View<'_>,
        src: View<'_>,
    ) -> Result<Storage> {
        let dtype = self.float_dtype("index_add", &[x, src])?;
        if indices.dtype() != DType::I64 {
            return Err(unsupported("index_add", indices.device(), indices.dtype()));
        }
        let mut p = [0; 64];
        p[2] = axis as u32;
        p[3] = x.layout().dims()[axis] as u32;
        p[4] = u32_checked(indices.layout().num_elements(), "index_add")?;
        p[1] = p[4];
        self.compute(
            if dtype == DType::F16 {
                "f16_index_add"
            } else {
                "index_add"
            },
            &[x, indices, src],
            dtype,
            x.layout().num_elements(),
            p,
            true,
        )
    }
    fn gather(&self, x: View<'_>, axis: usize, indices: View<'_>) -> Result<Storage> {
        let dtype = self.float_dtype("gather", &[x])?;
        if indices.dtype() != DType::I64 {
            return Err(unsupported("gather", indices.device(), indices.dtype()));
        }
        let mut p = [0; 64];
        p[2] = axis as u32;
        p[3] = x.layout().dims()[axis] as u32;
        p[1] = u32_checked(indices.layout().num_elements(), "gather")?;
        self.compute(
            if dtype == DType::F16 {
                "f16_gather"
            } else {
                "gather"
            },
            &[x, indices],
            dtype,
            indices.layout().num_elements(),
            p,
            true,
        )
    }
    fn scatter_add(
        &self,
        x: View<'_>,
        axis: usize,
        indices: View<'_>,
        src: View<'_>,
    ) -> Result<Storage> {
        let dtype = self.float_dtype("scatter_add", &[x, src])?;
        if indices.dtype() != DType::I64 {
            return Err(unsupported(
                "scatter_add",
                indices.device(),
                indices.dtype(),
            ));
        }
        let mut p = [0; 64];
        p[0] = x.layout().num_elements() as u32;
        p[2] = axis as u32;
        p[3] = x.layout().dims()[axis] as u32;
        p[4] = u32_checked(indices.layout().dims()[axis], "scatter_add")?;
        p[1] = u32_checked(indices.layout().num_elements(), "scatter_add")?;
        descriptor(&mut p, 8, x.layout(), "scatter_add")?;
        descriptor(&mut p, 26, indices.layout(), "scatter_add")?;
        virtual_descriptor(
            &mut p,
            44,
            src.layout(),
            indices.layout().dims(),
            src.layout().strides(),
            "scatter_add",
        )?;
        let context = context(self.ordinal)?;
        let values = [
            storage(x, "scatter_add")?,
            storage(indices, "scatter_add")?,
            storage(src, "scatter_add")?,
        ];
        let output = context.allocate(
            x.layout().num_elements() * element_bytes(dtype).unwrap(),
            "scatter_add",
        );
        context
            .dispatch(
                if dtype == DType::F16 {
                    "f16_scatter_add"
                } else {
                    "scatter_add"
                },
                &values,
                output,
                x.layout().num_elements(),
                x.layout().num_elements(),
                dtype,
                &p,
                true,
                None,
            )
            .map(Storage::Wgpu)
    }

    fn conv(&self, op: ConvOp, inputs: &[View<'_>], params: &Conv2dParams) -> Result<Storage> {
        let dtype = self.float_dtype("conv", inputs)?;
        let geo = match op {
            ConvOp::Conv2d => Conv2dGeometry::conv2d(
                "conv2d",
                inputs[0].layout().dims(),
                inputs[1].layout().dims(),
                params,
            )?,
            ConvOp::MaxPool2d | ConvOp::AvgPool2d => {
                Conv2dGeometry::pool("pool2d", inputs[0].layout().dims(), params)?
            }
            ConvOp::Conv2dInputGrad => Conv2dGeometry::conv2d(
                "conv2d_backward",
                inputs[2].layout().dims(),
                inputs[1].layout().dims(),
                params,
            )?,
            ConvOp::Conv2dWeightGrad => Conv2dGeometry::conv2d(
                "conv2d_backward",
                inputs[1].layout().dims(),
                inputs[2].layout().dims(),
                params,
            )?,
            ConvOp::MaxPool2dBackward | ConvOp::AvgPool2dBackward => {
                Conv2dGeometry::pool("pool2d_backward", inputs[1].layout().dims(), params)?
            }
        };
        let output_dims = match op {
            ConvOp::Conv2d | ConvOp::MaxPool2d | ConvOp::AvgPool2d => geo.output_dims(),
            ConvOp::Conv2dInputGrad | ConvOp::MaxPool2dBackward | ConvOp::AvgPool2dBackward => {
                geo.input_dims()
            }
            ConvOp::Conv2dWeightGrad => geo.weight_dims(),
        };
        let d = geo.input_dims();
        let w = geo.weight_dims();
        let o = geo.output_dims();
        let mut p = [0; 64];
        p[2] = match op {
            ConvOp::Conv2d => 0,
            ConvOp::MaxPool2d => 1,
            ConvOp::AvgPool2d => 2,
            ConvOp::Conv2dInputGrad => 3,
            ConvOp::Conv2dWeightGrad => 4,
            ConvOp::MaxPool2dBackward => 5,
            ConvOp::AvgPool2dBackward => 6,
        };
        p[3..8].copy_from_slice(&[
            d[0] as u32,
            d[1] as u32,
            d[2] as u32,
            d[3] as u32,
            w[0] as u32,
        ]);
        p[14..18].copy_from_slice(&[w[2] as u32, w[3] as u32, o[2] as u32, o[3] as u32]);
        p[22..26].copy_from_slice(&[
            params.stride.0 as u32,
            params.stride.1 as u32,
            params.padding.0 as u32,
            params.padding.1 as u32,
        ]);
        let pool = matches!(
            op,
            ConvOp::MaxPool2d
                | ConvOp::AvgPool2d
                | ConvOp::MaxPool2dBackward
                | ConvOp::AvgPool2dBackward
        );
        p[32] = if pool { 1 } else { params.dilation.0 as u32 };
        p[33] = if pool { 1 } else { params.dilation.1 as u32 };
        self.compute(
            if dtype == DType::F16 {
                "f16_conv"
            } else {
                "conv"
            },
            inputs,
            dtype,
            output_dims.into_iter().product(),
            p,
            false,
        )
    }

    fn fused(&self, op: FusedOp, inputs: &[View<'_>], scalars: &[f64]) -> Result<Vec<Storage>> {
        let dtype = inputs.first().map_or(DType::F32, View::dtype);
        let entry = match (op, inputs, scalars) {
            (FusedOp::Softmax, [x], []) if x.dtype() == DType::F32 => "softmax",
            (FusedOp::Softmax, [x], [])
                if x.dtype() == DType::F16 && supports_f16(Device::Wgpu(self.ordinal)) =>
            {
                "f16_softmax"
            }
            (FusedOp::LayerNorm, [x, weight, bias], [_])
                if x.dtype() == DType::F32
                    && weight.dtype() == DType::F32
                    && bias.dtype() == DType::F32 =>
            {
                "layer_norm"
            }
            (FusedOp::LayerNorm, [x, weight, bias], [_])
                if x.dtype() == DType::F16
                    && weight.dtype() == DType::F16
                    && bias.dtype() == DType::F16
                    && supports_f16(Device::Wgpu(self.ordinal)) =>
            {
                "f16_layer_norm"
            }
            _ => {
                return Err(unsupported(
                    match op {
                        FusedOp::Softmax => "softmax",
                        FusedOp::LayerNorm => "layer_norm",
                        FusedOp::SgdStep => "sgd_step",
                        FusedOp::AdamStep => "adam_step",
                    },
                    Device::Wgpu(self.ordinal),
                    dtype,
                ));
            }
        };
        let x = inputs[0];
        let width = *x.layout().dims().last().ok_or_else(|| Error::InvalidArg {
            op: entry,
            msg: "fused row operation requires rank at least one".to_owned(),
        })?;
        let rows = x.layout().num_elements() / width;
        let context = context(self.ordinal)?;
        let values = inputs
            .iter()
            .map(|view| storage(*view, entry))
            .collect::<Result<Vec<_>>>()?;
        if values
            .iter()
            .any(|value| !Arc::ptr_eq(&value.context, &context))
        {
            return Err(Error::DeviceMismatch {
                op: entry,
                expected: Device::Wgpu(self.ordinal),
                got: inputs[0].device(),
            });
        }
        let mut params = [0; 64];
        params[0] = u32_checked(x.layout().num_elements(), entry)?;
        params[2] = u32_checked(rows, entry)?;
        params[3] = u32_checked(width, entry)?;
        params[5] = u32::from(inputs.iter().all(|view| view.layout().is_contiguous()));
        if let Some(&eps) = scalars.first() {
            params[4] = (eps as f32).to_bits();
        }
        for (view, base) in inputs.iter().zip([8usize, 26, 44]) {
            descriptor(&mut params, base, view.layout(), entry)?;
        }
        let output = context.allocate(
            x.layout().num_elements() * element_bytes(dtype).unwrap(),
            entry,
        );
        let result = context.dispatch(
            entry,
            &values,
            output,
            x.layout().num_elements(),
            rows,
            dtype,
            &params,
            false,
            Some([u32_checked(rows, entry)?, 1, 1]),
        )?;
        Ok(vec![Storage::Wgpu(result)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter_available() -> bool {
        if std::env::var_os("RSTORCH_SKIP_WGPU_TESTS").is_some() {
            eprintln!("skipping WGPU hardware test: RSTORCH_SKIP_WGPU_TESTS is set");
            return false;
        }
        context(0).unwrap_or_else(|error| {
            panic!(
                "WGPU feature enabled but adapter initialization failed: {error}. Set \
                 RSTORCH_SKIP_WGPU_TESTS=1 only when this test environment intentionally has no WGPU adapter"
            )
        });
        true
    }

    #[test]
    fn transfer_and_compute_if_adapter_exists() -> Result<()> {
        if !adapter_available() {
            return Ok(());
        }
        let device = Device::Wgpu(0);
        let x = crate::Tensor::from_vec(vec![1.0f32, -2.0, 3.0], [3], &device).unwrap();
        assert_eq!(
            x.relu().unwrap().to_vec::<f32>().unwrap(),
            vec![1.0, 0.0, 3.0]
        );
        let indices = crate::Tensor::from_vec(vec![2i64, 0], [2], &device).unwrap();
        assert_eq!(
            x.index_select(0, &indices)
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            vec![3.0, 1.0]
        );
        assert_eq!(indices.to_vec::<i64>().unwrap(), vec![2, 0]);
        assert_eq!(
            indices
                .index_select(
                    0,
                    &crate::Tensor::from_vec(vec![1i64, 0], [2], &device).unwrap(),
                )
                .unwrap()
                .to_vec::<i64>()
                .unwrap(),
            vec![0, 2]
        );
        let wide = crate::Tensor::from_vec(
            vec![i64::MIN, -1, i64::MAX, i64::from(u32::MAX) + 1],
            [2, 2],
            &device,
        )?
        .transpose(0, 1)?;
        assert_eq!(
            wide.index_select(0, &crate::Tensor::from_vec(vec![1i64, 0], [2], &device)?,)?
                .to_vec::<i64>()?,
            vec![-1, i64::from(u32::MAX) + 1, i64::MIN, i64::MAX]
        );
        let flags = crate::Tensor::from_vec(vec![true, false, true], [3], &device)?;
        assert_eq!(
            flags.index_select(0, &indices)?.to_vec::<bool>()?,
            vec![true, true]
        );

        let narrowed = crate::Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2], &device)
            .unwrap()
            .narrow(0, 1, 1)
            .unwrap();
        assert_eq!(narrowed.to_vec::<f32>().unwrap(), vec![3.0, 4.0]);
        Ok(())
    }

    #[test]
    fn f16_is_native_or_loudly_unsupported() {
        if !adapter_available() {
            return;
        }
        let device = Device::Wgpu(0);
        let result = crate::Tensor::from_vec(
            vec![half::f16::from_f32(1.5), half::f16::from_f32(-2.0)],
            [2],
            &device,
        );
        if supports_f16(device) {
            let values = result
                .unwrap()
                .relu()
                .unwrap()
                .to_vec::<half::f16>()
                .unwrap();
            assert_eq!(values, vec![half::f16::from_f32(1.5), half::f16::ZERO]);
        } else {
            assert!(matches!(
                result,
                Err(Error::Unsupported {
                    dtype: DType::F16,
                    ..
                })
            ));
        }
    }

    #[test]
    fn conformance_if_adapter_exists() {
        if !adapter_available() {
            return;
        }
        let report = crate::backend::conformance::run_device(Device::Wgpu(0));
        assert!(
            report.skipped.is_empty(),
            "unexpected unsupported rows: {:?}",
            report.skipped
        );
        report.into_result(Device::Wgpu(0)).unwrap();
    }
}
