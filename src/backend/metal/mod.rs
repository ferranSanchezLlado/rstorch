//! Apple Metal backend: ordinal-scoped shared contexts and ordered async
//! command encoding. Kernel implementations are extended in T61b.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use ::metal as metal_rs;
use metal_rs::objc::rc::autoreleasepool;

use crate::backend::{
    ArgReduceOp, BackendOps, BinaryOp, CmpOp, Conv2dParams, ConvOp, FusedOp, ReduceOp, UnaryOp,
    View,
};
use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::storage::{CpuStorage, Storage};

const SOURCE: &str = include_str!("kernels.metal");
const COMMIT_THRESHOLD: usize = 64;
type ContextResult = std::result::Result<Arc<Context>, String>;
type ContextRegistry = Mutex<HashMap<usize, ContextResult>>;

pub(crate) struct MetalBackend {
    ordinal: usize,
}

#[derive(Clone)]
pub(crate) struct MetalStorage {
    buffer: Arc<metal_rs::Buffer>,
    dtype: DType,
    len: usize,
    context: Arc<Context>,
}

struct Context {
    ordinal: usize,
    raw: metal_rs::Device,
    queue: metal_rs::CommandQueue,
    library: metal_rs::Library,
    pipelines: Mutex<HashMap<String, Arc<metal_rs::ComputePipelineState>>>,
    submission: Mutex<Submission>,
}

struct Submission {
    open: Option<OpenBuffer>,
    pending: Vec<PendingBuffer>,
}

struct OpenBuffer {
    command: metal_rs::CommandBuffer,
    encoder: metal_rs::ComputeCommandEncoder,
    resources: Vec<metal_rs::Buffer>,
    dispatches: usize,
}

struct PendingBuffer {
    command: metal_rs::CommandBuffer,
    _resources: Vec<metal_rs::Buffer>,
}

pub(crate) fn backend(ordinal: usize) -> &'static dyn BackendOps {
    static BACKENDS: OnceLock<Mutex<HashMap<usize, &'static MetalBackend>>> = OnceLock::new();
    let mut backends = BACKENDS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("Metal backend registry poisoned");
    *backends
        .entry(ordinal)
        .or_insert_with(|| Box::leak(Box::new(MetalBackend { ordinal })))
}

fn context(ordinal: usize) -> Result<Arc<Context>> {
    static CONTEXTS: OnceLock<ContextRegistry> = OnceLock::new();
    let mut contexts = CONTEXTS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("Metal context registry poisoned");
    let value = contexts
        .entry(ordinal)
        .or_insert_with(|| create_context(ordinal));
    value.clone().map_err(|msg| Error::Backend {
        op: "metal_device",
        msg,
    })
}

fn create_context(ordinal: usize) -> std::result::Result<Arc<Context>, String> {
    autoreleasepool(|| {
        let devices = metal_rs::Device::all();
        let raw = devices.into_iter().nth(ordinal).ok_or_else(|| {
            format!("invalid Metal device ordinal {ordinal}; no such device is available")
        })?;
        let options = metal_rs::CompileOptions::new();
        let library = raw
            .new_library_with_source(SOURCE, &options)
            .map_err(|e| format!("runtime shader compilation failed: {e}"))?;
        let queue = raw.new_command_queue();
        Ok(Arc::new(Context {
            ordinal,
            raw,
            queue,
            library,
            pipelines: Mutex::new(HashMap::new()),
            submission: Mutex::new(Submission {
                open: None,
                pending: Vec::new(),
            }),
        }))
    })
}

impl MetalStorage {
    pub(crate) fn dtype(&self) -> DType {
        self.dtype
    }

    pub(crate) fn device(&self) -> Device {
        Device::Metal(self.context.ordinal)
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

fn metal_storage<'a>(op: &'static str, view: View<'a>) -> Result<&'a MetalStorage> {
    match view.storage() {
        Storage::Metal(storage) => Ok(storage),
        Storage::Cpu(_) => Err(Error::DeviceMismatch {
            op,
            expected: view.device(),
            got: Device::Cpu,
        }),
    }
}

fn check_context(op: &'static str, context: &Arc<Context>, views: &[View<'_>]) -> Result<()> {
    for view in views {
        let storage = metal_storage(op, *view)?;
        if !Arc::ptr_eq(context, &storage.context) {
            return Err(Error::DeviceMismatch {
                op,
                expected: Device::Metal(context.ordinal),
                got: storage.device(),
            });
        }
    }
    Ok(())
}

fn unsupported(op: &'static str, view: View<'_>) -> Error {
    Error::Unsupported {
        op,
        device: view.device(),
        dtype: view.dtype(),
    }
}

fn supported_dtype(op: &'static str, dtype: DType, device: Device) -> Result<()> {
    if matches!(dtype, DType::F16 | DType::F32 | DType::I64 | DType::Bool) {
        Ok(())
    } else {
        Err(Error::Unsupported { op, device, dtype })
    }
}

fn byte_len(dtype: DType, len: usize) -> u64 {
    let size = match dtype {
        DType::F16 => 2,
        DType::F32 => 4,
        DType::I64 => 8,
        DType::Bool => 1,
        DType::BF16 => 2,
        DType::F64 => 8,
    };
    len.max(1).saturating_mul(size) as u64
}

fn allocate(context: &Arc<Context>, dtype: DType, len: usize) -> MetalStorage {
    let buffer = context.raw.new_buffer(
        byte_len(dtype, len),
        metal_rs::MTLResourceOptions::StorageModeShared,
    );
    MetalStorage {
        buffer: Arc::new(buffer),
        dtype,
        len,
        context: Arc::clone(context),
    }
}

fn pipeline(context: &Context, name: &str) -> Result<Arc<metal_rs::ComputePipelineState>> {
    let mut pipelines = context
        .pipelines
        .lock()
        .expect("Metal pipeline cache poisoned");
    if let Some(pipeline) = pipelines.get(name) {
        return Ok(Arc::clone(pipeline));
    }
    let pipeline = autoreleasepool(|| {
        let function = context
            .library
            .get_function(name, None)
            .map_err(|e| Error::Backend {
                op: "metal_pipeline",
                msg: e,
            })?;
        context
            .raw
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| Error::Backend {
                op: "metal_pipeline",
                msg: e,
            })
    })?;
    let pipeline = Arc::new(pipeline);
    pipelines.insert(name.to_owned(), Arc::clone(&pipeline));
    Ok(pipeline)
}

fn reap(submission: &mut Submission) -> Result<()> {
    let mut first_error = None;
    submission
        .pending
        .retain(|pending| match pending.command.status() {
            metal_rs::MTLCommandBufferStatus::Completed => false,
            metal_rs::MTLCommandBufferStatus::Error => {
                first_error = Some(Error::Backend {
                    op: "metal_command",
                    msg: "Metal command buffer completed with an error".to_owned(),
                });
                false
            }
            _ => true,
        });
    first_error.map_or(Ok(()), Err)
}

fn commit_open(submission: &mut Submission) {
    if let Some(open) = submission.open.take() {
        open.encoder.end_encoding();
        open.command.commit();
        submission.pending.push(PendingBuffer {
            command: open.command,
            _resources: open.resources,
        });
    }
}

fn new_open(context: &Context) -> OpenBuffer {
    let command = context.queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    OpenBuffer {
        command: command.to_owned(),
        encoder: encoder.to_owned(),
        resources: Vec::new(),
        dispatches: 0,
    }
}

fn encode(
    context: &Arc<Context>,
    pipeline: &metal_rs::ComputePipelineStateRef,
    len: usize,
    resources: &[&metal_rs::Buffer],
    set_args: impl FnOnce(&metal_rs::ComputeCommandEncoderRef),
) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    autoreleasepool(|| {
        let mut submission = context
            .submission
            .lock()
            .expect("Metal submission state poisoned");
        reap(&mut submission)?;
        let open = submission.open.get_or_insert_with(|| new_open(context));
        open.encoder.set_compute_pipeline_state(pipeline);
        set_args(&open.encoder);
        let width = pipeline
            .thread_execution_width()
            .min(pipeline.max_total_threads_per_threadgroup())
            .max(1);
        open.encoder.dispatch_thread_groups(
            metal_rs::MTLSize::new(len.div_ceil(width as usize) as u64, 1, 1),
            metal_rs::MTLSize::new(width, 1, 1),
        );
        open.resources
            .extend(resources.iter().map(|buffer| (*buffer).clone()));
        open.dispatches += 1;
        if open.dispatches >= COMMIT_THRESHOLD {
            commit_open(&mut submission);
        }
        Ok(())
    })
}

fn synchronize(context: &Arc<Context>) -> Result<()> {
    autoreleasepool(|| {
        let pending = {
            let mut submission = context
                .submission
                .lock()
                .expect("Metal submission state poisoned");
            commit_open(&mut submission);
            std::mem::take(&mut submission.pending)
        };
        for pending in pending {
            pending.command.wait_until_completed();
            if pending.command.status() == metal_rs::MTLCommandBufferStatus::Error {
                return Err(Error::Backend {
                    op: "transfer_out",
                    msg: "Metal command buffer completed with an error".to_owned(),
                });
            }
        }
        Ok(())
    })
}

fn set_bytes<T>(encoder: &metal_rs::ComputeCommandEncoderRef, index: u64, values: &[T]) {
    encoder.set_bytes(
        index,
        std::mem::size_of_val(values) as u64,
        values.as_ptr().cast::<c_void>(),
    );
}

fn layout_args(encoder: &metal_rs::ComputeCommandEncoderRef, start: u64, layout: &Layout) {
    let dims: Vec<u64> = layout.dims().iter().map(|&v| v as u64).collect();
    let strides: Vec<u64> = layout.strides().iter().map(|&v| v as u64).collect();
    set_bytes(encoder, start, &dims);
    set_bytes(encoder, start + 1, &strides);
    set_bytes(encoder, start + 2, &[layout.rank() as u32]);
    set_bytes(encoder, start + 3, &[layout.offset() as u64]);
}

fn copy_name(dtype: DType, into: bool) -> &'static str {
    match (dtype, into) {
        (DType::F16, false) => "copy_f16",
        (DType::F32, false) => "copy_f32",
        (DType::I64, false) => "copy_i64",
        (DType::Bool, false) => "copy_bool",
        (DType::F16, true) => "copy_into_f16",
        (DType::F32, true) => "copy_into_f32",
        (DType::I64, true) => "copy_into_i64",
        (DType::Bool, true) => "copy_into_bool",
        _ => unreachable!("dtype validated before copy pipeline lookup"),
    }
}

impl BackendOps for MetalBackend {
    fn transfer_in(&self, host: CpuStorage) -> Result<Storage> {
        let context = context(self.ordinal)?;
        let dtype = host.dtype();
        supported_dtype("from_vec", dtype, Device::Metal(self.ordinal))?;
        let storage = allocate(&context, dtype, host.len());
        unsafe {
            match host {
                CpuStorage::F16(values) => std::ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    storage.buffer.contents().cast::<half::f16>(),
                    values.len(),
                ),
                CpuStorage::F32(values) => std::ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    storage.buffer.contents().cast::<f32>(),
                    values.len(),
                ),
                CpuStorage::I64(values) => std::ptr::copy_nonoverlapping(
                    values.as_ptr(),
                    storage.buffer.contents().cast::<i64>(),
                    values.len(),
                ),
                CpuStorage::Bool(values) => {
                    let dst = storage.buffer.contents().cast::<u8>();
                    for (index, value) in values.iter().enumerate() {
                        dst.add(index).write(u8::from(*value));
                    }
                }
                CpuStorage::BF16(_) | CpuStorage::F64(_) => unreachable!("dtype validated"),
            }
        }
        Ok(Storage::Metal(storage))
    }

    fn transfer_out(&self, x: View<'_>) -> Result<CpuStorage> {
        metal_storage("transfer_out", x)?;
        let context = context(self.ordinal)?;
        check_context("transfer_out", &context, &[x])?;
        let dense = self.copy_strided(x)?;
        synchronize(&context)?;
        let Storage::Metal(storage) = dense else {
            unreachable!()
        };
        unsafe {
            Ok(match storage.dtype {
                DType::F16 => CpuStorage::F16(Arc::new(
                    std::slice::from_raw_parts(
                        storage.buffer.contents().cast::<half::f16>(),
                        storage.len,
                    )
                    .to_vec(),
                )),
                DType::F32 => CpuStorage::F32(Arc::new(
                    std::slice::from_raw_parts(
                        storage.buffer.contents().cast::<f32>(),
                        storage.len,
                    )
                    .to_vec(),
                )),
                DType::I64 => CpuStorage::I64(Arc::new(
                    std::slice::from_raw_parts(
                        storage.buffer.contents().cast::<i64>(),
                        storage.len,
                    )
                    .to_vec(),
                )),
                DType::Bool => CpuStorage::Bool(Arc::new(
                    std::slice::from_raw_parts(storage.buffer.contents().cast::<u8>(), storage.len)
                        .iter()
                        .map(|&value| value != 0)
                        .collect(),
                )),
                DType::BF16 | DType::F64 => unreachable!("unsupported storage cannot exist"),
            })
        }
    }

    fn copy_strided(&self, x: View<'_>) -> Result<Storage> {
        let input = metal_storage("copy_strided", x)?;
        supported_dtype("copy_strided", x.dtype(), x.device())?;
        let context = context(self.ordinal)?;
        check_context("copy_strided", &context, &[x])?;
        let output = allocate(&context, x.dtype(), x.layout().num_elements());
        let pipe = pipeline(&context, copy_name(x.dtype(), false))?;
        encode(
            &context,
            &pipe,
            output.len,
            &[&input.buffer, &output.buffer],
            |encoder| {
                encoder.set_buffer(0, Some(&input.buffer), 0);
                encoder.set_buffer(1, Some(&output.buffer), 0);
                layout_args(encoder, 2, x.layout());
                set_bytes(encoder, 6, &[output.len as u64]);
            },
        )?;
        Ok(Storage::Metal(output))
    }

    fn copy_into(&self, src: View<'_>, dst: &mut Storage, dst_layout: &Layout) -> Result<()> {
        if src.layout().shape() != dst_layout.shape() {
            return Err(Error::ShapeMismatch {
                op: "copy_into",
                lhs: src.layout().shape().clone(),
                rhs: dst_layout.shape().clone(),
            });
        }
        let input = metal_storage("copy_into", src)?;
        let Storage::Metal(output) = dst else {
            return Err(Error::DeviceMismatch {
                op: "copy_into",
                expected: src.device(),
                got: dst.device(),
            });
        };
        if input.dtype != output.dtype {
            return Err(Error::DTypeMismatch {
                op: "copy_into",
                expected: input.dtype,
                got: output.dtype,
            });
        }
        if !Arc::ptr_eq(&input.context, &output.context) {
            return Err(Error::DeviceMismatch {
                op: "copy_into",
                expected: input.device(),
                got: output.device(),
            });
        }
        let pipe = pipeline(&input.context, copy_name(input.dtype, true))?;
        encode(
            &input.context,
            &pipe,
            src.layout().num_elements(),
            &[&input.buffer, &output.buffer],
            |encoder| {
                encoder.set_buffer(0, Some(&input.buffer), 0);
                encoder.set_buffer(1, Some(&output.buffer), 0);
                layout_args(encoder, 2, src.layout());
                layout_args(encoder, 6, dst_layout);
                set_bytes(encoder, 10, &[src.layout().num_elements() as u64]);
            },
        )
    }

    fn full(&self, len: usize, dtype: DType, value: f64) -> Result<Storage> {
        let context = context(self.ordinal)?;
        supported_dtype("full", dtype, Device::Metal(self.ordinal))?;
        let storage = allocate(&context, dtype, len);
        unsafe {
            match dtype {
                DType::F16 => std::slice::from_raw_parts_mut(
                    storage.buffer.contents().cast::<half::f16>(),
                    len,
                )
                .fill(half::f16::from_f64(value)),
                DType::F32 => {
                    std::slice::from_raw_parts_mut(storage.buffer.contents().cast::<f32>(), len)
                        .fill(value as f32)
                }
                DType::I64 => {
                    std::slice::from_raw_parts_mut(storage.buffer.contents().cast::<i64>(), len)
                        .fill(value as i64)
                }
                DType::Bool => {
                    std::slice::from_raw_parts_mut(storage.buffer.contents().cast::<u8>(), len)
                        .fill(u8::from(value != 0.0))
                }
                DType::BF16 | DType::F64 => unreachable!("dtype validated"),
            }
        }
        Ok(Storage::Metal(storage))
    }

    fn cast(&self, x: View<'_>, _to: DType) -> Result<Storage> {
        Err(unsupported("to_dtype", x))
    }
    fn binary(&self, _op: BinaryOp, lhs: View<'_>, _rhs: View<'_>) -> Result<Storage> {
        Err(unsupported("binary", lhs))
    }
    fn binary_scalar(&self, _op: BinaryOp, x: View<'_>, _scalar: f64) -> Result<Storage> {
        Err(unsupported("binary_scalar", x))
    }
    fn unary(&self, _op: UnaryOp, x: View<'_>) -> Result<Storage> {
        Err(unsupported("unary", x))
    }
    fn compare(&self, _op: CmpOp, lhs: View<'_>, _rhs: View<'_>) -> Result<Storage> {
        Err(unsupported("compare", lhs))
    }
    fn where_cond(
        &self,
        _cond: View<'_>,
        on_true: View<'_>,
        _on_false: View<'_>,
    ) -> Result<Storage> {
        Err(unsupported("where", on_true))
    }
    fn masked_fill(&self, x: View<'_>, _mask: View<'_>, _value: f64) -> Result<Storage> {
        Err(unsupported("masked_fill", x))
    }
    fn reduce(&self, _op: ReduceOp, x: View<'_>, _axis: usize) -> Result<Storage> {
        Err(unsupported("reduce", x))
    }
    fn arg_reduce(&self, _op: ArgReduceOp, x: View<'_>, _axis: usize) -> Result<Storage> {
        Err(unsupported("arg_reduce", x))
    }
    fn matmul(&self, lhs: View<'_>, _rhs: View<'_>) -> Result<Storage> {
        Err(unsupported("matmul", lhs))
    }
    fn index_select(&self, x: View<'_>, _axis: usize, _indices: View<'_>) -> Result<Storage> {
        Err(unsupported("index_select", x))
    }
    fn index_add(
        &self,
        x: View<'_>,
        _axis: usize,
        _indices: View<'_>,
        _src: View<'_>,
    ) -> Result<Storage> {
        Err(unsupported("index_add", x))
    }
    fn gather(&self, x: View<'_>, _axis: usize, _indices: View<'_>) -> Result<Storage> {
        Err(unsupported("gather", x))
    }
    fn scatter_add(
        &self,
        x: View<'_>,
        _axis: usize,
        _indices: View<'_>,
        _src: View<'_>,
    ) -> Result<Storage> {
        Err(unsupported("scatter_add", x))
    }
    fn conv(&self, _op: ConvOp, inputs: &[View<'_>], _params: &Conv2dParams) -> Result<Storage> {
        let Some(input) = inputs.first() else {
            return Err(Error::InvalidArg {
                op: "conv",
                msg: "missing input".to_owned(),
            });
        };
        Err(unsupported("conv", *input))
    }
    fn fused(&self, _op: FusedOp, inputs: &[View<'_>], _scalars: &[f64]) -> Result<Vec<Storage>> {
        let Some(input) = inputs.first() else {
            return Err(Error::InvalidArg {
                op: "fused",
                msg: "missing input".to_owned(),
            });
        };
        Err(unsupported("fused", *input))
    }
}

#[cfg(test)]
mod tests {
    use crate::{DType, Device, Error, Tensor};

    const METAL: Device = Device::Metal(0);

    #[test]
    fn storage_transfer_and_strided_copy_run_on_hardware() {
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], &METAL)
            .expect("Metal device 0 must exist in the hardware lane");
        assert_eq!(x.device(), METAL);
        assert_eq!(x.dtype(), DType::F32);
        assert_eq!(
            x.transpose(0, 1).unwrap().to_vec::<f32>().unwrap(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn cat_and_stack_are_device_side() {
        let a = Tensor::from_vec(vec![1i64, 2], [2], &METAL).unwrap();
        let b = Tensor::from_vec(vec![3i64, 4], [2], &METAL).unwrap();
        assert_eq!(
            Tensor::cat(&[&a, &b], 0).unwrap().to_vec::<i64>().unwrap(),
            vec![1, 2, 3, 4]
        );
        assert_eq!(
            Tensor::stack(&[&a, &b], 1)
                .unwrap()
                .to_vec::<i64>()
                .unwrap(),
            vec![1, 3, 2, 4]
        );
    }

    #[test]
    fn invalid_ordinal_is_a_structured_backend_error() {
        let err = Tensor::zeros([1], DType::F32, &Device::Metal(usize::MAX));
        assert!(matches!(
            err,
            Err(Error::Backend {
                op: "metal_device",
                ..
            })
        ));
    }
}
