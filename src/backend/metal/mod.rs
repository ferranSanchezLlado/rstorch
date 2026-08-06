//! Apple Metal backend: ordinal-scoped shared contexts and ordered async
//! command encoding. Kernel implementations are extended in T61b.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

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
/// Threads per threadgroup to aim for; see [`threads_per_group`].
const TARGET_THREADS_PER_GROUP: u64 = 256;
const MAX_GRID_SIZE: usize = u32::MAX as usize;
type ContextResult = std::result::Result<Arc<Context>, String>;
type ContextRegistry = Mutex<HashMap<usize, ContextResult>>;

#[cfg(test)]
#[derive(Default)]
struct Instrumentation {
    dispatches: AtomicUsize,
    commits: AtomicUsize,
    waits: AtomicUsize,
    transfer_in: AtomicUsize,
    transfer_out: AtomicUsize,
    max_dispatches_per_buffer: AtomicUsize,
    max_pending: AtomicUsize,
}

#[cfg(test)]
static INSTRUMENTATION: Instrumentation = Instrumentation {
    dispatches: AtomicUsize::new(0),
    commits: AtomicUsize::new(0),
    waits: AtomicUsize::new(0),
    transfer_in: AtomicUsize::new(0),
    transfer_out: AtomicUsize::new(0),
    max_dispatches_per_buffer: AtomicUsize::new(0),
    max_pending: AtomicUsize::new(0),
};

#[cfg(test)]
fn observe_max(counter: &AtomicUsize, value: usize) {
    counter.fetch_max(value, Ordering::Relaxed);
}

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
    pipelines: Mutex<HashMap<PipelineKey, Arc<metal_rs::ComputePipelineState>>>,
    submission: Mutex<Submission>,
    pool: Mutex<BufferPool>,
    validation: Mutex<Validation>,
}

/// Index bounds checks that have been **encoded but not yet read back**.
///
/// # Why this is deferred
///
/// Every indexed op — `index_select`, `gather`, `index_add`, `scatter_add` —
/// runs a validation kernel first. Reading its verdict immediately meant
/// committing the open command buffer and waiting for the whole queue to
/// drain, in the middle of encoding. An embedding lookup therefore forced a
/// full GPU round trip, and a transformer step paid several: removing those
/// waits measured −69% on forward and −36% on a full AdamW step.
///
/// So the verdict is now collected at the next host boundary instead, which is
/// exactly where the backend already synchronizes (see the module-level
/// asynchronous-result contract). An out-of-range index still produces the same
/// [`Error::IndexOutOfBounds`] naming the same op, index, axis, and size — but
/// it surfaces from the next operation that reads device memory back to the
/// host (`to_vec`, `to_scalar`, `item`) rather than from the indexing call
/// itself. This is the same bargain CUDA makes with asynchronous launches, and
/// no incorrect value can be observed in the meantime: the host cannot see any
/// result without passing through the check.
///
/// The shader's own per-thread guards are unchanged and still clamp reads, so a
/// bad index is never a memory-safety question — only a reporting one.
struct Validation {
    /// Two `i64` per slot — `[failed, offending_index]` — written by the
    /// validation kernel at a per-slot byte offset. Fixed capacity so it is
    /// never reallocated while a command buffer still references it.
    status: Arc<metal_rs::Buffer>,
    /// Host-side descriptors, one per encoded slot, in program order.
    pending: Vec<PendingValidation>,
}

/// What an encoded-but-unread bounds check would report if it failed.
struct PendingValidation {
    op: &'static str,
    axis: usize,
    bound: usize,
}

/// Slots in the validation status buffer. Reaching this many un-read checks
/// forces an early drain, which simply restores the old synchronous cost for
/// that one op rather than losing a verdict.
const VALIDATION_SLOTS: usize = 256;
/// Bytes per slot: two `i64`.
const VALIDATION_SLOT_BYTES: u64 = 16;

struct Submission {
    open: Option<OpenBuffer>,
    pending: Vec<PendingBuffer>,
}

struct OpenBuffer {
    command: metal_rs::CommandBuffer,
    encoder: metal_rs::ComputeCommandEncoder,
    resources: Vec<Arc<metal_rs::Buffer>>,
    dispatches: usize,
}

struct PendingBuffer {
    command: metal_rs::CommandBuffer,
    _resources: Vec<Arc<metal_rs::Buffer>>,
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
        let status = Arc::new(raw.new_buffer(
            VALIDATION_SLOTS as u64 * VALIDATION_SLOT_BYTES,
            metal_rs::MTLResourceOptions::StorageModeShared,
        ));
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
            pool: Mutex::new(BufferPool::default()),
            validation: Mutex::new(Validation {
                status,
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

fn element_size(dtype: DType) -> usize {
    match dtype {
        DType::F16 => 2,
        DType::F32 => 4,
        DType::I64 => 8,
        DType::Bool => 1,
        DType::BF16 => 2,
        DType::F64 => 8,
    }
}

fn byte_len(dtype: DType, len: usize) -> Result<u64> {
    let size = element_size(dtype);
    let bytes = len
        .max(1)
        .checked_mul(size)
        .ok_or_else(|| Error::InvalidArg {
            op: "metal_allocate",
            msg: format!("buffer byte length overflows usize for {len} {dtype} elements"),
        })?;
    u64::try_from(bytes).map_err(|_| Error::InvalidArg {
        op: "metal_allocate",
        msg: format!("buffer byte length {bytes} exceeds u64::MAX"),
    })
}

/// A caching allocator for device buffers, keyed by byte size.
///
/// `newBufferWithLength:` measures 1.7–7 µs on an M4 Pro, and this backend
/// allocates one buffer per op output. A transformer step encodes hundreds of
/// ops, so allocation alone accounted for milliseconds — far more than the
/// kernels themselves, which is why the backend was losing to CPU on small
/// models rather than on arithmetic.
///
/// # How a buffer is known to be free
///
/// The pool keeps an [`Arc`] to every buffer it has ever handed out and never
/// returns one whose `strong_count` exceeds 1. That single reference is the
/// pool's own, so a count of 1 proves that
///
/// - no live [`MetalStorage`] holds it (tensors clone the `Arc`), **and**
/// - no un-reaped command buffer references it — [`OpenBuffer::resources`] and
///   [`PendingBuffer`] hold `Arc` clones for exactly as long as the GPU may
///   touch the buffer, and [`reap`] drops them only after the command buffer
///   reports completion.
///
/// Tracking the `Arc` rather than the storage's lifetime is the whole safety
/// argument: recycling on `MetalStorage` drop alone would hand a buffer to a
/// new op while a command buffer still in flight was writing it.
///
/// Buffers are cached rather than freed, so the pool settles at the peak
/// concurrent footprint per size class. That is the same bargain PyTorch's
/// caching allocator makes.
#[derive(Default)]
struct BufferPool {
    by_size: HashMap<u64, Vec<Arc<metal_rs::Buffer>>>,
}

impl BufferPool {
    /// A buffer of exactly `bytes`, recycled if one is idle.
    fn take(&mut self, device: &metal_rs::Device, bytes: u64) -> Arc<metal_rs::Buffer> {
        let slots = self.by_size.entry(bytes).or_default();
        if let Some(idle) = slots
            .iter()
            .find(|buffer| Arc::strong_count(buffer) == 1)
            .map(Arc::clone)
        {
            return idle;
        }
        let buffer =
            Arc::new(device.new_buffer(bytes, metal_rs::MTLResourceOptions::StorageModeShared));
        slots.push(Arc::clone(&buffer));
        buffer
    }
}

fn allocate(context: &Arc<Context>, dtype: DType, len: usize) -> Result<MetalStorage> {
    let bytes = byte_len(dtype, len)?;
    let buffer = context
        .pool
        .lock()
        .expect("Metal buffer pool poisoned")
        .take(&context.raw, bytes);
    Ok(MetalStorage {
        buffer,
        dtype,
        len,
        context: Arc::clone(context),
    })
}

/// A cache key identifying one compiled shader.
///
/// The key is built from `Copy` parts rather than the shader's name string so
/// that a cache **hit** — the overwhelmingly common case, once a model is
/// warm — costs no allocation. Formatting `"reduce_f32"` on every dispatch put
/// a heap allocation on the hot path of a backend whose whole problem is
/// per-op overhead; the MSL name is now built only on a miss.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum PipelineKey {
    /// A shader whose name is fixed (`validate_indices`, the copy kernels).
    Named(&'static str),
    /// A dtype-specialized family: `("reduce", F32)` names `reduce_f32`.
    Typed(&'static str, DType),
    /// `cast_{from}_to_{to}`.
    Cast(DType, DType),
}

impl PipelineKey {
    /// The Metal function name this key selects. Only called on a cache miss.
    fn shader_name(self) -> String {
        match self {
            PipelineKey::Named(name) => name.to_owned(),
            PipelineKey::Typed(base, dtype) => format!("{base}_{}", suffix(dtype)),
            PipelineKey::Cast(from, to) => {
                format!("cast_{}_to_{}", suffix(from), suffix(to))
            }
        }
    }
}

/// The pipeline for a dtype-specialized shader family.
fn pipeline_typed(
    context: &Context,
    base: &'static str,
    dtype: DType,
) -> Result<Arc<metal_rs::ComputePipelineState>> {
    pipeline(context, PipelineKey::Typed(base, dtype))
}

fn pipeline(context: &Context, key: PipelineKey) -> Result<Arc<metal_rs::ComputePipelineState>> {
    let mut pipelines = context
        .pipelines
        .lock()
        .expect("Metal pipeline cache poisoned");
    if let Some(pipeline) = pipelines.get(&key) {
        return Ok(Arc::clone(pipeline));
    }
    let name = key.shader_name();
    let pipeline = autoreleasepool(|| {
        let function = context
            .library
            .get_function(&name, None)
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
    pipelines.insert(key, Arc::clone(&pipeline));
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
        #[cfg(test)]
        {
            INSTRUMENTATION.commits.fetch_add(1, Ordering::Relaxed);
            observe_max(&INSTRUMENTATION.max_dispatches_per_buffer, open.dispatches);
        }
        open.encoder.end_encoding();
        open.command.commit();
        submission.pending.push(PendingBuffer {
            command: open.command,
            _resources: open.resources,
        });
        #[cfg(test)]
        observe_max(&INSTRUMENTATION.max_pending, submission.pending.len());
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
    resources: &[&Arc<metal_rs::Buffer>],
    set_args: impl FnOnce(&metal_rs::ComputeCommandEncoderRef),
) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    if len > MAX_GRID_SIZE {
        return Err(Error::InvalidArg {
            op: "metal_dispatch",
            msg: format!("grid size {len} exceeds u32::MAX"),
        });
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
        let group = threads_per_group(pipeline);
        open.encoder.dispatch_thread_groups(
            metal_rs::MTLSize::new(len.div_ceil(group as usize) as u64, 1, 1),
            metal_rs::MTLSize::new(group, 1, 1),
        );
        open.resources
            .extend(resources.iter().map(|buffer| Arc::clone(buffer)));
        open.dispatches += 1;
        #[cfg(test)]
        INSTRUMENTATION.dispatches.fetch_add(1, Ordering::Relaxed);
        if open.dispatches >= COMMIT_THRESHOLD {
            commit_open(&mut submission);
        }
        Ok(())
    })
}

/// Threads per threadgroup this shader should be dispatched with.
///
/// The previous rule was `thread_execution_width().min(max_total)`, which on
/// every Apple GPU is `min(32, 1024)` — one SIMD group per threadgroup. That
/// leaves the hardware badly under-occupied: a GPU core interleaves several
/// SIMD groups to hide memory latency, and with one per group it has nothing to
/// switch to. It also multiplies the threadgroup count, and hence the
/// dispatch-side bookkeeping, by 8× against the value below.
///
/// [`TARGET_THREADS_PER_GROUP`] is preferred over the 1024-thread maximum
/// because a large group raises register pressure and can *reduce* the number
/// of groups resident per core. The result is rounded down to a whole number of
/// SIMD groups so no partial group is dispatched, and is never zero.
fn threads_per_group(pipeline: &metal_rs::ComputePipelineStateRef) -> u64 {
    let width = pipeline.thread_execution_width().max(1);
    let max = pipeline.max_total_threads_per_threadgroup().max(width);
    let target = TARGET_THREADS_PER_GROUP.min(max);
    (target / width).max(1) * width
}

/// Drain the queue, then report any bounds check that failed while it ran.
///
/// This is the host boundary. Deferred [`Validation`] verdicts are collected
/// here — after the wait, so the status buffer is complete — which is what
/// lets the indexed ops encode without a round trip of their own. A drain
/// failure is reported ahead of a bounds failure: a command buffer that errored
/// may be why a verdict never landed.
fn synchronize(context: &Arc<Context>) -> Result<()> {
    autoreleasepool(|| {
        // Keep submission serialization through the wait. Otherwise another
        // host reader can take this reader's pending buffers and return before
        // the commands producing its storage have completed.
        let mut submission = context
            .submission
            .lock()
            .expect("Metal submission state poisoned");
        commit_open(&mut submission);
        drain_results(
            std::mem::take(&mut submission.pending)
                .into_iter()
                .map(|pending| {
                    #[cfg(test)]
                    INSTRUMENTATION.waits.fetch_add(1, Ordering::Relaxed);
                    pending.command.wait_until_completed();
                    if pending.command.status() == metal_rs::MTLCommandBufferStatus::Error {
                        Err(Error::Backend {
                            op: "transfer_out",
                            msg: "Metal command buffer completed with an error".to_owned(),
                        })
                    } else {
                        Ok(())
                    }
                }),
        )
    })?;
    collect_validations(context)
}

fn drain_results(results: impl IntoIterator<Item = Result<()>>) -> Result<()> {
    let mut first_error = None;
    for result in results {
        if let Err(error) = result
            && first_error.is_none()
        {
            first_error = Some(error);
        }
    }
    first_error.map_or(Ok(()), Err)
}

fn checked_product(op: &'static str, values: impl IntoIterator<Item = usize>) -> Result<usize> {
    values.into_iter().try_fold(1usize, |product, value| {
        product.checked_mul(value).ok_or_else(|| Error::InvalidArg {
            op,
            msg: "output element count overflows usize".to_owned(),
        })
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

fn suffix(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => "f16",
        DType::F32 => "f32",
        DType::I64 => "i64",
        DType::Bool => "bool",
        DType::BF16 | DType::F64 => unreachable!("unsupported Metal dtype"),
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

fn same_dtype(op: &'static str, views: &[View<'_>]) -> Result<DType> {
    let Some(first) = views.first() else {
        return Err(Error::InvalidArg {
            op,
            msg: "missing input".to_owned(),
        });
    };
    for view in &views[1..] {
        if view.dtype() != first.dtype() {
            return Err(Error::DTypeMismatch {
                op,
                expected: first.dtype(),
                got: view.dtype(),
            });
        }
        if view.device() != first.device() {
            return Err(Error::DeviceMismatch {
                op,
                expected: first.device(),
                got: view.device(),
            });
        }
    }
    supported_dtype(op, first.dtype(), first.device())?;
    Ok(first.dtype())
}

fn check_axis(op: &'static str, view: View<'_>, axis: usize) -> Result<()> {
    if axis < view.layout().rank() {
        Ok(())
    } else {
        Err(Error::InvalidAxis {
            op,
            axis: axis as isize,
            rank: view.layout().rank(),
        })
    }
}

fn validate_fused_optimizer_views(
    op: &'static str,
    inputs: &[View<'_>],
    parameter_inputs: usize,
) -> Result<()> {
    let param = inputs[0];
    let acc_dtype = if param.dtype() == DType::F16 {
        DType::F32
    } else {
        param.dtype()
    };
    for (index, &input) in inputs.iter().enumerate() {
        let expected = if index < parameter_inputs {
            param.dtype()
        } else {
            acc_dtype
        };
        if input.dtype() != expected {
            return Err(Error::DTypeMismatch {
                op,
                expected,
                got: input.dtype(),
            });
        }
        if input.layout().shape() != param.layout().shape() {
            return Err(Error::ShapeMismatch {
                op,
                lhs: param.layout().shape().clone(),
                rhs: input.layout().shape().clone(),
            });
        }
    }
    Ok(())
}

fn validate_indices(op: &'static str, indices: View<'_>, axis: usize, bound: usize) -> Result<()> {
    let context = context(match indices.device() {
        Device::Metal(ordinal) => ordinal,
        Device::Cpu => {
            return Err(Error::DeviceMismatch {
                op,
                expected: indices.device(),
                got: Device::Cpu,
            });
        }
    })?;
    let input = metal_storage(op, indices)?;

    // A full slot table means the verdicts must be collected before this check
    // can claim a slot. Draining here costs what every check used to cost, and
    // only after 256 un-read checks.
    let slot = {
        let claimed = context
            .validation
            .lock()
            .expect("Metal validation state poisoned")
            .pending
            .len();
        if claimed < VALIDATION_SLOTS {
            claimed
        } else {
            // Not holding the lock: `synchronize` collects and clears the queue.
            synchronize(&context)?;
            0
        }
    };

    let pipe = pipeline(&context, PipelineKey::Named("validate_indices"))?;
    let status = Arc::clone(
        &context
            .validation
            .lock()
            .expect("Metal validation state poisoned")
            .status,
    );
    let offset = slot as u64 * VALIDATION_SLOT_BYTES;
    encode(&context, &pipe, 1, &[&input.buffer, &status], |encoder| {
        encoder.set_buffer(0, Some(&input.buffer), 0);
        encoder.set_buffer(1, Some(&status), offset);
        layout_args(encoder, 2, indices.layout());
        set_bytes(encoder, 6, &[indices.layout().num_elements() as u64]);
        set_bytes(encoder, 7, &[bound as u64]);
    })?;

    context
        .validation
        .lock()
        .expect("Metal validation state poisoned")
        .pending
        .push(PendingValidation { op, axis, bound });
    Ok(())
}

/// Collect every encoded bounds check and report the first failure in program
/// order.
///
/// Called from [`synchronize`] once the queue has drained, so the status buffer
/// is complete. Clears the queue either way: a reported failure is reported
/// once, and the slots are reused from zero.
fn collect_validations(context: &Arc<Context>) -> Result<()> {
    let mut validation = context
        .validation
        .lock()
        .expect("Metal validation state poisoned");
    if validation.pending.is_empty() {
        return Ok(());
    }
    // SAFETY: `status` is a `StorageModeShared` buffer of exactly
    // `VALIDATION_SLOTS * 2` `i64`s, so `contents()` is a non-null,
    // CPU-readable, 8-byte-aligned pointer to that many elements. The caller
    // has waited for every command buffer to complete, so the GPU is no longer
    // writing it, and the slice does not outlive the borrow of `validation`.
    // Only the first `pending.len()` slots have been written by a kernel; the
    // loop below reads no further.
    let words = unsafe {
        std::slice::from_raw_parts(
            validation.status.contents().cast::<i64>(),
            VALIDATION_SLOTS * 2,
        )
    };
    let failure = validation
        .pending
        .iter()
        .enumerate()
        .find(|(slot, _)| words[slot * 2] != 0)
        .map(|(slot, check)| Error::IndexOutOfBounds {
            op: check.op,
            index: words[slot * 2 + 1],
            axis: check.axis,
            size: check.bound,
        });
    validation.pending.clear();
    failure.map_or(Ok(()), Err)
}

fn output_for(context: &Arc<Context>, dtype: DType, len: usize) -> Result<MetalStorage> {
    allocate(context, dtype, len)
}

/// The output element count of a reduction of `layout` over `axis`: the product
/// of every *other* dimension.
///
/// Deliberately a product rather than `num_elements() / dims()[axis]`: an empty
/// reduction axis makes that division by zero, which panics instead of
/// returning the reduction's identity the way the CPU backend does. Empty axes
/// are reachable (`narrow(axis, i, 0)`, an empty batch), and `sum` over one is
/// legal.
fn reduced_len(layout: &Layout, axis: usize) -> usize {
    layout
        .dims()
        .iter()
        .enumerate()
        .filter(|&(a, _)| a != axis)
        .map(|(_, dim)| dim)
        .product()
}

fn encode_binary(
    backend: &MetalBackend,
    name: &'static str,
    lhs: View<'_>,
    rhs: View<'_>,
    output_dtype: DType,
    code: u32,
) -> Result<Storage> {
    let dtype = same_dtype(name, &[lhs, rhs])?;
    let context = context(backend.ordinal)?;
    check_context(name, &context, &[lhs, rhs])?;
    let a = metal_storage(name, lhs)?;
    let b = metal_storage(name, rhs)?;
    let output = output_for(&context, output_dtype, lhs.layout().num_elements())?;
    let pipe = pipeline_typed(&context, name, dtype)?;
    encode(
        &context,
        &pipe,
        output.len,
        &[&a.buffer, &b.buffer, &output.buffer],
        |encoder| {
            encoder.set_buffer(0, Some(&a.buffer), 0);
            encoder.set_buffer(1, Some(&b.buffer), 0);
            encoder.set_buffer(2, Some(&output.buffer), 0);
            layout_args(encoder, 3, lhs.layout());
            layout_args(encoder, 7, rhs.layout());
            set_bytes(encoder, 11, &[output.len as u64]);
            set_bytes(encoder, 12, &[code]);
        },
    )?;
    Ok(Storage::Metal(output))
}

struct MatmulPlan {
    batch: Vec<u64>,
    lhs_batch: Vec<u64>,
    rhs_batch: Vec<u64>,
    params: [u64; 9],
    len: usize,
}

fn matmul_plan(lhs: &Layout, rhs: &Layout) -> Result<MatmulPlan> {
    if lhs.rank() < 2 || rhs.rank() < 2 {
        return Err(Error::InvalidArg {
            op: "matmul",
            msg: "operands must be rank >= 2".to_owned(),
        });
    }
    let (lr, rr) = (lhs.rank(), rhs.rank());
    let (m, k, rk, n) = (
        lhs.dims()[lr - 2],
        lhs.dims()[lr - 1],
        rhs.dims()[rr - 2],
        rhs.dims()[rr - 1],
    );
    if k != rk {
        return Err(Error::ShapeMismatch {
            op: "matmul",
            lhs: lhs.shape().clone(),
            rhs: rhs.shape().clone(),
        });
    }
    let lb = &lhs.dims()[..lr - 2];
    let rb = &rhs.dims()[..rr - 2];
    let rank = lb.len().max(rb.len());
    let mut batch = vec![0u64; rank];
    let mut lhs_batch = vec![0u64; rank];
    let mut rhs_batch = vec![0u64; rank];
    for axis in 0..rank {
        let li = (axis + lb.len()).checked_sub(rank);
        let ri = (axis + rb.len()).checked_sub(rank);
        let ld = li.map_or(1, |i| lb[i]);
        let rd = ri.map_or(1, |i| rb[i]);
        let dim = if ld == rd {
            ld
        } else if ld == 1 {
            rd
        } else if rd == 1 {
            ld
        } else {
            return Err(Error::ShapeMismatch {
                op: "matmul",
                lhs: lhs.shape().clone(),
                rhs: rhs.shape().clone(),
            });
        };
        batch[axis] = dim as u64;
        lhs_batch[axis] = li
            .filter(|&i| lb[i] != 1)
            .map_or(0, |i| lhs.strides()[i] as u64);
        rhs_batch[axis] = ri
            .filter(|&i| rb[i] != 1)
            .map_or(0, |i| rhs.strides()[i] as u64);
    }
    let batches = batch.iter().try_fold(1usize, |count, &dim| {
        count
            .checked_mul(dim as usize)
            .ok_or_else(|| Error::InvalidArg {
                op: "matmul",
                msg: "broadcast batch size overflows usize".to_owned(),
            })
    })?;
    let len = batches
        .checked_mul(m)
        .and_then(|value| value.checked_mul(n))
        .ok_or_else(|| Error::InvalidArg {
            op: "matmul",
            msg: "output element count overflows usize".to_owned(),
        })?;
    Ok(MatmulPlan {
        batch,
        lhs_batch,
        rhs_batch,
        params: [
            m as u64,
            k as u64,
            n as u64,
            lhs.offset() as u64,
            rhs.offset() as u64,
            lhs.strides()[lr - 2] as u64,
            lhs.strides()[lr - 1] as u64,
            rhs.strides()[rr - 2] as u64,
            rhs.strides()[rr - 1] as u64,
        ],
        len,
    })
}

fn conv_params(
    geometry: &crate::backend::conv_geometry::Conv2dGeometry,
    params: &Conv2dParams,
) -> [u64; 15] {
    let [n, ci, h, w] = geometry.input_dims();
    let [co, _, kh, kw] = geometry.weight_dims();
    let [_, _, oh, ow] = geometry.output_dims();
    [
        n,
        ci,
        h,
        w,
        co,
        kh,
        kw,
        oh,
        ow,
        params.stride.0,
        params.stride.1,
        params.padding.0,
        params.padding.1,
        params.dilation.0,
        params.dilation.1,
    ]
    .map(|value| value as u64)
}

impl MetalBackend {
    fn fused_layer_norm(&self, inputs: &[View<'_>], scalars: &[f64]) -> Result<Vec<Storage>> {
        if inputs.len() == 4 {
            if !scalars.is_empty() {
                return Err(Error::InvalidArg {
                    op: "fused_layer_norm_backward_input",
                    msg: "backward accepts no scalars".to_owned(),
                });
            }
            let [grad, xhat, inv_std, weight] = inputs else {
                unreachable!()
            };
            if !matches!(grad.dtype(), DType::F16 | DType::F32)
                || weight.dtype() != grad.dtype()
                || xhat.dtype() != DType::F32
                || inv_std.dtype() != DType::F32
            {
                return Err(unsupported("fused_layer_norm_backward_input", *grad));
            }
            let width = grad.layout().dims()[grad.layout().rank() - 1];
            let rows = grad.layout().num_elements() / width;
            let context = context(self.ordinal)?;
            check_context("fused_layer_norm_backward_input", &context, inputs)?;
            let g = metal_storage("fused_layer_norm_backward_input", *grad)?;
            let h = metal_storage("fused_layer_norm_backward_input", *xhat)?;
            let i = metal_storage("fused_layer_norm_backward_input", *inv_std)?;
            let w = metal_storage("fused_layer_norm_backward_input", *weight)?;
            let output = output_for(&context, grad.dtype(), grad.layout().num_elements())?;
            let pipe = pipeline_typed(&context, "layer_norm_backward", grad.dtype())?;
            encode(
                &context,
                &pipe,
                rows,
                &[&g.buffer, &h.buffer, &i.buffer, &w.buffer, &output.buffer],
                |encoder| {
                    encoder.set_buffer(0, Some(&g.buffer), 0);
                    encoder.set_buffer(1, Some(&h.buffer), 0);
                    encoder.set_buffer(2, Some(&i.buffer), 0);
                    encoder.set_buffer(3, Some(&w.buffer), 0);
                    encoder.set_buffer(4, Some(&output.buffer), 0);
                    layout_args(encoder, 5, grad.layout());
                    layout_args(encoder, 9, xhat.layout());
                    set_bytes(
                        encoder,
                        13,
                        &inv_std
                            .layout()
                            .strides()
                            .iter()
                            .map(|&v| v as u64)
                            .collect::<Vec<_>>(),
                    );
                    set_bytes(encoder, 14, &[inv_std.layout().offset() as u64]);
                    set_bytes(
                        encoder,
                        15,
                        &weight
                            .layout()
                            .strides()
                            .iter()
                            .map(|&v| v as u64)
                            .collect::<Vec<_>>(),
                    );
                    set_bytes(encoder, 16, &[weight.layout().offset() as u64]);
                    set_bytes(encoder, 17, &[rows as u64]);
                    set_bytes(encoder, 18, &[width as u64]);
                },
            )?;
            return Ok(vec![Storage::Metal(output)]);
        }

        if inputs.len() != 3 || !(scalars.len() == 1 || scalars.len() == 2) {
            return Err(Error::InvalidArg {
                op: "fused_layer_norm",
                msg: "expected three inputs and one or two scalars".to_owned(),
            });
        }
        let [x, weight, bias] = inputs else {
            unreachable!()
        };
        let dtype = same_dtype("fused_layer_norm", inputs)?;
        if !matches!(dtype, DType::F16 | DType::F32) {
            return Err(unsupported("fused_layer_norm", *x));
        }
        let width = x.layout().dims()[x.layout().rank() - 1];
        let rows = x.layout().num_elements() / width;
        let save = scalars.get(1).is_some_and(|&value| value == 1.0);
        let context = context(self.ordinal)?;
        check_context("fused_layer_norm", &context, inputs)?;
        let xv = metal_storage("fused_layer_norm", *x)?;
        let wv = metal_storage("fused_layer_norm", *weight)?;
        let bv = metal_storage("fused_layer_norm", *bias)?;
        let output = output_for(&context, dtype, x.layout().num_elements())?;
        let xhat = output_for(&context, DType::F32, x.layout().num_elements())?;
        let inv = output_for(&context, DType::F32, rows)?;
        let pipe = pipeline_typed(&context, "layer_norm", dtype)?;
        encode(
            &context,
            &pipe,
            rows,
            &[
                &xv.buffer,
                &wv.buffer,
                &bv.buffer,
                &output.buffer,
                &xhat.buffer,
                &inv.buffer,
            ],
            |encoder| {
                encoder.set_buffer(0, Some(&xv.buffer), 0);
                encoder.set_buffer(1, Some(&wv.buffer), 0);
                encoder.set_buffer(2, Some(&bv.buffer), 0);
                encoder.set_buffer(3, Some(&output.buffer), 0);
                encoder.set_buffer(4, Some(&xhat.buffer), 0);
                encoder.set_buffer(5, Some(&inv.buffer), 0);
                layout_args(encoder, 6, x.layout());
                set_bytes(
                    encoder,
                    10,
                    &weight
                        .layout()
                        .strides()
                        .iter()
                        .map(|&v| v as u64)
                        .collect::<Vec<_>>(),
                );
                set_bytes(encoder, 11, &[weight.layout().offset() as u64]);
                set_bytes(
                    encoder,
                    12,
                    &bias
                        .layout()
                        .strides()
                        .iter()
                        .map(|&v| v as u64)
                        .collect::<Vec<_>>(),
                );
                set_bytes(encoder, 13, &[bias.layout().offset() as u64]);
                set_bytes(encoder, 14, &[rows as u64]);
                set_bytes(encoder, 15, &[width as u64]);
                set_bytes(encoder, 16, &[scalars[0] as f32]);
                set_bytes(encoder, 17, &[u32::from(save)]);
            },
        )?;
        let mut outputs = vec![Storage::Metal(output)];
        if save {
            outputs.push(Storage::Metal(xhat));
            outputs.push(Storage::Metal(inv));
        }
        Ok(outputs)
    }

    fn fused_sgd(&self, inputs: &[View<'_>], scalars: &[f64]) -> Result<Vec<Storage>> {
        if !(inputs.len() == 2 || inputs.len() == 3) || scalars.len() != 3 {
            return Err(Error::InvalidArg {
                op: "fused_sgd_step",
                msg: "expected two or three inputs and three scalars".to_owned(),
            });
        }
        let [lr, momentum, decay] = scalars else {
            unreachable!()
        };
        let dtype = same_dtype("fused_sgd_step", &inputs[..2])?;
        if !matches!(dtype, DType::F16 | DType::F32) {
            return Err(unsupported("fused_sgd_step", inputs[0]));
        }
        if let Some(velocity) = inputs.get(2)
            && velocity.dtype() != DType::F32
        {
            return Err(Error::DTypeMismatch {
                op: "fused_sgd_step",
                expected: DType::F32,
                got: velocity.dtype(),
            });
        }
        validate_fused_optimizer_views("fused_sgd_step", inputs, 2)?;
        let context = context(self.ordinal)?;
        check_context("fused_sgd_step", &context, inputs)?;
        let dense = inputs
            .iter()
            .map(|&input| self.copy_strided(input))
            .collect::<Result<Vec<_>>>()?;
        let values: Vec<&MetalStorage> = dense
            .iter()
            .map(|storage| match storage {
                Storage::Metal(value) => value,
                Storage::Cpu(_) => unreachable!(),
            })
            .collect();
        let p = values[0];
        let g = values[1];
        let velocity = values.get(2).copied();
        let next = output_for(&context, dtype, p.len)?;
        let next_velocity = output_for(&context, DType::F32, p.len)?;
        let pipe = pipeline_typed(&context, "sgd", dtype)?;
        let hp = [*lr as f32, *momentum as f32, *decay as f32];
        let mut resources = vec![&p.buffer, &g.buffer, &next.buffer, &next_velocity.buffer];
        if let Some(value) = velocity {
            resources.push(&value.buffer);
        }
        let use_momentum = *momentum != 0.0;
        encode(&context, &pipe, p.len, &resources, |encoder| {
            encoder.set_buffer(0, Some(&p.buffer), 0);
            encoder.set_buffer(1, Some(&g.buffer), 0);
            encoder.set_buffer(2, velocity.map(|value| &**value.buffer), 0);
            encoder.set_buffer(3, Some(&next.buffer), 0);
            encoder.set_buffer(4, Some(&next_velocity.buffer), 0);
            set_bytes(encoder, 5, &[p.len as u64]);
            set_bytes(encoder, 6, &hp);
            set_bytes(encoder, 7, &[u32::from(velocity.is_some())]);
            set_bytes(encoder, 8, &[u32::from(use_momentum)]);
        })?;
        let mut outputs = vec![Storage::Metal(next)];
        if use_momentum {
            outputs.push(Storage::Metal(next_velocity));
        }
        Ok(outputs)
    }

    fn fused_adam(&self, inputs: &[View<'_>], scalars: &[f64]) -> Result<Vec<Storage>> {
        if inputs.len() != 4 || scalars.len() != 8 {
            return Err(Error::InvalidArg {
                op: "fused_adam_step",
                msg: "expected four inputs and eight scalars".to_owned(),
            });
        }
        let dtype = same_dtype("fused_adam_step", &inputs[..2])?;
        if !matches!(dtype, DType::F16 | DType::F32)
            || inputs[2].dtype() != DType::F32
            || inputs[3].dtype() != DType::F32
        {
            return Err(unsupported("fused_adam_step", inputs[0]));
        }
        validate_fused_optimizer_views("fused_adam_step", inputs, 2)?;
        let context = context(self.ordinal)?;
        check_context("fused_adam_step", &context, inputs)?;
        let dense = inputs
            .iter()
            .map(|&input| self.copy_strided(input))
            .collect::<Result<Vec<_>>>()?;
        let values: Vec<&MetalStorage> = dense
            .iter()
            .map(|storage| match storage {
                Storage::Metal(value) => value,
                Storage::Cpu(_) => unreachable!(),
            })
            .collect();
        let [p, g, m, v] = values.as_slice() else {
            unreachable!()
        };
        let next = output_for(&context, dtype, p.len)?;
        let next_m = output_for(&context, DType::F32, p.len)?;
        let next_v = output_for(&context, DType::F32, p.len)?;
        let pipe = pipeline_typed(&context, "adam", dtype)?;
        let hp: Vec<f32> = scalars.iter().map(|&value| value as f32).collect();
        encode(
            &context,
            &pipe,
            p.len,
            &[
                &p.buffer,
                &g.buffer,
                &m.buffer,
                &v.buffer,
                &next.buffer,
                &next_m.buffer,
                &next_v.buffer,
            ],
            |encoder| {
                encoder.set_buffer(0, Some(&p.buffer), 0);
                encoder.set_buffer(1, Some(&g.buffer), 0);
                encoder.set_buffer(2, Some(&m.buffer), 0);
                encoder.set_buffer(3, Some(&v.buffer), 0);
                encoder.set_buffer(4, Some(&next.buffer), 0);
                encoder.set_buffer(5, Some(&next_m.buffer), 0);
                encoder.set_buffer(6, Some(&next_v.buffer), 0);
                set_bytes(encoder, 7, &[p.len as u64]);
                set_bytes(encoder, 8, &hp);
            },
        )?;
        Ok(vec![
            Storage::Metal(next),
            Storage::Metal(next_m),
            Storage::Metal(next_v),
        ])
    }
}

impl BackendOps for MetalBackend {
    fn transfer_in(&self, host: CpuStorage) -> Result<Storage> {
        #[cfg(test)]
        INSTRUMENTATION.transfer_in.fetch_add(1, Ordering::Relaxed);
        let context = context(self.ordinal)?;
        let dtype = host.dtype();
        supported_dtype("from_vec", dtype, Device::Metal(self.ordinal))?;
        let storage = allocate(&context, dtype, host.len())?;
        // SAFETY: `storage` holds `host.len()` elements of `dtype`, which is
        // `host`'s own dtype, in a `StorageModeShared` (CPU-writable) buffer.
        // Each arm below casts `contents()` to the matching element type and
        // copies exactly `values.len() == host.len()` elements, so the write
        // stays in bounds and correctly aligned. Source and destination are
        // distinct allocations, so `copy_nonoverlapping` is satisfied.
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
        #[cfg(test)]
        INSTRUMENTATION.transfer_out.fetch_add(1, Ordering::Relaxed);
        metal_storage("transfer_out", x)?;
        let context = context(self.ordinal)?;
        check_context("transfer_out", &context, &[x])?;
        let dense = self.copy_strided(x)?;
        synchronize(&context)?;
        let Storage::Metal(storage) = dense else {
            unreachable!()
        };
        // SAFETY: `storage` is the dense copy produced by `copy_strided`, so it
        // holds `storage.len` initialized elements of `storage.dtype` in a
        // `StorageModeShared` (CPU-readable) buffer, and the matched arm casts
        // `contents()` to that same element type. `synchronize` above has
        // completed, so the GPU is no longer writing the buffer. Each slice is
        // copied with `to_vec` before `storage` is dropped.
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
        let output = allocate(&context, x.dtype(), x.layout().num_elements())?;
        let pipe = pipeline(&context, PipelineKey::Named(copy_name(x.dtype(), false)))?;
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
        let pipe = pipeline(
            &input.context,
            PipelineKey::Named(copy_name(input.dtype, true)),
        )?;
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
        let storage = allocate(&context, dtype, len)?;
        // SAFETY: `storage` holds `len` elements of `dtype` in a
        // `StorageModeShared` (CPU-writable) buffer, and each arm casts
        // `contents()` to the element type matching the `dtype` it matched on,
        // then fills exactly `len` of them. Nothing has been enqueued against
        // this freshly allocated buffer, so no GPU work races the fill.
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

    fn cast(&self, x: View<'_>, to: DType) -> Result<Storage> {
        supported_dtype("to_dtype", x.dtype(), x.device())?;
        supported_dtype("to_dtype", to, x.device())?;
        if x.dtype() == to {
            return self.copy_strided(x);
        }
        let context = context(self.ordinal)?;
        check_context("to_dtype", &context, &[x])?;
        let input = metal_storage("to_dtype", x)?;
        let output = output_for(&context, to, x.layout().num_elements())?;
        let pipe = pipeline(&context, PipelineKey::Cast(x.dtype(), to))?;
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
    fn binary(&self, op: BinaryOp, lhs: View<'_>, rhs: View<'_>) -> Result<Storage> {
        encode_binary(self, "binary", lhs, rhs, lhs.dtype(), op_code_binary(op))
    }
    fn binary_scalar(&self, op: BinaryOp, x: View<'_>, scalar: f64) -> Result<Storage> {
        supported_dtype("binary_scalar", x.dtype(), x.device())?;
        if x.dtype() == DType::Bool {
            return Err(unsupported("binary_scalar", x));
        }
        let context = context(self.ordinal)?;
        check_context("binary_scalar", &context, &[x])?;
        let input = metal_storage("binary_scalar", x)?;
        let output = output_for(&context, x.dtype(), x.layout().num_elements())?;
        let pipe = pipeline_typed(&context, "scalar", x.dtype())?;
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
                if x.dtype() == DType::I64 {
                    set_bytes(encoder, 7, &[scalar as i64]);
                } else {
                    set_bytes(encoder, 7, &[scalar as f32]);
                }
                set_bytes(encoder, 8, &[op_code_binary(op)]);
            },
        )?;
        Ok(Storage::Metal(output))
    }
    fn unary(&self, op: UnaryOp, x: View<'_>) -> Result<Storage> {
        supported_dtype("unary", x.dtype(), x.device())?;
        if x.dtype() == DType::Bool
            || (x.dtype() == DType::I64 && !matches!(op, UnaryOp::Neg | UnaryOp::Abs))
        {
            return Err(unsupported("unary", x));
        }
        let context = context(self.ordinal)?;
        check_context("unary", &context, &[x])?;
        let input = metal_storage("unary", x)?;
        let output = output_for(&context, x.dtype(), x.layout().num_elements())?;
        let pipe = pipeline_typed(&context, "unary", x.dtype())?;
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
                set_bytes(encoder, 7, &[op_code_unary(op)]);
            },
        )?;
        Ok(Storage::Metal(output))
    }
    fn compare(&self, op: CmpOp, lhs: View<'_>, rhs: View<'_>) -> Result<Storage> {
        encode_binary(self, "compare", lhs, rhs, DType::Bool, op_code_cmp(op))
    }
    fn where_cond(&self, cond: View<'_>, on_true: View<'_>, on_false: View<'_>) -> Result<Storage> {
        if cond.dtype() != DType::Bool {
            return Err(Error::DTypeMismatch {
                op: "where",
                expected: DType::Bool,
                got: cond.dtype(),
            });
        }
        let dtype = same_dtype("where", &[on_true, on_false])?;
        let context = context(self.ordinal)?;
        check_context("where", &context, &[cond, on_true, on_false])?;
        let c = metal_storage("where", cond)?;
        let t = metal_storage("where", on_true)?;
        let f = metal_storage("where", on_false)?;
        let output = output_for(&context, dtype, cond.layout().num_elements())?;
        let pipe = pipeline_typed(&context, "where", dtype)?;
        encode(
            &context,
            &pipe,
            output.len,
            &[&c.buffer, &t.buffer, &f.buffer, &output.buffer],
            |encoder| {
                encoder.set_buffer(0, Some(&c.buffer), 0);
                encoder.set_buffer(1, Some(&t.buffer), 0);
                encoder.set_buffer(2, Some(&f.buffer), 0);
                encoder.set_buffer(3, Some(&output.buffer), 0);
                layout_args(encoder, 4, cond.layout());
                layout_args(encoder, 8, on_true.layout());
                layout_args(encoder, 12, on_false.layout());
                set_bytes(encoder, 16, &[output.len as u64]);
            },
        )?;
        Ok(Storage::Metal(output))
    }
    fn masked_fill(&self, x: View<'_>, mask: View<'_>, value: f64) -> Result<Storage> {
        supported_dtype("masked_fill", x.dtype(), x.device())?;
        if mask.dtype() != DType::Bool {
            return Err(Error::DTypeMismatch {
                op: "masked_fill",
                expected: DType::Bool,
                got: mask.dtype(),
            });
        }
        let context = context(self.ordinal)?;
        check_context("masked_fill", &context, &[x, mask])?;
        let input = metal_storage("masked_fill", x)?;
        let mask_storage = metal_storage("masked_fill", mask)?;
        let output = output_for(&context, x.dtype(), x.layout().num_elements())?;
        let pipe = pipeline_typed(&context, "masked", x.dtype())?;
        encode(
            &context,
            &pipe,
            output.len,
            &[&input.buffer, &mask_storage.buffer, &output.buffer],
            |encoder| {
                encoder.set_buffer(0, Some(&input.buffer), 0);
                encoder.set_buffer(1, Some(&mask_storage.buffer), 0);
                encoder.set_buffer(2, Some(&output.buffer), 0);
                layout_args(encoder, 3, x.layout());
                layout_args(encoder, 7, mask.layout());
                set_bytes(encoder, 11, &[output.len as u64]);
                if x.dtype() == DType::Bool {
                    set_bytes(encoder, 12, &[f32::from(value != 0.0)]);
                } else if x.dtype() == DType::I64 {
                    set_bytes(encoder, 12, &[value as i64]);
                } else {
                    set_bytes(encoder, 12, &[value as f32]);
                }
            },
        )?;
        Ok(Storage::Metal(output))
    }
    fn reduce(&self, op: ReduceOp, x: View<'_>, axis: usize) -> Result<Storage> {
        supported_dtype("reduce", x.dtype(), x.device())?;
        if matches!(x.dtype(), DType::Bool) {
            return Err(unsupported("reduce", x));
        }
        let context = context(self.ordinal)?;
        check_context("reduce", &context, &[x])?;
        let input = metal_storage("reduce", x)?;
        let len = reduced_len(x.layout(), axis);
        let output = output_for(&context, x.dtype(), len)?;
        let pipe = pipeline_typed(&context, "reduce", x.dtype())?;
        let code = match op {
            ReduceOp::Sum => 0,
            ReduceOp::Mean => 1,
            ReduceOp::Max => 2,
            ReduceOp::Min => 3,
        };
        encode(
            &context,
            &pipe,
            len,
            &[&input.buffer, &output.buffer],
            |encoder| {
                encoder.set_buffer(0, Some(&input.buffer), 0);
                encoder.set_buffer(1, Some(&output.buffer), 0);
                layout_args(encoder, 2, x.layout());
                set_bytes(encoder, 6, &[len as u64]);
                set_bytes(encoder, 7, &[axis as u32]);
                set_bytes(encoder, 8, &[code]);
            },
        )?;
        Ok(Storage::Metal(output))
    }
    fn arg_reduce(&self, op: ArgReduceOp, x: View<'_>, axis: usize) -> Result<Storage> {
        supported_dtype("arg_reduce", x.dtype(), x.device())?;
        if x.dtype() == DType::Bool {
            return Err(unsupported("arg_reduce", x));
        }
        let context = context(self.ordinal)?;
        check_context("arg_reduce", &context, &[x])?;
        let input = metal_storage("arg_reduce", x)?;
        let len = reduced_len(x.layout(), axis);
        let output = output_for(&context, DType::I64, len)?;
        let pipe = pipeline_typed(&context, "arg_reduce", x.dtype())?;
        encode(
            &context,
            &pipe,
            len,
            &[&input.buffer, &output.buffer],
            |encoder| {
                encoder.set_buffer(0, Some(&input.buffer), 0);
                encoder.set_buffer(1, Some(&output.buffer), 0);
                layout_args(encoder, 2, x.layout());
                set_bytes(encoder, 6, &[len as u64]);
                set_bytes(encoder, 7, &[axis as u32]);
                set_bytes(encoder, 8, &[u32::from(matches!(op, ArgReduceOp::ArgMin))]);
            },
        )?;
        Ok(Storage::Metal(output))
    }
    fn matmul(&self, lhs: View<'_>, rhs: View<'_>) -> Result<Storage> {
        let dtype = same_dtype("matmul", &[lhs, rhs])?;
        if dtype == DType::Bool {
            return Err(unsupported("matmul", lhs));
        }
        let plan = matmul_plan(lhs.layout(), rhs.layout())?;
        let context = context(self.ordinal)?;
        check_context("matmul", &context, &[lhs, rhs])?;
        let a = metal_storage("matmul", lhs)?;
        let b = metal_storage("matmul", rhs)?;
        let output = output_for(&context, dtype, plan.len)?;
        let pipe = pipeline_typed(&context, "matmul", dtype)?;
        encode(
            &context,
            &pipe,
            plan.len,
            &[&a.buffer, &b.buffer, &output.buffer],
            |encoder| {
                encoder.set_buffer(0, Some(&a.buffer), 0);
                encoder.set_buffer(1, Some(&b.buffer), 0);
                encoder.set_buffer(2, Some(&output.buffer), 0);
                set_bytes(encoder, 3, &plan.batch);
                set_bytes(encoder, 4, &plan.lhs_batch);
                set_bytes(encoder, 5, &plan.rhs_batch);
                set_bytes(encoder, 6, &[plan.batch.len() as u32]);
                set_bytes(encoder, 7, &plan.params);
                set_bytes(encoder, 8, &[plan.len as u64]);
            },
        )?;
        Ok(Storage::Metal(output))
    }
    fn index_select(&self, x: View<'_>, axis: usize, indices: View<'_>) -> Result<Storage> {
        supported_dtype("index_select", x.dtype(), x.device())?;
        check_axis("index_select", x, axis)?;
        if indices.dtype() != DType::I64 {
            return Err(Error::DTypeMismatch {
                op: "index_select",
                expected: DType::I64,
                got: indices.dtype(),
            });
        }
        if indices.layout().rank() != 1 {
            return Err(Error::RankMismatch {
                op: "index_select",
                expected: 1,
                got: indices.layout().rank(),
            });
        }
        let context = context(self.ordinal)?;
        check_context("index_select", &context, &[x, indices])?;
        validate_indices("index_select", indices, axis, x.layout().dims()[axis])?;
        let input = metal_storage("index_select", x)?;
        let index = metal_storage("index_select", indices)?;
        let mut dims = x.layout().dims().to_vec();
        dims[axis] = indices.layout().num_elements();
        let len = checked_product("index_select", dims.iter().copied())?;
        let out_dims: Vec<u64> = dims.iter().map(|&v| v as u64).collect();
        let output = output_for(&context, x.dtype(), len)?;
        let pipe = pipeline_typed(&context, "index_select", x.dtype())?;
        encode(
            &context,
            &pipe,
            len,
            &[&input.buffer, &index.buffer, &output.buffer],
            |encoder| {
                encoder.set_buffer(0, Some(&input.buffer), 0);
                encoder.set_buffer(1, Some(&index.buffer), 0);
                encoder.set_buffer(2, Some(&output.buffer), 0);
                layout_args(encoder, 3, x.layout());
                layout_args(encoder, 7, indices.layout());
                set_bytes(encoder, 11, &out_dims);
                set_bytes(encoder, 12, &[axis as u32]);
                set_bytes(encoder, 13, &[len as u64]);
            },
        )?;
        Ok(Storage::Metal(output))
    }
    fn index_add(
        &self,
        x: View<'_>,
        axis: usize,
        indices: View<'_>,
        src: View<'_>,
    ) -> Result<Storage> {
        let dtype = same_dtype("index_add", &[x, src])?;
        check_axis("index_add", x, axis)?;
        if dtype == DType::Bool {
            return Err(unsupported("index_add", x));
        }
        if indices.dtype() != DType::I64 {
            return Err(Error::DTypeMismatch {
                op: "index_add",
                expected: DType::I64,
                got: indices.dtype(),
            });
        }
        if indices.layout().rank() != 1 {
            return Err(Error::RankMismatch {
                op: "index_add",
                expected: 1,
                got: indices.layout().rank(),
            });
        }
        let mut expected = x.layout().dims().to_vec();
        expected[axis] = indices.layout().num_elements();
        if src.layout().dims() != expected {
            return Err(Error::ShapeMismatch {
                op: "index_add",
                lhs: crate::shape::Shape::from(expected),
                rhs: src.layout().shape().clone(),
            });
        }
        let context = context(self.ordinal)?;
        check_context("index_add", &context, &[x, indices, src])?;
        validate_indices("index_add", indices, axis, x.layout().dims()[axis])?;
        let xv = metal_storage("index_add", x)?;
        let iv = metal_storage("index_add", indices)?;
        let sv = metal_storage("index_add", src)?;
        let output = output_for(&context, dtype, x.layout().num_elements())?;
        let pipe = pipeline_typed(&context, "index_add", dtype)?;
        encode(
            &context,
            &pipe,
            output.len,
            &[&xv.buffer, &iv.buffer, &sv.buffer, &output.buffer],
            |encoder| {
                encoder.set_buffer(0, Some(&xv.buffer), 0);
                encoder.set_buffer(1, Some(&iv.buffer), 0);
                encoder.set_buffer(2, Some(&sv.buffer), 0);
                encoder.set_buffer(3, Some(&output.buffer), 0);
                layout_args(encoder, 4, x.layout());
                layout_args(encoder, 8, indices.layout());
                layout_args(encoder, 12, src.layout());
                set_bytes(encoder, 16, &[axis as u32]);
                set_bytes(encoder, 17, &[output.len as u64]);
                set_bytes(encoder, 18, &[src.layout().num_elements() as u64]);
            },
        )?;
        Ok(Storage::Metal(output))
    }
    fn gather(&self, x: View<'_>, axis: usize, indices: View<'_>) -> Result<Storage> {
        supported_dtype("gather", x.dtype(), x.device())?;
        check_axis("gather", x, axis)?;
        if indices.dtype() != DType::I64 {
            return Err(Error::DTypeMismatch {
                op: "gather",
                expected: DType::I64,
                got: indices.dtype(),
            });
        }
        if indices.layout().rank() != x.layout().rank() {
            return Err(Error::RankMismatch {
                op: "gather",
                expected: x.layout().rank(),
                got: indices.layout().rank(),
            });
        }
        for (other, (&index_dim, &source_dim)) in indices
            .layout()
            .dims()
            .iter()
            .zip(x.layout().dims())
            .enumerate()
        {
            if other != axis && index_dim > source_dim {
                return Err(Error::ShapeMismatch {
                    op: "gather",
                    lhs: x.layout().shape().clone(),
                    rhs: indices.layout().shape().clone(),
                });
            }
        }
        let context = context(self.ordinal)?;
        check_context("gather", &context, &[x, indices])?;
        validate_indices("gather", indices, axis, x.layout().dims()[axis])?;
        let xv = metal_storage("gather", x)?;
        let iv = metal_storage("gather", indices)?;
        let output = output_for(&context, x.dtype(), indices.layout().num_elements())?;
        let pipe = pipeline_typed(&context, "gather", x.dtype())?;
        encode(
            &context,
            &pipe,
            output.len,
            &[&xv.buffer, &iv.buffer, &output.buffer],
            |encoder| {
                encoder.set_buffer(0, Some(&xv.buffer), 0);
                encoder.set_buffer(1, Some(&iv.buffer), 0);
                encoder.set_buffer(2, Some(&output.buffer), 0);
                layout_args(encoder, 3, x.layout());
                layout_args(encoder, 7, indices.layout());
                set_bytes(encoder, 11, &[axis as u32]);
                set_bytes(encoder, 12, &[output.len as u64]);
            },
        )?;
        Ok(Storage::Metal(output))
    }
    fn scatter_add(
        &self,
        x: View<'_>,
        axis: usize,
        indices: View<'_>,
        src: View<'_>,
    ) -> Result<Storage> {
        let dtype = same_dtype("scatter_add", &[x, src])?;
        check_axis("scatter_add", x, axis)?;
        if dtype == DType::Bool {
            return Err(unsupported("scatter_add", x));
        }
        if indices.dtype() != DType::I64 {
            return Err(Error::DTypeMismatch {
                op: "scatter_add",
                expected: DType::I64,
                got: indices.dtype(),
            });
        }
        if indices.layout().rank() != x.layout().rank() {
            return Err(Error::RankMismatch {
                op: "scatter_add",
                expected: x.layout().rank(),
                got: indices.layout().rank(),
            });
        }
        if src.layout().rank() != x.layout().rank() {
            return Err(Error::RankMismatch {
                op: "scatter_add",
                expected: x.layout().rank(),
                got: src.layout().rank(),
            });
        }
        for (&index_dim, &src_dim) in indices.layout().dims().iter().zip(src.layout().dims()) {
            if index_dim > src_dim {
                return Err(Error::ShapeMismatch {
                    op: "scatter_add",
                    lhs: src.layout().shape().clone(),
                    rhs: indices.layout().shape().clone(),
                });
            }
        }
        for (other, (&index_dim, &output_dim)) in indices
            .layout()
            .dims()
            .iter()
            .zip(x.layout().dims())
            .enumerate()
        {
            if other != axis && index_dim > output_dim {
                return Err(Error::ShapeMismatch {
                    op: "scatter_add",
                    lhs: x.layout().shape().clone(),
                    rhs: indices.layout().shape().clone(),
                });
            }
        }
        let context = context(self.ordinal)?;
        check_context("scatter_add", &context, &[x, indices, src])?;
        validate_indices("scatter_add", indices, axis, x.layout().dims()[axis])?;
        let xv = metal_storage("scatter_add", x)?;
        let iv = metal_storage("scatter_add", indices)?;
        let sv = metal_storage("scatter_add", src)?;
        let output = output_for(&context, dtype, x.layout().num_elements())?;
        let pipe = pipeline_typed(&context, "scatter_add", dtype)?;
        encode(
            &context,
            &pipe,
            output.len,
            &[&xv.buffer, &iv.buffer, &sv.buffer, &output.buffer],
            |encoder| {
                encoder.set_buffer(0, Some(&xv.buffer), 0);
                encoder.set_buffer(1, Some(&iv.buffer), 0);
                encoder.set_buffer(2, Some(&sv.buffer), 0);
                encoder.set_buffer(3, Some(&output.buffer), 0);
                layout_args(encoder, 4, x.layout());
                layout_args(encoder, 8, indices.layout());
                layout_args(encoder, 12, src.layout());
                set_bytes(encoder, 16, &[axis as u32]);
                set_bytes(encoder, 17, &[output.len as u64]);
                set_bytes(encoder, 18, &[src.layout().num_elements() as u64]);
            },
        )?;
        Ok(Storage::Metal(output))
    }
    fn conv(&self, op: ConvOp, inputs: &[View<'_>], params: &Conv2dParams) -> Result<Storage> {
        let Some(input) = inputs.first() else {
            return Err(Error::InvalidArg {
                op: "conv",
                msg: "missing input".to_owned(),
            });
        };
        let dtype = same_dtype("conv", inputs)?;
        if dtype == DType::Bool {
            return Err(unsupported("conv", *input));
        }
        let (geometry, kernel_name, output_len, first, second, pool_code) = match (op, inputs) {
            (ConvOp::Conv2d, [x, weight]) => {
                let geometry = crate::backend::conv_geometry::Conv2dGeometry::conv2d(
                    "conv2d",
                    x.layout().dims(),
                    weight.layout().dims(),
                    params,
                )?;
                let len = checked_product("conv2d", geometry.output_dims())?;
                (geometry, "conv2d", len, *x, Some(*weight), None)
            }
            (ConvOp::MaxPool2d | ConvOp::AvgPool2d, [x]) => {
                let geometry = crate::backend::conv_geometry::Conv2dGeometry::pool(
                    "pool2d",
                    x.layout().dims(),
                    params,
                )?;
                let len = checked_product("pool2d", geometry.output_dims())?;
                (
                    geometry,
                    "pool",
                    len,
                    *x,
                    None,
                    Some(u32::from(matches!(op, ConvOp::AvgPool2d))),
                )
            }
            (ConvOp::Conv2dInputGrad, [grad, weight, original_input]) => {
                let geometry = crate::backend::conv_geometry::Conv2dGeometry::conv2d(
                    "conv2d_backward_input",
                    original_input.layout().dims(),
                    weight.layout().dims(),
                    params,
                )?;
                if grad.layout().dims() != geometry.output_dims()
                    || weight.layout().dims() != geometry.weight_dims()
                {
                    return Err(Error::ShapeMismatch {
                        op: "conv2d_backward_input",
                        lhs: grad.layout().shape().clone(),
                        rhs: crate::shape::Shape::from(geometry.output_dims()),
                    });
                }
                (
                    geometry,
                    "conv_input_grad",
                    original_input.layout().num_elements(),
                    *grad,
                    Some(*weight),
                    None,
                )
            }
            (ConvOp::Conv2dWeightGrad, [grad, original_input, original_weight]) => {
                let geometry = crate::backend::conv_geometry::Conv2dGeometry::conv2d(
                    "conv2d_backward_weight",
                    original_input.layout().dims(),
                    original_weight.layout().dims(),
                    params,
                )?;
                if grad.layout().dims() != geometry.output_dims()
                    || original_input.layout().dims() != geometry.input_dims()
                {
                    return Err(Error::ShapeMismatch {
                        op: "conv2d_backward_weight",
                        lhs: grad.layout().shape().clone(),
                        rhs: crate::shape::Shape::from(geometry.output_dims()),
                    });
                }
                (
                    geometry,
                    "conv_weight_grad",
                    original_weight.layout().num_elements(),
                    *grad,
                    Some(*original_input),
                    None,
                )
            }
            (ConvOp::MaxPool2dBackward | ConvOp::AvgPool2dBackward, [grad, original_input]) => {
                let geometry = crate::backend::conv_geometry::Conv2dGeometry::pool(
                    "pool2d_backward",
                    original_input.layout().dims(),
                    params,
                )?;
                if grad.layout().dims() != geometry.output_dims() {
                    return Err(Error::ShapeMismatch {
                        op: "pool2d_backward",
                        lhs: grad.layout().shape().clone(),
                        rhs: crate::shape::Shape::from(geometry.output_dims()),
                    });
                }
                (
                    geometry,
                    "pool_backward",
                    original_input.layout().num_elements(),
                    *grad,
                    Some(*original_input),
                    Some(u32::from(matches!(op, ConvOp::AvgPool2dBackward))),
                )
            }
            _ => {
                return Err(Error::InvalidArg {
                    op: "conv",
                    msg: format!("invalid {op:?} operand encoding"),
                });
            }
        };
        let context = context(self.ordinal)?;
        check_context("conv", &context, inputs)?;
        let a = metal_storage("conv", first)?;
        let b = second.map(|view| metal_storage("conv", view)).transpose()?;
        let output = output_for(&context, dtype, output_len)?;
        let pipe = pipeline_typed(&context, kernel_name, dtype)?;
        let packed = conv_params(&geometry, params);
        let mut resources = vec![&a.buffer, &output.buffer];
        if let Some(value) = &b {
            resources.push(&value.buffer);
        }
        encode(&context, &pipe, output_len, &resources, |encoder| {
            encoder.set_buffer(0, Some(&a.buffer), 0);
            if let Some(value) = b {
                encoder.set_buffer(1, Some(&value.buffer), 0);
                encoder.set_buffer(2, Some(&output.buffer), 0);
                set_bytes(
                    encoder,
                    3,
                    &first
                        .layout()
                        .strides()
                        .iter()
                        .map(|&v| v as u64)
                        .collect::<Vec<_>>(),
                );
                set_bytes(encoder, 4, &[first.layout().offset() as u64]);
                set_bytes(
                    encoder,
                    5,
                    &second
                        .unwrap()
                        .layout()
                        .strides()
                        .iter()
                        .map(|&v| v as u64)
                        .collect::<Vec<_>>(),
                );
                set_bytes(encoder, 6, &[second.unwrap().layout().offset() as u64]);
                set_bytes(encoder, 7, &packed);
                set_bytes(encoder, 8, &[output_len as u64]);
                if let Some(code) = pool_code {
                    set_bytes(encoder, 9, &[code]);
                }
            } else {
                encoder.set_buffer(1, Some(&output.buffer), 0);
                set_bytes(
                    encoder,
                    2,
                    &first
                        .layout()
                        .strides()
                        .iter()
                        .map(|&v| v as u64)
                        .collect::<Vec<_>>(),
                );
                set_bytes(encoder, 3, &[first.layout().offset() as u64]);
                set_bytes(encoder, 4, &packed);
                set_bytes(encoder, 5, &[output_len as u64]);
                set_bytes(encoder, 6, &[pool_code.unwrap_or(0)]);
            }
        })?;
        Ok(Storage::Metal(output))
    }
    fn fused(&self, op: FusedOp, inputs: &[View<'_>], scalars: &[f64]) -> Result<Vec<Storage>> {
        let Some(input) = inputs.first() else {
            return Err(Error::InvalidArg {
                op: "fused",
                msg: "missing input".to_owned(),
            });
        };
        match op {
            FusedOp::Softmax => {
                if inputs.len() != 1 || !scalars.is_empty() {
                    return Err(Error::InvalidArg {
                        op: "fused_softmax",
                        msg: "expected one input and no scalars".to_owned(),
                    });
                }
                let dtype = input.dtype();
                if !matches!(dtype, DType::F16 | DType::F32) {
                    return Err(unsupported("fused_softmax", *input));
                }
                let width = *input
                    .layout()
                    .dims()
                    .last()
                    .ok_or_else(|| Error::InvalidArg {
                        op: "fused_softmax",
                        msg: "requires rank >= 1".to_owned(),
                    })?;
                let rows = input.layout().num_elements() / width;
                let context = context(self.ordinal)?;
                check_context("fused_softmax", &context, inputs)?;
                let x = metal_storage("fused_softmax", *input)?;
                let output = output_for(&context, dtype, input.layout().num_elements())?;
                let pipe = pipeline_typed(&context, "softmax", dtype)?;
                encode(
                    &context,
                    &pipe,
                    rows,
                    &[&x.buffer, &output.buffer],
                    |encoder| {
                        encoder.set_buffer(0, Some(&x.buffer), 0);
                        encoder.set_buffer(1, Some(&output.buffer), 0);
                        layout_args(encoder, 2, input.layout());
                        set_bytes(encoder, 6, &[rows as u64]);
                        set_bytes(encoder, 7, &[width as u64]);
                    },
                )?;
                Ok(vec![Storage::Metal(output)])
            }
            FusedOp::LayerNorm => self.fused_layer_norm(inputs, scalars),
            FusedOp::SgdStep => self.fused_sgd(inputs, scalars),
            FusedOp::AdamStep => self.fused_adam(inputs, scalars),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, atomic::Ordering};

    use super::{COMMIT_THRESHOLD, INSTRUMENTATION};
    use crate::backend::{FusedOp, View, dispatch};
    use crate::layout::Layout;
    use crate::nn::{Forward, Linear, Mode, Param};
    use crate::optim::AdamW;
    use crate::rng::Rng;
    use crate::{DType, Device, Error, Tensor};

    const METAL: Device = Device::Metal(0);
    static HARDWARE_LANE: Mutex<()> = Mutex::new(());

    #[test]
    fn storage_transfer_and_strided_copy_run_on_hardware() {
        let _lane = HARDWARE_LANE.lock().unwrap();
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
        let _lane = HARDWARE_LANE.lock().unwrap();
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
        let _lane = HARDWARE_LANE.lock().unwrap();
        let err = Tensor::zeros([1], DType::F32, &Device::Metal(usize::MAX));
        assert!(matches!(
            err,
            Err(Error::Backend {
                op: "metal_device",
                ..
            })
        ));
    }

    #[test]
    fn required_backend_table_matches_cpu() {
        let _lane = HARDWARE_LANE.lock().unwrap();
        let report = crate::backend::conformance::run_device(METAL);
        eprintln!(
            "Metal conformance: {} matched, {} expected unsupported, {} unexpected skips, {} failed",
            report.matched.len(),
            report.expected_unsupported.len(),
            report.skipped.len(),
            report.failures.len()
        );
        assert!(
            report.skipped.is_empty(),
            "unexpected skips: {:?}",
            report.skipped
        );
        assert!(
            report.failures.is_empty(),
            "Metal conformance failures:\n{}",
            report.failures.join("\n")
        );
    }

    #[test]
    fn async_submission_batches_defers_waits_and_reaps() {
        let _lane = HARDWARE_LANE.lock().unwrap();
        let before_dispatch = INSTRUMENTATION.dispatches.load(Ordering::Relaxed);
        let before_commit = INSTRUMENTATION.commits.load(Ordering::Relaxed);
        let before_wait = INSTRUMENTATION.waits.load(Ordering::Relaxed);
        let mut value = Tensor::ones([16], DType::F32, &METAL).unwrap();
        for _ in 0..COMMIT_THRESHOLD + 8 {
            value = value.add_scalar(1.0).unwrap();
        }
        assert_eq!(
            INSTRUMENTATION.waits.load(Ordering::Relaxed),
            before_wait,
            "device-ordered operations must not wait"
        );
        assert!(
            INSTRUMENTATION.commits.load(Ordering::Relaxed) > before_commit,
            "the threshold must commit without waiting"
        );
        assert_eq!(
            value.to_vec::<f32>().unwrap(),
            vec![(COMMIT_THRESHOLD + 9) as f32; 16]
        );
        assert!(INSTRUMENTATION.waits.load(Ordering::Relaxed) > before_wait);
        assert!(
            INSTRUMENTATION.dispatches.load(Ordering::Relaxed) - before_dispatch
                >= COMMIT_THRESHOLD + 9
        );
        assert!(
            INSTRUMENTATION
                .max_dispatches_per_buffer
                .load(Ordering::Relaxed)
                >= COMMIT_THRESHOLD
        );
        let context = super::context(0).unwrap();
        let submission = context.submission.lock().unwrap();
        assert!(submission.open.is_none());
        assert!(
            submission.pending.is_empty(),
            "host read must reap pending buffers"
        );
        eprintln!(
            "async counters: dispatches={}, commits={}, waits={}, max_dispatches_per_buffer={}, max_pending={}",
            INSTRUMENTATION.dispatches.load(Ordering::Relaxed) - before_dispatch,
            INSTRUMENTATION.commits.load(Ordering::Relaxed) - before_commit,
            INSTRUMENTATION.waits.load(Ordering::Relaxed) - before_wait,
            INSTRUMENTATION
                .max_dispatches_per_buffer
                .load(Ordering::Relaxed),
            INSTRUMENTATION.max_pending.load(Ordering::Relaxed),
        );
    }

    #[test]
    fn cat_and_stack_do_not_cross_the_host_boundary() {
        let _lane = HARDWARE_LANE.lock().unwrap();
        let input_count = INSTRUMENTATION.transfer_in.load(Ordering::Relaxed);
        let output_count = INSTRUMENTATION.transfer_out.load(Ordering::Relaxed);
        let a = Tensor::ones([2, 3], DType::F32, &METAL).unwrap();
        let b = Tensor::full([2, 3], 2.0, DType::F32, &METAL).unwrap();
        let cat = Tensor::cat(&[&a, &b], 0).unwrap();
        let stack = Tensor::stack(&[&a, &b], 1).unwrap();
        assert_eq!(cat.device(), METAL);
        assert_eq!(stack.device(), METAL);
        assert_eq!(
            INSTRUMENTATION.transfer_in.load(Ordering::Relaxed),
            input_count
        );
        assert_eq!(
            INSTRUMENTATION.transfer_out.load(Ordering::Relaxed),
            output_count
        );
    }

    #[test]
    fn fused_mixed_layer_norm_and_optimizers_match_cpu() {
        let _lane = HARDWARE_LANE.lock().unwrap();
        let metal = dispatch::backend(METAL);
        let cpu = dispatch::backend(Device::Cpu);

        let x = Tensor::from_vec(
            vec![
                half::f16::from_f32(1.0),
                half::f16::from_f32(2.0),
                half::f16::from_f32(4.0),
                half::f16::from_f32(-1.0),
                half::f16::from_f32(0.5),
                half::f16::from_f32(3.0),
            ],
            [2, 3],
            &METAL,
        )
        .unwrap();
        let weight = Tensor::ones([3], DType::F16, &METAL).unwrap();
        let bias = Tensor::zeros([3], DType::F16, &METAL).unwrap();
        let outputs = metal
            .fused(
                FusedOp::LayerNorm,
                &[x.view(), weight.view(), bias.view()],
                &[1e-5, 1.0],
            )
            .unwrap();
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].dtype(), DType::F16);
        assert_eq!(outputs[1].dtype(), DType::F32);
        assert_eq!(outputs[2].dtype(), DType::F32);

        let layouts = [
            Layout::contiguous([2, 3]).unwrap(),
            Layout::contiguous([2, 3]).unwrap(),
            Layout::contiguous([2, 1]).unwrap(),
        ];
        let y = metal
            .transfer_out(View::new(&outputs[0], &layouts[0]))
            .unwrap();
        let xhat = metal
            .transfer_out(View::new(&outputs[1], &layouts[1]))
            .unwrap();
        let inv = metal
            .transfer_out(View::new(&outputs[2], &layouts[2]))
            .unwrap();
        assert_eq!(y.dtype(), DType::F16);
        assert_eq!(xhat.dtype(), DType::F32);
        assert_eq!(inv.dtype(), DType::F32);

        let host_param = crate::storage::CpuStorage::F16(std::sync::Arc::new(vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
        ]));
        let host_grad = crate::storage::CpuStorage::F16(std::sync::Arc::new(vec![
            half::f16::from_f32(0.5),
            half::f16::from_f32(-0.25),
        ]));
        for backend in [cpu, metal] {
            let p = backend.transfer_in(host_param.clone()).unwrap();
            let g = backend.transfer_in(host_grad.clone()).unwrap();
            let layout = Layout::contiguous([2]).unwrap();
            let next = backend
                .fused(
                    FusedOp::SgdStep,
                    &[View::new(&p, &layout), View::new(&g, &layout)],
                    &[0.1, 0.9, 0.0],
                )
                .unwrap();
            assert_eq!(next.len(), 2);
            let values = backend.transfer_out(View::new(&next[0], &layout)).unwrap();
            let crate::storage::CpuStorage::F16(values) = values else {
                panic!("fused SGD changed parameter dtype")
            };
            assert!((values[0].to_f32() - 0.950_2).abs() < 2e-3);
            assert!((values[1].to_f32() - 2.025).abs() < 2e-3);
        }
    }

    #[test]
    fn conv_and_pool_backward_execute_on_metal() {
        let _lane = HARDWARE_LANE.lock().unwrap();
        let x = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [1, 1, 3, 3],
            &METAL,
        )
        .unwrap()
        .traced()
        .unwrap();
        let w = Tensor::from_vec(vec![1.0f32, -1.0, 0.5, 2.0], [1, 1, 2, 2], &METAL)
            .unwrap()
            .traced()
            .unwrap();
        let loss = x
            .conv2d(&w, (1, 1), (0, 0), (1, 1))
            .unwrap()
            .sum_all()
            .unwrap();
        let grads = loss.backward().unwrap();
        assert_eq!(
            grads.wrt_input(&x).unwrap().to_vec::<f32>().unwrap(),
            vec![1.0, 0.0, -1.0, 1.5, 2.5, 1.0, 0.5, 2.5, 2.0]
        );
        assert_eq!(
            grads.wrt_input(&w).unwrap().to_vec::<f32>().unwrap(),
            vec![12.0, 16.0, 24.0, 28.0]
        );

        let pooled = x.max_pool2d((2, 2), (1, 1), (0, 0)).unwrap();
        let pool_grads = pooled.sum_all().unwrap().backward().unwrap();
        assert_eq!(
            pool_grads.wrt_input(&x).unwrap().to_vec::<f32>().unwrap(),
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        );

        let avg = x.avg_pool2d((2, 2), (1, 1), (0, 0)).unwrap();
        let avg_grads = avg.sum_all().unwrap().backward().unwrap();
        assert_eq!(
            avg_grads.wrt_input(&x).unwrap().to_vec::<f32>().unwrap(),
            vec![0.25, 0.5, 0.25, 0.5, 1.0, 0.5, 0.25, 0.5, 0.25]
        );
    }

    /// Bounds violations are reported with the same content as CPU, but at the
    /// next **host boundary** rather than from the indexing call — see
    /// [`Validation`] for why the synchronous check was removed.
    ///
    /// Each case therefore encodes the bad op, then forces a host read and
    /// expects the verdict there. Collecting a verdict also clears the queue,
    /// so the cases do not contaminate each other.
    #[test]
    fn index_family_reports_public_bounds_errors() {
        let _lane = HARDWARE_LANE.lock().unwrap();
        let backend = dispatch::backend(METAL);
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3], &METAL).unwrap();

        // Read any device tensor back; the pending verdict surfaces here.
        let host_boundary = || x.to_vec::<f32>();

        for index in [-1i64, 3] {
            let indices = Tensor::from_vec(vec![index], [1], &METAL).unwrap();
            // The op itself now succeeds: it has only *encoded* the check.
            let selected = x.index_select(0, &indices).unwrap();
            assert!(matches!(
                selected.to_vec::<f32>(),
                Err(Error::IndexOutOfBounds {
                    op: "index_select",
                    index: got,
                    axis: 0,
                    size: 3,
                }) if got == index
            ));

            let base = Tensor::zeros([3], DType::F32, &METAL).unwrap();
            let src = Tensor::ones([1], DType::F32, &METAL).unwrap();
            backend
                .index_add(base.view(), 0, indices.view(), src.view())
                .expect("index_add encodes its check rather than resolving it");
            assert!(matches!(
                host_boundary(),
                Err(Error::IndexOutOfBounds { op: "index_add", index: got, .. }) if got == index
            ));
        }

        let matrix = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2], &METAL).unwrap();
        for index in [-1i64, 2] {
            let bad = Tensor::from_vec(vec![0i64, index], [1, 2], &METAL).unwrap();
            let gathered = matrix.gather(1, &bad).unwrap();
            assert!(matches!(
                gathered.to_vec::<f32>(),
                Err(Error::IndexOutOfBounds { op: "gather", index: got, .. }) if got == index
            ));

            let src = Tensor::ones([1, 2], DType::F32, &METAL).unwrap();
            backend
                .scatter_add(matrix.view(), 1, bad.view(), src.view())
                .expect("scatter_add encodes its check rather than resolving it");
            assert!(matches!(
                host_boundary(),
                Err(Error::IndexOutOfBounds { op: "scatter_add", index: got, .. }) if got == index
            ));
        }

        // The queue is clean once every verdict has been collected, so a valid
        // program that follows is unaffected by the failures above.
        assert_eq!(host_boundary().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    /// A single host boundary resolves a whole batch of encoded checks and
    /// reports the **first** failure in program order, not the last.
    #[test]
    fn deferred_bounds_checks_batch_and_report_in_program_order() {
        let _lane = HARDWARE_LANE.lock().unwrap();
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3], &METAL).unwrap();
        let good = Tensor::from_vec(vec![0i64], [1], &METAL).unwrap();
        let first_bad = Tensor::from_vec(vec![7i64], [1], &METAL).unwrap();
        let second_bad = Tensor::from_vec(vec![9i64], [1], &METAL).unwrap();

        // Three indexed ops encoded back to back, none of them synchronizing.
        let _ = x.index_select(0, &good).unwrap();
        let _ = x.index_select(0, &first_bad).unwrap();
        let _ = x.index_select(0, &second_bad).unwrap();

        assert!(matches!(
            x.to_vec::<f32>(),
            Err(Error::IndexOutOfBounds { index: 7, .. })
        ));
        assert_eq!(x.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn gelu_matches_cpu_on_adversarial_inputs() {
        let _lane = HARDWARE_LANE.lock().unwrap();
        let mut values: Vec<f32> = (-12_000..=12_000)
            .map(|value| value as f32 / 1000.0)
            .collect();
        values.extend([
            -12.0f32,
            -8.0,
            -5.0,
            -3.0,
            -1.0,
            -0.1,
            -1e-4,
            -f32::EPSILON,
            0.0,
            f32::EPSILON,
            1e-4,
            0.1,
            1.0,
            3.0,
            5.0,
            8.0,
            12.0,
        ]);
        let cpu = Tensor::from_vec(values.clone(), [values.len()], &Device::Cpu)
            .unwrap()
            .gelu()
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let metal = Tensor::from_vec(values.clone(), [cpu.len()], &METAL)
            .unwrap()
            .gelu()
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let mut max_abs = 0.0f32;
        let mut max_ulp = 0u32;
        for (index, ((&input, &want), &got)) in values.iter().zip(&cpu).zip(&metal).enumerate() {
            max_abs = max_abs.max((want - got).abs());
            if input.abs() <= 5.0
                && want.abs() >= 1e-4
                && want.is_sign_positive() == got.is_sign_positive()
            {
                max_ulp = max_ulp.max(want.to_bits().abs_diff(got.to_bits()));
            }
            assert!(
                (want - got).abs() <= 2.0 * f32::EPSILON * want.abs().max(1.0),
                "GELU element {index}: expected {want:?}, got {got:?}"
            );
        }
        eprintln!("dense GELU sweep: max_abs={max_abs:e}, max_central_ulp={max_ulp}");
    }

    #[test]
    fn float_to_i64_casts_match_rust_saturation() {
        let _lane = HARDWARE_LANE.lock().unwrap();
        let values = vec![
            f32::NAN,
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::from_bits(0xdf00_0000),
            f32::from_bits(0x5f00_0000),
            f32::from_bits(0xdeff_ffff),
            f32::from_bits(0x5eff_ffff),
            f32::from_bits(0xdf00_0001),
            f32::from_bits(0x5f00_0001),
            -123.75,
            -0.75,
            0.75,
            123.75,
            16_777_216.0,
        ];
        let expected = Tensor::from_vec(values.clone(), [values.len()], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::I64)
            .unwrap()
            .to_vec::<i64>()
            .unwrap();
        let got = Tensor::from_vec(values, [expected.len()], &METAL)
            .unwrap()
            .to_dtype(DType::I64)
            .unwrap()
            .to_vec::<i64>()
            .unwrap();
        assert_eq!(got, expected);

        let halves = vec![
            half::f16::NAN,
            half::f16::NEG_INFINITY,
            half::f16::INFINITY,
            half::f16::from_f32(-123.75),
            half::f16::from_f32(-0.75),
            half::f16::from_f32(0.75),
            half::f16::from_f32(123.75),
        ];
        let expected = Tensor::from_vec(halves.clone(), [halves.len()], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::I64)
            .unwrap()
            .to_vec::<i64>()
            .unwrap();
        let got = Tensor::from_vec(halves, [expected.len()], &METAL)
            .unwrap()
            .to_dtype(DType::I64)
            .unwrap()
            .to_vec::<i64>()
            .unwrap();
        assert_eq!(got, expected);
    }

    #[test]
    fn checked_output_products_reject_overflow_without_allocating() {
        assert!(matches!(
            super::checked_product("test_grid", [usize::MAX, 2]),
            Err(Error::InvalidArg {
                op: "test_grid",
                ..
            })
        ));
        assert_eq!(
            super::checked_product("test_grid", [3, 0, usize::MAX]).unwrap(),
            0
        );
    }

    #[test]
    fn synchronization_error_drain_evaluates_every_pending_result() {
        use std::cell::Cell;

        let evaluated = Cell::new(0usize);
        let result = super::drain_results((0..3).map(|index| {
            evaluated.set(evaluated.get() + 1);
            if index < 2 {
                Err(Error::Backend {
                    op: if index == 0 { "first" } else { "second" },
                    msg: index.to_string(),
                })
            } else {
                Ok(())
            }
        }));
        assert_eq!(evaluated.get(), 3);
        assert!(matches!(result, Err(Error::Backend { op: "first", .. })));
    }

    #[test]
    fn max_pool_first_tie_and_first_nan_own_gradient() {
        let _lane = HARDWARE_LANE.lock().unwrap();
        for values in [
            vec![2.0f32, 2.0, 1.0, 0.0],
            vec![f32::NAN, f32::NAN, 1.0, 0.0],
        ] {
            let run = |device| {
                let x = Tensor::from_vec(values.clone(), [1, 1, 2, 2], &device)
                    .unwrap()
                    .traced()
                    .unwrap();
                let y = x.max_pool2d((2, 2), (1, 1), (0, 0)).unwrap();
                let forward = y.to_vec::<f32>().unwrap();
                let grad = y
                    .sum_all()
                    .unwrap()
                    .backward()
                    .unwrap()
                    .wrt_input(&x)
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap();
                (forward, grad)
            };
            let cpu = run(Device::Cpu);
            let metal = run(METAL);
            assert_eq!(cpu.1, metal.1);
            assert_eq!(metal.1, vec![1.0, 0.0, 0.0, 0.0]);
            assert_eq!(cpu.0[0].is_nan(), metal.0[0].is_nan());
        }
    }

    #[test]
    fn f16_accumulators_remain_wide_on_hardware() {
        let _lane = HARDWARE_LANE.lock().unwrap();
        let ones = Tensor::ones([4096], DType::F16, &METAL).unwrap();
        assert_eq!(ones.sum_all().unwrap().item().unwrap(), 4096.0);

        let indices = Tensor::zeros([4096], DType::I64, &METAL).unwrap();
        let base = Param::new(Tensor::zeros([1], DType::F16, &METAL).unwrap());
        let selected = base
            .get(Mode::TRAIN)
            .index_select(0, &indices)
            .unwrap()
            .sum_all()
            .unwrap();
        let grads = selected.backward().unwrap();
        assert_eq!(grads.wrt(&base).unwrap().item().unwrap(), 4096.0);
    }

    #[test]
    fn finite_differences_and_training_step_stay_device_resident() {
        let _lane = HARDWARE_LANE.lock().unwrap();
        let input = Tensor::from_vec(vec![0.25f32, -0.5, 1.0, 0.75], [2, 2], &METAL).unwrap();
        crate::testing::check_grad(
            |xs| xs[0].matmul(&xs[0].transpose(0, 1)?)?.gelu()?.sum_all(),
            &[input],
            1e-3,
            3e-3,
        )
        .unwrap();

        #[derive(rstorch::Module)]
        struct Tiny {
            linear: Linear,
        }
        impl Forward for Tiny {
            fn forward(&mut self, x: &Tensor, mode: Mode) -> crate::Result<Tensor> {
                self.linear.forward(x, mode)
            }
        }
        let mut model = Tiny {
            linear: Linear::new(2, 2, &METAL, &mut Rng::seed(3)).unwrap(),
        };
        let x = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], [2, 2], &METAL).unwrap();
        let labels = Tensor::from_vec(vec![0i64, 1], [2], &METAL).unwrap();
        let transfer_in = INSTRUMENTATION.transfer_in.load(Ordering::Relaxed);
        let transfer_out = INSTRUMENTATION.transfer_out.load(Ordering::Relaxed);
        let loss = model
            .forward(&x, Mode::TRAIN)
            .unwrap()
            .cross_entropy(&labels)
            .unwrap();
        AdamW::new(0.01, 0.0)
            .step(&mut model, loss.backward().unwrap())
            .unwrap();
        // The first Adam step allocates its initial moment state on-device.
        // This constructor-side write is not a device-to-host fallback.
        assert_eq!(
            INSTRUMENTATION.transfer_in.load(Ordering::Relaxed),
            transfer_in + 1
        );
        assert_eq!(
            INSTRUMENTATION.transfer_out.load(Ordering::Relaxed),
            transfer_out
        );
    }

    #[test]
    fn multithreaded_streams_remain_ordered() {
        let _lane = HARDWARE_LANE.lock().unwrap();
        let threads: Vec<_> = (0..4)
            .map(|thread| {
                std::thread::spawn(move || {
                    let mut value = Tensor::full([32], thread as f64, DType::F32, &METAL).unwrap();
                    for _ in 0..16 {
                        value = value.add_scalar(1.0).unwrap();
                    }
                    value.to_vec::<f32>().unwrap()
                })
            })
            .collect();
        for (thread, handle) in threads.into_iter().enumerate() {
            assert_eq!(handle.join().unwrap(), vec![thread as f32 + 16.0; 32]);
        }
    }
}
