//! NVIDIA CUDA backend, backed by the precompiled kernels in `kernels.ptx`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

use crate::backend::{
    ArgReduceOp, BackendOps, BinaryOp, CmpOp, Conv2dParams, ConvOp, FusedOp, ReduceOp, UnaryOp,
    View,
};
use crate::device::Device;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::storage::{CpuStorage, Storage};

const PTX: &str = include_str!("kernels.ptx");
const THREADS: u32 = 256;
const VALIDATION_SLOTS: usize = 256;

type ContextResult = std::result::Result<Arc<Context>, String>;

struct Context {
    ordinal: usize,
    _raw: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    functions: Mutex<HashMap<String, CudaFunction>>,
    validation: Mutex<Validation>,
}

struct Validation {
    status: CudaSlice<i64>,
    pending: Vec<PendingValidation>,
}

struct PendingValidation {
    op: &'static str,
    axis: usize,
    bound: usize,
}

impl Context {
    fn function(&self, name: &str) -> Result<CudaFunction> {
        let mut functions = self.functions.lock().map_err(|_| Error::Backend {
            op: "cuda_kernel",
            msg: "CUDA function cache poisoned".to_owned(),
        })?;
        if let Some(function) = functions.get(name) {
            return Ok(function.clone());
        }
        let function = self
            .module
            .load_function(name)
            .map_err(|error| backend_error("cuda_kernel", error))?;
        functions.insert(name.to_owned(), function.clone());
        Ok(function)
    }
}

pub(crate) struct CudaBackend {
    ordinal: usize,
}

#[derive(Clone)]
pub(crate) struct CudaStorage {
    buffer: Arc<CudaSlice<u8>>,
    dtype: DType,
    len: usize,
    context: Arc<Context>,
}

impl CudaStorage {
    pub(crate) fn dtype(&self) -> DType {
        self.dtype
    }

    pub(crate) fn device(&self) -> Device {
        Device::Cuda(self.context.ordinal)
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

pub(crate) fn backend(ordinal: usize) -> &'static dyn BackendOps {
    static BACKENDS: OnceLock<Mutex<HashMap<usize, &'static CudaBackend>>> = OnceLock::new();
    let mut backends = BACKENDS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("CUDA backend registry poisoned");
    *backends
        .entry(ordinal)
        .or_insert_with(|| Box::leak(Box::new(CudaBackend { ordinal })))
}

pub(crate) fn is_available(ordinal: usize) -> bool {
    context(ordinal).is_ok()
}

fn context(ordinal: usize) -> Result<Arc<Context>> {
    static CONTEXTS: OnceLock<Mutex<HashMap<usize, ContextResult>>> = OnceLock::new();
    let result = CONTEXTS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .expect("CUDA context registry poisoned")
        .entry(ordinal)
        .or_insert_with(|| create_context(ordinal))
        .clone();
    result.map_err(|msg| Error::Backend {
        op: "cuda_device",
        msg,
    })
}

fn create_context(ordinal: usize) -> ContextResult {
    let raw = std::panic::catch_unwind(|| CudaContext::new(ordinal))
        .map_err(|payload| {
            payload.downcast_ref::<&str>().map_or_else(
                || {
                    payload
                        .downcast_ref::<String>()
                        .cloned()
                        .unwrap_or_else(|| "CUDA driver initialization panicked".to_owned())
                },
                |message| (*message).to_owned(),
            )
        })?
        .map_err(|error| error.to_string())?;
    let stream = raw.default_stream();
    let module = raw
        .load_module(Ptx::from_src(PTX))
        .map_err(|error| format!("failed to load kernels.ptx: {error}"))?;
    let status = stream
        .alloc_zeros::<i64>(VALIDATION_SLOTS * 2)
        .map_err(|error| format!("failed to allocate validation status table: {error}"))?;
    Ok(Arc::new(Context {
        ordinal,
        _raw: raw,
        stream,
        module,
        functions: Mutex::new(HashMap::new()),
        validation: Mutex::new(Validation {
            status,
            pending: Vec::new(),
        }),
    }))
}

fn backend_error(op: &'static str, error: impl std::fmt::Display) -> Error {
    Error::Backend {
        op,
        msg: error.to_string(),
    }
}

fn supported(op: &'static str, dtype: DType, device: Device) -> Result<()> {
    if matches!(dtype, DType::F16 | DType::F32 | DType::I64 | DType::Bool) {
        Ok(())
    } else {
        Err(Error::Unsupported { op, device, dtype })
    }
}

fn unsupported(op: &'static str, view: View<'_>) -> Error {
    Error::Unsupported {
        op,
        device: view.device(),
        dtype: view.dtype(),
    }
}

fn element_size(dtype: DType) -> usize {
    match dtype {
        DType::F16 | DType::BF16 => 2,
        DType::F32 => 4,
        DType::I64 | DType::F64 => 8,
        DType::Bool => 1,
    }
}

fn byte_len(op: &'static str, dtype: DType, len: usize) -> Result<usize> {
    len.max(1)
        .checked_mul(element_size(dtype))
        .ok_or_else(|| Error::InvalidArg {
            op,
            msg: format!("buffer byte length overflows usize for {len} {dtype} elements"),
        })
}

fn allocate_raw(context: &Context, dtype: DType, len: usize) -> Result<CudaSlice<u8>> {
    let bytes = byte_len("cuda_allocate", dtype, len)?;
    // SAFETY: every byte is initialized by a host copy or a kernel before it is read.
    unsafe { context.stream.alloc(bytes) }.map_err(|error| backend_error("cuda_allocate", error))
}

fn finish(context: &Arc<Context>, dtype: DType, len: usize, buffer: CudaSlice<u8>) -> CudaStorage {
    CudaStorage {
        buffer: Arc::new(buffer),
        dtype,
        len,
        context: Arc::clone(context),
    }
}

fn storage<'a>(op: &'static str, view: View<'a>) -> Result<&'a CudaStorage> {
    match view.storage() {
        Storage::Cuda(storage) => Ok(storage),
        other => Err(Error::DeviceMismatch {
            op,
            expected: view.device(),
            got: other.device(),
        }),
    }
}

fn check_context(op: &'static str, context: &Arc<Context>, views: &[View<'_>]) -> Result<()> {
    for &view in views {
        let value = storage(op, view)?;
        if !Arc::ptr_eq(context, &value.context) {
            return Err(Error::DeviceMismatch {
                op,
                expected: Device::Cuda(context.ordinal),
                got: value.device(),
            });
        }
    }
    Ok(())
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
    supported(op, first.dtype(), first.device())?;
    Ok(first.dtype())
}

fn suffix(dtype: DType) -> &'static str {
    match dtype {
        DType::F16 => "f16",
        DType::F32 => "f32",
        DType::I64 => "i64",
        DType::Bool => "bool",
        DType::BF16 | DType::F64 => unreachable!("unsupported CUDA dtype"),
    }
}

fn typed_name(base: &str, dtype: DType) -> String {
    format!("{base}_{}", suffix(dtype))
}

fn launch_config(op: &'static str, len: usize) -> Result<LaunchConfig> {
    let blocks = len.div_ceil(THREADS as usize);
    let blocks = u32::try_from(blocks)
        .ok()
        .filter(|&value| value <= i32::MAX as u32)
        .ok_or_else(|| Error::InvalidArg {
            op,
            msg: format!("CUDA grid requires {blocks} blocks, exceeding the grid-x limit"),
        })?;
    Ok(LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn row_parallel_config(op: &'static str, rows: usize) -> Result<LaunchConfig> {
    let rows = u32::try_from(rows)
        .ok()
        .filter(|&value| value <= i32::MAX as u32)
        .ok_or_else(|| Error::InvalidArg {
            op,
            msg: format!("CUDA grid requires {rows} row blocks, exceeding the grid-x limit"),
        })?;
    Ok(LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

macro_rules! launch_configured {
    ($context:expr, $name:expr, $config:expr; $($arg:expr),+ $(,)?) => {{
        let function = $context.function(&$name)?;
        let mut launch = $context.stream.launch_builder(&function);
        $(launch.arg($arg);)+
        // SAFETY: each call site follows the corresponding `kernels.cu` ABI exactly,
        // and all buffers and metadata remain alive until cudarc's events complete.
        unsafe { launch.launch($config) }.map_err(|error| backend_error("cuda_launch", error))?;
        Ok::<(), Error>(())
    }};
}

macro_rules! launch {
    ($context:expr, $name:expr, $len:expr; $($arg:expr),+ $(,)?) => {{
        let len = $len;
        if len != 0 {
            launch_configured!($context, $name, launch_config("cuda_launch", len)?;
                $($arg),+
            )?;
        }
        Ok::<(), Error>(())
    }};
}

struct DeviceLayout {
    dims: CudaSlice<u64>,
    strides: CudaSlice<u64>,
    rank: u32,
    offset: u64,
}

fn upload_u64(context: &Context, values: impl IntoIterator<Item = u64>) -> Result<CudaSlice<u64>> {
    let mut values: Vec<u64> = values.into_iter().collect();
    if values.is_empty() {
        values.push(0);
    }
    context
        .stream
        .clone_htod(&values)
        .map_err(|error| backend_error("cuda_metadata", error))
}

fn device_layout(context: &Context, layout: &Layout) -> Result<DeviceLayout> {
    Ok(DeviceLayout {
        dims: upload_u64(context, layout.dims().iter().map(|&value| value as u64))?,
        strides: upload_u64(context, layout.strides().iter().map(|&value| value as u64))?,
        rank: layout.rank() as u32,
        offset: layout.offset() as u64,
    })
}

fn is_row_major(layout: &Layout) -> bool {
    let mut expected = 1usize;
    for (&dim, &stride) in layout.dims().iter().zip(layout.strides()).rev() {
        if stride != expected {
            return false;
        }
        let Some(next) = expected.checked_mul(dim) else {
            return false;
        };
        expected = next;
    }
    true
}

fn direct_contiguous(layout: &Layout) -> bool {
    layout.offset() == 0 && layout.is_contiguous()
}

fn checked_product(op: &'static str, values: impl IntoIterator<Item = usize>) -> Result<usize> {
    values.into_iter().try_fold(1usize, |product, value| {
        product.checked_mul(value).ok_or_else(|| Error::InvalidArg {
            op,
            msg: "output element count overflows usize".to_owned(),
        })
    })
}

fn reduced_len(layout: &Layout, axis: usize) -> usize {
    layout
        .dims()
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != axis)
        .map(|(_, value)| value)
        .product()
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

fn copy_name(dtype: DType, into: bool) -> String {
    typed_name(if into { "copy_into" } else { "copy" }, dtype)
}

fn validate_indices(
    context: &Arc<Context>,
    op: &'static str,
    indices: View<'_>,
    axis: usize,
    bound: usize,
) -> Result<()> {
    let input = storage(op, indices)?;
    let layout = device_layout(context, indices.layout())?;
    let len = indices.layout().num_elements() as u64;
    let bound64 = bound as u64;
    loop {
        // Keep reservation, launch, and descriptor publication atomic with
        // respect to collection so a slot cannot be read or reused early.
        let mut validation = context.validation.lock().map_err(|_| Error::Backend {
            op,
            msg: "CUDA validation state poisoned".to_owned(),
        })?;
        if validation.pending.len() >= VALIDATION_SLOTS {
            drop(validation);
            collect_validations(context)?;
            continue;
        }

        let slot = validation.pending.len();
        let slot64 = slot as u64;
        let status = &mut validation.status;
        launch!(context, "validate_indices", 1;
            input.buffer.as_ref(), status, &layout.dims, &layout.strides,
            &layout.rank, &layout.offset, &len, &bound64, &slot64
        )?;
        validation
            .pending
            .push(PendingValidation { op, axis, bound });
        return Ok(());
    }
}

fn collect_validations(context: &Arc<Context>) -> Result<()> {
    let mut validation = context.validation.lock().map_err(|_| Error::Backend {
        op: "transfer_out",
        msg: "CUDA validation state poisoned".to_owned(),
    })?;
    if validation.pending.is_empty() {
        return Ok(());
    }

    // This ordered DtoH also waits for validation kernels that a concurrent
    // host read may not have preceded when it copied its tensor output.
    let words = context
        .stream
        .clone_dtoh(&validation.status)
        .map_err(|error| backend_error("transfer_out", error))?;
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

struct MatmulPlan {
    batch: Vec<u64>,
    lhs_batch: Vec<u64>,
    rhs_batch: Vec<u64>,
    params: [u64; 9],
    batches: usize,
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
    let mut batch = vec![0; rank];
    let mut lhs_batch = vec![0; rank];
    let mut rhs_batch = vec![0; rank];
    for axis in 0..rank {
        let li = (axis + lb.len()).checked_sub(rank);
        let ri = (axis + rb.len()).checked_sub(rank);
        let ld = li.map_or(1, |index| lb[index]);
        let rd = ri.map_or(1, |index| rb[index]);
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
            .filter(|&index| lb[index] != 1)
            .map_or(0, |index| lhs.strides()[index] as u64);
        rhs_batch[axis] = ri
            .filter(|&index| rb[index] != 1)
            .map_or(0, |index| rhs.strides()[index] as u64);
    }
    let batches = checked_product("matmul", batch.iter().map(|&value| value as usize))?;
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
        batches,
        len,
    })
}

fn tiled_matmul_config(plan: &MatmulPlan) -> Option<LaunchConfig> {
    const MAX_GRID_X: u64 = i32::MAX as u64;
    const MAX_GRID_YZ: u64 = 65_535;

    let x = plan.params[2].div_ceil(16);
    let y = plan.params[0].div_ceil(16);
    let z = u64::try_from(plan.batches).ok()?;
    if x == 0 || y == 0 || z == 0 || x > MAX_GRID_X || y > MAX_GRID_YZ || z > MAX_GRID_YZ {
        return None;
    }
    Some(LaunchConfig {
        grid_dim: (x as u32, y as u32, z as u32),
        block_dim: (16, 16, 1),
        shared_mem_bytes: 0,
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

impl CudaBackend {
    fn context(&self) -> Result<Arc<Context>> {
        context(self.ordinal)
    }

    fn binary_kernel(
        &self,
        base: &'static str,
        lhs: View<'_>,
        rhs: View<'_>,
        output_dtype: DType,
        code: u32,
    ) -> Result<Storage> {
        let dtype = same_dtype(base, &[lhs, rhs])?;
        let context = self.context()?;
        check_context(base, &context, &[lhs, rhs])?;
        let a = storage(base, lhs)?;
        let b = storage(base, rhs)?;
        let ll = device_layout(&context, lhs.layout())?;
        let rl = device_layout(&context, rhs.layout())?;
        let len = lhs.layout().num_elements();
        let mut output = allocate_raw(&context, output_dtype, len)?;
        let len64 = len as u64;
        let contiguous =
            u32::from(direct_contiguous(lhs.layout()) && direct_contiguous(rhs.layout()));
        launch!(&context, typed_name(base, dtype), len;
            a.buffer.as_ref(), b.buffer.as_ref(), &mut output,
            &ll.dims, &ll.strides, &ll.rank, &ll.offset,
            &rl.dims, &rl.strides, &rl.rank, &rl.offset,
            &len64, &code, &contiguous
        )?;
        Ok(Storage::Cuda(finish(&context, output_dtype, len, output)))
    }

    fn dense_inputs(&self, inputs: &[View<'_>]) -> Result<Vec<Storage>> {
        inputs.iter().map(|&view| self.copy_strided(view)).collect()
    }

    fn fused_layer_norm(&self, inputs: &[View<'_>], scalars: &[f64]) -> Result<Vec<Storage>> {
        if inputs.len() == 4 {
            if !scalars.is_empty() {
                return Err(Error::InvalidArg {
                    op: "fused_layer_norm_backward_input",
                    msg: "backward accepts no scalars".to_owned(),
                });
            }
            let [grad, xhat, inv, weight] = inputs else {
                unreachable!()
            };
            if !matches!(grad.dtype(), DType::F16 | DType::F32) {
                return Err(unsupported("fused_layer_norm_backward_input", *grad));
            }
            if weight.dtype() != grad.dtype() {
                return Err(Error::DTypeMismatch {
                    op: "fused_layer_norm_backward_input",
                    expected: grad.dtype(),
                    got: weight.dtype(),
                });
            }
            for stat in [xhat, inv] {
                if stat.dtype() != DType::F32 {
                    return Err(Error::DTypeMismatch {
                        op: "fused_layer_norm_backward_input",
                        expected: DType::F32,
                        got: stat.dtype(),
                    });
                }
            }
            let width = *grad
                .layout()
                .dims()
                .last()
                .ok_or_else(|| Error::InvalidArg {
                    op: "fused_layer_norm_backward_input",
                    msg: "requires rank >= 1".to_owned(),
                })?;
            if width == 0 {
                return Err(Error::InvalidArg {
                    op: "fused_layer_norm_backward_input",
                    msg: "last-axis width must be non-zero".to_owned(),
                });
            }
            let rank = grad.layout().rank();
            if xhat.layout().shape() != grad.layout().shape()
                || inv.layout().rank() != rank
                || inv.layout().dims()[..rank - 1] != grad.layout().dims()[..rank - 1]
                || inv.layout().dims()[rank - 1] != 1
                || weight.layout().rank() != 1
                || weight.layout().dims()[0] != width
            {
                return Err(Error::ShapeMismatch {
                    op: "fused_layer_norm_backward_input",
                    lhs: grad.layout().shape().clone(),
                    rhs: xhat.layout().shape().clone(),
                });
            }
            let rows = grad.layout().num_elements() / width;
            let context = self.context()?;
            check_context("fused_layer_norm_backward_input", &context, inputs)?;
            let g = storage("fused_layer_norm_backward_input", *grad)?;
            let h = storage("fused_layer_norm_backward_input", *xhat)?;
            let i = storage("fused_layer_norm_backward_input", *inv)?;
            let w = storage("fused_layer_norm_backward_input", *weight)?;
            let gl = device_layout(&context, grad.layout())?;
            let hl = device_layout(&context, xhat.layout())?;
            let il = device_layout(&context, inv.layout())?;
            let ws = upload_u64(
                &context,
                weight.layout().strides().iter().map(|&v| v as u64),
            )?;
            let wo = weight.layout().offset() as u64;
            let rows64 = rows as u64;
            let width64 = width as u64;
            let mut output = allocate_raw(&context, grad.dtype(), grad.layout().num_elements())?;
            let parallel = width >= 64;
            let kernel = typed_name(
                if parallel {
                    "layer_norm_backward_parallel"
                } else {
                    "layer_norm_backward"
                },
                grad.dtype(),
            );
            if parallel && rows != 0 {
                launch_configured!(&context, kernel, row_parallel_config("cuda_launch", rows)?;
                    g.buffer.as_ref(), h.buffer.as_ref(), i.buffer.as_ref(), w.buffer.as_ref(),
                    &mut output, &gl.dims, &gl.strides, &gl.rank, &gl.offset,
                    &hl.dims, &hl.strides, &hl.rank, &hl.offset,
                    &il.dims, &il.strides, &il.rank, &il.offset, &ws, &wo, &rows64, &width64
                )?;
            } else {
                launch!(&context, kernel, rows;
                    g.buffer.as_ref(), h.buffer.as_ref(), i.buffer.as_ref(), w.buffer.as_ref(),
                    &mut output, &gl.dims, &gl.strides, &gl.rank, &gl.offset,
                    &hl.dims, &hl.strides, &hl.rank, &hl.offset,
                    &il.dims, &il.strides, &il.rank, &il.offset, &ws, &wo, &rows64, &width64
                )?;
            }
            return Ok(vec![Storage::Cuda(finish(
                &context,
                grad.dtype(),
                grad.layout().num_elements(),
                output,
            ))]);
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
        let width = *x.layout().dims().last().ok_or_else(|| Error::InvalidArg {
            op: "fused_layer_norm",
            msg: "requires rank >= 1".to_owned(),
        })?;
        if width == 0 {
            return Err(Error::InvalidArg {
                op: "fused_layer_norm",
                msg: "last-axis width must be non-zero".to_owned(),
            });
        }
        for affine in [weight, bias] {
            if affine.layout().rank() != 1 || affine.layout().dims()[0] != width {
                return Err(Error::ShapeMismatch {
                    op: "fused_layer_norm",
                    lhs: x.layout().shape().clone(),
                    rhs: affine.layout().shape().clone(),
                });
            }
        }
        let eps64 = scalars[0];
        if !(eps64.is_finite() && eps64 > 0.0) {
            return Err(Error::InvalidArg {
                op: "fused_layer_norm",
                msg: format!("eps must be finite and positive, got {eps64}"),
            });
        }
        let save = match scalars.get(1) {
            None => false,
            Some(&1.0) => true,
            Some(value) => {
                return Err(Error::InvalidArg {
                    op: "fused_layer_norm",
                    msg: format!("save_stats must be encoded as 1, got {value}"),
                });
            }
        };
        let rows = x.layout().num_elements() / width;
        let context = self.context()?;
        check_context("fused_layer_norm", &context, inputs)?;
        let xv = storage("fused_layer_norm", *x)?;
        let wv = storage("fused_layer_norm", *weight)?;
        let bv = storage("fused_layer_norm", *bias)?;
        let xl = device_layout(&context, x.layout())?;
        let ws = upload_u64(
            &context,
            weight.layout().strides().iter().map(|&v| v as u64),
        )?;
        let bs = upload_u64(&context, bias.layout().strides().iter().map(|&v| v as u64))?;
        let wo = weight.layout().offset() as u64;
        let bo = bias.layout().offset() as u64;
        let rows64 = rows as u64;
        let width64 = width as u64;
        let eps = eps64 as f32;
        let save32 = u32::from(save);
        let len = x.layout().num_elements();
        let mut output = allocate_raw(&context, dtype, len)?;
        let mut xhat = allocate_raw(&context, DType::F32, if save { len } else { 1 })?;
        let mut invout = allocate_raw(&context, DType::F32, if save { rows } else { 1 })?;
        let parallel = width >= 64;
        let kernel = typed_name(
            if parallel {
                "layer_norm_parallel"
            } else {
                "layer_norm"
            },
            dtype,
        );
        if parallel && rows != 0 {
            launch_configured!(&context, kernel, row_parallel_config("cuda_launch", rows)?;
                xv.buffer.as_ref(), wv.buffer.as_ref(), bv.buffer.as_ref(), &mut output,
                &mut xhat, &mut invout, &xl.dims, &xl.strides, &xl.rank, &xl.offset,
                &ws, &wo, &bs, &bo, &rows64, &width64, &eps, &save32
            )?;
        } else {
            launch!(&context, kernel, rows;
                xv.buffer.as_ref(), wv.buffer.as_ref(), bv.buffer.as_ref(), &mut output,
                &mut xhat, &mut invout, &xl.dims, &xl.strides, &xl.rank, &xl.offset,
                &ws, &wo, &bs, &bo, &rows64, &width64, &eps, &save32
            )?;
        }
        let mut values = vec![Storage::Cuda(finish(&context, dtype, len, output))];
        if save {
            values.push(Storage::Cuda(finish(&context, DType::F32, len, xhat)));
            values.push(Storage::Cuda(finish(&context, DType::F32, rows, invout)));
        }
        Ok(values)
    }

    fn fused_sgd(&self, inputs: &[View<'_>], scalars: &[f64]) -> Result<Vec<Storage>> {
        if !(inputs.len() == 2 || inputs.len() == 3) || scalars.len() != 3 {
            return Err(Error::InvalidArg {
                op: "fused_sgd_step",
                msg: "expected two or three inputs and three scalars".to_owned(),
            });
        }
        let dtype = same_dtype("fused_sgd_step", &inputs[..2])?;
        if !matches!(dtype, DType::F16 | DType::F32) {
            return Err(unsupported("fused_sgd_step", inputs[0]));
        }
        validate_optimizer("fused_sgd_step", inputs, 2)?;
        crate::backend::cpu::fused::validate_sgd_scalars(
            "fused_sgd_step",
            scalars[0],
            scalars[1],
            scalars[2],
            dtype,
        )?;
        let use_momentum = scalars[1] as f32 != 0.0;
        if inputs.len() == 3 && !use_momentum {
            return Err(Error::InvalidArg {
                op: "fused_sgd_step",
                msg: "a velocity input requires non-zero momentum".to_owned(),
            });
        }
        let context = self.context()?;
        check_context("fused_sgd_step", &context, inputs)?;
        let dense = self.dense_inputs(inputs)?;
        let values: Vec<&CudaStorage> = dense
            .iter()
            .map(|value| match value {
                Storage::Cuda(value) => value,
                _ => unreachable!(),
            })
            .collect();
        let len = inputs[0].layout().num_elements();
        let mut next = allocate_raw(&context, dtype, len)?;
        let mut next_velocity =
            allocate_raw(&context, DType::F32, if use_momentum { len } else { 1 })?;
        let hp = [scalars[0] as f32, scalars[1] as f32, scalars[2] as f32];
        let hp_dev = context
            .stream
            .clone_htod(&hp)
            .map_err(|e| backend_error("fused_sgd_step", e))?;
        let hasv = u32::from(values.len() == 3);
        let usem = u32::from(use_momentum);
        let len64 = len as u64;
        let velocity = values
            .get(2)
            .map_or(values[0].buffer.as_ref(), |v| v.buffer.as_ref());
        launch!(&context, typed_name("sgd", dtype), len;
            values[0].buffer.as_ref(), values[1].buffer.as_ref(), velocity,
            &mut next, &mut next_velocity, &len64, &hp_dev, &hasv, &usem
        )?;
        let mut output = vec![Storage::Cuda(finish(&context, dtype, len, next))];
        if use_momentum {
            output.push(Storage::Cuda(finish(
                &context,
                DType::F32,
                len,
                next_velocity,
            )));
        }
        Ok(output)
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
        validate_optimizer("fused_adam_step", inputs, 2)?;
        crate::backend::cpu::fused::validate_adam_scalars("fused_adam_step", scalars, dtype)?;
        let context = self.context()?;
        check_context("fused_adam_step", &context, inputs)?;
        let dense = self.dense_inputs(inputs)?;
        let values: Vec<&CudaStorage> = dense
            .iter()
            .map(|value| match value {
                Storage::Cuda(value) => value,
                _ => unreachable!(),
            })
            .collect();
        let len = inputs[0].layout().num_elements();
        let mut next = allocate_raw(&context, dtype, len)?;
        let mut next_m = allocate_raw(&context, DType::F32, len)?;
        let mut next_v = allocate_raw(&context, DType::F32, len)?;
        let hp: Vec<f32> = scalars.iter().map(|&value| value as f32).collect();
        let hp_dev = context
            .stream
            .clone_htod(&hp)
            .map_err(|e| backend_error("fused_adam_step", e))?;
        let len64 = len as u64;
        launch!(&context, typed_name("adam", dtype), len;
            values[0].buffer.as_ref(), values[1].buffer.as_ref(),
            values[2].buffer.as_ref(), values[3].buffer.as_ref(),
            &mut next, &mut next_m, &mut next_v, &len64, &hp_dev
        )?;
        Ok(vec![
            Storage::Cuda(finish(&context, dtype, len, next)),
            Storage::Cuda(finish(&context, DType::F32, len, next_m)),
            Storage::Cuda(finish(&context, DType::F32, len, next_v)),
        ])
    }
}

fn validate_optimizer(
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
    for (index, input) in inputs.iter().enumerate() {
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

impl BackendOps for CudaBackend {
    fn transfer_in(&self, host: CpuStorage) -> Result<Storage> {
        let context = self.context()?;
        let dtype = host.dtype();
        supported("from_vec", dtype, Device::Cuda(self.ordinal))?;
        let len = host.len();
        let mut raw = allocate_raw(&context, dtype, len)?;
        if len != 0 {
            match host {
                CpuStorage::F16(values) => {
                    let mut view = unsafe { raw.transmute_mut::<half::f16>(len) }.unwrap();
                    context.stream.memcpy_htod(values.as_slice(), &mut view)
                }
                CpuStorage::F32(values) => {
                    let mut view = unsafe { raw.transmute_mut::<f32>(len) }.unwrap();
                    context.stream.memcpy_htod(values.as_slice(), &mut view)
                }
                CpuStorage::I64(values) => {
                    let mut view = unsafe { raw.transmute_mut::<i64>(len) }.unwrap();
                    context.stream.memcpy_htod(values.as_slice(), &mut view)
                }
                CpuStorage::Bool(values) => {
                    let bytes: Vec<u8> = values.iter().map(|&value| u8::from(value)).collect();
                    context.stream.memcpy_htod(&bytes, &mut raw)
                }
                CpuStorage::BF16(_) | CpuStorage::F64(_) => unreachable!("dtype validated"),
            }
            .map_err(|error| backend_error("from_vec", error))?;
        }
        Ok(Storage::Cuda(finish(&context, dtype, len, raw)))
    }

    fn transfer_out(&self, x: View<'_>) -> Result<CpuStorage> {
        supported("transfer_out", x.dtype(), x.device())?;
        let context = self.context()?;
        check_context("transfer_out", &context, &[x])?;
        let dense = if x.layout().is_contiguous()
            && x.layout().offset() == 0
            && x.layout().num_elements() == storage("transfer_out", x)?.len
        {
            None
        } else {
            Some(self.copy_strided(x)?)
        };
        let value = match &dense {
            Some(Storage::Cuda(value)) => value,
            Some(_) => unreachable!(),
            None => storage("transfer_out", x)?,
        };
        let len = x.layout().num_elements();
        let result = match value.dtype {
            DType::F16 => {
                let view = unsafe { value.buffer.transmute::<half::f16>(len) }.unwrap();
                CpuStorage::F16(Arc::new(
                    context
                        .stream
                        .clone_dtoh(&view)
                        .map_err(|e| backend_error("transfer_out", e))?,
                ))
            }
            DType::F32 => {
                let view = unsafe { value.buffer.transmute::<f32>(len) }.unwrap();
                CpuStorage::F32(Arc::new(
                    context
                        .stream
                        .clone_dtoh(&view)
                        .map_err(|e| backend_error("transfer_out", e))?,
                ))
            }
            DType::I64 => {
                let view = unsafe { value.buffer.transmute::<i64>(len) }.unwrap();
                CpuStorage::I64(Arc::new(
                    context
                        .stream
                        .clone_dtoh(&view)
                        .map_err(|e| backend_error("transfer_out", e))?,
                ))
            }
            DType::Bool => {
                let bytes = context
                    .stream
                    .clone_dtoh(value.buffer.as_ref())
                    .map_err(|e| backend_error("transfer_out", e))?;
                CpuStorage::Bool(Arc::new(
                    bytes[..len].iter().map(|&value| value != 0).collect(),
                ))
            }
            DType::BF16 | DType::F64 => unreachable!("unsupported CUDA storage"),
        };
        collect_validations(&context)?;
        Ok(result)
    }

    fn copy_strided(&self, x: View<'_>) -> Result<Storage> {
        supported("copy_strided", x.dtype(), x.device())?;
        let context = self.context()?;
        check_context("copy_strided", &context, &[x])?;
        let input = storage("copy_strided", x)?;
        let layout = device_layout(&context, x.layout())?;
        let len = x.layout().num_elements();
        let mut output = allocate_raw(&context, x.dtype(), len)?;
        let len64 = len as u64;
        let contiguous = u32::from(is_row_major(x.layout()));
        launch!(&context, copy_name(x.dtype(), false), len;
            input.buffer.as_ref(), &mut output, &layout.dims, &layout.strides,
            &layout.rank, &layout.offset, &len64, &contiguous
        )?;
        Ok(Storage::Cuda(finish(&context, x.dtype(), len, output)))
    }

    fn copy_into(&self, src: View<'_>, dst: &mut Storage, dst_layout: &Layout) -> Result<()> {
        if src.layout().shape() != dst_layout.shape() {
            return Err(Error::ShapeMismatch {
                op: "copy_into",
                lhs: src.layout().shape().clone(),
                rhs: dst_layout.shape().clone(),
            });
        }
        let input = storage("copy_into", src)?;
        let Storage::Cuda(output) = dst else {
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
        let target = Arc::get_mut(&mut output.buffer).ok_or_else(|| Error::InvalidArg {
            op: "copy_into",
            msg: "destination storage is already shared".to_owned(),
        })?;
        let source_layout = device_layout(&input.context, src.layout())?;
        let target_layout = device_layout(&input.context, dst_layout)?;
        let len = src.layout().num_elements();
        let len64 = len as u64;
        let contiguous = u32::from(is_row_major(src.layout()) && is_row_major(dst_layout));
        launch!(&input.context, copy_name(input.dtype, true), len;
            input.buffer.as_ref(), target,
            &source_layout.dims, &source_layout.strides, &source_layout.rank, &source_layout.offset,
            &target_layout.dims, &target_layout.strides, &target_layout.rank, &target_layout.offset,
            &len64, &contiguous
        )
    }

    fn full(&self, len: usize, dtype: DType, value: f64) -> Result<Storage> {
        supported("full", dtype, Device::Cuda(self.ordinal))?;
        let context = self.context()?;
        let mut output = allocate_raw(&context, dtype, len)?;
        let len64 = len as u64;
        match dtype {
            DType::F16 => {
                let value = value as f32;
                launch!(&context, "full_f16", len; &mut output, &len64, &value)?;
            }
            DType::F32 => {
                let value = value as f32;
                launch!(&context, "full_f32", len; &mut output, &len64, &value)?;
            }
            DType::I64 => {
                let value = value as i64;
                launch!(&context, "full_i64", len; &mut output, &len64, &value)?;
            }
            DType::Bool => {
                let value = u32::from(value != 0.0);
                launch!(&context, "full_bool", len; &mut output, &len64, &value)?;
            }
            DType::BF16 | DType::F64 => unreachable!("dtype validated"),
        }
        Ok(Storage::Cuda(finish(&context, dtype, len, output)))
    }

    fn cast(&self, x: View<'_>, to: DType) -> Result<Storage> {
        supported("to_dtype", x.dtype(), x.device())?;
        supported("to_dtype", to, x.device())?;
        if x.dtype() == to {
            return self.copy_strided(x);
        }
        let context = self.context()?;
        check_context("to_dtype", &context, &[x])?;
        let input = storage("to_dtype", x)?;
        let layout = device_layout(&context, x.layout())?;
        let len = x.layout().num_elements();
        let mut output = allocate_raw(&context, to, len)?;
        let len64 = len as u64;
        let contiguous = u32::from(direct_contiguous(x.layout()));
        let name = format!("cast_{}_to_{}", suffix(x.dtype()), suffix(to));
        launch!(&context, name, len;
            input.buffer.as_ref(), &mut output, &layout.dims, &layout.strides,
            &layout.rank, &layout.offset, &len64, &contiguous
        )?;
        Ok(Storage::Cuda(finish(&context, to, len, output)))
    }

    fn binary(&self, op: BinaryOp, lhs: View<'_>, rhs: View<'_>) -> Result<Storage> {
        if lhs.dtype() == DType::Bool {
            return Err(unsupported("binary", lhs));
        }
        self.binary_kernel("binary", lhs, rhs, lhs.dtype(), op_code_binary(op))
    }

    fn binary_scalar(&self, op: BinaryOp, x: View<'_>, scalar: f64) -> Result<Storage> {
        supported("binary_scalar", x.dtype(), x.device())?;
        if x.dtype() == DType::Bool {
            return Err(unsupported("binary_scalar", x));
        }
        let context = self.context()?;
        check_context("binary_scalar", &context, &[x])?;
        let input = storage("binary_scalar", x)?;
        let layout = device_layout(&context, x.layout())?;
        let len = x.layout().num_elements();
        let mut output = allocate_raw(&context, x.dtype(), len)?;
        let len64 = len as u64;
        let code = op_code_binary(op);
        let contiguous = u32::from(direct_contiguous(x.layout()));
        if x.dtype() == DType::I64 {
            let scalar = scalar as i64;
            launch!(&context, typed_name("scalar", x.dtype()), len;
                input.buffer.as_ref(), &mut output, &layout.dims, &layout.strides,
                &layout.rank, &layout.offset, &len64, &scalar, &code, &contiguous
            )?;
        } else {
            let scalar = scalar as f32;
            launch!(&context, typed_name("scalar", x.dtype()), len;
                input.buffer.as_ref(), &mut output, &layout.dims, &layout.strides,
                &layout.rank, &layout.offset, &len64, &scalar, &code, &contiguous
            )?;
        }
        Ok(Storage::Cuda(finish(&context, x.dtype(), len, output)))
    }

    fn unary(&self, op: UnaryOp, x: View<'_>) -> Result<Storage> {
        supported("unary", x.dtype(), x.device())?;
        if x.dtype() == DType::Bool
            || (x.dtype() == DType::I64 && !matches!(op, UnaryOp::Neg | UnaryOp::Abs))
        {
            return Err(unsupported("unary", x));
        }
        let context = self.context()?;
        check_context("unary", &context, &[x])?;
        let input = storage("unary", x)?;
        let layout = device_layout(&context, x.layout())?;
        let len = x.layout().num_elements();
        let mut output = allocate_raw(&context, x.dtype(), len)?;
        let len64 = len as u64;
        let code = op_code_unary(op);
        let contiguous = u32::from(direct_contiguous(x.layout()));
        launch!(&context, typed_name("unary", x.dtype()), len;
            input.buffer.as_ref(), &mut output, &layout.dims, &layout.strides,
            &layout.rank, &layout.offset, &len64, &code, &contiguous
        )?;
        Ok(Storage::Cuda(finish(&context, x.dtype(), len, output)))
    }

    fn compare(&self, op: CmpOp, lhs: View<'_>, rhs: View<'_>) -> Result<Storage> {
        self.binary_kernel("compare", lhs, rhs, DType::Bool, op_code_cmp(op))
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
        let context = self.context()?;
        check_context("where", &context, &[cond, on_true, on_false])?;
        let c = storage("where", cond)?;
        let t = storage("where", on_true)?;
        let f = storage("where", on_false)?;
        let cl = device_layout(&context, cond.layout())?;
        let tl = device_layout(&context, on_true.layout())?;
        let fl = device_layout(&context, on_false.layout())?;
        let len = cond.layout().num_elements();
        let mut output = allocate_raw(&context, dtype, len)?;
        let len64 = len as u64;
        let contiguous = u32::from(
            direct_contiguous(cond.layout())
                && direct_contiguous(on_true.layout())
                && direct_contiguous(on_false.layout()),
        );
        launch!(&context, typed_name("where", dtype), len;
            c.buffer.as_ref(), t.buffer.as_ref(), f.buffer.as_ref(), &mut output,
            &cl.dims, &cl.strides, &cl.rank, &cl.offset,
            &tl.dims, &tl.strides, &tl.rank, &tl.offset,
            &fl.dims, &fl.strides, &fl.rank, &fl.offset, &len64, &contiguous
        )?;
        Ok(Storage::Cuda(finish(&context, dtype, len, output)))
    }

    fn masked_fill(&self, x: View<'_>, mask: View<'_>, value: f64) -> Result<Storage> {
        supported("masked_fill", x.dtype(), x.device())?;
        if mask.dtype() != DType::Bool {
            return Err(Error::DTypeMismatch {
                op: "masked_fill",
                expected: DType::Bool,
                got: mask.dtype(),
            });
        }
        let context = self.context()?;
        check_context("masked_fill", &context, &[x, mask])?;
        let input = storage("masked_fill", x)?;
        let mask_storage = storage("masked_fill", mask)?;
        let xl = device_layout(&context, x.layout())?;
        let ml = device_layout(&context, mask.layout())?;
        let len = x.layout().num_elements();
        let mut output = allocate_raw(&context, x.dtype(), len)?;
        let len64 = len as u64;
        let contiguous =
            u32::from(direct_contiguous(x.layout()) && direct_contiguous(mask.layout()));
        if x.dtype() == DType::I64 {
            let fill = value as i64;
            launch!(&context, typed_name("masked", x.dtype()), len;
                input.buffer.as_ref(), mask_storage.buffer.as_ref(), &mut output,
                &xl.dims, &xl.strides, &xl.rank, &xl.offset,
                &ml.dims, &ml.strides, &ml.rank, &ml.offset, &len64, &fill, &contiguous
            )?;
        } else {
            let fill = if x.dtype() == DType::Bool {
                f32::from(value != 0.0)
            } else {
                value as f32
            };
            launch!(&context, typed_name("masked", x.dtype()), len;
                input.buffer.as_ref(), mask_storage.buffer.as_ref(), &mut output,
                &xl.dims, &xl.strides, &xl.rank, &xl.offset,
                &ml.dims, &ml.strides, &ml.rank, &ml.offset, &len64, &fill, &contiguous
            )?;
        }
        Ok(Storage::Cuda(finish(&context, x.dtype(), len, output)))
    }

    fn reduce(&self, op: ReduceOp, x: View<'_>, axis: usize) -> Result<Storage> {
        supported("reduce", x.dtype(), x.device())?;
        check_axis("reduce", x, axis)?;
        if x.dtype() == DType::Bool {
            return Err(unsupported("reduce", x));
        }
        let context = self.context()?;
        check_context("reduce", &context, &[x])?;
        let input = storage("reduce", x)?;
        let layout = device_layout(&context, x.layout())?;
        let len = reduced_len(x.layout(), axis);
        let mut output = allocate_raw(&context, x.dtype(), len)?;
        let len64 = len as u64;
        let axis32 = axis as u32;
        let code = match op {
            ReduceOp::Sum => 0,
            ReduceOp::Mean => 1,
            ReduceOp::Max => 2,
            ReduceOp::Min => 3,
        };
        let parallel =
            matches!(x.dtype(), DType::F16 | DType::F32) && x.layout().dims()[axis] >= 64;
        let kernel = typed_name(
            if parallel {
                "reduce_parallel"
            } else {
                "reduce"
            },
            x.dtype(),
        );
        if parallel && len != 0 {
            launch_configured!(&context, kernel, row_parallel_config("cuda_launch", len)?;
                input.buffer.as_ref(), &mut output, &layout.dims, &layout.strides,
                &layout.rank, &layout.offset, &len64, &axis32, &code
            )?;
        } else {
            launch!(&context, kernel, len;
                input.buffer.as_ref(), &mut output, &layout.dims, &layout.strides,
                &layout.rank, &layout.offset, &len64, &axis32, &code
            )?;
        }
        Ok(Storage::Cuda(finish(&context, x.dtype(), len, output)))
    }

    fn arg_reduce(&self, op: ArgReduceOp, x: View<'_>, axis: usize) -> Result<Storage> {
        supported("arg_reduce", x.dtype(), x.device())?;
        check_axis("arg_reduce", x, axis)?;
        if x.dtype() == DType::Bool {
            return Err(unsupported("arg_reduce", x));
        }
        let context = self.context()?;
        check_context("arg_reduce", &context, &[x])?;
        let input = storage("arg_reduce", x)?;
        let layout = device_layout(&context, x.layout())?;
        let len = reduced_len(x.layout(), axis);
        let mut output = allocate_raw(&context, DType::I64, len)?;
        let len64 = len as u64;
        let axis32 = axis as u32;
        let code = u32::from(matches!(op, ArgReduceOp::ArgMin));
        launch!(&context, typed_name("arg_reduce", x.dtype()), len;
            input.buffer.as_ref(), &mut output, &layout.dims, &layout.strides,
            &layout.rank, &layout.offset, &len64, &axis32, &code
        )?;
        Ok(Storage::Cuda(finish(&context, DType::I64, len, output)))
    }

    fn matmul(&self, lhs: View<'_>, rhs: View<'_>) -> Result<Storage> {
        let dtype = same_dtype("matmul", &[lhs, rhs])?;
        if dtype == DType::Bool {
            return Err(unsupported("matmul", lhs));
        }
        let plan = matmul_plan(lhs.layout(), rhs.layout())?;
        let context = self.context()?;
        check_context("matmul", &context, &[lhs, rhs])?;
        let a = storage("matmul", lhs)?;
        let b = storage("matmul", rhs)?;
        let bd = upload_u64(&context, plan.batch.iter().copied())?;
        let lbs = upload_u64(&context, plan.lhs_batch.iter().copied())?;
        let rbs = upload_u64(&context, plan.rhs_batch.iter().copied())?;
        let p = upload_u64(&context, plan.params)?;
        let mut output = allocate_raw(&context, dtype, plan.len)?;
        let rank = plan.batch.len() as u32;
        let len64 = plan.len as u64;
        let tiled = matches!(dtype, DType::F16 | DType::F32)
            .then(|| tiled_matmul_config(&plan))
            .flatten();
        if let Some(config) = tiled {
            launch_configured!(&context, typed_name("matmul_tiled", dtype), config;
                a.buffer.as_ref(), b.buffer.as_ref(), &mut output,
                &bd, &lbs, &rbs, &rank, &p, &len64
            )?;
        } else {
            launch!(&context, typed_name("matmul", dtype), plan.len;
                a.buffer.as_ref(), b.buffer.as_ref(), &mut output,
                &bd, &lbs, &rbs, &rank, &p, &len64
            )?;
        }
        Ok(Storage::Cuda(finish(&context, dtype, plan.len, output)))
    }

    fn index_select(&self, x: View<'_>, axis: usize, indices: View<'_>) -> Result<Storage> {
        supported("index_select", x.dtype(), x.device())?;
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
        let context = self.context()?;
        check_context("index_select", &context, &[x, indices])?;
        validate_indices(
            &context,
            "index_select",
            indices,
            axis,
            x.layout().dims()[axis],
        )?;
        let xv = storage("index_select", x)?;
        let iv = storage("index_select", indices)?;
        let mut dims = x.layout().dims().to_vec();
        dims[axis] = indices.layout().num_elements();
        let len = checked_product("index_select", dims.iter().copied())?;
        let mut output = allocate_raw(&context, x.dtype(), len)?;
        let fast = x.dtype() != DType::Bool
            && axis == 0
            && direct_contiguous(x.layout())
            && direct_contiguous(indices.layout());
        if fast {
            let len64 = len as u64;
            let inner =
                checked_product("index_select", x.layout().dims()[1..].iter().copied())? as u64;
            let bound = x.layout().dims()[axis] as u64;
            launch!(&context, typed_name("index_select_axis0", x.dtype()), len;
                xv.buffer.as_ref(), iv.buffer.as_ref(), &mut output, &len64, &inner, &bound
            )?;
        } else {
            let xl = device_layout(&context, x.layout())?;
            let il = device_layout(&context, indices.layout())?;
            let od = upload_u64(&context, dims.iter().map(|&v| v as u64))?;
            let axis32 = axis as u32;
            let len64 = len as u64;
            launch!(&context, typed_name("index_select", x.dtype()), len;
                xv.buffer.as_ref(), iv.buffer.as_ref(), &mut output,
                &xl.dims, &xl.strides, &xl.rank, &xl.offset,
                &il.dims, &il.strides, &il.rank, &il.offset, &od, &axis32, &len64
            )?;
        }
        Ok(Storage::Cuda(finish(&context, x.dtype(), len, output)))
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
        let context = self.context()?;
        check_context("index_add", &context, &[x, indices, src])?;
        validate_indices(
            &context,
            "index_add",
            indices,
            axis,
            x.layout().dims()[axis],
        )?;
        let xv = storage("index_add", x)?;
        let iv = storage("index_add", indices)?;
        let sv = storage("index_add", src)?;
        let len = x.layout().num_elements();
        let mut output = allocate_raw(&context, dtype, len)?;
        let fast =
            axis == 0 && direct_contiguous(x.layout()) && direct_contiguous(indices.layout());
        let atomic = fast && matches!(dtype, DType::F32 | DType::I64);
        if atomic {
            if len != 0 {
                let bytes = len * element_size(dtype);
                let source = xv.buffer.slice(..bytes);
                let mut target = output.slice_mut(..bytes);
                context
                    .stream
                    .memcpy_dtod(&source, &mut target)
                    .map_err(|error| backend_error("index_add", error))?;
            }
            let src_len = src.layout().num_elements();
            let src_len64 = src_len as u64;
            let inner = checked_product("index_add", x.layout().dims()[1..].iter().copied())?;
            let inner64 = inner as u64;
            let bound = x.layout().dims()[axis] as u64;
            let sl = device_layout(&context, src.layout())?;
            launch!(&context, typed_name("index_add_axis0_atomic", dtype), src_len;
                iv.buffer.as_ref(), sv.buffer.as_ref(), &mut output,
                &src_len64, &inner64, &bound,
                &sl.dims, &sl.strides, &sl.rank, &sl.offset
            )?;
        } else if fast {
            let rows = x.layout().dims()[0] as u64;
            let count = indices.layout().num_elements() as u64;
            let inner =
                checked_product("index_add", x.layout().dims()[1..].iter().copied())? as u64;
            let bound = x.layout().dims()[axis] as u64;
            let sl = device_layout(&context, src.layout())?;
            launch!(&context, typed_name("index_add_axis0", dtype), rows as usize;
                xv.buffer.as_ref(), iv.buffer.as_ref(), sv.buffer.as_ref(), &mut output,
                &rows, &count, &inner, &bound,
                &sl.dims, &sl.strides, &sl.rank, &sl.offset
            )?;
        } else {
            let xl = device_layout(&context, x.layout())?;
            let il = device_layout(&context, indices.layout())?;
            let sl = device_layout(&context, src.layout())?;
            let axis32 = axis as u32;
            let len64 = len as u64;
            let src_len = src.layout().num_elements() as u64;
            launch!(&context, typed_name("index_add", dtype), len;
                xv.buffer.as_ref(), iv.buffer.as_ref(), sv.buffer.as_ref(), &mut output,
                &xl.dims, &xl.strides, &xl.rank, &xl.offset,
                &il.dims, &il.strides, &il.rank, &il.offset,
                &sl.dims, &sl.strides, &sl.rank, &sl.offset,
                &axis32, &len64, &src_len
            )?;
        }
        Ok(Storage::Cuda(finish(&context, dtype, len, output)))
    }

    fn gather(&self, x: View<'_>, axis: usize, indices: View<'_>) -> Result<Storage> {
        supported("gather", x.dtype(), x.device())?;
        check_axis("gather", x, axis)?;
        validate_gather_shape("gather", x, axis, indices)?;
        let context = self.context()?;
        check_context("gather", &context, &[x, indices])?;
        validate_indices(&context, "gather", indices, axis, x.layout().dims()[axis])?;
        let xv = storage("gather", x)?;
        let iv = storage("gather", indices)?;
        let len = indices.layout().num_elements();
        let mut output = allocate_raw(&context, x.dtype(), len)?;
        let fast = x.dtype() != DType::Bool
            && axis + 1 == x.layout().rank()
            && direct_contiguous(x.layout())
            && direct_contiguous(indices.layout());
        if fast {
            let len64 = len as u64;
            let classes = x.layout().dims()[axis] as u64;
            let picks = indices.layout().dims()[axis] as u64;
            let bound = x.layout().dims()[axis] as u64;
            launch!(&context, typed_name("gather_last", x.dtype()), len;
                xv.buffer.as_ref(), iv.buffer.as_ref(), &mut output,
                &len64, &classes, &picks, &bound
            )?;
        } else {
            let xl = device_layout(&context, x.layout())?;
            let il = device_layout(&context, indices.layout())?;
            let axis32 = axis as u32;
            let len64 = len as u64;
            launch!(&context, typed_name("gather", x.dtype()), len;
                xv.buffer.as_ref(), iv.buffer.as_ref(), &mut output,
                &xl.dims, &xl.strides, &xl.rank, &xl.offset,
                &il.dims, &il.strides, &il.rank, &il.offset, &axis32, &len64
            )?;
        }
        Ok(Storage::Cuda(finish(&context, x.dtype(), len, output)))
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
        validate_scatter_shape(x, axis, indices, src)?;
        let context = self.context()?;
        check_context("scatter_add", &context, &[x, indices, src])?;
        validate_indices(
            &context,
            "scatter_add",
            indices,
            axis,
            x.layout().dims()[axis],
        )?;
        let xv = storage("scatter_add", x)?;
        let iv = storage("scatter_add", indices)?;
        let sv = storage("scatter_add", src)?;
        let len = x.layout().num_elements();
        let mut output = allocate_raw(&context, dtype, len)?;
        let fast = axis + 1 == x.layout().rank()
            && direct_contiguous(x.layout())
            && direct_contiguous(indices.layout())
            && indices.layout().dims() == src.layout().dims();
        let atomic = fast && matches!(dtype, DType::F32 | DType::I64);
        if atomic {
            if len != 0 {
                let bytes = len * element_size(dtype);
                let source = xv.buffer.slice(..bytes);
                let mut target = output.slice_mut(..bytes);
                context
                    .stream
                    .memcpy_dtod(&source, &mut target)
                    .map_err(|error| backend_error("scatter_add", error))?;
            }
            let index_len = indices.layout().num_elements();
            let index_len64 = index_len as u64;
            let classes = x.layout().dims()[axis] as u64;
            let picks = indices.layout().dims()[axis] as u64;
            let bound = x.layout().dims()[axis] as u64;
            let sl = device_layout(&context, src.layout())?;
            launch!(&context, typed_name("scatter_add_last_atomic", dtype), index_len;
                iv.buffer.as_ref(), sv.buffer.as_ref(), &mut output,
                &index_len64, &classes, &picks, &bound,
                &sl.dims, &sl.strides, &sl.rank, &sl.offset
            )?;
        } else if fast {
            let len64 = len as u64;
            let classes = x.layout().dims()[axis] as u64;
            let picks = indices.layout().dims()[axis] as u64;
            let bound = x.layout().dims()[axis] as u64;
            let sl = device_layout(&context, src.layout())?;
            launch!(&context, typed_name("scatter_add_last", dtype), len;
                xv.buffer.as_ref(), iv.buffer.as_ref(), sv.buffer.as_ref(), &mut output,
                &len64, &classes, &picks, &bound,
                &sl.dims, &sl.strides, &sl.rank, &sl.offset
            )?;
        } else {
            let xl = device_layout(&context, x.layout())?;
            let il = device_layout(&context, indices.layout())?;
            let sl = device_layout(&context, src.layout())?;
            let axis32 = axis as u32;
            let len64 = len as u64;
            let index_len = indices.layout().num_elements() as u64;
            launch!(&context, typed_name("scatter_add", dtype), len;
                xv.buffer.as_ref(), iv.buffer.as_ref(), sv.buffer.as_ref(), &mut output,
                &xl.dims, &xl.strides, &xl.rank, &xl.offset,
                &il.dims, &il.strides, &il.rank, &il.offset,
                &sl.dims, &sl.strides, &sl.rank, &sl.offset,
                &axis32, &len64, &index_len
            )?;
        }
        Ok(Storage::Cuda(finish(&context, dtype, len, output)))
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
        let (geometry, kernel, len, first, second, code) = conv_plan(op, inputs, params)?;
        let context = self.context()?;
        check_context("conv", &context, inputs)?;
        let a = storage("conv", first)?;
        let b = second.map(|view| storage("conv", view)).transpose()?;
        let as_ = upload_u64(&context, first.layout().strides().iter().map(|&v| v as u64))?;
        let ao = first.layout().offset() as u64;
        let packed = conv_params(&geometry, params);
        let len64 = len as u64;
        let mut output = allocate_raw(&context, dtype, len)?;
        if let (Some(second), Some(b)) = (second, b) {
            let bs = upload_u64(
                &context,
                second.layout().strides().iter().map(|&v| v as u64),
            )?;
            let bo = second.layout().offset() as u64;
            if let Some(code) = code {
                launch!(&context, typed_name(kernel, dtype), len;
                    a.buffer.as_ref(), b.buffer.as_ref(), &mut output, &as_, &ao, &bs, &bo,
                    &packed[0], &packed[1], &packed[2], &packed[3], &packed[4],
                    &packed[5], &packed[6], &packed[7], &packed[8], &packed[9],
                    &packed[10], &packed[11], &packed[12], &packed[13], &packed[14],
                    &len64, &code
                )?;
            } else {
                launch!(&context, typed_name(kernel, dtype), len;
                    a.buffer.as_ref(), b.buffer.as_ref(), &mut output, &as_, &ao, &bs, &bo,
                    &packed[0], &packed[1], &packed[2], &packed[3], &packed[4],
                    &packed[5], &packed[6], &packed[7], &packed[8], &packed[9],
                    &packed[10], &packed[11], &packed[12], &packed[13], &packed[14], &len64
                )?;
            }
        } else {
            let code = code.unwrap_or(0);
            launch!(&context, typed_name(kernel, dtype), len;
                a.buffer.as_ref(), &mut output, &as_, &ao,
                &packed[0], &packed[1], &packed[2], &packed[3], &packed[4],
                &packed[5], &packed[6], &packed[7], &packed[8], &packed[9],
                &packed[10], &packed[11], &packed[12], &packed[13], &packed[14],
                &len64, &code
            )?;
        }
        Ok(Storage::Cuda(finish(&context, dtype, len, output)))
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
                if !matches!(input.dtype(), DType::F16 | DType::F32) {
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
                if width == 0 {
                    return Err(Error::InvalidArg {
                        op: "fused_softmax",
                        msg: "last-axis width must be non-zero".to_owned(),
                    });
                }
                let rows = input.layout().num_elements() / width;
                let context = self.context()?;
                check_context("fused_softmax", &context, inputs)?;
                let x = storage("fused_softmax", *input)?;
                let layout = device_layout(&context, input.layout())?;
                let mut output =
                    allocate_raw(&context, input.dtype(), input.layout().num_elements())?;
                let rows64 = rows as u64;
                let width64 = width as u64;
                let parallel = width >= 64;
                let kernel = typed_name(
                    if parallel {
                        "softmax_parallel"
                    } else {
                        "softmax"
                    },
                    input.dtype(),
                );
                if parallel && rows != 0 {
                    launch_configured!(&context, kernel, row_parallel_config("cuda_launch", rows)?;
                        x.buffer.as_ref(), &mut output, &layout.dims, &layout.strides,
                        &layout.rank, &layout.offset, &rows64, &width64
                    )?;
                } else {
                    launch!(&context, kernel, rows;
                        x.buffer.as_ref(), &mut output, &layout.dims, &layout.strides,
                        &layout.rank, &layout.offset, &rows64, &width64
                    )?;
                }
                Ok(vec![Storage::Cuda(finish(
                    &context,
                    input.dtype(),
                    input.layout().num_elements(),
                    output,
                ))])
            }
            FusedOp::LayerNorm => self.fused_layer_norm(inputs, scalars),
            FusedOp::SgdStep => self.fused_sgd(inputs, scalars),
            FusedOp::AdamStep => self.fused_adam(inputs, scalars),
        }
    }
}

fn validate_gather_shape(
    op: &'static str,
    x: View<'_>,
    axis: usize,
    indices: View<'_>,
) -> Result<()> {
    if indices.dtype() != DType::I64 {
        return Err(Error::DTypeMismatch {
            op,
            expected: DType::I64,
            got: indices.dtype(),
        });
    }
    if indices.layout().rank() != x.layout().rank() {
        return Err(Error::RankMismatch {
            op,
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
                op,
                lhs: x.layout().shape().clone(),
                rhs: indices.layout().shape().clone(),
            });
        }
    }
    Ok(())
}

fn validate_scatter_shape(
    x: View<'_>,
    axis: usize,
    indices: View<'_>,
    src: View<'_>,
) -> Result<()> {
    validate_gather_shape("scatter_add", x, axis, indices)?;
    if src.layout().rank() != x.layout().rank() {
        return Err(Error::RankMismatch {
            op: "scatter_add",
            expected: x.layout().rank(),
            got: src.layout().rank(),
        });
    }
    if indices
        .layout()
        .dims()
        .iter()
        .zip(src.layout().dims())
        .any(|(&a, &b)| a > b)
    {
        return Err(Error::ShapeMismatch {
            op: "scatter_add",
            lhs: src.layout().shape().clone(),
            rhs: indices.layout().shape().clone(),
        });
    }
    Ok(())
}

type ConvPlan<'a> = (
    crate::backend::conv_geometry::Conv2dGeometry,
    &'static str,
    usize,
    View<'a>,
    Option<View<'a>>,
    Option<u32>,
);

fn conv_plan<'a>(op: ConvOp, inputs: &[View<'a>], params: &Conv2dParams) -> Result<ConvPlan<'a>> {
    match (op, inputs) {
        (ConvOp::Conv2d, [x, weight]) => {
            let geometry = crate::backend::conv_geometry::Conv2dGeometry::conv2d(
                "conv2d",
                x.layout().dims(),
                weight.layout().dims(),
                params,
            )?;
            let len = checked_product("conv2d", geometry.output_dims())?;
            Ok((geometry, "conv2d", len, *x, Some(*weight), None))
        }
        (ConvOp::MaxPool2d | ConvOp::AvgPool2d, [x]) => {
            let geometry = crate::backend::conv_geometry::Conv2dGeometry::pool(
                "pool2d",
                x.layout().dims(),
                params,
            )?;
            let len = checked_product("pool2d", geometry.output_dims())?;
            Ok((
                geometry,
                "pool",
                len,
                *x,
                None,
                Some(u32::from(matches!(op, ConvOp::AvgPool2d))),
            ))
        }
        (ConvOp::Conv2dInputGrad, [grad, weight, original]) => {
            let geometry = crate::backend::conv_geometry::Conv2dGeometry::conv2d(
                "conv2d_backward_input",
                original.layout().dims(),
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
            Ok((
                geometry,
                "conv_input_grad",
                original.layout().num_elements(),
                *grad,
                Some(*weight),
                None,
            ))
        }
        (ConvOp::Conv2dWeightGrad, [grad, original, weight]) => {
            let geometry = crate::backend::conv_geometry::Conv2dGeometry::conv2d(
                "conv2d_backward_weight",
                original.layout().dims(),
                weight.layout().dims(),
                params,
            )?;
            if grad.layout().dims() != geometry.output_dims() {
                return Err(Error::ShapeMismatch {
                    op: "conv2d_backward_weight",
                    lhs: grad.layout().shape().clone(),
                    rhs: crate::shape::Shape::from(geometry.output_dims()),
                });
            }
            Ok((
                geometry,
                "conv_weight_grad",
                weight.layout().num_elements(),
                *grad,
                Some(*original),
                None,
            ))
        }
        (ConvOp::MaxPool2dBackward | ConvOp::AvgPool2dBackward, [grad, original]) => {
            let geometry = crate::backend::conv_geometry::Conv2dGeometry::pool(
                "pool2d_backward",
                original.layout().dims(),
                params,
            )?;
            if grad.layout().dims() != geometry.output_dims() {
                return Err(Error::ShapeMismatch {
                    op: "pool2d_backward",
                    lhs: grad.layout().shape().clone(),
                    rhs: crate::shape::Shape::from(geometry.output_dims()),
                });
            }
            Ok((
                geometry,
                "pool_backward",
                original.layout().num_elements(),
                *grad,
                Some(*original),
                Some(u32::from(matches!(op, ConvOp::AvgPool2dBackward))),
            ))
        }
        _ => Err(Error::InvalidArg {
            op: "conv",
            msg: format!("invalid {op:?} operand encoding"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static HARDWARE_LANE: Mutex<()> = Mutex::new(());

    #[test]
    fn invalid_ordinal_is_loud() {
        let error = crate::Tensor::zeros([1], DType::F32, &Device::Cuda(usize::MAX)).unwrap_err();
        assert!(matches!(
            error,
            Error::Backend {
                op: "cuda_device",
                ..
            }
        ));
    }

    #[test]
    fn transfer_and_compute_if_device_exists() {
        let _lane = HARDWARE_LANE.lock().unwrap();
        if !is_available(0) {
            return;
        }
        let device = Device::Cuda(0);
        let x = crate::Tensor::from_vec(vec![1.0f32, -2.0, 3.0], [3], &device).unwrap();
        assert_eq!(
            x.relu().unwrap().to_vec::<f32>().unwrap(),
            vec![1.0, 0.0, 3.0]
        );
    }

    #[test]
    fn conformance_if_device_exists() {
        let _lane = HARDWARE_LANE.lock().unwrap();
        if !is_available(0) {
            return;
        }
        let report = crate::backend::conformance::run_device(Device::Cuda(0));
        assert!(
            report.skipped.is_empty(),
            "unexpected unsupported rows: {:?}",
            report.skipped
        );
        report.into_result(Device::Cuda(0)).unwrap();
    }

    #[test]
    fn invalid_indices_are_reported_at_the_next_host_read_if_device_exists() {
        let _lane = HARDWARE_LANE.lock().unwrap();
        if !is_available(0) {
            return;
        }
        let device = Device::Cuda(0);
        let x = crate::Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3], &device).unwrap();
        let indices = crate::Tensor::from_vec(vec![3i64], [1], &device).unwrap();

        x.index_select(0, &indices)
            .expect("index_select must defer device index-value errors");
        assert!(matches!(
            x.to_vec::<f32>(),
            Err(Error::IndexOutOfBounds {
                op: "index_select",
                index: 3,
                axis: 0,
                size: 3,
            })
        ));
        assert_eq!(x.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn deferred_index_errors_follow_program_order_if_device_exists() {
        let _lane = HARDWARE_LANE.lock().unwrap();
        if !is_available(0) {
            return;
        }
        let device = Device::Cuda(0);
        let x = crate::Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3], &device).unwrap();
        let first = crate::Tensor::from_vec(vec![7i64], [1], &device).unwrap();
        let second = crate::Tensor::from_vec(vec![9i64], [1], &device).unwrap();

        x.index_select(0, &first).unwrap();
        x.index_select(0, &second).unwrap();
        assert!(matches!(
            x.to_vec::<f32>(),
            Err(Error::IndexOutOfBounds { index: 7, .. })
        ));
        assert_eq!(x.to_vec::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    }
}
