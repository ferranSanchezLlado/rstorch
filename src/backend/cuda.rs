use super::Backend;
use crate::dtype::{DType, bf16, f16};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, LaunchConfig,
    PushKernelArg, ValidAsZeroBits,
};
use cudarc::nvrtc::Ptx;
use std::error;
use std::fmt;
use std::sync::{Arc, OnceLock};

const PTX_F32: &str = include_str!("kernels/cuda_f32.ptx");
const PTX_F64: &str = include_str!("kernels/cuda_f64.ptx");
const PTX_F16: &str = include_str!("kernels/cuda_f16.ptx");
const PTX_BF16: &str = include_str!("kernels/cuda_bf16.ptx");
const THREADS: u32 = 128;

#[derive(Debug, Clone, Copy, Default)]
pub struct Cuda;

#[derive(Clone)]
pub struct CudaDevice {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module_f32: Arc<CudaModule>,
    module_f64: Arc<CudaModule>,
    module_f16: Arc<CudaModule>,
    module_bf16: Arc<CudaModule>,
    ordinal: usize,
    name: Arc<str>,
}

#[derive(Debug, Clone, Copy)]
enum CudaModuleKind {
    F16,
    BF16,
    F32,
    F64,
}

#[derive(Clone)]
pub struct CudaStorage<E: DeviceRepr> {
    data: CudaSlice<E>,
    len: usize,
}

trait CudaElement: DType + DeviceRepr + ValidAsZeroBits {
    type ScalarParam: DeviceRepr;

    const BINARY_FUNCTION: &'static str;
    const SCALAR_FUNCTION: &'static str;
    const MATMUL_FUNCTION: &'static str;
    const SUM_FUNCTION: &'static str;
    const MODULE: CudaModuleKind;

    fn scalar_param(value: Self) -> Self::ScalarParam;
}

#[derive(Debug, Clone)]
pub enum CudaError {
    Driver(String),
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

impl fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaDevice")
            .field("ordinal", &self.ordinal)
            .field("name", &self.name)
            .finish()
    }
}

impl PartialEq for CudaDevice {
    fn eq(&self, other: &Self) -> bool {
        self.context == other.context && self.ordinal == other.ordinal
    }
}

impl<E: CudaElement> fmt::Debug for CudaStorage<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaStorage")
            .field("dtype", &E::ID)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for CudaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Driver(message) => write!(f, "CUDA driver error: {message}"),
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
            Self::SizeOverflow => write!(f, "CUDA size exceeds u32::MAX"),
        }
    }
}

impl error::Error for CudaError {}

macro_rules! cuda_backend_impl {
    ($element:ty) => {
        fn default_device() -> std::result::Result<Self::Device, Self::Error> {
            static DEVICE: OnceLock<std::result::Result<CudaDevice, CudaError>> = OnceLock::new();
            DEVICE.get_or_init(create_default_device).clone()
        }

        fn zeros(
            device: &Self::Device,
            len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Self::from_vec(device, vec![<$element as DType>::zero(); len])
        }

        fn ones(
            device: &Self::Device,
            len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            Self::from_vec(device, vec![<$element as DType>::one(); len])
        }

        fn from_vec(
            device: &Self::Device,
            data: Vec<$element>,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            let data = device.stream.clone_htod(&data).map_err(cuda_err)?;
            let len = data.len();
            Ok(CudaStorage { data, len })
        }

        fn to_vec(
            device: &Self::Device,
            storage: &Self::Storage,
        ) -> std::result::Result<Vec<$element>, Self::Error> {
            device.stream.clone_dtoh(&storage.data).map_err(cuda_err)
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
            matmul(device, lhs, rhs, m, k, n)
        }

        fn add(
            device: &Self::Device,
            lhs: &Self::Storage,
            rhs: &Self::Storage,
            len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            binary(device, lhs, rhs, len, 0)
        }

        fn sub(
            device: &Self::Device,
            lhs: &Self::Storage,
            rhs: &Self::Storage,
            len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            binary(device, lhs, rhs, len, 1)
        }

        fn mul(
            device: &Self::Device,
            lhs: &Self::Storage,
            rhs: &Self::Storage,
            len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            binary(device, lhs, rhs, len, 2)
        }

        fn div(
            device: &Self::Device,
            lhs: &Self::Storage,
            rhs: &Self::Storage,
            len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            binary(device, lhs, rhs, len, 3)
        }

        fn add_scalar(
            device: &Self::Device,
            input: &Self::Storage,
            rhs: $element,
            len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            scalar(device, input, rhs, len, 0)
        }

        fn sub_scalar(
            device: &Self::Device,
            input: &Self::Storage,
            rhs: $element,
            len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            scalar(device, input, rhs, len, 1)
        }

        fn mul_scalar(
            device: &Self::Device,
            input: &Self::Storage,
            rhs: $element,
            len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            scalar(device, input, rhs, len, 2)
        }

        fn div_scalar(
            device: &Self::Device,
            input: &Self::Storage,
            rhs: $element,
            len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            scalar(device, input, rhs, len, 3)
        }

        fn sum(
            device: &Self::Device,
            input: &Self::Storage,
            len: usize,
        ) -> std::result::Result<Self::Storage, Self::Error> {
            sum(device, input, len)
        }
    };
}

impl Backend<f32> for Cuda {
    type Device = CudaDevice;
    type Storage = CudaStorage<f32>;
    type Error = CudaError;

    cuda_backend_impl!(f32);
}

impl Backend<f64> for Cuda {
    type Device = CudaDevice;
    type Storage = CudaStorage<f64>;
    type Error = CudaError;

    cuda_backend_impl!(f64);
}

impl Backend<f16> for Cuda {
    type Device = CudaDevice;
    type Storage = CudaStorage<f16>;
    type Error = CudaError;

    cuda_backend_impl!(f16);
}

impl Backend<bf16> for Cuda {
    type Device = CudaDevice;
    type Storage = CudaStorage<bf16>;
    type Error = CudaError;

    cuda_backend_impl!(bf16);
}

impl CudaElement for f32 {
    type ScalarParam = f32;

    const BINARY_FUNCTION: &'static str = "binary_f32_kernel";
    const SCALAR_FUNCTION: &'static str = "scalar_f32_kernel";
    const MATMUL_FUNCTION: &'static str = "matmul_f32_kernel";
    const SUM_FUNCTION: &'static str = "sum_f32_kernel";
    const MODULE: CudaModuleKind = CudaModuleKind::F32;

    fn scalar_param(value: Self) -> Self::ScalarParam {
        value
    }
}

impl CudaElement for f64 {
    type ScalarParam = f64;

    const BINARY_FUNCTION: &'static str = "binary_f64_kernel";
    const SCALAR_FUNCTION: &'static str = "scalar_f64_kernel";
    const MATMUL_FUNCTION: &'static str = "matmul_f64_kernel";
    const SUM_FUNCTION: &'static str = "sum_f64_kernel";
    const MODULE: CudaModuleKind = CudaModuleKind::F64;

    fn scalar_param(value: Self) -> Self::ScalarParam {
        value
    }
}

impl CudaElement for f16 {
    type ScalarParam = u16;

    const BINARY_FUNCTION: &'static str = "binary_f16_kernel";
    const SCALAR_FUNCTION: &'static str = "scalar_f16_kernel";
    const MATMUL_FUNCTION: &'static str = "matmul_f16_kernel";
    const SUM_FUNCTION: &'static str = "sum_f16_kernel";
    const MODULE: CudaModuleKind = CudaModuleKind::F16;

    fn scalar_param(value: Self) -> Self::ScalarParam {
        value.to_bits()
    }
}

impl CudaElement for bf16 {
    type ScalarParam = u16;

    const BINARY_FUNCTION: &'static str = "binary_bf16_kernel";
    const SCALAR_FUNCTION: &'static str = "scalar_bf16_kernel";
    const MATMUL_FUNCTION: &'static str = "matmul_bf16_kernel";
    const SUM_FUNCTION: &'static str = "sum_bf16_kernel";
    const MODULE: CudaModuleKind = CudaModuleKind::BF16;

    fn scalar_param(value: Self) -> Self::ScalarParam {
        value.to_bits()
    }
}

fn create_default_device() -> std::result::Result<CudaDevice, CudaError> {
    let context = CudaContext::new(0).map_err(cuda_err)?;
    let stream = context.default_stream();
    let module_f32 = context
        .load_module(Ptx::from_src(PTX_F32))
        .map_err(cuda_err)?;
    let module_f64 = context
        .load_module(Ptx::from_src(PTX_F64))
        .map_err(cuda_err)?;
    let module_f16 = context
        .load_module(Ptx::from_src(PTX_F16))
        .map_err(cuda_err)?;
    let module_bf16 = context
        .load_module(Ptx::from_src(PTX_BF16))
        .map_err(cuda_err)?;
    let ordinal = context.ordinal();
    let name = context.name().map_err(cuda_err)?;
    Ok(CudaDevice {
        context,
        stream,
        module_f32,
        module_f64,
        module_f16,
        module_bf16,
        ordinal,
        name: Arc::from(name),
    })
}

fn matmul<E: CudaElement>(
    device: &CudaDevice,
    lhs: &CudaStorage<E>,
    rhs: &CudaStorage<E>,
    m: usize,
    k: usize,
    n: usize,
) -> std::result::Result<CudaStorage<E>, CudaError> {
    if lhs.len != m.saturating_mul(k) || rhs.len != k.saturating_mul(n) {
        return Err(CudaError::BadMatmulDims {
            m,
            k,
            n,
            lhs_len: lhs.len,
            rhs_len: rhs.len,
        });
    }

    let len = m.saturating_mul(n);
    let mut output = empty_storage(device, len)?;
    if len == 0 {
        return Ok(output);
    }
    let function = function::<E>(device, E::MATMUL_FUNCTION)?;
    let m = checked_u32(m)?;
    let k = checked_u32(k)?;
    let n = checked_u32(n)?;
    unsafe {
        device
            .stream
            .launch_builder(&function)
            .arg(&lhs.data)
            .arg(&rhs.data)
            .arg(&mut output.data)
            .arg(&m)
            .arg(&k)
            .arg(&n)
            .launch(config(len)?)
            .map_err(cuda_err)?;
    }
    device.stream.synchronize().map_err(cuda_err)?;
    Ok(output)
}

fn binary<E: CudaElement>(
    device: &CudaDevice,
    lhs: &CudaStorage<E>,
    rhs: &CudaStorage<E>,
    len: usize,
    op: u32,
) -> std::result::Result<CudaStorage<E>, CudaError> {
    ensure_len(lhs.len, len)?;
    ensure_len(rhs.len, len)?;
    let mut output = empty_storage(device, len)?;
    if len == 0 {
        return Ok(output);
    }
    let function = function::<E>(device, E::BINARY_FUNCTION)?;
    let len = checked_u32(len)?;
    unsafe {
        device
            .stream
            .launch_builder(&function)
            .arg(&lhs.data)
            .arg(&rhs.data)
            .arg(&mut output.data)
            .arg(&op)
            .arg(&len)
            .launch(config(len as usize)?)
            .map_err(cuda_err)?;
    }
    device.stream.synchronize().map_err(cuda_err)?;
    Ok(output)
}

fn scalar<E: CudaElement>(
    device: &CudaDevice,
    input: &CudaStorage<E>,
    rhs: E,
    len: usize,
    op: u32,
) -> std::result::Result<CudaStorage<E>, CudaError> {
    ensure_len(input.len, len)?;
    let mut output = empty_storage(device, len)?;
    if len == 0 {
        return Ok(output);
    }
    let rhs = E::scalar_param(rhs);
    let function = function::<E>(device, E::SCALAR_FUNCTION)?;
    let len = checked_u32(len)?;
    unsafe {
        device
            .stream
            .launch_builder(&function)
            .arg(&input.data)
            .arg(&mut output.data)
            .arg(&rhs)
            .arg(&op)
            .arg(&len)
            .launch(config(len as usize)?)
            .map_err(cuda_err)?;
    }
    device.stream.synchronize().map_err(cuda_err)?;
    Ok(output)
}

fn sum<E: CudaElement>(
    device: &CudaDevice,
    input: &CudaStorage<E>,
    len: usize,
) -> std::result::Result<CudaStorage<E>, CudaError> {
    ensure_len(input.len, len)?;
    if len == 0 {
        let data = device.stream.clone_htod(&[E::zero()]).map_err(cuda_err)?;
        return Ok(CudaStorage { data, len: 1 });
    }
    let mut output = empty_storage(device, 1)?;
    let function = function::<E>(device, E::SUM_FUNCTION)?;
    let len = checked_u32(len)?;
    unsafe {
        device
            .stream
            .launch_builder(&function)
            .arg(&input.data)
            .arg(&mut output.data)
            .arg(&len)
            .launch(LaunchConfig::for_num_elems(1))
            .map_err(cuda_err)?;
    }
    device.stream.synchronize().map_err(cuda_err)?;
    Ok(output)
}

fn empty_storage<E: CudaElement>(
    device: &CudaDevice,
    len: usize,
) -> std::result::Result<CudaStorage<E>, CudaError> {
    let data = device.stream.alloc_zeros::<E>(len).map_err(cuda_err)?;
    Ok(CudaStorage { data, len })
}

fn function<E: CudaElement>(
    device: &CudaDevice,
    name: &'static str,
) -> std::result::Result<CudaFunction, CudaError> {
    let module = match E::MODULE {
        CudaModuleKind::F16 => &device.module_f16,
        CudaModuleKind::BF16 => &device.module_bf16,
        CudaModuleKind::F32 => &device.module_f32,
        CudaModuleKind::F64 => &device.module_f64,
    };
    module.load_function(name).map_err(cuda_err)
}

fn config(len: usize) -> std::result::Result<LaunchConfig, CudaError> {
    let len = checked_u32(len)?;
    Ok(LaunchConfig {
        grid_dim: (len.div_ceil(THREADS).max(1), 1, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn ensure_len(found: usize, expected: usize) -> std::result::Result<(), CudaError> {
    if found != expected {
        return Err(CudaError::LengthMismatch { expected, found });
    }
    Ok(())
}

fn checked_u32(value: usize) -> std::result::Result<u32, CudaError> {
    u32::try_from(value).map_err(|_| CudaError::SizeOverflow)
}

fn cuda_err(err: impl fmt::Debug) -> CudaError {
    CudaError::Driver(format!("{err:?}"))
}

#[cfg(test)]
mod tests {
    use super::{PTX_BF16, PTX_F16, PTX_F32, PTX_F64};

    #[test]
    fn bundled_ptx_contains_required_entrypoints() {
        assert!(PTX_F32.contains(".entry binary_f32_kernel"));
        assert!(PTX_F32.contains(".entry scalar_f32_kernel"));
        assert!(PTX_F32.contains(".entry matmul_f32_kernel"));
        assert!(PTX_F32.contains(".entry sum_f32_kernel"));
        assert!(PTX_F64.contains(".entry binary_f64_kernel"));
        assert!(PTX_F64.contains(".entry scalar_f64_kernel"));
        assert!(PTX_F64.contains(".entry matmul_f64_kernel"));
        assert!(PTX_F64.contains(".entry sum_f64_kernel"));
        assert!(PTX_F16.contains(".entry binary_f16_kernel"));
        assert!(PTX_F16.contains(".entry scalar_f16_kernel"));
        assert!(PTX_F16.contains(".entry matmul_f16_kernel"));
        assert!(PTX_F16.contains(".entry sum_f16_kernel"));
        assert!(PTX_BF16.contains(".entry binary_bf16_kernel"));
        assert!(PTX_BF16.contains(".entry scalar_bf16_kernel"));
        assert!(PTX_BF16.contains(".entry matmul_bf16_kernel"));
        assert!(PTX_BF16.contains(".entry sum_bf16_kernel"));
    }
}
