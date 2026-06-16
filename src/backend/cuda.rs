//! CUDA backend for Linux and Windows.
//!
//! This module keeps CUDA context, stream, buffer, and kernel launch code out of
//! tensor/autograd layers. CUDA support is feature-gated and only has a real
//! `Backend` implementation on CUDA-supported desktop targets where `cudarc` is
//! available.

/// CUDA backend marker.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Cuda;

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
impl Cuda {
    /// Returns whether CUDA execution is available on this build target.
    pub fn is_available() -> bool {
        false
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
mod real {
    use super::Cuda;
    use crate::backend::Backend;
    use cudarc::driver::{
        CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
    };
    use cudarc::nvrtc::compile_ptx;
    use std::marker::PhantomData;
    use std::sync::{Arc, OnceLock};

    const KERNELS: &str = include_str!("cuda/kernels.cu");

    /// Single CUDA context, default stream, and kernel module used by tensors.
    ///
    /// Multi-device execution is intentionally not implemented in this epoch.
    #[derive(Clone)]
    pub struct CudaDevice {
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        module: Arc<CudaModule>,
        ordinal: usize,
    }

    /// CUDA device-buffer-backed tensor storage.
    #[derive(Clone, Debug)]
    pub struct CudaStorage<E = f32> {
        data: CudaSlice<f32>,
        len: usize,
        device_ordinal: usize,
        stream: Arc<CudaStream>,
        dtype: PhantomData<E>,
    }

    impl Cuda {
        /// Returns whether at least one CUDA device is visible to the driver.
        pub fn is_available() -> bool {
            CudaContext::device_count()
                .map(|device_count| device_count > 0)
                .unwrap_or(false)
        }
    }

    impl CudaDevice {
        /// Creates a device wrapper for a CUDA ordinal.
        pub fn new(ordinal: usize) -> Option<Self> {
            let context = CudaContext::new(ordinal).ok()?;
            let stream = context.default_stream();
            let ptx = compile_ptx(KERNELS).ok()?;
            let module = context.load_module(ptx).ok()?;

            Some(Self {
                context,
                stream,
                module,
                ordinal,
            })
        }

        pub fn ordinal(&self) -> usize {
            self.ordinal
        }
    }

    impl Backend<f32> for Cuda {
        type Device = CudaDevice;
        type Storage = CudaStorage<f32>;

        fn default_device() -> Self::Device {
            static DEFAULT_DEVICE: OnceLock<CudaDevice> = OnceLock::new();

            DEFAULT_DEVICE
                .get_or_init(|| {
                    CudaDevice::new(0)
                        .expect("CUDA backend requested, but device 0 or NVRTC is unavailable")
                })
                .clone()
        }

        fn zeros(device: &Self::Device, len: usize) -> Self::Storage {
            empty_storage(device, len)
        }

        fn ones(device: &Self::Device, len: usize) -> Self::Storage {
            Self::from_vec(device, vec![1.0; len])
        }

        fn from_array<const N: usize>(device: &Self::Device, data: [f32; N]) -> Self::Storage
        where
            [(); N]:,
        {
            Self::from_vec(device, Vec::from(data))
        }

        fn from_vec(device: &Self::Device, data: Vec<f32>) -> Self::Storage {
            upload(device, &data)
        }

        fn to_vec(storage: &Self::Storage) -> Vec<f32> {
            storage
                .stream
                .clone_dtoh(&storage.data)
                .expect("failed to copy CUDA storage to host")
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
            assert_eq!(lhs.len, rows * inner, "CUDA matmul lhs length mismatch");
            assert_eq!(rhs.len, inner * cols, "CUDA matmul rhs length mismatch");

            let output_len = rows * cols;
            let mut output = empty_storage(device, output_len);
            let rows = checked_u32(rows, "matmul rows");
            let inner = checked_u32(inner, "matmul inner");
            let cols = checked_u32(cols, "matmul cols");
            let function = function(device, "matmul_f32");
            let config =
                LaunchConfig::for_num_elems(checked_u32(output_len.max(1), "matmul length"));

            unsafe {
                // SAFETY: Kernel name and argument order are fixed in this module.
                // Buffer lengths are checked above and the kernel bounds-checks each index.
                device
                    .stream
                    .launch_builder(&function)
                    .arg(&lhs.data)
                    .arg(&rhs.data)
                    .arg(&mut output.data)
                    .arg(&rows)
                    .arg(&inner)
                    .arg(&cols)
                    .launch(config)
                    .expect("failed to launch CUDA matmul kernel");
            }
            synchronize(device);

            output
        }

        fn transpose(
            device: &Self::Device,
            input: &Self::Storage,
            rows: usize,
            cols: usize,
        ) -> Self::Storage {
            assert_storage_on_device(device, input);
            assert_eq!(input.len, rows * cols, "CUDA transpose length mismatch");

            let mut output = empty_storage(device, input.len);
            let rows = checked_u32(rows, "transpose rows");
            let cols = checked_u32(cols, "transpose cols");
            let len = checked_u32(input.len, "transpose length");
            let function = function(device, "transpose_f32");
            let config = LaunchConfig::for_num_elems(len.max(1));

            unsafe {
                // SAFETY: Kernel name and argument order are fixed in this module.
                // `input.len == rows * cols` is checked above and the kernel bounds-checks.
                device
                    .stream
                    .launch_builder(&function)
                    .arg(&input.data)
                    .arg(&mut output.data)
                    .arg(&rows)
                    .arg(&cols)
                    .arg(&len)
                    .launch(config)
                    .expect("failed to launch CUDA transpose kernel");
            }
            synchronize(device);

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

    fn upload(device: &CudaDevice, data: &[f32]) -> CudaStorage<f32> {
        let data = device
            .stream
            .clone_htod(data)
            .expect("failed to copy host data to CUDA storage");

        CudaStorage {
            len: data.len(),
            data,
            device_ordinal: device.ordinal,
            stream: Arc::clone(&device.stream),
            dtype: PhantomData,
        }
    }

    fn empty_storage(device: &CudaDevice, len: usize) -> CudaStorage<f32> {
        let data = device
            .stream
            .alloc_zeros::<f32>(len)
            .expect("failed to allocate CUDA storage");

        CudaStorage {
            data,
            len,
            device_ordinal: device.ordinal,
            stream: Arc::clone(&device.stream),
            dtype: PhantomData,
        }
    }

    fn binary_kernel(
        device: &CudaDevice,
        lhs: &CudaStorage<f32>,
        rhs: &CudaStorage<f32>,
        name: &str,
    ) -> CudaStorage<f32> {
        assert_storage_on_device(device, lhs);
        assert_storage_on_device(device, rhs);
        assert_eq!(lhs.len, rhs.len, "CUDA elementwise length mismatch");

        let mut output = empty_storage(device, lhs.len);
        let len = checked_u32(lhs.len, "elementwise length");
        let function = function(device, name);
        let config = LaunchConfig::for_num_elems(len.max(1));

        unsafe {
            // SAFETY: Kernel name and argument order are fixed by the caller.
            // Input lengths match and kernels bounds-check against `len`.
            device
                .stream
                .launch_builder(&function)
                .arg(&lhs.data)
                .arg(&rhs.data)
                .arg(&mut output.data)
                .arg(&len)
                .launch(config)
                .expect("failed to launch CUDA binary kernel");
        }
        synchronize(device);

        output
    }

    fn unary_kernel(device: &CudaDevice, input: &CudaStorage<f32>, name: &str) -> CudaStorage<f32> {
        assert_storage_on_device(device, input);

        let mut output = empty_storage(device, input.len);
        let len = checked_u32(input.len, "unary length");
        let function = function(device, name);
        let config = LaunchConfig::for_num_elems(len.max(1));

        unsafe {
            // SAFETY: Kernel name and argument order are fixed by the caller.
            // The kernel writes exactly one checked output element per input element.
            device
                .stream
                .launch_builder(&function)
                .arg(&input.data)
                .arg(&mut output.data)
                .arg(&len)
                .launch(config)
                .expect("failed to launch CUDA unary kernel");
        }
        synchronize(device);

        output
    }

    fn scalar_kernel(
        device: &CudaDevice,
        input: &CudaStorage<f32>,
        rhs: f32,
        name: &str,
    ) -> CudaStorage<f32> {
        assert_storage_on_device(device, input);

        let mut output = empty_storage(device, input.len);
        let len = checked_u32(input.len, "scalar length");
        let function = function(device, name);
        let config = LaunchConfig::for_num_elems(len.max(1));

        unsafe {
            // SAFETY: Kernel name and argument order are fixed by the caller.
            // The kernel bounds-checks each output write against `len`.
            device
                .stream
                .launch_builder(&function)
                .arg(&input.data)
                .arg(&mut output.data)
                .arg(&rhs)
                .arg(&len)
                .launch(config)
                .expect("failed to launch CUDA scalar kernel");
        }
        synchronize(device);

        output
    }

    fn reduction_kernel(
        device: &CudaDevice,
        input: &CudaStorage<f32>,
        name: &str,
    ) -> CudaStorage<f32> {
        assert_storage_on_device(device, input);

        let mut output = empty_storage(device, 1);
        let len = checked_u32(input.len, "reduction length");
        let function = function(device, name);
        let config = LaunchConfig::for_num_elems(1);

        unsafe {
            // SAFETY: Kernel name and argument order are fixed by the caller.
            // Reduction kernels are single-threaded MVP kernels and write one scalar.
            device
                .stream
                .launch_builder(&function)
                .arg(&input.data)
                .arg(&mut output.data)
                .arg(&len)
                .launch(config)
                .expect("failed to launch CUDA reduction kernel");
        }
        synchronize(device);

        output
    }

    fn add_vector_kernel(
        device: &CudaDevice,
        input: &CudaStorage<f32>,
        vector: &CudaStorage<f32>,
        rows: usize,
        cols: usize,
        name: &str,
    ) -> CudaStorage<f32> {
        assert_storage_on_device(device, input);
        assert_storage_on_device(device, vector);
        assert_eq!(
            input.len,
            rows * cols,
            "CUDA broadcast input length mismatch"
        );

        let expected_vector_len = if name == "add_row_f32" { cols } else { rows };
        assert_eq!(
            vector.len, expected_vector_len,
            "CUDA broadcast vector length mismatch"
        );

        let mut output = empty_storage(device, input.len);
        let rows = checked_u32(rows, "broadcast rows");
        let cols = checked_u32(cols, "broadcast cols");
        let len = checked_u32(input.len, "broadcast length");
        let function = function(device, name);
        let config = LaunchConfig::for_num_elems(len.max(1));

        unsafe {
            // SAFETY: Kernel name and argument order are fixed by the caller.
            // Matrix/vector lengths are checked above and kernels bounds-check by index.
            device
                .stream
                .launch_builder(&function)
                .arg(&input.data)
                .arg(&vector.data)
                .arg(&mut output.data)
                .arg(&rows)
                .arg(&cols)
                .arg(&len)
                .launch(config)
                .expect("failed to launch CUDA broadcast kernel");
        }
        synchronize(device);

        output
    }

    fn function(device: &CudaDevice, name: &str) -> CudaFunction {
        device
            .module
            .load_function(name)
            .unwrap_or_else(|error| panic!("failed to load CUDA kernel `{name}`: {error}"))
    }

    fn synchronize(device: &CudaDevice) {
        device
            .stream
            .synchronize()
            .expect("failed to synchronize CUDA stream");
        device
            .context
            .check_err()
            .expect("CUDA context reported an asynchronous error");
    }

    fn assert_storage_on_device(device: &CudaDevice, storage: &CudaStorage<f32>) {
        assert_eq!(
            device.ordinal, storage.device_ordinal,
            "CUDA tensors from different devices cannot be used together"
        );
    }

    fn checked_u32(value: usize, name: &str) -> u32 {
        value
            .try_into()
            .unwrap_or_else(|_| panic!("CUDA {name} exceeds u32::MAX"))
    }
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
pub use real::{CudaDevice, CudaStorage};
