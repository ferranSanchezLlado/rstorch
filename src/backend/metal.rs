use super::{Backend, NativeBinaryOp, NativeRowOp, NativeUnaryOp, sealed};
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
    const UNARY_KERNEL: &'static str;
    const ROW_SOFTMAX_KERNEL: &'static str;
    const SUM_LAST_KERNEL: &'static str;
    const BMM_KERNEL: &'static str;
    const STRIDED_MATMUL_KERNEL: &'static str;
    const BROADCAST_KERNEL: &'static str;
    const MASK_KERNEL: &'static str;
    const INDEX_SELECT_ROWS_KERNEL: &'static str;
    const CROSS_ENTROPY_KERNEL: &'static str;
    const LAYER_NORM_KERNEL: &'static str;
    const RMS_NORM_KERNEL: &'static str;
    const SGD_STEP_KERNEL: &'static str;
    const ADAM_STEP_KERNEL: &'static str;
}

impl MetalDType for f32 {
    const SHADERS: &'static str = SHADERS_F32;
    const BINARY_KERNEL: &'static str = "binary_f32_kernel";
    const SCALAR_KERNEL: &'static str = "scalar_f32_kernel";
    const MATMUL_KERNEL: &'static str = "matmul_f32_kernel";
    const SUM_KERNEL: &'static str = "sum_f32_kernel";
    const UNARY_KERNEL: &'static str = "unary_f32_kernel";
    const ROW_SOFTMAX_KERNEL: &'static str = "row_softmax_f32_kernel";
    const SUM_LAST_KERNEL: &'static str = "sum_last_f32_kernel";
    const BMM_KERNEL: &'static str = "bmm_f32_kernel";
    const STRIDED_MATMUL_KERNEL: &'static str = "strided_matmul_f32_kernel";
    const BROADCAST_KERNEL: &'static str = "broadcast_f32_kernel";
    const MASK_KERNEL: &'static str = "mask_f32_kernel";
    const INDEX_SELECT_ROWS_KERNEL: &'static str = "index_select_rows_f32_kernel";
    const CROSS_ENTROPY_KERNEL: &'static str = "cross_entropy_f32_kernel";
    const LAYER_NORM_KERNEL: &'static str = "layer_norm_f32_kernel";
    const RMS_NORM_KERNEL: &'static str = "rms_norm_f32_kernel";
    const SGD_STEP_KERNEL: &'static str = "sgd_step_f32_kernel";
    const ADAM_STEP_KERNEL: &'static str = "adam_step_f32_kernel";
}

impl MetalDType for f16 {
    const SHADERS: &'static str = SHADERS_F16;
    const BINARY_KERNEL: &'static str = "binary_f16_kernel";
    const SCALAR_KERNEL: &'static str = "scalar_f16_kernel";
    const MATMUL_KERNEL: &'static str = "matmul_f16_kernel";
    const SUM_KERNEL: &'static str = "sum_f16_kernel";
    const UNARY_KERNEL: &'static str = "unary_f16_kernel";
    const ROW_SOFTMAX_KERNEL: &'static str = "row_softmax_f16_kernel";
    const SUM_LAST_KERNEL: &'static str = "sum_last_f16_kernel";
    const BMM_KERNEL: &'static str = "bmm_f16_kernel";
    const STRIDED_MATMUL_KERNEL: &'static str = "strided_matmul_f16_kernel";
    const BROADCAST_KERNEL: &'static str = "broadcast_f16_kernel";
    const MASK_KERNEL: &'static str = "mask_f16_kernel";
    const INDEX_SELECT_ROWS_KERNEL: &'static str = "index_select_rows_f16_kernel";
    const CROSS_ENTROPY_KERNEL: &'static str = "cross_entropy_f16_kernel";
    const LAYER_NORM_KERNEL: &'static str = "layer_norm_f16_kernel";
    const RMS_NORM_KERNEL: &'static str = "rms_norm_f16_kernel";
    const SGD_STEP_KERNEL: &'static str = "sgd_step_f16_kernel";
    const ADAM_STEP_KERNEL: &'static str = "adam_step_f16_kernel";
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

    fn try_unary(
        device: &Self::Device,
        input: &Self::Storage,
        len: usize,
        op: NativeUnaryOp,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        ensure_len(input.len, len)?;
        let output = empty_storage::<E>(device, len);
        if len == 0 {
            return Ok(Some(output));
        }

        let pipeline = pipeline::<E>(device, E::UNARY_KERNEL)?;
        let len_u32 = checked_u32(len, "unary length")?;
        encode_and_wait(device, &pipeline, len, |encoder| {
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&output.buffer), 0);
            set_u32(encoder, 2, native_unary_op(op));
            set_u32(encoder, 3, len_u32);
        })?;
        Ok(Some(output))
    }

    fn try_row_softmax(
        device: &Self::Device,
        input: &Self::Storage,
        rows: usize,
        cols: usize,
        op: NativeRowOp,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        let len = rows.saturating_mul(cols);
        ensure_len(input.len, len)?;
        let output = empty_storage::<E>(device, len);
        if len == 0 {
            return Ok(Some(output));
        }

        let pipeline = pipeline::<E>(device, E::ROW_SOFTMAX_KERNEL)?;
        let rows_u32 = checked_u32(rows, "row count")?;
        let cols_u32 = checked_u32(cols, "column count")?;
        encode_and_wait(device, &pipeline, rows, |encoder| {
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&output.buffer), 0);
            set_u32(encoder, 2, rows_u32);
            set_u32(encoder, 3, cols_u32);
            set_u32(encoder, 4, native_row_op(op));
        })?;
        Ok(Some(output))
    }

    fn try_sum_last(
        device: &Self::Device,
        input: &Self::Storage,
        rows: usize,
        cols: usize,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        ensure_len(input.len, rows.saturating_mul(cols))?;
        let output = empty_storage::<E>(device, rows);
        if rows == 0 {
            return Ok(Some(output));
        }

        let pipeline = pipeline::<E>(device, E::SUM_LAST_KERNEL)?;
        let rows_u32 = checked_u32(rows, "row count")?;
        let cols_u32 = checked_u32(cols, "column count")?;
        encode_and_wait(device, &pipeline, rows, |encoder| {
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&output.buffer), 0);
            set_u32(encoder, 2, rows_u32);
            set_u32(encoder, 3, cols_u32);
        })?;
        Ok(Some(output))
    }

    fn try_bmm(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        ensure_len(lhs.len, batch.saturating_mul(m).saturating_mul(k))?;
        ensure_len(rhs.len, batch.saturating_mul(k).saturating_mul(n))?;
        let len = batch.saturating_mul(m).saturating_mul(n);
        let output = empty_storage::<E>(device, len);
        if len == 0 {
            return Ok(Some(output));
        }

        let pipeline = pipeline::<E>(device, E::BMM_KERNEL)?;
        let batch_u32 = checked_u32(batch, "batch count")?;
        let m_u32 = checked_u32(m, "m")?;
        let k_u32 = checked_u32(k, "k")?;
        let n_u32 = checked_u32(n, "n")?;
        encode_and_wait(device, &pipeline, len, |encoder| {
            encoder.set_buffer(0, Some(&lhs.buffer), 0);
            encoder.set_buffer(1, Some(&rhs.buffer), 0);
            encoder.set_buffer(2, Some(&output.buffer), 0);
            set_u32(encoder, 3, batch_u32);
            set_u32(encoder, 4, m_u32);
            set_u32(encoder, 5, k_u32);
            set_u32(encoder, 6, n_u32);
        })?;
        Ok(Some(output))
    }

    fn try_strided_matmul(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        m: usize,
        k: usize,
        n: usize,
        lhs_offset: usize,
        lhs_row_stride: usize,
        lhs_col_stride: usize,
        rhs_offset: usize,
        rhs_row_stride: usize,
        rhs_col_stride: usize,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        let len = m.saturating_mul(n);
        let output = empty_storage::<E>(device, len);
        if len == 0 {
            return Ok(Some(output));
        }

        let pipeline = pipeline::<E>(device, E::STRIDED_MATMUL_KERNEL)?;
        encode_and_wait(device, &pipeline, len, |encoder| {
            encoder.set_buffer(0, Some(&lhs.buffer), 0);
            encoder.set_buffer(1, Some(&rhs.buffer), 0);
            encoder.set_buffer(2, Some(&output.buffer), 0);
            set_u32(
                encoder,
                3,
                checked_u32(m, "m").expect("checked before dispatch"),
            );
            set_u32(
                encoder,
                4,
                checked_u32(k, "k").expect("checked before dispatch"),
            );
            set_u32(
                encoder,
                5,
                checked_u32(n, "n").expect("checked before dispatch"),
            );
            set_u32(
                encoder,
                6,
                checked_u32(lhs_offset, "lhs offset").expect("checked before dispatch"),
            );
            set_u32(
                encoder,
                7,
                checked_u32(lhs_row_stride, "lhs row stride").expect("checked before dispatch"),
            );
            set_u32(
                encoder,
                8,
                checked_u32(lhs_col_stride, "lhs col stride").expect("checked before dispatch"),
            );
            set_u32(
                encoder,
                9,
                checked_u32(rhs_offset, "rhs offset").expect("checked before dispatch"),
            );
            set_u32(
                encoder,
                10,
                checked_u32(rhs_row_stride, "rhs row stride").expect("checked before dispatch"),
            );
            set_u32(
                encoder,
                11,
                checked_u32(rhs_col_stride, "rhs col stride").expect("checked before dispatch"),
            );
        })?;
        Ok(Some(output))
    }

    fn try_broadcast_last(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        rows: usize,
        cols: usize,
        op: NativeBinaryOp,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        broadcast::<E>(device, lhs, rhs, rows.saturating_mul(cols), cols, 0, op).map(Some)
    }

    fn try_broadcast_leading(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        leading: usize,
        inner: usize,
        op: NativeBinaryOp,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        broadcast::<E>(
            device,
            lhs,
            rhs,
            leading.saturating_mul(inner),
            inner,
            1,
            op,
        )
        .map(Some)
    }

    fn try_broadcast_channel(
        device: &Self::Device,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        op: NativeBinaryOp,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        let total = batch
            .saturating_mul(channels)
            .saturating_mul(height)
            .saturating_mul(width);
        broadcast::<E>(device, lhs, rhs, total, height.saturating_mul(width), 2, op).map(Some)
    }

    fn try_masked_fill(
        device: &Self::Device,
        input: &Self::Storage,
        mask: &[bool],
        value: E,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        ensure_len(input.len, mask.len())?;
        let output = empty_storage::<E>(device, input.len);
        if input.len == 0 {
            return Ok(Some(output));
        }
        let mask_storage = u8_storage(device, mask.iter().map(|&value| u8::from(value)).collect());
        let pipeline = pipeline::<E>(device, E::MASK_KERNEL)?;
        let len_u32 = checked_u32(input.len, "mask length")?;
        encode_and_wait(device, &pipeline, input.len, |encoder| {
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&mask_storage.buffer), 0);
            encoder.set_buffer(2, Some(&output.buffer), 0);
            encoder.set_buffer(3, None, 0);
            set_value(encoder, 4, value);
            set_u32(encoder, 5, 0);
            set_u32(encoder, 6, len_u32);
        })?;
        Ok(Some(output))
    }

    fn try_where_mask(
        device: &Self::Device,
        lhs: &Self::Storage,
        mask: &[bool],
        rhs: &Self::Storage,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        ensure_len(lhs.len, mask.len())?;
        ensure_len(rhs.len, mask.len())?;
        let output = empty_storage::<E>(device, lhs.len);
        if lhs.len == 0 {
            return Ok(Some(output));
        }
        let mask_storage = u8_storage(device, mask.iter().map(|&value| u8::from(value)).collect());
        let pipeline = pipeline::<E>(device, E::MASK_KERNEL)?;
        let len_u32 = checked_u32(lhs.len, "mask length")?;
        encode_and_wait(device, &pipeline, lhs.len, |encoder| {
            encoder.set_buffer(0, Some(&lhs.buffer), 0);
            encoder.set_buffer(1, Some(&mask_storage.buffer), 0);
            encoder.set_buffer(2, Some(&output.buffer), 0);
            encoder.set_buffer(3, Some(&rhs.buffer), 0);
            set_value(encoder, 4, E::ZERO);
            set_u32(encoder, 5, 1);
            set_u32(encoder, 6, len_u32);
        })?;
        Ok(Some(output))
    }

    fn try_index_select_rows(
        device: &Self::Device,
        input: &Self::Storage,
        indices: &[usize],
        rows: usize,
        cols: usize,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        ensure_len(input.len, rows.saturating_mul(cols))?;
        let output = empty_storage::<E>(device, indices.len().saturating_mul(cols));
        if indices.is_empty() || cols == 0 {
            return Ok(Some(output));
        }
        let index_storage =
            u32_storage(device, checked_u32_vec(indices, "index_select_rows index")?);
        let pipeline = pipeline::<E>(device, E::INDEX_SELECT_ROWS_KERNEL)?;
        let len = indices.len().saturating_mul(cols);
        let cols_u32 = checked_u32(cols, "column count")?;
        let total_u32 = checked_u32(len, "index_select_rows output length")?;
        encode_and_wait(device, &pipeline, len, |encoder| {
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&index_storage.buffer), 0);
            encoder.set_buffer(2, Some(&output.buffer), 0);
            set_u32(encoder, 3, cols_u32);
            set_u32(encoder, 4, total_u32);
        })?;
        Ok(Some(output))
    }

    fn try_cross_entropy(
        device: &Self::Device,
        logits: &Self::Storage,
        targets: &[usize],
        rows: usize,
        cols: usize,
        ignore_index: Option<usize>,
        label_smoothing: f64,
        mean_reduction: bool,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        ensure_len(logits.len, rows.saturating_mul(cols))?;
        let output = empty_storage::<E>(device, 1);
        let mut target_values = checked_u32_vec(targets, "cross_entropy target")?;
        let ignore = match ignore_index {
            Some(value) => checked_u32(value, "ignore_index")?,
            None => u32::MAX,
        };
        target_values.resize(rows, ignore);
        let target_storage = u32_storage(device, target_values);
        let pipeline = pipeline::<E>(device, E::CROSS_ENTROPY_KERNEL)?;
        encode_and_wait(device, &pipeline, 1, |encoder| {
            encoder.set_buffer(0, Some(&logits.buffer), 0);
            encoder.set_buffer(1, Some(&target_storage.buffer), 0);
            encoder.set_buffer(2, Some(&output.buffer), 0);
            set_u32(
                encoder,
                3,
                checked_u32(rows, "row count").expect("checked before dispatch"),
            );
            set_u32(
                encoder,
                4,
                checked_u32(cols, "column count").expect("checked before dispatch"),
            );
            set_u32(encoder, 5, ignore);
            set_f32(encoder, 6, label_smoothing as f32);
            set_u32(encoder, 7, u32::from(mean_reduction));
        })?;
        Ok(Some(output))
    }

    fn try_layer_norm(
        device: &Self::Device,
        input: &Self::Storage,
        weight: &Self::Storage,
        bias: &Self::Storage,
        rows: usize,
        cols: usize,
        eps: f64,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        ensure_len(input.len, rows.saturating_mul(cols))?;
        ensure_len(weight.len, cols)?;
        ensure_len(bias.len, cols)?;
        let output = empty_storage::<E>(device, input.len);
        if input.len == 0 {
            return Ok(Some(output));
        }
        let pipeline = pipeline::<E>(device, E::LAYER_NORM_KERNEL)?;
        encode_and_wait(device, &pipeline, rows, |encoder| {
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&weight.buffer), 0);
            encoder.set_buffer(2, Some(&bias.buffer), 0);
            encoder.set_buffer(3, Some(&output.buffer), 0);
            set_u32(
                encoder,
                4,
                checked_u32(rows, "row count").expect("checked before dispatch"),
            );
            set_u32(
                encoder,
                5,
                checked_u32(cols, "column count").expect("checked before dispatch"),
            );
            set_f32(encoder, 6, eps as f32);
        })?;
        Ok(Some(output))
    }

    fn try_rms_norm(
        device: &Self::Device,
        input: &Self::Storage,
        weight: &Self::Storage,
        rows: usize,
        cols: usize,
        eps: f64,
    ) -> std::result::Result<Option<Self::Storage>, Self::Error> {
        ensure_len(input.len, rows.saturating_mul(cols))?;
        ensure_len(weight.len, cols)?;
        let output = empty_storage::<E>(device, input.len);
        if input.len == 0 {
            return Ok(Some(output));
        }
        let pipeline = pipeline::<E>(device, E::RMS_NORM_KERNEL)?;
        encode_and_wait(device, &pipeline, rows, |encoder| {
            encoder.set_buffer(0, Some(&input.buffer), 0);
            encoder.set_buffer(1, Some(&weight.buffer), 0);
            encoder.set_buffer(2, Some(&output.buffer), 0);
            set_u32(
                encoder,
                3,
                checked_u32(rows, "row count").expect("checked before dispatch"),
            );
            set_u32(
                encoder,
                4,
                checked_u32(cols, "column count").expect("checked before dispatch"),
            );
            set_f32(encoder, 5, eps as f32);
        })?;
        Ok(Some(output))
    }

    #[allow(clippy::too_many_arguments)]
    fn sgd_step(
        device: &Self::Device,
        param: &Self::Storage,
        grad: &Self::Storage,
        velocity: Option<&Self::Storage>,
        len: usize,
        lr: E,
        momentum: Option<E>,
        weight_decay: E,
    ) -> std::result::Result<(Self::Storage, Option<Self::Storage>), Self::Error>
    where
        E: crate::dtype::FloatDType,
    {
        ensure_len(param.len, len)?;
        ensure_len(grad.len, len)?;
        if let Some(velocity) = velocity {
            ensure_len(velocity.len, len)?;
        }
        let output = empty_storage::<E>(device, len);
        let velocity_output = momentum.map(|_| empty_storage::<E>(device, len));
        if len == 0 {
            return Ok((output, velocity_output));
        }

        let pipeline = pipeline::<E>(device, E::SGD_STEP_KERNEL)?;
        let len_u32 = checked_u32(len, "sgd step length")?;
        let use_momentum = u32::from(momentum.is_some());
        encode_and_wait(device, &pipeline, len, |encoder| {
            encoder.set_buffer(0, Some(&param.buffer), 0);
            encoder.set_buffer(1, Some(&grad.buffer), 0);
            encoder.set_buffer(2, velocity.map(buffer_ref), 0);
            encoder.set_buffer(3, Some(&output.buffer), 0);
            encoder.set_buffer(4, velocity_output.as_ref().map(buffer_ref), 0);
            set_value(encoder, 5, lr);
            set_value(encoder, 6, momentum.unwrap_or(E::ZERO));
            set_value(encoder, 7, weight_decay);
            set_u32(encoder, 8, use_momentum);
            set_u32(encoder, 9, len_u32);
        })?;
        Ok((output, velocity_output))
    }

    #[allow(clippy::too_many_arguments)]
    fn adam_step(
        device: &Self::Device,
        param: &Self::Storage,
        grad: &Self::Storage,
        m: Option<&Self::Storage>,
        v: Option<&Self::Storage>,
        len: usize,
        lr: E,
        beta1: E,
        beta2: E,
        eps: E,
        weight_decay: E,
        beta1_pow: E,
        beta2_pow: E,
    ) -> std::result::Result<(Self::Storage, Self::Storage, Self::Storage), Self::Error>
    where
        E: crate::dtype::FloatDType,
    {
        ensure_len(param.len, len)?;
        ensure_len(grad.len, len)?;
        if let Some(m) = m {
            ensure_len(m.len, len)?;
        }
        if let Some(v) = v {
            ensure_len(v.len, len)?;
        }
        let output = empty_storage::<E>(device, len);
        let m_output = empty_storage::<E>(device, len);
        let v_output = empty_storage::<E>(device, len);
        if len == 0 {
            return Ok((output, m_output, v_output));
        }

        let pipeline = pipeline::<E>(device, E::ADAM_STEP_KERNEL)?;
        let len_u32 = checked_u32(len, "adam step length")?;
        encode_and_wait(device, &pipeline, len, |encoder| {
            encoder.set_buffer(0, Some(&param.buffer), 0);
            encoder.set_buffer(1, Some(&grad.buffer), 0);
            encoder.set_buffer(2, m.map(buffer_ref), 0);
            encoder.set_buffer(3, v.map(buffer_ref), 0);
            encoder.set_buffer(4, Some(&output.buffer), 0);
            encoder.set_buffer(5, Some(&m_output.buffer), 0);
            encoder.set_buffer(6, Some(&v_output.buffer), 0);
            set_value(encoder, 7, lr);
            set_value(encoder, 8, beta1);
            set_value(encoder, 9, beta2);
            set_value(encoder, 10, eps);
            set_value(encoder, 11, weight_decay);
            set_value(encoder, 12, beta1_pow);
            set_value(encoder, 13, beta2_pow);
            set_u32(encoder, 14, u32::from(m.is_some() && v.is_some()));
            set_u32(encoder, 15, len_u32);
        })?;
        Ok((output, m_output, v_output))
    }
}

fn native_unary_op(op: NativeUnaryOp) -> u32 {
    match op {
        NativeUnaryOp::Relu => 0,
        NativeUnaryOp::Neg => 1,
        NativeUnaryOp::Exp => 2,
        NativeUnaryOp::Ln => 3,
        NativeUnaryOp::Tanh => 4,
        NativeUnaryOp::Sigmoid => 5,
        NativeUnaryOp::Sqrt => 6,
        NativeUnaryOp::Abs => 7,
        NativeUnaryOp::Gelu => 8,
    }
}

fn native_row_op(op: NativeRowOp) -> u32 {
    match op {
        NativeRowOp::Softmax => 0,
        NativeRowOp::LogSoftmax => 1,
    }
}

fn native_binary_op(op: NativeBinaryOp) -> u32 {
    match op {
        NativeBinaryOp::Add => 0,
        NativeBinaryOp::Sub => 1,
        NativeBinaryOp::Mul => 2,
        NativeBinaryOp::Div => 3,
    }
}

fn broadcast<E: MetalDType>(
    device: &MetalDevice,
    lhs: &MetalStorage,
    rhs: &MetalStorage,
    total: usize,
    period: usize,
    mode: u32,
    op: NativeBinaryOp,
) -> std::result::Result<MetalStorage, MetalError> {
    ensure_len(lhs.len, total)?;
    let output = empty_storage::<E>(device, total);
    if total == 0 {
        return Ok(output);
    }
    let pipeline = pipeline::<E>(device, E::BROADCAST_KERNEL)?;
    encode_and_wait(device, &pipeline, total, |encoder| {
        encoder.set_buffer(0, Some(&lhs.buffer), 0);
        encoder.set_buffer(1, Some(&rhs.buffer), 0);
        encoder.set_buffer(2, Some(&output.buffer), 0);
        set_u32(
            encoder,
            3,
            checked_u32(total, "broadcast length").expect("checked before dispatch"),
        );
        set_u32(
            encoder,
            4,
            checked_u32(period, "broadcast period").expect("checked before dispatch"),
        );
        set_u32(encoder, 5, mode);
        set_u32(encoder, 6, native_binary_op(op));
        set_u32(
            encoder,
            7,
            checked_u32(rhs.len, "broadcast rhs length").expect("checked before dispatch"),
        );
    })?;
    Ok(output)
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

fn set_f32(encoder: &metal_rs::ComputeCommandEncoderRef, index: u64, value: f32) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<f32>() as u64,
        (&value as *const f32).cast::<c_void>(),
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

fn u32_storage(device: &MetalDevice, values: Vec<u32>) -> MetalStorage {
    raw_bytes_storage(device, values)
}

fn u8_storage(device: &MetalDevice, values: Vec<u8>) -> MetalStorage {
    raw_bytes_storage(device, values)
}

fn raw_bytes_storage<T>(device: &MetalDevice, values: Vec<T>) -> MetalStorage {
    let len = values.len();
    let byte_len = len.max(1).saturating_mul(std::mem::size_of::<T>()) as u64;
    let buffer = if len == 0 {
        device
            .raw
            .new_buffer(byte_len, metal_rs::MTLResourceOptions::StorageModeShared)
    } else {
        device.raw.new_buffer_with_data(
            values.as_ptr().cast::<c_void>(),
            byte_len,
            metal_rs::MTLResourceOptions::StorageModeShared,
        )
    };
    MetalStorage {
        buffer: Arc::new(buffer),
        len,
    }
}

fn buffer_ref(storage: &MetalStorage) -> &metal_rs::BufferRef {
    &storage.buffer
}

fn checked_u32_vec(
    values: &[usize],
    name: &'static str,
) -> std::result::Result<Vec<u32>, MetalError> {
    values
        .iter()
        .copied()
        .map(|value| checked_u32(value, name))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{Backend, Metal, SHADERS_F16, SHADERS_F32};
    use crate::dtype::{FloatDType, f16};

    #[test]
    fn bundled_kernel_source_contains_required_entrypoints() {
        assert!(SHADERS_F32.contains("float apply_op_f32"));
        assert!(SHADERS_F32.contains("kernel void binary_f32_kernel"));
        assert!(SHADERS_F32.contains("kernel void scalar_f32_kernel"));
        assert!(SHADERS_F32.contains("kernel void matmul_f32_kernel"));
        assert!(SHADERS_F32.contains("kernel void sum_f32_kernel"));
        assert!(SHADERS_F32.contains("kernel void unary_f32_kernel"));
        assert!(SHADERS_F32.contains("kernel void row_softmax_f32_kernel"));
        assert!(SHADERS_F32.contains("kernel void sum_last_f32_kernel"));
        assert!(SHADERS_F32.contains("kernel void bmm_f32_kernel"));
        assert!(SHADERS_F32.contains("kernel void strided_matmul_f32_kernel"));
        assert!(SHADERS_F32.contains("kernel void broadcast_f32_kernel"));
        assert!(SHADERS_F32.contains("kernel void mask_f32_kernel"));
        assert!(SHADERS_F32.contains("kernel void index_select_rows_f32_kernel"));
        assert!(SHADERS_F32.contains("kernel void cross_entropy_f32_kernel"));
        assert!(SHADERS_F32.contains("kernel void layer_norm_f32_kernel"));
        assert!(SHADERS_F32.contains("kernel void rms_norm_f32_kernel"));
        assert!(SHADERS_F32.contains("kernel void sgd_step_f32_kernel"));
        assert!(SHADERS_F32.contains("kernel void adam_step_f32_kernel"));
        assert!(SHADERS_F16.contains("half apply_op_f16"));
        assert!(SHADERS_F16.contains("kernel void binary_f16_kernel"));
        assert!(SHADERS_F16.contains("kernel void scalar_f16_kernel"));
        assert!(SHADERS_F16.contains("kernel void matmul_f16_kernel"));
        assert!(SHADERS_F16.contains("kernel void sum_f16_kernel"));
        assert!(SHADERS_F16.contains("kernel void unary_f16_kernel"));
        assert!(SHADERS_F16.contains("kernel void row_softmax_f16_kernel"));
        assert!(SHADERS_F16.contains("kernel void sum_last_f16_kernel"));
        assert!(SHADERS_F16.contains("kernel void bmm_f16_kernel"));
        assert!(SHADERS_F16.contains("kernel void strided_matmul_f16_kernel"));
        assert!(SHADERS_F16.contains("kernel void broadcast_f16_kernel"));
        assert!(SHADERS_F16.contains("kernel void mask_f16_kernel"));
        assert!(SHADERS_F16.contains("kernel void index_select_rows_f16_kernel"));
        assert!(SHADERS_F16.contains("kernel void cross_entropy_f16_kernel"));
        assert!(SHADERS_F16.contains("kernel void layer_norm_f16_kernel"));
        assert!(SHADERS_F16.contains("kernel void rms_norm_f16_kernel"));
        assert!(SHADERS_F16.contains("kernel void sgd_step_f16_kernel"));
        assert!(SHADERS_F16.contains("kernel void adam_step_f16_kernel"));
    }

    fn assert_optimizer_kernels<E>()
    where
        E: FloatDType,
        Metal: Backend<E>,
    {
        let Ok(device) = <Metal as Backend<E>>::default_device() else {
            return;
        };
        let param =
            <Metal as Backend<E>>::from_vec(&device, vec![E::from_f64(1.0), E::from_f64(2.0)])
                .unwrap();
        let grad =
            <Metal as Backend<E>>::from_vec(&device, vec![E::from_f64(0.5), E::from_f64(-0.25)])
                .unwrap();

        let (next, velocity) = <Metal as Backend<E>>::sgd_step(
            &device,
            &param,
            &grad,
            None,
            2,
            E::from_f64(0.1),
            Some(E::from_f64(0.9)),
            E::ZERO,
        )
        .unwrap();
        let next = <Metal as Backend<E>>::to_vec(&device, &next).unwrap();
        let velocity = <Metal as Backend<E>>::to_vec(&device, &velocity.unwrap()).unwrap();
        assert_close::<E>(&next, &[0.95, 2.025]);
        assert_close::<E>(&velocity, &[0.5, -0.25]);

        let (next, m, v) = <Metal as Backend<E>>::adam_step(
            &device,
            &param,
            &grad,
            None,
            None,
            2,
            E::from_f64(0.1),
            E::from_f64(0.9),
            E::from_f64(0.999),
            E::from_f64(1e-8),
            E::ZERO,
            E::from_f64(0.9),
            E::from_f64(0.999),
        )
        .unwrap();
        let next = <Metal as Backend<E>>::to_vec(&device, &next).unwrap();
        let m = <Metal as Backend<E>>::to_vec(&device, &m).unwrap();
        let v = <Metal as Backend<E>>::to_vec(&device, &v).unwrap();
        assert_close::<E>(&next, &[0.9, 2.1]);
        assert_close::<E>(&m, &[0.05, -0.025]);
        assert_close::<E>(&v, &[0.00025, 0.0000625]);
    }

    fn assert_close<E: FloatDType>(actual: &[E], expected: &[f64]) {
        let tol = if E::BYTE_SIZE == 2 { 5e-3 } else { 1e-5 };
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!((actual.to_f64() - expected).abs() <= tol);
        }
    }

    #[test]
    fn metal_f32_optimizer_kernels_update_storage() {
        assert_optimizer_kernels::<f32>();
    }

    #[test]
    fn metal_f16_optimizer_kernels_update_storage() {
        assert_optimizer_kernels::<f16>();
    }
}
