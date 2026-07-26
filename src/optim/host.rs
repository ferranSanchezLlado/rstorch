//! The `Tensor` ↔ `HostTensor` bridge (implementation-plan §4, T44).
//!
//! [`persist`](crate::persist) deliberately speaks only in
//! [`HostTensor`](crate::persist::HostTensor)s — dtype + dims + contiguous
//! little-endian bytes — and names "the `nn`/`optim` runtime (T40/T44)" as the
//! owner of the conversion. This module is that owner, kept deliberately
//! minimal: exactly the two directions optimizer-state persistence needs.
//!
//! A `HostTensor` is always canonical row-major, so the outbound direction
//! reads the tensor's *logical* order (a view of a permuted tensor converts
//! correctly) and the inbound direction produces a fresh contiguous tensor with
//! no autograd history.
//!
//! Scope note: the model-level save/load surface — which sections a checkpoint
//! carries, how a checkpoint reconstructs a model — is T52's decision, not
//! this bridge's. Nothing here is public.

use half::{bf16, f16};

use crate::device::Device;
use crate::dtype::DType;
use crate::error::Result;
use crate::persist::HostTensor;
use crate::tensor::Tensor;

/// Flatten host values into little-endian bytes.
fn pack<T, const N: usize>(values: Vec<T>, to_le: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * N);
    for v in values {
        out.extend_from_slice(&to_le(v));
    }
    out
}

/// The inverse of `pack`. `bytes.len()` is a multiple of `N` because
/// `HostTensor` validated it against its dtype at construction.
fn unpack<T, const N: usize>(bytes: &[u8], from_le: impl Fn([u8; N]) -> T) -> Vec<T> {
    bytes
        .chunks_exact(N)
        .map(|chunk| {
            let mut buf = [0u8; N];
            buf.copy_from_slice(chunk);
            from_le(buf)
        })
        .collect()
}

/// Copy `tensor` to the host as a canonical contiguous [`HostTensor`].
///
/// A host boundary: this synchronizes the backend (it is a `to_vec` underneath)
/// and is only ever on a checkpoint path.
pub(crate) fn to_host_tensor(tensor: &Tensor) -> Result<HostTensor> {
    let dtype = tensor.dtype();
    let bytes = match dtype {
        DType::F16 => pack(tensor.to_vec::<f16>()?, f16::to_le_bytes),
        DType::BF16 => pack(tensor.to_vec::<bf16>()?, bf16::to_le_bytes),
        DType::F32 => pack(tensor.to_vec::<f32>()?, f32::to_le_bytes),
        DType::F64 => pack(tensor.to_vec::<f64>()?, f64::to_le_bytes),
        DType::I64 => pack(tensor.to_vec::<i64>()?, i64::to_le_bytes),
        DType::Bool => pack(tensor.to_vec::<bool>()?, |b| [u8::from(b)]),
    };
    HostTensor::from_bytes(dtype, tensor.dims().to_vec(), bytes)
}

/// Build a fresh, contiguous, untraced tensor on `device` from `host`.
pub(crate) fn from_host_tensor(host: &HostTensor, device: &Device) -> Result<Tensor> {
    let dims = host.dims().to_vec();
    let bytes = host.bytes();
    match host.dtype() {
        DType::F16 => Tensor::from_vec(unpack(bytes, f16::from_le_bytes), dims, device),
        DType::BF16 => Tensor::from_vec(unpack(bytes, bf16::from_le_bytes), dims, device),
        DType::F32 => Tensor::from_vec(unpack(bytes, f32::from_le_bytes), dims, device),
        DType::F64 => Tensor::from_vec(unpack(bytes, f64::from_le_bytes), dims, device),
        DType::I64 => Tensor::from_vec(unpack(bytes, i64::from_le_bytes), dims, device),
        DType::Bool => Tensor::from_vec(unpack(bytes, |[b]| b != 0), dims, device),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CPU: Device = Device::Cpu;

    #[test]
    fn f32_round_trips_through_the_host_form() {
        let t = Tensor::from_vec(vec![1.5f32, -2.0, 0.25, 8.0], [2, 2], &CPU).unwrap();
        let host = to_host_tensor(&t).unwrap();
        assert_eq!(host.dtype(), DType::F32);
        assert_eq!(host.dims(), &[2, 2]);
        assert_eq!(host.bytes().len(), 16);

        let back = from_host_tensor(&host, &CPU).unwrap();
        assert_eq!(back.dims(), &[2, 2]);
        assert_eq!(back.to_vec::<f32>().unwrap(), t.to_vec::<f32>().unwrap());
        // No history rides along.
        assert!(back.backward().is_err());
    }

    #[test]
    fn i64_and_bool_round_trip() {
        let ints = Tensor::from_vec(vec![-1i64, 0, 7], [3], &CPU).unwrap();
        let host = to_host_tensor(&ints).unwrap();
        assert_eq!(
            from_host_tensor(&host, &CPU)
                .unwrap()
                .to_vec::<i64>()
                .unwrap(),
            vec![-1, 0, 7]
        );

        let mask = Tensor::from_vec(vec![true, false, true], [3], &CPU).unwrap();
        let host = to_host_tensor(&mask).unwrap();
        assert_eq!(host.bytes(), &[1u8, 0, 1]);
        assert_eq!(
            from_host_tensor(&host, &CPU)
                .unwrap()
                .to_vec::<bool>()
                .unwrap(),
            vec![true, false, true]
        );
    }

    #[test]
    fn a_view_converts_in_logical_order() {
        // The host form is canonical row-major, so a transposed view must be
        // materialized in *its* order, not the source's.
        let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], &CPU).unwrap();
        let view = t.transpose(0, 1).unwrap();
        assert!(!view.is_contiguous());
        let host = to_host_tensor(&view).unwrap();
        assert_eq!(host.dims(), &[3, 2]);
        assert_eq!(
            from_host_tensor(&host, &CPU)
                .unwrap()
                .to_vec::<f32>()
                .unwrap(),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn a_traced_tensor_loses_its_graph() {
        let t = Tensor::from_vec(vec![1.0f32], [1], &CPU)
            .unwrap()
            .traced()
            .unwrap();
        let host = to_host_tensor(&t).unwrap();
        assert!(from_host_tensor(&host, &CPU).unwrap().backward().is_err());
    }
}
