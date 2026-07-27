//! Fused CPU kernels for last-axis softmax and layer normalization.
//!
//! Optimizer fusion needs more than the frozen single-[`Storage`] return can
//! represent: momentum SGD produces a parameter and velocity, while Adam
//! produces a parameter and two moments. Those variants therefore remain loud
//! [`Error::Unsupported`] results rather than mutating immutable input storage
//! or silently dropping updated state.

use std::sync::Arc;

use crate::backend::{FusedOp, View};
use crate::device::Device;
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::layout::Layout;
use crate::storage::{CpuStorage, Storage};

trait FloatAcc:
    Copy
    + PartialEq
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
{
    const ZERO: Self;
    const NEG_INFINITY: Self;

    fn from_usize(value: usize) -> Self;
    fn exp(self) -> Self;
    fn sqrt(self) -> Self;
    fn is_nan(self) -> bool;
}

macro_rules! impl_float_acc {
    ($ty:ty) => {
        impl FloatAcc for $ty {
            const ZERO: Self = 0.0;
            const NEG_INFINITY: Self = Self::NEG_INFINITY;

            fn from_usize(value: usize) -> Self {
                value as Self
            }
            fn exp(self) -> Self {
                self.exp()
            }
            fn sqrt(self) -> Self {
                self.sqrt()
            }
            fn is_nan(self) -> bool {
                self.is_nan()
            }
        }
    };
}

impl_float_acc!(f32);
impl_float_acc!(f64);

/// See [`BackendOps::fused`](crate::backend::BackendOps::fused).
///
/// Operand encoding:
///
/// - `Softmax`: `[x]`, no scalars; normalizes the last axis.
/// - `LayerNorm`: `[x, weight, bias]`, `[eps]`; `weight` and `bias` are
///   rank-one views matching `x`'s last axis.
pub(crate) fn fused(op: FusedOp, inputs: &[View<'_>], scalars: &[f64]) -> Result<Storage> {
    match op {
        FusedOp::Softmax => softmax(inputs, scalars),
        FusedOp::LayerNorm => layer_norm(inputs, scalars),
        FusedOp::SgdStep => unsupported("fused_sgd_step", inputs),
        FusedOp::AdamStep => unsupported("fused_adam_step", inputs),
    }
}

fn softmax(inputs: &[View<'_>], scalars: &[f64]) -> Result<Storage> {
    const OP: &str = "fused_softmax";
    require_encoding(OP, inputs, 1, scalars, 0)?;
    let x = inputs[0];
    require_last_axis(OP, x.layout())?;
    require_float(OP, x)?;
    validate_view(OP, x)?;

    let output = match cpu_storage(OP, x)? {
        CpuStorage::F16(values) => CpuStorage::F16(Arc::new(softmax_generic(values, x.layout()))),
        CpuStorage::BF16(values) => CpuStorage::BF16(Arc::new(softmax_generic(values, x.layout()))),
        CpuStorage::F32(values) => CpuStorage::F32(Arc::new(softmax_generic(values, x.layout()))),
        CpuStorage::F64(values) => CpuStorage::F64(Arc::new(softmax_generic(values, x.layout()))),
        CpuStorage::I64(_) | CpuStorage::Bool(_) => unreachable!("validated float dtype"),
    };
    Ok(Storage::Cpu(output))
}

fn softmax_generic<E>(values: &[E], layout: &Layout) -> Vec<E>
where
    E: Element,
    E::Acc: FloatAcc,
{
    let width = layout.dims()[layout.rank() - 1];
    let rows = layout.num_elements() / width;
    let stride = layout.strides()[layout.rank() - 1];
    let mut output = Vec::with_capacity(layout.num_elements());
    let mut exponents = Vec::with_capacity(width);

    for row in 0..rows {
        let base = row_base(layout, row);
        let mut peak = E::Acc::NEG_INFINITY;
        for col in 0..width {
            let value = values[base + col * stride].to_acc();
            peak = if peak.is_nan() || value.is_nan() {
                // Match the composed Max reduction's NaN propagation.
                peak + value
            } else if value > peak {
                value
            } else {
                peak
            };
        }

        if peak == E::Acc::NEG_INFINITY {
            output.extend((0..width).map(|_| E::from_acc(E::Acc::ZERO)));
            continue;
        }

        exponents.clear();
        let mut denominator = E::Acc::ZERO;
        for col in 0..width {
            let exponent = (values[base + col * stride].to_acc() - peak).exp();
            denominator = denominator + exponent;
            exponents.push(exponent);
        }
        output.extend(
            exponents
                .iter()
                .copied()
                .map(|value| E::from_acc(value / denominator)),
        );
    }
    output
}

fn layer_norm(inputs: &[View<'_>], scalars: &[f64]) -> Result<Storage> {
    const OP: &str = "fused_layer_norm";
    require_encoding(OP, inputs, 3, scalars, 1)?;
    let [x, weight, bias] = inputs else {
        unreachable!("arity validated")
    };
    require_last_axis(OP, x.layout())?;
    require_float(OP, *x)?;
    require_same_dtype(OP, *x, *weight)?;
    require_same_dtype(OP, *x, *bias)?;
    validate_view(OP, *x)?;
    validate_view(OP, *weight)?;
    validate_view(OP, *bias)?;

    let width = x.layout().dims()[x.layout().rank() - 1];
    for affine in [weight, bias] {
        if affine.layout().rank() != 1 || affine.layout().dims()[0] != width {
            return Err(Error::ShapeMismatch {
                op: OP,
                lhs: x.layout().shape().clone(),
                rhs: affine.layout().shape().clone(),
            });
        }
    }
    let eps = scalars[0];
    if !(eps.is_finite() && eps > 0.0) {
        return Err(Error::InvalidArg {
            op: OP,
            msg: format!("eps must be finite and positive, got {eps}"),
        });
    }

    let storages = [
        cpu_storage(OP, *x)?,
        cpu_storage(OP, *weight)?,
        cpu_storage(OP, *bias)?,
    ];
    let output = match storages {
        [
            CpuStorage::F16(xv),
            CpuStorage::F16(wv),
            CpuStorage::F16(bv),
        ] => CpuStorage::F16(Arc::new(layer_norm_generic(
            xv,
            x.layout(),
            wv,
            weight.layout(),
            bv,
            bias.layout(),
            eps as f32,
        ))),
        [
            CpuStorage::BF16(xv),
            CpuStorage::BF16(wv),
            CpuStorage::BF16(bv),
        ] => CpuStorage::BF16(Arc::new(layer_norm_generic(
            xv,
            x.layout(),
            wv,
            weight.layout(),
            bv,
            bias.layout(),
            eps as f32,
        ))),
        [
            CpuStorage::F32(xv),
            CpuStorage::F32(wv),
            CpuStorage::F32(bv),
        ] => CpuStorage::F32(Arc::new(layer_norm_generic(
            xv,
            x.layout(),
            wv,
            weight.layout(),
            bv,
            bias.layout(),
            eps as f32,
        ))),
        [
            CpuStorage::F64(xv),
            CpuStorage::F64(wv),
            CpuStorage::F64(bv),
        ] => CpuStorage::F64(Arc::new(layer_norm_generic(
            xv,
            x.layout(),
            wv,
            weight.layout(),
            bv,
            bias.layout(),
            eps,
        ))),
        _ => unreachable!("dtypes validated equal and float"),
    };
    Ok(Storage::Cpu(output))
}

#[allow(clippy::too_many_arguments)]
fn layer_norm_generic<E>(
    values: &[E],
    layout: &Layout,
    weights: &[E],
    weight_layout: &Layout,
    biases: &[E],
    bias_layout: &Layout,
    eps: E::Acc,
) -> Vec<E>
where
    E: Element,
    E::Acc: FloatAcc,
{
    let width = layout.dims()[layout.rank() - 1];
    let rows = layout.num_elements() / width;
    let stride = layout.strides()[layout.rank() - 1];
    let mut output = Vec::with_capacity(layout.num_elements());

    for row in 0..rows {
        let base = row_base(layout, row);
        let mut sum = E::Acc::ZERO;
        for col in 0..width {
            sum = sum + values[base + col * stride].to_acc();
        }
        let mean = sum / E::Acc::from_usize(width);
        let mut squared = E::Acc::ZERO;
        for col in 0..width {
            let centered = values[base + col * stride].to_acc() - mean;
            squared = squared + centered * centered;
        }
        let scale = (squared / E::Acc::from_usize(width) + eps).sqrt();
        for col in 0..width {
            let normalized = (values[base + col * stride].to_acc() - mean) / scale;
            let weight =
                weights[weight_layout.offset() + col * weight_layout.strides()[0]].to_acc();
            let bias = biases[bias_layout.offset() + col * bias_layout.strides()[0]].to_acc();
            output.push(E::from_acc(normalized * weight + bias));
        }
    }
    output
}

fn row_base(layout: &Layout, row: usize) -> usize {
    let last = layout.rank() - 1;
    let mut base = layout.offset();
    let mut remainder = row;
    for axis in (0..last).rev() {
        let dim = layout.dims()[axis];
        base += (remainder % dim) * layout.strides()[axis];
        remainder /= dim;
    }
    base
}

fn require_encoding(
    op: &'static str,
    inputs: &[View<'_>],
    input_count: usize,
    scalars: &[f64],
    scalar_count: usize,
) -> Result<()> {
    if inputs.len() != input_count || scalars.len() != scalar_count {
        return Err(Error::InvalidArg {
            op,
            msg: format!(
                "expected {input_count} input(s) and {scalar_count} scalar(s), got {} and {}",
                inputs.len(),
                scalars.len()
            ),
        });
    }
    Ok(())
}

fn require_last_axis(op: &'static str, layout: &Layout) -> Result<()> {
    if layout.rank() == 0 {
        return Err(Error::RankMismatch {
            op,
            expected: 1,
            got: 0,
        });
    }
    if layout.dims()[layout.rank() - 1] == 0 {
        return Err(Error::InvalidArg {
            op,
            msg: "last axis must be non-empty".to_owned(),
        });
    }
    Ok(())
}

fn require_float(op: &'static str, view: View<'_>) -> Result<()> {
    if !view.dtype().is_float() {
        return Err(Error::Unsupported {
            op,
            device: view.device(),
            dtype: view.dtype(),
        });
    }
    Ok(())
}

fn require_same_dtype(op: &'static str, expected: View<'_>, got: View<'_>) -> Result<()> {
    if expected.dtype() != got.dtype() {
        return Err(Error::DTypeMismatch {
            op,
            expected: expected.dtype(),
            got: got.dtype(),
        });
    }
    Ok(())
}

fn validate_view(op: &'static str, view: View<'_>) -> Result<()> {
    if view.layout().num_elements() == 0 {
        return Ok(());
    }
    let mut highest = view.layout().offset();
    for (&dim, &stride) in view.layout().dims().iter().zip(view.layout().strides()) {
        highest = highest
            .checked_add(
                (dim - 1)
                    .checked_mul(stride)
                    .ok_or_else(|| Error::InvalidArg {
                        op,
                        msg: "view address overflows usize".to_owned(),
                    })?,
            )
            .ok_or_else(|| Error::InvalidArg {
                op,
                msg: "view address overflows usize".to_owned(),
            })?;
    }
    if highest >= view.storage().len() {
        return Err(Error::InvalidArg {
            op,
            msg: format!(
                "view reaches storage index {highest}, but backing storage has length {}",
                view.storage().len()
            ),
        });
    }
    Ok(())
}

fn unsupported(op: &'static str, inputs: &[View<'_>]) -> Result<Storage> {
    Err(Error::Unsupported {
        op,
        device: inputs.first().map_or(Device::Cpu, View::device),
        dtype: inputs.first().map_or(DType::F32, View::dtype),
    })
}

#[cfg_attr(not(feature = "metal"), allow(unused_variables))]
fn cpu_storage<'a>(op: &'static str, view: View<'a>) -> Result<&'a CpuStorage> {
    match view.storage() {
        Storage::Cpu(storage) => Ok(storage),
        #[cfg(feature = "metal")]
        Storage::Metal(_) => Err(Error::Unsupported {
            op,
            device: view.device(),
            dtype: view.dtype(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn storage(values: Vec<f32>) -> Storage {
        Storage::Cpu(CpuStorage::F32(Arc::new(values)))
    }

    fn values(storage: Storage) -> Vec<f32> {
        match storage {
            Storage::Cpu(CpuStorage::F32(values)) => Arc::unwrap_or_clone(values),
            _ => panic!("expected f32 CPU storage"),
        }
    }

    fn close(got: &[f32], expected: &[f32]) {
        assert_eq!(got.len(), expected.len());
        for (&got, &expected) in got.iter().zip(expected) {
            assert!((got - expected).abs() < 1e-6, "{got} != {expected}");
        }
    }

    #[test]
    fn softmax_is_stable_and_handles_fully_masked_rows() {
        let x = storage(vec![
            1000.0,
            1001.0,
            1002.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ]);
        let layout = Layout::contiguous([2, 3]).unwrap();
        let composed = crate::tensor::Tensor::from_parts(x.clone(), layout.clone())
            .softmax(-1)
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let got = values(fused(FusedOp::Softmax, &[View::new(&x, &layout)], &[]).unwrap());
        close(&got, &composed);
        close(&got[..3], &[0.090_030_57, 0.244_728_48, 0.665_240_94]);
        assert_eq!(&got[3..], &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn softmax_reads_a_strided_last_axis_and_writes_contiguous_output() {
        let x = storage(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
        let base = Layout::contiguous([3, 2]).unwrap();
        let transposed = base.transpose(0, 1).unwrap();
        let got = values(fused(FusedOp::Softmax, &[View::new(&x, &transposed)], &[]).unwrap());
        close(&got[..3], &[0.090_030_57, 0.244_728_48, 0.665_240_94]);
        assert!(got[3] < 1e-8);
        assert!(got[4] < 1e-4);
        assert!(got[5] > 0.9999);
    }

    #[test]
    fn layer_norm_applies_strided_affine_views() {
        let x = storage(vec![1.0, 2.0, 3.0, 4.0, 6.0, 8.0]);
        let x_layout = Layout::contiguous([2, 3]).unwrap();
        let weight = storage(vec![2.0, 99.0, 3.0, 99.0, 4.0]);
        let bias = storage(vec![1.0, 99.0, -1.0, 99.0, 0.5]);
        let affine_layout = Layout::from_parts(
            crate::shape::Shape::from([3]),
            vec![2].into_boxed_slice(),
            0,
        )
        .unwrap();
        let got = values(
            fused(
                FusedOp::LayerNorm,
                &[
                    View::new(&x, &x_layout),
                    View::new(&weight, &affine_layout),
                    View::new(&bias, &affine_layout),
                ],
                &[1e-5],
            )
            .unwrap(),
        );
        close(&got[..3], &[-1.449_471_2, -1.0, 5.398_942_5]);
        close(&got[3..], &[-1.449_485_3, -1.0, 5.398_970_6]);
    }

    #[test]
    fn half_layer_norm_accumulates_mean_in_f32() {
        let x = Storage::Cpu(CpuStorage::F16(Arc::new(vec![
            half::f16::from_f32(2048.0),
            half::f16::from_f32(1.0),
            half::f16::from_f32(-2048.0),
            half::f16::from_f32(-1.0),
        ])));
        let weight = Storage::Cpu(CpuStorage::F16(Arc::new(vec![half::f16::from_f32(1.0); 4])));
        let bias = Storage::Cpu(CpuStorage::F16(Arc::new(vec![half::f16::from_f32(0.0); 4])));
        let layout = Layout::contiguous([4]).unwrap();
        let result = fused(
            FusedOp::LayerNorm,
            &[
                View::new(&x, &layout),
                View::new(&weight, &layout),
                View::new(&bias, &layout),
            ],
            &[1e-5],
        )
        .unwrap();
        let Storage::Cpu(CpuStorage::F16(result)) = result else {
            panic!("expected f16 CPU storage")
        };
        assert_eq!(result[0].to_f32(), -result[2].to_f32());
        assert_eq!(result[1].to_f32(), -result[3].to_f32());
    }

    #[test]
    fn invalid_encodings_and_unsupported_variants_are_loud() {
        let x = storage(vec![1.0, 2.0]);
        let layout = Layout::contiguous([2]).unwrap();
        let view = View::new(&x, &layout);
        assert!(matches!(
            fused(FusedOp::Softmax, &[view], &[1.0]),
            Err(Error::InvalidArg {
                op: "fused_softmax",
                ..
            })
        ));
        assert!(matches!(
            fused(FusedOp::SgdStep, &[view], &[]),
            Err(Error::Unsupported {
                op: "fused_sgd_step",
                ..
            })
        ));
        assert!(matches!(
            fused(FusedOp::AdamStep, &[view], &[]),
            Err(Error::Unsupported {
                op: "fused_adam_step",
                ..
            })
        ));
    }

    #[test]
    fn non_float_and_out_of_bounds_views_are_rejected() {
        let integers = Storage::Cpu(CpuStorage::I64(Arc::new(vec![1, 2])));
        let layout = Layout::contiguous([2]).unwrap();
        assert!(matches!(
            fused(FusedOp::Softmax, &[View::new(&integers, &layout)], &[]),
            Err(Error::Unsupported {
                dtype: DType::I64,
                ..
            })
        ));

        let x = storage(vec![1.0]);
        assert!(matches!(
            fused(FusedOp::Softmax, &[View::new(&x, &layout)], &[]),
            Err(Error::InvalidArg {
                op: "fused_softmax",
                ..
            })
        ));
    }
}
