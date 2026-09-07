//! Tensor-level checks for deferred chains.
//!
//! This module intentionally does not reuse backend conformance. A backend
//! conformance case calls one primitive directly; these checks build a graph,
//! exercise admission and realization, and compare the resulting bytes.

use crate::device::Device;
use crate::dtype::DType;
use crate::error::Result;
use crate::tensor::Tensor;

/// Compare two CPU f32 tensors by their exact IEEE-754 bit patterns.
#[cfg(test)]
fn assert_f32_bits(left: &Tensor, right: &Tensor) {
    let lhs = left.to_vec::<f32>().expect("left host read");
    let rhs = right.to_vec::<f32>().expect("right host read");
    assert_eq!(lhs.len(), rhs.len());
    for (index, (&a, &b)) in lhs.iter().zip(&rhs).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "element {index}");
    }
}

/// Run one chain in eager and deferred modes on the CPU.
#[cfg(test)]
fn compare_chain(build: impl Fn(&Tensor) -> Result<Tensor>) {
    let input = Tensor::from_vec(
        (0..4096).map(|i| (i as f32 - 2048.0) / 257.0).collect(),
        [4096],
        &Device::Cpu,
    )
    .expect("harness input");
    let eager = {
        let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::Off);
        build(&input).expect("eager chain")
    };
    let deferred = {
        let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::On);
        build(&input).expect("deferred chain")
    };
    assert_f32_bits(&eager, &deferred);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chains_match_bit_for_bit() {
        compare_chain(|x| x.add_scalar(1.25)?.mul_scalar(-2.0)?.neg()?.sub_scalar(3.0));
    }

    #[test]
    fn long_private_chain_keeps_stage_order() {
        let input = Tensor::from_vec(vec![-1.5f32, 0.25, 4.0], [3], &Device::Cpu).unwrap();
        let eager = {
            let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::Off);
            let mut value = input.clone();
            for _ in 0..17 {
                value = value.add_scalar(0.5).unwrap();
            }
            value
        };
        let deferred = {
            let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::On);
            let mut value = input;
            for _ in 0..17 {
                value = value.add_scalar(0.5).unwrap();
            }
            value
        };
        assert_f32_bits(&eager, &deferred);
    }

    #[test]
    fn unary_and_materialized_rhs_chain_matches() {
        let input = Tensor::from_vec(vec![-2.0f32, -0.5, 0.25, 1.5], [4], &Device::Cpu).unwrap();
        let rhs = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [4], &Device::Cpu).unwrap();
        let eager = {
            let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::Off);
            input.relu().unwrap().mul(&rhs).unwrap()
        };
        let deferred = {
            let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::On);
            input.relu().unwrap().mul(&rhs).unwrap()
        };
        assert_f32_bits(&eager, &deferred);
    }

    #[test]
    fn narrowed_dense_base_fuses_without_losing_offset() {
        let source =
            Tensor::from_vec(vec![-4.0f32, -2.0, -0.5, 0.25, 1.5, 3.0], [6], &Device::Cpu).unwrap();
        let input = source.narrow(0, 1, 4).unwrap();
        let eager = {
            let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::Off);
            input.add_scalar(1.0).unwrap().mul_scalar(2.0).unwrap()
        };
        let deferred = {
            let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::On);
            input.add_scalar(1.0).unwrap().mul_scalar(2.0).unwrap()
        };
        assert_f32_bits(&eager, &deferred);
    }

    #[test]
    fn shape_view_terminates_pending_chain() {
        let source =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], &Device::Cpu).unwrap();
        let eager = {
            let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::Off);
            source
                .add_scalar(10.0)
                .unwrap()
                .transpose(0, 1)
                .unwrap()
                .mul_scalar(2.0)
                .unwrap()
        };
        let deferred = {
            let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::On);
            source
                .add_scalar(10.0)
                .unwrap()
                .transpose(0, 1)
                .unwrap()
                .mul_scalar(2.0)
                .unwrap()
        };
        assert_f32_bits(&eager, &deferred);
        assert_eq!(
            deferred.to_vec::<f32>().unwrap(),
            vec![22.0, 28.0, 24.0, 30.0, 26.0, 32.0]
        );
    }

    #[test]
    fn broadcast_view_terminates_pending_chain() {
        let lhs = Tensor::from_vec(vec![1.0f32, 2.0], [2, 1], &Device::Cpu).unwrap();
        let rhs = Tensor::from_vec(vec![10.0f32; 6], [2, 3], &Device::Cpu).unwrap();
        let eager = {
            let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::Off);
            lhs.add_scalar(1.0).unwrap().add(&rhs).unwrap()
        };
        let deferred = {
            let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::On);
            lhs.add_scalar(1.0).unwrap().add(&rhs).unwrap()
        };
        assert_f32_bits(&eager, &deferred);
        assert_eq!(
            deferred.to_vec::<f32>().unwrap(),
            vec![12.0, 12.0, 12.0, 13.0, 13.0, 13.0]
        );
    }

    #[test]
    fn every_fused_unary_stage_matches_eager_bits() {
        let input = Tensor::from_vec(vec![-2.25f32, -0.0, 0.5, 2.75], [4], &Device::Cpu).unwrap();
        let operations: [fn(&Tensor) -> Result<Tensor>; 15] = [
            Tensor::relu,
            Tensor::gelu,
            Tensor::exp,
            Tensor::ln,
            Tensor::sqrt,
            Tensor::tanh,
            Tensor::sigmoid,
            Tensor::neg,
            Tensor::abs,
            Tensor::sign,
            Tensor::recip,
            Tensor::floor,
            Tensor::ceil,
            Tensor::round,
            Tensor::erf,
        ];
        for operation in operations {
            let eager = {
                let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::Off);
                operation(&input).expect("eager unary")
            };
            let deferred = {
                let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::On);
                operation(&input).expect("deferred unary")
            };
            assert_f32_bits(&eager, &deferred);
        }
    }

    #[test]
    fn fused_non_f32_cpu_dtypes_match_eager_bits() {
        let f16 = Tensor::from_vec(
            vec![
                half::f16::from_f32(-2.25),
                half::f16::from_f32(-0.5),
                half::f16::from_f32(0.25),
                half::f16::from_f32(1.5),
            ],
            [4],
            &Device::Cpu,
        )
        .unwrap();
        let eager = {
            let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::Off);
            f16.add_scalar(1.25)
                .unwrap()
                .mul_scalar(-2.0)
                .unwrap()
                .neg()
                .unwrap()
        };
        let deferred = {
            let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::On);
            f16.add_scalar(1.25)
                .unwrap()
                .mul_scalar(-2.0)
                .unwrap()
                .neg()
                .unwrap()
        };
        assert_eq!(
            eager.to_vec::<half::f16>().unwrap(),
            deferred.to_vec::<half::f16>().unwrap()
        );

        let bf16 = Tensor::from_vec(
            vec![
                half::bf16::from_f32(-2.25),
                half::bf16::from_f32(-0.5),
                half::bf16::from_f32(0.25),
                half::bf16::from_f32(1.5),
            ],
            [4],
            &Device::Cpu,
        )
        .unwrap();
        let eager = {
            let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::Off);
            bf16.exp().unwrap().sub_scalar(0.5).unwrap()
        };
        let deferred = {
            let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::On);
            bf16.exp().unwrap().sub_scalar(0.5).unwrap()
        };
        assert_eq!(
            eager.to_vec::<half::bf16>().unwrap(),
            deferred.to_vec::<half::bf16>().unwrap()
        );

        let f64 = Tensor::from_vec(vec![-2.25f64, -0.5, 0.25, 1.5], [4], &Device::Cpu).unwrap();
        let eager = {
            let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::Off);
            f64.exp().unwrap().mul_scalar(1.25).unwrap()
        };
        let deferred = {
            let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::On);
            f64.exp().unwrap().mul_scalar(1.25).unwrap()
        };
        assert_eq!(
            eager.to_vec::<f64>().unwrap(),
            deferred.to_vec::<f64>().unwrap()
        );

        let i64 = Tensor::from_vec(vec![-4i64, -1, 2, 7], [4], &Device::Cpu).unwrap();
        let eager = {
            let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::Off);
            i64.add_scalar(3.0).unwrap().abs().unwrap()
        };
        let deferred = {
            let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::On);
            i64.add_scalar(3.0).unwrap().abs().unwrap()
        };
        assert_eq!(
            eager.to_vec::<i64>().unwrap(),
            deferred.to_vec::<i64>().unwrap()
        );
    }
    #[test]
    fn metadata_does_not_force_pending_storage() {
        let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::On);
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3], &Device::Cpu).unwrap();
        let y = x.add_scalar(2.0).unwrap();
        let crate::storage::Storage::Pending(node) = y.storage() else {
            panic!("expected a pending tensor")
        };
        assert!(node.cache.get().is_none());
        assert_eq!(y.dims(), &[3]);
        assert_eq!(y.shape().dims(), &[3]);
        assert_eq!(y.rank(), 1);
        assert_eq!(y.num_elements(), 3);
        assert_eq!(y.dtype(), DType::F32);
        assert_eq!(y.device(), Device::Cpu);
        assert!(node.cache.get().is_none());
    }

    #[test]
    fn terminating_reduction_realizes_its_input() {
        let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::On);
        let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [4], &Device::Cpu).unwrap();
        let y = x.add_scalar(1.0).unwrap().sum_all().unwrap();
        assert_eq!(y.to_vec::<f32>().unwrap(), vec![14.0]);
    }
    #[test]
    fn argument_errors_are_not_deferred() {
        let x = Tensor::from_vec(vec![1.0f32, 2.0], [2], &Device::Cpu).unwrap();
        let other = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], [3], &Device::Cpu).unwrap();
        assert!(matches!(
            x.add(&other),
            Err(crate::Error::ShapeMismatch { op: "add", .. })
        ));
        assert!(matches!(
            x.dims2(),
            Err(crate::Error::RankMismatch { op: "dims2", .. })
        ));
        assert!(matches!(
            x.sum(4),
            Err(crate::Error::InvalidAxis { op: "sum", .. })
        ));
        let integers = Tensor::from_vec(vec![1i64, 2], [2], &Device::Cpu).unwrap();
        assert!(matches!(
            x.add(&integers),
            Err(crate::Error::DTypeMismatch { op: "add", .. })
        ));
        assert!(matches!(
            x.reshape([3]),
            Err(crate::Error::ReshapeMismatch { op: "reshape", .. })
        ));
        assert!(matches!(
            x.clamp(f64::NAN, 1.0),
            Err(crate::Error::InvalidArg { op: "clamp", .. })
        ));
        let indices = Tensor::from_vec(vec![2i64], [1], &Device::Cpu).unwrap();
        assert!(matches!(
            x.index_select(0, &indices),
            Err(crate::Error::IndexOutOfBounds {
                op: "index_select",
                ..
            })
        ));
    }
    #[cfg(all(feature = "testing", not(feature = "wgpu")))]
    #[test]
    fn allocation_counts_for_four_stage_chain() {
        let x = Tensor::from_vec(vec![1.0f32; 1024], [1024], &Device::Cpu).unwrap();
        let eager_stats = {
            let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::Off);
            crate::testing::reset_allocation_stats();
            let value = x
                .add_scalar(1.0)
                .unwrap()
                .mul_scalar(2.0)
                .unwrap()
                .neg()
                .unwrap()
                .sub_scalar(0.5)
                .unwrap();
            let _ = value.to_vec::<f32>().unwrap();
            crate::testing::allocation_stats()
        };
        let fused_stats = {
            let _guard = crate::lazy::set_fusion(crate::lazy::Fusion::On);
            crate::testing::reset_allocation_stats();
            let value = x
                .add_scalar(1.0)
                .unwrap()
                .mul_scalar(2.0)
                .unwrap()
                .neg()
                .unwrap()
                .sub_scalar(0.5)
                .unwrap();
            let _ = value.to_vec::<f32>().unwrap();
            crate::testing::allocation_stats()
        };
        eprintln!("eager allocations: {eager_stats:?}");
        eprintln!("fused allocations: {fused_stats:?}");
        assert!(eager_stats.allocations > 0 && fused_stats.allocations > 0);
    }
}
