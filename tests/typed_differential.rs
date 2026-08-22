#![cfg(feature = "typed")]

use rstorch::typed::prelude::*;
use rstorch::{DType, Device, Element, Error, Result, Tensor};
use std::fs;
use std::path::Path;

struct ExecutableCase {
    name: &'static str,
    run: fn() -> Result<()>,
}

const FAMILY_CASES: [ExecutableCase; 8] = [
    ExecutableCase {
        name: "shape/views",
        run: shape_view_family,
    },
    ExecutableCase {
        name: "elementwise",
        run: elementwise_family,
    },
    ExecutableCase {
        name: "reductions",
        run: reduction_family,
    },
    ExecutableCase {
        name: "matmul",
        run: matmul_family,
    },
    ExecutableCase {
        name: "indexing",
        run: indexing_family,
    },
    ExecutableCase {
        name: "conv/pool",
        run: conv_pool_family,
    },
    ExecutableCase {
        name: "losses",
        run: loss_family,
    },
    ExecutableCase {
        name: "core/autograd",
        run: core_autograd_family,
    },
];

const DTYPE_CASES: [ExecutableCase; 5] = [
    ExecutableCase {
        name: "f64",
        run: f64_capabilities,
    },
    ExecutableCase {
        name: "f16",
        run: f16_capabilities,
    },
    ExecutableCase {
        name: "bf16",
        run: bf16_capabilities,
    },
    ExecutableCase {
        name: "i64",
        run: i64_capabilities,
    },
    ExecutableCase {
        name: "bool",
        run: bool_capabilities,
    },
];

fn cpu() -> DeviceCtx<Cpu> {
    DeviceCtx::cpu().unwrap()
}

fn assert_f32_parity(typed: &Tensor, dynamic: &Tensor) {
    assert_eq!(typed.dims(), dynamic.dims());
    assert_eq!(typed.dtype(), dynamic.dtype());
    assert_eq!(typed.device(), dynamic.device());
    assert_eq!(
        typed.to_vec::<f32>().unwrap(),
        dynamic.to_vec::<f32>().unwrap()
    );
}

fn assert_typed_dynamic_parity<E>(typed: &Tensor, dynamic: &Tensor)
where
    E: Element + PartialEq,
{
    assert_eq!(typed.dims(), dynamic.dims());
    assert_eq!(typed.dtype(), dynamic.dtype());
    assert_eq!(typed.device(), dynamic.device());
    assert_eq!(typed.to_vec::<E>().unwrap(), dynamic.to_vec::<E>().unwrap());
}

fn assert_error_parity(typed: Error, dynamic: Error) {
    assert_eq!(
        std::mem::discriminant(&typed),
        std::mem::discriminant(&dynamic)
    );
    assert_eq!(typed.to_string(), dynamic.to_string());
}

fn assert_grad_parity(typed: &Tensor, dynamic: &Tensor) {
    assert_eq!(typed.dims(), dynamic.dims());
    let lhs = typed.to_vec::<f32>().unwrap();
    let rhs = dynamic.to_vec::<f32>().unwrap();
    assert_eq!(lhs.len(), rhs.len());
    for (lhs, rhs) in lhs.into_iter().zip(rhs) {
        assert!(
            (lhs - rhs).abs() <= 1e-6,
            "gradient mismatch: {lhs} vs {rhs}"
        );
    }
}

macro_rules! assert_reduced_grad_parity {
    ($element:ty, $typed:expr, $dynamic:expr, $to_f64:expr, $tolerance:expr) => {{
        let typed = $typed.to_vec::<$element>()?;
        let dynamic = $dynamic.to_vec::<$element>()?;
        assert_eq!(typed.len(), dynamic.len());
        assert!(
            typed.iter().any(|value| $to_f64(*value).abs() > $tolerance),
            "reduced-precision gradient evidence must be non-vacuous"
        );
        for (typed, dynamic) in typed.into_iter().zip(dynamic) {
            let typed = $to_f64(typed);
            let dynamic = $to_f64(dynamic);
            assert!(
                (typed - dynamic).abs() <= $tolerance,
                "gradient mismatch: {typed} vs {dynamic}"
            );
        }
    }};
}

#[test]
fn operation_family_table_executes_all_differential_cases() {
    for case in FAMILY_CASES {
        (case.run)().unwrap_or_else(|error| panic!("{} case failed: {error}", case.name));
    }
}

#[test]
fn dtype_capability_table_executes_all_accepted_cases() {
    for case in DTYPE_CASES {
        (case.run)().unwrap_or_else(|error| panic!("{} case failed: {error}", case.name));
    }
}

fn shape_view_family() -> Result<()> {
    let ctx = cpu();
    let leaf =
        Tensor::from_vec((0..6).map(|x| x as f32).collect(), [2, 1, 3], &Device::Cpu)?.traced()?;
    let typed = Tensor3::<DYN, 1, 3>::try_from_dynamic(leaf.clone(), &ctx)?;
    let typed_view = typed.squeeze::<1>()?.transpose::<0, 1>()?;
    let dynamic_view = leaf.squeeze(1)?.transpose(0, 1)?;
    assert_f32_parity(typed_view.as_dynamic(), &dynamic_view);

    let empty = Tensor2::<0, 3>::from_vec(Vec::<f32>::new(), [0, 3], &ctx)?;
    assert_f32_parity(
        empty.transpose::<0, 1>()?.as_dynamic(),
        &empty.as_dynamic().transpose(0, 1)?,
    );

    let typed_error = typed.narrow::<2>(2, 2).unwrap_err();
    let dynamic_error = leaf.narrow(2, 2, 2).unwrap_err();
    assert_error_parity(typed_error, dynamic_error);

    let typed_grads = typed_view.sum_all()?.backward()?;
    let dynamic_grads = dynamic_view.sum_all()?.backward()?;
    assert_grad_parity(
        &typed_grads.wrt_input(&leaf)?,
        &dynamic_grads.wrt_input(&leaf)?,
    );
    Ok(())
}

fn elementwise_family() -> Result<()> {
    let ctx = cpu();
    let leaf = Tensor::from_vec(vec![-1.0f32, 2.0, 3.0, -4.0], [2, 2], &Device::Cpu)?.traced()?;
    let typed = Tensor2::<DYN, DYN>::try_from_dynamic(leaf.clone(), &ctx)?.transpose::<0, 1>()?;
    let rhs = Tensor2::<DYN, DYN>::from_vec(vec![2.0; 4], [2, 2], &ctx)?.transpose::<0, 1>()?;
    let actual = typed.mul(&rhs)?.tanh()?;
    let expected = leaf.transpose(0, 1)?.mul(rhs.as_dynamic())?.tanh()?;
    assert_f32_parity(actual.as_dynamic(), &expected);

    let ints = Tensor1::<3, i64>::from_vec(vec![1, 2, 3], [3], &ctx)?;
    assert_typed_dynamic_parity::<i64>(
        ints.add(&ints)?.as_dynamic(),
        &ints.as_dynamic().add(ints.as_dynamic())?,
    );
    assert_typed_dynamic_parity::<bool>(
        ints.eq(&ints)?.as_dynamic(),
        &ints.as_dynamic().eq(ints.as_dynamic())?,
    );
    let empty = Tensor1::<0, i64>::from_vec(Vec::new(), [0], &ctx)?;
    assert_typed_dynamic_parity::<i64>(
        empty.add(&empty)?.as_dynamic(),
        &empty.as_dynamic().add(empty.as_dynamic())?,
    );

    let short = Tensor2::<DYN, DYN>::from_vec(vec![1.0; 2], [1, 2], &ctx)?;
    let long = Tensor2::<DYN, DYN>::from_vec(vec![1.0; 4], [2, 2], &ctx)?;
    assert!(long.as_dynamic().add(short.as_dynamic()).is_ok());
    assert!(matches!(
        long.add(&short),
        Err(Error::ShapeMismatch { op: "add", .. })
    ));

    let typed_grads = actual.sum_all()?.backward()?;
    let dynamic_grads = expected.sum_all()?.backward()?;
    assert_grad_parity(
        &typed_grads.wrt_input(&leaf)?,
        &dynamic_grads.wrt_input(&leaf)?,
    );
    Ok(())
}

fn reduction_family() -> Result<()> {
    let ctx = cpu();
    let rank8 = Tensor8::<1, 1, 1, 1, 1, 1, 2, 3, i64>::from_vec(
        vec![1i64, 4, 2, 3, 0, 5],
        [1, 1, 1, 1, 1, 1, 2, 3],
        &ctx,
    )?;
    assert_typed_dynamic_parity::<i64>(rank8.sum::<7>()?.as_dynamic(), &rank8.as_dynamic().sum(7)?);
    assert_typed_dynamic_parity::<i64>(
        rank8.argmax::<6>()?.as_dynamic(),
        &rank8.as_dynamic().argmax(6)?,
    );

    let empty = Tensor2::<2, 0>::from_vec(Vec::<f32>::new(), [2, 0], &ctx)?;
    assert_f32_parity(empty.sum::<1>()?.as_dynamic(), &empty.as_dynamic().sum(1)?);
    assert_error_parity(
        empty.mean::<1>().unwrap_err(),
        empty.as_dynamic().mean(1).unwrap_err(),
    );

    let leaf = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2], &Device::Cpu)?.traced()?;
    let typed = Tensor2::<DYN, 2>::try_from_dynamic(leaf.clone(), &ctx)?.transpose::<0, 1>()?;
    let weights = Tensor2::<2, DYN>::from_vec(vec![0.5, -1.0, 2.0, 0.25], [2, 2], &ctx)?;
    let typed_out = typed.softmax::<1>()?.mul(&weights)?.sum::<0>()?;
    let dynamic_out = leaf
        .transpose(0, 1)?
        .softmax(1)?
        .mul(weights.as_dynamic())?
        .sum(0)?;
    assert_f32_parity(typed_out.as_dynamic(), &dynamic_out);
    assert!(matches!(
        typed.sum_dyn(2),
        Err(Error::InvalidAxis {
            op: "sum_dyn",
            axis: 2,
            rank: 2,
            ..
        })
    ));
    assert!(matches!(
        typed.as_dynamic().sum(2),
        Err(Error::InvalidAxis {
            op: "sum",
            axis: 2,
            rank: 2,
            ..
        })
    ));
    let typed_grad = typed_out.sum_all()?.backward()?.wrt_input(&leaf)?;
    let dynamic_grad = dynamic_out.sum_all()?.backward()?.wrt_input(&leaf)?;
    assert!(
        typed_grad
            .to_vec::<f32>()?
            .iter()
            .any(|value| value.abs() > 1e-4)
    );
    assert_grad_parity(&typed_grad, &dynamic_grad);
    Ok(())
}

fn matmul_family() -> Result<()> {
    let ctx = cpu();
    let high = Tensor8::<1, 1, 1, 1, 1, 2, 2, 3>::from_vec(
        (1..=12).map(|x| x as f32).collect(),
        [1, 1, 1, 1, 1, 2, 2, 3],
        &ctx,
    )?;
    let weight = Tensor2::<3, 2>::from_vec(vec![1.0; 6], [3, 2], &ctx)?;
    let high_out = high.matmul(&weight)?;
    assert_f32_parity(
        high_out.as_dynamic(),
        &high.as_dynamic().matmul(weight.as_dynamic())?,
    );

    let empty_lhs = Tensor2::<2, 0>::from_vec(Vec::<f32>::new(), [2, 0], &ctx)?;
    let empty_rhs = Tensor2::<0, 3>::from_vec(Vec::<f32>::new(), [0, 3], &ctx)?;
    assert_f32_parity(
        empty_lhs.matmul(&empty_rhs)?.as_dynamic(),
        &empty_lhs.as_dynamic().matmul(empty_rhs.as_dynamic())?,
    );

    let bad_lhs = Tensor2::<2, DYN>::from_vec(vec![0.0; 6], [2, 3], &ctx)?;
    let bad_rhs = Tensor2::<DYN, 2>::from_vec(vec![0.0; 8], [4, 2], &ctx)?;
    assert_error_parity(
        bad_lhs.matmul(&bad_rhs).unwrap_err(),
        bad_lhs
            .as_dynamic()
            .matmul(bad_rhs.as_dynamic())
            .unwrap_err(),
    );

    let leaf =
        Tensor::from_vec((1..=6).map(|x| x as f32).collect(), [3, 2], &Device::Cpu)?.traced()?;
    let view = Tensor2::<3, 2>::try_from_dynamic(leaf.clone(), &ctx)?.transpose::<0, 1>()?;
    let typed_out = view.matmul(&weight)?;
    let dynamic_out = leaf.transpose(0, 1)?.matmul(weight.as_dynamic())?;
    assert_grad_parity(
        &typed_out.sum_all()?.backward()?.wrt_input(&leaf)?,
        &dynamic_out.sum_all()?.backward()?.wrt_input(&leaf)?,
    );
    Ok(())
}

fn indexing_family() -> Result<()> {
    let ctx = cpu();
    let leaf = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2], &Device::Cpu)?.traced()?;
    let source = Tensor2::<DYN, 2>::try_from_dynamic(leaf.clone(), &ctx)?.transpose::<0, 1>()?;
    let ids = Tensor1::<2, i64>::from_indices(vec![1, 0], &ctx)?;
    let actual = source.index_select::<1, 2>(&ids)?;
    let expected = leaf.transpose(0, 1)?.index_select(1, ids.as_dynamic())?;
    assert_f32_parity(actual.as_dynamic(), &expected);

    let no_ids = Tensor1::<0, i64>::from_indices(Vec::new(), &ctx)?;
    assert_f32_parity(
        source.index_select::<1, 0>(&no_ids)?.as_dynamic(),
        &source.as_dynamic().index_select(1, no_ids.as_dynamic())?,
    );
    let bad = Tensor1::<1, i64>::from_indices(vec![2], &ctx)?;
    assert_error_parity(
        source.index_select::<1, 1>(&bad).unwrap_err(),
        source
            .as_dynamic()
            .index_select(1, bad.as_dynamic())
            .unwrap_err(),
    );
    assert_grad_parity(
        &actual.sum_all()?.backward()?.wrt_input(&leaf)?,
        &expected.sum_all()?.backward()?.wrt_input(&leaf)?,
    );
    Ok(())
}

fn conv_pool_family() -> Result<()> {
    let ctx = cpu();
    let input =
        Tensor4::<1, 1, 3, 3>::from_vec((0..9).map(|x| x as f32).collect(), [1, 1, 3, 3], &ctx)?;
    let weight = Tensor4::<1, 1, 2, 2>::from_vec(vec![1.0; 4], [1, 1, 2, 2], &ctx)?;
    assert_f32_parity(
        input.conv2d(&weight, (1, 1), (0, 0), (1, 1))?.as_dynamic(),
        &input
            .as_dynamic()
            .conv2d(weight.as_dynamic(), (1, 1), (0, 0), (1, 1))?,
    );
    let ints = Tensor4::<1, 1, 2, 2, i64>::from_vec(vec![1, 2, 3, 4], [1, 1, 2, 2], &ctx)?;
    assert_typed_dynamic_parity::<i64>(
        ints.max_pool2d((2, 2), (1, 1), (0, 0))?.as_dynamic(),
        &ints.as_dynamic().max_pool2d((2, 2), (1, 1), (0, 0))?,
    );

    let empty = Tensor4::<0, 1, 3, 3>::from_vec(Vec::<f32>::new(), [0, 1, 3, 3], &ctx)?;
    assert_f32_parity(
        empty.avg_pool2d((2, 2), (1, 1), (0, 0))?.as_dynamic(),
        &empty.as_dynamic().avg_pool2d((2, 2), (1, 1), (0, 0))?,
    );
    let dyn_input = Tensor4::<1, DYN, 3, 3>::from_vec(vec![0.0; 9], [1, 1, 3, 3], &ctx)?;
    let dyn_weight = Tensor4::<1, DYN, 2, 2>::from_vec(vec![0.0; 8], [1, 2, 2, 2], &ctx)?;
    assert_error_parity(
        dyn_input
            .conv2d(&dyn_weight, (1, 1), (0, 0), (1, 1))
            .unwrap_err(),
        dyn_input
            .as_dynamic()
            .conv2d(dyn_weight.as_dynamic(), (1, 1), (0, 0), (1, 1))
            .unwrap_err(),
    );

    let leaf = Tensor::from_vec(
        (0..9).map(|x| x as f32).collect(),
        [1, 1, 3, 3],
        &Device::Cpu,
    )?
    .traced()?;
    let typed_leaf = Tensor4::<1, 1, 3, 3>::try_from_dynamic(leaf.clone(), &ctx)?;
    let typed_out = typed_leaf.avg_pool2d((2, 2), (1, 1), (0, 0))?;
    let dynamic_out = leaf.avg_pool2d((2, 2), (1, 1), (0, 0))?;
    assert_grad_parity(
        &typed_out.sum_all()?.backward()?.wrt_input(&leaf)?,
        &dynamic_out.sum_all()?.backward()?.wrt_input(&leaf)?,
    );
    Ok(())
}

fn loss_family() -> Result<()> {
    let ctx = cpu();
    let leaf = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2], &Device::Cpu)?.traced()?;
    let prediction =
        Tensor2::<DYN, 2>::try_from_dynamic(leaf.clone(), &ctx)?.transpose::<0, 1>()?;
    let target = Tensor2::<2, DYN>::from_vec(vec![0.0; 4], [2, 2], &ctx)?;
    let actual = prediction.mse_loss(&target)?;
    let expected = leaf.transpose(0, 1)?.mse_loss(target.as_dynamic())?;
    assert_f32_parity(actual.as_dynamic(), &expected);

    let empty_logits = Tensor2::<0, 3>::from_vec(Vec::<f32>::new(), [0, 3], &ctx)?;
    let empty_labels = Tensor1::<0, i64>::from_indices(Vec::new(), &ctx)?;
    assert_f32_parity(
        empty_logits.cross_entropy(&empty_labels)?.as_dynamic(),
        &empty_logits
            .as_dynamic()
            .cross_entropy(empty_labels.as_dynamic())?,
    );
    let logits = Tensor2::<DYN, 3>::from_vec(vec![0.0; 6], [2, 3], &ctx)?;
    let labels = Tensor1::<DYN, i64>::from_indices(vec![0], &ctx)?;
    assert_error_parity(
        logits.cross_entropy(&labels).unwrap_err(),
        logits
            .as_dynamic()
            .cross_entropy(labels.as_dynamic())
            .unwrap_err(),
    );
    assert_grad_parity(
        &actual.backward()?.wrt_input(&leaf)?,
        &expected.backward()?.wrt_input(&leaf)?,
    );
    Ok(())
}

fn core_autograd_family() -> Result<()> {
    let ctx = cpu();
    let leaf =
        Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], &Device::Cpu)?.traced()?;
    let typed = Tensor2::<DYN, DYN>::try_from_dynamic(leaf.clone(), &ctx)?;
    let refined = typed.refine::<Tensor2<2, 3>>()?;
    let erased = refined.erase_shape()?;
    let roundtrip = Tensor2::<DYN, DYN>::try_from_dynamic(erased.into_dynamic(), &ctx)?;
    let cast: Tensor2<DYN, DYN, i64> = roundtrip.to_dtype()?;
    assert_typed_dynamic_parity::<i64>(cast.as_dynamic(), &leaf.to_dtype(DType::I64)?);

    let view = roundtrip.transpose::<0, 1>()?;
    // Contiguity has no public predicate; the `Debug` field is the affordance.
    assert!(format!("{:?}", view.as_dynamic()).contains("contiguous: false"));
    let typed_output = view.square()?.sum_all()?;
    let dynamic_view = leaf.transpose(0, 1)?;
    let dynamic_output = dynamic_view.mul(&dynamic_view)?.sum_all()?;
    let typed_grads = typed_output.backward()?;
    let dynamic_grads = dynamic_output.backward()?;
    assert_grad_parity(
        &typed_grads.wrt_input(&leaf)?,
        &dynamic_grads.wrt_input(&leaf)?,
    );
    let typed_lookup: Tensor2<DYN, DYN> = typed_grads.wrt_typed_input(&roundtrip)?;
    assert_grad_parity(typed_lookup.as_dynamic(), &typed_grads.wrt_input(&leaf)?);

    let empty = Tensor2::<0, DYN>::from_vec(Vec::<f32>::new(), [0, 3], &ctx)?;
    assert_eq!(empty.contiguous()?.dims(), [0, 3]);
    assert_error_parity(
        roundtrip.to_scalar().unwrap_err(),
        leaf.to_scalar::<f32>().unwrap_err(),
    );
    Ok(())
}

fn f64_capabilities() -> Result<()> {
    let ctx = cpu();
    let lhs = Tensor2::<DYN, 2, f64>::from_vec(vec![1.0, -2.0, 3.0, 4.0], [2, 2], &ctx)?;
    let rhs = Tensor2::<DYN, 2, f64>::from_vec(vec![0.5, 1.0, -1.0, 2.0], [2, 2], &ctx)?;
    let typed_numeric = lhs.add(&rhs)?.tanh()?.mean::<1>()?;
    let dynamic_numeric = lhs.as_dynamic().add(rhs.as_dynamic())?.tanh()?.mean(1)?;
    assert_typed_dynamic_parity::<f64>(typed_numeric.as_dynamic(), &dynamic_numeric);

    let typed_matmul = lhs.matmul(&rhs)?;
    let dynamic_matmul = lhs.as_dynamic().matmul(rhs.as_dynamic())?;
    assert_typed_dynamic_parity::<f64>(typed_matmul.as_dynamic(), &dynamic_matmul);

    let typed_loss = lhs.mse_loss(&rhs)?;
    let dynamic_loss = lhs.as_dynamic().mse_loss(rhs.as_dynamic())?;
    assert_typed_dynamic_parity::<f64>(typed_loss.as_dynamic(), &dynamic_loss);

    let leaf = Tensor::from_vec(vec![1.0f64, -2.0, 3.0, 4.0], [2, 2], &Device::Cpu)?.traced()?;
    let typed = Tensor2::<DYN, 2, f64>::try_from_dynamic(leaf.clone(), &ctx)?;
    let typed_grad = typed.square()?.sum_all()?.backward()?.wrt_input(&leaf)?;
    let dynamic_grad = leaf.mul(&leaf)?.sum_all()?.backward()?.wrt_input(&leaf)?;
    assert_reduced_grad_parity!(f64, typed_grad, dynamic_grad, |value: f64| value, 1e-12);
    Ok(())
}

fn f16_capabilities() -> Result<()> {
    let ctx = cpu();
    let one = half::f16::from_f32(1.0);
    let two = half::f16::from_f32(2.0);
    let value = Tensor2::<DYN, 2, half::f16>::from_vec(vec![one, two, two, one], [2, 2], &ctx)?;
    let typed_add = value.add(&value)?;
    let dynamic_add = value.as_dynamic().add(value.as_dynamic())?;
    assert_typed_dynamic_parity::<half::f16>(typed_add.as_dynamic(), &dynamic_add);

    let typed_sum = value.sum::<1>()?;
    let dynamic_sum = value.as_dynamic().sum(1)?;
    assert_typed_dynamic_parity::<half::f16>(typed_sum.as_dynamic(), &dynamic_sum);

    let typed_matmul = value.matmul(&value)?;
    let dynamic_matmul = value.as_dynamic().matmul(value.as_dynamic())?;
    assert_typed_dynamic_parity::<half::f16>(typed_matmul.as_dynamic(), &dynamic_matmul);

    let leaf = Tensor::from_vec(vec![one, two, two, one], [2, 2], &Device::Cpu)?.traced()?;
    let typed = Tensor2::<DYN, 2, half::f16>::try_from_dynamic(leaf.clone(), &ctx)?;
    let typed_grad = typed.square()?.sum_all()?.backward()?.wrt_input(&leaf)?;
    let dynamic_grad = leaf.mul(&leaf)?.sum_all()?.backward()?.wrt_input(&leaf)?;
    assert_reduced_grad_parity!(
        half::f16,
        typed_grad,
        dynamic_grad,
        |value: half::f16| value.to_f64(),
        1e-3
    );
    Ok(())
}

fn bf16_capabilities() -> Result<()> {
    let ctx = cpu();
    let one = half::bf16::from_f32(1.0);
    let two = half::bf16::from_f32(2.0);
    let value = Tensor2::<DYN, 2, half::bf16>::from_vec(vec![one, two, two, one], [2, 2], &ctx)?;
    let typed_scaled = value.mul_scalar(2.0)?;
    let dynamic_scaled = value.as_dynamic().mul_scalar(2.0)?;
    assert_typed_dynamic_parity::<half::bf16>(typed_scaled.as_dynamic(), &dynamic_scaled);

    let typed_mean = value.mean::<0>()?;
    let dynamic_mean = value.as_dynamic().mean(0)?;
    assert_typed_dynamic_parity::<half::bf16>(typed_mean.as_dynamic(), &dynamic_mean);

    let typed_loss = value.mse_loss(&value)?;
    let dynamic_loss = value.as_dynamic().mse_loss(value.as_dynamic())?;
    assert_typed_dynamic_parity::<half::bf16>(typed_loss.as_dynamic(), &dynamic_loss);

    let leaf = Tensor::from_vec(vec![one, two, two, one], [2, 2], &Device::Cpu)?.traced()?;
    let typed = Tensor2::<DYN, 2, half::bf16>::try_from_dynamic(leaf.clone(), &ctx)?;
    let typed_grad = typed.square()?.sum_all()?.backward()?.wrt_input(&leaf)?;
    let dynamic_grad = leaf.mul(&leaf)?.sum_all()?.backward()?.wrt_input(&leaf)?;
    assert_reduced_grad_parity!(
        half::bf16,
        typed_grad,
        dynamic_grad,
        |value: half::bf16| value.to_f64(),
        1e-2
    );
    Ok(())
}

fn i64_capabilities() -> Result<()> {
    let ctx = cpu();
    let matrix = Tensor2::<DYN, 2, i64>::from_vec(vec![1, 2, 3, 4], [2, 2], &ctx)?;
    let ids = Tensor1::<2, i64>::from_indices(vec![1, 0], &ctx)?;
    let typed_matmul = matrix.matmul(&matrix)?;
    let dynamic_matmul = matrix.as_dynamic().matmul(matrix.as_dynamic())?;
    assert_typed_dynamic_parity::<i64>(typed_matmul.as_dynamic(), &dynamic_matmul);

    let typed_selected = matrix.index_select::<0, 2>(&ids)?;
    let dynamic_selected = matrix.as_dynamic().index_select(0, ids.as_dynamic())?;
    assert_typed_dynamic_parity::<i64>(typed_selected.as_dynamic(), &dynamic_selected);

    let typed_argmax = matrix.argmax::<1>()?;
    let dynamic_argmax = matrix.as_dynamic().argmax(1)?;
    assert_typed_dynamic_parity::<i64>(typed_argmax.as_dynamic(), &dynamic_argmax);

    assert!(matches!(
        matrix.as_dynamic().tanh(),
        Err(Error::Unsupported { op: "tanh", .. })
    ));
    assert!(matches!(
        matrix.as_dynamic().mse_loss(matrix.as_dynamic()),
        Err(Error::Unsupported { op: "mse_loss", .. })
    ));
    Ok(())
}

fn bool_capabilities() -> Result<()> {
    let ctx = cpu();
    let mask = Tensor2::<DYN, 2, bool>::from_vec(vec![true, false, false, true], [2, 2], &ctx)?;
    let on_true = Tensor2::<DYN, 2, f64>::from_vec(vec![1.0; 4], [2, 2], &ctx)?;
    let on_false = Tensor2::<DYN, 2, f64>::from_vec(vec![-1.0; 4], [2, 2], &ctx)?;
    let typed_selected = mask.where_cond(&on_true, &on_false)?;
    let dynamic_selected = mask
        .as_dynamic()
        .where_cond(on_true.as_dynamic(), on_false.as_dynamic())?;
    assert_typed_dynamic_parity::<f64>(typed_selected.as_dynamic(), &dynamic_selected);

    let typed_transpose = mask.transpose::<0, 1>()?;
    let dynamic_transpose = mask.as_dynamic().transpose(0, 1)?;
    assert_typed_dynamic_parity::<bool>(typed_transpose.as_dynamic(), &dynamic_transpose);

    let empty = Tensor1::<0, bool>::from_vec(Vec::new(), [0], &ctx)?;
    let typed_equal = empty.eq(&empty)?;
    let dynamic_equal = empty.as_dynamic().eq(empty.as_dynamic())?;
    assert_typed_dynamic_parity::<bool>(typed_equal.as_dynamic(), &dynamic_equal);

    assert!(matches!(
        mask.as_dynamic().add(mask.as_dynamic()),
        Err(Error::Unsupported { op: "add", .. })
    ));
    assert!(matches!(
        mask.as_dynamic().sum(0),
        Err(Error::Unsupported { op: "sum", .. })
    ));
    assert!(matches!(
        mask.as_dynamic().matmul(mask.as_dynamic()),
        Err(Error::Unsupported { op: "matmul", .. })
    ));
    Ok(())
}

#[test]
fn strict_policy_rejections_are_separate_from_runtime_parity() -> Result<()> {
    let ctx = cpu();
    let matrix = Tensor2::<DYN, DYN>::from_vec(vec![1.0f32; 6], [2, 3], &ctx)?;
    let row = Tensor2::<DYN, DYN>::from_vec(vec![2.0f32; 3], [1, 3], &ctx)?;
    assert!(matrix.as_dynamic().add(row.as_dynamic()).is_ok());
    assert!(matches!(
        matrix.add(&row),
        Err(Error::ShapeMismatch { op: "add", .. })
    ));

    let lhs = Tensor3::<DYN, 2, 3>::from_vec(vec![1.0; 12], [2, 2, 3], &ctx)?;
    let rhs = Tensor3::<DYN, 3, 2>::from_vec(vec![1.0; 6], [1, 3, 2], &ctx)?;
    assert!(lhs.as_dynamic().matmul(rhs.as_dynamic()).is_ok());
    assert!(matches!(
        lhs.matmul(&rhs),
        Err(Error::ShapeMismatch { op: "matmul", .. })
    ));
    Ok(())
}

#[test]
fn typed_sources_do_not_import_private_execution_layers() {
    fn inspect(path: &Path) {
        for entry in fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                inspect(&path);
            } else if path.extension().and_then(|value| value.to_str()) == Some("rs") {
                let source = fs::read_to_string(&path).unwrap();
                assert!(
                    !has_private_execution_path(&source),
                    "{} imports a private execution layer",
                    path.display()
                );
            }
        }
    }

    for rejected in [
        "use crate::{ backend :: CpuKernel, Tensor };",
        "use crate :: layout :: Layout;",
        "fn f() { crate::storage::Storage::new(); }",
        "use super::super::{autograd::Node};",
        "use crate::{Error, tensor::PrivateTensor};",
        "use crate as root; use root::backend::CpuKernel;",
        "use super as root; use root::storage::Storage;",
        "use super::super as root; use root::autograd::Node;",
        "extern crate self as rstorch; use rstorch::backend::CpuKernel;",
        "use rstorch as root; use root as chained; use chained::layout::Layout;",
        "use rstorch::{Error, storage::Storage};",
        "use crate::backend;",
        "use rstorch::storage as execution;",
        "use rstorch::{layout as execution};",
        "macro_rules! leak { () => { crate::backend::CpuKernel } }",
        "macro_rules! leak { () => { crate::{layout::Layout} } }",
        "macro_rules! leak { ($root:path) => { $root::storage::Storage } }",
        "macro_rules! leak { () => { super::super::autograd::Node } }",
        "macro_rules! leak { () => { crate::tensor::PrivateTensor } }",
        "macro_rules! private_path { ($module:ident) => { crate::$module::PrivateTensor } } private_path!(tensor);",
        "use crate::Tensor as tensor;",
    ] {
        assert!(
            has_private_execution_path(rejected),
            "scanner accepted `{rejected}`"
        );
    }
    for accepted in [
        "mod tensor;",
        "pub(in crate::typed) mod tensor;",
        "use crate::typed::tensor::checked_wrap;",
        "use super::tensor::checked_wrap;",
        "fn wrap() { super::super::tensor::checked_wrap(); }",
        "// use rstorch::backend::Kernel;\nconst S: &str = \"root::storage::Storage\";",
    ] {
        assert!(
            !has_private_execution_path(accepted),
            "scanner rejected `{accepted}`"
        );
    }

    inspect(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src/typed"));
}

fn has_private_execution_path(source: &str) -> bool {
    let code = strip_comments_and_strings(source);
    let compact = code
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    let code_tokens = lexical_words(&code);
    if ["backend", "layout", "storage"]
        .iter()
        .any(|module| code_tokens.contains(module))
    {
        return true;
    }
    let normalized = compact
        .replace("modautograd;", "mod__typed_internal;")
        .replace(
            "pubuseautograd::TypedGradsExt;",
            "pubuse__typed_internal::TypedGradsExt;",
        )
        .replace(
            "pub(incrate::typed)modtensor;",
            "pub(incrate::typed)mod__typed_internal;",
        )
        .replace("modtensor;", "mod__typed_internal;")
        .replace("crate::typed::tensor::", "crate::typed::__typed_internal::")
        .replace(
            "crate::typed::{tensor::",
            "crate::typed::{__typed_internal::",
        )
        .replace("super::super::tensor::", "super::super::__typed_internal::")
        .replace("super::tensor::", "super::__typed_internal::")
        .replace("super::{tensor::", "super::{__typed_internal::");

    let words = lexical_words(&normalized);
    if words.contains(&"autograd")
        || normalized.contains("tensor::")
        || normalized
            .split(';')
            .any(|statement| statement.starts_with("use") && statement.contains("tensor"))
        || macro_invocation_contains_token(&normalized, "tensor")
    {
        return true;
    }
    false
}

fn macro_invocation_contains_token(source: &str, rejected: &str) -> bool {
    let bytes = source.as_bytes();
    let mut index = 0;
    while index + 1 < bytes.len() {
        if bytes[index] != b'!' || !matches!(bytes[index + 1], b'(' | b'[' | b'{') {
            index += 1;
            continue;
        }

        let open = bytes[index + 1];
        let close = match open {
            b'(' => b')',
            b'[' => b']',
            b'{' => b'}',
            _ => unreachable!(),
        };
        let start = index + 2;
        let mut cursor = start;
        let mut depth = 1usize;
        while cursor < bytes.len() && depth > 0 {
            if bytes[cursor] == open {
                depth += 1;
            } else if bytes[cursor] == close {
                depth -= 1;
            }
            cursor += 1;
        }
        if depth == 0 && lexical_words(&source[start..cursor - 1]).contains(&rejected) {
            return true;
        }
        index = cursor;
    }
    false
}

fn lexical_words(source: &str) -> Vec<&str> {
    source
        .split(|ch: char| !(ch.is_ascii_alphanumeric() || ch == '_'))
        .filter(|word| !word.is_empty())
        .collect()
}

fn strip_comments_and_strings(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut clean = String::with_capacity(bytes.len());
    let mut index = 0;
    let mut block_depth = 0usize;
    while index < bytes.len() {
        if block_depth > 0 {
            if bytes[index..].starts_with(b"/*") {
                block_depth += 1;
                clean.push_str("  ");
                index += 2;
            } else if bytes[index..].starts_with(b"*/") {
                block_depth -= 1;
                clean.push_str("  ");
                index += 2;
            } else {
                clean.push(if bytes[index] == b'\n' { '\n' } else { ' ' });
                index += 1;
            }
        } else if bytes[index..].starts_with(b"//") {
            while index < bytes.len() && bytes[index] != b'\n' {
                clean.push(' ');
                index += 1;
            }
        } else if bytes[index..].starts_with(b"/*") {
            block_depth = 1;
            clean.push_str("  ");
            index += 2;
        } else if bytes[index] == b'"' {
            clean.push(' ');
            index += 1;
            while index < bytes.len() {
                let byte = bytes[index];
                clean.push(if byte == b'\n' { '\n' } else { ' ' });
                index += 1;
                if byte == b'\\' && index < bytes.len() {
                    clean.push(' ');
                    index += 1;
                } else if byte == b'"' {
                    break;
                }
            }
        } else {
            clean.push(bytes[index] as char);
            index += 1;
        }
    }
    clean
}
