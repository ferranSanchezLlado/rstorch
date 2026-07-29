#![cfg(feature = "typed")]

use rstorch::typed::prelude::*;
use rstorch::{Device, Result, Tensor};

#[test]
fn downstream_typed_tensor_workflow_uses_only_public_api() -> Result<()> {
    let ctx = DeviceCtx::<Cpu>::cpu()?;
    let dynamic = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2], &Device::Cpu)?.traced()?;
    let input = Tensor2::<DYN, 2>::try_from_dynamic(dynamic.clone(), &ctx)?;
    let weight = Tensor2::<2, 1>::from_vec(vec![0.5, 1.0], [2, 1], &ctx)?;
    let output = input.matmul(&weight)?.relu()?.sum_all()?;
    let grads = output.backward()?;
    let typed_grad: Tensor2<DYN, 2> = grads.wrt_typed_input(&input)?;

    assert_eq!(output.to_scalar()?, 8.0);
    assert_eq!(typed_grad.to_vec()?, vec![0.5, 1.0, 0.5, 1.0]);
    assert_eq!(
        grads.wrt_input(&dynamic)?.to_vec::<f32>()?,
        vec![0.5, 1.0, 0.5, 1.0]
    );
    Ok(())
}

#[test]
fn downstream_static_dynamic_empty_and_view_paths_compile() -> Result<()> {
    let ctx = DeviceCtx::<Cpu>::cpu()?;
    let row = Tensor2::<1, 3, i64>::from_vec(vec![1, 2, 3], [1, 3], &ctx)?;
    let expanded = row.broadcast_to::<Tensor2<2, 3, i64>>([2, 3])?;
    let view = expanded.transpose::<0, 1>()?;
    let empty = Tensor2::<0, DYN, i64>::from_vec(Vec::new(), [0, 3], &ctx)?;

    assert_eq!(view.dims(), [3, 2]);
    assert_eq!(view.sum::<1>()?.to_vec()?, vec![2, 4, 6]);
    assert_eq!(empty.sum::<1>()?.to_vec()?, Vec::<i64>::new());
    Ok(())
}
