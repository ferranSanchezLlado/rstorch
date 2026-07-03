use std::fmt::Debug;

use rstorch::{
    Backend, C, Conv2dOptions, Cpu, D1, D2, DType, DTypeId, FloatDType, Padding2d, Pool2dOptions,
    Sym, Tensor, Tensor1D, Tensor2D, Tensor4D, bf16, f16,
};

struct Batch;

fn values<E: FloatDType>(data: &[f64]) -> Vec<E> {
    data.iter().map(|&value| E::from_f64(value)).collect()
}

fn assert_vec_eq<E>(actual: Vec<E>, expected: &[f64])
where
    E: FloatDType + Debug,
{
    assert_eq!(actual, values(expected));
}

fn parity_for_backend<E, B>()
where
    E: FloatDType + Debug,
    B: Backend<E>,
{
    let zeros = Tensor2D::<2, 3, E, B>::zeros().unwrap();
    assert_eq!(zeros.dtype(), E::ID);
    assert_vec_eq(zeros.to_vec().unwrap(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

    let ones = Tensor2D::<2, 3, E, B>::ones().unwrap();
    assert_vec_eq(ones.to_vec().unwrap(), &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

    let lhs = Tensor2D::<2, 2, E, B>::from_vec(values(&[8.0, 9.0, 10.0, 12.0])).unwrap();
    let rhs = Tensor2D::<2, 2, E, B>::from_vec(values(&[2.0, 3.0, 5.0, 6.0])).unwrap();
    assert_vec_eq(
        lhs.add(&rhs).unwrap().to_vec().unwrap(),
        &[10.0, 12.0, 15.0, 18.0],
    );
    assert_vec_eq(
        lhs.sub(&rhs).unwrap().to_vec().unwrap(),
        &[6.0, 6.0, 5.0, 6.0],
    );
    assert_vec_eq(
        lhs.mul(&rhs).unwrap().to_vec().unwrap(),
        &[16.0, 27.0, 50.0, 72.0],
    );
    assert_vec_eq(
        lhs.div(&rhs).unwrap().to_vec().unwrap(),
        &[4.0, 3.0, 2.0, 2.0],
    );

    let scalar = Tensor1D::<4, E, B>::from_vec(values(&[2.0, 4.0, 6.0, 8.0])).unwrap();
    assert_vec_eq(
        scalar
            .add_scalar(E::from_f64(1.0))
            .unwrap()
            .to_vec()
            .unwrap(),
        &[3.0, 5.0, 7.0, 9.0],
    );
    assert_vec_eq(
        scalar
            .sub_scalar(E::from_f64(1.0))
            .unwrap()
            .to_vec()
            .unwrap(),
        &[1.0, 3.0, 5.0, 7.0],
    );
    assert_vec_eq(
        scalar
            .mul_scalar(E::from_f64(0.5))
            .unwrap()
            .to_vec()
            .unwrap(),
        &[1.0, 2.0, 3.0, 4.0],
    );
    assert_vec_eq(
        scalar
            .div_scalar(E::from_f64(2.0))
            .unwrap()
            .to_vec()
            .unwrap(),
        &[1.0, 2.0, 3.0, 4.0],
    );

    let mat_lhs =
        Tensor2D::<2, 3, E, B>::from_vec(values(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])).unwrap();
    let mat_rhs =
        Tensor2D::<3, 2, E, B>::from_vec(values(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0])).unwrap();
    assert_vec_eq(
        mat_lhs.matmul(&mat_rhs).unwrap().to_vec().unwrap(),
        &[58.0, 64.0, 139.0, 154.0],
    );
    assert_vec_eq(mat_lhs.sum().unwrap().to_vec().unwrap(), &[21.0]);

    let transposed = mat_lhs.transpose().unwrap();
    assert_eq!(transposed.shape().dims(), &[3, 2]);
    assert_vec_eq(
        transposed.to_vec().unwrap(),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
    );

    let reshaped = mat_lhs
        .reshape_with_shape::<D2<Sym<Batch>, C<2>>>([3, 2])
        .unwrap();
    assert_eq!(reshaped.shape().dims(), &[3, 2]);
    assert_vec_eq(reshaped.to_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let flattened = mat_lhs.flatten::<6>().unwrap();
    assert_eq!(flattened.shape().dims(), &[6]);
    assert_vec_eq(flattened.to_vec().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let cat_lhs = Tensor::<D1<C<2>>, E, B>::from_vec(values(&[1.0, 2.0])).unwrap();
    let cat_rhs = Tensor1D::<3, E, B>::from_vec(values(&[3.0, 4.0, 5.0])).unwrap();
    assert_vec_eq(
        cat_lhs.cat::<C<3>, 5>(&cat_rhs).unwrap().to_vec().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0],
    );

    let row = Tensor1D::<3, E, B>::from_vec(values(&[10.0, 20.0, 30.0])).unwrap();
    assert_vec_eq(
        mat_lhs.add_last_dim(&row).unwrap().to_vec().unwrap(),
        &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0],
    );

    let relu = Tensor1D::<4, E, B>::from_vec(values(&[-2.0, 0.0, 3.0, -4.0])).unwrap();
    assert_vec_eq(
        relu.relu().unwrap().to_vec().unwrap(),
        &[0.0, 0.0, 3.0, 0.0],
    );

    let image = Tensor4D::<1, 1, 2, 2, E, B>::from_vec(values(&[1.0, 2.0, 3.0, 4.0])).unwrap();
    assert_vec_eq(
        image
            .pad2d::<4, 4>(Padding2d::new(1, 1))
            .unwrap()
            .to_vec()
            .unwrap(),
        &[
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    );
    let weight = Tensor4D::<1, 1, 2, 2, E, B>::from_vec(values(&[1.0, 0.0, 0.0, 1.0])).unwrap();
    assert_vec_eq(
        image
            .conv2d::<C<1>, C<2>, C<2>, 1, 1>(&weight, Conv2dOptions::default())
            .unwrap()
            .to_vec()
            .unwrap(),
        &[5.0],
    );
    let bias = Tensor1D::<1, E, B>::from_vec(values(&[10.0])).unwrap();
    assert_vec_eq(
        image.add_channel_dim(&bias).unwrap().to_vec().unwrap(),
        &[11.0, 12.0, 13.0, 14.0],
    );
    assert_vec_eq(
        image.flatten_spatial::<4>().unwrap().to_vec().unwrap(),
        &[1.0, 2.0, 3.0, 4.0],
    );
    assert_vec_eq(
        image
            .max_pool2d::<1, 1>(Pool2dOptions::new(2, 2))
            .unwrap()
            .to_vec()
            .unwrap(),
        &[4.0],
    );
    assert_vec_eq(
        image
            .avg_pool2d::<1, 1>(Pool2dOptions::new(2, 2))
            .unwrap()
            .to_vec()
            .unwrap(),
        &[2.5],
    );
}

fn autograd_parity_for_backend<E, B>()
where
    E: FloatDType + Debug,
    B: Backend<E>,
{
    let x = Tensor1D::<2, E, B>::from_vec(values(&[2.0, 3.0]))
        .unwrap()
        .with_requires_grad(true);
    x.mul(&x).unwrap().sum().unwrap().backward().unwrap();
    assert_vec_eq(x.grad().unwrap().to_vec().unwrap(), &[4.0, 6.0]);

    let lhs = Tensor2D::<2, 2, E, B>::from_vec(values(&[1.0, 2.0, 3.0, 4.0]))
        .unwrap()
        .with_requires_grad(true);
    let rhs = Tensor2D::<2, 2, E, B>::from_vec(values(&[5.0, 6.0, 7.0, 8.0]))
        .unwrap()
        .with_requires_grad(true);
    lhs.matmul(&rhs).unwrap().sum().unwrap().backward().unwrap();
    assert_vec_eq(
        lhs.grad().unwrap().to_vec().unwrap(),
        &[11.0, 15.0, 11.0, 15.0],
    );
    assert_vec_eq(rhs.grad().unwrap().to_vec().unwrap(), &[4.0, 4.0, 6.0, 6.0]);

    let image = Tensor4D::<1, 1, 2, 2, E, B>::from_vec(values(&[1.0, 2.0, 3.0, 4.0]))
        .unwrap()
        .with_requires_grad(true);
    let weight = Tensor4D::<1, 1, 2, 2, E, B>::from_vec(values(&[1.0, 1.0, 1.0, 1.0]))
        .unwrap()
        .with_requires_grad(true);
    image
        .conv2d::<C<1>, C<2>, C<2>, 1, 1>(&weight, Conv2dOptions::default())
        .unwrap()
        .sum()
        .unwrap()
        .backward()
        .unwrap();
    assert_vec_eq(
        image.grad().unwrap().to_vec().unwrap(),
        &[1.0, 1.0, 1.0, 1.0],
    );
    assert_vec_eq(
        weight.grad().unwrap().to_vec().unwrap(),
        &[1.0, 2.0, 3.0, 4.0],
    );
}

#[test]
fn dtype_ids_match_public_types() {
    assert_eq!(<f16 as DType>::ID, DTypeId::F16);
    assert_eq!(<bf16 as DType>::ID, DTypeId::BF16);
    assert_eq!(<f32 as DType>::ID, DTypeId::F32);
    assert_eq!(<f64 as DType>::ID, DTypeId::F64);
}

#[test]
fn cpu_f16_parity() {
    parity_for_backend::<f16, Cpu>();
}

#[test]
fn cpu_bf16_parity() {
    parity_for_backend::<bf16, Cpu>();
}

#[test]
fn cpu_f32_parity() {
    parity_for_backend::<f32, Cpu>();
}

#[test]
fn cpu_f64_parity() {
    parity_for_backend::<f64, Cpu>();
}

#[test]
fn to_and_cast_convert_dtype_and_detach_from_autograd() {
    let source = Tensor1D::<4, f32, Cpu>::from_vec(vec![1.0, 2.0, 3.0, 4.0])
        .unwrap()
        .with_requires_grad(true);

    let half: Tensor1D<4, f16, Cpu> = source.cast::<f16>().unwrap();
    assert_eq!(half.dtype(), DTypeId::F16);
    assert_eq!(half.to_vec().unwrap(), values::<f16>(&[1.0, 2.0, 3.0, 4.0]));
    assert!(!half.requires_grad());

    let bfloat: Tensor1D<4, bf16, Cpu> = half.cast::<bf16>().unwrap();
    assert_eq!(bfloat.dtype(), DTypeId::BF16);
    assert_eq!(
        bfloat.to_vec().unwrap(),
        values::<bf16>(&[1.0, 2.0, 3.0, 4.0])
    );
    assert!(!bfloat.requires_grad());
}

#[test]
fn cast_and_to_backend_convert_dtype_or_backend_and_detach() {
    let source = Tensor1D::<4, f32, Cpu>::from_vec(vec![1.0, 2.0, 3.0, 4.0])
        .unwrap()
        .with_requires_grad(true);

    let same_dtype: Tensor1D<4, f32, Cpu> = source.to_backend::<Cpu>().unwrap();
    assert_eq!(same_dtype.dtype(), DTypeId::F32);
    assert_eq!(same_dtype.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    assert!(!same_dtype.requires_grad());

    let changed_dtype: Tensor1D<4, f64, Cpu> = source.cast::<f64>().unwrap();
    assert_eq!(changed_dtype.dtype(), DTypeId::F64);
    assert_eq!(changed_dtype.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    assert!(!changed_dtype.requires_grad());
}

#[test]
fn cpu_f32_autograd_parity() {
    autograd_parity_for_backend::<f32, Cpu>();
}

#[test]
fn cpu_f16_autograd_parity() {
    autograd_parity_for_backend::<f16, Cpu>();
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
fn cuda_f32_matches_cpu_supported_ops() {
    use rstorch::Cuda;

    if <Cuda as Backend<f32>>::default_device().is_err() {
        return;
    }

    parity_for_backend::<f32, Cuda>();
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
fn cuda_f16_matches_cpu_supported_ops() {
    use rstorch::Cuda;

    if <Cuda as Backend<f16>>::default_device().is_err() {
        return;
    }

    parity_for_backend::<f16, Cuda>();
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
fn cuda_bf16_matches_cpu_supported_ops() {
    use rstorch::Cuda;

    if <Cuda as Backend<bf16>>::default_device().is_err() {
        return;
    }

    parity_for_backend::<bf16, Cuda>();
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
fn cuda_f64_matches_cpu_supported_ops() {
    use rstorch::Cuda;

    if <Cuda as Backend<f64>>::default_device().is_err() {
        return;
    }

    parity_for_backend::<f64, Cuda>();
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
fn cuda_f32_autograd_matches_cpu_supported_ops() {
    use rstorch::Cuda;

    if <Cuda as Backend<f32>>::default_device().is_err() {
        return;
    }

    autograd_parity_for_backend::<f32, Cuda>();
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
fn cuda_f16_autograd_matches_cpu_supported_ops() {
    use rstorch::Cuda;

    if <Cuda as Backend<f16>>::default_device().is_err() {
        return;
    }

    autograd_parity_for_backend::<f16, Cuda>();
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
fn cuda_bf16_autograd_matches_cpu_supported_ops() {
    use rstorch::Cuda;

    if <Cuda as Backend<bf16>>::default_device().is_err() {
        return;
    }

    autograd_parity_for_backend::<bf16, Cuda>();
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
fn cuda_f64_autograd_matches_cpu_supported_ops() {
    use rstorch::Cuda;

    if <Cuda as Backend<f64>>::default_device().is_err() {
        return;
    }

    autograd_parity_for_backend::<f64, Cuda>();
}

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
#[test]
fn cuda_transfers_dtype_and_device() {
    use rstorch::Cuda;

    if <Cuda as Backend<f32>>::default_device().is_err() {
        return;
    }

    let cpu = Tensor1D::<4, f32, Cpu>::from_vec(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let cuda: Tensor1D<4, f32, Cuda> = cpu.to_backend::<Cuda>().unwrap();
    assert_eq!(cuda.dtype(), DTypeId::F32);
    assert_eq!(cuda.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

    let back: Tensor1D<4, f32, Cpu> = cuda.to_backend::<Cpu>().unwrap();
    assert_eq!(back.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);

    let half: Tensor1D<4, f16, Cuda> = cpu.cast::<f16>().unwrap().to_backend::<Cuda>().unwrap();
    assert_eq!(half.dtype(), DTypeId::F16);
    assert_eq!(half.to_vec().unwrap(), values::<f16>(&[1.0, 2.0, 3.0, 4.0]));

    let bfloat: Tensor1D<4, bf16, Cuda> = half.cast::<bf16>().unwrap();
    assert_eq!(bfloat.dtype(), DTypeId::BF16);
    assert_eq!(
        bfloat.to_vec().unwrap(),
        values::<bf16>(&[1.0, 2.0, 3.0, 4.0])
    );

    let double: Tensor1D<4, f64, Cuda> = bfloat.cast::<f64>().unwrap();
    assert_eq!(double.dtype(), DTypeId::F64);
    assert_eq!(double.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_f32_matches_cpu_supported_ops() {
    use rstorch::Metal;

    if <Metal as Backend<f32>>::default_device().is_err() {
        return;
    }

    parity_for_backend::<f32, Metal>();
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_f16_matches_cpu_supported_ops() {
    use rstorch::Metal;

    if <Metal as Backend<f16>>::default_device().is_err() {
        return;
    }

    parity_for_backend::<f16, Metal>();
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_f32_autograd_matches_cpu_supported_ops() {
    use rstorch::Metal;

    if <Metal as Backend<f32>>::default_device().is_err() {
        return;
    }

    autograd_parity_for_backend::<f32, Metal>();
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_f16_autograd_matches_cpu_supported_ops() {
    use rstorch::Metal;

    if <Metal as Backend<f16>>::default_device().is_err() {
        return;
    }

    autograd_parity_for_backend::<f16, Metal>();
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_transfers_dtype_and_device() {
    use rstorch::Metal;

    if <Metal as Backend<f16>>::default_device().is_err() {
        return;
    }

    let cpu = Tensor1D::<4, f32, Cpu>::from_vec(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let metal: Tensor1D<4, f16, Metal> = cpu.cast::<f16>().unwrap().to_backend::<Metal>().unwrap();
    assert_eq!(metal.dtype(), DTypeId::F16);
    assert_eq!(
        metal.to_vec().unwrap(),
        values::<f16>(&[1.0, 2.0, 3.0, 4.0])
    );

    let back: Tensor1D<4, f32, Cpu> = metal.cast::<f32>().unwrap().to_backend::<Cpu>().unwrap();
    assert_eq!(back.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_f32_matches_cpu_supported_ops() {
    use rstorch::Wgpu;

    if <Wgpu as Backend<f32>>::default_device().is_err() {
        return;
    }

    parity_for_backend::<f32, Wgpu>();
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_f16_matches_cpu_supported_ops_when_shader_f16_is_available() {
    use rstorch::Wgpu;

    if <Wgpu as Backend<f16>>::default_device().is_err() {
        return;
    }

    parity_for_backend::<f16, Wgpu>();
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_f32_autograd_matches_cpu_supported_ops() {
    use rstorch::Wgpu;

    if <Wgpu as Backend<f32>>::default_device().is_err() {
        return;
    }

    autograd_parity_for_backend::<f32, Wgpu>();
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_f16_autograd_matches_cpu_supported_ops_when_shader_f16_is_available() {
    use rstorch::Wgpu;

    if <Wgpu as Backend<f16>>::default_device().is_err() {
        return;
    }

    autograd_parity_for_backend::<f16, Wgpu>();
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_transfers_dtype_and_device_when_shader_f16_is_available() {
    use rstorch::Wgpu;

    if <Wgpu as Backend<f16>>::default_device().is_err() {
        return;
    }

    let cpu = Tensor1D::<4, f32, Cpu>::from_vec(vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let wgpu: Tensor1D<4, f16, Wgpu> = cpu.cast::<f16>().unwrap().to_backend::<Wgpu>().unwrap();
    assert_eq!(wgpu.dtype(), DTypeId::F16);
    assert_eq!(wgpu.to_vec().unwrap(), values::<f16>(&[1.0, 2.0, 3.0, 4.0]));

    let back: Tensor1D<4, f32, Cpu> = wgpu.cast::<f32>().unwrap().to_backend::<Cpu>().unwrap();
    assert_eq!(back.to_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}
