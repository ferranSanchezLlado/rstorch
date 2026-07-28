#![cfg(feature = "typed")]

use rstorch::typed::{
    Cpu, DYN, DeviceCtx, Placement, Tensor0, Tensor1, Tensor2, Tensor3, Tensor4, Tensor5, Tensor6,
    Tensor7, Tensor8, TypedTensor,
};
use rstorch::{DType, Device, Error, Tensor};

fn cpu() -> DeviceCtx<Cpu> {
    DeviceCtx::cpu().unwrap()
}

fn assert_traits<T: Clone + Send + Sync + 'static>() {}

#[test]
fn every_rank_constructs_and_reports_exact_runtime_dimensions() {
    let ctx = cpu();
    let t0 = Tensor0::from_vec(vec![0.0f32], [], &ctx).unwrap();
    let t1 = Tensor1::<1>::from_vec(vec![0.0], [1], &ctx).unwrap();
    let t2 = Tensor2::<1, 1>::from_vec(vec![0.0], [1, 1], &ctx).unwrap();
    let t3 = Tensor3::<1, 1, 1>::from_vec(vec![0.0], [1, 1, 1], &ctx).unwrap();
    let t4 = Tensor4::<1, 1, 1, 1>::from_vec(vec![0.0], [1, 1, 1, 1], &ctx).unwrap();
    let t5 = Tensor5::<1, 1, 1, 1, 1>::from_vec(vec![0.0], [1, 1, 1, 1, 1], &ctx).unwrap();
    let t6 = Tensor6::<1, 1, 1, 1, 1, 1>::from_vec(vec![0.0], [1, 1, 1, 1, 1, 1], &ctx).unwrap();
    let t7 =
        Tensor7::<1, 1, 1, 1, 1, 1, 1>::from_vec(vec![0.0], [1, 1, 1, 1, 1, 1, 1], &ctx).unwrap();
    let t8 = Tensor8::<1, 1, 1, 1, 1, 1, 1, 1>::from_vec(vec![0.0], [1, 1, 1, 1, 1, 1, 1, 1], &ctx)
        .unwrap();

    assert_eq!(t0.dims(), [] as [usize; 0]);
    assert_eq!(t1.dims(), [1]);
    assert_eq!(t2.dims(), [1, 1]);
    assert_eq!(t3.dims(), [1, 1, 1]);
    assert_eq!(t4.dims(), [1, 1, 1, 1]);
    assert_eq!(t5.dims(), [1, 1, 1, 1, 1]);
    assert_eq!(t6.dims(), [1, 1, 1, 1, 1, 1]);
    assert_eq!(t7.dims(), [1, 1, 1, 1, 1, 1, 1]);
    assert_eq!(t8.dims(), [1, 1, 1, 1, 1, 1, 1, 1]);
    assert_eq!(Tensor8::<1, 1, 1, 1, 1, 1, 1, 1>::RANK, 8);

    Tensor0::<f32, Cpu>::try_from_dynamic(t0.into_dynamic(), &ctx).unwrap();
    Tensor1::<1>::try_from_dynamic(t1.into_dynamic(), &ctx).unwrap();
    Tensor2::<1, 1>::try_from_dynamic(t2.into_dynamic(), &ctx).unwrap();
    Tensor3::<1, 1, 1>::try_from_dynamic(t3.into_dynamic(), &ctx).unwrap();
    Tensor4::<1, 1, 1, 1>::try_from_dynamic(t4.into_dynamic(), &ctx).unwrap();
    Tensor5::<1, 1, 1, 1, 1>::try_from_dynamic(t5.into_dynamic(), &ctx).unwrap();
    Tensor6::<1, 1, 1, 1, 1, 1>::try_from_dynamic(t6.into_dynamic(), &ctx).unwrap();
    Tensor7::<1, 1, 1, 1, 1, 1, 1>::try_from_dynamic(t7.into_dynamic(), &ctx).unwrap();
    Tensor8::<1, 1, 1, 1, 1, 1, 1, 1>::try_from_dynamic(t8.into_dynamic(), &ctx).unwrap();
}

#[test]
fn dynamic_markers_retain_independent_changing_values() {
    let ctx = cpu();
    let first = Tensor2::<DYN, DYN>::from_vec(vec![0.0f32; 6], [2, 3], &ctx).unwrap();
    let second = Tensor2::<DYN, DYN>::from_vec(vec![0.0f32; 20], [4, 5], &ctx).unwrap();
    assert_eq!(first.dims(), [2, 3]);
    assert_eq!(second.dims(), [4, 5]);
}

#[test]
fn zero_is_a_static_dimension_and_empty_data_is_valid() {
    let ctx = cpu();
    let empty = Tensor2::<0, 3>::from_vec(Vec::<f32>::new(), [0, 3], &ctx).unwrap();
    assert_eq!(empty.dims(), [0, 3]);
    assert_eq!(empty.as_dynamic().num_elements(), 0);

    assert!(matches!(
        Tensor2::<0, 3>::from_vec(Vec::<f32>::new(), [1, 3], &ctx),
        Err(Error::ShapeMismatch { op: "from_vec", .. })
    ));
}

#[test]
fn dynamic_reentry_rejects_rank_static_dimension_dtype_and_device() {
    let ctx = cpu();
    let wrong_rank = Tensor::zeros([2], DType::F32, &Device::Cpu).unwrap();
    assert!(matches!(
        Tensor2::<DYN, DYN>::try_from_dynamic(wrong_rank, &ctx),
        Err(Error::RankMismatch {
            op: "try_from_dynamic",
            expected: 2,
            got: 1
        })
    ));

    let wrong_dim = Tensor::zeros([2, 4], DType::F32, &Device::Cpu).unwrap();
    assert!(matches!(
        Tensor2::<DYN, 3>::try_from_dynamic(wrong_dim, &ctx),
        Err(Error::ShapeMismatch {
            op: "try_from_dynamic",
            ..
        })
    ));

    let wrong_dtype = Tensor::zeros([2, 3], DType::I64, &Device::Cpu).unwrap();
    assert!(matches!(
        Tensor2::<DYN, 3>::try_from_dynamic(wrong_dtype, &ctx),
        Err(Error::DTypeMismatch {
            op: "try_from_dynamic",
            expected: DType::F32,
            got: DType::I64
        })
    ));

    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        struct OtherDevice;
        impl Placement for OtherDevice {}
        let other = DeviceCtx::<OtherDevice>::bind(Device::Metal(0)).unwrap();
        let runtime = Tensor::zeros([1], DType::F32, &Device::Cpu).unwrap();
        assert!(matches!(
            Tensor1::<1, f32, OtherDevice>::try_from_dynamic(runtime, &other),
            Err(Error::DeviceMismatch {
                op: "try_from_dynamic",
                ..
            })
        ));
    }
}

#[test]
fn refinement_erasure_and_dynamic_roundtrip_preserve_values_and_metadata() {
    let ctx = cpu();
    let runtime = Tensor::from_vec(vec![1i64, 2, 3, 4, 5, 6], [2, 3], &Device::Cpu).unwrap();
    let typed = Tensor2::<DYN, DYN, i64>::try_from_dynamic(runtime, &ctx).unwrap();
    let refined = typed.refine::<Tensor2<2, 3, i64>>().unwrap();
    assert_eq!(refined.dims(), [2, 3]);
    assert_eq!(refined.as_dynamic().dtype(), DType::I64);
    assert_eq!(refined.as_dynamic().device(), Device::Cpu);

    let erased = refined.erase_shape().unwrap();
    let runtime = erased.into_dynamic();
    assert_eq!(runtime.to_vec::<i64>().unwrap(), vec![1, 2, 3, 4, 5, 6]);
    let roundtrip = Tensor2::<2, 3, i64>::try_from_dynamic(runtime, &ctx).unwrap();
    assert_eq!(roundtrip.dims(), [2, 3]);
}

#[test]
fn refinement_rejects_an_incorrect_static_target() {
    let ctx = cpu();
    let typed = Tensor2::<DYN, DYN>::from_vec(vec![0.0f32; 6], [2, 3], &ctx).unwrap();
    assert!(matches!(
        typed.refine::<Tensor2<2, 4>>(),
        Err(Error::ShapeMismatch { op: "refine", .. })
    ));
}

#[test]
fn wrappers_have_required_thread_and_lifetime_traits() {
    struct Main;
    impl Placement for Main {}
    assert_traits::<Tensor0<f32, Main>>();
    assert_traits::<Tensor8<DYN, 1, 2, 3, 4, 5, 6, 7, i64, Main>>();
}

#[test]
fn formatting_is_exactly_the_runtime_tensor_formatting() {
    let ctx = cpu();
    let runtime = Tensor::from_vec(vec![1.0f32, 2.0], [2], &Device::Cpu).unwrap();
    let expected_debug = format!("{runtime:?}");
    let expected_display = format!("{runtime}");
    let typed = Tensor1::<2>::try_from_dynamic(runtime, &ctx).unwrap();
    assert_eq!(format!("{typed:?}"), expected_debug);
    assert_eq!(format!("{typed}"), expected_display);
}
