#![cfg(feature = "typed")]

use rstorch::typed::{
    Cpu, DYN, DeviceCtx, Placement, Tensor0, Tensor1, Tensor2, Tensor3, Tensor4, Tensor5, Tensor6,
    Tensor7, Tensor8, TypedTensor,
};
use rstorch::{DType, Device, Error, Tensor};

#[cfg(all(feature = "metal", target_os = "macos"))]
#[path = "common/metal.rs"]
mod metal;

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

    Tensor0::<f32>::try_from_dynamic(t0.into_dynamic(), &ctx).unwrap();
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

/// Both entry points must report the shape the target actually requires, and
/// must agree with each other for the identical failure. The payload is pinned
/// exactly: reporting only the first contradicting axis names a shape the
/// target also rejects, which is a misleading diagnostic rather than a wrong
/// result, and no `..` matcher can catch it.
#[test]
fn shape_mismatch_reports_the_required_shape_on_every_axis() {
    let ctx = cpu();

    // Two axes contradict their markers at once.
    let both_wrong = Tensor::zeros([2, 5], DType::F32, &Device::Cpu).unwrap();
    match Tensor2::<4, 3>::try_from_dynamic(both_wrong, &ctx) {
        Err(Error::ShapeMismatch { op, lhs, rhs, .. }) => {
            assert_eq!(op, "try_from_dynamic");
            assert_eq!(lhs.dims(), [2, 5], "lhs is the observed shape");
            assert_eq!(rhs.dims(), [4, 3], "rhs must be a shape the target accepts");
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }

    // `from_vec` reports the same required shape for the same contradiction,
    // and describes `lhs` as the requested dims rather than the data length.
    match Tensor2::<4, 3>::from_vec(vec![0.0f32; 10], [2, 5], &ctx) {
        Err(Error::ShapeMismatch { op, lhs, rhs, .. }) => {
            assert_eq!(op, "from_vec");
            assert_eq!(lhs.dims(), [2, 5], "lhs is the requested dims");
            assert_eq!(rhs.dims(), [4, 3]);
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }

    // A `DYN` marker keeps the observed dimension in the required shape.
    let one_wrong = Tensor::zeros([2, 5], DType::F32, &Device::Cpu).unwrap();
    match Tensor2::<DYN, 3>::try_from_dynamic(one_wrong, &ctx) {
        Err(Error::ShapeMismatch { rhs, .. }) => {
            assert_eq!(rhs.dims(), [2, 3], "DYN axis keeps the observed 2");
        }
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn dynamic_reentry_rejects_rank_static_dimension_and_dtype() {
    let ctx = cpu();
    let wrong_rank = Tensor::zeros([2], DType::F32, &Device::Cpu).unwrap();
    assert!(matches!(
        Tensor2::<DYN, DYN>::try_from_dynamic(wrong_rank, &ctx),
        Err(Error::RankMismatch {
            op: "try_from_dynamic",
            expected: 2,
            got: 1,
            ..
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
            got: DType::I64,
            ..
        })
    ));
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
fn public_refine_and_erase_work_for_every_rank() {
    macro_rules! assert_refine_erase {
        ($source:ty => $target:ty, $dims:expr) => {{
            let ctx = cpu();
            let source = <$source>::from_vec(vec![1.0f32], $dims, &ctx).unwrap();
            let refined: $target = source.refine().unwrap();
            assert_eq!(refined.dims(), $dims);
            let erased = refined.erase_shape().unwrap();
            assert_eq!(erased.dims(), $dims);
            assert_eq!(erased.into_dynamic().to_vec::<f32>().unwrap(), vec![1.0]);
        }};
    }

    assert_refine_erase!(Tensor0<f32, Cpu> => Tensor0<f32, Cpu>, [] as [usize; 0]);
    assert_refine_erase!(Tensor1<DYN> => Tensor1<1>, [1]);
    assert_refine_erase!(Tensor2<DYN, DYN> => Tensor2<1, 1>, [1, 1]);
    assert_refine_erase!(Tensor3<DYN, DYN, DYN> => Tensor3<1, 1, 1>, [1, 1, 1]);
    assert_refine_erase!(Tensor4<DYN, DYN, DYN, DYN> => Tensor4<1, 1, 1, 1>, [1, 1, 1, 1]);
    assert_refine_erase!(Tensor5<DYN, DYN, DYN, DYN, DYN> => Tensor5<1, 1, 1, 1, 1>, [1, 1, 1, 1, 1]);
    assert_refine_erase!(Tensor6<DYN, DYN, DYN, DYN, DYN, DYN> => Tensor6<1, 1, 1, 1, 1, 1>, [1, 1, 1, 1, 1, 1]);
    assert_refine_erase!(Tensor7<DYN, DYN, DYN, DYN, DYN, DYN, DYN> => Tensor7<1, 1, 1, 1, 1, 1, 1>, [1, 1, 1, 1, 1, 1, 1]);
    assert_refine_erase!(Tensor8<DYN, DYN, DYN, DYN, DYN, DYN, DYN, DYN> => Tensor8<1, 1, 1, 1, 1, 1, 1, 1>, [1, 1, 1, 1, 1, 1, 1, 1]);
}

#[test]
fn public_relabel_is_zero_copy_for_every_rank() {
    struct AlternateCpu;
    impl Placement for AlternateCpu {}

    macro_rules! assert_relabel {
        ($source:ty => $target:ty, $dims:expr) => {{
            let source_ctx = cpu();
            let target_ctx = DeviceCtx::<AlternateCpu>::bind(Device::Cpu).unwrap();
            let runtime = Tensor::from_vec(vec![1.0f32], $dims, &Device::Cpu)
                .unwrap()
                .traced()
                .unwrap();
            let identity = runtime.clone();
            let source = <$source>::try_from_dynamic(runtime, &source_ctx).unwrap();
            let relabeled: $target = source.relabel(&target_ctx).unwrap();
            assert_eq!(relabeled.dims(), $dims);
            let loss = relabeled.as_dynamic().sum_all().unwrap();
            let gradients = loss.backward().unwrap();
            assert_eq!(
                gradients
                    .wrt_input(&identity)
                    .unwrap()
                    .to_vec::<f32>()
                    .unwrap(),
                vec![1.0]
            );
        }};
    }

    assert_relabel!(Tensor0<f32, Cpu> => Tensor0<f32, AlternateCpu>, [] as [usize; 0]);
    assert_relabel!(Tensor1<1> => Tensor1<1, f32, AlternateCpu>, [1]);
    assert_relabel!(Tensor2<1, 1> => Tensor2<1, 1, f32, AlternateCpu>, [1, 1]);
    assert_relabel!(Tensor3<1, 1, 1> => Tensor3<1, 1, 1, f32, AlternateCpu>, [1, 1, 1]);
    assert_relabel!(Tensor4<1, 1, 1, 1> => Tensor4<1, 1, 1, 1, f32, AlternateCpu>, [1, 1, 1, 1]);
    assert_relabel!(Tensor5<1, 1, 1, 1, 1> => Tensor5<1, 1, 1, 1, 1, f32, AlternateCpu>, [1, 1, 1, 1, 1]);
    assert_relabel!(Tensor6<1, 1, 1, 1, 1, 1> => Tensor6<1, 1, 1, 1, 1, 1, f32, AlternateCpu>, [1, 1, 1, 1, 1, 1]);
    assert_relabel!(Tensor7<1, 1, 1, 1, 1, 1, 1> => Tensor7<1, 1, 1, 1, 1, 1, 1, f32, AlternateCpu>, [1, 1, 1, 1, 1, 1, 1]);
    assert_relabel!(Tensor8<1, 1, 1, 1, 1, 1, 1, 1> => Tensor8<1, 1, 1, 1, 1, 1, 1, 1, f32, AlternateCpu>, [1, 1, 1, 1, 1, 1, 1, 1]);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn public_relabel_rejects_a_different_physical_device() {
    use rstorch::typed::Metal;

    if !metal::available() {
        return;
    }

    let cpu = cpu();
    let metal = DeviceCtx::<Metal<0>>::bind(Device::Metal(0)).unwrap();
    let tensor = Tensor1::<1>::from_vec(vec![1.0], [1], &cpu).unwrap();
    assert!(matches!(
        tensor.relabel(&metal),
        Err(Error::DeviceMismatch {
            op: "relabel",
            expected: Device::Cpu,
            got: Device::Metal(0),
            ..
        })
    ));
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
