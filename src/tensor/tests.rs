//! The `Tensor` core: construction, accessors, host transfer, movement, and
//! the `sum_to` broadcast-gradient reducer every backward funnels through.

use super::*;

const CPU: Device = Device::Cpu;

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

/// A contiguous f32 tensor on CPU.
fn t_f32(data: &[f32], shape: impl Into<Shape>) -> Tensor {
    Tensor::from_vec(data.to_vec(), shape, &CPU).unwrap()
}

/// Re-view `t` through `layout` over the same storage. The only way to
/// build a non-contiguous `Tensor` without the public view ops.
fn re_view(t: &Tensor, layout: Layout) -> Tensor {
    Tensor::from_parts(t.storage().clone(), layout)
}

/// The address of the f32 buffer behind `t`, for "did this share storage
/// or allocate a copy?" assertions.
fn f32_buf_ptr(t: &Tensor) -> *const f32 {
    match t.storage() {
        Storage::Cpu(CpuStorage::F32(v)) => v.as_ptr(),
        _ => panic!("expected an f32 CPU tensor"),
    }
}

// ------------------------------------------------------------------
// Constructors
// ------------------------------------------------------------------

#[test]
fn zeros_ones_full_shape_dtype_device() {
    let z = Tensor::zeros([2, 3], DType::F32, &CPU).unwrap();
    assert_eq!(z.dims(), &[2, 3]);
    assert_eq!(z.shape(), &Shape::from([2, 3]));
    assert_eq!(z.rank(), 2);
    assert_eq!(z.num_elements(), 6);
    assert_eq!(z.dtype(), DType::F32);
    assert_eq!(z.device(), CPU);
    assert!(z.is_contiguous());
    assert_eq!(z.to_vec::<f32>().unwrap(), vec![0.0; 6]);

    let o = Tensor::ones([4], DType::F32, &CPU).unwrap();
    assert_eq!(o.to_vec::<f32>().unwrap(), vec![1.0; 4]);

    let f = Tensor::full([2, 2], -1.5, DType::F32, &CPU).unwrap();
    assert_eq!(f.to_vec::<f32>().unwrap(), vec![-1.5; 4]);
}

#[test]
fn constructors_cover_every_dtype_flavour() {
    // Integer fills truncate toward zero; bool fills are `value != 0`.
    let i = Tensor::full([3], 2.9, DType::I64, &CPU).unwrap();
    assert_eq!(i.dtype(), DType::I64);
    assert_eq!(i.to_vec::<i64>().unwrap(), vec![2, 2, 2]);

    let b = Tensor::ones([2], DType::Bool, &CPU).unwrap();
    assert_eq!(b.to_vec::<bool>().unwrap(), vec![true, true]);
    let b = Tensor::zeros([2], DType::Bool, &CPU).unwrap();
    assert_eq!(b.to_vec::<bool>().unwrap(), vec![false, false]);

    // f16/bf16/f64 tensors can be *built* even though their kernels are
    // deferred: the six Element impls exist from the start.
    let h = Tensor::full([2], 1.0, DType::F16, &CPU).unwrap();
    assert_eq!(h.dtype(), DType::F16);
    assert_eq!(
        h.to_vec::<half::f16>().unwrap(),
        vec![half::f16::from_f32(1.0); 2]
    );
    let d = Tensor::full([2], 0.25, DType::F64, &CPU).unwrap();
    assert_eq!(d.to_vec::<f64>().unwrap(), vec![0.25, 0.25]);
}

#[test]
fn scalar_and_empty_shapes_are_constructible() {
    let s = Tensor::full((), 7.0, DType::F32, &CPU).unwrap();
    assert_eq!(s.rank(), 0);
    assert_eq!(s.num_elements(), 1);
    assert_eq!(s.to_scalar::<f32>().unwrap(), 7.0);

    let e = Tensor::zeros([0, 3], DType::F32, &CPU).unwrap();
    assert_eq!(e.num_elements(), 0);
    assert!(e.to_vec::<f32>().unwrap().is_empty());
}

#[test]
fn from_vec_round_trips_and_checks_length() {
    let t = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    assert_eq!(t.dims(), &[2, 3]);
    assert_eq!(t.dtype(), DType::F32);
    assert_eq!(
        t.to_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );

    // dtype comes from `T`, not from an argument.
    let i = Tensor::from_vec(vec![1i64, 2, 3], [3], &CPU).unwrap();
    assert_eq!(i.dtype(), DType::I64);
    let b = Tensor::from_vec(vec![true, false], [2], &CPU).unwrap();
    assert_eq!(b.dtype(), DType::Bool);

    // Length must equal the shape's element count.
    assert!(matches!(
        Tensor::from_vec(vec![1.0f32, 2.0], [3], &CPU),
        Err(Error::ShapeMismatch { op: "from_vec", .. })
    ));
}

#[test]
fn full_overflowing_shape_is_invalid_arg() {
    // The single overflow-validation point is the contiguous layout.
    assert!(matches!(
        Tensor::zeros([usize::MAX, usize::MAX], DType::F32, &CPU),
        Err(Error::InvalidArg { op: "layout", .. })
    ));
}

#[test]
fn like_constructors_inherit_shape_dtype_and_device() {
    // A non-default dtype and a non-trivial shape, so inheriting them is
    // observable rather than accidentally equal to the default.
    let src = Tensor::full([2, 3], 9.0, DType::I64, &CPU).unwrap();

    for (like, expected) in [
        (src.zeros_like().unwrap(), 0i64),
        (src.ones_like().unwrap(), 1),
        (src.full_like(-4.5).unwrap(), -4),
    ] {
        assert_eq!(like.dims(), src.dims());
        assert_eq!(like.dtype(), src.dtype());
        assert_eq!(like.device(), src.device());
        assert_eq!(like.to_vec::<i64>().unwrap(), vec![expected; 6]);
    }

    // Every dtype flavour, not just the integer one above.
    for dtype in [DType::F32, DType::F64, DType::F16, DType::BF16, DType::Bool] {
        let src = Tensor::zeros([4], dtype, &CPU).unwrap();
        assert_eq!(src.ones_like().unwrap().dtype(), dtype);
        assert_eq!(src.ones_like().unwrap().device(), CPU);
    }

    // Device inheritance is only distinguishable from "defaults to CPU" when a
    // second device exists, so assert it against a Metal source when one is
    // present.
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if let Ok(gpu) = Tensor::zeros([2, 2], DType::F32, &Device::Metal(0)) {
        assert_eq!(gpu.zeros_like().unwrap().device(), Device::Metal(0));
        assert_eq!(gpu.full_like(2.0).unwrap().device(), Device::Metal(0));
    }
}

#[test]
fn like_constructors_are_untraced_even_from_a_traced_source() {
    let src = t_f32(&[1.0, 2.0], [2]).traced().unwrap();
    assert!(src.node().is_some());

    for like in [
        src.zeros_like().unwrap(),
        src.ones_like().unwrap(),
        src.full_like(3.0).unwrap(),
    ] {
        assert!(like.node().is_none());
        // No graph means no gradient: `backward` on the result is the
        // `NotTraced` error, not a silent zero.
        assert!(matches!(
            like.sum_all().unwrap().backward(),
            Err(Error::NotTraced { .. })
        ));
    }
}

#[test]
fn rand_is_in_unit_interval_and_reproducible() {
    let mut rng = Rng::seed(1234);
    let a = Tensor::rand([64], DType::F32, &CPU, &mut rng).unwrap();
    let values = a.to_vec::<f32>().unwrap();
    assert_eq!(values.len(), 64);
    assert!(values.iter().all(|&v| (0.0..1.0).contains(&v)));
    // Not a constant tensor.
    assert!(values.iter().any(|&v| v != values[0]));

    // Same seed, same draw sequence.
    let mut rng2 = Rng::seed(1234);
    let b = Tensor::rand([64], DType::F32, &CPU, &mut rng2).unwrap();
    assert_eq!(b.to_vec::<f32>().unwrap(), values);
}

#[test]
fn randn_is_roughly_standard_normal() {
    let mut rng = Rng::seed(7);
    let a = Tensor::randn([4096], DType::F32, &CPU, &mut rng).unwrap();
    let v = a.to_vec::<f32>().unwrap();
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
    assert!(mean.abs() < 0.1, "mean {mean}");
    assert!((var - 1.0).abs() < 0.15, "var {var}");
}

#[test]
fn rand_requires_a_float_dtype() {
    let mut rng = Rng::seed(1);
    assert!(matches!(
        Tensor::rand([4], DType::I64, &CPU, &mut rng),
        Err(Error::InvalidArg { op: "rand", .. })
    ));
    assert!(matches!(
        Tensor::randn([4], DType::Bool, &CPU, &mut rng),
        Err(Error::InvalidArg { op: "randn", .. })
    ));
    // Every float dtype is accepted.
    for dtype in [DType::F16, DType::BF16, DType::F32, DType::F64] {
        let t = Tensor::rand([2], dtype, &CPU, &mut rng).unwrap();
        assert_eq!(t.dtype(), dtype);
    }
}

#[test]
fn arange_counts_and_values() {
    let a = Tensor::arange(0.0, 5.0, 1.0, DType::F32, &CPU).unwrap();
    assert_eq!(a.dims(), &[5]);
    assert_eq!(a.to_vec::<f32>().unwrap(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);

    // Non-integral count rounds up (PyTorch's ceil rule).
    let a = Tensor::arange(0.0, 1.0, 0.3, DType::F32, &CPU).unwrap();
    assert_eq!(a.dims(), &[4]); // ceil(1/0.3) == 4
    let v = a.to_vec::<f32>().unwrap();
    assert!((v[3] - 0.9).abs() < 1e-6);

    // Negative step counts down.
    let a = Tensor::arange(3.0, 0.0, -1.0, DType::F32, &CPU).unwrap();
    assert_eq!(a.to_vec::<f32>().unwrap(), vec![3.0, 2.0, 1.0]);

    // I64 is the indexing dtype the index utilities build on.
    let a = Tensor::arange(0.0, 4.0, 1.0, DType::I64, &CPU).unwrap();
    assert_eq!(a.dtype(), DType::I64);
    assert_eq!(a.to_vec::<i64>().unwrap(), vec![0, 1, 2, 3]);
}

#[test]
fn arange_empty_and_invalid_ranges() {
    // A step pointing away from `end` yields an empty tensor, not an error.
    let a = Tensor::arange(0.0, 5.0, -1.0, DType::F32, &CPU).unwrap();
    assert_eq!(a.dims(), &[0]);
    assert!(a.to_vec::<f32>().unwrap().is_empty());
    let a = Tensor::arange(2.0, 2.0, 1.0, DType::F32, &CPU).unwrap();
    assert_eq!(a.num_elements(), 0);

    assert!(matches!(
        Tensor::arange(0.0, 5.0, 0.0, DType::F32, &CPU),
        Err(Error::InvalidArg { op: "arange", .. })
    ));
    assert!(matches!(
        Tensor::arange(0.0, 5.0, 1.0, DType::Bool, &CPU),
        Err(Error::InvalidArg { op: "arange", .. })
    ));
    // Non-finite bounds are rejected rather than silently saturating the
    // element count.
    for (s, e, st) in [
        (f64::NAN, 5.0, 1.0),
        (0.0, f64::INFINITY, 1.0),
        (0.0, 5.0, f64::NAN),
    ] {
        assert!(matches!(
            Tensor::arange(s, e, st, DType::F32, &CPU),
            Err(Error::InvalidArg { op: "arange", .. })
        ));
    }
    // A finite range whose count cannot fit in `usize` is loud, not an
    // allocation abort.
    assert!(matches!(
        Tensor::arange(0.0, f64::MAX, 1.0, DType::F32, &CPU),
        Err(Error::InvalidArg { op: "arange", .. })
    ));
}

// ------------------------------------------------------------------
// Host transfer
// ------------------------------------------------------------------

#[test]
fn to_vec_walks_strided_views_in_row_major_order() {
    let t = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let transposed = re_view(&t, t.layout().transpose(0, 1).unwrap());
    assert_eq!(transposed.dims(), &[3, 2]);
    assert!(!transposed.is_contiguous());
    assert_eq!(
        transposed.to_vec::<f32>().unwrap(),
        vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    );

    // Broadcast (stride-0) axes repeat elements on the way out.
    let row = t_f32(&[10.0, 20.0, 30.0], [1, 3]);
    let b = re_view(
        &row,
        row.layout().broadcast_to(&Shape::from([2, 3])).unwrap(),
    );
    assert_eq!(
        b.to_vec::<f32>().unwrap(),
        vec![10.0, 20.0, 30.0, 10.0, 20.0, 30.0]
    );
}

#[test]
fn to_vec_is_dtype_strict() {
    let t = t_f32(&[1.0, 2.0], [2]);
    // No implicit promotion on the way to the host either: reading an f32
    // tensor as i64/f64 is a structured error, not a conversion.
    assert!(matches!(
        t.to_vec::<i64>(),
        Err(Error::DTypeMismatch {
            op: "to_vec",
            expected: DType::I64,
            got: DType::F32
        })
    ));
    assert!(matches!(
        t.to_vec::<f64>(),
        Err(Error::DTypeMismatch { op: "to_vec", .. })
    ));
    // `to_scalar` inherits the strictness (element count is checked
    // first, so use a single-element tensor here).
    assert!(matches!(
        t_f32(&[1.0], ()).to_scalar::<i64>(),
        Err(Error::DTypeMismatch { .. })
    ));
    // `item` is the deliberate exception: dtype-agnostic by design.
    assert_eq!(t_f32(&[1.0], ()).item().unwrap(), 1.0);
}

#[test]
fn to_scalar_requires_exactly_one_element() {
    let one = t_f32(&[42.0], [1, 1, 1]);
    assert_eq!(one.to_scalar::<f32>().unwrap(), 42.0);
    let many = t_f32(&[1.0, 2.0], [2]);
    assert!(matches!(
        many.to_scalar::<f32>(),
        Err(Error::InvalidArg {
            op: "to_scalar",
            ..
        })
    ));
    let empty = Tensor::zeros([0], DType::F32, &CPU).unwrap();
    assert!(matches!(
        empty.to_scalar::<f32>(),
        Err(Error::InvalidArg { .. })
    ));
}

#[test]
fn item_reads_any_dtype_as_f64() {
    assert_eq!(t_f32(&[2.5], ()).item().unwrap(), 2.5);
    assert_eq!(
        Tensor::from_vec(vec![7i64], (), &CPU)
            .unwrap()
            .item()
            .unwrap(),
        7.0
    );
    assert_eq!(
        Tensor::from_vec(vec![true], (), &CPU)
            .unwrap()
            .item()
            .unwrap(),
        1.0
    );
    assert_eq!(
        Tensor::from_vec(vec![false], (), &CPU)
            .unwrap()
            .item()
            .unwrap(),
        0.0
    );
    assert_eq!(
        Tensor::full((), 1.5, DType::F16, &CPU)
            .unwrap()
            .item()
            .unwrap(),
        1.5
    );
    assert_eq!(
        Tensor::full((), 1.5, DType::BF16, &CPU)
            .unwrap()
            .item()
            .unwrap(),
        1.5
    );
    assert_eq!(
        Tensor::full((), 0.5, DType::F64, &CPU)
            .unwrap()
            .item()
            .unwrap(),
        0.5
    );
    assert!(matches!(
        t_f32(&[1.0, 2.0], [2]).item(),
        Err(Error::InvalidArg { op: "item", .. })
    ));
}

#[test]
fn item_reads_the_first_logical_element_of_a_view() {
    // A single-element *view* into a larger buffer must read the element
    // the view points at, not `storage[0]`.
    let t = t_f32(&[1.0, 2.0, 3.0, 4.0], [4]);
    let last = re_view(&t, t.layout().narrow(0, 3, 1).unwrap());
    assert_eq!(last.item().unwrap(), 4.0);
    assert_eq!(last.to_scalar::<f32>().unwrap(), 4.0);
}

// ------------------------------------------------------------------
// Movement: to_device / to_dtype / contiguous
// ------------------------------------------------------------------

#[test]
fn to_device_same_device_is_the_identity() {
    let t = t_f32(&[1.0, 2.0], [2]);
    let moved = t.to_device(&CPU).unwrap();
    assert_eq!(moved.device(), CPU);
    // Identity: the very same buffer, no copy.
    assert_eq!(f32_buf_ptr(&t), f32_buf_ptr(&moved));
}

#[test]
fn to_dtype_casts_explicitly_and_never_promotes_implicitly() {
    let f = t_f32(&[1.9, -1.9, 0.0], [3]);
    let i = f.to_dtype(DType::I64).unwrap();
    assert_eq!(i.dtype(), DType::I64);
    assert_eq!(i.to_vec::<i64>().unwrap(), vec![1, -1, 0]);
    // ...and back.
    assert_eq!(
        i.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap(),
        vec![1.0, -1.0, 0.0]
    );
    // Numeric -> Bool is `x != 0`; Bool -> numeric is 0/1.
    let b = f.to_dtype(DType::Bool).unwrap();
    assert_eq!(b.to_vec::<bool>().unwrap(), vec![true, true, false]);
    assert_eq!(
        b.to_dtype(DType::F32).unwrap().to_vec::<f32>().unwrap(),
        vec![1.0, 1.0, 0.0]
    );

    // The identity cast is free and keeps the same buffer.
    let same = f.to_dtype(DType::F32).unwrap();
    assert_eq!(f32_buf_ptr(&f), f32_buf_ptr(&same));

    let half = f.to_dtype(DType::F16).unwrap();
    assert_eq!(half.dtype(), DType::F16);
    assert_eq!(
        half.to_dtype(DType::BF16)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap(),
        vec![1.8984375, -1.8984375, 0.0]
    );
    // F64 cast scope remains deferred and loud.
    assert!(matches!(
        f.to_dtype(DType::F64),
        Err(Error::Unsupported { op: "to_dtype", .. })
    ));
}

#[test]
fn to_dtype_materializes_strided_sources() {
    let t = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let transposed = re_view(&t, t.layout().transpose(0, 1).unwrap());
    let cast = transposed.to_dtype(DType::I64).unwrap();
    assert_eq!(cast.dims(), &[3, 2]);
    assert!(cast.is_contiguous());
    assert_eq!(cast.to_vec::<i64>().unwrap(), vec![1, 4, 2, 5, 3, 6]);
}

#[test]
fn contiguous_is_free_when_already_contiguous() {
    let t = t_f32(&[1.0, 2.0, 3.0], [3]);
    assert!(t.is_contiguous());
    let c = t.contiguous().unwrap();
    assert_eq!(f32_buf_ptr(&t), f32_buf_ptr(&c));
}

#[test]
fn contiguous_copies_permuted_and_narrowed_views() {
    let t = t_f32(&(0..24).map(|x| x as f32).collect::<Vec<_>>(), [2, 3, 4]);

    // Permuted: dims [4, 2, 3].
    let permuted = re_view(&t, t.layout().permute(&[2, 0, 1]).unwrap());
    assert!(!permuted.is_contiguous());
    let expected = permuted.to_vec::<f32>().unwrap();
    let c = permuted.contiguous().unwrap();
    assert!(c.is_contiguous());
    assert_eq!(c.dims(), &[4, 2, 3]);
    assert_ne!(f32_buf_ptr(&permuted), f32_buf_ptr(&c));
    assert_eq!(c.to_vec::<f32>().unwrap(), expected);

    // Narrowed (non-zero offset, gaps between rows).
    let narrowed = re_view(&t, t.layout().narrow(2, 1, 2).unwrap());
    assert!(!narrowed.is_contiguous());
    let expected = narrowed.to_vec::<f32>().unwrap();
    let c = narrowed.contiguous().unwrap();
    assert!(c.is_contiguous());
    assert_eq!(c.dims(), &[2, 3, 2]);
    assert_eq!(c.to_vec::<f32>().unwrap(), expected);
    // The copy is dense: 12 live elements, not the original 24-slot buffer.
    assert_eq!(c.storage().len(), 12);

    // Broadcast views materialize their repeats.
    let row = t_f32(&[1.0, 2.0], [1, 2]);
    let b = re_view(
        &row,
        row.layout().broadcast_to(&Shape::from([3, 2])).unwrap(),
    );
    let c = b.contiguous().unwrap();
    assert_eq!(c.dims(), &[3, 2]);
    assert_eq!(c.storage().len(), 6);
    assert_eq!(
        c.to_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    );
}

// ------------------------------------------------------------------
// detach
// ------------------------------------------------------------------

#[test]
fn detach_shares_storage_and_layout_and_carries_no_node() {
    let t = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let view = re_view(&t, t.layout().transpose(0, 1).unwrap());

    let d = view.detach();
    assert!(d.node().is_none());
    // Storage is shared (an Arc bump, not an element copy)...
    assert_eq!(f32_buf_ptr(&view), f32_buf_ptr(&d));
    // ...and so is the layout: detach is not a `contiguous()`.
    assert_eq!(d.layout(), view.layout());
    assert_eq!(d.dims(), &[3, 2]);
    assert!(!d.is_contiguous());
    assert_eq!(d.to_vec::<f32>().unwrap(), view.to_vec::<f32>().unwrap());

    // `detach` is exactly the public spelling of `detach_shallow`.
    let shallow = view.detach_shallow();
    assert_eq!(f32_buf_ptr(&shallow), f32_buf_ptr(&d));
    assert_eq!(shallow.layout(), d.layout());

    // An untraced tensor detaches to an equivalent untraced tensor; the
    // node-dropping half of the contract cannot be exercised until the engine
    // makes `record`/`make_leaf` build real nodes.
    assert!(t.detach().node().is_none());
}

// ------------------------------------------------------------------
// sum_to — the broadcast-gradient reducer
// ------------------------------------------------------------------

#[test]
fn sum_to_identity_returns_the_same_buffer() {
    let t = t_f32(&[1.0, 2.0, 3.0, 4.0], [2, 2]);
    let s = t.sum_to(&[2, 2]).unwrap();
    assert_eq!(s.dims(), &[2, 2]);
    assert_eq!(f32_buf_ptr(&t), f32_buf_ptr(&s));

    // Size-1 axes that were never expanded are also a no-op.
    let t = t_f32(&[1.0, 2.0, 3.0], [1, 3]);
    let s = t.sum_to(&[1, 3]).unwrap();
    assert_eq!(f32_buf_ptr(&t), f32_buf_ptr(&s));

    // Rank-0 to rank-0.
    let t = t_f32(&[5.0], ());
    assert_eq!(t.sum_to(&[]).unwrap().item().unwrap(), 5.0);
}

#[test]
fn sum_to_scalar_target_sums_everything() {
    let t = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let s = t.sum_to(&[]).unwrap();
    assert_eq!(s.rank(), 0);
    assert_eq!(s.num_elements(), 1);
    assert_eq!(s.to_scalar::<f32>().unwrap(), 21.0);

    // The rank-1 size-1 target is the *other* scalar spelling: shape [1].
    let s = t.sum_to(&[1]).unwrap();
    assert_eq!(s.dims(), &[1]);
    assert_eq!(s.to_vec::<f32>().unwrap(), vec![21.0]);
}

#[test]
fn sum_to_partial_broadcast_keeps_size_one_axes() {
    // [[1,2,3],[4,5,6]]
    let t = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);

    // Column vector operand: sum over the last axis, keep it at 1.
    let s = t.sum_to(&[2, 1]).unwrap();
    assert_eq!(s.dims(), &[2, 1]);
    assert_eq!(s.to_vec::<f32>().unwrap(), vec![6.0, 15.0]);

    // Row vector operand: sum over the leading axis, keep it at 1.
    let s = t.sum_to(&[1, 3]).unwrap();
    assert_eq!(s.dims(), &[1, 3]);
    assert_eq!(s.to_vec::<f32>().unwrap(), vec![5.0, 7.0, 9.0]);

    // Both axes at 1: everything summed, rank preserved.
    let s = t.sum_to(&[1, 1]).unwrap();
    assert_eq!(s.dims(), &[1, 1]);
    assert_eq!(s.to_vec::<f32>().unwrap(), vec![21.0]);
}

#[test]
fn sum_to_drops_leading_axes() {
    // A [2, 3] operand broadcast against a [4, 2, 3] one: the cotangent
    // is [4, 2, 3] and the leading axis must be summed away and dropped.
    let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
    let t = t_f32(&data, [4, 2, 3]);
    let s = t.sum_to(&[2, 3]).unwrap();
    assert_eq!(s.dims(), &[2, 3]);
    // out[i,j] = sum over b of data[b, i, j]; the four planes are offset
    // by 6 each, so out = base + (0+6+12+18) = base + 36.
    assert_eq!(
        s.to_vec::<f32>().unwrap(),
        vec![36.0, 40.0, 44.0, 48.0, 52.0, 56.0]
    );

    // A [3] operand: two leading axes dropped.
    let s = t.sum_to(&[3]).unwrap();
    assert_eq!(s.dims(), &[3]);
    // Column sums of all 8 rows: rows start at 0,3,6,...,21.
    let expected: Vec<f32> = (0..3)
        .map(|c| (0..8).map(|r| (r * 3 + c) as f32).sum())
        .collect();
    assert_eq!(s.to_vec::<f32>().unwrap(), expected);

    // Leading axes and an expanded aligned axis at once.
    let s = t.sum_to(&[1, 3]).unwrap();
    assert_eq!(s.dims(), &[1, 3]);
    assert_eq!(s.to_vec::<f32>().unwrap(), expected);
}

#[test]
fn sum_to_is_the_transpose_of_broadcast_to() {
    // Property: for every operand shape that broadcasts to `[2, 3, 4]`,
    // summing a cotangent of ones back to that operand must yield the
    // number of output elements each operand element fed, i.e. the
    // broadcast multiplicity.
    let out_dims = [2usize, 3, 4];
    let total: usize = out_dims.iter().product();
    let g = Tensor::ones(out_dims, DType::F32, &CPU).unwrap();
    for target in [
        vec![2usize, 3, 4],
        vec![1, 3, 4],
        vec![2, 1, 4],
        vec![2, 3, 1],
        vec![1, 1, 4],
        vec![1, 1, 1],
        vec![3, 4],
        vec![1, 4],
        vec![4],
        vec![1],
        vec![],
    ] {
        let reduced = g.sum_to(&target).unwrap();
        assert_eq!(reduced.dims(), target.as_slice(), "target {target:?}");
        let n: usize = target.iter().product();
        let multiplicity = (total / n) as f32;
        assert_eq!(
            reduced.to_vec::<f32>().unwrap(),
            vec![multiplicity; n],
            "target {target:?}"
        );
    }
}

#[test]
fn sum_to_handles_strided_sources_and_other_dtypes() {
    // A non-contiguous cotangent must reduce correctly (the backend reduce
    // walks strides; no `contiguous()` is required first).
    let t = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);
    let transposed = re_view(&t, t.layout().transpose(0, 1).unwrap()); // [3, 2]
    let s = transposed.sum_to(&[1, 2]).unwrap();
    assert_eq!(s.dims(), &[1, 2]);
    // Transposed rows are [1,4],[2,5],[3,6]; column sums are [6, 15].
    assert_eq!(s.to_vec::<f32>().unwrap(), vec![6.0, 15.0]);

    // dtype and device survive the reduction.
    let i = Tensor::from_vec(vec![1i64, 2, 3, 4], [2, 2], &CPU).unwrap();
    let s = i.sum_to(&[2, 1]).unwrap();
    assert_eq!(s.dtype(), DType::I64);
    assert_eq!(s.device(), CPU);
    assert_eq!(s.to_vec::<i64>().unwrap(), vec![3, 7]);
}

#[test]
fn sum_to_accumulates_in_the_wide_acc_type() {
    for dtype in [DType::F16, DType::BF16] {
        // Native f16 addition stalls at 2048 and bf16 at 256; Acc = f32.
        let t = Tensor::full([4096], 1.0, dtype, &CPU).unwrap();
        let s = t.sum_to(&[]).unwrap();
        assert_eq!(s.dtype(), dtype);
        assert_eq!(s.item().unwrap(), 4096.0);
    }
}

#[test]
fn sum_to_keeps_reduced_multi_axis_accumulation_wide_until_the_target_shape() {
    let width = 65_520;
    let mut f16_values = vec![half::f16::ONE; width];
    f16_values.extend(vec![half::f16::NEG_ONE; width]);
    let f16 = Tensor::from_vec(f16_values, [2, width], &CPU).unwrap();
    assert_eq!(f16.sum_to(&[1, 1]).unwrap().item().unwrap(), 0.0);

    let mut bf16_values = Vec::with_capacity(514);
    for index in 0..257 {
        bf16_values.push(half::bf16::ONE);
        bf16_values.push(if index < 256 {
            half::bf16::NEG_ONE
        } else {
            half::bf16::ZERO
        });
    }
    let base = Tensor::from_vec(bf16_values, [257, 2], &CPU).unwrap();
    let strided = re_view(&base, base.layout().transpose(0, 1).unwrap());
    assert_eq!(strided.sum_to(&[1, 1]).unwrap().item().unwrap(), 1.0);
}

#[test]
fn sum_to_over_an_empty_axis_is_zero() {
    // Broadcasting a [3] operand to [0, 3] produces no outputs, so the
    // gradient contribution is zero, not an error.
    let t = Tensor::zeros([0, 3], DType::F32, &CPU).unwrap();
    let s = t.sum_to(&[3]).unwrap();
    assert_eq!(s.dims(), &[3]);
    assert_eq!(s.to_vec::<f32>().unwrap(), vec![0.0, 0.0, 0.0]);
}

#[test]
fn sum_to_rejects_shapes_that_never_broadcast() {
    let t = t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]);

    // Target of higher rank than the source.
    assert!(matches!(
        t.sum_to(&[1, 2, 3]),
        Err(Error::ShapeMismatch { op: "sum_to", .. })
    ));
    // Aligned axis that is neither equal nor 1 on the target side.
    assert!(matches!(
        t.sum_to(&[2, 2]),
        Err(Error::ShapeMismatch { op: "sum_to", .. })
    ));
    // Target axis *larger* than the source's: that is a broadcast, not a
    // reduction.
    assert!(matches!(
        t.sum_to(&[4, 3]),
        Err(Error::ShapeMismatch { op: "sum_to", .. })
    ));
    // The reported shapes name both sides.
    match t.sum_to(&[5]) {
        Err(Error::ShapeMismatch { lhs, rhs, .. }) => {
            assert_eq!(lhs, Shape::from([2, 3]));
            assert_eq!(rhs, Shape::from([5]));
        }
        _ => panic!("expected a ShapeMismatch"),
    }

    // Bool has no sum: the backend says so rather than inventing one.
    let b = Tensor::from_vec(vec![true, false], [2], &CPU).unwrap();
    assert!(matches!(
        b.sum_to(&[1]),
        Err(Error::Unsupported { op: "reduce", .. })
    ));
}
